//! The converted programs that also draw the frame's own motion for DLSS ([`super::super::motion`]).
//!
//! - A skinned character's program (its vertex shader reads the joints from `matrices_ubo`) keeps
//!   its vertex shader's body a second time as `monaka_prev_main`, run only while `monaka_mv.x`
//!   is 1: the same vertex with the previous frame's joints (`monaka_prev_joints`, a storage
//!   buffer our pass fills), `_va_` rows (`monaka_prev_va`: the morph and blend weights) and
//!   skinning switches (`monaka_prev_hs`), placed by `monaka_prev_rows`: this frame's projection of
//!   where the model was last frame (the camera's own motion is the depth pass's to add). Both
//!   positions go to the fragment shader for each eye (the other eye's by the eye matrix), which
//!   writes the motion in UV units (x right, y down, towards where the pixel was) and its window
//!   depth: `vec4( motion, 0, depth )`.
//! - A particle's program (smoke, fire, sparks) runs as it is and, while `monaka_mv.x` is 2 or 3,
//!   writes how much of the pixel it covers into the blue channel only: its alpha (2, blended by
//!   it) or its brightest channel (3, added).
//!
//! With `monaka_mv.x` 0 both draw exactly as before.

use super::{STEREO_UNIFORM, after_version, find_main, outputs, position_rows};

/// The storage buffer binding the previous joints are read from.
pub const JOINTS_BINDING: u32 = 7;
/// The uniform switching a program's motion draw (x: 0 off, 1 motion, 2 alpha share, 3 brightness share).
pub const SWITCH: &str = "monaka_mv";
pub const PREVIOUS_VA: &str = "monaka_prev_va";
pub const PREVIOUS_ROWS: &str = "monaka_prev_rows";
pub const PREVIOUS_SKINNING: &str = "monaka_prev_hs";

/// What a converted program draws for the motion pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A skinned character: its `_va_` rows giving the position (x, y, z, w) and the world
    /// position (x, y, z), and how many `_va_` rows it has.
    Character { rows: [usize; 4], world: [usize; 3], va: usize },
    Particle,
}

/// The programs drawn as particles: those a pixel's colour is partly made of over what is behind.
fn particle(name: &str) -> bool {
    (name.starts_with("softparticle") || name.starts_with("particle")) && !name.contains("distortion")
}

/// The `_va_` row `dot4( localPosition, _va_[n ] );` names, after `prefix` on a trimmed line.
fn row_after(line: &str, prefix: &str) -> Option<usize> {
    let rest = line.trim().strip_prefix(prefix)?.trim_start().strip_prefix('=')?.trim();
    let inner = rest.strip_prefix("dot4")?.trim_start().strip_prefix('(')?.strip_suffix(';')?.trim_end().strip_suffix(')')?;
    let (_, row) = inner.split_once(',')?;
    row.trim().strip_prefix("_va_")?.trim_start().strip_prefix('[')?.trim().strip_suffix(']')?.trim().parse().ok()
}

/// How many rows `uniform vec4 _va_ [n];` declares.
fn va_rows(vertex: &str) -> Option<usize> {
    vertex.lines().find_map(|l| l.trim().strip_prefix("uniform vec4 _va_")?.trim().strip_prefix('[')?.split(']').next()?.trim().parse().ok())
}

/// What render program `name` (its engine vertex shader `vertex`) draws for the motion pass.
pub fn kind(name: &str, vertex: &str) -> Option<Kind> {
    if particle(name) {
        return Some(Kind::Particle);
    }
    // A light's own pass over a lit model (added on top of its first): the first pass's motion is
    // the same, drawn once.
    // Only characters and the held weapon: a skinned prop (a chair, ropes: `outside`) stays still
    // in its own frame, and the camera's motion is already right for it.
    if !["character", "skin", "hair", "eye", "weapon", "selfshadow"].iter().any(|part| name.contains(part)) {
        return None;
    }
    if !vertex.contains("uniform matrices_ubo") || !vertex.contains("hardwareSkinning") || ["_point", "_spot", "_parallel"].iter().any(|light| name.ends_with(light)) {
        return None;
    }
    let rows = position_rows(vertex)?;
    let mut world = [None; 3];
    for line in vertex.lines() {
        for (i, c) in ["x", "y", "z"].iter().enumerate() {
            if let Some(row) = row_after(line, &format!("worldPosition.{c}")) {
                match world[i] {
                    None => world[i] = Some(row),
                    Some(known) if known != row => return None,
                    Some(_) => {}
                }
            }
        }
    }
    let [Some(x), Some(y), Some(z)] = world else { return None };
    Some(Kind::Character { rows, world: [x, y, z], va: va_rows(vertex)? })
}

/// The byte range of the body (braces included) of the function whose header starts at `at`.
fn body(source: &str, at: usize) -> Option<(usize, usize)> {
    let open = at + source[at..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in source[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, open + i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// `text` with every `name` not part of a longer identifier before it replaced by `with`.
fn replace_word(text: &str, name: &str, with: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(name) {
        let before = rest[..i].chars().next_back();
        out.push_str(&rest[..i]);
        if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.push_str(name);
        } else {
            out.push_str(with);
        }
        rest = &rest[i + name.len()..];
    }
    out.push_str(rest);
    out
}

/// The converted vertex shader (`stereo`, from [`super::stereo_vertex`]) of a character `kind`
/// that also places its vertex as it was last frame; none when its text is not the one known.
pub fn vertex(stereo: &str, kind: Kind) -> Option<String> {
    let Kind::Character { rows, va, .. } = kind else { return None };
    let header = stereo.find("void monaka_main()")?;
    let (open, close) = body(stereo, header)?;
    let mut previous = String::new();
    for line in stereo[open..close].split_inclusive('\n') {
        let mut line = replace_word(line, "gl_Position", "monaka_prev_pos");
        if line.trim_start().starts_with("monaka_prev_pos.") {
            for (i, row) in rows.iter().enumerate() {
                line = line.replace(&format!("_va_[{row} ]"), &format!("{PREVIOUS_ROWS}[{i}]"));
            }
        }
        line = replace_word(&line, "_va_[", &format!("{PREVIOUS_VA}["));
        line = replace_word(&line, "hardwareSkinning", PREVIOUS_SKINNING);
        line = replace_word(&line, "matrices[", "monaka_prev_joints[");
        previous.push_str(&line);
    }
    let call = "\tmonaka_main();\n";
    if stereo.matches(call).count() != 1 || !stereo[close..].contains(call) {
        return None;
    }
    let s = STEREO_UNIFORM;
    let declarations = format!(
        "layout( std430, binding = {JOINTS_BINDING} ) readonly buffer monaka_prev_joint_rows {{ vec4 monaka_prev_joints[]; }};\n\
         uniform vec4 {PREVIOUS_VA}[{va}];\n\
         uniform vec4 {PREVIOUS_ROWS}[4];\n\
         uniform vec4 {PREVIOUS_SKINNING};\n\
         uniform vec4 {SWITCH};\n\
         vec4 monaka_prev_pos;\n\
         out vec4 monaka_m0;\nout vec4 monaka_m1;\nout vec4 monaka_m2;\nout vec4 monaka_m3;\n"
    );
    let run = format!(
        "\tbool monaka_moving = {SWITCH}.x == 1.0;\n\
         \tmonaka_prev_pos = vec4( 0.0 );\n\
         \tif ( monaka_moving ) {{\n\t\tmonaka_prev_main();\n\t}}\n\
         \tmonaka_main();\n\
         \tif ( monaka_moving ) {{\n\
         \t\tmonaka_m0 = gl_Position;\n\
         \t\tmonaka_m1 = monaka_prev_pos;\n\
         \t\tmonaka_m2 = vec4( dot( {s}[0], gl_Position ), dot( {s}[1], gl_Position ), dot( {s}[2], gl_Position ), dot( {s}[3], gl_Position ) );\n\
         \t\tmonaka_m3 = vec4( dot( {s}[0], monaka_prev_pos ), dot( {s}[1], monaka_prev_pos ), dot( {s}[2], monaka_prev_pos ), dot( {s}[3], monaka_prev_pos ) );\n\
         \t}}\n"
    );
    let mut out = String::with_capacity(stereo.len() * 2 + 1024);
    out.push_str(&stereo[..header]);
    out.push_str(&declarations);
    out.push_str(&stereo[header..close]);
    out.push_str("\nvoid monaka_prev_main() ");
    out.push_str(&previous);
    out.push_str(&stereo[close..].replacen(call, &run, 1));
    Some(after_version(&out, "#extension GL_ARB_shader_storage_buffer_object : require\n#extension GL_ARB_shading_language_420pack : require\n"))
}

/// The fragment shader of a program of `kind` that also draws for the motion pass; none when its
/// text is not the one known.
pub fn fragment(fragment: &str, kind: Kind) -> Option<String> {
    let output = outputs(fragment).into_iter().next()?;
    if !fragment.lines().any(|l| l.trim() == format!("out vec4 {output};")) {
        return None;
    }
    let (start, end) = find_main(fragment)?;
    let mut out = String::with_capacity(fragment.len() + 1024);
    out.push_str(&fragment[..start]);
    out.push_str("void monaka_engine_main()");
    out.push_str(&fragment[end..]);
    let o = output;
    match kind {
        Kind::Character { .. } => out.push_str(&format!(
            "\nuniform vec4 {SWITCH};\n\
             in vec4 monaka_m0;\nin vec4 monaka_m1;\nin vec4 monaka_m2;\nin vec4 monaka_m3;\n\
             void main() {{\n\
             \tmonaka_engine_main();\n\
             \tif ( {SWITCH}.x == 1.0 ) {{\n\
             \t\tvec4 monaka_now = gl_Layer == 0 ? monaka_m0 : monaka_m2;\n\
             \t\tvec4 monaka_was = gl_Layer == 0 ? monaka_m1 : monaka_m3;\n\
             \t\tvec2 monaka_motion = vec2( 0.0 );\n\
             \t\tif ( monaka_now.w > 1e-6 && monaka_was.w > 1e-6 ) {{\n\
             \t\t\tvec2 monaka_a = monaka_now.xy / monaka_now.w;\n\
             \t\t\tvec2 monaka_b = monaka_was.xy / monaka_was.w;\n\
             \t\t\tmonaka_motion = vec2( monaka_b.x - monaka_a.x, monaka_a.y - monaka_b.y ) * 0.5;\n\
             \t\t\tif ( any( isnan( monaka_motion ) ) || any( isinf( monaka_motion ) ) ) monaka_motion = vec2( 0.0 );\n\
             \t\t\tmonaka_motion = clamp( monaka_motion, vec2( -0.25 ), vec2( 0.25 ) );\n\
             \t\t}}\n\
             \t\t{o} = vec4( monaka_motion, 0.0, gl_FragCoord.z );\n\
             \t}}\n\
             }}\n"
        )),
        Kind::Particle => out.push_str(&format!(
            "\nuniform vec4 {SWITCH};\n\
             void main() {{\n\
             \tmonaka_engine_main();\n\
             \tif ( {SWITCH}.x >= 2.0 ) {{\n\
             \t\tfloat monaka_share = {SWITCH}.x == 2.0 ? {o}.a : max( {o}.r, max( {o}.g, {o}.b ) );\n\
             \t\t{o} = vec4( 0.0, 0.0, clamp( ( monaka_share - 0.1 ) * 1.25, 0.0, 1.0 ), 0.0 );\n\
             \t}}\n\
             }}\n"
        )),
    }
    Some(match kind {
        Kind::Character { .. } => after_version(&out, "#extension GL_ARB_fragment_layer_viewport : require\n"),
        Kind::Particle => out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKINNED: &str = "#version 150\n\nuniform vec4 _va_ [8];\nuniform vec4 hardwareSkinning ;\nuniform matrices_ubo { vec4 matrices[768]; };\nin vec4 in_Position;\nin vec4 in_Position1;\nout vec4 vofi_TexCoord0;\n\nvoid main() {\n\tvec4 localPosition = in_Position + _va_[0 ].x * in_Position1;\n\tif ( hardwareSkinning.x != 0.0 ) {\n\t\tlocalPosition.x = dot4( localPosition, matrices[int(in_Position.w)] );\n\t}\n\tgl_Position.x = dot4( localPosition, _va_[1 ] );\n\tgl_Position.y = dot4( localPosition, _va_[2 ] );\n\tgl_Position.z = dot4( localPosition, _va_[3 ] );\n\tgl_Position.w = dot4( localPosition, _va_[4 ] );\n\tvec4 worldPosition;\n\tworldPosition.x = dot4( localPosition, _va_[5 ] );\n\tworldPosition.y = dot4( localPosition, _va_[6 ] );\n\tworldPosition.z = dot4( localPosition, _va_[7 ] );\n\tvofi_TexCoord0 = worldPosition;\n}\n";

    #[test]
    fn a_skinned_program_is_known_by_its_joints_and_rows() {
        assert_eq!(kind("skinsss", SKINNED), Some(Kind::Character { rows: [1, 2, 3, 4], world: [5, 6, 7], va: 8 }));
        assert_eq!(kind("softparticleaddmasklut", "void main() {}"), Some(Kind::Particle));
        assert_eq!(kind("particledistortion", "void main() {}"), None);
        assert_eq!(kind("particledistortiontwosided", "void main() {}"), None);
        assert_eq!(kind("interskin_point", SKINNED), None);
        assert_eq!(kind("outside", SKINNED), None);
        assert_eq!(kind("landuniquebump", "uniform vec4 _va_ [8];\nvoid main() {}"), None);
    }

    #[test]
    fn the_previous_vertex_reads_the_previous_joints_rows_and_weights() {
        let k = kind("skinsss", SKINNED).unwrap();
        let stereo = super::super::stereo_vertex(SKINNED).unwrap();
        let out = vertex(&stereo, k).expect("converted");
        let previous = &out[out.find("void monaka_prev_main()").unwrap()..];
        let previous = &previous[..previous.find("\n}\n").unwrap()];
        assert!(previous.contains("monaka_prev_pos.x = dot4( localPosition, monaka_prev_rows[0] );"), "{previous}");
        assert!(previous.contains("monaka_prev_pos.w = dot4( localPosition, monaka_prev_rows[3] );"), "{previous}");
        assert!(previous.contains("monaka_prev_va[0 ].x * in_Position1"), "{previous}");
        assert!(previous.contains("monaka_prev_hs.x"), "{previous}");
        assert!(previous.contains("monaka_prev_joints[int("), "{previous}");
        assert!(previous.contains("worldPosition.x = dot4( localPosition, monaka_prev_va[5 ] );"), "{previous}");
        assert!(!previous.contains("gl_Position") && !previous.contains("_va_[") && !previous.contains(" matrices["), "{previous}");
        // The engine's own body is untouched, and runs after the previous one.
        assert!(out.contains("gl_Position.x = dot4( localPosition, _va_[1 ] );"));
        let main = &out[out.rfind("void main()").unwrap()..];
        assert!(main.find("monaka_prev_main();").unwrap() < main.find("monaka_main();\n\tif ( monaka_moving )").unwrap(), "{main}");
        assert!(out.contains("#extension GL_ARB_shader_storage_buffer_object : require"));
        assert!(out.contains("uniform vec4 monaka_prev_va[8];"));
    }

    #[test]
    fn the_fragment_writes_motion_or_coverage_only_when_switched() {
        let engine = "#version 150\nout vec4 out_FragColor0;\nvoid main() {\n\tout_FragColor0 = vec4( 1.0 );\n}\n";
        let character = fragment(engine, Kind::Character { rows: [0; 4], world: [0; 3], va: 1 }).unwrap();
        assert!(character.contains("void monaka_engine_main()"));
        assert!(character.contains("out_FragColor0 = vec4( monaka_motion, 0.0, gl_FragCoord.z );"));
        assert!(character.contains("#extension GL_ARB_fragment_layer_viewport : require"));
        let particle = fragment(engine, Kind::Particle).unwrap();
        assert!(particle.contains("out_FragColor0 = vec4( 0.0, 0.0, clamp( ( monaka_share - 0.1 ) * 1.25, 0.0, 1.0 ), 0.0 );"));
        assert!(!particle.contains("in vec4 monaka_m0"));
        assert_eq!(fragment("void main() {}", Kind::Particle), None);
    }
}
