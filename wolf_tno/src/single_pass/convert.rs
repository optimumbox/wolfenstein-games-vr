//! The shader conversion: a render program's GLSL made to draw both eyes in one draw with
//! `GL_NV_stereo_view_rendering`.
//!
//! - The vertex shader's `main` is kept as `monaka_main`; a new `main` runs it, sends the vertex
//!   to viewport 0 (the eye the engine is drawing, as `glViewport` set it) and, while the stereo
//!   uniform's switch is on, writes the other eye's position into viewport 1: the left eye's clip
//!   position times the eye matrix ([`STEREO_UNIFORM`], rows 0 to 3; row 4's x is the switch). The
//!   eye matrix is the right eye's view-projection over the left eye's, the same for every draw of
//!   a frame, so no per-draw constant changes.
//! - Every plain uniform gets the location the engine's program gave it ([`with_locations`]), so
//!   the converted program takes the engine's own uniform calls unchanged.
//!
//! Each id Tech 5 render program is one vertex and one fragment shader, written alike: no
//! `return` in `main`, plain `uniform <type> <name> [<n>] ;` declarations, one uniform block.

/// The converted vertex shader's uniform: rows 0-3 the eye matrix, row 4's x the switch.
pub const STEREO_UNIFORM: &str = "monaka_sp";
/// Its rows.
pub const STEREO_ROWS: usize = 5;

const VIEWPORT_EXTENSIONS: &str = "#extension GL_NV_viewport_array2 : require\n#extension GL_NV_stereo_view_rendering : require\n";
const LOCATION_EXTENSION: &str = "#extension GL_ARB_explicit_uniform_location : require\n";

/// `source` with `lines` put right after its `#version` line.
fn after_version(source: &str, lines: &str) -> String {
    match source.strip_prefix("#version") {
        Some(rest) => {
            let end = rest.find('\n').map_or(source.len(), |i| "#version".len() + i + 1);
            let mut out = String::with_capacity(source.len() + lines.len() + 1);
            out.push_str(&source[..end]);
            if !source[..end].ends_with('\n') {
                out.push('\n');
            }
            out.push_str(lines);
            out.push_str(&source[end..]);
            out
        }
        None => format!("{lines}{source}"),
    }
}

/// The vertex shader drawing both eyes, or none when it has no `void main()` of its own.
pub fn stereo_vertex(vertex: &str) -> Option<String> {
    let at = find_main(vertex)?;
    let mut out = String::with_capacity(vertex.len() + 600);
    out.push_str(&vertex[..at.0]);
    out.push_str("void monaka_main()");
    out.push_str(&vertex[at.1..]);
    out.push_str(&format!(
        "\nuniform vec4 {u}[{n}];\n\
         void main() {{\n\
         \tmonaka_main();\n\
         \tgl_ViewportMask[0] = 1;\n\
         \tif ( {u}[4].x != 0.0 ) {{\n\
         \t\tgl_SecondaryPositionNV = vec4( dot( {u}[0], gl_Position ), dot( {u}[1], gl_Position ), dot( {u}[2], gl_Position ), dot( {u}[3], gl_Position ) );\n\
         \t\tgl_SecondaryViewportMaskNV[0] = 2;\n\
         \t}} else {{\n\
         \t\tgl_SecondaryPositionNV = gl_Position;\n\
         \t\tgl_SecondaryViewportMaskNV[0] = 0;\n\
         \t}}\n\
         }}\n",
        u = STEREO_UNIFORM,
        n = STEREO_ROWS
    ));
    Some(after_version(&out, VIEWPORT_EXTENSIONS))
}

/// Where `void main()` is written (start and end of the signature).
fn find_main(source: &str) -> Option<(usize, usize)> {
    let mut from = 0;
    while let Some(found) = source[from..].find("void") {
        let start = from + found;
        let rest = &source[start + 4..];
        let trimmed = rest.trim_start();
        if let Some(after) = trimmed.strip_prefix("main") {
            let after_name = after.trim_start();
            if let Some(inside) = after_name.strip_prefix('(') {
                let inside_trimmed = inside.trim_start();
                if let Some(_closed) = inside_trimmed.strip_prefix(')') {
                    let end = source.len() - inside_trimmed.len() + 1;
                    let before_ok = start == 0 || !source.as_bytes()[start - 1].is_ascii_alphanumeric();
                    if before_ok {
                        return Some((start, end));
                    }
                }
            }
        }
        from = start + 4;
    }
    None
}

/// A plain uniform declaration: `uniform <type> <name> [<n>] ;` (no block, no qualifier).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Uniform {
    pub kind: String,
    pub name: String,
    /// Elements (1 when not an array).
    pub count: usize,
}

/// The plain uniforms `source` declares, each with the line it is on.
pub fn uniforms(source: &str) -> Vec<(usize, Uniform)> {
    source.lines().enumerate().filter_map(|(line, text)| uniform(text).map(|u| (line, u))).collect()
}

fn uniform(line: &str) -> Option<Uniform> {
    let rest = line.trim().strip_prefix("uniform ")?;
    let body = rest.strip_suffix(';')?.trim();
    if body.contains('{') || body.contains('}') {
        return None;
    }
    let (kind, rest) = body.split_once(char::is_whitespace)?;
    let rest = rest.trim();
    let (name, count) = match rest.split_once('[') {
        Some((name, size)) => (name.trim(), size.trim().strip_suffix(']')?.trim().parse().ok()?),
        None => (rest, 1),
    };
    let identifier = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !s.starts_with(|c: char| c.is_ascii_digit());
    (identifier(kind) && identifier(name)).then(|| Uniform { kind: kind.to_owned(), name: name.to_owned(), count })
}

/// `source` with each plain uniform that `location` knows put at that location
/// (`layout(location = n)`), the others left as they are.
pub fn with_locations(source: &str, location: &dyn Fn(&Uniform) -> Option<i32>) -> String {
    let mut out = String::with_capacity(source.len() + 512);
    let mut any = false;
    for line in source.split_inclusive('\n') {
        match uniform(line).and_then(|u| location(&u)) {
            Some(at) => {
                any = true;
                out.push_str(&format!("layout(location = {at}) "));
                out.push_str(line.trim_start());
            }
            None => out.push_str(line),
        }
    }
    if any { after_version(&out, LOCATION_EXTENSION) } else { out }
}

/// The rows of the vertex shader's `_va_` that give `gl_Position` (x, y, z, w), when it is
/// written as one `dot` of a position with each (`gl_Position.x = dot4( position, _va_[2 ] );`):
/// the draw's model-view-projection rows. None for a shader that writes it otherwise (a full-screen
/// pass passes its input through).
pub fn position_rows(vertex: &str) -> Option<[usize; 4]> {
    let mut rows = [None; 4];
    for line in vertex.lines() {
        let line = line.trim();
        for (i, component) in ["x", "y", "z", "w"].iter().enumerate() {
            let Some(rest) = line.strip_prefix(&format!("gl_Position.{component}")) else { continue };
            let rest = rest.trim_start().strip_prefix('=')?.trim();
            let call = rest.strip_prefix("dot4").or_else(|| rest.strip_prefix("dot"))?.trim_start();
            let inner = call.strip_prefix('(')?.strip_suffix(';')?.trim_end().strip_suffix(')')?;
            let (_position, row) = inner.split_once(',')?;
            let index = row.trim().strip_prefix("_va_")?.trim_start().strip_prefix('[')?.trim().strip_suffix(']')?.trim();
            rows[i] = Some(index.parse().ok()?);
        }
    }
    Some([rows[0]?, rows[1]?, rows[2]?, rows[3]?])
}

/// The names of the fragment shader's outputs (`out vec4 out_FragColor0;`).
pub fn outputs(fragment: &str) -> Vec<String> {
    fragment
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("out ")?.strip_suffix(';')?;
            let (_kind, name) = rest.trim().split_once(char::is_whitespace)?;
            Some(name.trim().to_owned())
        })
        .collect()
}

#[cfg(test)]
mod tests;
