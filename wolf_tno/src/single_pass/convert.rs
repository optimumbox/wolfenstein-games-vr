//! The shader conversion: a render program's GLSL made to draw both eyes in one draw with
//! `GL_NV_stereo_view_rendering`.
//!
//! - The vertex shader's `main` is kept as `monaka_main`; a new `main` runs it, sends the vertex
//!   to layer 0 (the eye the engine is drawing) and, while the stereo uniform's switch is on,
//!   writes the other eye's position into layer 1 of the same viewport (`secondary_view_offset`):
//!   the left eye's clip position times the eye matrix ([`STEREO_UNIFORM`], rows 0 to 3; row 4's
//!   x is the switch). A framebuffer that is not layered ignores the layer. The
//!   eye matrix is the right eye's view-projection over the left eye's, the same for every draw of
//!   a frame, so no per-draw constant changes.
//! - Every plain uniform gets the location the engine's program gave it ([`with_locations`]), so
//!   the converted program takes the engine's own uniform calls unchanged.
//!
//! Each id Tech 5 render program is one vertex and one fragment shader, written alike: no
//! `return` in `main`, plain `uniform <type> <name> [<n>] ;` declarations, one uniform block.

use std::collections::HashSet;

/// The converted vertex shader's uniform: rows 0-3 the eye matrix, row 4's x the switch.
pub const STEREO_UNIFORM: &str = "monaka_sl";
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
        "\nlayout( secondary_view_offset = 1 ) out int gl_Layer;\n\
         uniform vec4 {u}[{n}];\n\
         void main() {{\n\
         \tmonaka_main();\n\
         \tgl_Layer = 0;\n\
         \tgl_ViewportMask[0] = 1;\n\
         \tif ( {u}[4].x != 0.0 ) {{\n\
         \t\tgl_SecondaryPositionNV = vec4( dot( {u}[0], gl_Position ), dot( {u}[1], gl_Position ), dot( {u}[2], gl_Position ), dot( {u}[3], gl_Position ) );\n\
         \t\tgl_SecondaryViewportMaskNV[0] = 1;\n\
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

/// Whether the fragment shader reads what the scene drew so far at its own pixel: `gl_FragCoord`
/// into the view's depth, normal or colour maps (decals, lights and soft particles place, light or
/// fade themselves by the depth under them). In the other eye's copy both the pixel's place and the
/// maps are the first eye's, so such a program is drawn by each eye.
pub fn reads_screen(fragment: &str) -> bool {
    fragment.contains("gl_FragCoord") && ["samp_view", "samp_targetdepth"].iter().any(|map| fragment.contains(map))
}

/// The prefix of the layered fragment shader's array sampler for each screen map
/// (`monaka_l_samp_viewdepthmap`).
pub const LAYERED_PREFIX: &str = "monaka_l_";
/// The fragment shader's uniform rows (`uniform vec4 _fa_ [n];`).
pub const FRAGMENT_ROWS: &str = "_fa_";
/// The layered fragment shader's rows for the other eye's copy, in place of [`FRAGMENT_ROWS`].
pub const RIGHT_ROWS: &str = "monaka_fr";

const LAYER_EXTENSION: &str = "#extension GL_ARB_fragment_layer_viewport : require\n";

/// A fragment shader reading the scene's maps made to read them at the layer it draws into
/// ([`layered_fragment`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayeredFragment {
    pub source: String,
    /// The screen maps (2D samplers) it reads layered, each also as `LAYERED_PREFIX` + its name.
    pub maps: Vec<String>,
    /// Its [`FRAGMENT_ROWS`] (0 without), given to the other eye's copy as [`RIGHT_ROWS`].
    pub rows: usize,
}

fn screen_map(name: &str) -> bool {
    name.starts_with("samp_view") || name.starts_with("samp_targetdepth")
}

/// A fragment shader reading the scene's 2D maps ([`reads_screen`]) made to read them at the
/// layer its fragment is drawn into, for the copy of the other eye: each map is also declared as a
/// `sampler2DArray` (to be given the stand-in's array), each helper taking a `sampler2D` gets an
/// overload taking the array (its lookups at `gl_Layer`), and each call given a map calls the
/// array's overload in layer 1. The shader's rows are read from [`RIGHT_ROWS`] in layer 1 (the
/// other eye's, set for each draw). Layer 0 reads as the engine's shader did.
///
/// None when a map is used other than as a helper's first argument, or a helper given one
/// samples it some other way.
pub fn layered_fragment(fragment: &str) -> Option<LayeredFragment> {
    let declared = uniforms(fragment);
    let maps: Vec<String> = declared.iter().filter(|(_, u)| u.kind == "sampler2D" && u.count == 1 && screen_map(&u.name)).map(|(_, u)| u.name.clone()).collect();
    if maps.is_empty() {
        return None;
    }
    let mut out = fragment.to_owned();
    // The helpers' overloads, each after its helper (so it may call the overloads before it).
    let mut overloaded = HashSet::new();
    let mut inserts = Vec::new();
    for function in functions(&out) {
        if function.first_param.as_ref().is_some_and(|(kind, _)| kind == "sampler2D")
            && let Some(text) = array_overload(&out, &function, &overloaded)
        {
            overloaded.insert(function.name.clone());
            inserts.push((function.end, text));
        }
    }
    for (at, text) in inserts.into_iter().rev() {
        out.insert_str(at, &format!("\n{text}"));
    }
    // Each use of a map: a helper's call, made in layer 1 with the map's array.
    let mut uses: Vec<(usize, String)> = Vec::new();
    for map in &maps {
        uses.extend(identifiers(&out).filter(|&(at, name)| name == map && !declares(&out, at)).map(|(at, _)| (at, map.clone())));
    }
    uses.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (at, map) in uses {
        let call = call_around(&out, at)?;
        if !overloaded.contains(call.name) || call.args.first().is_none_or(|&(start, end)| out[start..end].trim() != map) {
            return None;
        }
        let rest = out[call.args[0].1..call.close].trim_end();
        let text = format!("( gl_Layer != 0 ? {f}( {LAYERED_PREFIX}{map}{rest} ) : {f}( {map}{rest} ) )", f = call.name);
        let (start, close) = (call.start, call.close);
        out.replace_range(start..=close, &text);
    }
    // The rows: the other eye's in layer 1.
    let rows = declared.iter().find(|(_, u)| u.kind == "vec4" && u.name == FRAGMENT_ROWS).map_or(0, |(_, u)| u.count);
    if rows > 0 {
        let mut renamed = String::with_capacity(out.len() + 64);
        let mut from = 0;
        for (at, name) in identifiers(&out) {
            if name == FRAGMENT_ROWS && !declares(&out, at) {
                renamed.push_str(&out[from..at]);
                renamed.push_str("monaka_fa");
                from = at + name.len();
            }
        }
        renamed.push_str(&out[from..]);
        out = renamed;
        let at = find_main(&out)?;
        out.replace_range(at.0..at.1, "void monaka_fmain()");
        out.push_str(&format!(
            "\nvoid main() {{\n\
             \tfor ( int i = 0; i < {rows}; i++ ) {{\n\
             \t\tmonaka_fa[i] = gl_Layer != 0 ? {RIGHT_ROWS}[i] : {FRAGMENT_ROWS}[i];\n\
             \t}}\n\
             \tmonaka_fmain();\n\
             }}\n"
        ));
    }
    // The arrays and rows declared after the engine's declarations.
    let mut declared_out = String::with_capacity(out.len() + 256);
    for line in out.split_inclusive('\n') {
        declared_out.push_str(line);
        let Some(uniform) = uniform(line) else { continue };
        let ending = if line.ends_with('\n') { "" } else { "\n" };
        if maps.contains(&uniform.name) {
            declared_out.push_str(&format!("{ending}uniform sampler2DArray {LAYERED_PREFIX}{};\n", uniform.name));
        } else if uniform.name == FRAGMENT_ROWS && rows > 0 {
            declared_out.push_str(&format!("{ending}uniform vec4 {RIGHT_ROWS}[{rows}];\nvec4 monaka_fa[{rows}];\n"));
        }
    }
    Some(LayeredFragment { source: after_version(&declared_out, LAYER_EXTENSION), maps, rows })
}

/// A function defined at the top level: its name, first parameter (type and name), and where its
/// text is (from its line's start to after its closing brace) and its body opens.
struct Function {
    name: String,
    first_param: Option<(String, String)>,
    start: usize,
    body: usize,
    end: usize,
}

fn functions(source: &str) -> Vec<Function> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] != b'{' {
            at += 1;
            continue;
        }
        let Some(close) = closing(source, at, b'{', b'}') else { break };
        let before = source[..at].trim_end();
        if let Some(signature_end) = before.len().checked_sub(1).filter(|&i| bytes[i] == b')')
            && let Some(open) = opening(source, signature_end)
        {
            let name_end = source[..open].trim_end().len();
            let name_start = source[..name_end].rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |i| i + 1);
            let start = source[..name_start].rfind('\n').map_or(0, |i| i + 1);
            let first_param = split_args(source, open, signature_end).first().and_then(|&(s, e)| {
                let words: Vec<&str> = source[s..e].split_whitespace().collect();
                (words.len() >= 2).then(|| (words[words.len() - 2].to_owned(), words[words.len() - 1].to_owned()))
            });
            if name_start < name_end {
                out.push(Function { name: source[name_start..name_end].to_owned(), first_param, start, body: at, end: close + 1 });
            }
        }
        at = close + 1;
    }
    out
}

/// `function`'s text taking a `sampler2DArray` for its first parameter, its lookups of it at
/// `gl_Layer`; none when it uses the parameter other than in a lookup or a call of a helper in
/// `overloaded`.
fn array_overload(source: &str, function: &Function, overloaded: &HashSet<String>) -> Option<String> {
    let (_, param) = function.first_param.as_ref()?;
    let mut text = source[function.start..function.end].to_owned();
    let body = function.body - function.start;
    let uses: Vec<usize> = identifiers(&text).filter(|&(at, name)| at > body && name == param).map(|(at, _)| at).collect();
    for &at in uses.iter().rev() {
        let call = call_around(&text, at)?;
        if call.args.first().is_none_or(|&(start, end)| text[start..end].trim() != param) {
            return None;
        }
        let wrap = match call.name {
            "texture" | "textureLod" | "textureGrad" => "vec3( {}, float( gl_Layer ) )",
            "texelFetch" => "ivec3( {}, gl_Layer )",
            name if overloaded.contains(name) => continue,
            _ => return None,
        };
        let &(start, end) = call.args.get(1)?;
        let raw = &text[start..end];
        let (lead, trail) = (&raw[..raw.len() - raw.trim_start().len()], &raw[raw.trim_end().len()..]);
        let coordinate = format!("{lead}{}{trail}", wrap.replace("{}", raw.trim()));
        text.replace_range(start..end, &coordinate);
    }
    let signature = text[..body].replacen("sampler2D", "sampler2DArray", 1);
    Some(format!("{signature}{}", &text[body..]))
}

/// A call around the argument at `at`: where its name starts, the name, its arguments and its
/// closing parenthesis.
struct Call<'a> {
    start: usize,
    name: &'a str,
    args: Vec<(usize, usize)>,
    close: usize,
}

fn call_around(source: &str, at: usize) -> Option<Call<'_>> {
    let open = source[..at].trim_end().len().checked_sub(1).filter(|&i| source.as_bytes()[i] == b'(')?;
    let name_end = source[..open].trim_end().len();
    let start = source[..name_end].rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |i| i + 1);
    let name = &source[start..name_end];
    if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let close = closing(source, open, b'(', b')')?;
    Some(Call { start, name, args: split_args(source, open, close), close })
}

/// The arguments between the parentheses at `open` and `close`, split at their top-level commas.
fn split_args(source: &str, open: usize, close: usize) -> Vec<(usize, usize)> {
    let mut args = Vec::new();
    let (mut depth, mut from) = (0i32, open + 1);
    for (i, b) in source.as_bytes()[open + 1..close].iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                args.push((from, open + 1 + i));
                from = open + 2 + i;
            }
            _ => {}
        }
    }
    if !source[from..close].trim().is_empty() || !args.is_empty() {
        args.push((from, close));
    }
    args
}

/// Where the bracket opened at `open` closes.
fn closing(source: &str, open: usize, opens: u8, closes: u8) -> Option<usize> {
    let mut depth = 0;
    for (i, &b) in source.as_bytes().iter().enumerate().skip(open) {
        if b == opens {
            depth += 1;
        } else if b == closes {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Where the parenthesis closed at `close` opens.
fn opening(source: &str, close: usize) -> Option<usize> {
    let mut depth = 0;
    for i in (0..=close).rev() {
        match source.as_bytes()[i] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The identifiers in `source` and where each starts.
fn identifiers(source: &str) -> impl Iterator<Item = (usize, &str)> {
    let bytes = source.as_bytes();
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut at = 0;
    std::iter::from_fn(move || {
        while at < bytes.len() {
            if word(bytes[at]) && (at == 0 || !word(bytes[at - 1])) {
                let start = at;
                while at < bytes.len() && word(bytes[at]) {
                    at += 1;
                }
                if !bytes[start].is_ascii_digit() {
                    return Some((start, &source[start..at]));
                }
            } else {
                at += 1;
            }
        }
        None
    })
}

/// Whether `at` is on a plain uniform's declaration line.
fn declares(source: &str, at: usize) -> bool {
    let start = source[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = source[at..].find('\n').map_or(source.len(), |i| at + i);
    uniform(&source[start..end]).is_some()
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
