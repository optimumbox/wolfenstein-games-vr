//! The game's render programs, read from its own archive (`base\chunk0.index` and
//! `chunk0.resources`): each `generated/renderprogs/<name>_pc.bin` holds one program's GLSL, a
//! vertex and a fragment shader ([`Renderprog`]). Nothing of the game's is shipped with the pack.
//!
//! The index: 0x24 bytes of header and the entry count (u32 big-endian); then per entry a u32
//! big-endian id, three strings (u32 little-endian length and bytes: type, name, path), the offset,
//! size and stored size (u32 big-endian each) and 9 bytes of flags. A stored size below the size
//! is raw deflate, flushed without a final block. Only the render programs' part of the index is
//! read: entries of other types further on are laid out differently.

use std::path::Path;

/// One archive entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub kind: String,
    pub path: String,
    pub offset: u32,
    pub size: u32,
    pub stored: u32,
}

/// The longest string an entry is believed to hold; a longer one means the layout was lost.
const LONGEST: usize = 4096;

fn u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

fn u32_le(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// The index's entries, up to where their layout stops being the one above.
pub fn entries(index: &[u8]) -> Vec<Entry> {
    let mut out = Vec::new();
    let Some(count) = u32_be(index, 0x24) else { return out };
    let mut at = 0x28;
    'entries: for _ in 0..count {
        at += 4;
        let mut strings = [String::new(), String::new(), String::new()];
        for string in &mut strings {
            let Some(length) = u32_le(index, at).map(|n| n as usize).filter(|&n| n <= LONGEST) else { break 'entries };
            let Some(bytes) = index.get(at + 4..at + 4 + length) else { break 'entries };
            *string = String::from_utf8_lossy(bytes).into_owned();
            at += 4 + length;
        }
        let (Some(offset), Some(size), Some(stored)) = (u32_be(index, at), u32_be(index, at + 4), u32_be(index, at + 8)) else { break };
        at += 12 + 9;
        let [kind, _name, path] = strings;
        out.push(Entry { kind, path, offset, size, stored });
    }
    out
}

/// An entry's bytes from the part of the resources file that starts at offset `base`.
pub fn read(resources: &[u8], base: u32, entry: &Entry) -> Result<Vec<u8>, String> {
    let start = entry.offset.checked_sub(base).ok_or_else(|| format!("{}: before the part read", entry.path))? as usize;
    let stored = resources.get(start..start + entry.stored as usize).ok_or_else(|| format!("{}: past the end of the resources", entry.path))?;
    if entry.stored >= entry.size {
        return Ok(stored.to_vec());
    }
    // Flushed without a final block: the inflater stops short of "done" with all the bytes out
    // (and may run on into the bytes after them).
    let size = entry.size as usize;
    let mut output = match miniz_oxide::inflate::decompress_to_vec_with_limit(stored, size) {
        Ok(output) => output,
        Err(error) => error.output,
    };
    output.truncate(size);
    if output.len() != size {
        return Err(format!("{}: inflated {} of {} bytes", entry.path, output.len(), entry.size));
    }
    Ok(output)
}

/// One render program: its name (`wireframe` for `generated/renderprogs/wireframe_pc.bin`) and
/// its two shaders.
#[derive(Clone, Debug)]
pub struct Renderprog {
    pub name: String,
    pub vertex: String,
    pub fragment: String,
}

/// The program's two GLSL sources out of a `_pc.bin`: each is preceded by its length (u32
/// little-endian) and starts with `#version`; the vertex shader is the one writing
/// `gl_Position`.
pub fn renderprog(name: &str, bytes: &[u8]) -> Option<Renderprog> {
    const VERSION: &[u8] = b"#version ";
    let mut sources = Vec::new();
    let mut from = 4;
    while let Some(found) = bytes.get(from..)?.windows(VERSION.len()).position(|w| w == VERSION) {
        let at = from + found;
        let length = u32_le(bytes, at - 4)? as usize;
        match bytes.get(at..at + length) {
            Some(text) => {
                sources.push(String::from_utf8_lossy(text).into_owned());
                from = at + length;
            }
            None => from = at + VERSION.len(),
        }
        if from + VERSION.len() > bytes.len() {
            break;
        }
    }
    let vertex = sources.iter().position(|s| s.contains("gl_Position"))?;
    let fragment = (0..sources.len()).find(|&i| i != vertex)?;
    Some(Renderprog { name: name.to_owned(), vertex: sources[vertex].clone(), fragment: sources[fragment].clone() })
}

/// Every render program in the game installed at `base` (its `base` folder).
pub fn renderprogs(base: &Path) -> Result<Vec<Renderprog>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let index = std::fs::read(base.join("chunk0.index")).map_err(|e| format!("chunk0.index: {e}"))?;
    const FOLDER: &str = "generated/renderprogs/";
    let wanted: Vec<Entry> = entries(&index).into_iter().filter(|e| e.kind == "renderProg" && e.path.starts_with(FOLDER)).collect();
    // The programs lie together in the resources file: only their stretch of it is read.
    let (Some(first), Some(end)) = (wanted.iter().map(|e| e.offset).min(), wanted.iter().map(|e| u64::from(e.offset) + u64::from(e.stored)).max()) else {
        return Err("no render programs in chunk0.index".to_owned());
    };
    let mut resources = vec![0u8; (end - u64::from(first)) as usize];
    let mut file = std::fs::File::open(base.join("chunk0.resources")).map_err(|e| format!("chunk0.resources: {e}"))?;
    file.seek(SeekFrom::Start(first.into())).and_then(|_| file.read_exact(&mut resources)).map_err(|e| format!("chunk0.resources: {e}"))?;
    let mut out = Vec::new();
    for entry in &wanted {
        let name = entry.path[FOLDER.len()..].trim_end_matches(".bin").trim_end_matches("_pc");
        let bytes = read(&resources, first, entry)?;
        if let Some(program) = renderprog(name, &bytes) {
            out.push(program);
        }
    }
    if out.is_empty() {
        return Err("no render programs in chunk0".to_owned());
    }
    Ok(out)
}
