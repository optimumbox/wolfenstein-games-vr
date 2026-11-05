//! The programs the single pass compiles (the render programs for their assembly, the converted
//! ones), compiled once: the same program asked for again in a run is the one already made (the
//! engine has thousands of programs from the same render program with the same locations, hair's
//! above all), and each one's binary is kept on disk for the next run, loaded with
//! `glProgramBinary` instead of compiled. A program is known by everything it is made of (both
//! shaders, the attribute and output locations); the file by the driver that made its binaries (a
//! new driver starts it afresh).

use super::gl_api::Gl;
use monaka_producer::log;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"MWPC0001";

struct Cache {
    driver: String,
    /// Binaries by program key: from the file and kept this run ([`keep`]).
    stored: HashMap<u64, (u32, Vec<u8>)>,
    /// Binaries of programs alive this run, read once (thousands of engine programs share one).
    read: HashMap<u64, (u32, Vec<u8>)>,
    /// Programs alive this run: by key, each with its users; and each program's key.
    live: HashMap<u64, (u32, usize)>,
    keys: HashMap<u32, u64>,
    added: usize,
    from_disk: usize,
    compiled: usize,
    shared: usize,
    refused: usize,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn path() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("monaka").join("cache").join("wolfenstein-games-vr").join("programs.bin"))
}

/// The driver's renderer and version (`glGetString`).
fn driver() -> String {
    let get = |name: u32| {
        // SAFETY: a plain query on the current context; the driver's NUL-terminated string.
        let text = unsafe { windows::Win32::Graphics::OpenGL::glGetString(name) };
        if text.is_null() {
            return String::new();
        }
        // SAFETY: as above.
        unsafe { std::ffi::CStr::from_ptr(text.cast()) }.to_string_lossy().into_owned()
    };
    format!("{} | {}", get(0x1F01), get(0x1F02))
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    // A separator, so that parts do not run into each other.
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

/// A program's key: what it is made of.
pub fn key(vertex: &str, fragment: &str, attributes: &[(String, u32)], outputs: &[(String, u32)]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    fnv(&mut hash, vertex.as_bytes());
    fnv(&mut hash, fragment.as_bytes());
    for (name, location) in attributes.iter().chain(std::iter::once(&(String::from("|"), u32::MAX))).chain(outputs) {
        fnv(&mut hash, name.as_bytes());
        fnv(&mut hash, &location.to_le_bytes());
    }
    hash
}

fn read(driver: &str) -> HashMap<u64, (u32, Vec<u8>)> {
    let mut out = HashMap::new();
    let Some(bytes) = path().and_then(|p| std::fs::read(p).ok()) else { return out };
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let slice = bytes.get(at..at.checked_add(n)?)?;
        at += n;
        Some(slice)
    };
    let u32_of = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap_or([0; 4]));
    if take(8) != Some(MAGIC.as_slice()) {
        return out;
    }
    let Some(length) = take(4).map(u32_of) else { return out };
    let Some(theirs) = take(length as usize) else { return out };
    if theirs != driver.as_bytes() {
        log!("single pass: the kept programs were made by another driver ({}): made afresh", String::from_utf8_lossy(theirs));
        return out;
    }
    let Some(count) = take(4).map(u32_of) else { return out };
    for _ in 0..count {
        let (Some(k), Some(format), Some(length)) = (take(8), take(4).map(u32_of), take(4).map(u32_of)) else { break };
        let Some(binary) = take(length as usize) else { break };
        out.insert(u64::from_le_bytes(k.try_into().unwrap_or([0; 8])), (format, binary.to_vec()));
    }
    out
}

fn with<T>(f: impl FnOnce(&mut Cache) -> T) -> T {
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = guard.get_or_insert_with(|| {
        let driver = driver();
        let stored = read(&driver);
        if !stored.is_empty() {
            log!("single pass: {} programs kept from an earlier run", stored.len());
        }
        Cache { driver, stored, read: HashMap::new(), live: HashMap::new(), keys: HashMap::new(), added: 0, from_disk: 0, compiled: 0, shared: 0, refused: 0 }
    });
    f(cache)
}

/// The program made of these, as [`Gl::program`]: the one alive already, else from its kept binary,
/// else compiled. Each one asked for is given back with [`release`].
pub fn program(gl: &Gl, vertex: &str, fragment: &str, attributes: &[(String, u32)], outputs: &[(String, u32)]) -> u32 {
    let key = key(vertex, fragment, attributes, outputs);
    with(|cache| {
        if let Some((program, users)) = cache.live.get_mut(&key) {
            *users += 1;
            cache.shared += 1;
            return *program;
        }
        let from_disk = cache.stored.get(&key).and_then(|(format, binary)| {
            // SAFETY: plain GL calls on the current context with a binary this driver made.
            let program = unsafe { (gl.create_program)() };
            unsafe { (gl.program_binary)(program, *format, binary.as_ptr().cast(), binary.len() as i32) };
            if gl.linked(program).is_ok() {
                Some(program)
            } else {
                // SAFETY: our own program.
                unsafe { (gl.delete_program)(program) };
                None
            }
        });
        let program = match from_disk {
            Some(program) => {
                cache.from_disk += 1;
                program
            }
            None => {
                if cache.stored.remove(&key).is_some() {
                    cache.refused += 1;
                }
                cache.compiled += 1;
                gl.program(vertex, fragment, attributes, outputs)
            }
        };
        cache.live.insert(key, (program, 1));
        cache.keys.insert(program, key);
        program
    })
}

/// A program's binary ([`Gl::binary`]), read from the driver once.
pub fn binary(gl: &Gl, program: u32) -> Option<(u32, Vec<u8>)> {
    with(|cache| {
        let key = cache.keys.get(&program).copied();
        if let Some(found) = key.and_then(|k| cache.stored.get(&k).or_else(|| cache.read.get(&k))) {
            return Some(found.clone());
        }
        let binary = gl.binary(program)?;
        if let Some(key) = key {
            cache.read.insert(key, binary.clone());
        }
        Some(binary)
    })
}

/// A program that did its work (linked, its binary taken into the engine's program) kept for the
/// next run; only these, not every one compiled (most of the engine's thousands are not swapped).
pub fn keep(gl: &Gl, program: u32) {
    with(|cache| {
        let Some(&key) = cache.keys.get(&program) else { return };
        if cache.stored.contains_key(&key) {
            return;
        }
        let binary = match cache.read.get(&key) {
            Some(binary) => Some(binary.clone()),
            None => gl.binary(program),
        };
        if let Some(binary) = binary {
            cache.stored.insert(key, binary);
            cache.added += 1;
        }
    })
}

/// A program from [`program`] given back: deleted with its last user.
pub fn release(gl: &Gl, program: u32) {
    let last = with(|cache| {
        let Some(&key) = cache.keys.get(&program) else { return true };
        let Some((_, users)) = cache.live.get_mut(&key) else { return true };
        *users -= 1;
        if *users > 0 {
            return false;
        }
        cache.live.remove(&key);
        cache.keys.remove(&program);
        cache.read.remove(&key);
        true
    });
    if last {
        // SAFETY: our own program, its last user done with it.
        unsafe { (gl.delete_program)(program) };
    }
}

/// The run's programs written for the next run (on a thread of its own), if any were added; and
/// what this run took from where.
pub fn save() {
    let (bytes, summary) = with(|cache| {
        let summary = format!(
            "single pass: programs {} loaded as kept, {} compiled, {} the same as one made already ({} kept ones refused by the driver)",
            cache.from_disk, cache.compiled, cache.shared, cache.refused
        );
        if cache.added == 0 {
            return (None, summary);
        }
        cache.added = 0;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(cache.driver.len() as u32).to_le_bytes());
        out.extend_from_slice(cache.driver.as_bytes());
        out.extend_from_slice(&(cache.stored.len() as u32).to_le_bytes());
        for (key, (format, binary)) in &cache.stored {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&format.to_le_bytes());
            out.extend_from_slice(&(binary.len() as u32).to_le_bytes());
            out.extend_from_slice(binary);
        }
        (Some((out, cache.stored.len())), summary)
    });
    log!("{summary}");
    let (Some((bytes, count)), Some(path)) = (bytes, path()) else { return };
    std::thread::spawn(move || {
        let written = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|_| {
            let temporary = path.with_extension("tmp");
            std::fs::write(&temporary, &bytes)?;
            std::fs::rename(&temporary, &path)
        });
        match written {
            Ok(()) => log!("single pass: {count} programs kept for the next run ({:.1} MB, {})", bytes.len() as f64 / 1e6, path.display()),
            Err(e) => log!("single pass: the programs could not be kept for the next run: {e}"),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_is_known_by_all_it_is_made_of() {
        let a = key("v", "f", &[("pos".into(), 0)], &[("out".into(), 0)]);
        assert_eq!(a, key("v", "f", &[("pos".into(), 0)], &[("out".into(), 0)]));
        assert_ne!(a, key("v", "f", &[("pos".into(), 1)], &[("out".into(), 0)]));
        assert_ne!(a, key("v", "f", &[], &[("pos".into(), 0), ("out".into(), 0)]));
        assert_ne!(key("vf", "", &[], &[]), key("v", "f", &[], &[]));
    }
}
