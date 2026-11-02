//! Single pass, checked on this PC without the game running: every render program read from the
//! game's archive, compiled as it is and converted, in a hidden window's GL context; and, given a
//! `probe_shaders` dump, which of the engine's program binaries each matches.
//!
//!     cargo run --release --example sp_offline -- <game>\base [<run>\shaders] [<out folder>]

use monaka_wolf_tno::single_pass::{archive, convert, gl_api};
use std::collections::HashMap;
use std::path::PathBuf;
use windows::Win32::Graphics::Gdi::GetDC;
use windows::Win32::Graphics::OpenGL::{
    ChoosePixelFormat, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR, SetPixelFormat, glGetString, wglCreateContext,
    wglMakeCurrent, GL_RENDERER, GL_VERSION,
};
use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW};
use windows::core::w;

fn context() {
    // SAFETY: a hidden window of a predefined class and a GL context on it, made current here.
    unsafe {
        let window = CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("sp_offline"), WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, None, None, None, None).expect("window");
        let dc = GetDC(Some(window));
        let pfd = PIXELFORMATDESCRIPTOR {
            nSize: size_of::<PIXELFORMATDESCRIPTOR>() as u16,
            nVersion: 1,
            dwFlags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            iPixelType: PFD_TYPE_RGBA,
            cColorBits: 32,
            cDepthBits: 24,
            ..Default::default()
        };
        let format = ChoosePixelFormat(dc, &pfd);
        SetPixelFormat(dc, format, &pfd).expect("pixel format");
        let gl = wglCreateContext(dc).expect("context");
        wglMakeCurrent(dc, gl).expect("current");
        let text = |name| std::ffi::CStr::from_ptr(glGetString(name).cast()).to_string_lossy().into_owned();
        println!("{} / {}", text(GL_RENDERER), text(GL_VERSION));
    }
}

fn first_lines(text: &str) -> String {
    text.lines().take(3).collect::<Vec<_>>().join(" | ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let base = PathBuf::from(args.first().expect("the game's base folder"));
    let dump = args.get(1).map(PathBuf::from);
    let out = args.get(2).map(PathBuf::from);
    context();
    let gl = gl_api::Gl::load().expect("GL entry points");
    let programs = archive::renderprogs(&base).expect("render programs");
    println!("{} render programs", programs.len());
    let outputs = |p: &archive::Renderprog| convert::outputs(&p.fragment).into_iter().enumerate().map(|(i, n)| (n, i as u32)).collect::<Vec<_>>();
    let (mut original_failed, mut converted_failed, mut stereo) = (0, 0, 0);
    let mut keys: HashMap<String, Vec<String>> = HashMap::new();
    for program in &programs {
        let original = gl.program(&program.vertex, &program.fragment, &[], &outputs(program));
        match gl.linked(original) {
            Ok(()) => {
                if let Some(key) = gl.binary(original).and_then(|(_, b)| gl_api::assembly_key(&b)) {
                    keys.entry(key).or_default().push(program.name.clone());
                }
            }
            Err(log) => {
                original_failed += 1;
                println!("{}: original does not link: {}", program.name, first_lines(&log));
            }
        }
        if convert::position_rows(&program.vertex).is_some() {
            stereo += 1;
        }
        let Some(vertex) = convert::stereo_vertex(&program.vertex) else {
            println!("{}: no main to convert", program.name);
            continue;
        };
        // Locations as the engine's program gave them (here: the original's own).
        let uniforms = gl.uniforms(original);
        let located = |u: &convert::Uniform| uniforms.iter().find(|a| a.name == u.name).and_then(|a| a.locations.first().copied()).filter(|&l| l >= 0);
        let vertex = convert::with_locations(&vertex, &located);
        let fragment = convert::with_locations(&program.fragment, &located);
        let converted = gl.program(&vertex, &fragment, &gl.attributes(original), &outputs(program));
        match gl.linked(converted) {
            Ok(()) => {
                let after = gl.uniforms(converted);
                for u in &uniforms {
                    let same = after.iter().find(|a| a.name == u.name).is_some_and(|a| a.locations == u.locations);
                    if !same {
                        println!("{}: {} moved", program.name, u.name);
                    }
                }
            }
            Err(log) => {
                converted_failed += 1;
                println!("{}: converted does not link: {}", program.name, first_lines(&log));
            }
        }
        if let Some(out) = &out {
            let _ = std::fs::create_dir_all(out);
            let _ = std::fs::write(out.join(format!("{}.vert", program.name)), &vertex);
            let _ = std::fs::write(out.join(format!("{}.frag", program.name)), &fragment);
        }
        // SAFETY: programs of our own.
        unsafe {
            (gl.delete_program)(original);
            (gl.delete_program)(converted);
        }
    }
    let shared = keys.values().filter(|names| names.len() > 1).count();
    println!(
        "originals failing: {original_failed}; converted failing: {converted_failed}; with plain position rows: {stereo}; distinct keys {} ({shared} shared by more than one program)",
        keys.len()
    );
    for names in keys.values().filter(|names| names.len() > 1).take(10) {
        println!("  same assembly: {}", names.join(", "));
    }
    let Some(dump) = dump else { return };
    let (mut matched, mut unmatched, mut files) = (0, 0, 0);
    for entry in std::fs::read_dir(&dump).expect("dump folder").flatten() {
        if entry.path().extension().is_none_or(|e| e != "bin") {
            continue;
        }
        files += 1;
        let Some(key) = std::fs::read(entry.path()).ok().and_then(|b| gl_api::assembly_key(&b)) else { continue };
        match keys.get(&key) {
            Some(_) => matched += 1,
            None => {
                unmatched += 1;
                if unmatched <= 5 {
                    let first = key.lines().take(3).collect::<Vec<_>>().join(" | ");
                    println!("  unmatched {}: {first}", entry.path().display());
                }
            }
        }
    }
    println!("engine binaries: {files}, matched {matched}, unmatched {unmatched}");
}
