//! The driver's GL 2+ entry points single pass calls (asked of the current context with
//! `wglGetProcAddress`), and the program work built on them: compiling a render program,
//! reading a program's binary and interface, and copying its uniform values.

use std::ffi::c_void;
use windows::Win32::Graphics::OpenGL::{wglGetCurrentContext, wglGetProcAddress};
use windows::core::PCSTR;

pub const GL_VERTEX_SHADER: u32 = 0x8B31;
pub const GL_FRAGMENT_SHADER: u32 = 0x8B30;
pub const GL_COMPILE_STATUS: u32 = 0x8B81;
pub const GL_LINK_STATUS: u32 = 0x8B82;
pub const GL_INFO_LOG_LENGTH: u32 = 0x8B84;
pub const GL_ACTIVE_UNIFORMS: u32 = 0x8B86;
pub const GL_ACTIVE_ATTRIBUTES: u32 = 0x8B89;
pub const GL_ACTIVE_UNIFORM_BLOCKS: u32 = 0x8A36;
pub const GL_UNIFORM_BLOCK_BINDING: u32 = 0x8A3F;
pub const GL_UNIFORM_BLOCK_INDEX: u32 = 0x8A3A;
pub const GL_PROGRAM_BINARY_LENGTH: u32 = 0x8741;
pub const GL_PROGRAM_BINARY_RETRIEVABLE_HINT: u32 = 0x8257;
pub const GL_COMPLETION_STATUS: u32 = 0x91B1;

pub const GL_DRAW_FRAMEBUFFER_BINDING: u32 = 0x8CA6;
pub const GL_SCISSOR_TEST: u32 = 0x0C11;
pub const GL_SCISSOR_BOX: u32 = 0x0C10;
pub const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const GL_DEPTH_ATTACHMENT: u32 = 0x8D00;
pub const GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: u32 = 0x8CD0;
pub const GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: u32 = 0x8CD1;
pub const GL_TEXTURE: u32 = 0x1702;
pub const GL_RENDERBUFFER: u32 = 0x8D41;
pub const GL_TEXTURE_WIDTH: u32 = 0x1000;
pub const GL_RENDERBUFFER_WIDTH: u32 = 0x8D42;

pub const GL_FLOAT: u32 = 0x1406;
pub const GL_FLOAT_VEC2: u32 = 0x8B50;
pub const GL_FLOAT_VEC3: u32 = 0x8B51;
pub const GL_FLOAT_VEC4: u32 = 0x8B52;
pub const GL_INT: u32 = 0x1404;
pub const GL_INT_VEC2: u32 = 0x8B53;
pub const GL_INT_VEC3: u32 = 0x8B54;
pub const GL_INT_VEC4: u32 = 0x8B55;
pub const GL_BOOL: u32 = 0x8B56;
pub const GL_FLOAT_MAT4: u32 = 0x8B5C;

macro_rules! entry_points {
    ($($field:ident: $name:literal fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        /// The entry points, loaded together ([`Gl::load`]).
        #[allow(non_snake_case)]
        pub struct Gl {
            $(pub $field: unsafe extern "system" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Gl {
            /// Every entry point from the current context's driver; the first missing one named.
            pub fn load() -> Result<Self, &'static str> {
                // SAFETY: queries this thread's current context.
                if unsafe { wglGetCurrentContext() }.is_invalid() {
                    return Err("no GL context");
                }
                Ok(Self { $($field: load($name).ok_or($name)?,)* })
            }
        }
    };
}

entry_points! {
    create_shader: "glCreateShader" fn(u32) -> u32;
    shader_source: "glShaderSource" fn(u32, i32, *const *const u8, *const i32);
    compile_shader: "glCompileShader" fn(u32);
    get_shader_iv: "glGetShaderiv" fn(u32, u32, *mut i32);
    get_shader_info_log: "glGetShaderInfoLog" fn(u32, i32, *mut i32, *mut u8);
    delete_shader: "glDeleteShader" fn(u32);
    create_program: "glCreateProgram" fn() -> u32;
    attach_shader: "glAttachShader" fn(u32, u32);
    detach_shader: "glDetachShader" fn(u32, u32);
    link_program: "glLinkProgram" fn(u32);
    get_program_iv: "glGetProgramiv" fn(u32, u32, *mut i32);
    get_program_info_log: "glGetProgramInfoLog" fn(u32, i32, *mut i32, *mut u8);
    delete_program: "glDeleteProgram" fn(u32);
    is_program: "glIsProgram" fn(u32) -> u8;
    program_parameter_i: "glProgramParameteri" fn(u32, u32, i32);
    bind_attrib_location: "glBindAttribLocation" fn(u32, u32, *const u8);
    bind_frag_data_location: "glBindFragDataLocation" fn(u32, u32, *const u8);
    get_attrib_location: "glGetAttribLocation" fn(u32, *const u8) -> i32;
    get_frag_data_location: "glGetFragDataLocation" fn(u32, *const u8) -> i32;
    get_uniform_location: "glGetUniformLocation" fn(u32, *const u8) -> i32;
    get_active_attrib: "glGetActiveAttrib" fn(u32, u32, i32, *mut i32, *mut i32, *mut u32, *mut u8);
    get_active_uniform: "glGetActiveUniform" fn(u32, u32, i32, *mut i32, *mut i32, *mut u32, *mut u8);
    get_active_uniforms_iv: "glGetActiveUniformsiv" fn(u32, i32, *const u32, u32, *mut i32);
    get_active_uniform_block_iv: "glGetActiveUniformBlockiv" fn(u32, u32, u32, *mut i32);
    uniform_block_binding: "glUniformBlockBinding" fn(u32, u32, u32);
    get_uniform_fv: "glGetUniformfv" fn(u32, i32, *mut f32);
    get_uniform_iv: "glGetUniformiv" fn(u32, i32, *mut i32);
    program_uniform_4fv: "glProgramUniform4fv" fn(u32, i32, i32, *const f32);
    program_uniform_3fv: "glProgramUniform3fv" fn(u32, i32, i32, *const f32);
    program_uniform_2fv: "glProgramUniform2fv" fn(u32, i32, i32, *const f32);
    program_uniform_1fv: "glProgramUniform1fv" fn(u32, i32, i32, *const f32);
    program_uniform_4iv: "glProgramUniform4iv" fn(u32, i32, i32, *const i32);
    program_uniform_3iv: "glProgramUniform3iv" fn(u32, i32, i32, *const i32);
    program_uniform_2iv: "glProgramUniform2iv" fn(u32, i32, i32, *const i32);
    program_uniform_1iv: "glProgramUniform1iv" fn(u32, i32, i32, *const i32);
    program_uniform_matrix_4fv: "glProgramUniformMatrix4fv" fn(u32, i32, i32, u8, *const f32);
    get_program_binary: "glGetProgramBinary" fn(u32, i32, *mut i32, *mut u32, *mut c_void);
    program_binary: "glProgramBinary" fn(u32, u32, *const c_void, i32);
    viewport_indexed_f: "glViewportIndexedf" fn(u32, f32, f32, f32, f32);
    scissor_indexed: "glScissorIndexed" fn(u32, i32, i32, i32, i32);
    get_integer_i_v: "glGetIntegeri_v" fn(u32, u32, *mut i32);
    get_named_framebuffer_attachment_parameter_iv: "glGetNamedFramebufferAttachmentParameteriv" fn(u32, u32, u32, *mut i32);
    get_texture_level_parameter_iv: "glGetTextureLevelParameteriv" fn(u32, i32, u32, *mut i32);
    get_named_renderbuffer_parameter_iv: "glGetNamedRenderbufferParameteriv" fn(u32, u32, *mut i32);
    is_framebuffer: "glIsFramebuffer" fn(u32) -> u8;
    bind_framebuffer: "glBindFramebuffer" fn(u32, u32);
    create_framebuffers: "glCreateFramebuffers" fn(i32, *mut u32);
    delete_framebuffers: "glDeleteFramebuffers" fn(i32, *const u32);
    named_framebuffer_texture: "glNamedFramebufferTexture" fn(u32, u32, u32, i32);
    named_framebuffer_texture_layer: "glNamedFramebufferTextureLayer" fn(u32, u32, u32, i32, i32);
    named_framebuffer_renderbuffer: "glNamedFramebufferRenderbuffer" fn(u32, u32, u32, u32);
    named_framebuffer_draw_buffers: "glNamedFramebufferDrawBuffers" fn(u32, i32, *const u32);
    named_framebuffer_read_buffer: "glNamedFramebufferReadBuffer" fn(u32, u32);
    check_named_framebuffer_status: "glCheckNamedFramebufferStatus" fn(u32, u32) -> u32;
    create_textures: "glCreateTextures" fn(u32, i32, *mut u32);
    gen_textures: "glGenTextures" fn(i32, *mut u32);
    delete_textures: "glDeleteTextures" fn(i32, *const u32);
    texture_storage_3d: "glTextureStorage3D" fn(u32, i32, u32, i32, i32, i32);
    texture_view: "glTextureView" fn(u32, u32, u32, u32, u32, u32, u32, u32);
    get_texture_parameter_iv: "glGetTextureParameteriv" fn(u32, u32, *mut i32);
    texture_parameter_i: "glTextureParameteri" fn(u32, u32, i32);
    copy_image_sub_data: "glCopyImageSubData" fn(u32, u32, i32, i32, i32, i32, u32, u32, i32, i32, i32, i32, i32, i32, i32);
    bind_multi_texture: "glBindMultiTextureEXT" fn(u32, u32, u32);
}

fn load<F: Copy>(name: &str) -> Option<F> {
    let text = format!("{name}\0");
    // SAFETY: asks the current context's driver for a function by NUL-terminated name.
    let function = unsafe { wglGetProcAddress(PCSTR(text.as_ptr())) }?;
    if (function as usize) < 4 {
        return None;
    }
    // SAFETY: the driver's function, of the type its specification gives.
    Some(unsafe { std::mem::transmute_copy(&function) })
}

/// The driver's address for `name` (as the engine's own loader got it), none where missing.
pub fn address(name: &str) -> Option<usize> {
    let text = format!("{name}\0");
    // SAFETY: as in `load`.
    let function = unsafe { wglGetProcAddress(PCSTR(text.as_ptr())) }?;
    (function as usize >= 4).then_some(function as usize)
}

/// `name` NUL-terminated, for the GL calls that take a C string.
pub fn c_name(name: &str) -> Vec<u8> {
    let mut out = name.as_bytes().to_vec();
    out.push(0);
    out
}

/// A uniform of a linked program: its name (an array's with `[0]` taken off), type, elements and
/// the location of each element.
#[derive(Clone, Debug)]
pub struct ActiveUniform {
    pub name: String,
    pub kind: u32,
    pub locations: Vec<i32>,
}

/// A uniform's components and whether they are integers (samplers are ints); none for a type the
/// copy does not handle.
pub fn components(kind: u32) -> Option<(usize, bool)> {
    Some(match kind {
        GL_FLOAT => (1, false),
        GL_FLOAT_VEC2 => (2, false),
        GL_FLOAT_VEC3 => (3, false),
        GL_FLOAT_VEC4 => (4, false),
        GL_FLOAT_MAT4 => (16, false),
        GL_INT | GL_BOOL => (1, true),
        GL_INT_VEC2 => (2, true),
        GL_INT_VEC3 => (3, true),
        GL_INT_VEC4 => (4, true),
        // Samplers (2D, cube, shadow, multisample, ...): a texture unit.
        0x8B5D..=0x8B64 | 0x9108..=0x910A | 0x8DC0..=0x8DD8 => (1, true),
        _ => return None,
    })
}

/// A uniform's value: each element's components, as floats or ints.
#[derive(Clone, Debug)]
pub enum Values {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
}

impl Gl {
    /// The width of what framebuffer `framebuffer` draws into (its depth attachment's, else its
    /// first colour attachment's); none for the window's own (0).
    pub fn framebuffer_width(&self, framebuffer: u32) -> Option<i32> {
        if framebuffer == 0 {
            return None;
        }
        for attachment in [GL_DEPTH_ATTACHMENT, GL_COLOR_ATTACHMENT0] {
            let (mut kind, mut name, mut width) = (0, 0, 0);
            // SAFETY: plain queries into locals.
            unsafe {
                (self.get_named_framebuffer_attachment_parameter_iv)(framebuffer, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &mut kind);
                (self.get_named_framebuffer_attachment_parameter_iv)(framebuffer, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &mut name);
                match kind as u32 {
                    GL_TEXTURE => (self.get_texture_level_parameter_iv)(name as u32, 0, GL_TEXTURE_WIDTH, &mut width),
                    GL_RENDERBUFFER => (self.get_named_renderbuffer_parameter_iv)(name as u32, GL_RENDERBUFFER_WIDTH, &mut width),
                    _ => continue,
                }
            }
            if width > 0 {
                return Some(width);
            }
        }
        None
    }

    /// A shader compiled from `source` (the log when it does not compile).
    pub fn shader(&self, kind: u32, source: &str) -> Result<u32, String> {
        // SAFETY: plain GL calls on the current context, with buffers of the lengths given.
        unsafe {
            let shader = (self.create_shader)(kind);
            let text = source.as_ptr();
            let length = source.len() as i32;
            (self.shader_source)(shader, 1, &text, &length);
            (self.compile_shader)(shader);
            Ok(shader)
        }
    }

    /// Whether `shader` compiled, with its log otherwise.
    pub fn compiled(&self, shader: u32) -> Result<(), String> {
        let mut status = 0;
        // SAFETY: as above.
        unsafe { (self.get_shader_iv)(shader, GL_COMPILE_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut length = 0;
        unsafe { (self.get_shader_iv)(shader, GL_INFO_LOG_LENGTH, &mut length) };
        let mut log = vec![0u8; length.max(1) as usize];
        let mut written = 0;
        unsafe { (self.get_shader_info_log)(shader, length.max(1), &mut written, log.as_mut_ptr()) };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    /// Whether `program` linked, with its log otherwise.
    pub fn linked(&self, program: u32) -> Result<(), String> {
        let mut status = 0;
        // SAFETY: as above.
        unsafe { (self.get_program_iv)(program, GL_LINK_STATUS, &mut status) };
        if status != 0 {
            return Ok(());
        }
        let mut length = 0;
        unsafe { (self.get_program_iv)(program, GL_INFO_LOG_LENGTH, &mut length) };
        let mut log = vec![0u8; length.max(1) as usize];
        let mut written = 0;
        unsafe { (self.get_program_info_log)(program, length.max(1), &mut written, log.as_mut_ptr()) };
        log.truncate(written.max(0) as usize);
        Err(String::from_utf8_lossy(&log).into_owned())
    }

    /// Whether the driver has finished compiling or linking (`KHR_parallel_shader_compile`): an
    /// object asked about before then would make the caller wait.
    pub fn program_done(&self, program: u32) -> bool {
        let mut done = 1;
        // SAFETY: as above; without the extension the query leaves `done` as it was.
        unsafe { (self.get_program_iv)(program, GL_COMPLETION_STATUS, &mut done) };
        done != 0
    }

    /// A program of a vertex and a fragment shader, linked with the attribute and output
    /// locations given (the shaders are let go after the link). The link may still be running
    /// ([`Gl::program_done`]); its result is [`Gl::linked`].
    pub fn program(&self, vertex: &str, fragment: &str, attributes: &[(String, u32)], outputs: &[(String, u32)]) -> u32 {
        // SAFETY: plain GL calls on the current context with NUL-terminated names.
        unsafe {
            let program = (self.create_program)();
            let shaders = [self.shader(GL_VERTEX_SHADER, vertex), self.shader(GL_FRAGMENT_SHADER, fragment)];
            for shader in shaders.iter().flatten() {
                (self.attach_shader)(program, *shader);
            }
            for (name, location) in attributes {
                (self.bind_attrib_location)(program, *location, c_name(name).as_ptr());
            }
            for (name, location) in outputs {
                (self.bind_frag_data_location)(program, *location, c_name(name).as_ptr());
            }
            (self.program_parameter_i)(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, 1);
            (self.link_program)(program);
            for shader in shaders.iter().flatten() {
                (self.detach_shader)(program, *shader);
                (self.delete_shader)(*shader);
            }
            program
        }
    }

    /// A linked program's binary and its format.
    pub fn binary(&self, program: u32) -> Option<(u32, Vec<u8>)> {
        let mut length = 0;
        // SAFETY: plain queries into a buffer of the length given.
        unsafe { (self.get_program_iv)(program, GL_PROGRAM_BINARY_LENGTH, &mut length) };
        if length <= 0 {
            return None;
        }
        let mut bytes = vec![0u8; length as usize];
        let (mut written, mut format) = (0, 0u32);
        unsafe { (self.get_program_binary)(program, length, &mut written, &mut format, bytes.as_mut_ptr().cast()) };
        bytes.truncate(written.clamp(0, length) as usize);
        (!bytes.is_empty()).then_some((format, bytes))
    }

    /// A linked program's active attributes and their locations.
    pub fn attributes(&self, program: u32) -> Vec<(String, u32)> {
        let mut count = 0;
        // SAFETY: plain queries into buffers of the lengths given.
        unsafe { (self.get_program_iv)(program, GL_ACTIVE_ATTRIBUTES, &mut count) };
        let mut out = Vec::new();
        for index in 0..count.max(0) as u32 {
            let mut name = [0u8; 256];
            let (mut length, mut size, mut kind) = (0, 0, 0);
            unsafe { (self.get_active_attrib)(program, index, name.len() as i32, &mut length, &mut size, &mut kind, name.as_mut_ptr()) };
            let name = String::from_utf8_lossy(&name[..length.clamp(0, 255) as usize]).into_owned();
            if name.starts_with("gl_") {
                continue;
            }
            let location = unsafe { (self.get_attrib_location)(program, c_name(&name).as_ptr()) };
            if let Ok(location) = u32::try_from(location) {
                out.push((name, location));
            }
        }
        out
    }

    /// A linked program's plain (not block) uniforms.
    pub fn uniforms(&self, program: u32) -> Vec<ActiveUniform> {
        let mut count = 0;
        // SAFETY: plain queries into buffers of the lengths given.
        unsafe { (self.get_program_iv)(program, GL_ACTIVE_UNIFORMS, &mut count) };
        let mut out = Vec::new();
        for index in 0..count.max(0) as u32 {
            let mut block = -1;
            unsafe { (self.get_active_uniforms_iv)(program, 1, &index, GL_UNIFORM_BLOCK_INDEX, &mut block) };
            if block >= 0 {
                continue;
            }
            let mut name = [0u8; 256];
            let (mut length, mut size, mut kind) = (0, 0, 0u32);
            unsafe { (self.get_active_uniform)(program, index, name.len() as i32, &mut length, &mut size, &mut kind, name.as_mut_ptr()) };
            let full = String::from_utf8_lossy(&name[..length.clamp(0, 255) as usize]).into_owned();
            let base = full.strip_suffix("[0]").unwrap_or(&full).to_owned();
            let locations = (0..size.max(1))
                .map(|element| {
                    let element_name = if size > 1 || full.ends_with("[0]") { format!("{base}[{element}]") } else { base.clone() };
                    unsafe { (self.get_uniform_location)(program, c_name(&element_name).as_ptr()) }
                })
                .collect();
            out.push(ActiveUniform { name: base, kind, locations });
        }
        out
    }

    /// Each uniform block's binding, by block index.
    pub fn block_bindings(&self, program: u32) -> Vec<u32> {
        let mut count = 0;
        // SAFETY: plain queries.
        unsafe { (self.get_program_iv)(program, GL_ACTIVE_UNIFORM_BLOCKS, &mut count) };
        (0..count.max(0) as u32)
            .map(|block| {
                let mut binding = 0;
                unsafe { (self.get_active_uniform_block_iv)(program, block, GL_UNIFORM_BLOCK_BINDING, &mut binding) };
                binding.max(0) as u32
            })
            .collect()
    }

    /// The current values of `uniform` in `program` (one entry per element), none for a type the
    /// copy does not handle.
    pub fn values(&self, program: u32, uniform: &ActiveUniform) -> Option<Vec<Values>> {
        let (components, ints) = components(uniform.kind)?;
        let mut out = Vec::with_capacity(uniform.locations.len());
        for &location in &uniform.locations {
            if location < 0 {
                continue;
            }
            // SAFETY: plain queries into buffers as large as the type's components.
            out.push(if ints {
                let mut value = vec![0i32; components];
                unsafe { (self.get_uniform_iv)(program, location, value.as_mut_ptr()) };
                Values::Ints(value)
            } else {
                let mut value = vec![0f32; components];
                unsafe { (self.get_uniform_fv)(program, location, value.as_mut_ptr()) };
                Values::Floats(value)
            });
        }
        Some(out)
    }

    /// Sets `uniform` in `program` to `values` (as [`Gl::values`] read them).
    pub fn set_values(&self, program: u32, uniform: &ActiveUniform, values: &[Values]) {
        let locations = uniform.locations.iter().filter(|&&l| l >= 0);
        for (&location, value) in locations.zip(values) {
            // SAFETY: plain GL calls with buffers as large as the type's components.
            unsafe {
                match (value, uniform.kind) {
                    (Values::Floats(v), GL_FLOAT_MAT4) => (self.program_uniform_matrix_4fv)(program, location, 1, 0, v.as_ptr()),
                    (Values::Floats(v), _) => match v.len() {
                        1 => (self.program_uniform_1fv)(program, location, 1, v.as_ptr()),
                        2 => (self.program_uniform_2fv)(program, location, 1, v.as_ptr()),
                        3 => (self.program_uniform_3fv)(program, location, 1, v.as_ptr()),
                        _ => (self.program_uniform_4fv)(program, location, 1, v.as_ptr()),
                    },
                    (Values::Ints(v), _) => match v.len() {
                        1 => (self.program_uniform_1iv)(program, location, 1, v.as_ptr()),
                        2 => (self.program_uniform_2iv)(program, location, 1, v.as_ptr()),
                        3 => (self.program_uniform_3iv)(program, location, 1, v.as_ptr()),
                        _ => (self.program_uniform_4iv)(program, location, 1, v.as_ptr()),
                    },
                }
            }
        }
    }
}

/// The NVIDIA assembly a program binary carries (each stage's `!!NV...` text), with attribute
/// numbers taken out so programs linked with different attribute locations compare equal: what
/// identifies which render program an engine program was made from.
pub fn assembly_key(binary: &[u8]) -> Option<String> {
    const START: &[u8] = b"!!NV";
    let mut out = String::new();
    let mut from = 0;
    while let Some(found) = binary.get(from..)?.windows(START.len()).position(|w| w == START) {
        let at = from + found;
        let end = binary[at..].iter().position(|&b| b == 0).map_or(binary.len(), |n| at + n);
        let text = String::from_utf8_lossy(&binary[at..end]);
        for line in text.lines() {
            // `vertex.attrib[3]` and `vertex.attrib[0..10]` depend on the attribute locations.
            let mut line = line.to_owned();
            while let Some(i) = line.find("attrib[") {
                let close = line[i..].find(']').map_or(line.len(), |n| i + n + 1);
                line.replace_range(i..close, "attrib#");
            }
            out.push_str(&line);
            out.push('\n');
        }
        out.push('\0');
        from = end;
    }
    (!out.is_empty()).then_some(out)
}
