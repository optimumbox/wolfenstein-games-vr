//! The layered eye targets: each of the engine's eye textures stood in for by a two-layer array
//! (layer 0 the left eye, layer 1 the right), and each engine framebuffer drawing into one by
//! framebuffers of ours on the arrays.
//!
//! The engine draws its eyes one after the other into the same textures (one eye's pass, then the
//! other's). With the arrays in their place a scene draw of the left eye's pass fills both layers
//! (the converted program's second view goes to layer 1), and the right eye's pass finds its
//! layer drawn. Each engine framebuffer gets up to three of ours: one with every attachment
//! layered (for the draws filling both eyes, where all its attachments are eye textures), and one
//! per layer. Each engine texture gets a 2D view per layer, given in its place to the draws of
//! that eye's pass.
//!
//! The engine's own textures are left as they are; [`Layers::copy_back`] copies a layer into them
//! for what reads them outside the engine's draws (the frame's depth, at the present).

use super::gl_api::{self, Gl};
use std::collections::HashMap;

const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_TEXTURE_2D_ARRAY: u32 = 0x8C1A;
const GL_TEXTURE_HEIGHT: u32 = 0x1001;
const GL_TEXTURE_INTERNAL_FORMAT: u32 = 0x1003;
const GL_TEXTURE_SAMPLES: u32 = 0x9106;
const GL_TEXTURE_TARGET: u32 = 0x1006;
const GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: u32 = 0x8CD2;
const GL_STENCIL_ATTACHMENT: u32 = 0x8D20;
const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
const GL_READ_FRAMEBUFFER_BINDING: u32 = 0x8CAA;
const GL_DRAW_BUFFER0: u32 = 0x8825;
const GL_READ_BUFFER: u32 = 0x0C02;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
/// The sampling state a view takes from the engine's texture.
const SAMPLING: [u32; 7] = [0x2801, 0x2800, 0x2802, 0x2803, 0x884C, 0x884D, 0x90EA];
/// The colour attachments looked at.
const COLOURS: u32 = 8;
/// The framebuffer names swept for those drawing into an eye texture.
const FRAMEBUFFERS_SWEPT: u32 = 1024;

/// An engine eye texture's stand-in.
struct Layered {
    array: u32,
    views: [u32; 2],
    size: (i32, i32),
    format: u32,
}

/// An engine framebuffer's stand-ins: all layers (when every attachment is an eye texture), and
/// one per layer.
#[derive(Clone, Copy)]
pub struct Stand {
    pub layered: Option<u32>,
    pub layer: [u32; 2],
}

pub struct Layers {
    textures: HashMap<u32, Layered>,
    framebuffers: HashMap<u32, Stand>,
    /// For each engine framebuffer stood in for: the eye texture its first draw buffer writes (if
    /// one), and every eye texture attached.
    drawn: HashMap<u32, (Option<u32>, Vec<u32>)>,
    /// The engine's scene depth (the eye texture at a depth attachment).
    depth: Option<u32>,
    /// A copy of a layer of the scene depth (a one-layer array, and its 2D view).
    depth_copy: Option<(u32, u32)>,
    /// Both eyes' scene depth side by side (a one-layer array twice the eye's width, and its 2D view).
    depth_pair: Option<(u32, u32)>,
    /// A copy of an eye texture's both layers ([`Self::keep`]): the eye texture, and the array.
    kept: Option<(u32, u32)>,
}

/// An attachment of an engine framebuffer.
#[derive(Clone, Copy)]
struct Attachment {
    point: u32,
    kind: u32,
    name: u32,
    level: i32,
}

fn attachments(gl: &Gl, framebuffer: u32) -> Vec<Attachment> {
    let points = (0..COLOURS).map(|i| gl_api::GL_COLOR_ATTACHMENT0 + i).chain([gl_api::GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT]);
    points
        .filter_map(|point| {
            let (mut kind, mut name, mut level) = (0, 0, 0);
            // SAFETY: plain queries into locals.
            unsafe {
                (gl.get_named_framebuffer_attachment_parameter_iv)(framebuffer, point, gl_api::GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &mut kind);
                if kind == 0 {
                    return None;
                }
                (gl.get_named_framebuffer_attachment_parameter_iv)(framebuffer, point, gl_api::GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &mut name);
                if kind as u32 == gl_api::GL_TEXTURE {
                    (gl.get_named_framebuffer_attachment_parameter_iv)(framebuffer, point, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &mut level);
                }
            }
            Some(Attachment { point, kind: kind as u32, name: name as u32, level })
        })
        .collect()
}

fn texture_parameter(gl: &Gl, texture: u32, pname: u32) -> i32 {
    let mut value = 0;
    // SAFETY: a plain query into a local.
    unsafe { (gl.get_texture_parameter_iv)(texture, pname, &mut value) };
    value
}

fn level_parameter(gl: &Gl, texture: u32, pname: u32) -> i32 {
    let mut value = 0;
    // SAFETY: a plain query into a local.
    unsafe { (gl.get_texture_level_parameter_iv)(texture, 0, pname, &mut value) };
    value
}

/// Engine texture `texture` as found now: its size, format and target, and the framebuffers it is
/// attached to (and where): for the log, once.
pub fn describe(gl: &Gl, texture: u32) -> String {
    let size = (level_parameter(gl, texture, gl_api::GL_TEXTURE_WIDTH), level_parameter(gl, texture, GL_TEXTURE_HEIGHT));
    let format = level_parameter(gl, texture, GL_TEXTURE_INTERNAL_FORMAT);
    let target = texture_parameter(gl, texture, GL_TEXTURE_TARGET);
    let mut on = Vec::new();
    for framebuffer in 1..=FRAMEBUFFERS_SWEPT {
        // SAFETY: a plain query.
        if unsafe { (gl.is_framebuffer)(framebuffer) } == 0 {
            continue;
        }
        for a in attachments(gl, framebuffer) {
            if a.kind == gl_api::GL_TEXTURE && a.name == texture {
                on.push(format!("fb {framebuffer} at {:#x} level {}", a.point, a.level));
            }
        }
    }
    format!("texture {texture}: {}x{} format {format:#x} target {target:#x}, attached to {}", size.0, size.1, if on.is_empty() { "no framebuffer".to_owned() } else { on.join(", ") })
}

impl Layers {
    /// Stand-ins for the eye textures (`size`) attached to `seen` (the framebuffers the scene was
    /// drawn into), and for every framebuffer drawing into one of them.
    pub fn build(gl: &Gl, seen: &[u32], size: (i32, i32)) -> Result<Self, String> {
        let mut layers = Self { textures: HashMap::new(), framebuffers: HashMap::new(), drawn: HashMap::new(), depth: None, depth_copy: None, depth_pair: None, kept: None };
        let made = layers.make(gl, seen, size);
        if made.is_err() {
            layers.release(gl);
        }
        made.map(|()| layers)
    }

    fn make(&mut self, gl: &Gl, seen: &[u32], size: (i32, i32)) -> Result<(), String> {
        for &framebuffer in seen {
            for attachment in attachments(gl, framebuffer) {
                if attachment.kind != gl_api::GL_TEXTURE || attachment.level != 0 || self.textures.contains_key(&attachment.name) {
                    continue;
                }
                let depth = attachment.point == gl_api::GL_DEPTH_ATTACHMENT;
                let texture = attachment.name;
                let own = (level_parameter(gl, texture, gl_api::GL_TEXTURE_WIDTH), level_parameter(gl, texture, GL_TEXTURE_HEIGHT));
                if own != size {
                    continue;
                }
                let target = texture_parameter(gl, texture, GL_TEXTURE_TARGET) as u32;
                if target != GL_TEXTURE_2D || level_parameter(gl, texture, GL_TEXTURE_SAMPLES) > 1 {
                    return Err(format!("eye texture {texture} is not a plain 2D texture (target {target:#x})"));
                }
                let format = level_parameter(gl, texture, GL_TEXTURE_INTERNAL_FORMAT) as u32;
                let (mut array, mut views) = (0, [0u32; 2]);
                // SAFETY: new objects of ours, with storage like the engine's texture.
                unsafe {
                    (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut array);
                    (gl.texture_storage_3d)(array, 1, format, size.0, size.1, 2);
                    // Sampled whole too, by the programs reading the scene's maps layered.
                    for pname in SAMPLING {
                        (gl.texture_parameter_i)(array, pname, texture_parameter(gl, texture, pname));
                    }
                    (gl.gen_textures)(2, views.as_mut_ptr());
                    for (layer, &view) in views.iter().enumerate() {
                        (gl.texture_view)(view, GL_TEXTURE_2D, array, format, 0, 1, layer as u32, 1);
                        for pname in SAMPLING {
                            (gl.texture_parameter_i)(view, pname, texture_parameter(gl, texture, pname));
                        }
                    }
                }
                self.textures.insert(texture, Layered { array, views, size, format });
                if depth && self.depth.is_none() {
                    self.depth = Some(texture);
                }
            }
        }
        if self.textures.is_empty() {
            return Err("no eye textures on the scene's framebuffers".to_owned());
        }
        for framebuffer in 1..=FRAMEBUFFERS_SWEPT {
            // SAFETY: a plain query.
            if unsafe { (gl.is_framebuffer)(framebuffer) } == 0 {
                continue;
            }
            let attached = attachments(gl, framebuffer);
            if !attached.iter().any(|a| a.kind == gl_api::GL_TEXTURE && self.textures.contains_key(&a.name)) {
                continue;
            }
            let buffers = draw_buffers(gl, framebuffer);
            let all_eye = attached.iter().all(|a| a.kind == gl_api::GL_TEXTURE && a.level == 0 && self.textures.contains_key(&a.name));
            let layered = if all_eye { Some(self.stand_in(gl, &attached, &buffers, None)?) } else { None };
            let layer = [self.stand_in(gl, &attached, &buffers, Some(0))?, self.stand_in(gl, &attached, &buffers, Some(1))?];
            self.framebuffers.insert(framebuffer, Stand { layered, layer });
            let first = buffers.0.first().and_then(|&point| attached.iter().find(|a| a.point == point && a.kind == gl_api::GL_TEXTURE && a.level == 0)).map(|a| a.name).filter(|t| self.textures.contains_key(t));
            let eyes = attached.iter().filter(|a| a.kind == gl_api::GL_TEXTURE && self.textures.contains_key(&a.name)).map(|a| a.name).collect();
            self.drawn.insert(framebuffer, (first, eyes));
        }
        Ok(())
    }

    /// A framebuffer of ours like the engine's `attached`, its eye textures' arrays attached at
    /// `layer` (all layers with none).
    fn stand_in(&self, gl: &Gl, attached: &[Attachment], buffers: &(Vec<u32>, u32), layer: Option<i32>) -> Result<u32, String> {
        let mut framebuffer = 0;
        // SAFETY: a new framebuffer of ours, given the engine's attachments or our arrays.
        unsafe {
            (gl.create_framebuffers)(1, &mut framebuffer);
            for a in attached {
                match (self.textures.get(&a.name).filter(|_| a.kind == gl_api::GL_TEXTURE), layer) {
                    (Some(eye), None) => (gl.named_framebuffer_texture)(framebuffer, a.point, eye.array, 0),
                    (Some(eye), Some(layer)) => (gl.named_framebuffer_texture_layer)(framebuffer, a.point, eye.array, 0, layer),
                    (None, _) if a.kind == gl_api::GL_TEXTURE => (gl.named_framebuffer_texture)(framebuffer, a.point, a.name, a.level),
                    (None, _) => (gl.named_framebuffer_renderbuffer)(framebuffer, a.point, gl_api::GL_RENDERBUFFER, a.name),
                }
            }
            (gl.named_framebuffer_draw_buffers)(framebuffer, buffers.0.len() as i32, buffers.0.as_ptr());
            (gl.named_framebuffer_read_buffer)(framebuffer, buffers.1);
            let status = (gl.check_named_framebuffer_status)(framebuffer, GL_DRAW_FRAMEBUFFER);
            if status != GL_FRAMEBUFFER_COMPLETE {
                (gl.delete_framebuffers)(1, &framebuffer);
                return Err(format!("a stand-in framebuffer (layer {layer:?}) is incomplete: {status:#x}"));
            }
        }
        Ok(framebuffer)
    }

    /// The engine framebuffers stood in for, with ours.
    pub fn stand(&self, framebuffer: u32) -> Option<Stand> {
        self.framebuffers.get(&framebuffer).copied()
    }

    /// The array standing in for an engine eye texture.
    pub fn array(&self, texture: u32) -> Option<u32> {
        self.textures.get(&texture).map(|t| t.array)
    }

    /// The eye texture engine framebuffer `framebuffer`'s first draw buffer writes, if one.
    pub fn first_colour(&self, framebuffer: u32) -> Option<u32> {
        self.drawn.get(&framebuffer)?.0
    }

    /// Whether engine framebuffer `framebuffer` has eye texture `texture` attached.
    pub fn attaches(&self, framebuffer: u32, texture: u32) -> bool {
        self.drawn.get(&framebuffer).is_some_and(|(_, eyes)| eyes.contains(&texture))
    }

    /// The view of an engine eye texture's `layer`.
    pub fn view(&self, texture: u32, layer: usize) -> Option<u32> {
        self.textures.get(&texture).map(|t| t.views[layer])
    }

    /// What a texture of ours is: the engine eye texture and layer it is a view of, or the engine
    /// eye texture it is the array of (layer none).
    pub fn ours(&self, texture: u32) -> Option<(u32, Option<usize>)> {
        self.textures.iter().find_map(|(&engine, t)| {
            if t.array == texture {
                Some((engine, None))
            } else {
                t.views.iter().position(|&v| v == texture).map(|layer| (engine, Some(layer)))
            }
        })
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        (self.textures.len(), self.framebuffers.len(), self.framebuffers.values().filter(|s| s.layered.is_some()).count())
    }

    /// Every eye texture's `layer` copied into the engine's own texture.
    pub fn copy_back(&self, gl: &Gl, layer: usize) {
        for (&texture, eye) in &self.textures {
            // SAFETY: a copy between two textures of the same format and size.
            unsafe {
                (gl.copy_image_sub_data)(eye.array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer as i32, texture, GL_TEXTURE_2D, 0, 0, 0, 0, eye.size.0, eye.size.1, 1);
            }
        }
    }

    /// The scene depth's `layer` copied into a texture of ours sampled like the engine's `like`
    /// (the depth the decals read): its 2D view; none without a layered scene depth.
    pub fn depth_copy(&mut self, gl: &Gl, layer: usize, like: u32) -> Option<u32> {
        let eye = self.textures.get(&self.depth?)?;
        let (array, view) = match self.depth_copy {
            Some(made) => made,
            None => {
                let (mut array, mut view) = (0, 0);
                // SAFETY: new objects of ours, with storage like the scene depth's.
                unsafe {
                    (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut array);
                    (gl.texture_storage_3d)(array, 1, eye.format, eye.size.0, eye.size.1, 1);
                    (gl.gen_textures)(1, &mut view);
                    (gl.texture_view)(view, GL_TEXTURE_2D, array, eye.format, 0, 1, 0, 1);
                    for pname in SAMPLING {
                        (gl.texture_parameter_i)(view, pname, texture_parameter(gl, like, pname));
                    }
                }
                self.depth_copy = Some((array, view));
                (array, view)
            }
        };
        // SAFETY: a copy between two textures of the same format and size.
        unsafe {
            (gl.copy_image_sub_data)(eye.array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer as i32, array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, eye.size.0, eye.size.1, 1);
        }
        Some(view)
    }

    /// Both eyes' scene depth side by side, left eye first, as the presented frame has their
    /// colour (the upscaler's depth): its 2D view and size. The engine's own depth texture holds
    /// one eye only (the one copied back), and DLSS stretched it over both eyes: the camera's motion
    /// vectors were right for turning (no depth in them) and wrong for walking.
    pub fn depth_pair(&mut self, gl: &Gl) -> Option<(u32, (i32, i32))> {
        let eye = self.textures.get(&self.depth?)?;
        let (array, view) = match self.depth_pair {
            Some(made) => made,
            None => {
                let (mut array, mut view) = (0, 0);
                // SAFETY: new objects of ours, with storage like the scene depth's.
                unsafe {
                    (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut array);
                    (gl.texture_storage_3d)(array, 1, eye.format, eye.size.0 * 2, eye.size.1, 1);
                    (gl.gen_textures)(1, &mut view);
                    (gl.texture_view)(view, GL_TEXTURE_2D, array, eye.format, 0, 1, 0, 1);
                }
                self.depth_pair = Some((array, view));
                (array, view)
            }
        };
        // SAFETY: copies between textures of the same format, each layer into its half.
        unsafe {
            for layer in 0..2 {
                (gl.copy_image_sub_data)(eye.array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, array, GL_TEXTURE_2D_ARRAY, 0, layer * eye.size.0, 0, 0, eye.size.0, eye.size.1, 1);
            }
        }
        Some((view, (eye.size.0 * 2, eye.size.1)))
    }

    /// Both layers of eye texture `texture` copied into an array of ours sampled the same way (the
    /// self-shadow, kept before the scene is drawn over it): that array.
    pub fn keep(&mut self, gl: &Gl, texture: u32) -> Option<u32> {
        let eye = self.textures.get(&texture)?;
        let array = match self.kept {
            Some((kept, array)) if kept == texture => array,
            _ => {
                if let Some((_, old)) = self.kept.take() {
                    // SAFETY: our own texture.
                    unsafe { (gl.delete_textures)(1, &old) };
                }
                let mut array = 0;
                // SAFETY: a new texture of ours, with storage and sampling like the eye texture's array.
                unsafe {
                    (gl.create_textures)(GL_TEXTURE_2D_ARRAY, 1, &mut array);
                    (gl.texture_storage_3d)(array, 1, eye.format, eye.size.0, eye.size.1, 2);
                    for pname in SAMPLING {
                        (gl.texture_parameter_i)(array, pname, texture_parameter(gl, eye.array, pname));
                    }
                }
                self.kept = Some((texture, array));
                array
            }
        };
        // SAFETY: a copy between two arrays of the same format and size, both layers.
        unsafe {
            (gl.copy_image_sub_data)(eye.array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, array, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, eye.size.0, eye.size.1, 2);
        }
        Some(array)
    }

    /// The array [`Self::keep`] last copied into, once made.
    pub fn kept(&self) -> Option<u32> {
        self.kept.map(|(_, array)| array)
    }

    /// The copy of the scene depth's layer ([`Self::depth_copy`]), once made.
    pub fn depth_copy_view(&self) -> Option<u32> {
        self.depth_copy.map(|(_, view)| view)
    }

    /// Every object of ours deleted.
    pub fn release(&mut self, gl: &Gl) {
        let framebuffers: Vec<u32> = self.framebuffers.values().flat_map(|s| s.layered.into_iter().chain(s.layer)).collect();
        let textures: Vec<u32> = self.textures.values().flat_map(|t| std::iter::once(t.array).chain(t.views)).chain(self.depth_copy.take().into_iter().flat_map(|(a, v)| [v, a])).chain(self.depth_pair.take().into_iter().flat_map(|(a, v)| [v, a])).chain(self.kept.take().map(|(_, a)| a)).collect();
        // SAFETY: our own objects, unbound by the caller.
        unsafe {
            (gl.delete_framebuffers)(framebuffers.len() as i32, framebuffers.as_ptr());
            (gl.delete_textures)(textures.len() as i32, textures.as_ptr());
        }
        self.framebuffers.clear();
        self.drawn.clear();
        self.textures.clear();
    }
}

/// An engine framebuffer's draw buffers and read buffer (read with it bound, the bindings put back).
fn draw_buffers(gl: &Gl, framebuffer: u32) -> (Vec<u32>, u32) {
    let (mut draw, mut read) = (0, 0);
    let mut buffers = Vec::new();
    let mut read_buffer = 0;
    // SAFETY: state queries into locals; the bindings put back as they were.
    unsafe {
        windows::Win32::Graphics::OpenGL::glGetIntegerv(gl_api::GL_DRAW_FRAMEBUFFER_BINDING, &mut draw);
        windows::Win32::Graphics::OpenGL::glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &mut read);
        (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, framebuffer);
        (gl.bind_framebuffer)(0x8CA8, framebuffer);
        for i in 0..COLOURS {
            let mut buffer = 0;
            windows::Win32::Graphics::OpenGL::glGetIntegerv(GL_DRAW_BUFFER0 + i, &mut buffer);
            buffers.push(buffer as u32);
        }
        windows::Win32::Graphics::OpenGL::glGetIntegerv(GL_READ_BUFFER, &mut read_buffer);
        (gl.bind_framebuffer)(GL_DRAW_FRAMEBUFFER, draw as u32);
        (gl.bind_framebuffer)(0x8CA8, read as u32);
    }
    while buffers.last() == Some(&0) {
        buffers.pop();
    }
    (buffers, read_buffer as u32)
}
