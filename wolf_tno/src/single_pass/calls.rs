//! The game's GL calls that move data on the GPU without drawing (reading the frame back, mapping
//! and copying buffers, fences), each stamped by [`super::pieces`] before it goes to the driver, so
//! the GPU time after one is put down to it and not to the draw before it.

use monaka_hook::{InFlight, Original};
use std::ffi::c_void;

macro_rules! stamped {
    ($original:ident, $detour:ident, $name:literal, fn($($arg:ident: $ty:ty),*) $(-> $ret:ty)?) => {
        pub static $original: Original<unsafe extern "system" fn($($ty),*) $(-> $ret)?> = Original::new();
        pub unsafe extern "system" fn $detour($($arg: $ty),*) $(-> $ret)? {
            let flight = InFlight::enter();
            super::pieces::call($name);
            // SAFETY: forwards the game's own call.
            unsafe { $original.get(&flight)($($arg),*) }
        }
    };
}

stamped!(READ_PIXELS, read_pixels, "glReadPixels", fn(x: i32, y: i32, w: i32, h: i32, format: u32, kind: u32, pixels: *mut c_void));
stamped!(GET_TEX_IMAGE, get_tex_image, "glGetTexImage", fn(target: u32, level: i32, format: u32, kind: u32, pixels: *mut c_void));
stamped!(MAP_BUFFER, map_buffer, "glMapBuffer", fn(target: u32, access: u32) -> *mut c_void);
stamped!(MAP_BUFFER_RANGE, map_buffer_range, "glMapBufferRange", fn(target: u32, offset: isize, length: isize, access: u32) -> *mut c_void);
stamped!(UNMAP_BUFFER, unmap_buffer, "glUnmapBuffer", fn(target: u32) -> u8);
stamped!(COPY_BUFFER_SUB_DATA, copy_buffer_sub_data, "glCopyBufferSubData", fn(read: u32, write: u32, from: isize, to: isize, size: isize));
stamped!(GENERATE_MIPMAP, generate_mipmap, "glGenerateMipmap", fn(target: u32));
stamped!(FENCE_SYNC, fence_sync, "glFenceSync", fn(condition: u32, flags: u32) -> *const c_void);
stamped!(WAIT_SYNC, wait_sync, "glWaitSync", fn(sync: *const c_void, flags: u32, timeout: u64));
stamped!(FINISH, finish, "glFinish", fn());
stamped!(FLUSH, flush, "glFlush", fn());
stamped!(DISPATCH_COMPUTE, dispatch_compute, "glDispatchCompute", fn(x: u32, y: u32, z: u32));
stamped!(CLEAR_BUFFER_FV, clear_buffer_fv, "glClearBufferfv", fn(buffer: u32, draw: i32, value: *const f32));
