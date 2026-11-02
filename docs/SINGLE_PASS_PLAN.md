# Single-pass stereo for Wolfenstein (id Tech 5, OpenGL): plan

Goal: the engine walks and draws the scene once a frame, and each draw lands in both eyes. Today's
same-frame stereo draws the scene twice (the left eye's scene call, then the right's) into the two
halves of a double-width window, and the render thread's second walk costs frame time.

## What the two reference ports teach

- **KF2VR** (`KF2VR-src/docs/SINGLE_PASS.md`): UE3 already supports two views in one view family,
  so its single pass needed no shader conversion. Its lessons are about one-view state:
  - occlusion queries per eye;
  - full-screen quads that assume x = 0;
  - screen-space passes (HBAO+, SSR) that read the double-width buffer as one view;
  - clears per eye rectangle;
  - DLSS per eye, fed real world depth.

  Also, each fix gets its own switch.
- **RDRVR** (`RDRVR-src/docs/SINGLE_PASS_PLAN.md`) is the shader-conversion reference:
  - It dumps every pipeline's shaders, finds which constant rows hold the view (per-pass
    view-projection, camera position), and rewrites each vertex shader so view 1 reads eye 2's
    rows.
  - One draw then covers both eyes (D3D12 view instancing), and the game's second walk records no
    draws.
  - 133 of 145 vertex shaders converted. The rest read no view block (full-screen and UI passes).
  - The lessons: find the view-dependent constants by diffing pass 1 against pass 2; keep the
    sequential path as the reference and fallback; compare each eye against that eye drawn alone.

## The OpenGL equivalent

The driver (RTX 4090, 616.64) offers `GL_NV_stereo_view_rendering` with `GL_NV_viewport_array2`, and
`GL_OVR_multiview2` (both listed in qconsole.log).

`NV_stereo_view_rendering` fits the double-width window as it is:
- A vertex shader writes `gl_Position` for the left eye and `gl_SecondaryPositionNV` for the right.
- Viewport 0 is the left half and viewport 1 the right half (`gl_ViewportMask[0] = 1`,
  `gl_SecondaryViewportMaskNV[0] = 2`).
- One draw rasterises into both halves. No layered targets are needed, so the window, the copy-out,
  DLSS per eye and the viewer stay as they are.

## The shaders

- The engine builds its shaders from `renderprogs` decls and can dump its generated GLSL itself:
  - `r_dumpGeneratedGLSL`, `generated/glsl`;
  - an override path: `r_renderProgProdOverridePath`, `reloadRenderProgFromOveride`.
- The executable is Steam-DRM wrapped (`.bind`), so its code cannot be read from the file.
- Monaka attaches at the menu, after the programs were made. So **`probe_shaders=1`**
  (`research/shaders.rs`) reads them back out of the driver: it sweeps the GL names on the render
  thread and writes, to the run's `shaders\` folder, each program's GLSL sources, the text of each
  assembly program (the engine also uses `GL_ARB_vertex_program`), or the binary where no source
  is attached.
- What the dump shows decides the converter:
  - **GLSL**: a text pass (Python, like RDR's `tools/sp`), then the converted source compiled into
    the same program name at runtime (`glShaderSource` + `glLinkProgram`), uniform locations
    checked unchanged:
    - find the view uniforms (view-projection, camera origin, per-eye projection offset);
    - add eye 2's copy;
    - write `gl_SecondaryPositionNV` and the viewport masks;
    - raise `#version 150` to a version that takes the extensions.
  - **NV assembly** (`!!NVvp5.0`): the same rewrite on the program text, through
    `glProgramStringARB` with `OPTION NV_viewport_array2; OPTION NV_stereo_view_rendering;`
    (to be confirmed against the extension's assembly grammar).

## Milestones (each a switch, off by default)

1. **S0 dump**: run with `probe_shaders=1`. Count the programs and kinds, and find the view uniforms
   by diffing the left and right scene calls' uniform/UBO uploads (RDR's census).
2. **S1 convert offline**: write the converter and compile every converted shader in a test context.
3. **S2 one draw, both eyes**: with the left scene call, convert-bound programs draw both halves
   (viewport array set to the two halves), and the right scene call skips its draws. Compare each
   eye against today's two-pass image, with the eye captures.
4. **S3 the one-view state**: occlusion queries, full-screen passes, clears per half, and screen-space
   passes per eye (KF2's list).
5. **S4 headset**: frame time against same-frame stereo.
