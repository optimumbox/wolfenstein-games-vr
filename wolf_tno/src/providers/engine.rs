//! id Tech 5 engine facts for the Wolfenstein games built on it (OpenGL), as they stand in each
//! running game's decrypted image (the files on disk are SteamStub-packed), valid only for the
//! fingerprinted builds in [`ROWS`]. The engine code is shared, so the per-game facts are
//! addresses and a few layout sizes; the rest is common to every build.

use eng_idt::Cvar;
use monaka_hook::module::Module;
use monaka_producer::builds::{Builds, Measured};
use monaka_hook::{Instruction, Site, mem};

/// A screen layout's first view record (rva) and the name its header carries.
pub type Layout = (usize, &'static str);

/// One game build: where its engine keeps what the producer uses.
pub struct Build {
    pub game: &'static str,
    pub exe: &'static str,
    pub sha256: &'static str,
    /// The engine keeps a pointer to the current screen layout's first view here. A layout is a
    /// header record followed by two view records of 0x28 bytes each: name, player (-1: unused),
    /// x, y, w, h (screen fractions), two flags, eye sign. The per-frame view builder renders one
    /// view per used record; a non-zero eye sign moves that view's origin sideways by
    /// `stereoRender_separation` and shifts its projection by `stereoRender_screenSeparation`.
    /// With one local player nothing rewrites the pointer after start-up (the only other writer
    /// picks split screen for two players).
    pub layout_global: usize,
    pub layout_single: Layout,
    pub layout_left_right: Layout,
    /// World units from the centre to each eye (default 1.5).
    pub stereo_separation: Cvar,
    /// Projection shift per eye in screen units (default -0.3).
    pub stereo_screen_separation: Cvar,
    /// The 2D layer's shift per eye (default -0.3, "so they don't appear at infinity in HMDs").
    pub stereo_gui_offset: Cvar,
    /// 1 pauses the game when its window loses focus (default 1).
    /// `g_stopTime` (int): 1 stops every entity's update but the player's own (the world stands
    /// still; the player moves and looks around it).
    pub stop_time: Cvar,
    pub pause_on_alt_tab: Cvar,
    /// The camera's horizontal field of view at rest (default 80). Aiming down sights narrows the
    /// render view's below it (The Old Blood: 80 to 65 over 7 frames).
    pub fov: Cvar,
    /// The cvar and value that turn god rays off, and the one for lens flares. With the
    /// side-by-side layout the engine draws both for the last view only (the right eye); with them
    /// off, both eyes match.
    pub god_rays_off: (Cvar, i32),
    pub skip_flares: Cvar,
    /// 1 skips the haze/flare post effect: ghosts of bright lights (the aperture's shape) laid over
    /// each eye in screen space, so they fall in a different place in each eye.
    pub haze_off: Cvar,
    /// 1 skips dim shadows (the cheap darkening shadows of lights, the window light shafts' among
    /// them). Drawn for each eye they can disagree: a large straight-edged shadow in one eye only.
    pub dim_shadows_off: Cvar,
    /// Screen-space reflections (0 off; the graphics settings set 2). With the side-by-side layout
    /// each eye's reflections are traced with another view's camera: glossy floors show ghosts of
    /// things where they are not.
    pub reflections: Cvar,
    /// 1 turns off the view bob and weapon sway (default 0).
    pub no_bob: Cvar,
    /// How far the hands and their weapons' depth is squeezed toward the eye so the world never
    /// clips them (default 0.1; 1 leaves it as drawn): their depth then says nothing true about
    /// how far they are.
    pub hands_depth_hack: Cvar,
    /// The hands' field of view scale, `(hands) -> float`: the hands and their
    /// weapon are drawn with the view's field of view times it (`hands_FovScale`, else the held
    /// weapon's own, eased to 1 while aiming down sights), so in colour they are not where the
    /// world's projection (and the scene's depth) puts them.
    pub hands_fov_scale: Site<HandsFovScaleFn>,
    /// The hands' depth squeeze, `(hands) -> float` (`hands_depthHack`), asked
    /// each frame as the hands' items are drawn. The hands' own model takes the value only now and
    /// then (shown, swimming: into its render entity, [`HANDS_PRESENTABLE`]), so a changed cvar
    /// reaches the weapons at once and the hands not.
    pub hands_depth_hack_get: Site<HandsDepthHackFn>,
    pub gui_draw: Site<GuiDrawFn>,
    pub view_builder: Site<ViewBuilderFn>,
    /// The frame's view slots the view builder fills ([`ScreenViews`]; none: not mapped).
    pub screen_views: Option<ScreenViews>,
    pub player_view_stride: usize,
    pub projection: Site<ProjectionFn>,
    pub make_frustum: Site<MakeFrustumFn>,
    pub usercmd_builder: Site<UsercmdBuilderFn>,
    /// The float ANGLE2SHORT scale the usercmd builder uses (65536 / 360).
    pub angle_to_short: usize,
    /// Where the hands place their guns and the weapons queue their shots; None where not mapped.
    pub guns: Option<Guns>,
    /// Where the player's melee hit is found; None where not mapped.
    pub melee: Option<Melee>,
    /// Pad 0's "connected" byte in the joystick state (then a "was connected" byte): the
    /// joystick thread reads XInput only for pads marked connected, and looks for new ones every
    /// 5 s by a 32-bit microsecond clock, a check that can stay blocked for over half an hour once
    /// the clock wraps. None where not mapped.
    pub pad_connected: Option<usize>,
    /// The player's view camera `(idPresentablePlayer) -> presentable`: the top of the player's
    /// `controlledPresentables` while it is still alive (a cutscene's camera, a takedown's), else
    /// null. The player's view update asks it every frame and builds the first-person view from
    /// the player's own angles only when it is null.
    pub view_camera: Site<ViewCameraFn>,
    /// The player's control camera ([`ControlCamera`]).
    pub control_camera: ControlCamera,
    /// The game's entities, for checking an entity handle ([`Entities`]).
    pub entities: Entities,
    /// The minigame the player is in ([`Minigame`]).
    pub minigame: Minigame,
    /// `idPresentableWeaponStatic`'s vtable (rva): the view camera a mounted gun gives while the
    /// player uses it. The gun turns by the change in the player's command angles each frame
    /// (`idProp_WeaponStatic::Think`: times its own factor, then clamped to its limits); its view
    /// is the gun's, not the player's.
    pub mounted_gun_vtable: usize,
    /// A machine the player drives ([`Driven`]); None where not mapped.
    pub driven: Option<Driven>,
    /// How the mech turns its aim ([`MechAim`]); None where there is no mech or it is not mapped.
    pub mech_aim: Option<MechAim>,
    /// The hotwiring minigame's hands ([`Hotwire`]); None where not mapped.
    pub hotwire: Option<Hotwire>,
    /// `idPresentablePlayer`'s weapon selection flags ([`WEAPON_WHEEL_OPEN`]).
    pub weapon_selection_flags: usize,
    /// The engine's check whether the player's control is held (rva, [`ControlInhibitedFn`]): an
    /// open in-game menu (the journal) or the game itself holds the player's control. The view
    /// update then takes no look input, and the camera keeps the angles it had.
    pub control_inhibited: usize,
    /// The game's rate, its setter `(hz)` ([`GameRateFn`]), and the engine's one
    /// `idTimeManager` it sets (ticks a second first: 60). The game runs one tick a frame, paced
    /// to real time at that rate, so the rate is also the frame rate's cap; each frame the game's
    /// clock takes the rate and the tick's whole milliseconds from it, so a new
    /// rate holds from the next frame. A frame rate below the rate slows the game down.
    pub set_game_rate: usize,
    pub time_manager: usize,
    /// How long the frame loop's last frame waited for its tick, in microseconds (`time_spinning`,
    /// an int): the frame loop waits out the rest of each tick (while `com_syncToTime` is on) and
    /// records the wait here, so a frame's own time is its period less this.
    pub time_spinning: usize,
    /// Where the scene's depth is, for frame warping; None where not mapped.
    pub scene_depth: Option<SceneDepth>,
    /// The engine's one video player keeps its open Bink video here (null with none open). Every
    /// video it plays fills the screen (the cutscenes shown while a map loads, scripted
    /// `idTarget_Bink` cutscenes, the credits) and is drawn over the whole window, not per eye.
    pub video: usize,
    /// Where the HUD's ammo for each gun is told apart from the rest of the HUD; None where not
    /// mapped.
    pub hud_pieces: Option<HudPieces>,
    /// Where the things around the player that it can use are found, and taken; None where not
    /// mapped.
    pub pickups: Option<Pickups>,
    /// The player's focus update ([`FocusUpdateFn`]; None where not mapped): each think it traces
    /// along the player's view object (in a cutscene, the cutscene's camera turned by the free
    /// look) for what the player looks at: the usable it offers to use, a choice a scene asks for by
    /// looking, the crosshair's target.
    pub focus_update: Option<Site<FocusUpdateFn>>,
    /// The frame loop's frame ([`FrameFn`]): once per frame, menus and loads included, it first
    /// waits out the rest of the tick on the engine's millisecond clock (while
    /// [`Build::sync_to_time`] is on), then runs the frame. The world advances one tick a frame
    /// whatever the wait, so frames started on another clock keep the game's speed.
    pub frame: Site<FrameFn>,
    /// `com_syncToTime` (1: the frame loop waits for its tick, 0: it runs the next frame at once).
    pub sync_to_time: Cvar,
    /// `r_fullscreen` (0: windowed). Windowed, the engine renders at its window's client size and
    /// follows its resizes (`r_windowWidth` and `r_windowHeight` track `WM_SIZE`); None where not
    /// mapped ([`crate::providers::window_size`]).
    pub fullscreen: Option<Cvar>,
    /// The damage handlers of what the laser cuts ([`Cutting`]; None where not mapped).
    pub cutting: Option<Cutting>,
}

/// `(idCommonLocal) -> ?`.
pub type FrameFn = unsafe extern "system" fn(usize) -> usize;
pub const FRAME_PROLOGUE: [Instruction; 3] = [
    Instruction::plain(&[0x40, 0x57]), // push rdi
    Instruction::plain(&[0x41, 0x54]), // push r12
    Instruction::plain(&[0x41, 0x55]), // push r13
];

/// `(focus tracker, idPlayer, &view origin, &view axis rows, ?)`.
pub type FocusUpdateFn = unsafe extern "system" fn(usize, usize, *const f32, *const f32, usize);

/// What the laser cuts takes its cut as damage ([`CutDamageFn`], each class's own handler): a chain
/// breaks the link nearest the hit, a cuttable panel cuts a hole there, a simple cuttable loses
/// health until it falls.
pub struct Cutting {
    /// `idFuncChain::Damage`: takes damage whose flags meet the chain's mask
    /// ([`CHAIN_MASK`], or [`DAMAGE_ANY_CHAIN`] set) while unbroken ([`CHAIN_BROKEN_AT`] < 0).
    pub chain: Site<CutDamageFn>,
    /// `idCuttable::Damage`: cuts with damage flagged [`DAMAGE_CUTS`] at least its strength
    /// ([`CUTTABLE_STRENGTH`] up to the damage's least, [`DAMAGE_LEAST`]).
    pub cuttable: Site<CutDamageFn>,
    /// `idEntitySimpleCuttable::Damage`: takes damage whose flags meet its mask
    /// ([`SIMPLE_CUTTABLE_MASK`]) while standing ([`SIMPLE_CUTTABLE_FALLEN_AT`] -1), off its
    /// health ([`SIMPLE_CUTTABLE_HEALTH`]).
    pub simple: Site<CutDamageFn>,
}

/// `(entity, inflictor, attacker, damage decl, scale, &direction, trace result or 0) -> ?`.
pub type CutDamageFn = unsafe extern "system" fn(usize, usize, usize, usize, f32, usize, usize) -> usize;

/// A damage decl's flags (u32).
pub const DAMAGE_FLAGS: usize = 0x98;
/// The damage's least and most amount (f32 each, a hit takes one between).
pub const DAMAGE_LEAST: usize = 0xcc;
pub const DAMAGE_MOST: usize = 0xd0;
/// A damage flag every chain takes.
pub const DAMAGE_ANY_CHAIN: u32 = 1 << 7;
/// A damage flag cuttable panels take.
pub const DAMAGE_CUTS: u32 = 1 << 14;
/// A trace result's hit point (3 floats).
pub const TRACE_POINT: usize = 0x38;
/// An entity's origin (3 floats).
pub const ENTITY_ORIGIN: usize = 0x40;
/// `idFuncChain`: the damage flags it takes (u32), and when it broke (i32 game ms, < 0 unbroken).
pub const CHAIN_MASK: usize = 0x4d0;
pub const CHAIN_BROKEN_AT: usize = 0x570;
/// `idCuttable`: the least damage that cuts it (f32), and its cut graph (i32, < 0 none: cut away).
pub const CUTTABLE_STRENGTH: usize = 0x3c0;
pub const CUTTABLE_GRAPH: usize = 0x3c8;
/// `idEntitySimpleCuttable`: the damage flags it takes (u32), when it fell (i32 game ms, -1
/// standing), its health (f32).
pub const SIMPLE_CUTTABLE_MASK: usize = 0x3d8;
pub const SIMPLE_CUTTABLE_FALLEN_AT: usize = 0x3e0;
pub const SIMPLE_CUTTABLE_HEALTH: usize = 0x3e8;

/// The things around the player that it can use. The use button uses the one in focus, which for
/// an item on the ground (an `idProp_Moveable`) takes what its inventory holds. The entities near a
/// place are found by a clip query: a gather of the bounds is asked for ([`GatherFn`], given a key),
/// made during the frame, and read on a later think by that key while the clip world still keeps it.
/// The player's own update of its usables gathers so only in multiplayer; in the campaign the
/// player's think ([`Pickups::think`]) is where a gather of its surroundings is read and asked for.
pub struct Pickups {
    /// The player's think `(idPlayer)` ([`PlayerThinkFn`]), each game tick on the game thread.
    pub think: Site<PlayerThinkFn>,
    /// The gather of the entities touching some bounds ([`GatherFn`], rva), with the contents the
    /// player's usables gather asks for, and the player's entity number (u16 at this offset in the
    /// entity), passed by the gather.
    pub gather: usize,
    pub contents: u32,
    pub entity_number: usize,
    /// The client game (rva of a pointer to it), its clip world, and in the clip world the oldest
    /// and newest keys whose gathers can still be read (u64 each).
    pub client_game: usize,
    pub clip: usize,
    pub keys: usize,
    /// The gathered entities by a gather's key ([`GatheredFn`], rva).
    pub gathered: usize,
    /// Whether an entity is an `idProp_Moveable` (an item on the ground) and an inventory item an
    /// `idAmmoItem` or an `idWeapon` ([`TypeTestFn`], rva).
    pub is_moveable: usize,
    pub is_ammo: usize,
    pub is_weapon: usize,
    /// An inventory item's virtual method: whether it fits in a player's inventory ([`FitsFn`]):
    /// ammo, its type's total under the most the player can carry; a gun, one the player has not, a
    /// second to hold in the other hand, or its ammo fitting. Always for a gun carried in both hands.
    pub fits: usize,
    /// In a weapon's decl: a gun carried in both hands (u8; a heavy gun torn off its mount, held
    /// with the heavy grip in place of the player's guns).
    pub carried_gun: usize,
    /// An entity's virtual methods (offsets into its vtable): whether the player can ever use it and
    /// can use it now (both [`UsableFn`]), its use ([`UseFn`]), its inventory ([`InventoryFn`]).
    pub ever_usable: usize,
    pub usable_now: usize,
    pub use_it: usize,
    pub inventory: usize,
    /// `idEntity`: its physics (a pointer); in the physics' vtable, its absolute bounds
    /// ([`AbsBoundsFn`]).
    pub physics: usize,
    pub abs_bounds: usize,
}

/// `(idPlayer)`.
pub type PlayerThinkFn = unsafe extern "system" fn(usize);
/// `(clip world, &key out, &bounds, contents, entity number to pass, label) -> &key`.
pub type GatherFn = unsafe extern "system" fn(usize, *mut u64, *const [f32; 6], u32, u16, *const u8) -> *const u64;
/// `(clip world, &key, out entity numbers, most) -> count`.
pub type GatheredFn = unsafe extern "system" fn(usize, *const u64, *mut i32, i32) -> u32;
/// `(object) -> bool`.
pub type TypeTestFn = unsafe extern "system" fn(usize) -> u8;
/// `(inventory item, idPlayer, 0) -> bool`.
pub type FitsFn = unsafe extern "system" fn(usize, usize, u8) -> u8;
/// `(entity, idPlayer) -> bool` (the low byte).
pub type UsableFn = unsafe extern "system" fn(usize, usize) -> u8;
/// `(entity, user, use state) -> bool`: the use button's use, the state 0 (a plain use).
pub type UseFn = unsafe extern "system" fn(usize, usize, i32) -> u8;
/// `(entity) -> inventory` (a list: items, pointers; then their count, i32).
pub type InventoryFn = unsafe extern "system" fn(usize) -> usize;
/// `(physics, body: -1 all) -> &bounds` (min x, y, z, then max).
pub type AbsBoundsFn = unsafe extern "system" fn(usize, i32) -> *const [f32; 6];
/// An inventory list's count (i32), after its items.
pub const LIST_COUNT: usize = 8;
/// An object's type (its first virtual method, `(object) -> idTypeInfo`), and the type's name (a
/// pointer to a C string).
pub type TypeFn = unsafe extern "system" fn(usize) -> usize;
pub const TYPE_NAME: usize = 0;

/// The scene's depth: the engine's depth-only render destination (a pointer to an
/// `idRenderDestination`), made with the scene's colour destinations when the default framebuffer
/// is, on the image `_frontDepthStencil`, which every scene
/// destination draws its depth into. At `SwapBuffers` it holds the frame's depth, every view's at
/// its place in the window (scaled with the render size).
pub struct SceneDepth {
    pub destination: usize,
}

/// An `idRenderDestination`'s depth image (`idImage *`).
pub const DESTINATION_DEPTH: usize = 0x30;
/// An `idImage`'s GL texture (u32), width and height (i32), and samples (i32; above 1 it is a
/// `GL_TEXTURE_2D_MULTISAMPLE`).
pub const IMAGE_TEXTURE: usize = 0xc8;
pub const IMAGE_WIDTH: usize = 0x58;
pub const IMAGE_HEIGHT: usize = 0x5c;
pub const IMAGE_SAMPLES: usize = 0x68;

/// The HUD is a Flash movie (`idSWF`) drawn into a GUI model (`idRenderModelGui`) on the game
/// thread: each sprite instance's shapes become triangles in the model's current surface (a run
/// of one material and one GL state), which the render thread draws surface by surface. Each
/// gun's ammo is a sprite of its own, which the HUD's ammo group for that gun finds by name when
/// the HUD starts.
pub struct HudPieces {
    /// `idSWF::RenderSprite` ([`RenderSpriteFn`]).
    pub render_sprite: Site<RenderSpriteFn>,
    /// `idPlayerHud_Render::Render` ([`HudRenderFn`]).
    pub hud_render: Site<HudRenderFn>,
    /// `idPlayerHud_Render`: its ammo groups (`idAmmoGroupLeft`, `idAmmoGroupRight`), left first
    /// ([`AMMO_GROUP_SPRITE`], [`AMMO_GROUP_READY`]).
    pub ammo_groups: [usize; 2],
    /// `idPlayerHud_Render`: its item checklists (`idCombinerGroup`: the items a task gathers,
    /// ticked as found, at the top left).
    pub checklists: Checklists,
    /// The GL draw the render thread draws each GUI surface with ([`DrawElementsFn`]).
    pub draw_elements: Site<DrawElementsFn>,
    /// `idRenderModelGui::AdvanceSurf(gui)`: starts the model's next surface with the current
    /// one's material and state.
    pub advance_surface: usize,
    /// The GUI model's virtual width and height getters, `(gui) -> float`: the size of the
    /// layout the model's positions are in.
    pub virtual_width: usize,
    pub virtual_height: usize,
    /// The GUI models' shared vertex buffer as mapped this frame (a pointer; [`GUI_VERTEX_SIZE`]
    /// bytes a vertex, its x and y first) and how many vertices are in it.
    pub gui_vertices: usize,
    pub gui_vertex_count: usize,
    /// `idRenderModelGui`: its current surface (a pointer); the surface count follows 0xc later
    /// and the surfaces themselves ([`GUI_SURFACE_SIZE`] bytes each) 0x10 later.
    pub current_surface: usize,
}

/// The HUD's item checklists (`idCombinerGroup`, `hudmain/combiner1`..`combiner3`, each up to five
/// item icons): where in `idPlayerHud_Render` the group is, and where in it each checklist's sprite
/// is kept (a sprite instance pointer, 0 until found).
pub struct Checklists {
    pub group: usize,
    pub sprites: [usize; 3],
}

/// The functions that place what the hands hold and queue its shots, with their prologues where
/// they differ between builds, and where the objects they work on keep what the producer uses.
pub struct Guns {
    /// The hands' item placer `(item, hands)` ([`PlaceItemFn`]).
    pub place_item: Site<PlaceItemFn>,
    pub tag_transform: Site<TagTransformFn>,
    /// The deferred fire queue ([`FireQueueFn`]).
    pub fire_queue: Site<FireQueueFn>,
    /// The hands' blend kick ([`ArmsKickFn`]).
    pub arms_kick: Site<ArmsKickFn>,
    /// A blend job's pose bounds ([`PoseBoundsFn`]).
    pub pose_bounds: Site<PoseBoundsFn>,
    pub layout: HandsLayout,
    /// Where the arms model is placed from the player's view, for it to follow the head
    /// ([`FollowHead`]; none: not mapped for this build).
    pub follow_head: Option<FollowHead>,
    /// The laser's check whether a cut point is aimed at ([`LaserTargetFn`]; none: not
    /// mapped): it locks onto a point only within a small angle of the origin and direction it is
    /// given, which the laser's draw takes from the player's view.
    pub laser_target: Option<Site<LaserTargetFn>>,
}

/// `(idLaserCutter, ?, &cut point, &origin, &direction, &out cosine) -> bool`.
pub type LaserTargetFn = unsafe extern "system" fn(usize, usize, *const f32, *const f32, *const f32, *mut f32) -> u8;

/// The player's melee hit: a trace the hands own (`idHands::handsMeleeTrace`, an `idMeleeTrace`),
/// opened and closed by the hands' animation events. While it is open, each update sweeps a box
/// from a point on the weapon's previous position to its current one ([`MeleeCoreFn`]); the core
/// that does it, and how it deals the damage, depend on `gameFrameParms_t::isServer`.
pub struct Melee {
    /// The trace in `idHands`, and its fields.
    pub trace: usize,
    pub fields: MeleeTraceFields,
    /// The hands' per-frame melee update ([`MeleeUpdateFn`]): while the trace is open, it sweeps
    /// it from a point on the right hand's item, then from the eye when that hit nothing.
    pub update: Site<MeleeUpdateFn>,
    /// The trace's start at a point ([`StartPointFn`]): it opens with the settings given, the
    /// sweep starting at the point.
    pub start_point: usize,
    /// `idHands::isInMeleeSwing` (bool): set only by the quick melee (the melee button), which
    /// draws the knife or the pipe for one attack and goes back to the gun.
    pub in_swing: usize,
    /// `idPresentable::entity`: the player's game entity (the server core's attacker).
    pub player_entity: usize,
    /// `idDeclWeapon::pistolWhipProjectile`: the projectile decl a weapon's melee hit deals
    /// damage as; in it, its damage decl ([`PROJECTILE_IMPACT_EFFECT`] beside).
    pub pistol_whip: usize,
    pub projectile_damage: usize,
    /// The hands' animation events ([`AnimEventFn`]) that start the trace at a joint of the hands
    /// ([`START_JOINT_PROLOGUE`]), at a tag on the held item's model and along the eye (both
    /// [`START_TAG_PROLOGUE`]), and that end it ([`START_JOINT_PROLOGUE`] too).
    pub start_joint: usize,
    pub start_tag: usize,
    pub start_simple: usize,
    pub end: usize,
    /// The core when the game is the server (damages the victim at once; the attacker is the
    /// player entity) and when it is a client (sends a melee-hit message; the attacker is the
    /// presentable player).
    pub core_server: usize,
    pub core_server_prologue: &'static [Instruction],
    /// [`CORE_CLIENT_PROLOGUE`].
    pub core_client: usize,
    /// `[exe + game_local] + is_server`: the byte that picks the core.
    pub game_local: usize,
    pub is_server: usize,
}

/// `idMeleeTrace` fields that moved between builds.
pub struct MeleeTraceFields {
    /// Updates left (int; above 0: open).
    pub num_samples: usize,
    /// One more update collects the last queries (bool), then the trace is shut; followed by
    /// `skipBounds` (bool: no box around the point).
    pub one_last_update: usize,
    /// The melee type (u8: 0 impact, 1 semi-continuous, 2 first plus actor, 3 continuous), followed
    /// by the bounds type (u8: 0 a line, 1 to 6 boxes of 8 to 96 units, 7 the player melee box).
    pub melee_type: usize,
    /// Where this update's sweep starts (3 floats).
    pub prev_pos: usize,
    /// The last entity damaged.
    pub hit_entity: usize,
}

/// `idMeleeTrace`: the second blade joint (i16, below 0: none), whether it reads a tag, whether it
/// has hit.
pub const MELEE_JOINT2: usize = 0x1e;
pub const MELEE_USE_TAG: usize = 0x59;
pub const MELEE_HAS_HIT: usize = 0x5c;
/// `idMeleeTrace::upgradDamageScale` (int): the hands set 1.
pub const MELEE_DAMAGE_SCALE: usize = 0x18;
/// `idDeclProjectile::impactEffectTable`: what a hit spawns where it lands.
pub const PROJECTILE_IMPACT_EFFECT: usize = 0xa8;

pub const START_JOINT_PROLOGUE: [Instruction; 1] = [Instruction::plain(&[0x48, 0x89, 0x5c, 0x24, 0x10])]; // mov [rsp+0x10], rbx
pub const START_TAG_PROLOGUE: [Instruction; 1] = [Instruction::plain(&[0x48, 0x89, 0x5c, 0x24, 0x08])]; // mov [rsp+0x8], rbx
pub const CORE_CLIENT_PROLOGUE: [Instruction; 4] = [
    Instruction::plain(&[0x40, 0x55]), // push rbp
    Instruction::plain(&[0x53]),       // push rbx
    Instruction::plain(&[0x56]),       // push rsi
    Instruction::plain(&[0x57]),       // push rdi
];
/// An `idHands::AnimEvent_*` method `(hands, ...)`: six pointer-sized arguments, the last two
/// stack slots passed through as found.
pub type AnimEventFn = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize) -> usize;
/// The melee trace core `(trace, attacker, const idVec3* point, const idVec3* point2, bool
/// secondary) -> hit`: collects the previous update's queries (damaging what they hit) and
/// sweeps from the trace's previous position to `point`, and from `point2` to `point` when
/// [`MELEE_JOINT2`] is 0 or more.
pub type MeleeCoreFn = unsafe extern "system" fn(usize, usize, *const f32, *const f32, usize) -> usize;
/// The hands' per-frame melee update `(hands)`, on the game thread early in the hands' think.
pub type MeleeUpdateFn = unsafe extern "system" fn(usize);
/// The melee trace's start at a point `(trace, const idVec3* point, numSamples, damageCap, meleeType,
/// boundsType, damage decl, impact effect)`: the trace opens for `numSamples` updates (each core
/// call takes one), its sweep starting at `point`, its hits cleared.
pub type StartPointFn = unsafe extern "system" fn(usize, *const f32, i32, f32, u8, u8, usize, usize);

/// Object fields that moved between builds (the rest are shared constants below).
pub struct HandsLayout {
    /// `idPresentablePlayer`: the first-person view, origin (3 floats) followed by axis rows (9).
    pub player_view: usize,
    /// `idHands`: the right hand's `idHandsItem`.
    pub right_item: usize,
    /// `idDeclWeapon::oneHandedWeapon` (bool): the knife, the grenades and The Old Blood's pipes;
    /// the guns are held with both hands.
    pub one_handed: usize,
    /// `idDeclWeapon::weaponSlot` (i32), and the slots of the one-handed weapons that are melee
    /// weapons (a slot number can be shared with a gun: The Old Blood's pipes and its LKW).
    pub weapon_slot: usize,
    pub melee_slots: &'static [i32],
    /// The right wrist in the melee weapon's attach joint frame at the game's resting hold (the
    /// grip the hand rig's end-of-run summary says it holds longest, measured standing still): the
    /// weapon then goes along the controller's grip ([`crate::providers::held_item::melee_at`])
    /// and the hand holds it at this grip. None (not measured): the weapon goes where a gun would,
    /// the hand on it at `skeleton::RIGHT_GRIP`.
    pub melee_rest: Option<[f32; 12]>,
    /// `idPresentableAnimatedEntity`: its animated entity (an `idTreeAnimator`).
    pub model_animated: usize,
    /// `idTreeAnimator`: its joint animator (`idJointAnimator`).
    pub joint_animator: usize,
    /// `idJointAnimator`: the model-space joint matrices (3x4 row-major, rotation and translation
    /// for column vectors), and the pose the next blend job builds (swapped with them when it is
    /// finished).
    pub joints: usize,
    pub back_joints: usize,
    /// The skeleton header (`idMD6SkelData`: joint count +0x2, offset of the parent table (i16,
    /// parents first) +0xe) is `[[[[joint animator] + 0x60] + 0x128] + skeleton_header]`.
    pub skeleton_header: usize,
    /// How a joint's rotation gives the game's axis rows (forward, left, up) in model space.
    pub joint_basis: Basis,
}

/// How a joint's rotation gives the game's axis rows (forward, left, up) in model space: its
/// columns, or its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    Columns,
    Rows,
}

// SAFETY: every `Site` below starts a function of its type in its build's executable, with the
// prologue as measured there; a build's facts are used only once its fingerprint (`sha256`) is
// checked, and each prologue is checked again before a hook is written.
pub static ROWS: [Build; 2] = unsafe { [
    Build {
        game: "Wolfenstein: The New Order",
        exe: "WolfNewOrder_x64.exe",
        sha256: "9E413197E097BA41E85F0DE47DDD3918F7656C432F8431F5F15E4B7EEB7C371C",
        // Object 0x1cc5260, field +0x8520.
        layout_global: 0x1ccd780,
        layout_single: (0xe6da48, "single"),
        layout_left_right: (0xe6dbd8, "leftRightStereo"),
        stereo_separation: Cvar::id5(0x1ccd7c0, "stereoRender_separation"),
        stereo_screen_separation: Cvar::id5(0x1ccd840, "stereoRender_screenSeparation"),
        stereo_gui_offset: Cvar::id5(0x1ccd8c0, "stereoRender_guiOffset"),
        pause_on_alt_tab: Cvar::id5(0x1d70e50, "win_pauseOnAltTab"),
        stop_time: Cvar::id5(0x1da3000, "g_stopTime"),
        fov: Cvar::id5(0x1da5a00, "g_fov"),
        god_rays_off: (Cvar::id5(0x1d16820, "r_skipGodRays"), 1),
        skip_flares: Cvar::id5(0x1ce8650, "r_skipFlares"),
        haze_off: Cvar::id5(0x1d169a0, "r_skipHazeFlare"),
        dim_shadows_off: Cvar::id5(0x1d163a0, "r_skipDimShadows"),
        reflections: Cvar::id5(0x1d16920, "r_screenSpaceReflections"),
        no_bob: Cvar::id5(0x1da5480, "pm_noBob"),
        hands_depth_hack: Cvar::id5(0x1dadb00, "hands_depthHack"),
        hands_fov_scale: Site::new("hands' field of view", 0x8fbda0, &HANDS_FOV_SCALE_PROLOGUE),
        hands_depth_hack_get: Site::new("hands' depth squeeze", 0x8fb0d0, &[
            Instruction::plain(&[0x48, 0x8b, 0x41, 0x18]),                   // mov rax, [rcx+0x18]
            Instruction::plain(&[0x83, 0xb8, 0x24, 0xd7, 0x00, 0x00, 0x01]), // cmp dword [rax+0xd724], 1
        ]),
        // Called from exe+0x4726d8 in play, exe+0x472c67 in a paused menu.
        gui_draw: Site::new("2D layer", 0x453c50, &GUI_DRAW_PROLOGUE),
        view_builder: Site::new("view builder", 0x162750, &VIEW_BUILDER_PROLOGUE),
        screen_views: Some(ScreenViews { list: 0, stride: 0x1a90, env_lists: 0x1080 }),
        player_view_stride: 0x19f0,
        projection: Site::new("projection", 0xa23570, &PROJECTION_PROLOGUE),
        make_frustum: eng_idt::make_frustum(0xa20ec0),
        usercmd_builder: Site::new("usercmd builder", 0x179910, &[
            Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
            Instruction::plain(&[0x53]),             // push rbx
            Instruction::plain(&[0x41, 0x54]),       // push r12
        ]),
        angle_to_short: 0xe740f0,
        guns: Some(Guns {
            place_item: Site::new("hand item placer", 0x916780, &[
                Instruction::plain(&[0x4c, 0x8b, 0xdc]), // mov r11, rsp
                Instruction::plain(&[0x55]),             // push rbp
                Instruction::plain(&[0x41, 0x54]),       // push r12
            ]),
            tag_transform: Site::new("tag transform", 0x3806a0, &TAG_TRANSFORM_PROLOGUE),
            fire_queue: Site::new("fire queue", 0x991c10, &[
                Instruction::plain(&[0x40, 0x53]), // push rbx
                Instruction::plain(&[0x56]),       // push rsi
                Instruction::plain(&[0x57]),       // push rdi
                Instruction::plain(&[0x41, 0x55]), // push r13
            ]),
            arms_kick: Site::new("hands blend kick", 0x901b40, &[
                Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
                Instruction::plain(&[0x41, 0x55]),       // push r13
            ]),
            pose_bounds: Site::new("pose bounds", 0x3323f0, &[
                Instruction::plain(&[0x48, 0x83, 0xec, 0x38]),                             // sub rsp, 0x38
                Instruction::rip_relative(&[0x0f, 0x28, 0x1d, 0xd5, 0x5d, 0xb7, 0x00], 3), // movaps xmm3, [rip+...]
            ]),
            layout: HandsLayout {
                player_view: 0x10780,
                right_item: 0x1458,
                one_handed: 0x730,
                // SLOT_KNIFE.
                weapon_slot: 0x358,
                melee_slots: &[0],
                // The knife, the only melee weapon: all of a 16 s idle at this one grip.
                melee_rest: Some([-0.389, -0.387, 0.836, -4.416, 0.13, -0.921, -0.366, 0.571, 0.912, -0.034, 0.409, -1.414]),
                model_animated: 0x540,
                joint_animator: 0x888,
                joints: 0x110,
                back_joints: 0x120,
                skeleton_header: 0x38,
                joint_basis: Basis::Columns,
            },
            follow_head: Some(FollowHead { update_position: 0x906860, animator: 0x8fabb0, joint_index: 0x380dd0, item_draw: 0x916c40, view_position: 0x47e0 }),
            laser_target: Some(Site::new("laser target", 0x9a30d0, &[
                Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
                Instruction::plain(&[0x55]),             // push rbp
                Instruction::plain(&[0x53]),             // push rbx
            ])),
        }),
        melee: Some(Melee {
            trace: 0x2678,
            fields: MeleeTraceFields { num_samples: 0x64, one_last_update: 0x60, melee_type: 0x70, prev_pos: 0x74, hit_entity: 0x88 },
            update: Site::new("melee update", 0x8fb340, &[
                Instruction::plain(&[0x40, 0x56]),             // push rsi
                Instruction::plain(&[0x57]),                   // push rdi
                Instruction::plain(&[0x48, 0x83, 0xec, 0x58]), // sub rsp, 0x58
            ]),
            start_point: 0x9aecc0,
            in_swing: 0x262f,
            player_entity: 0x4c8,
            pistol_whip: 0x698,
            projectile_damage: 0x1c0,
            start_joint: 0x8ff8a0,
            start_tag: 0x8ffad0,
            start_simple: 0x8ffe90,
            end: 0x8ffdb0,
            core_server: 0x9af270,
            core_server_prologue: &[
                Instruction::plain(&[0x40, 0x55]), // push rbp
                Instruction::plain(&[0x57]),       // push rdi
                Instruction::plain(&[0x41, 0x55]), // push r13
            ],
            core_client: 0x9b0270,
            game_local: 0x1da60e0,
            is_server: 0x180ffc,
        }),
        pad_connected: Some(0x1d5af4c),
        view_camera: Site::new("view camera", 0x602d70, &[
            Instruction::plain(&[0x48, 0x83, 0xec, 0x28]),             // sub rsp, 0x28
            Instruction::plain(&[0x8b, 0x81, 0xd0, 0xbc, 0x00, 0x00]), // mov eax, [rcx+0xbcd0]
        ]),
        control_camera: ControlCamera { player: 0x4c8, get: 0x611400 },
        entities: Entities { game_local: 0x1da60e0, spawn_ids: 0x261d0, entities: 0x161d0, bits: 13 },
        minigame: Minigame { hud: 0xbd10, handle: 0x100 },
        mounted_gun_vtable: 0xf53728,
        driven: Some(Driven { entity: 0x4c8, cast: 0x31670 }),
        mech_aim: Some(MechAim {
            view_angles: Site::new("mech view angles", 0x74bfe0, &[
                Instruction::plain(&[0x48, 0x8b, 0xc4]),       // mov rax, rsp
                Instruction::plain(&[0x48, 0x89, 0x58, 0x08]), // mov [rax+8], rbx
                Instruction::plain(&[0x57]),                   // push rdi
            ]),
            pitch: 0xd68,
            min_pitch: 0xcf0,
            max_pitch: 0xcfc,
            pitch_scale: 0x17a8,
        }),
        hotwire: Some(Hotwire {
            update: Site::new("hotwiring", 0x6e84f0, &[
                Instruction::plain(&[0x48, 0x85, 0xd2]),                             // test rdx, rdx
                Instruction::rip_relative(&[0x0f, 0x84, 0xb3, 0x10, 0x00, 0x00], 2), // je +0x10b3
            ]),
            command: 0x10684,
        }),
        weapon_selection_flags: 0xc37c,
        control_inhibited: 0x60bc20,
        set_game_rate: 0x15cce0,
        time_manager: 0x1595dd8,
        time_spinning: 0x161e128,
        scene_depth: Some(SceneDepth { destination: 0x1d19e00 }),
        video: 0x1d1e580,
        hud_pieces: Some(HudPieces {
            render_sprite: Site::new("SWF sprite", 0x2725d0, &RENDER_SPRITE_PROLOGUE),
            hud_render: Site::new("HUD", 0x5f13b0, &HUD_RENDER_PROLOGUE),
            ammo_groups: [0x1d0, 0x108],
            // Three checklists of 0xd0 bytes from +8, each one's sprite first.
            checklists: Checklists { group: 0x938, sprites: [0x8, 0xd8, 0x1a8] },
            draw_elements: Site::new("GUI surface draw", 0x3f9570, &START_TAG_PROLOGUE),
            advance_surface: 0x313810,
            virtual_width: 0x313790,
            virtual_height: 0x3137d0,
            gui_vertices: 0x1cea608,
            gui_vertex_count: 0x1cea5fc,
            current_surface: 0x880,
        }),
        pickups: Some(Pickups {
            think: Site::new("player think", 0x9357c0, &[
                Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
                Instruction::plain(&[0x55]),             // push rbp
                Instruction::plain(&[0x57]),             // push rdi
                Instruction::plain(&[0x41, 0x54]),       // push r12
            ]),
            gather: 0x20d430,
            contents: 0x21b8c1,
            entity_number: 0x218,
            client_game: 0x1d7a360,
            clip: 0x4248,
            keys: 0x792b0,
            gathered: 0x210f30,
            is_moveable: 0x3df20,
            is_ammo: 0x1b010,
            is_weapon: 0x16ce0,
            fits: 0xb0,
            carried_gun: 0x1a2,
            ever_usable: 0x370,
            usable_now: 0x378,
            use_it: 0x380,
            inventory: 0x398,
            physics: 0x310,
            abs_bounds: 0x58,
        }),
        focus_update: Some(Site::new("player focus", 0x8f9ad0, &[
            Instruction::plain(&[0x48, 0x85, 0xd2]),                             // test rdx, rdx
            Instruction::rip_relative(&[0x0f, 0x84, 0x1b, 0x06, 0x00, 0x00], 2), // je +0x61b
        ])),
        frame: Site::new("frame", 0x13dff0, &FRAME_PROLOGUE),
        sync_to_time: Cvar::id5(0x16ae1e0, "com_syncToTime"),
        fullscreen: Some(Cvar::id5(0x1d1f240, "r_fullscreen")),
        cutting: Some(Cutting {
            chain: Site::new("chain damage", 0x6b5bd0, &[
                Instruction::plain(&[0x40, 0x55]), // push rbp
                Instruction::plain(&[0x53]),       // push rbx
                Instruction::plain(&[0x56]),       // push rsi
                Instruction::plain(&[0x41, 0x54]), // push r12
            ]),
            cuttable: Site::new("cuttable damage", 0x6bf940, &[
                Instruction::plain(&[0x4c, 0x8b, 0xdc]),       // mov r11, rsp
                Instruction::plain(&[0x49, 0x89, 0x5b, 0x10]), // mov [r11+0x10], rbx
            ]),
            simple: Site::new("simple cuttable damage", 0x7242b0, &[
                Instruction::plain(&[0x40, 0x53]),                               // push rbx
                Instruction::plain(&[0x55]),                                     // push rbp
                Instruction::plain(&[0x57]),                                     // push rdi
                Instruction::plain(&[0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00]), // sub rsp, 0x80
            ]),
        }),
    },
    Build {
        game: "Wolfenstein: The Old Blood",
        exe: "WolfOldBlood_x64.exe",
        sha256: "DED2EE4F65A5560EF0D716E622FA5146121D45E3736ED013D74964D02FBA1CBD",
        layout_global: 0x1ed2be0,
        layout_single: (0xe53518, "single"),
        layout_left_right: (0xe536a8, "leftRightStereo"),
        stereo_separation: Cvar::id5(0x1ed2c20, "stereoRender_separation"),
        stereo_screen_separation: Cvar::id5(0x1ed2ca0, "stereoRender_screenSeparation"),
        stereo_gui_offset: Cvar::id5(0x1ed2d20, "stereoRender_guiOffset"),
        pause_on_alt_tab: Cvar::id5(0x1f82d50, "win_pauseOnAltTab"),
        stop_time: Cvar::id5(0x1fa8be0, "g_stopTime"),
        fov: Cvar::id5(0x1fab6e0, "g_fov"),
        // No r_skipGodRays in this build; quality 0 is what the engine stores when god rays are
        // unavailable (exe+0x46d8da).
        god_rays_off: (Cvar::id5(0x1f1f630, "r_godrayQuality"), 0),
        skip_flares: Cvar::id5(0x1eed100, "r_skipFlares"),
        haze_off: Cvar::id5(0x1f1c1b0, "r_skipHazeFlare"),
        dim_shadows_off: Cvar::id5(0x1f1bab0, "r_skipDimShadows"),
        reflections: Cvar::id5(0x1f1c130, "r_screenSpaceReflections"),
        no_bob: Cvar::id5(0x1fab060, "pm_noBob"),
        hands_depth_hack: Cvar::id5(0x1fb3b50, "hands_depthHack"),
        hands_fov_scale: Site::new("hands' field of view", 0x8d4950, &HANDS_FOV_SCALE_PROLOGUE),
        hands_depth_hack_get: Site::new("hands' depth squeeze", 0x8d4930, &[
            Instruction::plain(&[0x48, 0x8b, 0x41, 0x18]),                   // mov rax, [rcx+0x18]
            Instruction::plain(&[0x83, 0xb8, 0x08, 0x9a, 0x00, 0x00, 0x01]), // cmp dword [rax+0x9a08], 1
        ]),
        gui_draw: Site::new("2D layer", 0x46e430, &GUI_DRAW_PROLOGUE),
        view_builder: Site::new("view builder", 0x15ccb0, &VIEW_BUILDER_PROLOGUE),
        // Not mapped yet.
        screen_views: None,
        player_view_stride: 0x1a00,
        projection: Site::new("projection", 0x9f7dc0, &PROJECTION_PROLOGUE),
        make_frustum: eng_idt::make_frustum(0x9f7cc0),
        usercmd_builder: Site::new("usercmd builder", 0x174c60, &[
            Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
            Instruction::plain(&[0x53]),             // push rbx
            Instruction::plain(&[0x55]),             // push rbp
            Instruction::plain(&[0x41, 0x55]),       // push r13
        ]),
        angle_to_short: 0xe5a484,
        guns: Some(Guns {
            place_item: Site::new("hand item placer", 0x8eb880, &[
                Instruction::plain(&[0x4c, 0x8b, 0xdc]), // mov r11, rsp
                Instruction::plain(&[0x55]),             // push rbp
                Instruction::plain(&[0x41, 0x56]),       // push r14
            ]),
            tag_transform: Site::new("tag transform", 0x395c60, &TAG_TRANSFORM_PROLOGUE),
            fire_queue: Site::new("fire queue", 0x963ad0, &[
                Instruction::plain(&[0x40, 0x53]),                               // push rbx
                Instruction::plain(&[0x56]),                                     // push rsi
                Instruction::plain(&[0x57]),                                     // push rdi
                Instruction::plain(&[0x48, 0x81, 0xec, 0x90, 0x00, 0x00, 0x00]), // sub rsp, 0x90
            ]),
            arms_kick: Site::new("hands blend kick", 0x8e5530, &[
                Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
                Instruction::plain(&[0x41, 0x57]),       // push r15
            ]),
            pose_bounds: Site::new("pose bounds", 0x33a840, &[
                Instruction::plain(&[0x48, 0x83, 0xec, 0x38]),                             // sub rsp, 0x38
                Instruction::rip_relative(&[0x0f, 0x28, 0x1d, 0x35, 0x27, 0xb5, 0x00], 3), // movaps xmm3, [rip+...]
            ]),
            // Each of the joint animator's ten joint buffers is 8 bytes longer than The New
            // Order's, moving everything after them by 0x50.
            layout: HandsLayout {
                player_view: 0xcad4,
                right_item: 0x14a8,
                one_handed: 0x788,
                // SLOT_KNIFE, SLOT_PIPES.
                weapon_slot: 0x36c,
                melee_slots: &[0, 11],
                // Not measured yet (the knife and the pipes may differ).
                melee_rest: None,
                model_animated: 0x590,
                joint_animator: 0x898,
                joints: 0x160,
                back_joints: 0x170,
                skeleton_header: 0x130,
                joint_basis: Basis::Columns,
            },
            // Not mapped yet.
            follow_head: None,
            laser_target: None,
        }),
        melee: Some(Melee {
            trace: 0x26c8,
            fields: MeleeTraceFields { num_samples: 0x68, one_last_update: 0x61, melee_type: 0x7c, prev_pos: 0x80, hit_entity: 0x98 },
            update: Site::new("melee update", 0x8e6190, &[
                Instruction::plain(&[0x48, 0x89, 0x74, 0x24, 0x18]), // mov [rsp+0x18], rsi
            ]),
            start_point: 0x9839f0,
            in_swing: 0x2681,
            player_entity: 0x518,
            pistol_whip: 0x6f0,
            projectile_damage: 0x1c8,
            start_joint: 0x8d0010,
            start_tag: 0x8d04e0,
            start_simple: 0x8d0250,
            end: 0x8cfe80,
            core_server: 0x983dc0,
            core_server_prologue: &[
                Instruction::plain(&[0x40, 0x55]), // push rbp
                Instruction::plain(&[0x57]),       // push rdi
                Instruction::plain(&[0x41, 0x54]), // push r12
            ],
            core_client: 0x985010,
            game_local: 0x1fabdc0,
            is_server: 0x1b5084,
        }),
        pad_connected: Some(0x1f6cdec),
        view_camera: Site::new("view camera", 0x625fd0, &[
            Instruction::plain(&[0x48, 0x83, 0xec, 0x28]),             // sub rsp, 0x28
            Instruction::plain(&[0x8b, 0x81, 0x98, 0x7e, 0x00, 0x00]), // mov eax, [rcx+0x7e98]
        ]),
        control_camera: ControlCamera { player: 0x518, get: 0x630030 },
        entities: Entities { game_local: 0x1fabdc0, spawn_ids: 0x361d0, entities: 0x161d0, bits: 14 },
        minigame: Minigame { hud: 0x7ed8, handle: 0x108 },
        mounted_gun_vtable: 0xf32ce8,
        driven: None,
        mech_aim: None,
        hotwire: None,
        weapon_selection_flags: 0x8554,
        control_inhibited: 0x631340,
        set_game_rate: 0x158470,
        time_manager: 0x177d4a8,
        time_spinning: 0x1823170,
        scene_depth: Some(SceneDepth { destination: 0x1f202d0 }),
        video: 0x1f24cf8,
        hud_pieces: Some(HudPieces {
            render_sprite: Site::new("SWF sprite", 0x27b3a0, &RENDER_SPRITE_PROLOGUE),
            hud_render: Site::new("HUD", 0x6105f0, &HUD_RENDER_PROLOGUE),
            ammo_groups: [0x220, 0x158],
            checklists: Checklists { group: 0x9c0, sprites: [0x8, 0xd8, 0x1a8] },
            draw_elements: Site::new("GUI surface draw", 0x40d5f0, &START_TAG_PROLOGUE),
            advance_surface: 0x321a10,
            virtual_width: 0x322930,
            virtual_height: 0x3228f0,
            gui_vertices: 0x1eef160,
            gui_vertex_count: 0x1eef170,
            current_surface: 0x890,
        }),
        // Not mapped yet.
        pickups: None,
        focus_update: None,
        frame: Site::new("frame", 0x1371a0, &FRAME_PROLOGUE),
        sync_to_time: Cvar::id5(0x187fd70, "com_syncToTime"),
        fullscreen: None,
        // Not mapped yet.
        cutting: None,
    },
] };

/// The build for a game executable name.
/// The builds, the running one picked by its executable (`monaka_producer::builds`).
pub static BUILDS: Builds<Build> = Builds::new(&ROWS);

impl Measured for Build {
    fn module(&self) -> &str {
        self.exe
    }

    fn sha256(&self) -> &str {
        self.sha256
    }
}

/// A layout's header record, before its first view record, and each record's size: name, player
/// (-1: unused), x, y, w, h (screen fractions), two flag bytes, eye sign (f32 at
/// [`VIEW_EYE_SIGN`]: -1 left, 1 right, 0 no eye offset).
pub const LAYOUT_HEADER: usize = 0x28;
pub const LAYOUT_RECORD: usize = 0x28;
pub const VIEW_EYE_SIGN: usize = 0x20;

/// The 2D layer drawer `(backend, guis, count, ?, record, shift)`: draws one eye's GUIs (HUD, menus)
/// with the viewport already set: in play into the eye's render target after the 3D, in a paused
/// menu into the eye's half of the window.
pub const GUI_DRAW_PROLOGUE: [Instruction; 2] = [
    Instruction::plain(&[0x45, 0x85, 0xc0]),                                 // test r8d, r8d
    Instruction::rip_relative(&[0x0f, 0x8e, 0xf6, 0x00, 0x00, 0x00], 2),     // jle +0xf6
];
pub type ViewCameraFn = unsafe extern "system" fn(usize) -> usize;

/// A camera that rewrites the player's own view each frame without taking control of it (a grab's
/// spring camera: `idSpringCamera` sets it on the player as it activates and clears it as it
/// deactivates). The player's view is built first, then this camera rewrites it; the player's
/// angles do not move it.
#[derive(Clone, Copy, Debug)]
pub struct ControlCamera {
    /// `idPresentablePlayer`: its `idPlayer`.
    pub player: usize,
    /// The player's control camera getter `(player) -> idCamera` (rva): the camera, or null (its handle
    /// checked against the entity's spawn id, and the entity checked to be a camera).
    pub get: usize,
}

pub type ControlCameraFn = unsafe extern "system" fn(usize) -> usize;
/// `Build::control_inhibited`: `(idPresentablePlayer) -> byte` (1: control held).
pub type ControlInhibitedFn = unsafe extern "system" fn(usize) -> u8;

/// The game's entities, by which an entity handle (u32: the entity's number in the low [`bits`],
/// its spawn id above them) is checked as the game checks it: the handle names a live entity when
/// the spawn id at its number matches and an entity is there.
///
/// [`bits`]: Entities::bits
#[derive(Clone, Copy, Debug)]
pub struct Entities {
    /// `gameLocal` (rva): a pointer to `idGameLocal`.
    pub game_local: usize,
    /// `idGameLocal`: the spawn ids (i32 a number) and the entities (pointers) by number.
    pub spawn_ids: usize,
    pub entities: usize,
    pub bits: u32,
}

/// The minigame the player is in (hotwiring): a minigame sets itself on the player's HUD as it
/// starts, and the HUD clears it when it ends. The hotwiring game reads both sticks whole.
#[derive(Clone, Copy, Debug)]
pub struct Minigame {
    /// `idPresentablePlayer`: its `idPlayerHud`.
    pub hud: usize,
    /// `idPlayerHud`: the minigame's entity handle ([`Entities`]).
    pub handle: usize,
}

/// A machine the player climbs into and drives (`idPlayerControlled`: TNO's mech, power loader and
/// tunnel glider). As the player climbs on, it puts its presentable on top of the player's
/// controlled presentables, so the view camera getter answers it. It turns its view by the change
/// in the player's command angles each frame (times factors of its own, pitch clamped to its look
/// limits), as a mounted gun does.
#[derive(Clone, Copy, Debug)]
pub struct Driven {
    /// A presentable: its entity (as the presentable player reads it when it takes control).
    pub entity: usize,
    /// `idPlayerControlled::CastTo(entity) -> entity or null` (rva, [`CastFn`]): asks the entity's
    /// type, nothing else.
    pub cast: usize,
}

/// `Driven::cast`.
pub type CastFn = unsafe extern "system" fn(usize) -> usize;

/// How the mech (`idPlayerControlledFaust`) turns its aim: each command, its use of the player's
/// commands hands the change in their angles to its view angles' update, which adds the pitch's
/// change times the mech's own scale to its pitch and clamps that to its look limits (degrees,
/// down positive). The driver's arms pull the levers by that change.
#[derive(Clone, Copy, Debug)]
pub struct MechAim {
    /// The mech's view angles update (mech, pitch change, yaw change).
    pub view_angles: Site<MechViewAnglesFn>,
    /// `idPlayerControlled`: the view angles' pitch, the look limits' pitches (f32).
    pub pitch: usize,
    pub min_pitch: usize,
    pub max_pitch: usize,
    /// `idPlayerControlledFaust`: the scale on the pitch's change (f32).
    pub pitch_scale: usize,
}

pub type MechViewAnglesFn = unsafe extern "system" fn(usize, f32, f32);

/// The hotwiring minigame (`idHotWireEvent`): each frame of it, its update reads four stick
/// values from the player's command and sets each hand's pose spring from them: the left stick's
/// sideways and forward values move the left hand right and up ([`HOTWIRE_STICKS`]), the right
/// stick's the right hand, as positions (a stick held halfway holds the hand halfway), with the
/// difficulty's tremble added. The wires' tips are tags on the panel; they touch when both are
/// close enough in the player's view, and a touch held long enough clears a stage.
#[derive(Clone, Copy, Debug)]
pub struct Hotwire {
    /// The hotwiring minigame's update (event, `idPlayer`) ([`HotwireUpdateFn`]): only while the
    /// player hotwires; the player null when there is none.
    pub update: Site<HotwireUpdateFn>,
    /// `idPresentablePlayer`: the command its use of the player reads (`usercmd_t`).
    pub command: usize,
}

pub type HotwireUpdateFn = unsafe extern "system" fn(usize, usize);
/// `usercmd_t` (engine reflection): the left hand's sideways and forward values, then the right
/// hand's (`rightmove`, `forwardmove`, `rightmoveSec`, `forwardmoveSec`; signed bytes, right and
/// up positive).
pub const HOTWIRE_STICKS: [usize; 4] = [0x7, 0x6, 0xa, 0x9];
/// `idHotWireEvent`: the left hand's pose spring target, then the right's (`idVec2`, -1..1 each).
pub const HOTWIRE_SPRINGS: [usize; 2] = [0x1008, 0x1048];
pub type HandsFovScaleFn = unsafe extern "system" fn(usize) -> f32;
pub type HandsDepthHackFn = unsafe extern "system" fn(usize) -> f32;
/// `idHands`: its own model's presentable; a presentable's render entity; a render entity's depth
/// squeeze (float; 1 none).
pub const HANDS_PRESENTABLE: usize = 0x378;
pub const PRESENTABLE_RENDER_ENTITY: usize = 0xa8;
pub const RENDER_ENTITY_DEPTH_HACK: usize = 0xd4;
pub const HANDS_FOV_SCALE_PROLOGUE: [Instruction; 2] = [
    Instruction::plain(&[0x40, 0x53]),             // push rbx
    Instruction::plain(&[0x48, 0x83, 0xec, 0x40]), // sub rsp, 0x40
];
/// The game's rate setter `(hz)`: the time manager's rate, its tick in microseconds and whole
/// milliseconds, from `hz` (ignored unless above 0).
pub type GameRateFn = unsafe extern "system" fn(i32);
/// In the weapon selection flags (`Build::weapon_selection_flags`): the weapon wheel is open. The
/// selection update sets it at the wheel button's press and clears it when the wheel closes; while
/// it is set with the pad in use, the player's view update leaves the view angles alone and the
/// wheel picks by the right stick.
pub const WEAPON_WHEEL_OPEN: u32 = 4;

/// `idSWF::RenderSprite(swf, gui, sprite instance, render state, ?, ?)`: draws a sprite instance
/// and, depth first, every sprite and shape in its display list into the GUI model. Six
/// arguments, the last two stack slots passed through as found.
pub type RenderSpriteFn = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize);
pub const RENDER_SPRITE_PROLOGUE: [Instruction; 2] = [
    Instruction::plain(&[0x48, 0x8b, 0xc4]),       // mov rax, rsp
    Instruction::plain(&[0x4c, 0x89, 0x48, 0x20]), // mov [rax+0x20], r9
];
/// `idPlayerHud_Render::Render(hud, gui, player, time)`: each frame on the game thread, the HUD
/// updates its pieces and draws its movie into the GUI model.
pub type HudRenderFn = unsafe extern "system" fn(usize, usize, usize, usize);
pub const HUD_RENDER_PROLOGUE: [Instruction; 3] = [
    Instruction::plain(&[0x48, 0x8b, 0xc4]), // mov rax, rsp
    Instruction::plain(&[0x55]),             // push rbp
    Instruction::plain(&[0x53]),             // push rbx
];
/// In each ammo group, left first: its sprite (`ammo_group_left_new`, `ammo_group_right_new`),
/// and the byte set once everything it draws was found.
pub const AMMO_GROUP_SPRITE: [usize; 2] = [0xa8, 0x18];
pub const AMMO_GROUP_READY: [usize; 2] = [0x98, 0x8];
/// The GL draw of a material's geometry, `(&material, &geometry, GL state, ?)`: binds the
/// material's program for the GL state (a 64-bit set of flags, of which bits 50 and up are unused) and draws the geometry. The
/// render thread draws each GUI surface with it, the surface's state as the GUI model kept it.
pub type DrawElementsFn = unsafe extern "system" fn(usize, usize, u64, usize);
/// A GUI model surface: its GL state (u64), material, vertex count (i32), first index, index
/// count.
/// `idSWFSpriteInstance`: its name (a C string pointer) and its parent sprite.
pub const SPRITE_NAME: usize = 0x8;
pub const SPRITE_PARENT: usize = 0x28;
pub const GUI_SURFACE_SIZE: usize = 0x20;
pub const GUI_SURFACE_STATE: usize = 0;
pub const GUI_SURFACE_VERTICES: usize = 0x10;
/// A GUI model holds at most this many surfaces.
pub const GUI_SURFACES: i32 = 0x100;
pub const GUI_VERTEX_SIZE: usize = 0x20;
pub type GuiDrawFn =unsafe extern "system" fn(usize, usize, i32, usize, usize, f32);
/// The record's rectangle in the window: x0, y0, x1, y1 (the right eye's starts at half width).
pub const GUI_RECORD_RECT: usize = 0;
/// The record's world (pointer): set when the view draws the 3D world, the 2D layer then drawn
/// over it; null for a view with the 2D layer alone (a paused menu, a loading screen).
pub const GUI_RECORD_WORLD: usize = 0x18;

/// Whether the layout table holds `layout` with its expected name.
pub fn layout_present(exe: &Module, layout: (usize, &str)) -> bool {
    mem::read::<usize>(exe.at(layout.0 - LAYOUT_HEADER)).and_then(|p| mem::read_c_string(p, 32)).is_some_and(|n| n == layout.1)
}

/// The layout currently in use, if it is one of `known`.
pub fn current_layout(exe: &Module, build: &Build, known: &[Layout]) -> Option<Layout> {
    let pointer = mem::read::<usize>(exe.at(build.layout_global))?;
    known.iter().copied().find(|layout| exe.at(layout.0) == pointer)
}


/// The view builder `(this, player_views, ...)`: once per frame on the game thread it makes one
/// render view per layout view from the player's render view (eye offset and projection
/// included). The player views (`Build::player_view_stride` apart) are on its caller's stack,
/// rebuilt every frame.
pub const VIEW_BUILDER_PROLOGUE: [Instruction; 3] = [
    Instruction::plain(&[0x4c, 0x8b, 0xdc]),                               // mov r11, rsp
    Instruction::plain(&[0x56]),                                           // push rsi
    Instruction::plain(&[0x48, 0x81, 0xec, 0x10, 0x01, 0x00, 0x00]),       // sub rsp, 0x110
];
/// Six pointer-sized arguments: the four in registers and two stack slots passed through as found.
pub type ViewBuilderFn = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize) -> usize;

/// The view slots of the frame the view builder fills, one per record of the screen layout (the
/// side-by-side layout fills two where the game's own fills one): a list in the view builder's
/// object (its storage pointer, then count and capacity as i32), slots `stride` apart, each with
/// its render view's environment blend lists ([`ENV_LISTS`] of them, [`ENV_LIST_SIZE`] apart from
/// `env_lists`: storage pointer, count, capacity), which the builder fills by copying into them.
#[derive(Clone, Copy, Debug)]
pub struct ScreenViews {
    pub list: usize,
    pub stride: usize,
    pub env_lists: usize,
}

pub const ENV_LISTS: usize = 3;
pub const ENV_LIST_SIZE: usize = 0x18;
pub const PLAYER_VIEWS: usize = 4;
/// Byte: the player view is in use.
pub const PLAYER_ACTIVE: usize = 0x20;
pub const PLAYER_RENDER_VIEW: usize = 0x30;
/// Render view fields: the full horizontal and vertical field of view (2 floats, degrees), and the
/// origin (3 floats) directly followed by the axis rows forward, left, up (9 floats).
pub const VIEW_FOV: usize = 0x10;
pub const VIEW_ORIGIN_AXIS: usize = 0x60;

/// The projection builder `(fov_x, fov_y, near, far, shift_x, shift_y, out)`: the view builder
/// calls it once per stereo view, in layout order (left eye first). It hands a glFrustum-style
/// `(left, right, bottom, top, near, far, out)` to the frustum maker (`Build::make_frustum`).
pub const PROJECTION_PROLOGUE: [Instruction; 1] = [Instruction::plain(&[0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00])]; // sub rsp, 0x88
pub type ProjectionFn = unsafe extern "system" fn(f32, f32, f32, f32, f32, f32, *mut f32);
use eng_idt::MakeFrustumFn;

/// The usercmd builder `(generator, out usercmd, player, settings, flag) -> usercmd*` (a virtual
/// method): once per command it adds the keyboard, mouse and stick look input into the player's
/// accumulated view angles, normalises them and packs them into the command. Its prologue
/// differs between builds (`Build::usercmd_builder_prologue`).
pub type UsercmdBuilderFn = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize) -> usize;
/// Where the command's look angles are: the player's accumulated view angles at `generator +
/// 0xc + player * 0x15c`, then pitch (+4), yaw (+8), roll (+0xc); the command's angles as
/// shorts at +0xc; the generator's own copy of the command at +0x5f8.
pub const COMMANDS: eng_idt::usercmd::Commands =
    eng_idt::usercmd::Commands { accumulator: 0xc, stride: 0x15c, pitch: 4, yaw: 8, angles: 0xc, generator_command: 0x5f8, players: 4 };

/// The hands' item placer `(item, hands)` (an `idHandsItem` and its `idHands`): each frame on the
/// game thread it puts the item's model (a gun) on its tag of the hands model (`"RightHand"`,
/// `"LeftHand"`, by [`ITEM_TAG`]) and points its muzzle and laser from there. Its first tag
/// transform ([`TAG_TRANSFORM_PROLOGUE`]) is that hand tag's.
pub type PlaceItemFn = unsafe extern "system" fn(usize, usize);
/// A tag's world transform `(animated entity, tag, out origin, out axis)`: three floats in world
/// units, then three axis rows (forward, left, up).
pub const TAG_TRANSFORM_PROLOGUE: [Instruction; 1] = [Instruction::plain(&[0x48, 0x89, 0x5c, 0x24, 0x10])]; // mov [rsp+0x10], rbx
pub type TagTransformFn = unsafe extern "system" fn(usize, usize, *mut f32, *mut f32);
/// The deferred fire queue `(manager, fire parms) -> handle`: every shot's trace starts at the parms'
/// start and runs along the first row of their fire axis (spread added around it).
pub type FireQueueFn = unsafe extern "system" fn(usize, usize) -> u32;
/// The hands' blend kick `(idHands*)`: each frame on the game thread, before the hands place their
/// items, it starts the arms model's blend job for this frame (drawn the next frame), which builds
/// the model-space joints into the animator's back pose ([`HandsLayout::back_joints`]).
pub type ArmsKickFn = unsafe extern "system" fn(usize);
/// A blend job's pose bounds `(skeleton header, final pose, joint to skip, blend outputs)`: called
/// once the job's model-space joints are final (its overrides applied) and right before it skins
/// from them, on a worker thread.
pub type PoseBoundsFn = unsafe extern "system" fn(usize, usize, i16, usize);

/// The arms model's placement and the engine's own test of a scripted first-person moment: the
/// hands place their model each frame from the player's first-person view
/// (`HandsLayout::player_view`), and while the arms' animation has [`CAMERA_JOINT`] the view is
/// that joint instead (the game's camera rides the arms).
#[derive(Clone, Copy, Debug)]
pub struct FollowHead {
    /// The hands' placement from the player's view ([`UpdatePositionFn`], rva).
    pub update_position: usize,
    /// The hands' animator ([`HandsAnimatorFn`], rva).
    pub animator: usize,
    /// An animator's joint by name ([`JointIndexFn`], rva).
    pub joint_index: usize,
    /// A hand item's draw ([`ItemDrawFn`], rva): an item worn on the view ([`ITEM_ON_VIEW`],
    /// goggles) is placed there again, on the player's view object's position.
    pub item_draw: usize,
    /// `idPresentablePlayer`: its view object's position (origin, then axis rows), where a worn
    /// item is drawn.
    pub view_position: usize,
}

/// `(item, hands)`.
pub type ItemDrawFn = unsafe extern "system" fn(usize, usize);
/// A hand item's kind (i32 at its start) when it is worn on the view rather than held.
pub const ITEM_ON_VIEW: i32 = 3;

/// `(hands)`.
pub type UpdatePositionFn = unsafe extern "system" fn(usize);
/// `(hands) -> animator` (null without a model).
pub type HandsAnimatorFn = unsafe extern "system" fn(usize) -> usize;
/// `(animator, out, name) -> out`: the joint's index written to `out` (-1: none of that name).
pub type JointIndexFn = unsafe extern "system" fn(usize, *mut i16, *const u8) -> *mut i16;
/// The arms' joint a scripted first-person moment puts the game's camera on.
pub const CAMERA_JOINT: &[u8] = b"player_camera_game\0";

/// `idHands`: its player (`idPresentablePlayer`) and its arms model (`idPresentableAnimatedEntity`).
pub const HANDS_PLAYER: usize = 0x18;
pub const HANDS_ARMS_MODEL: usize = 0x378;
/// A model's animated entity: its origin (3 floats), axis rows (9) and scale (3).
pub const ANIMATED_ORIGIN: usize = 0x68;
pub const ANIMATED_AXIS: usize = 0x74;
pub const ANIMATED_SCALE: usize = 0x98;
pub const JOINT_SIZE: usize = 0x30;
/// `idHandsItem`: the hand tag it sits on (i32: 0 right hand, 1 left hand), the item (the
/// `idWeapon` for a gun) and its model.
pub const ITEM_TAG: usize = 0x90;
pub const ITEM_WEAPON: usize = 0x48;
/// `idInventoryItem` (an `idWeapon`'s base): its decl (for a weapon an `idDeclWeapon`).
pub const INVENTORY_ITEM_DECL: usize = 0x18;
/// A decl's name (a pointer to a C string, as `weapon/...`: what the console's `testWeapon` takes).
pub const DECL_NAME: usize = 0x8;
/// The weapons (by decl name) the game throws by the right stick's press, as a shot of their own
/// (the knife: throwing knives).
pub const THROWS_KNIVES: &[&str] = &["weapon/knife"];

/// The weapons (by decl name) whose target is the player's focus (what the crosshair is on: the
/// laser that cuts chains finds what it cuts there): while one is held the focus traces along its
/// barrel.
pub const AIMED_BY_FOCUS: &[&str] = &["weapon/lasercutter"]; // monaka-lint: allow

/// The weapons (by decl name) carried in both hands from the middle: the right hand's tag is the
/// item's centre, each hand on an edge (the tea tray). They go between both controllers.
pub const CARRIED_TWO_HANDED: &[&str] = &["weapon/specialweapon/teatray"];
/// `idFireParms`: the weapon firing, the trace's start (3 floats) and the fire axis (rows forward,
/// left, up).
pub const FIRE_WEAPON: usize = 0x8;
pub const FIRE_START: usize = 0x48;
pub const FIRE_AXIS: usize = 0x64;

#[cfg(test)]
mod tests;
