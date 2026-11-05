//! Probe (`probe_hud_sprites=1`, once per sprite, capped): which of the HUD movie's sprites are drawn where.
//! The first time a sprite draws anything (`providers::hud_pieces`' `idSWF::RenderSprite`), its
//! path (its name and its parents', `hudmain/commanderSlots/item0`) is logged with the box it drew
//! in, as fractions of the HUD's layout (left, top, right, bottom): which sprite a piece of the HUD
//! seen at a place is. A sprite's box holds its children's. Changes nothing.

use crate::providers::engine::{SPRITE_NAME, SPRITE_PARENT};
use monaka_hook::mem;
use monaka_producer::log;
use std::collections::HashSet;
use std::sync::Mutex;

/// Sprites logged at most.
const MOST: usize = 300;

static SEEN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether sprite `sprite`'s drawing is wanted (the probe on, not logged yet, room left).
pub fn wanted(sprite: usize) -> bool {
    sprite != 0 && super::options().hud_sprites && SEEN.try_lock().is_ok_and(|seen| seen.as_ref().is_none_or(|s| s.len() < MOST && !s.contains(&sprite)))
}

/// Sprite `sprite` drew within `bounds` (layout units) of a layout of `layout`.
pub fn drawn(sprite: usize, bounds: [f32; 4], layout: (f32, f32)) {
    let Ok(mut seen) = SEEN.lock() else { return };
    let seen = seen.get_or_insert_with(HashSet::new);
    if seen.len() >= MOST || !seen.insert(sprite) {
        return;
    }
    let at = [bounds[0] / layout.0, bounds[1] / layout.1, bounds[2] / layout.0, bounds[3] / layout.1];
    log!("HUD sprite {} drawn at {at:.3?} of the layout", path(sprite));
}

/// `sprite`'s path: its parents' names, then its own.
fn path(sprite: usize) -> String {
    let mut names = Vec::new();
    let mut at = sprite;
    while at != 0 && names.len() < 10 {
        let name = mem::read::<usize>(at + SPRITE_NAME).filter(|&p| p != 0).and_then(|p| mem::read_c_string(p, 64)).unwrap_or_else(|| "?".to_owned());
        names.push(name);
        at = mem::read::<usize>(at + SPRITE_PARENT).unwrap_or(0);
    }
    names.reverse();
    names.join("/")
}
