//! Probe (`probe_cvars=<name>@<rva>=<value>,...`): integer cvars set for the run (put back at the
//! stop), each named and placed as the game's cvar list has it (the rva of its name pointer), to try
//! a renderer switch against a fault seen in the headset. A cvar whose name is not at its rva in
//! this build is left alone.

use eng_idt::Cvar;
use monaka_core::sync::Locked;
use monaka_hook::module::Module;
use monaka_producer::ledger::Ledger;
use monaka_producer::log;
use std::sync::Mutex;

/// How long each step of `probe_cycle` lasts: two of the GPU-times lines' 10 s windows, so one
/// window lies wholly inside it.
const CYCLE_STEP: std::time::Duration = std::time::Duration::from_secs(20);

/// `probe_cycle`: the candidates, what each cvar had, and which candidate is on.
struct Cycle {
    exe: Module,
    /// Each candidate's label (as asked) and the integer values it sets.
    candidates: Vec<(String, Vec<(&'static Cvar, i32)>)>,
    /// Each cvar any candidate sets, with its integer and float values at the start.
    originals: Vec<(&'static Cvar, i32, f32)>,
    started: Option<std::time::Instant>,
    on: Option<usize>,
}

static CYCLE: Mutex<Option<Cycle>> = Mutex::new(None);

/// Probe (`probe_cycle=<candidate>,...`, a candidate `name=value` or several joined by `+`): in play,
/// the game as it is for [`CYCLE_STEP`], then the next candidate for a step, the game as it is
/// again, and so on round; each switch logged, so the GPU-times lines between tell each
/// candidate's cost. A menu puts the game as it was (nothing tried is saved in its config on
/// quitting), and the ledger puts each cvar back at the stop. Cvars found by name in the running
/// game ([`Cvar::find`]).
pub fn prepare_cycle(ledger: &mut Ledger, exe: &Module) {
    let asked = super::options().cycle.clone();
    if asked.is_empty() {
        return;
    }
    let mut candidates = Vec::new();
    let mut originals: Vec<(&'static Cvar, i32, f32)> = Vec::new();
    for candidate in asked.split(',').map(str::trim).filter(|c| !c.is_empty()) {
        let mut sets = Vec::new();
        for part in candidate.split('+') {
            let Some((name, value)) = part.split_once('=').and_then(|(n, v)| Some((n.trim(), v.trim().parse::<i32>().ok()?))) else {
                log!("probe_cycle: {part:?} is not name=value");
                continue;
            };
            let name: &'static str = Box::leak(name.to_owned().into_boxed_str());
            let Some(cvar) = Cvar::find(exe, name, eng_idt::Layout::ID_TECH_5) else { continue };
            if !originals.iter().any(|(c, ..)| c.name_pointer == cvar.name_pointer) {
                let (int, float) = (cvar.int(exe).unwrap_or(0), cvar.float(exe).unwrap_or(0.0));
                let _ = ledger.keep(eng_idt::IntCvar { exe: *exe, cvar });
                log!("probe_cycle: {name} is {int} ({float})");
                originals.push((cvar, int, float));
            }
            sets.push((cvar, value));
        }
        if !sets.is_empty() {
            candidates.push((candidate.to_owned(), sets));
        }
    }
    log!("probe_cycle: {} candidates, {} s each between steps as the game is", candidates.len(), CYCLE_STEP.as_secs());
    *CYCLE.locked() = Some(Cycle { exe: *exe, candidates, originals, started: None, on: None });
}

/// At each SwapBuffers (render thread): the `probe_cycle` step due ([`prepare_cycle`]).
pub fn cycle_frame() {
    let mut cycle = CYCLE.locked();
    let Some(cycle) = cycle.as_mut().filter(|c| !c.candidates.is_empty()) else { return };
    let playing = crate::providers::eye_site::builds() > 0 && !crate::providers::eye_site::in_menu();
    let due = playing.then(|| {
        let step = cycle.started.get_or_insert_with(std::time::Instant::now).elapsed().as_secs() / CYCLE_STEP.as_secs();
        (step % 2 == 1).then(|| (step / 2) as usize % cycle.candidates.len())
    });
    let due = due.flatten();
    if due == cycle.on {
        return;
    }
    for (cvar, int, float) in &cycle.originals {
        cvar.set_int(&cycle.exe, *int);
        cvar.set_float(&cycle.exe, *float);
    }
    match due {
        Some(index) => {
            let (label, sets) = &cycle.candidates[index];
            for (cvar, value) in sets {
                cvar.set_int(&cycle.exe, *value);
            }
            log!("probe_cycle: on {label}");
        }
        None => log!("probe_cycle: the game as it is{}", if playing { "" } else { " (a menu)" }),
    }
    cycle.on = due;
}

/// The cvars asked for so far: the ledger keeps a cvar for the run, so each lives for good, made
/// once whatever the number of runs.
static KNOWN: Mutex<Vec<&'static Cvar>> = Mutex::new(Vec::new());

/// Sets the cvars `probe_cvars` asks for, on `ledger`.
pub fn apply(ledger: &mut Ledger, exe: &Module) {
    let asked = super::options().cvars.clone();
    for entry in asked.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let Some((name, rva, value)) = parse(entry) else {
            log!("probe_cvars: {entry:?} is not name@rva=value");
            continue;
        };
        eng_idt::set_int(ledger, exe, known(name, rva), value, "probe_cvars leaves it as it is");
    }
}

/// The cvar `name` at `rva`, made the first time it is asked for.
fn known(name: &str, rva: usize) -> &'static Cvar {
    let mut known = KNOWN.locked();
    if let Some(cvar) = known.iter().find(|c| c.name == name && c.name_pointer == rva) {
        return cvar;
    }
    let cvar: &'static Cvar = Box::leak(Box::new(Cvar::id5(rva, Box::leak(name.to_owned().into_boxed_str()))));
    known.push(cvar);
    cvar
}

/// `name@rva=value` (rva in hex, `0x` optional) as the name, the rva and the value.
pub fn parse(entry: &str) -> Option<(&str, usize, i32)> {
    let (name, rest) = entry.split_once('@')?;
    let (rva, value) = rest.split_once('=')?;
    let rva = usize::from_str_radix(rva.trim().trim_start_matches("0x"), 16).ok()?;
    Some((name.trim(), rva, value.trim().parse().ok()?))
}

#[cfg(test)]
mod tests;
