//! id Tech 5 (Wolfenstein: The New Order and The Old Blood), id Tech 6 (Wolfenstein II) and the
//! Void engine fork (Dishonored 2 and Death of the Outsider): what the producers share about the
//! engine's cvars, written once. A cvar is found by the RVA of the pointer to its name
//! and read or written at fixed offsets before that pointer (which differ between 5 and 6:
//! [`Layout`]); [`set_int`] and [`set_float`] set cvars for a run on its ledger
//! (`monaka_producer::ledger`), put back at the stop. Each game crate supplies its build's RVAs.
//!
//! [`view`]: the player's render view at the view builder (id Tech 5 and the Void engine).
//! [`usercmd`]: the input command's look angles, rewritten after the game built them.

pub mod pacing;
pub mod usercmd;
pub mod view;

use monaka_core::convention::CameraConvention;
use monaka_hook::{Instruction, Site, mem};
use monaka_hook::module::Module;
use monaka_producer::ledger::{Change, Ledger, Setting};
use monaka_producer::log;

/// Where an `idCVar` keeps its values, before the pointer to its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Bytes before the name pointer to the integer value, and to the float value.
    pub int_before: usize,
    pub float_before: usize,
}

impl Layout {
    /// id Tech 5 and the Void engine: the integer 8 bytes and the float 4 bytes before the name.
    pub const ID_TECH_5: Layout = Layout { int_before: 8, float_before: 4 };
    /// id Tech 6: 0x18 and 0x14 before.
    pub const ID_TECH_6: Layout = Layout { int_before: 0x18, float_before: 0x14 };
}

/// A cvar by the RVA of the pointer to its name (checked against the name: [`Cvar::present`]).
#[derive(Clone, Copy, Debug)]
pub struct Cvar {
    pub name_pointer: usize,
    pub name: &'static str,
    pub layout: Layout,
}

impl Cvar {
    pub const fn new(name_pointer: usize, name: &'static str, layout: Layout) -> Self {
        Self { name_pointer, name, layout }
    }

    /// An id Tech 5 (or Void engine) cvar.
    pub const fn id5(name_pointer: usize, name: &'static str) -> Self {
        Self::new(name_pointer, name, Layout::ID_TECH_5)
    }

    /// An id Tech 6 cvar.
    pub const fn id6(name_pointer: usize, name: &'static str) -> Self {
        Self::new(name_pointer, name, Layout::ID_TECH_6)
    }

    /// A cvar of `exe` its build's table does not list, found by `name` as the engine holds it
    /// now (its name pointers are written at start-up, so it is looked for in the running image,
    /// never the file): the pointer to the name string in the image's data whose values before it
    /// agree (the integer the float's whole part, as the engine keeps both). A scan of the image's
    /// data, once a run.
    pub fn find(exe: &Module, name: &'static str, layout: Layout) -> Option<&'static Cvar> {
        let code = exe.kept_code()?;
        let data = || code.sections().iter().filter(|s| !s.executable);
        let needle = [&[0u8][..], name.as_bytes(), &[0]].concat();
        // The name may be in the data more than once (a format string, a menu's binding): each copy
        // is tried, the cvar's own being the one a cvar points at.
        let strings: Vec<u64> = data()
            .flat_map(|s| {
                let rva = s.rva as usize;
                s.bytes.windows(needle.len()).enumerate().filter(|(_, w)| *w == needle.as_slice()).map(move |(at, _)| (exe.base() + rva + at + 1) as u64).collect::<Vec<_>>()
            })
            .collect();
        let found = |target: u64| {
            data()
                .flat_map(move |s| {
                    let rva = s.rva as usize;
                    s.bytes.as_chunks::<8>().0.iter().enumerate().filter(move |(_, b)| u64::from_le_bytes(**b) == target).map(move |(i, _)| rva + i * 8)
                })
                .find_map(|at| {
                    let cvar = Cvar::new(at, name, layout);
                    cvar.int(exe).zip(cvar.float(exe)).is_some_and(|(i, f)| f.is_finite() && f as i32 == i).then_some(cvar)
                })
        };
        let cvar = strings.into_iter().find_map(found);
        match cvar {
            Some(cvar) => log!("{name}: found at {:#x}", cvar.name_pointer),
            None => log!("{name}: not found in the running game"),
        }
        Some(Box::leak(Box::new(cvar?)))
    }

    /// Whether this build has the cvar where expected (the name pointer names it).
    pub fn present(&self, exe: &Module) -> bool {
        mem::read::<usize>(exe.at(self.name_pointer)).and_then(|p| mem::read_c_string(p, 64)).is_some_and(|n| n == self.name)
    }

    pub fn float(&self, exe: &Module) -> Option<f32> {
        mem::read::<f32>(exe.at(self.name_pointer - self.layout.float_before))
    }

    pub fn set_float(&self, exe: &Module, value: f32) -> bool {
        mem::write(exe.at(self.name_pointer - self.layout.float_before), value)
    }

    pub fn int(&self, exe: &Module) -> Option<i32> {
        mem::read::<i32>(exe.at(self.name_pointer - self.layout.int_before))
    }

    /// Sets the integer and float values together (what the engine's direct reads use; the value
    /// string keeps the old text).
    pub fn set_int(&self, exe: &Module, value: i32) -> bool {
        mem::write(exe.at(self.name_pointer - self.layout.int_before), value) && self.set_float(exe, value as f32)
    }
}

/// An integer cvar of one module, as a [`Setting`] (its integer and float values written together).
#[derive(Clone, Copy, Debug)]
pub struct IntCvar {
    pub exe: Module,
    pub cvar: &'static Cvar,
}

impl Setting for IntCvar {
    type Value = i32;
    fn name(&self) -> String {
        self.cvar.name.to_owned()
    }
    fn read(&self) -> Option<i32> {
        self.cvar.int(&self.exe)
    }
    fn write(&self, value: &i32) -> bool {
        self.cvar.set_int(&self.exe, *value)
    }
}

/// A float cvar of one module, as a [`Setting`].
#[derive(Clone, Copy, Debug)]
pub struct FloatCvar {
    pub exe: Module,
    pub cvar: &'static Cvar,
}

impl Setting for FloatCvar {
    type Value = f32;
    fn name(&self) -> String {
        self.cvar.name.to_owned()
    }
    fn read(&self) -> Option<f32> {
        self.cvar.float(&self.exe)
    }
    fn write(&self, value: &f32) -> bool {
        self.cvar.set_float(&self.exe, *value)
    }
}

/// Sets an integer cvar for the run on `ledger` (the run's: put back at the stop), remembering what
/// it had; `why` says what stays as it is when the build keeps the cvar elsewhere. Whether it was
/// changed.
pub fn set_int(ledger: &mut Ledger, exe: &Module, cvar: &'static Cvar, value: i32, why: &str) -> bool {
    present(exe, cvar, why) && ledger.set(IntCvar { exe: *exe, cvar }, value) == Change::Changed
}

/// [`set_int`] for a float cvar.
pub fn set_float(ledger: &mut Ledger, exe: &Module, cvar: &'static Cvar, value: f32, why: &str) -> bool {
    present(exe, cvar, why) && ledger.set(FloatCvar { exe: *exe, cvar }, value) == Change::Changed
}

/// Whether the build has `cvar` where expected; logs `why` (what stays as it is) when not.
fn present(exe: &Module, cvar: &Cvar, why: &str) -> bool {
    let present = cvar.present(exe);
    if !present {
        log!("{} is not where this build keeps it; {why}", cvar.name);
    }
    present
}

/// How the engine stores a camera (render view: origin, then axis rows forward, left, up), in
/// `units_per_metre` game units a metre.
pub fn camera_convention(units_per_metre: f32) -> CameraConvention {
    CameraConvention { units_per_metre, ..CameraConvention::ID_TECH }
}

/// The engine's glFrustum-style matrix maker `(left, right, bottom, top, near, far, out)`, the one
/// its projection builder ends in: the same code in id Tech 5 and 6.
pub type MakeFrustumFn = unsafe extern "system" fn(f32, f32, f32, f32, f32, f32, *mut f32);

/// The frustum maker's first instructions: `sub rsp, 0x38; mov rax, [rsp+0x70]`.
pub const MAKE_FRUSTUM_PROLOGUE: &[Instruction] = &[Instruction::plain(&[0x48, 0x83, 0xec, 0x38]), Instruction::plain(&[0x48, 0x8b, 0x44, 0x24, 0x70])];

/// The frustum maker at `rva` of a build.
///
/// # Safety
/// As `Site::new`: `rva` is the frustum maker in the build the table's fingerprint names.
pub const unsafe fn make_frustum(rva: usize) -> Site<MakeFrustumFn> {
    // SAFETY: forwarded from the caller.
    unsafe { Site::new("make_frustum", rva, MAKE_FRUSTUM_PROLOGUE) }
}
