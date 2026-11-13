//! Probe (`probe_profile`): where the render and game threads' time goes. Once gameplay has run a
//! while, a sampler thread stops each of them about 2000 times a second for [`SPAN`], reads where
//! it is (and the nearest return address into the game on its stack), and logs per thread the
//! shares by phase (the render thread's own work, Monaka's at SwapBuffers, the game's SwapBuffers),
//! by module, the busiest places, and the game's callers of the time spent outside the game. Over
//! the same span it logs every thread's CPU time (a share of one core) and where it started.

use monaka_core::sync::Locked;
use monaka_hook::mem;
use monaka_hook::module::Module;
use monaka_producer::log;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering::*};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{CloseHandle, FILETIME, HANDLE};
use windows::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next};
use windows::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext};
use windows::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadTimes, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION,
    THREAD_SUSPEND_RESUME,
};

/// Gameplay time before sampling starts, and how long it samples.
const SETTLE: Duration = Duration::from_secs(12);
const SPAN: Duration = Duration::from_secs(10);
/// Between rounds (one sample of each thread).
const GAP: Duration = Duration::from_micros(500);
/// Stack scanned for a return address into the game (bytes).
const STACK_SCAN: usize = 4096;

/// What the render thread is doing, for the samples.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Phase {
    /// The game's own rendering, between SwapBuffers calls.
    Game = 0,
    /// Monaka's work in the SwapBuffers hook.
    Monaka = 1,
    /// The game's own SwapBuffers.
    Swap = 2,
}

static PHASE: AtomicU8 = AtomicU8::new(Phase::Game as u8);
static STARTED: AtomicBool = AtomicBool::new(false);
static FIRST: Mutex<Option<Instant>> = Mutex::new(None);
static GAME_THREAD: AtomicU32 = AtomicU32::new(0);

/// One sample: where the thread was, the first return address into the game above it, and the
/// render thread's phase.
#[derive(Clone, Copy)]
struct Sample {
    at: usize,
    caller: usize,
    phase: u8,
    /// The single pass's part of the frame, and the draws made so far.
    part: u8,
    draws: u64,
}

/// Whether the sampler runs this run: asked for, or a single-pass profiling build.
fn wanted() -> bool {
    super::options().profile || crate::single_pass::draws::profiling()
}

/// The draws made at each `SwapBuffers` while sampling, in order.
static SWAPS: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static SAMPLING: AtomicBool = AtomicBool::new(false);

/// The render thread's phase from here on.
pub fn phase(phase: Phase) {
    PHASE.store(phase as u8, Relaxed);
    if phase == Phase::Monaka && SAMPLING.load(Relaxed) {
        SWAPS.locked().push(crate::single_pass::draws::DRAWS_MADE.load(Relaxed));
    }
}

/// Each frame on the game thread (the view builder): which thread that is.
pub fn game_thread() {
    if wanted() && GAME_THREAD.load(Relaxed) == 0 {
        // SAFETY: the calling thread's own id.
        GAME_THREAD.store(unsafe { GetCurrentThreadId() }, Relaxed);
    }
}

/// At each SwapBuffers (render thread): starts the sampler once gameplay has run [`SETTLE`].
pub fn frame() {
    if !wanted() || STARTED.load(Relaxed) || crate::providers::eye_site::builds() == 0 {
        return;
    }
    let first = *FIRST.locked().get_or_insert_with(Instant::now);
    if first.elapsed() < SETTLE || STARTED.swap(true, AcqRel) {
        return;
    }
    // SAFETY: the calling thread's own id.
    let render = unsafe { GetCurrentThreadId() };
    let game = GAME_THREAD.load(Relaxed);
    std::thread::spawn(move || sample(render, game));
}

/// A thread being sampled: its handle and samples.
struct Sampled {
    name: &'static str,
    handle: HANDLE,
    samples: Vec<Sample>,
}

fn sample(render: u32, game: u32) {
    let Some(exe) = Module::find("WolfNewOrder_x64.exe").or_else(|| Module::find("WolfOldBlood_x64.exe")) else { return };
    let rounds = SPAN.as_micros() as usize / GAP.as_micros() as usize * 2;
    let mut threads = Vec::new();
    for (name, id) in [("render", render), ("game", game)] {
        if id == 0 {
            continue;
        }
        // SAFETY: a handle to one of our own threads, closed below.
        match unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, false, id) } {
            // Room for every sample, so nothing allocates while a thread is stopped (it may hold
            // the heap's lock).
            Ok(handle) => threads.push(Sampled { name, handle, samples: Vec::with_capacity(rounds) }),
            Err(_) => log!("probe_profile: the {name} thread could not be opened"),
        }
    }
    let mut stack = vec![0u8; STACK_SCAN];
    let mut context = CONTEXT { ContextFlags: CONTEXT_CONTROL_AMD64, ..Default::default() };
    let before = thread_times();
    SWAPS.locked().reserve(SPAN.as_secs() as usize * 400);
    SAMPLING.store(true, Relaxed);
    let started = Instant::now();
    'rounds: while started.elapsed() < SPAN {
        for thread in &mut threads {
            if thread.samples.len() == thread.samples.capacity() {
                break 'rounds;
            }
            // SAFETY: stops one of our threads; resumed below before anything else.
            if unsafe { SuspendThread(thread.handle) } == u32::MAX {
                break 'rounds;
            }
            let phase = PHASE.load(Relaxed);
            let part = crate::single_pass::draws::profile_part() as u8;
            let draws = crate::single_pass::draws::DRAWS_MADE.load(Relaxed);
            // SAFETY: the stopped thread's context into our local.
            let read = unsafe { GetThreadContext(thread.handle, &mut context) }.is_ok();
            let mut caller = 0;
            if read {
                let scanned = if mem::read_bytes(context.Rsp as usize, &mut stack) { STACK_SCAN } else { 0 };
                caller = stack[..scanned]
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|word| u64::from_le_bytes(*word) as usize)
                    .find(|&word| word != exe.base() && exe.contains(word))
                    .unwrap_or(0);
            }
            // SAFETY: the thread stopped above.
            unsafe { ResumeThread(thread.handle) };
            if read {
                thread.samples.push(Sample { at: context.Rip as usize, caller, phase, part, draws });
            }
        }
        std::thread::sleep(GAP);
    }
    let took = started.elapsed();
    SAMPLING.store(false, Relaxed);
    let swaps = std::mem::take(&mut *SWAPS.locked());
    let after = thread_times();
    let names = ModuleNames::now();
    log_threads(&names, &before, &after, took, render, game);
    for thread in &threads {
        // SAFETY: opened above.
        unsafe {
            let _ = CloseHandle(thread.handle);
        }
        report(thread.name, &exe, &names, &thread.samples, took, thread.name == "render");
        if thread.name == "render" {
            report_parts(&exe, &names, &thread.samples, &swaps);
        }
    }
}

/// The loaded modules and their file names.
struct ModuleNames(Vec<(Module, String)>);

impl ModuleNames {
    fn now() -> Self {
        Self(
            Module::loaded()
                .into_iter()
                .map(|m| {
                    let name = m.path().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_else(|| "?".into());
                    // The pack's file name carries the run's stamp.
                    let name = if name.starts_with("monaka_") { "monaka pack".to_owned() } else { name };
                    (m, name)
                })
                .collect(),
        )
    }

    fn of(&self, address: usize) -> Option<&(Module, String)> {
        self.0.iter().find(|(m, _)| m.contains(address))
    }

    fn name(&self, address: usize) -> String {
        self.of(address).map_or_else(|| "(none)".to_owned(), |(_, n)| n.clone())
    }
}

fn report(thread: &str, exe: &Module, names: &ModuleNames, samples: &[Sample], took: Duration, phased: bool) {
    let total = samples.len().max(1) as f64;
    let share = |count: usize| count as f64 * 100.0 / total;
    let place = |address: usize| match names.of(address) {
        Some((m, name)) if m == exe => format!("{name}+{:#x}", address - m.base()),
        Some((m, name)) => match nearest_export(m, address) {
            Some((export, offset)) => format!("{name}!{export}+{offset:#x}"),
            None => format!("{name}+{:#x}", address - m.base()),
        },
        None => format!("{address:#x}"),
    };
    log!("probe_profile: {thread} thread, {} samples over {:.1} s", samples.len(), took.as_secs_f64());

    if phased {
        log!("probe_profile: {thread}: by phase, then module");
        for (phase, label) in [(Phase::Game, "the game's rendering"), (Phase::Monaka, "Monaka at SwapBuffers"), (Phase::Swap, "the game's SwapBuffers")] {
            let of_phase: Vec<&Sample> = samples.iter().filter(|s| s.phase == phase as u8).collect();
            let waiting = of_phase.iter().filter(|s| is_wait(&place(s.at))).count();
            log!("  {:5.1}%  {label} ({:.1}% of it in a kernel wait)", share(of_phase.len()), waiting as f64 * 100.0 / of_phase.len().max(1) as f64);
            let mut by_module: HashMap<String, usize> = HashMap::new();
            for s in &of_phase {
                *by_module.entry(names.name(s.at)).or_default() += 1;
            }
            for (name, count) in top(by_module, 6) {
                log!("           {:5.1}%  {name}", share(count));
            }
        }
    }

    let mut by_module: HashMap<String, usize> = HashMap::new();
    for s in samples {
        *by_module.entry(names.name(s.at)).or_default() += 1;
    }
    log!("probe_profile: {thread}: by module");
    for (name, count) in top(by_module, 10) {
        log!("  {:5.1}%  {name}", share(count));
    }

    // The busiest places, a function's worth apart (64 bytes) folded together.
    let mut by_place: HashMap<usize, usize> = HashMap::new();
    for s in samples {
        *by_place.entry(s.at & !0x3f).or_default() += 1;
    }
    log!("probe_profile: {thread}: busiest places (64-byte buckets)");
    for (at, count) in top(by_place, 15) {
        log!("  {:5.1}%  {}", share(count), place(at));
    }

    // Time outside the game, by the game's call that led there.
    let mut by_caller: HashMap<(String, usize), usize> = HashMap::new();
    for s in samples.iter().filter(|s| !exe.contains(s.at)) {
        *by_caller.entry((names.name(s.at), s.caller)).or_default() += 1;
    }
    log!("probe_profile: {thread}: outside the game, by module and the game's return address");
    for ((module, caller), count) in top(by_caller, 20) {
        let caller = if caller == 0 { "(none on the stack)".to_owned() } else { format!("exe+{:#x}", caller - exe.base()) };
        log!("  {:5.1}%  {module} from {caller}", share(count));
    }
}

/// Each thread of this process: its CPU time so far and where it started.
fn thread_times() -> HashMap<u32, (Duration, usize)> {
    let mut out = HashMap::new();
    let query = Module::find("ntdll.dll").and_then(|m| m.export("NtQueryInformationThread"));
    // SAFETY: a snapshot of the system's threads, walked and closed here; each thread opened only to
    // read its times and start address.
    unsafe {
        let Ok(snapshot) = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) else { return out };
        let process = GetCurrentProcessId();
        let mut entry = THREADENTRY32 { dwSize: size_of::<THREADENTRY32>() as u32, ..Default::default() };
        let mut more = Thread32First(snapshot, &mut entry).is_ok();
        while more {
            if entry.th32OwnerProcessID == process
                && let Ok(handle) = OpenThread(THREAD_QUERY_INFORMATION, false, entry.th32ThreadID)
            {
                let (mut created, mut exited, mut kernel, mut user) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
                let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
                let cpu = match GetThreadTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) {
                    Ok(()) => Duration::from_nanos((ticks(kernel) + ticks(user)) * 100),
                    Err(_) => Duration::ZERO,
                };
                let mut start = 0usize;
                if let Some(query) = query {
                    // ThreadQuerySetWin32StartAddress (9): the thread's start routine.
                    type QueryFn = unsafe extern "system" fn(HANDLE, u32, *mut usize, u32, *mut u32) -> i32;
                    let query: QueryFn = std::mem::transmute(query);
                    query(handle, 9, &mut start, size_of::<usize>() as u32, std::ptr::null_mut());
                }
                out.insert(entry.th32ThreadID, (cpu, start));
                let _ = CloseHandle(handle);
            }
            more = Thread32Next(snapshot, &mut entry).is_ok();
        }
        let _ = CloseHandle(snapshot);
    }
    out
}

/// The busiest threads over the span, as a share of one core.
fn log_threads(names: &ModuleNames, before: &HashMap<u32, (Duration, usize)>, after: &HashMap<u32, (Duration, usize)>, took: Duration, render: u32, game: u32) {
    let mut busy: Vec<(u32, f64, usize)> = after
        .iter()
        .filter_map(|(id, (cpu, start))| {
            let (was, _) = before.get(id)?;
            Some((*id, cpu.saturating_sub(*was).as_secs_f64() * 100.0 / took.as_secs_f64(), *start))
        })
        .collect();
    busy.sort_by(|a, b| b.1.total_cmp(&a.1));
    log!("probe_profile: busiest threads (share of one core over {:.1} s)", took.as_secs_f64());
    for (id, share, start) in busy.into_iter().take(12) {
        let role = if id == render { " (render)" } else if id == game { " (game)" } else { "" };
        let start = match names.of(start) {
            Some((m, name)) => match nearest_export(m, start) {
                Some((export, offset)) => format!("{name}!{export}+{offset:#x}"),
                None => format!("{name}+{:#x}", start - m.base()),
            },
            None => format!("{start:#x}"),
        };
        log!("  {share:5.1}%  thread {id}{role} from {start}");
    }
}

/// The render thread's rendering by the single pass's part of the frame, the right eye's rest
/// split at its last draw: where the time from the frame's last draw to its `SwapBuffers` goes.
fn report_parts(exe: &Module, names: &ModuleNames, samples: &[Sample], swaps: &[u64]) {
    let rendering: Vec<&Sample> = samples.iter().filter(|s| s.phase == Phase::Game as u8).collect();
    let total = samples.len().max(1) as f64;
    let share = |count: usize| count as f64 * 100.0 / total;
    let place = |address: usize| match names.of(address) {
        Some((m, name)) if m == exe => format!("{name}+{:#x}", address - m.base()),
        Some((m, name)) => match nearest_export(m, address) {
            Some((export, offset)) => format!("{name}!{export}+{offset:#x}"),
            None => format!("{name}+{:#x}", address - m.base()),
        },
        None => format!("{address:#x}"),
    };
    // After the frame's last draw: no draw made between the sample and the next swap.
    let after_last = |s: &Sample| {
        let next = swaps.partition_point(|&at| at < s.draws);
        swaps.get(next).is_some_and(|&at| at == s.draws)
    };
    let labels = ["before the left pass", "left stretch", "left rest", "right stretch", "right rest, to its last draw"];
    log!("probe_profile: render: the game's rendering by the single pass's part of the frame ({} swaps seen)", swaps.len());
    let mut gap = Vec::new();
    let mut by_part = [0usize; 5];
    for s in &rendering {
        if after_last(s) {
            gap.push(*s);
        } else {
            by_part[usize::from(s.part).min(4)] += 1;
        }
    }
    for (label, count) in labels.iter().zip(by_part) {
        log!("  {:5.1}%  {label}", share(count));
    }
    let waiting = gap.iter().filter(|s| is_wait(&place(s.at))).count();
    log!("  {:5.1}%  after the frame's last draw, to its SwapBuffers ({:.1}% of it in a kernel wait)", share(gap.len()), waiting as f64 * 100.0 / gap.len().max(1) as f64);
    let mut by_module: HashMap<String, usize> = HashMap::new();
    let mut by_place: HashMap<usize, usize> = HashMap::new();
    let mut by_caller: HashMap<(String, usize), usize> = HashMap::new();
    for s in &gap {
        *by_module.entry(names.name(s.at)).or_default() += 1;
        *by_place.entry(s.at & !0x3f).or_default() += 1;
        *by_caller.entry((names.name(s.at), s.caller)).or_default() += 1;
    }
    log!("probe_profile: render: after the frame's last draw, by module");
    for (name, count) in top(by_module, 8) {
        log!("  {:5.1}%  {name}", share(count));
    }
    log!("probe_profile: render: after the frame's last draw, busiest places");
    for (at, count) in top(by_place, 15) {
        log!("  {:5.1}%  {}", share(count), place(at));
    }
    log!("probe_profile: render: after the frame's last draw, by module and the game's return address");
    for ((module, caller), count) in top(by_caller, 15) {
        let caller = if caller == 0 { "(none on the stack)".to_owned() } else { format!("exe+{:#x}", caller - exe.base()) };
        log!("  {:5.1}%  {module} from {caller}", share(count));
    }
}

/// Whether a place is a system call that waits.
fn is_wait(place: &str) -> bool {
    ["NtWaitFor", "NtDelayExecution", "NtYield", "NtRemoveIoCompletion", "WaitForSynchronizationObject", "NtGdiDdDDIWaitFor"].iter().any(|w| place.contains(w))
}

fn top<K>(counts: HashMap<K, usize>, n: usize) -> Vec<(K, usize)> {
    let mut list: Vec<_> = counts.into_iter().collect();
    list.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    list.truncate(n);
    list
}

/// The export at or below `address` in `module` and the distance from it.
fn nearest_export(module: &Module, address: usize) -> Option<(String, usize)> {
    let base = module.base();
    let pe = base + mem::read::<u32>(base + 0x3c)? as usize;
    let directory = mem::read::<u32>(pe + 0x88)? as usize;
    if directory == 0 {
        return None;
    }
    let exports = base + directory;
    let names = mem::read::<u32>(exports + 0x18)? as usize;
    let functions = base + mem::read::<u32>(exports + 0x1c)? as usize;
    let name_table = base + mem::read::<u32>(exports + 0x20)? as usize;
    let ordinals = base + mem::read::<u32>(exports + 0x24)? as usize;
    let rva = address - base;
    let mut best: Option<(usize, usize)> = None;
    for i in 0..names {
        let ordinal = mem::read::<u16>(ordinals + i * 2)? as usize;
        let function = mem::read::<u32>(functions + ordinal * 4)? as usize;
        if function <= rva && best.is_none_or(|(f, _)| function > f) {
            best = Some((function, i));
        }
    }
    let (function, i) = best?;
    let name_at = base + mem::read::<u32>(name_table + i * 4)? as usize;
    let mut bytes = Vec::new();
    while bytes.len() < 128 {
        let b = mem::read::<u8>(name_at + bytes.len())?;
        if b == 0 {
            break;
        }
        bytes.push(b);
    }
    Some((String::from_utf8_lossy(&bytes).into_owned(), rva - function))
}
