//! Compile MPC programs and spawn party processes.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::constants::*;

use super::{SpdzConfig, SpdzError};

fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Hash file name + content of every path that exists, in the caller's order.
/// Absent paths contribute nothing, so a file appearing or disappearing moves
/// the hash.
pub(crate) fn dependency_hash(paths: &[PathBuf]) -> u64 {
    let mut h = DefaultHasher::new();
    for p in paths {
        if let Ok(bytes) = fs::read(p) {
            p.file_name().hash(&mut h);
            bytes.hash(&mut h);
        }
    }
    h.finish()
}

/// What a `compile_program` call cost; `cached` says the bytecode cache hit, so
/// `elapsed` is then the file copies plus the hash check and not a compile.
#[derive(Debug, Clone)]
pub(crate) struct CompileOutcome {
    /// `<prog>-<args...>`: the compiled tape's name, passed to every party.
    pub prog: String,
    pub elapsed: Duration,
    pub cached: bool,
}

/// Copy source + mpc_utils into the MP-SPDZ install and compile.
pub(crate) fn compile_program(
    cfg: &SpdzConfig,
    program: &str,
    args: &[String],
) -> Result<CompileOutcome, SpdzError> {
    let t_compile = Instant::now();
    let mpc_dir = std::env::current_dir()?.join("mp-spdz");
    let src = mpc_dir.join(format!("{}.mpc", program));
    let programs_source = cfg.spdz_path.join("Programs/Source");
    let dst = programs_source.join(format!("{}.mpc", program));

    if src.exists() {
        // Always copy: staleness is decided by the content hash below, so the
        // compiled source must mirror the repo regardless of mtimes.
        fs::copy(&src, &dst)?;
    }

    let utils_src = mpc_dir.join("mpc_utils");
    let utils_dst = programs_source.join("mpc_utils");
    if utils_src.is_dir() {
        copy_dir_all(&utils_src, &utils_dst)?;
    }

    let prog_name = if args.is_empty() {
        program.to_string()
    } else {
        format!("{}-{}", program, args.join("-"))
    };

    // Compiled bytecode bakes in more than the .mpc source: every mpc_utils/*.py
    // and the data files read at COMPILE time. Destination mtimes carry no
    // staleness information, so recompile iff the dependency CONTENT differs.
    let mut dep_paths = vec![src.clone()];
    if utils_src.is_dir() {
        let mut utils: Vec<PathBuf> = fs::read_dir(&utils_src)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "py"))
            .collect();
        utils.sort();
        dep_paths.extend(utils);
    }
    dep_paths.push(programs_source.join("projection_matrix.txt"));
    let dep_hash = format!("{:016x}{:016x}", dependency_hash(&dep_paths), compile_flag_hash(cfg));

    let bytecode_dir = cfg.spdz_path.join("Programs/Bytecode");
    let bc = bytecode_dir.join(format!("{}-0.bc", prog_name));
    let sidecar = bytecode_dir.join(format!("{}.dephash", prog_name));
    let cache_hit =
        bc.exists() && fs::read_to_string(&sidecar).map(|s| s.trim() == dep_hash).unwrap_or(false);

    if !cache_hit {
        eprintln!("[SPDZ] Compiling {} with args {:?}...", program, args);
        // A failed compile can leave partial bytecode next to a stale
        // sidecar; drop the sidecar first so only a completed compile caches.
        let _ = fs::remove_file(&sidecar);
        let mut compile_args =
            vec!["./compile.py".to_string(), "-R".to_string(), MPC_RING_SIZE.to_string()];
        compile_args.extend_from_slice(&cfg.compile_flags);
        compile_args.push(program.to_string());
        compile_args.extend_from_slice(args);

        let out =
            Command::new("python3").current_dir(&cfg.spdz_path).args(&compile_args).output()?;

        if !out.status.success() {
            return Err(SpdzError::Runtime(format!(
                "compile failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        // The compiler prints hazards on stdout and still exits zero, so every
        // warning line is forwarded. "Order of memory instructions not preserved"
        // fires for ANY long vectorised access, so it cannot be a hard error.
        for line in String::from_utf8_lossy(&out.stdout)
            .lines()
            .chain(String::from_utf8_lossy(&out.stderr).lines())
        {
            if line.contains("WARNING") || line.contains("not preserved") {
                eprintln!("[SPDZ compile] {}", line);
            }
        }
        fs::write(&sidecar, &dep_hash)?;
        eprintln!("[SPDZ] Compiled {}", prog_name);
    }

    let compile_time = t_compile.elapsed();
    eprintln!(
        "[MPC] compiled {} in {:.1}s{}",
        program,
        compile_time.as_secs_f64(),
        if cache_hit { " (bytecode cache hit)" } else { "" }
    );
    Ok(CompileOutcome { prog: prog_name, elapsed: compile_time, cached: cache_hit })
}

/// Write newline-separated integers to `Programs/Public-Input/<prog_name>`, read
/// at RUNTIME by the MAIN tape (threads read a `-<thread>`-suffixed file). Every
/// party parses the same file — the consistency assumption of identical args.
pub(crate) fn write_public_input(
    cfg: &SpdzConfig,
    prog_name: &str,
    values: &[u64],
) -> Result<(), SpdzError> {
    let dir = cfg.spdz_path.join("Programs/Public-Input");
    fs::create_dir_all(&dir)?;
    let mut out = String::with_capacity(values.len() * 16);
    for v in values {
        out.push_str(&v.to_string());
        out.push('\n');
    }
    fs::write(dir.join(prog_name), out)?;
    Ok(())
}

/// Twin of `random.Random(seed).randrange(1, 1 << 48)` repeated `n` times: the
/// weight sequence `secure_extract_rsecret.mpc` consumes, so the printed
/// per-run seed alone reproduces the fingerprint weights.
pub(crate) fn expand_hash_weights(seed: u64, n: usize) -> Vec<u64> {
    let mut rng = PyMt19937::new(seed);
    // randrange(1, 2^48) = 1 + _randbelow(2^48 - 1): draw 48 bits, reject >= 2^48 - 1.
    let width = (1u64 << 48) - 1;
    (0..n)
        .map(|_| loop {
            let r = rng.getrandbits48();
            if r < width {
                return 1 + r;
            }
        })
        .collect()
}

/// CPython's MT19937 as seeded by `random.Random(int)`: the |seed| is split
/// into little-endian 32-bit words and fed to `init_by_array`.
struct PyMt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl PyMt19937 {
    fn new(seed: u64) -> Self {
        let mut s = Self { mt: [0u32; 624], mti: 624 };
        s.init_genrand(19650218);
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        s.init_by_array(&key);
        s
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn genrand_u32(&mut self) -> u32 {
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        if self.mti >= 624 {
            for kk in 0..624 {
                let y = (self.mt[kk] & UPPER) | (self.mt[(kk + 1) % 624] & LOWER);
                self.mt[kk] =
                    self.mt[(kk + M) % 624] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// CPython `getrandbits(48)`: low 32-bit word first, then 16 top bits.
    fn getrandbits48(&mut self) -> u64 {
        let lo = self.genrand_u32() as u64;
        let hi = (self.genrand_u32() >> 16) as u64;
        lo | (hi << 32)
    }
}

/// End-of-run statistics from a party process: the wall clock plus the `Time =`
/// / `Data sent =` / `Global data sent =` lines (`None` when a line was absent).
#[derive(Debug, Default, Clone)]
pub(crate) struct MpcRunStats {
    pub exec: Duration,
    pub time_s: Option<f64>,
    pub data_sent_mb: Option<f64>,
    pub global_data_sent_mb: Option<f64>,
    /// Scraped from the `Data sent = X MB in ~N rounds` line; MP-SPDZ counts
    /// rounds double under threading, so this is a ~2x upper bound.
    pub rounds: Option<u64>,
}

fn stat_after_eq(line: &str) -> Option<f64> {
    line.split('=').nth(1)?.split_whitespace().next()?.parse().ok()
}

/// `Data sent = 12.3 MB in ~4567 rounds (party 0 only; ...)` -> `4567`.
fn rounds_in_line(line: &str) -> Option<u64> {
    line.split("in ~").nth(1)?.split_whitespace().next()?.parse().ok()
}

impl MpcRunStats {
    /// Scrape the end-of-run block from party 0's stderr, then let stdout fill
    /// whatever is still missing, per field: a build that splits the block
    /// across the two streams must not leave fields null.
    pub(crate) fn scrape_with_fallback(&mut self, stderr: &str, stdout: &str) {
        self.scrape(stderr);
        if self.time_s.is_none()
            || self.data_sent_mb.is_none()
            || self.global_data_sent_mb.is_none()
            || self.rounds.is_none()
        {
            self.scrape(stdout);
        }
    }

    fn scrape(&mut self, text: &str) {
        for line in text.lines() {
            let l = line.trim_start();
            if l.starts_with("Time = ") {
                self.time_s = self.time_s.or_else(|| stat_after_eq(l));
            } else if l.starts_with("Data sent = ") {
                self.data_sent_mb = self.data_sent_mb.or_else(|| stat_after_eq(l));
                self.rounds = self.rounds.or_else(|| rounds_in_line(l));
            } else if l.starts_with("Global data sent = ") {
                self.global_data_sent_mb = self.global_data_sent_mb.or_else(|| stat_after_eq(l));
            }
        }
    }
}

/// True iff the MP-SPDZ party binary accepts `-N` (a variable party count).
/// Fixed-party binaries omit it from their usage and reject it, so the usage
/// text is the authority.
fn binary_accepts_nparties(bin: &Path) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&v) = cache.lock().unwrap().get(bin) {
        return v;
    }
    // No progname => the binary prints its usage and exits without networking.
    let usage = Command::new(bin).output().ok().map_or(String::new(), |o| {
        let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
        s.push_str(&String::from_utf8_lossy(&o.stderr));
        s
    });
    let accepts = usage.contains("-N,");
    cache.lock().unwrap().insert(bin.to_path_buf(), accepts);
    accepts
}

/// Extra compile flags and the ring width `-R <MPC_RING_SIZE>` change the
/// emitted bytecode but are not in prog_name, so both fold into the dephash.
/// Runtime party flags are excluded: they change no bytecode.
pub(crate) fn compile_flag_hash(cfg: &SpdzConfig) -> u64 {
    compile_flag_hash_for(&cfg.compile_flags, MPC_RING_SIZE)
}

fn compile_flag_hash_for(compile_flags: &[String], ring_size: u32) -> u64 {
    let mut flag_hasher = DefaultHasher::new();
    compile_flags.hash(&mut flag_hasher);
    ring_size.hash(&mut flag_hasher);
    flag_hasher.finish()
}

/// Party argv: the player number as a bare leading positional (the fixed-party
/// ring binaries read the first positional as the playerno, so `-p <n>` would
/// make them read the progname), then the program, `-N`, then runtime flags.
pub(crate) fn party_argv(
    party: u8,
    prog: &str,
    takes_nparties: bool,
    n_parties: u8,
    party_flags: &[String],
) -> Vec<String> {
    let mut args = vec![party.to_string(), prog.to_string()];
    if takes_nparties {
        args.push("-N".to_string());
        args.push(n_parties.to_string());
    }
    // `no_memory_output` suppresses MP-SPDZ's end-of-run dump of the tape's
    // global sint memory (raw shares of the matrices and Y). It joins the
    // user's `-o` group rather than adding a second `-o`, which `OnlineOptions`
    // would drop; a wrong `-o` value is ignored, so `SecretsCleanup` is the net.
    let mut flags = party_flags.to_vec();
    match flags.iter().position(|f| f == "-o" || f == "--options") {
        Some(i) if i + 1 < flags.len() => {
            flags[i + 1] = format!("{},{}", flags[i + 1], NO_MEMORY_OUTPUT_OPTION);
        }
        _ => {
            flags.push("-o".to_string());
            flags.push(NO_MEMORY_OUTPUT_OPTION.to_string());
        }
    }
    args.extend(flags);
    args
}

/// MP-SPDZ `-o` value that suppresses the end-of-run memory dump.
pub(crate) const NO_MEMORY_OUTPUT_OPTION: &str = "no_memory_output";

/// PIDs of the party processes this process supervises, so a failing party's
/// siblings and the signal teardown can kill them.
#[derive(Default)]
pub(crate) struct ChildRegistry {
    pids: Mutex<Vec<(usize, u32)>>,
    /// Set the moment a round is abandoned: a party failing before a sibling is
    /// even spawned would otherwise leave it running to its full deadline, since
    /// the kill cannot see a pid that does not exist yet.
    aborting: std::sync::atomic::AtomicBool,
}

impl ChildRegistry {
    fn aborting(&self) -> bool {
        self.aborting.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn reset(&self) {
        self.aborting.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Register a live child, or report that the round is already lost (the
    /// caller kills it immediately rather than supervising it to its deadline).
    fn register(&self, party: usize, pid: u32) -> bool {
        self.pids.lock().unwrap_or_else(|e| e.into_inner()).push((party, pid));
        !self.aborting()
    }

    fn unregister(&self, party: usize) {
        self.pids.lock().unwrap_or_else(|e| e.into_inner()).retain(|(p, _)| *p != party);
    }

    /// SIGKILL every registered child except `except` (the one that exited).
    /// SIGKILL rather than SIGTERM: a party blocked in a socket read on a peer
    /// that will never answer has no state worth flushing. Returns how many.
    fn kill_others(&self, except: Option<usize>) -> usize {
        self.aborting.store(true, std::sync::atomic::Ordering::SeqCst);
        let pids: Vec<(usize, u32)> = self
            .pids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(p, _)| Some(*p) != except)
            .copied()
            .collect();
        for (_, pid) in &pids {
            // SAFETY: kill(2) on a pid this process spawned and has not yet
            // reaped, so the pid cannot have been recycled.
            unsafe { libc::kill(*pid as libc::pid_t, libc::SIGKILL) };
        }
        pids.len()
    }
}

fn global_children() -> &'static ChildRegistry {
    static CHILDREN: OnceLock<ChildRegistry> = OnceLock::new();
    CHILDREN.get_or_init(ChildRegistry::default)
}

/// Kill every party process this run spawned — the signal handler's first act,
/// before it wipes the secret material out from under them.
pub(crate) fn kill_party_children() {
    let n = global_children().kill_others(None);
    if n > 0 {
        eprintln!("[SPDZ] Killed {} running party process(es)", n);
    }
}

/// How one party process ended. `Failed` covers a non-zero exit and death by
/// signal (an OOM kill reports no code), `Timeout` only the orchestrator's own
/// deadline, `ReaderStalled` a clean exit whose output was incomplete.
#[derive(Debug)]
pub(crate) enum PartyOutcome {
    Ok { stdout: String, stderr: String },
    Failed { how: String, stderr: String },
    Timeout,
    ReaderStalled { pipe: &'static str },
}

/// The failure message for one party's outcome, or `None` when it exited 0.
pub(crate) fn party_failure(party: usize, outcome: &PartyOutcome) -> Option<String> {
    match outcome {
        PartyOutcome::Ok { .. } => None,
        PartyOutcome::Failed { how, stderr } => {
            // The tail is what says WHY; the head is protocol setup chatter.
            let tail: Vec<&str> = stderr.lines().rev().take(8).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            Some(format!("party {} {}: {}", party, how, tail.join("\n")))
        }
        PartyOutcome::Timeout => Some(format!("party {} timeout", party)),
        PartyOutcome::ReaderStalled { pipe } => Some(format!(
            "party {} exited 0 but its {} reader did not reach EOF within the reader-drain \
             bound ({} s in a pipeline run; a process still holds the write end). The output \
             would be incomplete, so the round fails here rather than parsing a truncated \
             payload",
            party,
            pipe,
            READER_DRAIN_BOUND.as_secs()
        )),
    }
}

/// Read a child pipe to EOF on its own thread, handing the text back over a
/// channel rather than a `JoinHandle`: a killed shell can leave a grandchild
/// holding the write end, and an unbounded join would hang the round forever.
fn drain_pipe(pipe: impl std::io::Read + Send + 'static) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let text =
            BufReader::new(pipe).lines().map_while(Result::ok).collect::<Vec<_>>().join("\n");
        let _ = tx.send(text);
    });
    rx
}

/// The output a drained pipe produced, or `None` if the reader did not reach
/// EOF within `bound` of the child's exit. `None` is not substitutable by an
/// empty string: party 0's stdout is the round's parsed payload.
fn collect_pipe(rx: mpsc::Receiver<String>, bound: Duration) -> Option<String> {
    rx.recv_timeout(bound).ok()
}

/// How long to wait for a reader thread after the child is gone: the only
/// legitimate cost is draining what is still in flight, so anything past it is
/// a wedged grandchild, a failure rather than a truncation.
const READER_DRAIN_BOUND: Duration = Duration::from_secs(600);

fn spawn_party(cfg: &SpdzConfig, party: u8, n_parties: u8, prog: &str) -> std::io::Result<Child> {
    let bin = cfg.spdz_path.join(&cfg.protocol);
    // Fixed-party honest-majority binaries hardcode their party count and
    // reject -N; the variable-party dishonest-majority ones accept it.
    let takes_nparties = binary_accepts_nparties(&bin);
    let args = party_argv(party, prog, takes_nparties, n_parties, &cfg.party_flags);
    Command::new(&bin)
        .current_dir(&cfg.spdz_path)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// Drain a spawned child's pipes, wait for it (or the deadline), and classify
/// the outcome. Registers the pid for the lifetime of the wait so a sibling's
/// failure — or a signal — can kill it.
fn supervise_party(
    mut child: Child,
    party: usize,
    timeout: Duration,
    drain_bound: Duration,
    registry: &ChildRegistry,
) -> PartyOutcome {
    let live_round = registry.register(party, child.id());

    // Drain stdout/stderr in threads; otherwise the child blocks once its
    // output exceeds the OS pipe buffer (~64 KB).
    let stdout_rx = drain_pipe(child.stdout.take().unwrap());
    let stderr_rx = drain_pipe(child.stderr.take().unwrap());

    if !live_round {
        // A sibling already failed: this child was spawned into a lost round.
        let _ = child.kill();
    }
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) if registry.aborting() => {
                let _ = child.kill();
                break child.wait().ok();
            }
            Ok(None) if start.elapsed() >= timeout => {
                timed_out = true;
                let _ = child.kill();
                // Reap, so the readers see EOF and the pid is not recycled
                // under the registry entry we are about to remove.
                break child.wait().ok();
            }
            Ok(None) => thread::sleep(Duration::from_millis(MPC_POLL_INTERVAL_MS)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                registry.unregister(party);
                let stderr = collect_pipe(stderr_rx, drain_bound).unwrap_or_default();
                let _ = collect_pipe(stdout_rx, drain_bound);
                return PartyOutcome::Failed { how: format!("wait failed ({})", e), stderr };
            }
        }
    };
    registry.unregister(party);
    // Collected on EVERY path (a timeout branch returning early leaks both
    // reader threads). The child is gone, so each reader is at most
    // `drain_bound` from EOF unless something else holds the write end.
    let stdout = collect_pipe(stdout_rx, drain_bound);
    let stderr = collect_pipe(stderr_rx, drain_bound);

    if timed_out {
        return PartyOutcome::Timeout;
    }
    match status.and_then(|st| st.code()) {
        // A clean exit is only Ok when BOTH pipes reached EOF: party 0's stdout
        // is the round's payload and its stderr the scraped statistics, so a
        // stalled reader fails the round instead of substituting empty text.
        Some(0) => match (stdout, stderr) {
            (Some(stdout), Some(stderr)) => PartyOutcome::Ok { stdout, stderr },
            (None, _) => PartyOutcome::ReaderStalled { pipe: "stdout" },
            (_, None) => PartyOutcome::ReaderStalled { pipe: "stderr" },
        },
        // On a failure the stderr tail is diagnostic only: report the failure
        // itself, with whatever text arrived.
        Some(code) => PartyOutcome::Failed {
            how: format!("exited {}", code),
            stderr: stderr.unwrap_or_default(),
        },
        // No exit code = killed by a signal (an OOM kill, or our own sibling
        // kill), which is a failure but explicitly NOT a timeout.
        None => PartyOutcome::Failed {
            how: "killed by signal".to_string(),
            stderr: stderr.unwrap_or_default(),
        },
    }
}

/// Run `n` parties concurrently and FAIL FAST: the first party whose outcome is
/// not a clean exit kills its siblings and aborts the round, instead of every
/// survivor blocking in its socket read until its own 12 h deadline.
fn run_parties<F>(
    n_parties: usize,
    registry: &ChildRegistry,
    run: F,
) -> Result<Vec<PartyOutcome>, SpdzError>
where
    F: Fn(usize) -> PartyOutcome + Sync,
{
    let mut arrived: Vec<(usize, PartyOutcome)> = Vec::with_capacity(n_parties);
    registry.reset();
    let failure = thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        for p in 0..n_parties {
            let tx = tx.clone();
            let run = &run;
            scope.spawn(move || {
                let _ = tx.send((p, run(p)));
            });
        }
        drop(tx);
        for _ in 0..n_parties {
            match rx.recv() {
                Ok((p, outcome)) => {
                    let failed = party_failure(p, &outcome);
                    arrived.push((p, outcome));
                    if let Some(msg) = failed {
                        // Killing the siblings here is also what lets this
                        // scope join promptly instead of waiting them out.
                        let killed = registry.kill_others(Some(p));
                        eprintln!(
                            "[SPDZ] {} — killed {} sibling party process(es)",
                            msg.lines().next().unwrap_or(&msg),
                            killed
                        );
                        return Some(msg);
                    }
                }
                // Every sender dropped without n messages: a supervising
                // thread panicked. Its panic is re-raised when the scope ends.
                Err(_) => return Some("a party supervisor thread died".to_string()),
            }
        }
        None
    });
    match failure {
        Some(msg) => Err(SpdzError::Runtime(msg)),
        None => {
            // Restore party order (arrival order is a race).
            arrived.sort_by_key(|(p, _)| *p);
            Ok(arrived.into_iter().map(|(_, o)| o).collect())
        }
    }
}

/// Spawn n parties in parallel (simulator); return party 0's stdout plus the
/// end-of-run statistics scraped from its stderr (stdout fills any field the
/// stderr block lacked). A real deployment runs one Player.x per machine.
pub(crate) fn run_n_party(
    cfg: &SpdzConfig,
    n_parties: usize,
    prog_name: &str,
) -> Result<(String, MpcRunStats), SpdzError> {
    eprintln!("[SPDZ] Running {} parties...", n_parties);
    let t_exec = Instant::now();
    let registry = global_children();

    let outcomes = run_parties(n_parties, registry, |p| {
        match spawn_party(cfg, p as u8, n_parties as u8, prog_name) {
            Ok(child) => supervise_party(child, p, cfg.timeout, READER_DRAIN_BOUND, registry),
            Err(e) => {
                PartyOutcome::Failed { how: format!("spawn failed ({})", e), stderr: String::new() }
            }
        }
    })?;

    let mut stats = MpcRunStats { exec: t_exec.elapsed(), ..Default::default() };
    let (stdout, stderr) = match outcomes.into_iter().next() {
        Some(PartyOutcome::Ok { stdout, stderr }) => (stdout, stderr),
        // Unreachable: run_parties returns Err unless every outcome is Ok.
        _ => return Err(SpdzError::Runtime("party 0 produced no output".to_string())),
    };
    stats.scrape_with_fallback(&stderr, &stdout);
    Ok((stdout, stats))
}

/// Forward stdout lines matching any of `tags` as `[SPDZ-{label}]` to stderr.
pub(crate) fn forward_progress(stdout: &str, label: &str, tags: &[&str]) {
    for line in stdout.lines() {
        if tags.iter().any(|t| line.starts_with(t) || line.contains(t)) {
            eprintln!("[SPDZ-{}] {}", label, line);
        }
    }
}
