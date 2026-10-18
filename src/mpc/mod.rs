//! MPC layer: SPDZ process glue, input/output codecs, and round runners.
//! `runtime` compiles a `.mpc` and spawns the parties, `parse` reads party-0
//! stdout, `rounds` holds the runners `pipeline::mpc` calls.

mod extract_reuse;
mod input_encoding;
mod parse;
pub mod rounds;
mod runtime;

pub use rounds::{
    run_kmeans, run_relabel, run_round1_rsecret, run_round1_secure, run_round2_secure,
    run_sequence_cluster,
};

pub(crate) use extract_reuse::{
    enc_slab_offset, mode_layout_end_bound, ordered_digest, resolve as resolve_extract_reuse,
    write_sidecar, ExtractReuse, SlabKey,
};
pub(crate) use input_encoding::{
    input_payload_digest, install_signal_teardown, slab_shape, SecretsCleanup,
};
#[cfg(test)]
pub(crate) use parse::parse_row_output_backbone;
#[cfg(test)]
pub(crate) use runtime::expand_hash_weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::constants::MPC_TIMEOUT_SECS;

/// Program names match the `.mpc` filenames under `mp-spdz/`.
pub(crate) const MPC_PROGRAM: &str = "secure_extract";
pub(crate) const MPC_PROGRAM_RSECRET: &str = "secure_extract_rsecret";
pub(crate) const KMEANS_PROGRAM: &str = "secure_kmeans";
pub(crate) const ROUND2_PROGRAM: &str = "secure_aggregate";
pub(crate) const RELABEL_PROGRAM: &str = "secure_relabel";
pub(crate) const SEQUENCE_PROGRAM: &str = "secure_sequence_cluster";

#[derive(Debug)]
pub enum SpdzError {
    Io(std::io::Error),
    Runtime(String),
    /// A revealed `Y_π` whose zero mask disagrees with the opened `|A_kpriv|`:
    /// a backbone row projected to the exact zero vector (a balanced profile in
    /// the kernel of the in-box ±1 R). `run_round1_rsecret` answers it with the
    /// protocol's restart rule — one re-run with a fresh in-box R.
    DegenerateReveal {
        n_nonzero: usize,
        n_backbone: usize,
    },
}

impl std::fmt::Display for SpdzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpdzError::Io(e) => write!(f, "IO: {e}"),
            SpdzError::Runtime(s) => write!(f, "Runtime: {s}"),
            SpdzError::DegenerateReveal { n_nonzero, n_backbone } => write!(
                f,
                "Degenerate reveal: {n_nonzero} nonzero revealed rows but MPC reported \
                 |A_kpriv| = {n_backbone} (a backbone row projected to the exact zero vector)"
            ),
        }
    }
}

impl std::error::Error for SpdzError {}
impl From<std::io::Error> for SpdzError {
    fn from(e: std::io::Error) -> Self {
        SpdzError::Io(e)
    }
}
impl From<String> for SpdzError {
    fn from(s: String) -> Self {
        SpdzError::Runtime(s)
    }
}

/// One executed MPC round's cost, drained into the run's `timing.json`.
#[derive(Debug, Clone)]
pub struct RoundTiming {
    pub round: String,
    /// Wall clock around the compile step; when `compile_cached` it is the
    /// dependency-hash check alone, so the two regimes must never be averaged.
    pub compile_s: f64,
    pub compile_cached: bool,
    /// Wall clock around the n co-located party processes (the simulator figure).
    pub exec_s: f64,
    /// Party 0's own `Time = ` line — the per-machine number a deployment would
    /// measure; `exec_s - time_s` is the simulator's co-location overhead.
    pub time_s: Option<f64>,
    pub data_sent_mb: Option<f64>,
    pub global_data_sent_mb: Option<f64>,
    /// Round count, an upper bound: MP-SPDZ double-counts rounds under threading.
    pub rounds: Option<u64>,
}

/// Timing keys of the rsec Round 1: the attempt the run USES is
/// `round1_rsecret`; an attempt the restart rule discarded is renamed, so its
/// cost stays in the file without being read as the round's figure.
pub const ROUND1_RSECRET: &str = "round1_rsecret";
pub const ROUND1_RSECRET_DEGENERATE: &str = "round1_rsecret_degenerate";

/// Restart bookkeeping: the discarded attempt was recorded under the round's
/// name before its output was parsed, so rename it and zero its compile cost —
/// the one compile belongs to the attempt the run uses.
pub(crate) fn discard_last_round(cfg: &SpdzConfig, from: &str, to: &str) {
    if let Ok(mut log) = cfg.timings.lock() {
        if let Some(last) = log.last_mut() {
            if last.round == from {
                last.round = to.to_string();
                last.compile_s = 0.0;
                last.compile_cached = true;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpdzConfig {
    pub spdz_path: PathBuf,
    pub protocol: String,
    /// Extra `compile.py` flags (after `-R <ring>`), e.g. `["-Z", "2"]` for
    /// local share-conversion. Folded into the bytecode dependency hash so a
    /// flag change forces a recompile.
    pub compile_flags: Vec<String>,
    /// Extra RUNTIME flags appended to every `<protocol>-party.x` invocation,
    /// e.g. `["-o","softspoken=5"]`. NOT in the bytecode hash — runtime flags
    /// change no bytecode — but recorded in timing.json; every party needs them.
    pub party_flags: Vec<String>,
    pub timeout: Duration,
    /// Per-round cost log, shared across clones (one entry per executed round).
    pub timings: Arc<Mutex<Vec<RoundTiming>>>,
}

impl Default for SpdzConfig {
    fn default() -> Self {
        let path = std::env::var("MP_SPDZ_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("MP-SPDZ"))
                .unwrap_or_else(|| PathBuf::from("MP-SPDZ"))
        });
        let compile_flags = std::env::var("CLUSTERING_MPC_COMPILE_FLAGS")
            .ok()
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        let party_flags: Vec<String> = std::env::var("CLUSTERING_MPC_PARTY_FLAGS")
            .ok()
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        for flag in research_party_flags(&party_flags) {
            eprintln!(
                "[SPDZ] WARNING: party flag '{}' is a research-only option \
                 (weakens security, breaks the player/prog argv contract, or \
                 silently invalidates the --reuse-extraction slab); never \
                 deploy it",
                flag
            );
        }
        SpdzConfig {
            spdz_path: path,
            // The protocol of every recorded run (dishonest-majority, semi-honest,
            // 2^64 ring); the `-Z N` split benchmark rides the compile flags.
            protocol: "semi2k-party.x".to_string(),
            compile_flags,
            party_flags,
            timeout: Duration::from_secs(MPC_TIMEOUT_SECS),
            timings: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Party flags that must never appear in a deployed configuration: security
/// knobs, `-p` (which would read the player number as the progname), and the
/// prep-dir family, which the `--reuse-extraction` slab identity does not cover.
pub fn research_party_flags(flags: &[String]) -> Vec<String> {
    let named = ["comp_sec", "zero_seed", "use_kos", "insecure"];
    let exact = ["-p", "-P", "--prep-dir", "-F", "--file-prep-per-thread"];
    flags
        .iter()
        .filter(|f| named.iter().any(|n| f.starts_with(n)) || exact.contains(&f.as_str()))
        .cloned()
        .collect()
}

/// Write every executed round's cost to `<dir>/timing.json`. A repeated round
/// name (a repeated program within one run) gets a `_2`, `_3`, … suffix.
pub fn write_timing_json(cfg: &SpdzConfig, dir: &Path) -> std::io::Result<()> {
    let rounds = cfg.timings.lock().map(|g| g.clone()).unwrap_or_default();
    let mut map = serde_json::Map::new();
    for rt in rounds {
        let mut key = rt.round.clone();
        let mut n = 1usize;
        while map.contains_key(&key) {
            n += 1;
            key = format!("{}_{}", rt.round, n);
        }
        map.insert(
            key,
            serde_json::json!({
                "compile_s": rt.compile_s,
                "compile_cached": rt.compile_cached,
                "exec_s": rt.exec_s,
                "time_s": rt.time_s,
                "data_sent_mb": rt.data_sent_mb,
                "global_data_sent_mb": rt.global_data_sent_mb,
                "rounds": rt.rounds,
            }),
        );
    }
    // A cell measured under non-default runtime flags is not comparable to one
    // without them, and timing numbers compare only within one generation.
    if !cfg.party_flags.is_empty() {
        map.insert("_party_flags".to_string(), serde_json::json!(cfg.party_flags));
    }
    map.insert("_generation".to_string(), serde_json::json!(crate::constants::TIMING_GENERATION));
    // The rsec Round-1 restart on a degenerate reveal is a public event (every
    // party observes the re-run); always emitted, so 0 is a positive statement.
    let restarts = map.keys().filter(|k| k.starts_with(ROUND1_RSECRET_DEGENERATE)).count();
    map.insert("_round1_restarts".to_string(), serde_json::json!(restarts));
    let s = serde_json::to_string_pretty(&serde_json::Value::Object(map))
        .map_err(std::io::Error::other)?;
    std::fs::write(dir.join("timing.json"), s)
}

#[derive(Debug, Clone)]
pub struct CaseData {
    pub id: String,
    pub events: Vec<(u64, u32)>,
}
