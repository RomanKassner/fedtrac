//! Pipeline-wide tunables.

// -- Clustering --
pub const MAX_ITERATIONS: usize = 10;
pub const N_INIT: usize = 10;
/// Thesis edge filter: keep an activity edge iff its weight is at least
/// `max(MIN_EDGE_COUNT, MIN_EDGE_FRACTION * max activity weight)`.
pub const MIN_EDGE_COUNT: f64 = 2.0;
pub const MIN_EDGE_FRACTION: f64 = 0.02;
/// Merge floor: a realized cluster below this is re-homed into its nearest larger one.
pub const MIN_CLUSTER_SIZE: usize = 10;
pub const K_SELECT_K_MIN: usize = 3;

// -- Random projection --
pub const DEFAULT_PROJECTION_SEED: u64 = 42;

// -- Adaptive k_proj (compressed sensing bound) --
/// k_proj >= CS_COEFF * s * ln(d^2 / s); see compute_adaptive_kproj.
pub const CS_COEFF: f64 = 4.0;
pub const CS_FULL_SPARSITY_MULTIPLIER: f64 = 2.0;
pub const ADAPTIVE_KPROJ_SAFETY_FACTOR: f64 = 0.85;
pub const ADAPTIVE_KPROJ_MIN: usize = 20;

// -- Trace length selection --
/// Target fraction of total bigrams that must survive per-party truncation.
pub const BIGRAM_COVERAGE_TARGET: f64 = 0.97;
pub const TRACE_LEN_ALIGNMENT: usize = 4;
pub const TRACE_LEN_MIN: usize = 8;
pub const TRACE_LEN_FALLBACK_PER_PARTY: usize = 8;
pub const BIGRAM_COVERAGE_SCAN_START: usize = 4;

// -- Reveal-secret-projection suppression guard --
/// Fingerprint k-anonymity floor k0 and singleton budget tau.
pub const VARIANT_K_ANONYMITY: usize = 5;
pub const MAX_SINGLETON_FRACTION: f64 = 0.20;

// -- MPC / SPDZ --
/// 12 h: the guard catches hangs, not slowness (a 4-party org-split rsec Round 1
/// legitimately runs past 4 h).
pub const MPC_TIMEOUT_SECS: u64 = 43_200;
pub const MPC_POLL_INTERVAL_MS: u64 = 50;
/// Bound on the conformance-eval subprocess: token replay can hang on a
/// discovered model, and two hours passes the largest legitimate evaluation.
pub const EVAL_TIMEOUT_SECS: u64 = 7_200;
pub const MPC_RING_SIZE: u32 = 64;
pub const MPC_N_THREADS: u32 = 8;
/// Stamped into every timing.json as `_generation`: timing numbers are
/// comparable only within one generation.
pub const TIMING_GENERATION: u32 = 4;

/// `MPC_N_THREADS` with a `CLUSTERING_MPC_THREADS` override. MP-SPDZ counts
/// rounds DOUBLE under threading, so an exact round figure needs a 1-thread
/// probe; results themselves are thread-count-independent.
pub fn mpc_n_threads() -> u32 {
    std::env::var("CLUSTERING_MPC_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(MPC_N_THREADS)
}
/// Fixed-iteration Lloyd's: the iteration count is public, so it must not
/// depend on the data.
pub const SECURE_KMEANS_MAX_ITER: usize = 5;
/// Fixed-iteration Markov-chain hard-EM; the count is public likewise.
pub const SECURE_SEQUENCE_MAX_ITER: usize = 5;

// -- I/O defaults --
pub const DEFAULT_DATA_DIR: &str = "data/splits";
pub const DEFAULT_N_PARTIES: usize = 2;

// -- Encoding --
pub const MS_TO_SECS: u64 = 1000;
/// Public epoch granularity for the federation-wide `min_ts` handshake scalar.
/// A multiple of `MS_TO_SECS`, so flooring shifts every scaled timestamp alike.
pub const MIN_TS_EPOCH_MS: u64 = 86_400_000;
/// Public width bound for the in-MPC edge-filter comparisons: every aggregated
/// DFG cell is a pure count below `2^AGG_CELL_BITS`. Twin of
/// mpc_utils/constants.py::AGG_CELL_BITS.
pub const AGG_CELL_BITS: u32 = 41;
/// Fixed-point scale of the in-MPC L2-row-normalisation reveal: cells arrive as
/// `unit_value * NORM_SCALE`. Twin of mpc_utils/constants.py::NORM_SCALE.
pub const NORM_SCALE: f64 = 131_072.0; // 2^17 = 2^(16 + 2/2)
/// `NORM_SCALE` generalised to the selected `NORM_PRESHIFT`: revealed cells are
/// `unit_value * 2^(16 + p/2)`, 16 being the sfix fractional width.
pub fn norm_scale_for_preshift(preshift: u32) -> f64 {
    debug_assert!(preshift >= 2 && preshift.is_multiple_of(2));
    (1u64 << (16 + preshift / 2)) as f64
}
