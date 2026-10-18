//! Round runners called by `pipeline::mpc`: compile `.mpc`, spawn parties,
//! forward progress lines, parse structured output (rows / LABEL / ASSIGN).

use std::fs;

use crate::constants::*;

use super::input_encoding::{partial_lens_arg, slab_shape, write_inputs_nparty};
use super::parse::{parse_row_output, parse_row_output_backbone};
use super::runtime::{compile_program, forward_progress, run_n_party, CompileOutcome, MpcRunStats};
use super::{
    CaseData, ExtractReuse, RoundTiming, SpdzConfig, SpdzError, KMEANS_PROGRAM, MPC_PROGRAM,
    MPC_PROGRAM_RSECRET, RELABEL_PROGRAM, ROUND2_PROGRAM, SEQUENCE_PROGRAM,
};

/// Append the extract-once reuse compile args (the PARTIAL_LENS slot plus
/// `CORE_MODE PERSIST_ENC ENC_OFFSET BIGRAM_OFFSET`) or, for a non-reuse run,
/// only the optional PARTIAL_LENS token, so its bytecode is unchanged.
fn push_reuse_args(args: &mut Vec<String>, partial_lens: &[usize], reuse: ExtractReuse) {
    match reuse.compile_tuple() {
        Some((core_mode, persist_enc, enc_off, big_off)) => {
            // PARTIAL_LENS must occupy its fixed slot so the trailing reuse args
            // land at stable positions; "0" is the uniform-shape sentinel.
            args.push(partial_lens_arg(partial_lens).unwrap_or_else(|| "0".to_string()));
            args.push(core_mode.to_string());
            args.push(persist_enc.to_string());
            args.push(enc_off.to_string());
            args.push(big_off.to_string());
        }
        None => {
            if let Some(plens) = partial_lens_arg(partial_lens) {
                args.push(plens);
            }
        }
    }
}

/// MP-SPDZ requires Input-P*-0 to exist even when never read.
fn ensure_empty_party_inputs(cfg: &SpdzConfig, n_parties: usize) -> Result<(), SpdzError> {
    let dir = cfg.spdz_path.join("Player-Data");
    fs::create_dir_all(&dir)?;
    for p in 0..n_parties {
        let path = dir.join(format!("Input-P{}-0", p));
        if !path.exists() {
            std::fs::File::create(&path)?;
        }
    }
    Ok(())
}

/// Record one executed round on the shared timing log (drained into the
/// run's timing.json by `write_timing_json`).
fn record_round(cfg: &SpdzConfig, round: &str, compile: &CompileOutcome, stats: &MpcRunStats) {
    if let Ok(mut log) = cfg.timings.lock() {
        log.push(RoundTiming {
            round: round.to_string(),
            compile_s: compile.elapsed.as_secs_f64(),
            compile_cached: compile.cached,
            exec_s: stats.exec.as_secs_f64(),
            time_s: stats.time_s,
            data_sent_mb: stats.data_sent_mb,
            global_data_sent_mb: stats.global_data_sent_mb,
            rounds: stats.rounds,
        });
    }
}

// Round 1: secret-Y (secure_kmeans) and secret-R + shuffle (rsecret) variants.

#[derive(Debug)]
pub struct Round1SecureOutput {
    pub num_cases: usize,
    /// Persistence offset where secure_kmeans reads Y; equals `n_cases * d²`.
    pub y_offset: usize,
}

/// secure_extract with SECURE_Y=1: project and persist Y, no reveal. `reuse`
/// selects the extract-once path (`TailOnly` skips the input read and merge).
/// `y_offset` is `n_cases * d²` either way: the encoding slab rides above the
/// mode slabs, so it never moves.
#[allow(clippy::too_many_arguments)]
pub fn run_round1_secure(
    cfg: &SpdzConfig,
    all_cases: &[Vec<CaseData>],
    dict_size: usize,
    k_proj: usize,
    r_matrix_path: &std::path::Path,
    encoding: usize,
    enc_dim: usize,
    reuse: ExtractReuse,
) -> Result<Round1SecureOutput, SpdzError> {
    let n_parties = all_cases.len();
    if all_cases.iter().map(|c| c.len()).max().unwrap_or(0) == 0 {
        return Ok(Round1SecureOutput { num_cases: 0, y_offset: 0 });
    }

    // A tail run must NOT re-write the raw party streams; it reads the slab.
    let (n_cases, partial_lens) = if reuse.is_tail_only() {
        ensure_empty_party_inputs(cfg, n_parties)?;
        slab_shape(all_cases)?
    } else {
        write_inputs_nparty(cfg, all_cases, dict_size)?
    };
    let partial_len = *partial_lens.iter().max().unwrap_or(&1);
    fs::copy(r_matrix_path, cfg.spdz_path.join("Programs/Source/projection_matrix.txt"))?;

    let mut args = vec![
        n_cases.to_string(),
        partial_len.to_string(),
        dict_size.to_string(),
        mpc_n_threads().to_string(),
        k_proj.to_string(),
        n_parties.to_string(),
        encoding.to_string(),
        "1".to_string(), // SECURE_Y
        enc_dim.to_string(),
    ];
    push_reuse_args(&mut args, &partial_lens, reuse);
    let compiled = compile_program(cfg, MPC_PROGRAM, &args)?;
    let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
    record_round(cfg, "round1_secure", &compiled, &stats);
    forward_progress(&stdout, "NP", &["===", "MODE:", "Level", "Saved", "tail_only"]);

    Ok(Round1SecureOutput { num_cases: n_cases, y_offset: n_cases * dict_size * dict_size })
}

#[derive(Debug)]
pub struct Round1RsecretOutput {
    pub projected_shuffled: Vec<Vec<u64>>,
    pub num_cases: usize,
    pub feature_dim: usize,
    /// Offset of the secret π in persistence; `run_relabel` reads from here.
    pub pi_offset: usize,
    /// Backbone flag per shuffled row: `true` = revealed (in A_kpriv),
    /// `false` = suppressed tail (revealed as zeros).
    pub backbone: Vec<bool>,
    /// The budget-bounded threshold the MPC selected inside the protocol.
    pub k_priv_selected: usize,
}

/// secure_extract_rsecret: R sampled inside MPC, rows shuffled by secret π,
/// only the anonymous (backbone) rows of the shuffled Y revealed. The MPC
/// raises `k0` to the largest k_priv whose suppressed share stays within `tau`.
/// `hash_seed` derives the fingerprint weights, shipped as runtime public input.
#[allow(clippy::too_many_arguments)]
pub fn run_round1_rsecret(
    cfg: &SpdzConfig,
    all_cases: &[Vec<CaseData>],
    dict_size: usize,
    k_proj: usize,
    encoding: usize,
    enc_dim: usize,
    k0: usize,
    hash_seed: u64,
    tau: f64,
    norm_preshift: u32,
    reuse: ExtractReuse,
) -> Result<Round1RsecretOutput, SpdzError> {
    let n_parties = all_cases.len();
    if all_cases.iter().map(|c| c.len()).max().unwrap_or(0) == 0 {
        return Ok(Round1RsecretOutput {
            projected_shuffled: vec![],
            num_cases: 0,
            feature_dim: k_proj,
            pi_offset: 0,
            backbone: vec![],
            k_priv_selected: k0,
        });
    }

    // A tail run reads the slab; it must NOT re-write the raw party streams.
    let (n_cases, partial_lens) = if reuse.is_tail_only() {
        ensure_empty_party_inputs(cfg, n_parties)?;
        slab_shape(all_cases)?
    } else {
        write_inputs_nparty(cfg, all_cases, dict_size)?
    };
    let partial_len = *partial_lens.iter().max().unwrap_or(&1);

    let tau_num = (tau * 10_000.0).round().clamp(0.0, 10_000.0) as u64;
    let mut args = vec![
        n_cases.to_string(),
        partial_len.to_string(),
        dict_size.to_string(),
        mpc_n_threads().to_string(),
        k_proj.to_string(),
        n_parties.to_string(),
        encoding.to_string(),
        enc_dim.to_string(),
        k0.to_string(),
        tau_num.to_string(),
        "1".to_string(), // ADAPTIVE=1
        norm_preshift.to_string(),
    ];
    push_reuse_args(&mut args, &partial_lens, reuse);
    let compiled = compile_program(cfg, MPC_PROGRAM_RSECRET, &args)?;
    // Fingerprint hash weights ride the Public-Input channel: fresh per run
    // without recompiling the per-run-identical bytecode.
    let enc_dim_eff = if enc_dim > 0 { enc_dim } else { dict_size * dict_size };
    super::runtime::write_public_input(
        cfg,
        &compiled.prog,
        &super::runtime::expand_hash_weights(hash_seed, enc_dim_eff),
    )?;
    // Restart rule: a backbone row whose balanced profile lies in the kernel of
    // the in-box ±1 R opens as the exact zero vector. The answer is one re-run
    // of Round 1 with a freshly sampled R, a public event whose discarded
    // attempt stays in timing.json as `round1_rsecret_degenerate`.
    let mut attempt = 1usize;
    loop {
        let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
        record_round(cfg, super::ROUND1_RSECRET, &compiled, &stats);
        forward_progress(&stdout, "NP", &["===", "MODE:", "Saved", "Sort", "tail_only"]);
        match parse_round1_rsecret_output(&stdout, n_cases, k_proj) {
            Err(SpdzError::DegenerateReveal { n_nonzero, n_backbone })
                if attempt <= DEGENERATE_REVEAL_RESTARTS =>
            {
                eprintln!(
                    "[MPC] Round 1 restarted: degenerate reveal on attempt {attempt} \
                     ({n_nonzero} nonzero revealed rows vs |A_kpriv| = {n_backbone}); \
                     re-running Round 1 with a fresh in-box R"
                );
                super::discard_last_round(
                    cfg,
                    super::ROUND1_RSECRET,
                    super::ROUND1_RSECRET_DEGENERATE,
                );
                attempt += 1;
            }
            Err(e) => return Err(e),
            Ok((rows, backbone, k_priv_selected)) => {
                return Ok(Round1RsecretOutput {
                    projected_shuffled: rows,
                    num_cases: n_cases,
                    feature_dim: k_proj,
                    pi_offset: n_cases * dict_size * dict_size,
                    backbone,
                    k_priv_selected,
                })
            }
        }
    }
}

/// How many times a degenerate reveal may restart Round 1 before it is a failure.
pub(crate) const DEGENERATE_REVEAL_RESTARTS: usize = 1;

/// `(revealed rows, backbone flags, enforced k_priv)`.
type Round1RsecretParse = (Vec<Vec<u64>>, Vec<bool>, usize);

/// The per-attempt parse of the rsecret stdout.
fn parse_round1_rsecret_output(
    stdout: &str,
    n_cases: usize,
    k_proj: usize,
) -> Result<Round1RsecretParse, SpdzError> {
    // A missing K_PRIV_SELECTED line can only be dropped stdout and would
    // misreport the enforced threshold.
    let k_priv_selected = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("K_PRIV_SELECTED "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .ok_or_else(|| {
            SpdzError::Runtime("K_PRIV_SELECTED line missing from rsecret stdout".into())
        })?;
    let (rows, backbone) = parse_row_output_backbone(stdout, n_cases, k_proj)?;
    Ok((rows, backbone, k_priv_selected))
}

// Round 2 always runs SECURE_LABELS=1: secure-kmeans reads its one-hot from
// secure_kmeans's output offset, rsec from secure_relabel's.

/// Aggregated DFG rows plus the START-row sum of each opened graph. No cluster
/// size is opened: Round 3 prunes the dead ends inside the MPC before the open
/// (`secure_aggregate.mpc` Step 3c), so a case whose start activity was pruned
/// has no opened START cell and `cluster_sizes[c]` counts only the cases of
/// cluster c whose start activity is kept.
#[derive(Debug)]
pub struct Round2SecureOutput {
    pub aggregated: Vec<Vec<u64>>,
    pub cluster_sizes: Vec<usize>,
}

/// Compile args of `secure_aggregate.mpc`, in the order the tape reads them.
/// The trailing `1` is PRUNE: every pipeline run prunes the dead ends inside
/// the MPC, so no cell of a removed activity is opened.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_compile_args(
    total_rows: usize,
    num_revealed: usize,
    dfg_dim: usize,
    k: usize,
    label_offset: usize,
    n_threads: u32,
    edge_min_count: u64,
    edge_frac_num: u64,
) -> Vec<String> {
    vec![
        total_rows.to_string(),
        num_revealed.to_string(),
        dfg_dim.to_string(),
        k.to_string(),
        "1".to_string(), // SECURE_LABELS=1
        label_offset.to_string(),
        n_threads.to_string(),
        edge_min_count.to_string(),
        edge_frac_num.to_string(),
        "1".to_string(), // PRUNE=1
    ]
}

/// The opened START-row total of all clusters before the pruning, which a
/// correct run makes equal to N.
pub(crate) fn parse_start_total(stdout: &str) -> Result<usize, SpdzError> {
    stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("START_TOTAL "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .ok_or_else(|| {
            SpdzError::Runtime("START_TOTAL line missing from secure_aggregate stdout".into())
        })
}

/// Integer twin of the `filter_dfg_edges` cutoff for the in-MPC pre-reveal
/// filter: keep iff `w >= mc` AND `w * 10^4 >= frac_num * max`. Exact only when
/// the fraction is a plain decimal; otherwise `None`, which callers refuse
/// rather than fall back to a post-reveal filter that opens every cell.
pub(crate) fn edge_filter_int_args(min_count: f64, min_fraction: f64) -> Option<(u64, u64)> {
    if !(0.0..=1.0).contains(&min_fraction) || min_count < 0.0 {
        return None;
    }
    let frac_num = (min_fraction * 10_000.0).round();
    if (frac_num / 10_000.0 - min_fraction).abs() > 1e-13 {
        return None;
    }
    let mc = min_count.ceil();
    // The .mpc compares EDGE_MIN_COUNT at the AGG_CELL_BITS-bounded width.
    if mc >= (1u64 << crate::constants::AGG_CELL_BITS) as f64 {
        return None;
    }
    Some((mc as u64, frac_num as u64))
}

/// Round 2 reading secret one-hot labels from persistence. `edge_filter` zeroes
/// sub-cutoff activity cells inside the MPC before the reveal, so the rare
/// transitions the published DFG would drop are never opened; the dead-end
/// pruning then removes every activity on no start-to-end path, also before the
/// reveal, so its start and end counts are never opened either.
#[allow(clippy::too_many_arguments)]
pub fn run_round2_secure(
    cfg: &SpdzConfig,
    total_rows: usize,
    num_revealed: usize,
    dict_size: usize,
    k: usize,
    label_offset: usize,
    n_parties: usize,
    edge_filter: Option<(f64, f64)>,
) -> Result<Round2SecureOutput, SpdzError> {
    let dfg_dim = dict_size * dict_size;
    // Resolved before any side effect so the refusal below touches nothing. A
    // refusal, not a fallback: disabling the in-MPC pass would open every
    // aggregated cell, which both modes' published envelope rules out.
    let (edge_min_count, edge_frac_num) = match edge_filter {
        Some((mc, mf)) => edge_filter_int_args(mc, mf).ok_or_else(|| {
            SpdzError::Runtime(format!(
                "edge cutoff (min_edge_count {mc}, min_edge_fraction {mf}) has no exact \
                 integer form, so the in-MPC pre-reveal edge filter cannot run and every \
                 aggregated cell would be opened; refusing to widen the reveal"
            ))
        })?,
        None => (0, 0),
    };
    ensure_empty_party_inputs(cfg, n_parties)?;
    let args = aggregate_compile_args(
        total_rows,
        num_revealed,
        dfg_dim,
        k,
        label_offset,
        mpc_n_threads(),
        edge_min_count,
        edge_frac_num,
    );
    let compiled = compile_program(cfg, ROUND2_PROGRAM, &args)?;
    let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
    record_round(cfg, "round2_aggregate", &compiled, &stats);
    forward_progress(&stdout, "R2", &["===", "MODE:", "Aggregate", "Filtered", "Pruned"]);

    let aggregated = parse_row_output(&stdout, k, dfg_dim)?;
    // START-row sum of each opened (pruned) graph of the always-bigram offset-0
    // slab: the cases of the cluster whose start activity survived the pruning.
    let start_base = crate::types::START as usize * dict_size;
    let cluster_sizes: Vec<usize> = aggregated
        .iter()
        .map(|row| row[start_base..start_base + dict_size].iter().map(|&v| v as usize).sum())
        .collect();
    // Every case contributes exactly one START-row cell, so the START rows of
    // all clusters before the pruning must sum to the population, which the tape
    // opens as a public total. A shortfall means cases were lost upstream.
    let start_total = parse_start_total(&stdout)?;
    if start_total != total_rows {
        return Err(SpdzError::Runtime(format!(
            "START-row total of all clusters is {start_total}, expected N = {total_rows}: \
             cases were lost before aggregation; re-run with CLUSTERING_MPC_THREADS=1 \
             to isolate it"
        )));
    }
    Ok(Round2SecureOutput { aggregated, cluster_sizes })
}

#[derive(Debug)]
pub struct KmeansOutput {
    pub label_offset: usize,
    /// Some only under `--evaluate-full`; the persisted secret matrix is unaffected.
    pub revealed_labels: Option<Vec<usize>>,
}

/// Compile args of `secure_kmeans.mpc`, in the order the tape reads them.
/// `argmin_bits` is the E-step comparison width, the same value
/// `parameters.rs::secure_kmeans_argmin_bits` gates the run on; `min_cluster` is
/// the oblivious small-cluster merge's threshold, 0 when the refinement follows
/// and merges instead (the merge runs once, in the last tape of the chain).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kmeans_compile_args(
    n_cases: usize,
    k_proj: usize,
    k: usize,
    max_iter: usize,
    n_threads: u32,
    y_offset: usize,
    reveal_labels: bool,
    argmin_bits: u32,
    min_cluster: usize,
) -> Vec<String> {
    vec![
        n_cases.to_string(),
        k_proj.to_string(),
        k.to_string(),
        max_iter.to_string(),
        n_threads.to_string(),
        "1".to_string(), // INPUT_SOURCE=1 (persistence)
        y_offset.to_string(),
        if reveal_labels { "1" } else { "0" }.to_string(),
        argmin_bits.to_string(),
        min_cluster.to_string(),
    ]
}

/// Compile args of `secure_sequence_cluster.mpc`. PROFILE (the tenth) is always
/// 0 on a pipeline run; `min_cluster` rides the eleventh slot behind it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sequence_compile_args(
    n_cases: usize,
    d_sq: usize,
    k: usize,
    max_iter: usize,
    n_threads: u32,
    label_offset: usize,
    reveal_labels: bool,
    init_mode: usize,
    min_cluster: usize,
) -> Vec<String> {
    vec![
        n_cases.to_string(),
        d_sq.to_string(),
        k.to_string(),
        max_iter.to_string(),
        n_threads.to_string(),
        "1".to_string(), // INPUT_SOURCE=1 (persistence)
        label_offset.to_string(),
        if reveal_labels { "1" } else { "0" }.to_string(),
        init_mode.to_string(),
        "0".to_string(), // PROFILE off on every pipeline run
        min_cluster.to_string(),
    ]
}

/// secure_kmeans.mpc: reads Y at `y_offset`, writes one-hot labels at
/// `y_offset + n_cases * k_proj`. `reveal_labels` adds an additive `LABEL`
/// reveal for `--evaluate-full`.
#[allow(clippy::too_many_arguments)]
pub fn run_kmeans(
    cfg: &SpdzConfig,
    n_cases: usize,
    k_proj: usize,
    k: usize,
    max_iter: usize,
    y_offset: usize,
    n_parties: usize,
    reveal_labels: bool,
    argmin_bits: u32,
    min_cluster: usize,
) -> Result<KmeansOutput, SpdzError> {
    ensure_empty_party_inputs(cfg, n_parties)?;
    let args = kmeans_compile_args(
        n_cases,
        k_proj,
        k,
        max_iter,
        mpc_n_threads(),
        y_offset,
        reveal_labels,
        argmin_bits,
        min_cluster,
    );
    let compiled = compile_program(cfg, KMEANS_PROGRAM, &args)?;
    let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
    record_round(cfg, "secure_kmeans", &compiled, &stats);
    forward_progress(&stdout, "KM", &["===", "MODE:", "Iteration", "Persisting"]);

    let revealed = if reveal_labels {
        Some(collect_indexed_labels(&stdout, "LABEL ", n_cases, k)?)
    } else {
        None
    };

    Ok(KmeansOutput { label_offset: y_offset + n_cases * k_proj, revealed_labels: revealed })
}

/// Markov-chain hard-EM over the d² count rows persisted at offset 0;
/// `init_mode = 1` reads the initial one-hot partition at `label_offset`. The
/// refined one-hot is written back there, so Round 2 consumes it unchanged.
#[allow(clippy::too_many_arguments)]
pub fn run_sequence_cluster(
    cfg: &SpdzConfig,
    n_cases: usize,
    d_sq: usize,
    k: usize,
    max_iter: usize,
    label_offset: usize,
    n_parties: usize,
    reveal_labels: bool,
    init_mode: usize,
    min_cluster: usize,
) -> Result<Option<Vec<usize>>, SpdzError> {
    ensure_empty_party_inputs(cfg, n_parties)?;
    let args = sequence_compile_args(
        n_cases,
        d_sq,
        k,
        max_iter,
        mpc_n_threads(),
        label_offset,
        reveal_labels,
        init_mode,
        min_cluster,
    );
    let compiled = compile_program(cfg, SEQUENCE_PROGRAM, &args)?;
    let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
    record_round(cfg, "secure_sequence_cluster", &compiled, &stats);
    forward_progress(&stdout, "SQ", &["===", "MODE:", "Iteration", "Persisting"]);

    if reveal_labels {
        Ok(Some(collect_indexed_labels(&stdout, "LABEL ", n_cases, k)?))
    } else {
        Ok(None)
    }
}

// secure_relabel: un-shuffle public labels via the persisted secret π,
// build a secret one-hot at `label_offset` for Round 2 (SECURE_LABELS=1).
// The ASSIGN reveal is opt-in via `reveal_labels` (tests / --evaluate-full).

/// Apply π⁻¹ to shuffled labels and persist secret one-hot labels for Round 2.
/// Under `hybrid` the suppressed tail rows carry a sentinel and the program
/// assigns each to its nearest centroid in-MPC from the withheld unit-norm rows
/// at `y_offset`; `centroids` (k × k_proj) are scaled by `NORM_SCALE` into
/// `Input-P0-0`, with the tail count and indices.
pub type RelabelHybrid<'a> = (usize, usize, usize, &'a [Vec<f64>], &'a [usize], u32);

/// Declared width of `secure_relabel.mpc`'s tail-score comparison (its tenth
/// compile arg). Twin of the bound `parameters.rs::rsec_norm_preshift` enforces:
/// a tail score `‖c‖² − 2·y·c` is bounded by `81 · k_proj · 2^(32 + p)` (the
/// `FAR = 8` empty-cluster sentinel puts the 81 there), the comparison sees a
/// difference, and `less_than` needs a sign bit.
pub(crate) fn relabel_tail_cmp_bits(k_proj: usize, norm_preshift: u32) -> u32 {
    let ceil_log2 = |x: u64| x.max(1).next_power_of_two().trailing_zeros();
    ceil_log2(81 * k_proj as u64) + 32 + norm_preshift + 3
}

/// Compile args of `secure_relabel.mpc`. `hybrid` is
/// `(k_proj, y_offset, bb_offset, norm_preshift)`; without it the six-arg
/// non-hybrid form is emitted.
pub(crate) fn relabel_compile_args(
    n: usize,
    pi_offset: usize,
    n_threads: u32,
    k: usize,
    label_offset: usize,
    reveal_labels: bool,
    hybrid: Option<(usize, usize, usize, u32)>,
) -> Vec<String> {
    let mut args = vec![
        n.to_string(),
        pi_offset.to_string(),
        n_threads.to_string(),
        k.to_string(),
        label_offset.to_string(),
        if reveal_labels { "1" } else { "0" }.to_string(),
    ];
    if let Some((k_proj, y_offset, bb_offset, norm_preshift)) = hybrid {
        args.push(k_proj.to_string());
        args.push(y_offset.to_string());
        args.push(bb_offset.to_string());
        args.push(relabel_tail_cmp_bits(k_proj, norm_preshift).to_string());
    }
    args
}

#[allow(clippy::too_many_arguments)]
pub fn run_relabel(
    cfg: &SpdzConfig,
    shuffled_labels: &[usize],
    pi_offset: usize,
    k: usize,
    label_offset: usize,
    reveal_labels: bool,
    n_parties: usize,
    hybrid: Option<RelabelHybrid<'_>>,
) -> Result<Option<Vec<usize>>, SpdzError> {
    let n = shuffled_labels.len();
    if n == 0 {
        return Ok(if reveal_labels { Some(vec![]) } else { None });
    }

    let dir = cfg.spdz_path.join("Player-Data");
    fs::create_dir_all(&dir)?;
    {
        use std::io::{BufWriter, Write};
        let mut f = BufWriter::new(std::fs::File::create(dir.join("Input-P0-0"))?);
        for &c in shuffled_labels {
            writeln!(f, "{}", c)?;
        }
        // Hybrid reveal: append K*K_PROJ centroid cells (×NORM_SCALE,
        // row-major), then the tail count and tail row indices;
        // secure_relabel reads them as player-0 inputs.
        if let Some((k_proj, _, _, centroids, tail_idx, _)) = hybrid {
            if centroids.len() != k {
                return Err(SpdzError::Runtime(format!(
                    "relabel: {} centroid rows, expected K={}",
                    centroids.len(),
                    k
                )));
            }
            for (j, row) in centroids.iter().enumerate() {
                if row.len() != k_proj {
                    return Err(SpdzError::Runtime(format!(
                        "relabel: centroid row {} has length {}, expected K_PROJ={}",
                        j,
                        row.len(),
                        k_proj
                    )));
                }
                for &v in row {
                    writeln!(f, "{}", (v * NORM_SCALE).round() as i64)?;
                }
            }
            if let Some(&bad) = tail_idx.iter().find(|&&i| i >= n) {
                return Err(SpdzError::Runtime(format!(
                    "relabel: tail index {} out of range (N={})",
                    bad, n
                )));
            }
            writeln!(f, "{}", tail_idx.len())?;
            for &i in tail_idx {
                writeln!(f, "{}", i)?;
            }
        }
        f.flush()?;
    }
    ensure_empty_party_inputs(cfg, n_parties)?;

    let args = relabel_compile_args(
        n,
        pi_offset,
        mpc_n_threads(),
        k,
        label_offset,
        reveal_labels,
        hybrid.map(|(k_proj, y_offset, bb_offset, _, _, preshift)| {
            (k_proj, y_offset, bb_offset, preshift)
        }),
    );
    let compiled = compile_program(cfg, RELABEL_PROGRAM, &args)?;
    let (stdout, stats) = run_n_party(cfg, n_parties, &compiled.prog)?;
    record_round(cfg, "secure_relabel", &compiled, &stats);
    forward_progress(&stdout, "RL", &["===", "MODE:"]);

    if reveal_labels {
        // The relabel program is its own ground truth on label validity.
        Ok(Some(collect_indexed_labels(&stdout, "ASSIGN ", n, usize::MAX)?))
    } else {
        Ok(None)
    }
}

/// Parse `<tag> <i> <c>` into a length-`n` vector. Hard error on any missing index.
fn collect_indexed_labels(
    stdout: &str,
    tag: &str,
    n: usize,
    k_max: usize,
) -> Result<Vec<usize>, SpdzError> {
    let mut labels = vec![usize::MAX; n];
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix(tag) {
            let mut it = rest.split_whitespace();
            if let (Some(i_s), Some(c_s)) = (it.next(), it.next()) {
                if let (Ok(i), Ok(c)) = (i_s.parse::<usize>(), c_s.parse::<usize>()) {
                    if i < n && c < k_max {
                        labels[i] = c;
                    }
                }
            }
        }
    }
    let missing = labels.iter().filter(|&&c| c == usize::MAX).count();
    if missing > 0 {
        return Err(SpdzError::Runtime(format!(
            "{}: {}/{} lines missing in stdout",
            tag.trim(),
            missing,
            n
        )));
    }
    Ok(labels)
}
