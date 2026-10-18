//! `--reveal-secret-projection` mode: secure_extract_rsecret samples R inside
//! MPC, shuffles rows by secret π and reveals Y_π; plaintext clustering ->
//! secure_relabel applies π⁻¹ -> secure_aggregate under SECURE_LABELS=1.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::PipelineConfig;
use crate::constants::*;
use crate::error::{PipelineError, Result};
use crate::evaluation;
use crate::mpc;
use crate::types::{ActivityDictionary, BigramIndex, EncodedTrace, FeatureMatrix};
use crate::utils::l2_normalize;

use super::shared::{dfgs_from_aggregated, run_plaintext_baseline};

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    cfg: &PipelineConfig,
    cfg_k: &PipelineConfig,
    spdz_cfg: &mpc::SpdzConfig,
    all_cases: &[Vec<mpc::CaseData>],
    dict_size: usize,
    k_proj: usize,
    encoding_flag: usize,
    enc_dim: usize,
    reuse: mpc::ExtractReuse,
    k0: usize,
    norm_preshift: u32,
    dictionary: &ActivityDictionary,
    pt_traces: &[(String, EncodedTrace)],
    pt_bigram_index: &BigramIndex,
    pt_features_raw: &FeatureMatrix,
    output_base: &Path,
    effective_max_trace_len: usize,
    t_mpc: Instant,
    n_parties: usize,
) -> Result<()> {
    println!("\n--- Protocol: Reveal-Secret-Projection ({} parties) ---", n_parties);
    println!("  Round 1: secret R, dense secret projection, row shuffle, reveal Y_shuffled");
    println!("  Middle:  plaintext clustering on Y_shuffled (any algorithm, any k)");
    println!("  Relabel: tiny MPC pass applies π⁻¹, persists secret one-hot labels");
    println!("  Round 2: secure aggregation under SECURE_LABELS=1 (DFG cells)");

    // Fresh public fingerprint-weight seed per run: a hash collision between two
    // distinct profiles (false anonymity) is then a per-run probability, not a
    // fixed property of the log. Printed so a run can be re-audited.
    let hash_seed: u64 = rand::random();
    println!("[Privacy] Fingerprint hash seed: {}", hash_seed);

    // Revealed cells are `unit_value * 2^(16 + preshift/2)`; the validator
    // in mod.rs picked the preshift so the in-MPC sfix cast cannot overflow.
    let norm_scale = crate::constants::norm_scale_for_preshift(norm_preshift);

    let t_r1 = Instant::now();
    let r1 = mpc::run_round1_rsecret(
        spdz_cfg,
        all_cases,
        dict_size,
        k_proj,
        encoding_flag,
        enc_dim,
        k0,
        hash_seed,
        MAX_SINGLETON_FRACTION,
        norm_preshift,
        reuse,
    )?;
    println!("[MPC] Round 1 (rsecret) completed in {:?}", t_r1.elapsed());
    // The threshold the MPC chose inside the protocol on the secret class
    // sizes, with no plaintext input.
    let k_priv = r1.k_priv_selected;
    println!(
        "[Privacy] Budget-bounded k_priv = {} (selected in-MPC; k0={}, tau={:.0}%)",
        k_priv,
        k0,
        MAX_SINGLETON_FRACTION * 100.0
    );
    println!(
        "[MPC] {} shuffled rows, dim={}, π at offset {}",
        r1.num_cases, r1.feature_dim, r1.pi_offset
    );
    if r1.num_cases == 0 {
        return Err(PipelineError::Mpc(
            "Reveal-secret-projection path requires at least one case; got 0".into(),
        ));
    }

    // L2-row-normalisation against the norm-matching attack happens inside
    // MPC (Step 2c-norm in secure_extract_rsecret.mpc); the Rust-side call
    // below is idempotent quantisation polish only.
    let features_shuffled: FeatureMatrix = r1
        .projected_shuffled
        .iter()
        .map(|row| row.iter().map(|&v| (v as i64 as f64) / norm_scale).collect())
        .collect();
    // Sanity: each revealed row should be ≈ unit-norm; a row near the NORM_FLOOR
    // clamp still carries magnitude signal, so log it. Suppressed-tail rows are
    // masked to zero before the reveal, so this covers backbone rows only.
    let (n_low, min_n2) = features_shuffled.iter().zip(&r1.backbone).filter(|(_, &b)| b).fold(
        (0usize, f64::INFINITY),
        |(c, m), (r, _)| {
            let n2: f64 = r.iter().map(|v| v * v).sum();
            (c + (n2 < 0.5) as usize, m.min(n2))
        },
    );
    if n_low > 0 {
        eprintln!("[Privacy] WARNING: {} revealed rows had ‖y‖² < 0.5 (min={:.4}); norm-matching defence weakened on these rows", n_low, min_n2);
    } else {
        println!("[Privacy] Unit-norm check: all revealed rows ‖y‖² ≥ 0.5 (min={:.4})", min_n2);
    }
    let features_shuffled = l2_normalize(&features_shuffled);

    // Hybrid reveal: only the backbone (anonymous-set) rows are non-zero in the
    // revealed Y. Cluster those alone; the suppressed tail is assigned to the
    // nearest backbone centroid in secure_relabel and enters the Round 2 DFGs.
    let suppress = k_priv >= 2 && r1.backbone.iter().any(|&b| !b);
    let n_backbone = r1.backbone.iter().filter(|&&b| b).count();
    if suppress {
        println!(
            "[Privacy] Hybrid reveal: {}/{} rows in anonymous set A_kpriv (k_priv={}); \
             {} suppressed-tail rows assigned in-MPC by nearest centroid",
            n_backbone,
            r1.num_cases,
            k_priv,
            r1.num_cases - n_backbone
        );
    }

    let cluster_input: FeatureMatrix = if suppress {
        features_shuffled
            .iter()
            .zip(&r1.backbone)
            .filter(|(_, &b)| b)
            .map(|(row, _)| row.clone())
            .collect()
    } else {
        features_shuffled.clone()
    };

    // Degenerate split: every case fell below the k_priv floor, so the anonymous
    // set is empty and there is nothing to cluster. Fail here rather than handing
    // k_select.py a 0-row matrix, on which it crashes.
    if cluster_input.is_empty() {
        return Err(PipelineError::Mpc(format!(
            "empty revealed backbone: all {} cases fall below the k_priv floor \
             (k_priv={}); structural org-split degeneracy — the split has too few \
             cross-party cases to form a k_priv-anonymous set",
            r1.num_cases, k_priv
        )));
    }

    // Degenerate encoding: a non-empty backbone collapsing to fewer than two
    // distinct rows — an encoding that does not fit the log. Report it rather
    // than failing inside sklearn on a single-label score.
    if cfg_k.k_fixed().is_none() {
        let mut distinct: Vec<&Vec<f64>> = Vec::new();
        for row in &cluster_input {
            if !distinct.iter().any(|d| {
                d.len() == row.len() && d.iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-12)
            }) {
                distinct.push(row);
                if distinct.len() >= 2 {
                    break;
                }
            }
        }
        if distinct.len() < 2 {
            return Err(PipelineError::Mpc(format!(
                "degenerate revealed backbone: {} rows collapse to a single distinct \
                 embedding profile, so no clustering structure exists; the encoding \
                 does not fit this log (check for traces too short to produce any \
                 feature, e.g. trigram on traces of fewer than three events)",
                cluster_input.len()
            )));
        }
    }

    // A pinned k is bounded by the AGREED cases before Round 1, but the
    // plaintext middle step sees only the revealed backbone, which hybrid
    // suppression can shrink well below N.
    if let Some(k) = cfg_k.k_fixed() {
        super::validate_k_against_cases(k, cluster_input.len(), "revealed backbone rows")?;
    }

    // k resolves HERE when --k was not given: run_clustering auto-selects on
    // cluster_input — the revealed backbone rows of Y_pi, envelope data any party
    // could run the same selection on. The merged plaintext is never consulted.
    let t_local = Instant::now();
    let mut cluster_result = super::super::clustering::run_clustering(
        &cluster_input,
        cfg_k,
        Some(effective_max_trace_len),
        output_base,
        // The revealed rows are the JL-projected Y, not d² count rows:
        // ClusteringAlgo::Sequence is not available on this path.
        None,
    )?;
    let local_clustering_elapsed = t_local.elapsed();
    println!("[Local] clustering completed in {:?}", local_clustering_elapsed);
    let k = cluster_result.representatives.len();
    if cfg_k.k_fixed().is_none() {
        println!(
            "[Auto-k] Resolved k={} (selected on the revealed backbone rows of Y_pi; \
             shared with the plaintext baseline)",
            k
        );
    }
    // Extract-once safety net: the rsec write range (π + Y + backbone +
    // labels) must stay strictly below the shared encoding slab so a branch run
    // never clobbers it. enc_slab_offset is unit-tested to clear this.
    if let Some(enc_off) = reuse.enc_offset() {
        let end = mpc::mode_layout_end_bound(r1.num_cases, dict_size * dict_size, k_proj, k);
        if end > enc_off {
            return Err(PipelineError::Mpc(format!(
                "extract-once layout: rsec mode slabs end at {} but the encoding slab sits at \
                 {} (N={}, k_proj={}, k={})",
                end, enc_off, r1.num_cases, k_proj, k
            )));
        }
    }

    // merge_small_clusters is permutation-invariant — safe before un-shuffle.
    super::super::clustering::merge_small_clusters(
        &mut cluster_result.assignments,
        &cluster_result.representatives,
        MIN_CLUSTER_SIZE,
    );

    // Build the full-length per-shuffled-position label vector. Backbone rows
    // carry their plaintext label; suppressed-tail rows carry sentinel 0 and
    // are overwritten by the in-MPC nearest-centroid argmin in secure_relabel.
    let (shuffled_labels, hybrid_centroids): (Vec<usize>, Option<Vec<Vec<f64>>>) = if suppress {
        let mut full = vec![0usize; r1.num_cases];
        let mut it = cluster_result.assignments.iter();
        for (pos, &bb) in r1.backbone.iter().enumerate() {
            if bb {
                full[pos] = *it.next().expect("backbone count matches clustered rows");
            }
        }
        // Post-merge centroids over the BACKBONE rows, indexed by final label.
        // Empty (merged-away) clusters get a far centroid so the tail argmin
        // never selects them.
        let mut centroids =
            recompute_centroids(&cluster_input, &cluster_result.assignments, k, k_proj);
        // run_relabel quantises as round(v * NORM_SCALE), but the persisted Y the
        // tail argmin compares against is at this run's norm_scale; pre-scale by
        // the exact power-of-two ratio so both sides share one integer scale.
        let ratio = norm_scale / NORM_SCALE;
        if ratio != 1.0 {
            for row in &mut centroids {
                for v in row {
                    *v *= ratio;
                }
            }
        }
        (full, Some(centroids))
    } else {
        (cluster_result.assignments.clone(), None)
    };

    // rsec persistence layout (mirrored in secure_extract_rsecret.mpc): bigrams
    // at offset 0; π at pi_offset; shuffled Y at pi_offset + N; backbone bits at
    // pi_offset + N + N*K_PROJ; one-hot labels (relabel writes) at
    // pi_offset + 2N + N*K_PROJ.
    let y_offset = r1.pi_offset + r1.num_cases;
    let bb_offset = y_offset + r1.num_cases * k_proj;
    let label_offset = bb_offset + r1.num_cases;

    // Tail row indices are public envelope data (the zero rows of the
    // revealed Y_π); the relabel argmin loops only these rows.
    let tail_idx: Vec<usize> =
        r1.backbone.iter().enumerate().filter(|(_, &b)| !b).map(|(i, _)| i).collect();
    // `norm_preshift` rides along so `rounds.rs::relabel_tail_cmp_bits` can
    // declare the tail comparison at the run's own score bound, not the ring.
    let hybrid_arg = hybrid_centroids
        .as_ref()
        .map(|c| (k_proj, y_offset, bb_offset, c.as_slice(), tail_idx.as_slice(), norm_preshift));

    // secure_relabel un-shuffles by shuffle-then-open on the perfect key π (a
    // permutation of 0..num_cases-1), so a short label vector would fail its
    // in-MPC guard only after the extract round is paid; enforce it here too.
    assert_eq!(
        shuffled_labels.len(),
        r1.num_cases,
        "relabel needs one label per shuffled slot (perfect-key un-shuffle)"
    );
    let t_rl = Instant::now();
    let revealed_labels = mpc::run_relabel(
        spdz_cfg,
        &shuffled_labels,
        r1.pi_offset,
        k,
        label_offset,
        cfg.evaluate_full,
        n_parties,
        hybrid_arg,
    )?;
    println!("[MPC] secure_relabel completed in {:?}", t_rl.elapsed());

    let t_r2 = Instant::now();
    let r2 = mpc::run_round2_secure(
        spdz_cfg,
        r1.num_cases,
        r1.num_cases,
        dict_size,
        k,
        label_offset,
        n_parties,
        Some((MIN_EDGE_COUNT, MIN_EDGE_FRACTION)),
    )?;
    println!("[MPC] Round 2 (secret labels) completed in {:?}", t_r2.elapsed());
    println!("[MPC] Cluster sizes (derived from START rows): {:?}", r2.cluster_sizes);

    let cluster_sizes = r2.cluster_sizes.clone();
    let dfgs = dfgs_from_aggregated(&r2.aggregated, &cluster_sizes, dict_size);
    // The plaintext middle step is booked separately under [Local] above,
    // so the MPC total covers protocol work only.
    println!("MPC completed in {:?}", t_mpc.elapsed().saturating_sub(local_clustering_elapsed));

    let output_dir: PathBuf =
        if cfg.evaluate_full { output_base.join("federated") } else { output_base.to_path_buf() };
    let secure_suppress = revealed_labels.is_none();
    let export_assignments: &[usize] = revealed_labels.as_deref().unwrap_or(&[]);
    super::super::export::export_mpc_results(
        &output_dir,
        &dfgs,
        dictionary,
        pt_traces,
        export_assignments,
        k,
        secure_suppress,
    )?;

    if cfg.evaluate_full {
        super::super::export::export_features(&features_shuffled, &output_dir)?;
        println!("\n--- Evaluation: federated ---");
        evaluation::process_metrics::run_evaluate_all(&output_dir, "federated", cfg.fast_eval)?;

        // No projected tier here: R is sampled inside MPC and never opened, and
        // the federated tier already clusters the revealed Y. Pin the k that
        // tier resolved so both stay comparable even under auto-k.
        let mut cfg_k_pinned = cfg_k.clone();
        cfg_k_pinned.k = Some(crate::config::KSpec::Fixed(k));
        run_plaintext_baseline(
            cfg,
            &cfg_k_pinned,
            output_base,
            None,
            pt_traces,
            pt_bigram_index,
            pt_features_raw,
            dictionary,
            dict_size,
        )?;
    }

    if let Err(e) = mpc::write_timing_json(spdz_cfg, output_base) {
        eprintln!("[MPC] WARNING: could not write timing.json: {}", e);
    }
    Ok(())
}

/// Mean of the rows assigned to each cluster index, in unit-norm float space.
/// Empty cluster slots get a sentinel far from any unit-norm row so the tail
/// nearest-centroid argmin in secure_relabel never selects them.
fn recompute_centroids(
    features: &FeatureMatrix,
    assignments: &[usize],
    k: usize,
    k_proj: usize,
) -> Vec<Vec<f64>> {
    // The sentinel is bounded by secure_relabel's sint score accumulation on the
    // 2^64 ring: cells scale by NORM_SCALE = 2^17, so a sentinel cell is 2^20 and
    // the tail score stays inside the signed range (a sentinel whose scaled square
    // exceeds 2^63 wraps, turns the "far" score negative and WINS the argmin). It
    // still loses legitimately: (2^20-2^17)² exceeds the max real (2·2^17)².
    const FAR: f64 = 8.0;
    let mut sums = vec![vec![0.0f64; k_proj]; k];
    let mut counts = vec![0usize; k];
    for (row, &c) in features.iter().zip(assignments) {
        if c < k {
            for (acc, &v) in sums[c].iter_mut().zip(row) {
                *acc += v;
            }
            counts[c] += 1;
        }
    }
    sums.iter_mut()
        .zip(&counts)
        .map(
            |(s, &n)| {
                if n == 0 {
                    vec![FAR; k_proj]
                } else {
                    s.iter().map(|v| v / n as f64).collect()
                }
            },
        )
        .collect()
}
