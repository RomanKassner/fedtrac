//! `--secure-kmeans` mode: secure_extract (SECURE_Y=1) -> secure_kmeans ->
//! secure_aggregate (SECURE_LABELS=1). Y and labels stay secret-shared;
//! per-cluster DFG + sizes leak at Round 2.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::{KSpec, PipelineConfig, SecureClustering};
use crate::constants::*;
use crate::error::{PipelineError, Result};
use crate::evaluation;
use crate::mpc;
use crate::types::{ActivityDictionary, BigramIndex, EncodedTrace, FeatureMatrix};

use super::shared::{dfgs_from_aggregated, run_plaintext_baseline};

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    cfg: &PipelineConfig,
    cfg_k: &PipelineConfig,
    k: usize,
    spdz_cfg: &mpc::SpdzConfig,
    all_cases: &[Vec<mpc::CaseData>],
    dict_size: usize,
    k_proj: usize,
    r_path: &Path,
    r_matrix: &crate::projection::AchlioptasMatrix,
    encoding_flag: usize,
    enc_dim: usize,
    reuse: mpc::ExtractReuse,
    dictionary: &ActivityDictionary,
    pt_traces: &[(String, EncodedTrace)],
    pt_bigram_index: &BigramIndex,
    pt_features_raw: &FeatureMatrix,
    output_base: &Path,
    t_mpc: Instant,
    n_parties: usize,
    // `parameters.rs::secure_kmeans_argmin_bits` for this config, handed to the
    // tape as `ARGMIN_BITS` so the E-step comparison is no wider than needed.
    argmin_bits: u32,
) -> Result<()> {
    // `sequence` appends secure_sequence_cluster to the middle step.
    let sequence = cfg.secure_clustering == SecureClustering::Sequence;
    println!("\n--- Protocol: Secure k-means ({} parties) ---", n_parties);
    println!("  Round 1: encoding + projection (persisted, not revealed)");
    if sequence {
        println!(
            "  Middle:  Lloyd's k-means (k={}, max_iter={}) on secret Y, then Markov-chain \
             refinement (max_iter={}, init=kmeans), then the oblivious small-cluster merge (min {} cases)",
            k, SECURE_KMEANS_MAX_ITER, SECURE_SEQUENCE_MAX_ITER, MIN_CLUSTER_SIZE
        );
    } else {
        println!(
            "  Middle:  Lloyd's k-means (k={}, max_iter={}) on secret Y, then the oblivious \
             small-cluster merge (min {} cases)",
            k, SECURE_KMEANS_MAX_ITER, MIN_CLUSTER_SIZE
        );
    }
    println!("  Round 2: secure aggregation with secret labels (DFG)");

    let t_r1 = Instant::now();
    let r1s = mpc::run_round1_secure(
        spdz_cfg,
        all_cases,
        dict_size,
        k_proj,
        r_path,
        encoding_flag,
        enc_dim,
        reuse,
    )?;
    let _ = fs::remove_file(r_path);
    println!("[MPC] Round 1 (secure) completed in {:?}", t_r1.elapsed());
    println!("[MPC] {} cases, Y persisted at offset {}", r1s.num_cases, r1s.y_offset);
    if r1s.num_cases == 0 {
        return Err(PipelineError::Mpc(
            "Secure k-means path requires at least one case; got 0".into(),
        ));
    }

    // Extract-once safety net: the SK write range (Y + labels) must stay
    // strictly below the shared encoding slab so a branch run never clobbers
    // it. enc_slab_offset is unit-tested, so this fires only on a coding error.
    if let Some(enc_off) = reuse.enc_offset() {
        let end = mpc::mode_layout_end_bound(r1s.num_cases, dict_size * dict_size, k_proj, k);
        if end > enc_off {
            return Err(PipelineError::Mpc(format!(
                "extract-once layout: SK mode slabs end at {} but the encoding slab sits at \
                 {} (N={}, k_proj={}, k={})",
                end, enc_off, r1s.num_cases, k_proj, k
            )));
        }
    }

    // --evaluate-full adds an additive label reveal so per-trace artefacts
    // exist for bridge/evaluate.py; the persisted one-hot matrix is unchanged.
    let reveal_labels = cfg.evaluate_full;
    // Both middle steps write the one-hot label matrix to the same slab.
    let (label_offset, mut revealed_labels) = {
        let t_km = Instant::now();
        let km_out = mpc::run_kmeans(
            spdz_cfg,
            r1s.num_cases,
            k_proj,
            k,
            SECURE_KMEANS_MAX_ITER,
            r1s.y_offset,
            n_parties,
            // Under `sequence` the k-means labels are superseded; only the
            // refined ones are revealed for --evaluate-full.
            reveal_labels && !sequence,
            argmin_bits,
            // The oblivious small-cluster merge runs ONCE, in the last tape of
            // the chain: here unless the refinement follows.
            if sequence { 0 } else { MIN_CLUSTER_SIZE },
        )?;
        println!(
            "[MPC] secure_kmeans completed in {:?} (labels at offset {})",
            t_km.elapsed(),
            km_out.label_offset
        );
        (km_out.label_offset, km_out.revealed_labels)
    };

    if sequence {
        let t_sq = Instant::now();
        revealed_labels = mpc::run_sequence_cluster(
            spdz_cfg,
            r1s.num_cases,
            dict_size * dict_size,
            k,
            SECURE_SEQUENCE_MAX_ITER,
            label_offset,
            n_parties,
            reveal_labels,
            1, // INIT_MODE: refine the k-means partition
            MIN_CLUSTER_SIZE,
        )?;
        println!(
            "[MPC] secure_sequence_cluster completed in {:?} (refined labels at offset {})",
            t_sq.elapsed(),
            label_offset
        );
    }

    let t_r2 = Instant::now();
    let r2 = mpc::run_round2_secure(
        spdz_cfg,
        r1s.num_cases,
        r1s.num_cases,
        dict_size,
        k,
        label_offset,
        n_parties,
        Some((MIN_EDGE_COUNT, MIN_EDGE_FRACTION)),
    )?;
    println!("[MPC] Round 2 (secure labels) completed in {:?}", t_r2.elapsed());
    println!("[MPC] Cluster sizes (derived from START rows): {:?}", r2.cluster_sizes);

    let cluster_sizes = r2.cluster_sizes.clone();
    let dfgs = dfgs_from_aggregated(&r2.aggregated, &cluster_sizes, dict_size);
    println!("MPC completed in {:?}", t_mpc.elapsed());

    let output_dir: PathBuf =
        if cfg.evaluate_full { output_base.join("federated") } else { output_base.to_path_buf() };

    let secure_suppress = revealed_labels.is_none();
    let assignments_slice: &[usize] = revealed_labels.as_deref().unwrap_or(&[]);
    super::super::export::export_mpc_results(
        &output_dir,
        &dfgs,
        dictionary,
        pt_traces,
        assignments_slice,
        k,
        secure_suppress,
    )?;

    if cfg.evaluate_full {
        if reveal_labels {
            println!("\n--- Evaluation: federated (labels revealed for eval only) ---");
        } else {
            println!("\n--- Evaluation: federated (DFG-level only - labels secret) ---");
        }
        evaluation::process_metrics::run_evaluate_all(&output_dir, "federated", cfg.fast_eval)?;

        // The baseline tier needs the plaintext COUNTERPART of the middle step,
        // so it runs the sequence twin whenever the MPC tier did, at the same k.
        let mut cfg_tiers = cfg_k.clone();
        cfg_tiers.k = Some(KSpec::Fixed(k));
        if sequence {
            cfg_tiers.clustering_algo = crate::config::ClusteringAlgo::Sequence;
        }
        run_plaintext_baseline(
            cfg,
            &cfg_tiers,
            output_base,
            Some(r_matrix),
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
