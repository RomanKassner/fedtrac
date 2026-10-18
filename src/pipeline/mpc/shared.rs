//! Helpers shared by both MPC modes: encoding dispatch, DFG assembly from a
//! Round-2 aggregate, and the `--evaluate-full` plaintext baseline tiers.

use std::path::Path;

use crate::config::{ClusteringAlgo, Encoding, PipelineConfig};
use crate::constants::{MIN_EDGE_COUNT, MIN_EDGE_FRACTION};
use crate::error::Result;
use crate::phases;
use crate::types::{ActivityDictionary, BigramIndex, ClusterDFG, EncodedTrace, FeatureMatrix};
use crate::utils::l2_normalize;

/// L2-normalised plaintext clustering features for the given encoding.
pub(super) fn extract_clustering_features(
    encoding: Encoding,
    pt_traces: &[(String, EncodedTrace)],
    pt_bigram_index: &BigramIndex,
    pt_features_raw: &FeatureMatrix,
    dict_size: usize,
) -> Result<FeatureMatrix> {
    let (raw, _) = extract_raw_features_dense_cols(
        encoding,
        pt_traces,
        pt_bigram_index,
        pt_features_raw,
        dict_size,
    );
    Ok(l2_normalize(&raw))
}

/// Raw (un-normalised) clustering features plus, for the sparse-indexed
/// encodings (bigram/trigram), the map from each sparse column to its dense
/// `enc_dim` column — the space R is generated over. `None` = already dense.
pub(in crate::pipeline) fn extract_raw_features_dense_cols(
    encoding: Encoding,
    pt_traces: &[(String, EncodedTrace)],
    pt_bigram_index: &BigramIndex,
    pt_features_raw: &FeatureMatrix,
    dict_size: usize,
) -> (FeatureMatrix, Option<Vec<usize>>) {
    match encoding {
        Encoding::Bigram => {
            (pt_features_raw.clone(), Some(bigram_dense_cols(pt_bigram_index, dict_size)))
        }
        Encoding::Ef => (phases::p2_bigrams::extract_ef(pt_traces, dict_size), None),
        Encoding::Activity => (phases::p2_bigrams::extract_activity(pt_traces, dict_size), None),
        Encoding::Trigram => {
            let (idx, tri) = phases::p2_bigrams::extract_trigram(pt_traces);
            // Dense trigram columns live in the (d-3)³ USER-activity cube:
            // PAD/START/END never appear in a trigram, so the flat index is
            // (a-3)·du² + (b-3)·du + (c-3), matching encoding_trigram.py.
            let first = crate::types::FIRST_USER_ACTIVITY as usize;
            let du = dict_size - first;
            let mut map = vec![0usize; idx.len()];
            for (&(a, b, c), &l) in idx.iter() {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                assert!(
                    a >= first && b >= first && c >= first,
                    "trigram over reserved activity ids ({}, {}, {})",
                    a,
                    b,
                    c
                );
                map[l] = (a - first) * du * du + (b - first) * du + (c - first);
            }
            (tri, Some(map))
        }
        Encoding::SkipBigram => {
            (phases::p2_bigrams::extract_skip_bigram(pt_traces, dict_size), None)
        }
    }
}

fn bigram_dense_cols(index: &BigramIndex, dict_size: usize) -> Vec<usize> {
    let mut map = vec![0usize; index.len()];
    for (&(a, b), &l) in index.iter() {
        map[l] = (a as usize) * dict_size + b as usize;
    }
    map
}

/// Twin of the `ENCODING` dispatch ladder of both extract programs.
pub(super) fn encoding_flag(encoding: Encoding) -> Result<usize> {
    Ok(match encoding {
        Encoding::Bigram => 0,
        Encoding::Ef => 1,
        // 2 was ppmi-bigram; the .mpc ladder keeps the retired slot so the
        // other flags stay stable.
        Encoding::Activity => 3,
        Encoding::Trigram => 4,
        Encoding::SkipBigram => 5,
    })
}

/// Encoding output dim: activity d, trigram the dense (d-3)³ user-activity cube
/// (twin of mpc_utils/encoding_trigram.py), skip-bigram 2·d², bigram/EF d².
pub(crate) fn encoding_dim(encoding: Encoding, dict_size: usize) -> usize {
    match encoding {
        Encoding::Activity => dict_size,
        Encoding::Trigram => {
            let du = dict_size - crate::types::FIRST_USER_ACTIVITY as usize;
            du * du * du
        }
        Encoding::SkipBigram => 2 * dict_size * dict_size,
        _ => dict_size * dict_size,
    }
}

/// Per-cluster DFGs from a Round-2 aggregate. Both release rules are re-applied
/// as idempotent post-reveal safety nets; the tape ran them before the open.
pub(super) fn dfgs_from_aggregated(
    aggregated_raw: &[Vec<u64>],
    cluster_sizes: &[usize],
    dict_size: usize,
) -> Vec<ClusterDFG> {
    let aggregated: Vec<Vec<f64>> =
        aggregated_raw.iter().map(|row| row.iter().map(|&v| v as i64 as f64).collect()).collect();
    let mut dfgs =
        phases::p4_aggregate::build_dfgs_from_dense(&aggregated, dict_size, cluster_sizes);
    phases::p4_aggregate::filter_dfg_edges(&mut dfgs, MIN_EDGE_COUNT, MIN_EDGE_FRACTION);
    phases::p4_aggregate::prune_dead_ends(&mut dfgs);
    dfgs
}

/// Does the projected-plaintext middle tier measure anything? Only when the
/// tier's algorithm clusters the projected features: the sequence twin scores
/// the dense d² count rows, so its JL gap is structurally zero.
pub(crate) fn projected_tier_applies(algo: ClusteringAlgo) -> bool {
    algo != ClusteringAlgo::Sequence
}

/// `--evaluate-full` plaintext baseline. `r_matrix` (secure-kmeans only) adds
/// the projected tier, so plaintext-vs-projected isolates the JL loss and
/// projected-vs-federated the MPC cost.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_plaintext_baseline(
    cfg: &PipelineConfig,
    cfg_k: &PipelineConfig,
    output_base: &Path,
    r_matrix: Option<&crate::projection::AchlioptasMatrix>,
    pt_traces: &[(String, EncodedTrace)],
    pt_bigram_index: &BigramIndex,
    pt_features_raw: &FeatureMatrix,
    dictionary: &ActivityDictionary,
    dict_size: usize,
) -> Result<()> {
    println!("\n--- Plaintext Baseline ---");
    let pt_clustering_features = extract_clustering_features(
        cfg.encoding,
        pt_traces,
        pt_bigram_index,
        pt_features_raw,
        dict_size,
    )?;
    // Both tiers cluster the same cases in the same order, so one matrix of
    // dense d² rows serves both; built only when the sequence twin needs it.
    let dfg_rows = (cfg_k.clustering_algo == ClusteringAlgo::Sequence).then(|| {
        phases::p2_bigrams::dense_bigram_rows(pt_features_raw, pt_bigram_index, dict_size)
    });
    let pt_cluster = super::super::clustering::run_clustering(
        &pt_clustering_features,
        cfg_k,
        None,
        output_base,
        dfg_rows.as_ref(),
    )?;
    super::super::export::export_plaintext_results(
        pt_bigram_index,
        pt_features_raw,
        pt_traces,
        &pt_cluster,
        dictionary,
        cfg,
        &output_base.join("plaintext"),
        "plaintext",
    )?;

    let seq_tier = !projected_tier_applies(cfg_k.clustering_algo);
    if seq_tier && r_matrix.is_some() {
        println!(
            "\n--- Projected-Plaintext Baseline: SKIPPED (the sequence twin scores the \
             same d^2 count rows as the plaintext tier, so plaintext-vs-projected \
             measures no JL loss) ---"
        );
    }
    if let Some(r) = r_matrix.filter(|_| !seq_tier) {
        println!("\n--- Projected-Plaintext Baseline (JL k_proj={}) ---", r.k_proj);
        let (raw, col_map) = extract_raw_features_dense_cols(
            cfg.encoding,
            pt_traces,
            pt_bigram_index,
            pt_features_raw,
            dict_size,
        );
        // R scales every norm by the same sqrt(k_proj/3) factor in expectation,
        // so normalising after projection is projecting the normalised rows.
        let projected = l2_normalize(&r.project(&raw, col_map.as_deref()));
        let proj_cluster = super::super::clustering::run_clustering(
            &projected,
            cfg_k,
            None,
            output_base,
            dfg_rows.as_ref(),
        )?;
        super::super::export::export_plaintext_results(
            pt_bigram_index,
            pt_features_raw,
            pt_traces,
            &proj_cluster,
            dictionary,
            cfg,
            &output_base.join("projected"),
            "projected",
        )?;
    }
    Ok(())
}
