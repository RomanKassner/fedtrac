//! Export helpers: features, plaintext results, and MPC results.

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::config::PipelineConfig;
use crate::constants::{MIN_EDGE_COUNT, MIN_EDGE_FRACTION};
use crate::error::{PipelineError, Result};
use crate::evaluation;
use crate::output;
use crate::phases;
use crate::types::{
    ActivityDictionary, BigramIndex, ClusterDFG, ClusterResult, EncodedTrace, FeatureMatrix,
};

pub(crate) struct ExportContext<'a> {
    pub dictionary: &'a ActivityDictionary,
    pub output_dir: &'a Path,
}

pub(crate) fn export_features(features: &FeatureMatrix, output_dir: &Path) -> Result<()> {
    let path = output_dir.join("features.csv");
    let mut f = std::io::BufWriter::new(fs::File::create(&path).map_err(PipelineError::Io)?);
    for row in features {
        let line: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(f, "{}", line.join(",")).map_err(PipelineError::Io)?;
    }
    println!(
        "Exported feature matrix ({} x {}) to {}",
        features.len(),
        features.first().map_or(0, |r| r.len()),
        path.display()
    );
    Ok(())
}

/// The raw encoded feature matrix in the integrated case order, counts as
/// integers, and the case ids beside it: the truth matrix of the attack-battery
/// rescore (`--dump-encoded-features`). No protocol path reads it.
pub(crate) fn export_encoded_features(
    features: &FeatureMatrix,
    traces: &[(String, EncodedTrace)],
    output_dir: &Path,
) -> Result<()> {
    assert_eq!(features.len(), traces.len(), "one encoded row per integrated trace");
    fs::create_dir_all(output_dir).map_err(PipelineError::Io)?;
    let path = output_dir.join("features_encoded.csv");
    let mut f = std::io::BufWriter::new(fs::File::create(&path).map_err(PipelineError::Io)?);
    for row in features {
        let line: Vec<String> = row
            .iter()
            .map(|v| if v.fract() == 0.0 { format!("{}", *v as i64) } else { format!("{:.6}", v) })
            .collect();
        writeln!(f, "{}", line.join(",")).map_err(PipelineError::Io)?;
    }
    let ids = output_dir.join("features_encoded_cases.txt");
    let mut g = std::io::BufWriter::new(fs::File::create(&ids).map_err(PipelineError::Io)?);
    for (case_id, _) in traces {
        writeln!(g, "{}", case_id).map_err(PipelineError::Io)?;
    }
    println!(
        "Exported encoded feature matrix ({} x {}) to {} (case ids in {})",
        features.len(),
        features.first().map_or(0, |r| r.len()),
        path.display(),
        ids.display()
    );
    Ok(())
}

fn export_common_artifacts(dfgs: &[ClusterDFG], ctx: &ExportContext) -> Result<()> {
    output::clean_output_dir(ctx.output_dir)?;
    phases::p4_aggregate::export_dfg_to_file(ctx.output_dir, dfgs, ctx.dictionary)?;
    println!("Exported {} DFGs to {}", dfgs.len(), ctx.output_dir.display());
    Ok(())
}

fn export_traces_and_assignments(
    traces: &[(String, EncodedTrace)],
    assignments_data: &[(String, usize)],
    k: usize,
    ctx: &ExportContext,
) -> Result<()> {
    if traces.is_empty() {
        return Ok(());
    }
    evaluation::trace_eval::export_cluster_traces(
        traces,
        assignments_data,
        ctx.dictionary,
        k,
        ctx.output_dir,
    )?;
    evaluation::trace_eval::export_assignments(assignments_data, ctx.output_dir)?;
    Ok(())
}

/// Export results for a plaintext tier; `tier` is the evaluation label.
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_plaintext_results(
    bigram_index: &BigramIndex,
    features: &FeatureMatrix,
    traces: &[(String, EncodedTrace)],
    cluster_result: &ClusterResult,
    dictionary: &ActivityDictionary,
    cfg: &PipelineConfig,
    output_dir: &Path,
    tier: &str,
) -> Result<()> {
    println!("\n--- Phase 4: Build Cluster DFGs ---");
    let mut dfgs = phases::p4_aggregate::build_cluster_dfgs(bigram_index, features, cluster_result);

    let k = cluster_result.representatives.len();

    // The baseline tiers apply the same public release rule as the federated
    // export (the edge filter, then the dead-end pruning), so a tier delta
    // measures the secure pipeline and not the rule.
    phases::p4_aggregate::filter_dfg_edges(&mut dfgs, MIN_EDGE_COUNT, MIN_EDGE_FRACTION);
    phases::p4_aggregate::prune_dead_ends(&mut dfgs);

    let ctx = ExportContext { dictionary, output_dir };
    export_common_artifacts(&dfgs, &ctx)?;

    let assignments: Vec<(String, usize)> = traces
        .iter()
        .zip(&cluster_result.assignments)
        .map(|((id, _), &c)| (id.clone(), c))
        .collect();
    export_traces_and_assignments(traces, &assignments, k, &ctx)?;

    if cfg.evaluate_full {
        export_features(features, output_dir)?;
        println!("\n--- Evaluation: {} ---", tier);
        evaluation::process_metrics::run_evaluate_all(output_dir, tier, cfg.fast_eval)?;
    }
    Ok(())
}

/// Export results for the MPC path. `secure_suppress` skips the per-trace
/// assignment file when labels stayed secret all the way through Round 2
/// (the `--secure-kmeans` path without `--evaluate-full` label reveal).
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_mpc_results(
    output_dir: &Path,
    dfgs: &[ClusterDFG],
    dictionary: &ActivityDictionary,
    pt_traces: &[(String, EncodedTrace)],
    kmeans_assignments: &[usize],
    k: usize,
    secure_suppress: bool,
) -> Result<()> {
    let ctx = ExportContext { dictionary, output_dir };
    export_common_artifacts(dfgs, &ctx)?;

    if secure_suppress {
        println!("[Secure] Suppressed per-trace cluster_assignments (labels are secret-shared)");
    } else {
        let assignments: Vec<(String, usize)> = pt_traces
            .iter()
            .zip(kmeans_assignments.iter())
            .map(|((id, _), &c)| (id.clone(), c))
            .collect();
        export_traces_and_assignments(pt_traces, &assignments, k, &ctx)?;
    }

    Ok(())
}
