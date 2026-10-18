//! Clustering, k-selection and the small-cluster merge.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config::{ClusteringAlgo, PipelineConfig};
use crate::constants::*;
use crate::error::{PipelineError, Result};
use crate::evaluation;
use crate::phases;
use crate::types::{ClusterResult, FeatureMatrix, Representative};

/// Cluster `features`. `dfg_rows` carries the dense d² directly-follows rows of
/// the SAME cases in the SAME order — the only input `ClusteringAlgo::Sequence`
/// scores; paths that cannot build them pass `None`.
pub(crate) fn run_clustering(
    features: &FeatureMatrix,
    cfg: &PipelineConfig,
    trace_len: Option<usize>,
    work_dir: &Path,
    dfg_rows: Option<&FeatureMatrix>,
) -> Result<ClusterResult> {
    let k = match cfg.k_fixed() {
        Some(k) => k,
        None => select_k_auto(features, trace_len, work_dir)?,
    };
    if k == 0 {
        return Err(PipelineError::Clustering("k must be at least 1".into()));
    }

    match cfg.clustering_algo {
        ClusteringAlgo::Kmeans => {
            let r = phases::p3_clustering::cluster_n_init(features, k, MAX_ITERATIONS, N_INIT)?;
            println!(
                "Clustered into {} groups (kmeans, k={}, max_iter={}, n_init={})",
                r.representatives.len(),
                k,
                MAX_ITERATIONS,
                N_INIT
            );
            Ok(r)
        }
        ClusteringAlgo::Sequence => {
            let rows = dfg_rows.ok_or_else(|| {
                PipelineError::Clustering(
                    "sequence clustering needs the dense d² directly-follows rows; \
                     this path does not build them"
                        .into(),
                )
            })?;
            run_sequence(features, rows, k, work_dir)
        }
    }
}

/// Feature CSV handed to a Python subprocess, removed on drop. Lives under the
/// per-run output dir: a fixed temp path would let concurrent runs read each
/// other's features.
struct TempFeatureCsv {
    path: PathBuf,
}

impl TempFeatureCsv {
    fn write(dir: &Path, name: &str, features: &FeatureMatrix) -> Result<Self> {
        fs::create_dir_all(dir).map_err(PipelineError::Io)?;
        let csv = TempFeatureCsv { path: dir.join(name) };
        let mut f =
            std::io::BufWriter::new(fs::File::create(&csv.path).map_err(PipelineError::Io)?);
        for row in features {
            let line: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(f, "{}", line.join(",")).map_err(PipelineError::Io)?;
        }
        f.flush().map_err(PipelineError::Io)?;
        Ok(csv)
    }

    /// One integer per line — the `--init-labels` partition `bridge/cluster.py` reads.
    fn write_labels(dir: &Path, name: &str, labels: &[usize]) -> Result<Self> {
        fs::create_dir_all(dir).map_err(PipelineError::Io)?;
        let f = TempFeatureCsv { path: dir.join(name) };
        let mut w = std::io::BufWriter::new(fs::File::create(&f.path).map_err(PipelineError::Io)?);
        for &c in labels {
            writeln!(w, "{}", c).map_err(PipelineError::Io)?;
        }
        w.flush().map_err(PipelineError::Io)?;
        Ok(f)
    }
}

impl Drop for TempFeatureCsv {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Per-cluster means of `features` (zero vector for an empty cluster) — the
/// representatives every downstream consumer expects in FEATURE space, which
/// the sequence arm cannot take from the bridge (it clusters the count rows).
fn feature_means(features: &FeatureMatrix, assignments: &[usize], k: usize) -> Vec<Representative> {
    let dim = features.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; dim]; k];
    let mut counts = vec![0usize; k];
    for (row, &c) in features.iter().zip(assignments) {
        if c < k {
            counts[c] += 1;
            for (s, v) in sums[c].iter_mut().zip(row) {
                *s += v;
            }
        }
    }
    sums.into_iter()
        .zip(counts)
        .map(|(mut sum, n)| {
            if n > 0 {
                for v in sum.iter_mut() {
                    *v /= n as f64;
                }
            }
            Representative { centroid: sum }
        })
        .collect()
}

/// The plaintext twin of `--secure-clustering sequence`: k-means seeds the
/// partition, then `bridge/cluster.py --algo sequence` runs the hard-EM.
fn run_sequence(
    features: &FeatureMatrix,
    dfg_rows: &FeatureMatrix,
    k: usize,
    work_dir: &Path,
) -> Result<ClusterResult> {
    if dfg_rows.len() != features.len() {
        return Err(PipelineError::Clustering(format!(
            "sequence clustering: {} count rows for {} feature rows",
            dfg_rows.len(),
            features.len()
        )));
    }
    let width = dfg_rows.first().map(|r| r.len()).unwrap_or(0);
    let dh = (width as f64).sqrt().round() as usize;
    if dh * dh != width {
        return Err(PipelineError::Clustering(format!(
            "sequence clustering needs d² directly-follows rows; got width {}",
            width
        )));
    }

    let seed = phases::p3_clustering::cluster_n_init(features, k, MAX_ITERATIONS, N_INIT)?;
    let init =
        TempFeatureCsv::write_labels(work_dir, "sequence_init_labels.txt", &seed.assignments)?;
    let rows = TempFeatureCsv::write(work_dir, "sequence_dfg_rows.csv", dfg_rows)?;
    let out = evaluation::process_metrics::run_cluster_dispatch_sequence(
        &rows.path,
        k,
        dh,
        SECURE_SEQUENCE_MAX_ITER,
        &init.path,
        features.len(),
    )?;
    drop(rows);
    drop(init);

    println!(
        "Clustered into {} groups (sequence, k={}, max_iter={})",
        out.k_actual, k, SECURE_SEQUENCE_MAX_ITER
    );
    let representatives = feature_means(features, &out.assignments, out.k_actual);
    Ok(ClusterResult { assignments: out.assignments, representatives })
}

// Prediction-strength k selection via scripts/k_select.py. The leakage depends
// on the caller's features: the rsec path passes the revealed backbone rows,
// which are envelope data; --secure-kmeans rejects auto-k upstream.
pub(crate) fn select_k_auto(
    features: &FeatureMatrix,
    trace_len: Option<usize>,
    work_dir: &Path,
) -> Result<usize> {
    let csv = TempFeatureCsv::write(work_dir, "auto_k_features.csv", features)?;
    println!("[Auto-k] Running k_select.py (n={})...", features.len());
    evaluation::process_metrics::run_k_select(&csv.path, K_SELECT_K_MIN, trace_len)
}

pub(crate) fn merge_small_clusters(
    assignments: &mut [usize],
    representatives: &[Representative],
    min_cluster_size: usize,
) -> usize {
    let k = representatives.len();
    let mut cluster_sizes = vec![0usize; k];
    for &c in assignments.iter() {
        if c < k {
            cluster_sizes[c] += 1;
        }
    }

    let small_clusters: Vec<usize> = cluster_sizes
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0 && s < min_cluster_size)
        .map(|(i, _)| i)
        .collect();

    if small_clusters.is_empty() {
        return 0;
    }

    println!(
        "[Merge] {} clusters below {} traces: {:?}",
        small_clusters.len(),
        min_cluster_size,
        small_clusters.iter().map(|&c| (c, cluster_sizes[c])).collect::<Vec<_>>()
    );

    let large_clusters: Vec<usize> =
        (0..k).filter(|&c| cluster_sizes[c] >= min_cluster_size).collect();
    if large_clusters.is_empty() {
        println!("[Merge] All clusters below minimum size, proceeding anyway.");
        return 0;
    }

    let mut merges = 0usize;
    for &small_c in &small_clusters {
        let sq_dist_to_small = |c: usize| -> f64 {
            representatives[small_c]
                .centroid
                .iter()
                .zip(&representatives[c].centroid)
                .map(|(x, y)| (x - y).powi(2))
                .sum()
        };
        let nearest = large_clusters
            .iter()
            .copied()
            .min_by(|&a, &b| sq_dist_to_small(a).partial_cmp(&sq_dist_to_small(b)).unwrap())
            .unwrap();
        let old_size = cluster_sizes[small_c];
        for a in assignments.iter_mut() {
            if *a == small_c {
                *a = nearest;
            }
        }
        cluster_sizes[nearest] += old_size;
        cluster_sizes[small_c] = 0;
        println!("[Merge] Merged cluster {} ({} traces) -> cluster {}", small_c, old_size, nearest);
        merges += 1;
    }
    merges
}
