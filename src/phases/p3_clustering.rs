//! Phase 3: k-means over feature vectors, seeded so a run is reproducible.

use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::error::{PipelineError, Result};
use crate::types::{ClusterResult, FeatureVector, Representative};

/// Run k-means with seeds `0..n_init` and keep the lowest-WCSS result.
pub fn cluster_n_init(
    data: &[FeatureVector],
    k: usize,
    iters: usize,
    n_init: usize,
) -> Result<ClusterResult> {
    let mut best_result: Option<ClusterResult> = None;
    let mut best_inertia = f64::INFINITY;

    for seed in 0..n_init {
        let result = run_kmeans_impl(data, k, iters, Some(seed as u64))?;
        let inertia = compute_inertia(data, &result);
        if inertia < best_inertia {
            best_inertia = inertia;
            best_result = Some(result);
        }
    }

    match best_result {
        Some(r) => Ok(r),
        None => run_kmeans_impl(data, k, iters, None),
    }
}

fn compute_inertia(data: &[FeatureVector], result: &ClusterResult) -> f64 {
    let mut total = 0.0;
    for (i, point) in data.iter().enumerate() {
        let cluster = result.assignments[i];
        if cluster < result.representatives.len() {
            let centroid = &result.representatives[cluster].centroid;
            let dist_sq: f64 =
                point.iter().zip(centroid.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            total += dist_sq;
        }
    }
    total
}

fn run_kmeans_impl(
    data: &[FeatureVector],
    k: usize,
    iters: usize,
    seed: Option<u64>,
) -> Result<ClusterResult> {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { 0 };
    if n == 0 || d == 0 || k == 0 {
        return Ok(ClusterResult { assignments: vec![], representatives: vec![] });
    }
    if k > n {
        return Err(PipelineError::Clustering(format!(
            "k = {} exceeds the {} rows to cluster; lower --k",
            k, n
        )));
    }

    let flat: Vec<f64> = data.iter().flat_map(|v| v.iter().copied()).collect();
    let array = Array2::from_shape_vec((n, d), flat)
        .map_err(|e| PipelineError::Clustering(format!("feature matrix {}x{}: {}", n, d, e)))?;
    let dataset = Dataset::from(array.clone());

    // n_runs(1): the seeded outer loop is what makes a run reproducible, and
    // linfa's own restart loop would multiply it.
    let fit_err = |e| PipelineError::Clustering(format!("k-means (k={}, n={}): {}", k, n, e));
    let model = if let Some(s) = seed {
        let rng = Xoshiro256Plus::seed_from_u64(s);
        KMeans::params_with_rng(k, rng)
            .n_runs(1)
            .max_n_iterations(iters as u64)
            .fit(&dataset)
            .map_err(fit_err)?
    } else {
        KMeans::params(k).n_runs(1).max_n_iterations(iters as u64).fit(&dataset).map_err(fit_err)?
    };

    let predictions = model.predict(&array);
    let assignments: Vec<usize> = predictions.iter().copied().collect();

    let representatives: Vec<Representative> = model
        .centroids()
        .rows()
        .into_iter()
        .map(|row| Representative { centroid: row.to_vec() })
        .collect();

    Ok(ClusterResult { assignments, representatives })
}
