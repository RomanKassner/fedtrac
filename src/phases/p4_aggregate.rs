//! Phase 4: Cluster DFG construction and export.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::types::{
    ActivityDictionary, BigramIndex, ClusterDFG, ClusterDFGExport, ClusterResult, DFGEdgeExport,
    FeatureMatrix, END, START,
};

pub fn build_cluster_dfgs(
    bigram_index: &BigramIndex,
    features: &FeatureMatrix,
    cluster_result: &ClusterResult,
) -> Vec<ClusterDFG> {
    let k = cluster_result.representatives.len();
    let mut dfgs = Vec::with_capacity(k);

    let reverse_index: HashMap<usize, (u32, u32)> =
        bigram_index.iter().map(|(&pair, &idx)| (idx, pair)).collect();

    for cluster_id in 0..k {
        let member_indices: Vec<usize> = cluster_result
            .assignments
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster_id)
            .map(|(i, _)| i)
            .collect();

        let size = member_indices.len();
        let dim = if !features.is_empty() { features[0].len() } else { 0 };
        let mut aggregated = vec![0.0; dim];

        for &trace_idx in &member_indices {
            if trace_idx < features.len() {
                for (i, &v) in features[trace_idx].iter().enumerate() {
                    aggregated[i] += v;
                }
            }
        }

        let mut edges: HashMap<(u32, u32), f64> = HashMap::new();
        for (idx, &count) in aggregated.iter().enumerate() {
            if count > 0.0 {
                if let Some(&pair) = reverse_index.get(&idx) {
                    edges.insert(pair, count);
                }
            }
        }

        dfgs.push(ClusterDFG { cluster_id, size, edges });
    }

    dfgs
}

/// MPC path. Reads the aggregated d² cells (pure counts) into edges.
pub fn build_dfgs_from_dense(
    aggregated: &[Vec<f64>],
    dict_size: usize,
    cluster_sizes: &[usize],
) -> Vec<ClusterDFG> {
    let mut dfgs = Vec::with_capacity(aggregated.len());

    for (cluster_id, row) in aggregated.iter().enumerate() {
        let size = cluster_sizes.get(cluster_id).copied().unwrap_or(0);
        let mut edges: HashMap<(u32, u32), f64> = HashMap::new();

        for (flat_idx, &count) in row.iter().enumerate() {
            if count == 0.0 {
                continue;
            }
            let from_act = (flat_idx / dict_size) as u32;
            let to_act = (flat_idx % dict_size) as u32;
            if from_act == 0 || to_act == 0 {
                continue;
            }
            edges.insert((from_act, to_act), count);
        }

        dfgs.push(ClusterDFG { cluster_id, size, edges });
    }

    dfgs
}

/// Drop edges below the cutoff; START/END edges are always kept. The predicate
/// uses the exact integer form of the in-MPC pre-reveal pass (`w >= mc` AND
/// `w * 10^4 >= frac_num * max`) whenever the cutoff has one, so a cell exactly
/// on the rational cutoff cannot be zeroed in-MPC and kept in the clear.
pub fn filter_dfg_edges(dfgs: &mut [ClusterDFG], min_count: f64, min_fraction: f64) {
    let int_args = crate::mpc::rounds::edge_filter_int_args(min_count, min_fraction);
    for dfg in dfgs.iter_mut() {
        let weights = dfg
            .edges
            .iter()
            .filter(|&(&(from, to), _)| from != START && to != END)
            .map(|(_, &w)| w);
        let max_weight = weights.fold(0.0_f64, f64::max);
        let effective = min_count.max(min_fraction * max_weight);
        let keep = |count: f64| match int_args {
            Some((mc, frac_num)) => {
                count >= mc as f64 && count * 10_000.0 >= frac_num as f64 * max_weight
            }
            None => count >= effective,
        };

        let before = dfg.edges.len();
        dfg.edges.retain(|&(from, to), &mut count| from == START || to == END || keep(count));
        let removed = before - dfg.edges.len();
        if removed > 0 {
            println!(
                "  Cluster {}: filtered {} edges below {:.1} (kept {})",
                dfg.cluster_id,
                removed,
                effective,
                dfg.edges.len()
            );
        }
    }
}

/// Drop every activity on no path from a start to an end activity, with its
/// START, END and activity edges. Twin of `secure_aggregate.mpc` Step 3c (which
/// runs it before the open, so on the MPC path this is an idempotent safety
/// net) and of `bridge/utils.py::prune_to_start_end_paths`.
pub fn prune_dead_ends(dfgs: &mut [ClusterDFG]) {
    for dfg in dfgs.iter_mut() {
        let mut succ: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut pred: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut starts: Vec<u32> = Vec::new();
        let mut ends: Vec<u32> = Vec::new();
        for (&(from, to), &w) in &dfg.edges {
            if w <= 0.0 || (from == START && to == END) {
                continue;
            }
            if from == START {
                starts.push(to);
            } else if to == END {
                ends.push(from);
            } else {
                succ.entry(from).or_default().push(to);
                pred.entry(to).or_default().push(from);
            }
        }
        let reach = |seed: &[u32], step: &HashMap<u32, Vec<u32>>| -> HashSet<u32> {
            let mut seen: HashSet<u32> = seed.iter().copied().collect();
            let mut todo: Vec<u32> = seed.to_vec();
            while let Some(x) = todo.pop() {
                for &y in step.get(&x).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if seen.insert(y) {
                        todo.push(y);
                    }
                }
            }
            seen
        };
        let fwd = reach(&starts, &succ);
        let bwd = reach(&ends, &pred);
        let keep = |a: u32| fwd.contains(&a) && bwd.contains(&a);

        let before = dfg.edges.len();
        dfg.edges.retain(|&(from, to), _| match (from == START, to == END) {
            (true, true) => true,
            (true, false) => keep(to),
            (false, true) => keep(from),
            (false, false) => keep(from) && keep(to),
        });
        let removed = before - dfg.edges.len();
        if removed > 0 {
            println!(
                "  Cluster {}: pruned {} edges of activities on no start-to-end path (kept {})",
                dfg.cluster_id,
                removed,
                dfg.edges.len()
            );
        }
    }
}

fn dfg_to_export(dfg: &ClusterDFG, dictionary: &ActivityDictionary) -> ClusterDFGExport {
    let mut activities_set: HashSet<String> = HashSet::new();
    let mut edges: Vec<DFGEdgeExport> = Vec::new();
    let mut start_activities: std::collections::BTreeMap<String, f64> = Default::default();
    let mut end_activities: std::collections::BTreeMap<String, f64> = Default::default();

    for (&(from, to), &weight) in &dfg.edges {
        let from_name = dictionary.decode_activity(from).unwrap_or("?").to_string();
        let to_name = dictionary.decode_activity(to).unwrap_or("?").to_string();

        if from == START {
            start_activities.insert(to_name.clone(), weight);
            activities_set.insert(to_name.clone());
        } else if to == END {
            end_activities.insert(from_name.clone(), weight);
            activities_set.insert(from_name.clone());
        } else {
            activities_set.insert(from_name.clone());
            activities_set.insert(to_name.clone());
        }

        edges.push(DFGEdgeExport { from: from_name, to: to_name, weight });
    }

    edges.sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));

    let mut activities: Vec<String> = activities_set.into_iter().collect();
    activities.sort();

    ClusterDFGExport {
        cluster_id: dfg.cluster_id,
        num_traces: dfg.size,
        activities,
        edges,
        start_activities,
        end_activities,
    }
}

pub fn export_dfg_to_file(
    dir: &Path,
    dfgs: &[ClusterDFG],
    dictionary: &ActivityDictionary,
) -> crate::error::Result<()> {
    std::fs::create_dir_all(dir)?;
    for dfg in dfgs {
        let export = dfg_to_export(dfg, dictionary);
        let json = serde_json::to_string_pretty(&export)
            .map_err(|e| crate::error::PipelineError::Data(e.to_string()))?;
        let path = dir.join(format!("cluster_{}_dfg.json", dfg.cluster_id));
        std::fs::write(&path, json)?;
        println!("  Written: {}", path.display());
    }
    Ok(())
}
