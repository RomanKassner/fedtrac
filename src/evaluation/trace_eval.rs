//! Trace assignment and per-cluster trace export.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{PipelineError, Result};
use crate::types::{ActivityDictionary, EncodedTrace};

pub fn export_cluster_traces(
    traces: &[(String, EncodedTrace)],
    assignments: &[(String, usize)],
    dictionary: &ActivityDictionary,
    k: usize,
    dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    let trace_map: HashMap<&str, &EncodedTrace> =
        traces.iter().map(|(id, trace)| (id.as_str(), trace)).collect();

    let mut clusters: Vec<Vec<Vec<String>>> = vec![Vec::new(); k];
    for (case_id, cluster_id) in assignments {
        if *cluster_id >= k {
            continue;
        }
        if let Some(trace) = trace_map.get(case_id.as_str()) {
            let decoded: Vec<String> = trace
                .iter()
                .filter_map(|e| dictionary.decode_activity(e.activity).map(|s| s.to_string()))
                .collect();
            clusters[*cluster_id].push(decoded);
        }
    }

    for (cluster_id, traces) in clusters.iter().enumerate() {
        let obj = serde_json::json!({
            "num_traces": traces.len(),
            "traces": traces,
        });
        let path = dir.join(format!("cluster_{}_traces.json", cluster_id));
        let json =
            serde_json::to_string_pretty(&obj).map_err(|e| PipelineError::Data(e.to_string()))?;
        std::fs::write(&path, json)?;
        println!("  Written: {} ({} traces)", path.display(), traces.len());
    }

    Ok(())
}

pub fn export_assignments(assignments: &[(String, usize)], dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let case_ids: Vec<&str> = assignments.iter().map(|(id, _)| id.as_str()).collect();
    let cluster_ids: Vec<usize> = assignments.iter().map(|(_, c)| *c).collect();
    let obj = serde_json::json!({
        "case_ids": case_ids,
        "assignments": cluster_ids,
    });
    let path = dir.join("cluster_assignments.json");
    let json =
        serde_json::to_string_pretty(&obj).map_err(|e| PipelineError::Data(e.to_string()))?;
    std::fs::write(&path, json)?;
    println!("  Written: {}", path.display());
    Ok(())
}
