//! Leaf helpers: XES loading and the L2 row normalisation every clustering path
//! applies before distances.

use crate::error::{PipelineError, Result};
use crate::types::{FeatureMatrix, RawEvent};
use process_mining::event_log::XESEditableAttribute;
use process_mining::{import_xes_file, XESImportOptions};
use std::path::Path;

pub fn l2_normalize(features: &FeatureMatrix) -> FeatureMatrix {
    features
        .iter()
        .map(|row| {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-9 {
                row.iter().map(|v| v / norm).collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

pub fn load_xes(path: &Path) -> Result<Vec<RawEvent>> {
    let log = import_xes_file(path, XESImportOptions::default()).map_err(|e| {
        PipelineError::Data(format!("Failed to load XES {}: {}", path.display(), e))
    })?;
    let mut events = Vec::new();

    for (t, trace) in log.traces.into_iter().enumerate() {
        // A defaulted "" case id would collapse every such trace into one case.
        let case_id = trace
            .attributes
            .get_by_key("concept:name")
            .and_then(|a| a.value.try_as_string().cloned())
            .ok_or_else(|| {
                PipelineError::Data(format!(
                    "trace {} in {} has no concept:name case id",
                    t,
                    path.display()
                ))
            })?;
        if case_id.is_empty() {
            return Err(PipelineError::Data(format!(
                "trace {} in {} has an empty concept:name case id",
                t,
                path.display()
            )));
        }

        for (i, event) in trace.events.into_iter().enumerate() {
            let activity = event
                .attributes
                .get_by_key("concept:name")
                .and_then(|a| a.value.try_as_string().cloned())
                .ok_or_else(|| {
                    PipelineError::Data(format!(
                        "event {} of case '{}' in {} has no concept:name activity",
                        i,
                        case_id,
                        path.display()
                    ))
                })?;

            let timestamp = event
                .attributes
                .get_by_key("time:timestamp")
                .and_then(|a| a.value.try_as_date())
                // Pre-epoch timestamps must not wrap into huge u64 sort keys.
                .map(|dt| dt.timestamp_millis().max(0) as u64)
                .unwrap_or(0);

            events.push(RawEvent { case_id: case_id.clone(), activity, timestamp });
        }
    }

    Ok(events)
}
