//! Data loading, case intersection, and trace integration.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::config::PipelineConfig;
use crate::constants::MS_TO_SECS;
use crate::error::{PipelineError, Result};
use crate::mpc::CaseData;
use crate::phases;
use crate::types::{
    ActivityDictionary, BigramIndex, EncodedEvent, EncodedTrace, FeatureMatrix, PartyState,
};
pub(crate) fn load_data(cfg: &PipelineConfig) -> Result<Vec<PartyState>> {
    if cfg.n_parties > 2 {
        // party_{i}.xes preferred; otherwise sorted glob.
        let dir = Path::new(&cfg.data_dir);
        let numbered: Vec<std::path::PathBuf> =
            (0..cfg.n_parties).map(|i| dir.join(format!("party_{}.xes", i))).collect();
        let paths = if numbered.iter().all(|p| p.exists()) {
            numbered
        } else {
            let mut candidates: Vec<std::path::PathBuf> = fs::read_dir(dir)
                .map_err(PipelineError::Io)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|ext| ext == "xes"))
                .collect();
            candidates.sort();
            if candidates.len() < cfg.n_parties {
                return Err(PipelineError::Data(format!(
                    "Need {} XES files in {}, found {}",
                    cfg.n_parties,
                    dir.display(),
                    candidates.len()
                )));
            }
            candidates.truncate(cfg.n_parties);
            candidates
        };
        let mut states = Vec::new();
        for (i, path) in paths.iter().enumerate() {
            let s = PartyState::from_xes(path)?;
            println!(
                "Loaded party {}: {} events ({})",
                i,
                s.events.len(),
                path.file_name().unwrap().to_string_lossy()
            );
            states.push(s);
        }
        println!();
        Ok(states)
    } else {
        // Party_A/Party_B preferred; otherwise sorted glob (org-derived
        // splits use descriptive names like Party_Support/Party_QA).
        let dir = Path::new(&cfg.data_dir);
        let (path_a, path_b) = {
            let a = dir.join("Party_A.xes");
            let b = dir.join("Party_B.xes");
            if a.exists() && b.exists() {
                (a, b)
            } else {
                let mut candidates: Vec<std::path::PathBuf> = fs::read_dir(dir)
                    .map_err(PipelineError::Io)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|ext| ext == "xes"))
                    .collect();
                candidates.sort();
                if candidates.len() < 2 {
                    return Err(PipelineError::Data(format!(
                        "Need 2 XES files in {}, found {}",
                        dir.display(),
                        candidates.len()
                    )));
                }
                (candidates[0].clone(), candidates[1].clone())
            }
        };
        let a = PartyState::from_xes(&path_a)?;
        let b = PartyState::from_xes(&path_b)?;
        println!(
            "Loaded XES: {} events ({}), {} events ({}) from {}\n",
            a.events.len(),
            path_a.file_name().unwrap().to_string_lossy(),
            b.events.len(),
            path_b.file_name().unwrap().to_string_lossy(),
            cfg.data_dir
        );
        Ok(vec![a, b])
    }
}

// LEAKAGE (simulator only): plaintext case-id intersection. Real deployment
// would use PSI or out-of-band agreement.
pub(crate) fn intersect_all_cases(states: &[PartyState]) -> HashSet<String> {
    let mut common = states[0].case_ids();
    for s in &states[1..] {
        let ids = s.case_ids();
        common = common.intersection(&ids).cloned().collect();
    }
    common
}

pub(crate) fn prepare_cases_for_mpc(
    state: &PartyState,
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
    max_trace_len: Option<usize>,
) -> Vec<CaseData> {
    // One CaseData per AGREED case, sorted by id, empty where this party holds
    // nothing: the per-party vectors are index-aligned (mpc::slab_shape).
    let mut by_case: HashMap<String, Vec<(u64, u32)>> =
        common_cases.iter().map(|id| (id.clone(), Vec::new())).collect();
    for e in state.filter_to_cases(common_cases) {
        if let Some(act) = dictionary.encode_activity(&e.activity) {
            by_case.entry(e.case_id).or_default().push((e.timestamp, act));
        }
    }
    for events in by_case.values_mut() {
        // Must match the MPC merge tree's composite key, built from the SCALED
        // timestamp `ts / MS_TO_SECS`: sorting on raw ms would leave a party's
        // own input unsorted under that key on a sub-second log.
        events.sort_by_key(|(ts, act)| (*ts / MS_TO_SECS, *act));
        if let Some(max_len) = max_trace_len {
            events.truncate(max_len);
        }
    }
    let mut cases: Vec<_> =
        by_case.into_iter().map(|(id, events)| CaseData { id, events }).collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    cases
}

/// Re-merge the per-party truncated MPC inputs in the merge tree's order
/// (`(ts / MS_TO_SECS, act)`), so the suppression guard's profile classes match
/// what the MPC fingerprints after truncation.
// LEAKAGE (simulator only): same class as integrate_traces.
pub(crate) fn merge_truncated_cases(all_cases: &[Vec<CaseData>]) -> Vec<(String, EncodedTrace)> {
    let mut by_case: BTreeMap<String, Vec<(u64, u32)>> = BTreeMap::new();
    for party in all_cases {
        for case in party {
            by_case.entry(case.id.clone()).or_default().extend(case.events.iter().copied());
        }
    }
    by_case
        .into_iter()
        .map(|(id, mut events)| {
            events.sort_by_key(|(ts, act)| (*ts / MS_TO_SECS, *act));
            let trace: EncodedTrace = events
                .into_iter()
                .map(|(ts, act)| EncodedEvent { case_id: id.clone(), activity: act, timestamp: ts })
                .collect();
            (id, trace)
        })
        .collect()
}

pub(crate) struct IntegratedTraces {
    pub traces: Vec<(String, EncodedTrace)>,
    pub bigram_index: BigramIndex,
    pub features: FeatureMatrix,
}

// LEAKAGE (simulator only): merges every party's events in plaintext for the
// --evaluate-full baseline. A real federation cannot assemble this.
pub(crate) fn integrate_traces(
    states: &[&PartyState],
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
) -> Result<IntegratedTraces> {
    let mut all_events: Vec<EncodedEvent> = Vec::new();
    for state in states {
        for e in state.filter_to_cases(common_cases) {
            if let Some(act) = dictionary.encode_activity(&e.activity) {
                all_events.push(EncodedEvent {
                    case_id: e.case_id,
                    activity: act,
                    timestamp: e.timestamp,
                });
            }
        }
    }
    // Mirrors the MPC merge composite key exactly, so same-second events order
    // identically across the tiers and the gap carries no ordering artefact.
    all_events.sort_by(|a, b| {
        a.case_id
            .cmp(&b.case_id)
            .then((a.timestamp / MS_TO_SECS).cmp(&(b.timestamp / MS_TO_SECS)))
            .then(a.activity.cmp(&b.activity))
    });

    let mut trace_map: BTreeMap<String, EncodedTrace> = BTreeMap::new();
    for e in all_events {
        trace_map.entry(e.case_id.clone()).or_default().push(e);
    }
    let traces: Vec<_> = trace_map.into_iter().collect();
    println!("[Plaintext] Integrated {} traces", traces.len());

    let (bigram_index, features) = phases::p2_bigrams::extract_plaintext(&traces);
    println!("[Plaintext] {} bigrams, {} features", bigram_index.len(), features.len());
    Ok(IntegratedTraces { traces, bigram_index, features })
}

/// `--truncate-baseline`: the baseline consumes the SAME per-party truncated
/// input as the MPC path (`prepare_cases_for_mpc` at L, merge-tree order),
/// isolating the truncation term of the tier decomposition.
pub(crate) fn integrate_traces_truncated(
    states: &[&PartyState],
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
    max_trace_len: usize,
) -> Result<IntegratedTraces> {
    let all_cases: Vec<Vec<CaseData>> = states
        .iter()
        .map(|s| prepare_cases_for_mpc(s, common_cases, dictionary, Some(max_trace_len)))
        .collect();
    let traces = merge_truncated_cases(&all_cases);
    println!("[Plaintext] Integrated {} traces (truncated, L={})", traces.len(), max_trace_len);
    let (bigram_index, features) = phases::p2_bigrams::extract_plaintext(&traces);
    println!("[Plaintext] {} bigrams, {} features", bigram_index.len(), features.len());
    Ok(IntegratedTraces { traces, bigram_index, features })
}
