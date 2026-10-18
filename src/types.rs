//! Shared types. Activities flow [`RawEvent`] (XES `String`) ->
//! [`EncodedEvent`] (dictionary `u32`) -> [`EncodedTrace`].

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Serialize;

use crate::error::Result;
use crate::utils::load_xes;

// Reserved activity ids: PAD is an implicit 0, then START, END and the user
// activities upward. Twins of mpc_utils/constants.py::START_ACT/END_ACT.
pub const START: u32 = 1;
pub const END: u32 = 2;
pub const START_SYMBOL: &str = ">";
pub const END_SYMBOL: &str = "|";
pub const FIRST_USER_ACTIVITY: u32 = 3;
// mpc_utils/encoding_trigram.py derives FIRST_USER_ACT = END_ACT + 1.
const _: () = assert!(FIRST_USER_ACTIVITY == END + 1);
/// Names a log activity may not carry — every DFG consumer filters these out.
pub const RESERVED_ACTIVITY_NAMES: [&str; 4] = [START_SYMBOL, END_SYMBOL, "?", ""];

#[derive(Debug, Clone)]
pub struct RawEvent {
    pub case_id: String,
    pub activity: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct EncodedEvent {
    pub case_id: String,
    pub activity: u32,
    pub timestamp: u64,
}

pub type EncodedTrace = Vec<EncodedEvent>;
pub type FeatureVector = Vec<f64>;
pub type BigramIndex = HashMap<(u32, u32), usize>;
pub type FeatureMatrix = Vec<FeatureVector>;

#[derive(Debug, Clone)]
pub struct Representative {
    pub centroid: FeatureVector,
}

#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub assignments: Vec<usize>,
    pub representatives: Vec<Representative>,
}

#[derive(Debug, Clone)]
pub struct ClusterDFG {
    pub cluster_id: usize,
    pub size: usize,
    pub edges: HashMap<(u32, u32), f64>,
}

#[derive(Debug, Clone)]
pub struct ActivityDictionary {
    pub encode: HashMap<String, u32>,
    pub decode: Vec<String>,
}

impl ActivityDictionary {
    /// Indices 0=PAD, 1=START, 2=END, 3+=user activities (alphabetical).
    // LEAKAGE (simulator only): the caller passes the union of every party's
    // vocabulary, and the resulting (size, ordering) is a public compile-time
    // constant for the .mpc programs.
    pub fn from_activities(activities: HashSet<String>) -> Result<Self> {
        let mut sorted: Vec<String> = activities.into_iter().collect();
        sorted.sort();
        // Every DFG consumer drops edges touching a marker name.
        if let Some(bad) = sorted.iter().find(|a| RESERVED_ACTIVITY_NAMES.contains(&a.as_str())) {
            return Err(crate::error::PipelineError::Data(format!(
                "activity name {:?} is reserved (DFG markers {:?}); rename it in the log",
                bad, RESERVED_ACTIVITY_NAMES
            )));
        }

        let mut decode = vec![String::new(), START_SYMBOL.to_string(), END_SYMBOL.to_string()];
        decode.extend(sorted.iter().cloned());

        let encode: HashMap<String, u32> = sorted
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), (i + FIRST_USER_ACTIVITY as usize) as u32))
            .collect();

        Ok(ActivityDictionary { encode, decode })
    }

    pub fn encode_activity(&self, activity: &str) -> Option<u32> {
        self.encode.get(activity).copied()
    }

    pub fn decode_activity(&self, code: u32) -> Option<&str> {
        self.decode.get(code as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.decode.len()
    }

    pub fn user_activity_count(&self) -> usize {
        self.decode.len().saturating_sub(FIRST_USER_ACTIVITY as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.user_activity_count() == 0
    }
}

pub struct PartyState {
    pub events: Vec<RawEvent>,
}

impl PartyState {
    pub fn from_xes(path: &Path) -> Result<Self> {
        Ok(PartyState { events: load_xes(path)? })
    }

    pub fn case_ids(&self) -> HashSet<String> {
        self.events.iter().map(|e| e.case_id.clone()).collect()
    }

    pub fn filter_to_cases(&self, cases: &HashSet<String>) -> Vec<RawEvent> {
        self.events.iter().filter(|e| cases.contains(&e.case_id)).cloned().collect()
    }

    pub fn activities_for_cases(&self, cases: &HashSet<String>) -> HashSet<String> {
        self.events
            .iter()
            .filter(|e| cases.contains(&e.case_id))
            .map(|e| e.activity.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterDFGExport {
    pub cluster_id: usize,
    pub num_traces: usize,
    pub activities: Vec<String>,
    pub edges: Vec<DFGEdgeExport>,
    // BTreeMap: the serialized key order feeds pm4py's dict order, and miner
    // tie-breaking follows it, so a HashMap here made F vary run to run.
    pub start_activities: std::collections::BTreeMap<String, f64>,
    pub end_activities: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DFGEdgeExport {
    pub from: String,
    pub to: String,
    pub weight: f64,
}
