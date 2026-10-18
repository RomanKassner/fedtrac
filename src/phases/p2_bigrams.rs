//! Phase 2: trace feature extraction. The pair-based encoders wrap traces with
//! START/END markers; the twins live in `mp-spdz/mpc_utils/encoding_*.py`.

use std::collections::HashMap;

use crate::types::{BigramIndex, EncodedTrace, FeatureMatrix, END, START};

pub fn extract_plaintext(traces: &[(String, EncodedTrace)]) -> (BigramIndex, FeatureMatrix) {
    let mut bigram_counts: HashMap<(u32, u32), usize> = HashMap::new();

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        if activities.is_empty() {
            continue;
        }
        let extended: Vec<u32> = std::iter::once(START)
            .chain(activities.iter().copied())
            .chain(std::iter::once(END))
            .collect();
        for w in extended.windows(2) {
            *bigram_counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
    }

    let mut pairs: Vec<(u32, u32)> = bigram_counts.keys().cloned().collect();
    pairs.sort();
    let index: BigramIndex = pairs.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    let dim = index.len();
    let mut features: FeatureMatrix = Vec::with_capacity(traces.len());

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        let mut vec = vec![0.0f64; dim];
        if !activities.is_empty() {
            let extended: Vec<u32> = std::iter::once(START)
                .chain(activities.iter().copied())
                .chain(std::iter::once(END))
                .collect();
            for w in extended.windows(2) {
                if let Some(&i) = index.get(&(w[0], w[1])) {
                    vec[i] += 1.0;
                }
            }
        }
        features.push(vec);
    }

    (index, features)
}

/// Scatter the sparse-indexed bigram matrix into dense `d²` rows at
/// `a * dict_size + b`: the layout the extract programs persist at offset 0.
pub fn dense_bigram_rows(
    raw: &FeatureMatrix,
    index: &BigramIndex,
    dict_size: usize,
) -> FeatureMatrix {
    let mut cols = vec![usize::MAX; index.len()];
    for (&(a, b), &l) in index.iter() {
        let (a, b) = (a as usize, b as usize);
        if a < dict_size && b < dict_size {
            cols[l] = a * dict_size + b;
        }
    }
    raw.iter()
        .map(|row| {
            let mut dense = vec![0.0f64; dict_size * dict_size];
            for (l, &v) in row.iter().enumerate() {
                if let Some(&c) = cols.get(l) {
                    if c != usize::MAX {
                        dense[c] += v;
                    }
                }
            }
            dense
        })
        .collect()
}

/// Bag-of-activities (Song et al. 2008); START/END are not counted.
pub fn extract_activity(traces: &[(String, EncodedTrace)], d: usize) -> FeatureMatrix {
    let mut features: FeatureMatrix = Vec::with_capacity(traces.len());
    for (_case_id, trace) in traces.iter() {
        let mut vec = vec![0.0f64; d];
        for ev in trace.iter() {
            let a = ev.activity as usize;
            if a < d {
                vec[a] += 1.0;
            }
        }
        features.push(vec);
    }
    features
}

/// Directly-follows trigrams (a, b, c), with no START/END wrap, as in
/// `encoding_trigram.py`.
pub fn extract_trigram(
    traces: &[(String, EncodedTrace)],
) -> (HashMap<(u32, u32, u32), usize>, FeatureMatrix) {
    let mut trigram_counts: HashMap<(u32, u32, u32), usize> = HashMap::new();

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        for w in activities.windows(3) {
            *trigram_counts.entry((w[0], w[1], w[2])).or_insert(0) += 1;
        }
    }

    let mut triples: Vec<(u32, u32, u32)> = trigram_counts.keys().cloned().collect();
    triples.sort();
    let index: HashMap<(u32, u32, u32), usize> =
        triples.iter().enumerate().map(|(i, &t)| (t, i)).collect();

    let dim = index.len();
    let mut features: FeatureMatrix = Vec::with_capacity(traces.len());

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        let mut vec = vec![0.0f64; dim];
        for w in activities.windows(3) {
            if let Some(&i) = index.get(&(w[0], w[1], w[2])) {
                vec[i] += 1.0;
            }
        }
        features.push(vec);
    }

    (index, features)
}

/// Bigram concatenated with skip-bigram at distance 2: output dim 2·d², the
/// first block direct bigrams, the second pairs two positions apart.
pub fn extract_skip_bigram(traces: &[(String, EncodedTrace)], d: usize) -> FeatureMatrix {
    let block = d * d;
    let dim = 2 * block;
    let mut features: FeatureMatrix = Vec::with_capacity(traces.len());

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        let mut vec = vec![0.0f64; dim];
        if !activities.is_empty() {
            let extended: Vec<u32> = std::iter::once(START)
                .chain(activities.iter().copied())
                .chain(std::iter::once(END))
                .collect();
            for w in extended.windows(2) {
                let a = w[0] as usize;
                let b = w[1] as usize;
                if a < d && b < d {
                    vec[a * d + b] += 1.0;
                }
            }
            for w in extended.windows(3) {
                let a = w[0] as usize;
                let c = w[2] as usize;
                if a < d && c < d {
                    vec[block + a * d + c] += 1.0;
                }
            }
        }
        features.push(vec);
    }

    features
}

/// Eventually-follows: cell (a, b) counts the a-before-b pairs of the trace.
pub fn extract_ef(traces: &[(String, EncodedTrace)], d: usize) -> FeatureMatrix {
    let dim = d * d;
    let mut features: FeatureMatrix = Vec::with_capacity(traces.len());

    for (_case_id, trace) in traces.iter() {
        let activities: Vec<u32> = trace.iter().map(|e| e.activity).collect();
        let mut vec = vec![0.0f64; dim];
        if !activities.is_empty() {
            let extended: Vec<u32> = std::iter::once(START)
                .chain(activities.iter().copied())
                .chain(std::iter::once(END))
                .collect();
            let mut prefix = vec![0.0f64; d];
            for &act in &extended {
                let a = act as usize;
                if a < d {
                    for b in 0..d {
                        vec[b * d + a] += prefix[b];
                    }
                    prefix[a] += 1.0;
                }
            }
        }
        features.push(vec);
    }

    features
}
