//! Adaptive parameter selection: dynamic L, sparsity, k_proj, the ring-width
//! guards and the suppression threshold.

use std::collections::{HashMap, HashSet};

use crate::config::Encoding;
use crate::constants::*;
use crate::error::{PipelineError, Result};
use crate::mpc;
use crate::phases;
use crate::types::{ActivityDictionary, EncodedTrace, FeatureMatrix, FeatureVector, PartyState};

pub(crate) fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// LEAKAGE (simulator only): touches per-party event counts in plaintext. A real
// deployment fixes L by policy; with --max-trace-len the override returns
// before any per-case data is read.
pub(crate) fn compute_dynamic_trace_len(
    states: &[&PartyState],
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
    user_override: Option<usize>,
) -> usize {
    if let Some(l) = user_override {
        println!("[MPC] Using user-specified max_trace_len = {} (policy input; joint length distribution not computed)", l);
        return l;
    }
    println!(
        "[Harness] Auto-L reads the joint per-case length distribution across parties; \
         a deployment fixes --max-trace-len by policy instead"
    );

    let count_per_case = |state: &PartyState| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for e in state.filter_to_cases(common_cases) {
            if dictionary.encode_activity(&e.activity).is_some() {
                *counts.entry(e.case_id).or_default() += 1;
            }
        }
        counts
    };

    let per_party_maps: Vec<HashMap<String, usize>> =
        states.iter().map(|s| count_per_case(s)).collect();
    let case_ids: Vec<&String> = common_cases.iter().collect();
    let party_counts: Vec<Vec<usize>> = per_party_maps
        .iter()
        .map(|m| case_ids.iter().map(|cid| m.get(*cid).copied().unwrap_or(0)).collect())
        .collect();

    let mut merged_lens: Vec<usize> =
        (0..case_ids.len()).map(|c| party_counts.iter().map(|p| p[c]).sum()).collect();
    merged_lens.sort();

    let p50 = percentile(&merged_lens, 50.0);
    let p95 = percentile(&merged_lens, 95.0);
    let p99 = percentile(&merged_lens, 99.0);
    let max_l = merged_lens.last().copied().unwrap_or(0);
    println!(
        "[MPC] Trace length distribution (merged, {} parties): P50={}, P95={}, P99={}, max={}",
        states.len(),
        p50,
        p95,
        p99,
        max_l
    );

    let (per_party, bigram_coverage) = select_l_from_bigram_coverage(&party_counts);
    let truncated = merged_lens.iter().filter(|&&m| m > party_counts.len() * per_party).count();
    let trunc_pct = 100.0 * truncated as f64 / merged_lens.len().max(1) as f64;
    println!(
        "[MPC] Dynamic L = {} (per-party, bigram coverage {:.1}%), truncated = {:.1}%",
        per_party,
        bigram_coverage * 100.0,
        trunc_pct
    );
    per_party
}

/// Smallest per-party L with at least `BIGRAM_COVERAGE_TARGET` bigram coverage
/// after truncation, aligned up to a multiple of four and at least eight.
// LEAKAGE: L summarises the joint length distribution.
pub(crate) fn select_l_from_bigram_coverage(party_counts: &[Vec<usize>]) -> (usize, f64) {
    let n_cases = party_counts.first().map_or(0, |p| p.len());
    if n_cases == 0 {
        return (TRACE_LEN_FALLBACK_PER_PARTY, 1.0);
    }

    let total_bigrams: usize = (0..n_cases)
        .map(|c| {
            let merged: usize = party_counts.iter().map(|p| p[c]).sum();
            merged.saturating_sub(1)
        })
        .sum();
    if total_bigrams == 0 {
        return (TRACE_LEN_FALLBACK_PER_PARTY, 1.0);
    }

    let max_local = party_counts.iter().flat_map(|p| p.iter()).copied().max().unwrap_or(0);
    let target = BIGRAM_COVERAGE_TARGET;

    for l in BIGRAM_COVERAGE_SCAN_START..=max_local {
        let captured: usize = (0..n_cases)
            .map(|c| {
                let surviving: usize = party_counts.iter().map(|p| p[c].min(l)).sum();
                surviving.saturating_sub(1)
            })
            .sum();
        let coverage = captured as f64 / total_bigrams as f64;
        if coverage >= target {
            let l_aligned = l.div_ceil(TRACE_LEN_ALIGNMENT) * TRACE_LEN_ALIGNMENT;
            return (l_aligned.max(TRACE_LEN_MIN), coverage);
        }
    }

    let l = max_local.div_ceil(TRACE_LEN_ALIGNMENT) * TRACE_LEN_ALIGNMENT;
    (l.max(TRACE_LEN_MIN), 1.0)
}

/// Distinct nonzero columns one case contributes under `encoding`, over the
/// party's own slice. Unwrapped (no START/END markers): a party holds a
/// fragment and cannot know whether its first event follows START.
fn nonzero_columns(encoding: Encoding, acts: &[u32]) -> usize {
    match encoding {
        Encoding::Activity => acts.iter().collect::<HashSet<_>>().len(),
        Encoding::Bigram => acts.windows(2).map(|p| (p[0], p[1])).collect::<HashSet<_>>().len(),
        Encoding::SkipBigram => {
            let d1: HashSet<(u32, u32)> = acts.windows(2).map(|p| (p[0], p[1])).collect();
            let d2: HashSet<(u32, u32)> = acts.windows(3).map(|p| (p[0], p[2])).collect();
            d1.len() + d2.len()
        }
        Encoding::Trigram => {
            acts.windows(3).map(|t| (t[0], t[1], t[2])).collect::<HashSet<_>>().len()
        }
        Encoding::Ef => {
            let mut pairs: HashSet<(u32, u32)> = HashSet::new();
            let mut prefix: HashSet<u32> = HashSet::new();
            for &b in acts {
                for &a in &prefix {
                    pairs.insert((a, b));
                }
                prefix.insert(b);
            }
            pairs.len()
        }
    }
}

/// Distinct nonzero feature columns per case in the selected encoding's space:
/// the `s` of the Candes-Tao bound, whose ambient dimension is
/// `shared::encoding_dim`.
// LEAKAGE: per-party distinct-column count.
pub(crate) fn estimate_sparsity(encoding: Encoding, cases: &[mpc::CaseData]) -> f64 {
    let mut total_nonzero = 0usize;
    for case in cases {
        let acts: Vec<u32> = case.events.iter().map(|(_, a)| *a).filter(|a| *a != 0).collect();
        total_nonzero += nonzero_columns(encoding, &acts);
    }
    // The denominator counts only the cases this party HOLDS: dividing by the
    // all-pad rows too would scale this published handshake scalar by
    // |cases_p| / |universe|, a disclosure of the party's coverage.
    let held = cases.iter().filter(|c| !c.events.is_empty()).count();
    if held == 0 {
        0.0
    } else {
        total_nonzero as f64 / held as f64
    }
}

/// Embedding-profile class histogram for the suppression guard.
#[derive(Debug, Clone)]
pub(crate) struct VariantAnonymity {
    pub n_cases: usize,
    pub distinct: usize,
    pub class_sizes: Vec<usize>,
}

impl VariantAnonymity {
    pub fn singletons(&self) -> usize {
        self.class_sizes.iter().filter(|&&s| s == 1).count()
    }

    pub fn singleton_fraction(&self) -> f64 {
        if self.n_cases == 0 {
            0.0
        } else {
            self.singletons() as f64 / self.n_cases as f64
        }
    }

    pub fn min_non_singleton_class(&self) -> usize {
        self.class_sizes.iter().copied().filter(|&s| s >= 2).min().unwrap_or(0)
    }
}

/// Per-case embedding-profile rows for the suppression guard: the raw φ row,
/// the disclosure unit of `Y_π` and the input the MPC fingerprints. Raw, since
/// normalizing would merge scalar multiples into one class.
pub(crate) fn encoding_profiles(
    encoding: Encoding,
    traces: &[(String, EncodedTrace)],
    dict_size: usize,
) -> Result<FeatureMatrix> {
    Ok(match encoding {
        Encoding::Bigram => phases::p2_bigrams::extract_plaintext(traces).1,
        Encoding::Ef => phases::p2_bigrams::extract_ef(traces, dict_size),
        Encoding::Activity => phases::p2_bigrams::extract_activity(traces, dict_size),
        Encoding::Trigram => phases::p2_bigrams::extract_trigram(traces).1,
        Encoding::SkipBigram => phases::p2_bigrams::extract_skip_bigram(traces, dict_size),
    })
}

pub(crate) fn measure_variant_anonymity(profiles: &FeatureMatrix) -> VariantAnonymity {
    let mut counts: HashMap<Vec<u64>, usize> = HashMap::new();
    for row in profiles {
        let key: Vec<u64> = row.iter().map(|v| v.to_bits()).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    let class_sizes: Vec<usize> = counts.values().copied().collect();
    VariantAnonymity { n_cases: profiles.len(), distinct: counts.len(), class_sizes }
}

/// Reference implementation of the budget-bounded threshold
/// `k_priv = max{ k >= k0 : suppressed_fraction(k) <= tau }`; the production
/// selection runs inside `secure_extract_rsecret.mpc` on the secret class sizes.
pub(crate) fn select_budget_bounded_k_priv(profiles: &FeatureMatrix, k0: usize, tau: f64) -> usize {
    let v = measure_variant_anonymity(profiles);
    let k0 = k0.max(2); // k_priv < 2 disables suppression entirely; floor at 2.
    let n = v.n_cases.max(1) as f64;

    // Zero-φ rows (e.g. trigram over a two-event trace) L2-normalise to zero,
    // so the hybrid reveal forces them out of the anonymous set and they count
    // as always suppressed, as the in-MPC search does.
    let is_zero_row = |row: &FeatureVector| row.iter().all(|&x| x == 0.0);
    let n_zero = profiles.iter().filter(|row| is_zero_row(row)).count();
    let nonzero_class_sizes: Vec<usize> = {
        let mut counts: HashMap<Vec<u64>, usize> = HashMap::new();
        for row in profiles {
            if is_zero_row(row) {
                continue;
            }
            let key: Vec<u64> = row.iter().map(|x| x.to_bits()).collect();
            *counts.entry(key).or_insert(0) += 1;
        }
        counts.into_values().collect()
    };

    // Effective suppression at k: every zero-φ row + nonzero rows whose class
    // is smaller than k.
    let suppressed_fraction = |k: usize| -> f64 {
        let nonzero_supp: usize = nonzero_class_sizes.iter().filter(|&&s| s < k).copied().sum();
        (n_zero + nonzero_supp) as f64 / n
    };

    // Largest k in [k0, n_cases] whose suppressed fraction stays <= tau,
    // mirroring the in-MPC binary search. Beyond the max class size every larger
    // k suppresses the same cases, so the monotone crossing is in range.
    let mut chosen = k0;
    for k in k0..=v.n_cases.max(k0) {
        if suppressed_fraction(k) <= tau {
            chosen = k;
        }
    }

    println!(
        "[Privacy] Variant anonymity (embedding profile): {} classes / {} cases; \
         singletons {} ({:.1}%); min non-singleton class {}",
        v.distinct,
        v.n_cases,
        v.singletons(),
        v.singleton_fraction() * 100.0,
        v.min_non_singleton_class()
    );
    println!(
        "[Privacy] Budget-bounded k_priv = {} (k0={}, tau={:.0}%): suppresses {:.1}% of cases \
         (assigned in-MPC by nearest centroid, hybrid reveal)",
        chosen,
        k0,
        tau * 100.0,
        suppressed_fraction(chosen) * 100.0
    );
    chosen
}

/// Longest merged trace of one case: each party contributes at most its own
/// public padded length, so their sum bounds it, capped by `L * n_parties`.
pub(crate) fn merged_len_bound(
    partial_lens: &[usize],
    l_per_party: usize,
    n_parties: usize,
) -> usize {
    partial_lens.iter().sum::<usize>().min(l_per_party * n_parties)
}

/// Worst-case `‖x‖₁` of one merged case's raw feature row, per encoding; twin of
/// `mpc_utils/encoding_*.py` (ef's `(m+1)(m+2)/2` is quadratic in L). With the
/// ±1 Achlioptas entries `|y_d| <= ‖x‖₁`, so both ring guards read it.
pub(crate) fn encoding_mass_bound(encoding: Encoding, l_total: usize) -> u128 {
    let l = l_total as u128;
    match encoding {
        Encoding::Bigram => l + 1,
        Encoding::Activity => l.max(1),
        Encoding::Trigram => l.saturating_sub(2).max(1),
        Encoding::SkipBigram => 2 * l + 1,
        Encoding::Ef => (l + 1) * (l + 2) / 2,
    }
}

/// Smallest even `NORM_PRESHIFT` keeping the in-MPC L2-norm step of
/// `secure_extract_rsecret.mpc` sound: `sq_shifted = (Σ y²) >> p` is cast into
/// sfix(f=16, k=31), whose integer range is ±2^14, and with R ∈ {±1} the bound
/// is `‖y‖² <= K_PROJ · ‖x‖₁²`. Errors when no preshift fits: the raw cell must
/// survive the cast, and `secure_relabel`'s FAR=8 sentinel needs 2^62 headroom.
pub(crate) fn rsec_norm_preshift(encoding: Encoding, l_total: usize, k_proj: usize) -> Result<u32> {
    let b1 = encoding_mass_bound(encoding, l_total);
    const SFIX_INT_MAX: u128 = 1 << 14; // sfix(f=16, k=31) integer range
    if b1 >= SFIX_INT_MAX {
        return Err(PipelineError::Mpc(format!(
            "--reveal-secret-projection: encoding {:?} worst-case cell mass {} >= 2^14 \
             overflows the in-MPC sfix cast regardless of preshift (merged length {}). \
             Lower --max-trace-len.",
            encoding, b1, l_total
        )));
    }
    let y2_max = k_proj as u128 * b1 * b1;
    let mut p = 2u32;
    while (y2_max >> p) >= SFIX_INT_MAX {
        p += 2;
        if 81 * k_proj as u128 * (1u128 << (32 + p)) >= (1u128 << 62) {
            return Err(PipelineError::Mpc(format!(
                "--reveal-secret-projection: encoding {:?} needs NORM_PRESHIFT > {} \
                 (worst-case ‖y‖² = {}), which breaks the secure_relabel sentinel \
                 headroom at K_PROJ={}. Lower --max-trace-len or --projection-dim.",
                encoding,
                p - 2,
                y2_max,
                k_proj
            )));
        }
    }
    Ok(p)
}

/// Centroid scale of the E-step: twins of `secure_kmeans.mpc::FRAC_BITS` and of
/// `scripts/plaintext_kmeans_reference.py`. `chat_j` carries `2^-SK_FRAC_BITS`,
/// built from `floor(2^RECIP_BITS / n_j)` with
/// `RECIP_BITS = SK_FRAC_BITS + ceil_log2(N) + SK_CENTROID_YMAX_SLACK_BITS`.
pub(crate) const SK_FRAC_BITS: u32 = 16;
/// Precision headroom the reciprocal reserves for `|y_d|`: above it the
/// centroid rounding exceeds one ulp, so the guard refuses instead.
pub(crate) const SK_CENTROID_YMAX_SLACK_BITS: u32 = 15;

/// Worst-case bit width of `secure_kmeans.mpc` at `-R 64`: the max of the
/// E-step distance `ceil_log2(k_proj) + 2·(ceil_log2(y_max)+f) + 3`, which does
/// not depend on N, and the centroid product `s_j[d]·r_j`, bounded by
/// `y_max·2^RECIP_BITS` non-strictly — the tape truncates it with `TruncRing`,
/// whose `a += 2^62` requires `|prod| < 2^62`.
pub(crate) fn secure_kmeans_argmin_bits(k_proj: usize, n_cases: usize, y_max: u64) -> u32 {
    let ceil_log2 = |x: u64| x.max(1).next_power_of_two().trailing_zeros();
    let y_bits = ceil_log2(y_max);
    if y_bits > SK_CENTROID_YMAX_SLACK_BITS {
        return 64;
    }
    let recip_bits = SK_FRAC_BITS + ceil_log2(n_cases as u64) + SK_CENTROID_YMAX_SLACK_BITS;
    let distance_bits = ceil_log2(k_proj as u64) + 2 * (y_bits + SK_FRAC_BITS) + 3;
    // +2, not +1: `|prod| <= 2^(y_bits+recip_bits)` is attainable, so the
    // magnitude needs one bit more and the sign one more; the `> 63` refusal at
    // the call site is what enforces TruncRing's strict `|prod| < 2^62`.
    let centroid_product_bits = y_bits + recip_bits + 2;
    distance_bits.max(centroid_product_bits)
}

/// `ambient_dim` is the dimension actually projected — `shared::encoding_dim`
/// for the selected encoding, NOT d².
pub(crate) fn compute_adaptive_kproj(
    s_per_party: f64,
    ambient_dim: f64,
    safety_factor: f64,
    min_kproj: usize,
) -> usize {
    let cs_per_party = CS_COEFF * s_per_party * (ambient_dim / s_per_party).ln();
    let s_full = s_per_party * CS_FULL_SPARSITY_MULTIPLIER;
    let cs_full = CS_COEFF * s_full * (ambient_dim / s_full).ln();
    let cs_binding = cs_per_party.min(cs_full);
    // s = 0 (no party has a two-event trace) makes the bound NaN, and any
    // bound below min_kproj + 1 would underflow the `- 1` cap below.
    if !cs_binding.is_finite() || cs_binding < (min_kproj + 1) as f64 {
        println!(
            "[Auto-kproj] WARNING: degenerate sparsity (s={:.2}, CS bound {:.1}); \
             falling back to k_proj = {}",
            s_per_party, cs_binding, min_kproj
        );
        return min_kproj;
    }
    let k_proj = (safety_factor * cs_binding).floor() as usize;
    let k_proj = k_proj.max(min_kproj).min(cs_binding.floor() as usize - 1);
    println!(
        "[Auto-kproj] CS_per_party={:.0}, CS_full={:.0}, CS_binding={:.0}, k_proj={} (safety={:.1})",
        cs_per_party, cs_full, cs_binding, k_proj, safety_factor
    );
    k_proj
}
