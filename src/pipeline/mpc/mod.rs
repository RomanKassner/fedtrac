//! MPC pipeline orchestrator: shared setup, mode validation, dispatch to
//! [`secure_kmeans`] or [`reveal_projection`].

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::config::PipelineConfig;
use crate::constants::*;
use crate::error::{PipelineError, Result};
use crate::mpc;
use crate::projection;
use crate::types::{ActivityDictionary, PartyState};

mod parameters;
mod reveal_projection;
mod secure_kmeans;
pub(in crate::pipeline) mod shared;

use parameters::{
    compute_adaptive_kproj, compute_dynamic_trace_len, encoding_mass_bound, encoding_profiles,
    estimate_sparsity, merged_len_bound, rsec_norm_preshift, secure_kmeans_argmin_bits,
    select_budget_bounded_k_priv,
};
// The --truncate-baseline control feeds the plaintext tier the same L.
pub(in crate::pipeline) use parameters::compute_dynamic_trace_len as dynamic_trace_len;
#[cfg(test)]
pub(crate) use parameters::{
    compute_adaptive_kproj as compute_adaptive_kproj_for_test,
    encoding_mass_bound as encoding_mass_bound_for_test,
    encoding_profiles as encoding_profiles_for_test,
    estimate_sparsity as estimate_sparsity_for_test, measure_variant_anonymity,
    merged_len_bound as merged_len_bound_for_test,
    rsec_norm_preshift as rsec_norm_preshift_for_test,
    secure_kmeans_argmin_bits as secure_kmeans_argmin_bits_for_test,
    select_budget_bounded_k_priv as select_k_priv_for_test, select_l_from_bigram_coverage,
    SK_CENTROID_YMAX_SLACK_BITS, SK_FRAC_BITS,
};
#[cfg(test)]
pub(crate) use shared::encoding_dim;

pub(crate) fn run_mpc_pipeline(
    cfg: &PipelineConfig,
    states: &[PartyState],
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
    output_base: &Path,
) -> Result<()> {
    let t_mpc = Instant::now();
    let n_parties = states.len();
    let state_refs: Vec<&PartyState> = states.iter().collect();

    // Reject invalid configs before any cross-party plaintext is touched.
    validate_mode_flags(cfg)?;

    let effective_max_trace_len =
        compute_dynamic_trace_len(&state_refs, common_cases, dictionary, cfg.max_trace_len);

    let all_cases: Vec<Vec<mpc::CaseData>> = states
        .iter()
        .map(|s| {
            super::data_loading::prepare_cases_for_mpc(
                s,
                common_cases,
                dictionary,
                Some(effective_max_trace_len),
            )
        })
        .collect();

    // In-MPC edge-filter soundness: secure_aggregate compares aggregated cells
    // at the fixed AGG_CELL_BITS width, so the worst-case per-cluster edge count
    // must stay below 2^AGG_CELL_BITS; fail loud rather than compare wrong.
    let n_cases_total = all_cases.iter().map(|c| c.len()).max().unwrap_or(0);
    // Earliest point at which N is known, and no `.mpc` program has compiled or
    // run yet. A pinned k > N would otherwise reach init_centroids, whose
    // stride N/k is 0.
    if let Some(k) = cfg.k_fixed() {
        validate_k_against_cases(k, n_cases_total, "agreed cases")?;
    }
    // The sum of the public per-party lengths, not L * n_parties: a party whose
    // longest kept trace is shorter than L cannot fill its share of the merge.
    let (_, partial_lens) = mpc::slab_shape(&all_cases).map_err(PipelineError::Io)?;
    let max_merged_len = merged_len_bound(&partial_lens, effective_max_trace_len, n_parties);
    let worst_edge_count = n_cases_total as u64 * max_merged_len.saturating_sub(1) as u64;
    if worst_edge_count >= 1u64 << AGG_CELL_BITS {
        return Err(PipelineError::Mpc(format!(
            "In-MPC edge filter can compare wrong: {} cases x merged length {} allows a \
             per-cluster edge count of {} >= 2^AGG_CELL_BITS. Lower --max-trace-len.",
            n_cases_total, max_merged_len, worst_edge_count
        )));
    }

    let spdz_cfg = mpc::SpdzConfig::default();
    // Raw party inputs and secret-share persistence must not outlive the run.
    // Every round shares one persistence file, so cleanup may fire only at
    // pipeline end; under --reuse-extraction a successful run arms survival.
    let secrets_cleanup = mpc::SecretsCleanup::new(&spdz_cfg);
    let dict_size = dictionary.len();
    let d_sq = dict_size * dict_size;

    // Plaintext trace integration is harness-only and lazy: it runs under
    // --evaluate-full alone.
    let (pt_traces, pt_bigram_index, pt_features_raw) = if cfg.evaluate_full {
        let super::data_loading::IntegratedTraces { traces, bigram_index, features } =
            super::data_loading::integrate_traces(&state_refs, common_cases, dictionary)?;
        (traces, bigram_index, features)
    } else {
        println!(
            "[Privacy] No plaintext trace integration: this run touches no merged \
             cross-party data outside the MPC (--evaluate-full not set)"
        );
        (Vec::new(), crate::types::BigramIndex::new(), Vec::new())
    };

    let enc_dim = shared::encoding_dim(cfg.encoding, dict_size);
    let encoding_flag = shared::encoding_flag(cfg.encoding)?;

    // Sparsity AND ambient dimension are both the selected encoding's.
    // k_proj > enc_dim would be a lift, not a reduction, and breaks JL.
    let sparsities: Vec<f64> =
        all_cases.iter().map(|c| estimate_sparsity(cfg.encoding, c)).collect();
    let s_max = sparsities.iter().cloned().fold(0.0f64, f64::max);
    for (i, s) in sparsities.iter().enumerate() {
        println!("[Auto-kproj] Party {} sparsity ({:?}): {:.1}", i, cfg.encoding, s);
    }
    let k_proj_raw = compute_adaptive_kproj(
        s_max,
        enc_dim as f64,
        ADAPTIVE_KPROJ_SAFETY_FACTOR,
        ADAPTIVE_KPROJ_MIN,
    );
    let k_proj = k_proj_raw.min(enc_dim);
    if k_proj < k_proj_raw {
        println!("[Auto-kproj] Clamped k_proj {} -> {} (enc_dim={})", k_proj_raw, k_proj, enc_dim);
    }
    fs::create_dir_all(output_base).map_err(PipelineError::Io)?;
    for (i, c) in all_cases.iter().enumerate() {
        println!("[MPC] Party {}: {} cases", i, c.len());
    }

    // Extract-once / branch-many: does this run create the core slab or read it?
    // A present-but-divergent sidecar fails loud here. k, k_proj and R are absent
    // from the key, so one slab feeds an rsec family and its same-encoding SK pair.
    let reuse_state: Option<(mpc::SlabKey, usize)> = if cfg.reuse_extraction {
        let (slab_n, slab_plens) = mpc::slab_shape(&all_cases)?;
        let key = mpc::SlabKey {
            data_dir: cfg.data_dir.clone(),
            // The resolved ids, in the index order the parties are written in,
            // plus the ordered dictionary: together with the payload digest
            // they pin the event content of the slab.
            case_set_sha256: mpc::ordered_digest(
                all_cases.first().map(Vec::as_slice).unwrap_or(&[]).iter().map(|c| c.id.as_str()),
            ),
            dictionary_sha256: mpc::ordered_digest(&dictionary.decode),
            // Computed from the in-memory cases, so a tail-only run (which
            // writes no input stream) still proves identity before reuse.
            input_payload_sha256: mpc::input_payload_digest(&all_cases, dict_size)?,
            n_cases: slab_n,
            max_trace_len_effective: effective_max_trace_len,
            partial_lens: slab_plens,
            dict_size,
            encoding: format!("{:?}", cfg.encoding),
            enc_dim,
            protocol: spdz_cfg.protocol.clone(),
            ring_size: crate::constants::MPC_RING_SIZE,
            compile_flags: spdz_cfg.compile_flags.clone(),
            n_parties,
        };
        let enc_offset = mpc::enc_slab_offset(slab_n, d_sq, enc_dim);
        Some((key, enc_offset))
    } else {
        None
    };
    let reuse_mode = match &reuse_state {
        Some((key, enc_offset)) => {
            let m = mpc::resolve_extract_reuse(&spdz_cfg.spdz_path, key, *enc_offset)?;
            println!(
                "[MPC] --reuse-extraction: {} (encoding slab @{})",
                if m.is_tail_only() {
                    "tail-only (skip Round-1 input read + merge + extract)"
                } else {
                    "creating core slab (full extract + persist encoding)"
                },
                enc_offset
            );
            m
        }
        None => mpc::ExtractReuse::None,
    };

    if cfg.secure_kmeans {
        // Achlioptas R is generated here and shipped to secure_extract
        // (SECURE_Y=1); the rsec path samples R in-MPC and writes no file.
        let skip_projection = enc_dim <= k_proj_raw;
        let seed = cfg.projection_seed.unwrap_or(DEFAULT_PROJECTION_SEED);
        let r_matrix = if skip_projection {
            // No projection needed when enc_dim already fits — substitute identity.
            println!(
                "[Auto-kproj] Skipping projection (enc_dim={} ≤ k_proj_raw={}): using identity",
                enc_dim, k_proj_raw
            );
            projection::AchlioptasMatrix::identity(enc_dim)
        } else {
            projection::AchlioptasMatrix::generate(k_proj, enc_dim, seed)
        };
        let r_path = output_base.join("projection_matrix.txt");
        r_matrix.write_to_file(&r_path).map_err(PipelineError::Io)?;

        // Ring headroom for secure_kmeans.mpc: what wraps silently past 2^63
        // (a sign flip, so wrong labels and no error) is the E-step distance and
        // the centroid product, both bounded by `secure_kmeans_argmin_bits`,
        // whose `y_max` is the exact per-encoding L1 mass.
        let y_max =
            u64::try_from(encoding_mass_bound(cfg.encoding, max_merged_len)).unwrap_or(u64::MAX);
        let bits = secure_kmeans_argmin_bits(k_proj, n_cases_total, y_max);
        if bits > 63 {
            return Err(PipelineError::Mpc(format!(
                "--secure-kmeans can overflow the 64-bit ring: worst-case width {} bits > 63 \
                 (N={}, K_PROJ={}, encoding {:?}, y_max bound {}). Lower --max-trace-len.",
                bits, n_cases_total, k_proj, cfg.encoding, y_max
            )));
        }
        // validate_mode_flags refuses a missing or `auto` --k under
        // --secure-kmeans, so k is pinned by the time the mode runs.
        let k = cfg.k_fixed().ok_or_else(|| {
            PipelineError::Mpc("--secure-kmeans requires an explicit --k <n>".into())
        })?;
        let r = secure_kmeans::run(
            cfg,
            cfg,
            k,
            &spdz_cfg,
            &all_cases,
            dict_size,
            k_proj,
            &r_path,
            &r_matrix,
            encoding_flag,
            enc_dim,
            reuse_mode,
            dictionary,
            &pt_traces,
            &pt_bigram_index,
            &pt_features_raw,
            output_base,
            t_mpc,
            n_parties,
            // The guard's own width, passed to the tape so the E-step argmin
            // compares at it instead of the ring maximum.
            bits,
        );
        return finalize_reuse(r, &reuse_state, &spdz_cfg, &secrets_cleanup);
    }

    if cfg.reveal_secret_projection {
        // In-MPC L2-norm soundness: the sfix cast in Step 2c-norm overflows for
        // encodings whose worst-case ‖y‖² exceeds 2^(14+preshift) (EF is
        // quadratic in trace length); take the smallest even preshift that fits.
        let norm_preshift =
            rsec_norm_preshift(cfg.encoding, effective_max_trace_len * n_parties, k_proj)?;
        if norm_preshift > 2 {
            println!(
                "[MPC] NORM_PRESHIFT auto-raised to {} (encoding {:?} worst-case ‖y‖² \
                 exceeds the sfix range at the default preshift 2; revealed cells scale \
                 by 2^(16+{}/2))",
                norm_preshift, cfg.encoding, norm_preshift
            );
        }

        // Budget-bounded privacy floor: the MPC itself raises k_priv to the
        // largest value >= k0 whose suppressed fraction stays within tau, on the
        // SECRET class sizes. Suppressed cases are not rejected — the hybrid
        // reveal assigns them in-MPC, so they enter the Round 2 DFGs.
        let k0 = VARIANT_K_ANONYMITY.max(2);

        // Harness-only cross-check of the in-MPC selection against the plaintext
        // calibration on the truncated merged profiles.
        if cfg.evaluate_full {
            let truncated_traces = super::data_loading::merge_truncated_cases(&all_cases);
            let profiles = encoding_profiles(cfg.encoding, &truncated_traces, dict_size)?;
            let expected = select_budget_bounded_k_priv(
                &profiles,
                VARIANT_K_ANONYMITY,
                MAX_SINGLETON_FRACTION,
            );
            println!(
                "[Harness] Plaintext calibration cross-check: expected k_priv = {} \
                 (compare with the in-MPC K_PRIV_SELECTED below)",
                expected
            );
        }

        // R is sampled inside MPC; no projection matrix file was written.
        let r = reveal_projection::run(
            cfg,
            cfg,
            &spdz_cfg,
            &all_cases,
            dict_size,
            k_proj,
            encoding_flag,
            enc_dim,
            reuse_mode,
            k0,
            norm_preshift,
            dictionary,
            &pt_traces,
            &pt_bigram_index,
            &pt_features_raw,
            output_base,
            effective_max_trace_len,
            t_mpc,
            n_parties,
        );
        return finalize_reuse(r, &reuse_state, &spdz_cfg, &secrets_cleanup);
    }

    unreachable!()
}

/// Earn slab survival: on a fully successful `--reuse-extraction` run, write (or
/// refresh) the identity sidecar and disarm the Transactions removal. Any error
/// path returns the error untouched, so the Drop guard wipes everything.
fn finalize_reuse(
    result: Result<()>,
    reuse_state: &Option<(mpc::SlabKey, usize)>,
    spdz_cfg: &mpc::SpdzConfig,
    secrets_cleanup: &mpc::SecretsCleanup,
) -> Result<()> {
    result?;
    if let Some((key, enc_offset)) = reuse_state {
        mpc::write_sidecar(&spdz_cfg.spdz_path, key, 0, *enc_offset)
            .map_err(|e| PipelineError::Mpc(format!("extract-slab sidecar: {e}")))?;
        secrets_cleanup.preserve_persistence();
    }
    Ok(())
}

pub(crate) fn validate_mode_flags(cfg: &PipelineConfig) -> Result<()> {
    if cfg.secure_kmeans && cfg.k_fixed().is_none() {
        // Plaintext auto-k leaks per-dataset stats in a non-research deployment,
        // and in-box k selection is not offered, so the mode takes k as input.
        return Err(PipelineError::Mpc(
            "--secure-kmeans requires an explicit --k <n>: in-box k selection is not \
             offered, and the plaintext auto-k leaks per-dataset stats"
                .into(),
        ));
    }
    if cfg.secure_kmeans && cfg.reveal_secret_projection {
        return Err(PipelineError::Mpc(
            "--secure-kmeans and --reveal-secret-projection are mutually exclusive".into(),
        ));
    }
    if !cfg.secure_kmeans && !cfg.reveal_secret_projection {
        return Err(PipelineError::Mpc(
            "--mpc requires --secure-kmeans or --reveal-secret-projection".into(),
        ));
    }
    if cfg.reveal_secret_projection
        && cfg.secure_clustering == crate::config::SecureClustering::Sequence
    {
        // clap's `requires = "secure_kmeans"` already rejects the pair on the
        // command line; the rule stays so a programmatic config cannot produce
        // an RP run in a directory named like a sequence cell.
        return Err(PipelineError::Mpc(
            "--secure-clustering sequence is a middle step of --secure-kmeans and does \
             not apply to --reveal-secret-projection"
                .into(),
        ));
    }
    // Both MPC modes filter DFG cells inside the box before the open, and the
    // published leakage envelope states that unconditionally. A cutoff with no
    // exact integer form is refused here rather than after Round 1.
    if crate::mpc::rounds::edge_filter_int_args(MIN_EDGE_COUNT, MIN_EDGE_FRACTION).is_none() {
        return Err(PipelineError::Mpc(format!(
            "edge cutoff ({}, {}) has no exact integer form, so the in-MPC pre-reveal \
             edge filter cannot run and every cell would be opened.",
            MIN_EDGE_COUNT, MIN_EDGE_FRACTION
        )));
    }
    Ok(())
}

/// k must fit the population it partitions: `secure_kmeans.mpc::init_centroids`
/// strides by `N_CASES // K`, which is 0 for `k > N` — every centroid seeds from
/// row 0, with no error anywhere. `population` names it in the message.
pub(crate) fn validate_k_against_cases(k: usize, n_cases: usize, population: &str) -> Result<()> {
    if k < 1 {
        return Err(PipelineError::Mpc(format!("--k must be at least 1 (got {k})")));
    }
    if k > n_cases {
        return Err(PipelineError::Mpc(format!(
            "--k {} exceeds the {} {}: the in-MPC centroid seeding strides by N/k = 0, \
             seeding every centroid from row 0 and leaving k-1 clusters empty",
            k, n_cases, population
        )));
    }
    Ok(())
}
