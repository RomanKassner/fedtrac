//! Federated Clustering Pipeline.

mod config;
mod constants;
mod error;
mod evaluation;
mod mpc;
mod output;
mod phases;
mod pipeline;
mod projection;
mod types;
mod utils;

#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::time::Instant;

use config::PipelineConfig;
use error::{PipelineError, Result};
use types::{ActivityDictionary, PartyState};

fn main() -> Result<()> {
    // Drop covers returns, `?` and panic-unwind but not a signal, and the
    // campaign driver kills over-budget cells with `timeout --kill-after`.
    mpc::install_signal_teardown();

    let args: Vec<String> = std::env::args().collect();
    let cfg = PipelineConfig::from_args(&args);

    let output_base = output::resolve_output_dir(&cfg.run_name);
    println!("--- Start ---");
    println!("Output: {}", output_base.display());

    let states = pipeline::data_loading::load_data(&cfg)?;
    let state_refs: Vec<&PartyState> = states.iter().collect();

    // Phase 0: the agreed case set is the intersection of the parties' case ids.
    println!("--- Phase 0: Case ID intersection ({} parties) ---", states.len());
    let t0 = Instant::now();
    let mut common_cases = pipeline::data_loading::intersect_all_cases(&states);
    // Pinning the case set keeps a comparison varying only the studied factor.
    // Phase 0 already exchanges case ids in plaintext, so this discloses nothing new.
    if let Some(path) = &cfg.case_ids {
        let requested: Vec<String> = std::fs::read_to_string(path)
            .map_err(|e| error::PipelineError::Data(format!("--case-ids {}: {}", path, e)))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        let missing: Vec<&String> =
            requested.iter().filter(|c| !common_cases.contains(*c)).collect();
        if !missing.is_empty() {
            // Silently dropping these would reintroduce exactly the divergence
            // this flag exists to remove, so fail instead.
            return Err(error::PipelineError::Data(format!(
                "--case-ids: {} of {} pinned cases are absent from the agreed set \
                 (first missing: {}). The pinned set must be a subset of the agreed \
                 universe; regenerate it with build_full_case_ids.py for this split.",
                missing.len(),
                requested.len(),
                missing.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            )));
        }
        common_cases = requested.into_iter().collect();
        println!("Phase 0: pinned to {} cases from {}", common_cases.len(), path);
    }
    println!("Phase 0: {} common cases in {:?}", common_cases.len(), t0.elapsed());

    // LEAKAGE (simulator only): unions every party's vocabulary in plaintext;
    // a deployment must agree it out of band (size and order become public).
    println!("\n--- Phase 1: Activity Dictionary ---");
    let t1 = Instant::now();
    let mut all_activities: HashSet<String> = HashSet::new();
    for s in &states {
        all_activities.extend(s.activities_for_cases(&common_cases));
    }
    let dictionary = ActivityDictionary::from_activities(all_activities)?;
    println!(
        "Phase 1: {} user activities ({} total with markers) in {:?}",
        dictionary.user_activity_count(),
        dictionary.len(),
        t1.elapsed()
    );
    if dictionary.is_empty() {
        return Err(PipelineError::Data("No user activities found in common cases".into()));
    }

    if cfg.mpc {
        return pipeline::mpc::run_mpc_pipeline(
            &cfg,
            &states,
            &common_cases,
            &dictionary,
            &output_base,
        );
    }

    pipeline::plaintext::run_plaintext_pipeline(
        &cfg,
        &state_refs,
        &common_cases,
        &dictionary,
        &output_base,
    )
}
