//! Plaintext pipeline path.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::config::{Encoding, PipelineConfig};
use crate::error::Result;
use crate::phases;
use crate::types::{ActivityDictionary, PartyState};
use crate::utils::l2_normalize;

pub(crate) fn run_plaintext_pipeline(
    cfg: &PipelineConfig,
    states: &[&PartyState],
    common_cases: &HashSet<String>,
    dictionary: &ActivityDictionary,
    output_base: &Path,
) -> Result<()> {
    println!("\n--- Phase 2-3: Plaintext Integration & Clustering ---");
    let t_plain = Instant::now();
    let super::data_loading::IntegratedTraces { traces, bigram_index, features: bigram_features } =
        if cfg.truncate_baseline {
            let l =
                super::mpc::dynamic_trace_len(states, common_cases, dictionary, cfg.max_trace_len);
            super::data_loading::integrate_traces_truncated(states, common_cases, dictionary, l)?
        } else {
            super::data_loading::integrate_traces(states, common_cases, dictionary)?
        };

    // Evaluation-only dump of the raw encoded rows; nothing is clustered.
    if cfg.dump_encoded_features {
        let (raw, _) = super::mpc::shared::extract_raw_features_dense_cols(
            cfg.encoding,
            &traces,
            &bigram_index,
            &bigram_features,
            dictionary.len(),
        );
        super::export::export_encoded_features(&raw, &traces, output_base)?;
        println!("Plaintext dump completed in {:?}", t_plain.elapsed());
        return Ok(());
    }

    let clustering_features = match cfg.encoding {
        Encoding::Ef => {
            let ef = phases::p2_bigrams::extract_ef(&traces, dictionary.len());
            println!(
                "[EF] {} features (d^2={})",
                ef.first().map(|r| r.len()).unwrap_or(0),
                dictionary.len().pow(2)
            );
            l2_normalize(&ef)
        }
        Encoding::Bigram => l2_normalize(&bigram_features),
        Encoding::Activity => {
            let act = phases::p2_bigrams::extract_activity(&traces, dictionary.len());
            println!(
                "[Activity] {} features (d={})",
                act.first().map(|r| r.len()).unwrap_or(0),
                dictionary.len()
            );
            l2_normalize(&act)
        }
        Encoding::Trigram => {
            let (_idx, tri) = phases::p2_bigrams::extract_trigram(&traces);
            println!(
                "[Trigram] {} features (sparse user-cube observed, (d-3)^3_max={})",
                tri.first().map(|r| r.len()).unwrap_or(0),
                dictionary
                    .len()
                    .saturating_sub(crate::types::FIRST_USER_ACTIVITY as usize)
                    .saturating_pow(3)
            );
            l2_normalize(&tri)
        }
        Encoding::SkipBigram => {
            let sb = phases::p2_bigrams::extract_skip_bigram(&traces, dictionary.len());
            println!(
                "[SkipBigram] {} features (2·d^2={}, bigram + distance-2 block)",
                sb.first().map(|r| r.len()).unwrap_or(0),
                2 * dictionary.len().pow(2)
            );
            l2_normalize(&sb)
        }
    };

    let cluster_result =
        super::clustering::run_clustering(&clustering_features, cfg, None, output_base, None)?;
    println!("Plaintext completed in {:?}", t_plain.elapsed());

    let output_dir =
        if cfg.evaluate_full { output_base.join("plaintext") } else { output_base.to_path_buf() };
    super::export::export_plaintext_results(
        &bigram_index,
        &bigram_features,
        &traces,
        &cluster_result,
        dictionary,
        cfg,
        &output_dir,
        "plaintext",
    )
}
