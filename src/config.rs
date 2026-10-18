//! CLI surface: the clap-derive `PipelineConfig` and the enums its flags parse into.

use clap::{Parser, ValueEnum};

use crate::constants::*;

#[derive(Parser, Debug, Clone)]
#[command(version, about = "Federated trace clustering with MPC")]
pub struct PipelineConfig {
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    pub data_dir: String,

    #[arg(long)]
    pub run_name: Option<String>,

    #[arg(long, value_enum, default_value_t = Encoding::Bigram)]
    pub encoding: Encoding,

    /// Cluster count: `--k <n>`, or `--k auto`. `--secure-kmeans` requires an
    /// explicit count; `auto` is identical to omitting the flag.
    #[arg(long, value_parser = KSpec::parse)]
    pub k: Option<KSpec>,

    #[arg(long, default_value_t = DEFAULT_N_PARTIES, value_parser = parse_n_parties)]
    pub n_parties: usize,

    /// Restrict the run to exactly these case IDs (newline-separated file), so a
    /// comparison varies only the factor under study.
    #[arg(long)]
    pub case_ids: Option<String>,

    #[arg(long)]
    pub max_trace_len: Option<usize>,

    /// The plaintext baseline consumes the same per-party truncated input as the
    /// MPC path, isolating the truncation term of the tier decomposition.
    #[arg(long, conflicts_with = "mpc")]
    pub truncate_baseline: bool,

    /// RNG seed for the projection matrix (deterministic).
    #[arg(long)]
    pub projection_seed: Option<u64>,

    /// Run the secure MPC pipeline. Pick exactly one of --secure-kmeans or
    /// --reveal-secret-projection when set.
    #[arg(long)]
    pub mpc: bool,

    /// Full MPC end-to-end (k-means inside MPC). Requires --k.
    #[arg(long, requires = "mpc", conflicts_with = "reveal_secret_projection")]
    pub secure_kmeans: bool,

    /// Middle step of `--secure-kmeans`. `kmeans` is Lloyd's on the secret Y;
    /// `sequence` follows it with a Markov-chain hard-EM refinement on the
    /// secret d² rows. Round 2 and the disclosure envelope are identical either way.
    #[arg(long, value_enum, default_value_t = SecureClustering::Kmeans,
          requires = "secure_kmeans")]
    pub secure_clustering: SecureClustering,

    /// Reveal a secret-R + secret-row-shuffled projected Y for plaintext
    /// clustering, then un-shuffle via secure_relabel before Round 2.
    #[arg(long, requires = "mpc")]
    pub reveal_secret_projection: bool,

    /// Extract-once / branch-many amortization: the first matching run persists
    /// the RNG-free core slab and its identity key; a later run whose key
    /// matches EXACTLY skips Round 1's read + merge + extract. A mismatch fails loud.
    #[arg(long, requires = "mpc")]
    pub reuse_extraction: bool,

    /// Compute both the plaintext and the federated tier.
    #[arg(long)]
    pub evaluate_full: bool,

    /// Write the raw encoded feature matrix and its case ids, then stop before
    /// clustering: the truth matrix of the attack battery rescore.
    #[arg(long, conflicts_with = "mpc")]
    pub dump_encoded_features: bool,

    /// Skip the log-based baseline; use token replay (faster eval).
    #[arg(long)]
    pub fast_eval: bool,

    /// Not a flag: the tiers of a `--secure-clustering sequence` run set the
    /// plaintext twin internally so the comparison scores the same middle step.
    #[arg(skip = ClusteringAlgo::Kmeans)]
    pub clustering_algo: ClusteringAlgo,
}

impl PipelineConfig {
    pub fn from_args(args: &[String]) -> Self {
        Self::parse_from(args)
    }

    /// The user-pinned k, or None when k is auto-selected (or absent).
    pub fn k_fixed(&self) -> Option<usize> {
        match self.k {
            Some(KSpec::Fixed(n)) => Some(n),
            _ => None,
        }
    }
}

/// `--k` argument: a pinned count, or `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KSpec {
    Auto,
    Fixed(usize),
}

impl KSpec {
    fn parse(s: &str) -> std::result::Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(KSpec::Auto);
        }
        let n: usize =
            s.parse().map_err(|_| format!("expected a positive integer or `auto`, got `{s}`"))?;
        if n == 0 {
            return Err("k must be at least 1".into());
        }
        Ok(KSpec::Fixed(n))
    }
}

impl std::str::FromStr for KSpec {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(s)
    }
}

// n_parties < 2 would silently fall into the 2-party loader branch.
fn parse_n_parties(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s.parse().map_err(|e| format!("{e}"))?;
    if n < 2 {
        return Err(format!("a federation needs at least 2 parties (got {n})"));
    }
    Ok(n)
}

/// Trace encoding used for clustering features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum Encoding {
    /// Directly-follows bigram counts (default).
    Bigram,
    /// Eventually-follows counts.
    Ef,
    /// Bag-of-activities (Song et al. 2008).
    Activity,
    /// Directly-follows trigram counts.
    Trigram,
    /// Bigram concatenated with skip-bigram at distance 2; output dim 2·d².
    SkipBigram,
}

/// Which plaintext clustering a tier runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ClusteringAlgo {
    /// Lloyd's k-means via linfa.
    Kmeans,
    /// k-means-seeded Markov-chain hard-EM on the d² directly-follows rows, the
    /// plaintext twin of `--secure-clustering sequence`; needs those rows too.
    Sequence,
}

/// Middle step under `--secure-kmeans`: Lloyd's alone, or Lloyd's followed by
/// the in-MPC Markov-chain refinement (`secure_sequence_cluster.mpc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum SecureClustering {
    /// Lloyd's k-means on the secret Y (default).
    Kmeans,
    /// Lloyd's, then Markov-chain hard-EM on the secret count rows.
    Sequence,
}
