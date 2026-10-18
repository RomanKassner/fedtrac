//! Algorithmic phases of the pipeline (P0 case intersection and P1 dictionary
//! run inline in `main.rs`). Each phase is pure; the pipelines own I/O.

pub mod p2_bigrams;
pub mod p3_clustering;
pub mod p4_aggregate;
