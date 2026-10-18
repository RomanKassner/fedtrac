//! Tests that pin a rule of the thesis, a twin across the three languages, or
//! the correctness of what a figure script reads.

use crate::phases::{p2_bigrams, p3_clustering, p4_aggregate};
use crate::types::{EncodedEvent, EncodedTrace, END, START};

fn make_trace(case_id: &str, activities: &[u32]) -> (String, EncodedTrace) {
    let events: EncodedTrace = activities
        .iter()
        .enumerate()
        .map(|(i, &act)| EncodedEvent {
            case_id: case_id.to_string(),
            activity: act,
            timestamp: i as u64,
        })
        .collect();
    (case_id.to_string(), events)
}

/// Twin of `mpc_utils/encoding_skip_bigram.py`: block 0 holds the bigrams,
/// block 1 the distance-2 pairs, both over the START/END-wrapped trace.
#[test]
fn test_skip_bigram_layout_and_counts() {
    let traces = vec![make_trace("c1", &[3, 4, 5]), make_trace("c2", &[3, 3, 3])];
    let d = 8usize;
    let feats = p2_bigrams::extract_skip_bigram(&traces, d);

    assert_eq!(feats.len(), 2);
    assert_eq!(feats[0].len(), 2 * d * d, "expected 2·d² dim");

    let block = d * d;
    let bg = |row: &[f64], a: u32, b: u32| row[a as usize * d + b as usize];
    let sk = |row: &[f64], a: u32, b: u32| row[block + a as usize * d + b as usize];

    let r = &feats[0];
    assert_eq!(bg(r, START, 3), 1.0);
    assert_eq!(bg(r, 3, 4), 1.0);
    assert_eq!(bg(r, 4, 5), 1.0);
    assert_eq!(bg(r, 5, END), 1.0);
    assert_eq!(sk(r, START, 4), 1.0);
    assert_eq!(sk(r, 3, 5), 1.0);
    assert_eq!(sk(r, 4, END), 1.0);

    let r = &feats[1];
    assert_eq!(bg(r, 3, 3), 2.0, "two (3,3) bigrams in [3,3,3]");
    assert_eq!(sk(r, 3, 3), 1.0, "one (3,_,3) skip-2 in [3,3,3]");
    assert_eq!(sk(r, START, 3), 1.0);
    assert_eq!(sk(r, 3, END), 1.0);
}

#[test]
fn test_projection_is_deterministic_and_sparse_matches_dense() {
    use crate::projection::AchlioptasMatrix;
    // --projection-seed pins R, so the seed families of the campaign are
    // reproducible run to run.
    let r = AchlioptasMatrix::generate(6, 32, 7);
    assert_eq!(r.data, AchlioptasMatrix::generate(6, 32, 7).data);
    assert_eq!(r.data.len(), 6);
    assert_eq!(r.data[0].len(), 32);

    // A sparse row plus its dense-column map must project exactly like the
    // dense row: the projected tier reads the sparse-indexed encodings.
    let mut dense = vec![0.0f64; 32];
    dense[3] = 2.0;
    dense[17] = 5.0;
    dense[31] = 1.0;
    let y_dense = r.project(&[dense], None);
    let y_sparse = r.project(&[vec![2.0f64, 5.0, 1.0]], Some(&[3usize, 17, 31]));
    assert_eq!(y_dense, y_sparse);
    assert_eq!(y_dense[0].len(), 6);

    // The identity passthrough projects to the input itself.
    let x = vec![vec![1.0f64, 0.0, 3.0, -2.0]];
    assert_eq!(AchlioptasMatrix::identity(4).project(&x, None), x);
}

// --- The per-party length bound L ---

#[test]
fn test_bigram_coverage_selects_the_length_bound() {
    use crate::pipeline::mpc::select_l_from_bigram_coverage as select_l;
    // Party A holds 20 events per case, B holds 3, so the bottleneck is A's
    // tail rather than the merged length: L must keep A's events.
    let (per_party, coverage) = select_l(&[vec![20; 100], vec![3; 100]]);
    assert!(coverage >= 0.97, "coverage {:.3} < 0.97", coverage);
    assert!(per_party >= 20, "per_party {} too low, party A has 20 events", per_party);

    // Uniform lengths capture every bigram at L = 10 per party, and the bound
    // is rounded up to the next multiple of four.
    let (per_party, coverage) = select_l(&[vec![10; 500], vec![10; 500]]);
    assert!((coverage - 1.0).abs() < 1e-6);
    assert_eq!(per_party, 12);

    // No bigrams at all, and an empty input: the floor L = 8 holds.
    let (per_party, coverage) = select_l(&[vec![1; 100], vec![0; 100]]);
    assert_eq!(per_party, 8, "a degenerate log must return the floor L=8");
    assert!((coverage - 1.0).abs() < 1e-6);
    assert_eq!(select_l(&[vec![], vec![]]).0, 8);
}

// --- Constants twinned with the secure programs ---

/// Parse `<NAME> = <int>` or `<NAME> = <base> << <shift>` from constants.py.
fn parse_python_int_constant(name: &str) -> f64 {
    let py = std::fs::read_to_string("mp-spdz/mpc_utils/constants.py")
        .expect("mp-spdz/mpc_utils/constants.py must exist - run from project root");
    let line = py
        .lines()
        .find(|l| l.trim_start().starts_with(name))
        .unwrap_or_else(|| panic!("{} not defined in constants.py", name));
    let rhs = line.split('=').nth(1).unwrap().split('#').next().unwrap().trim();
    if let Some((base, shift)) = rhs.split_once("<<") {
        let base: u64 = base.trim().parse().expect("parse base of <<");
        let shift: u32 = shift.trim().parse().expect("parse shift amount");
        (base << shift) as f64
    } else {
        rhs.parse::<u64>().unwrap_or_else(|_| panic!("{} literal not an int or shift", name)) as f64
    }
}

/// Parse a `<NAME> = <int>` assignment out of any Python-syntax source. Exactly
/// one integer definition must exist, so a constant that grows a second one
/// fails loudly instead of reading the first.
fn parse_named_int_constant(path: &str, name: &str) -> u64 {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} must be readable from the project root: {}", path, e));
    let defs: Vec<u64> = src
        .lines()
        .filter_map(|l| {
            let rest = l.trim_start().strip_prefix(name)?;
            let rhs = rest.trim_start().strip_prefix('=')?;
            // `==` is a comparison, not a definition.
            if rhs.starts_with('=') {
                return None;
            }
            rhs.split('#').next().unwrap().trim().parse::<u64>().ok()
        })
        .collect();
    assert_eq!(defs.len(), 1, "{} must define {} exactly once as an int literal", path, name);
    defs[0]
}

/// The centroid scale is twinned three ways; drift rescales every centroid.
#[test]
fn test_kmeans_fixed_point_constants_match_twins() {
    use crate::pipeline::mpc::{SK_CENTROID_YMAX_SLACK_BITS, SK_FRAC_BITS};
    const MPC: &str = "mp-spdz/secure_kmeans.mpc";
    const PY: &str = "scripts/plaintext_kmeans_reference.py";

    for path in [MPC, PY] {
        assert_eq!(
            parse_named_int_constant(path, "FRAC_BITS"),
            SK_FRAC_BITS as u64,
            "{}: FRAC_BITS drifted from parameters.rs::SK_FRAC_BITS",
            path
        );
        assert_eq!(
            parse_named_int_constant(path, "CENTROID_YMAX_SLACK_BITS"),
            SK_CENTROID_YMAX_SLACK_BITS as u64,
            "{}: CENTROID_YMAX_SLACK_BITS drifted from parameters.rs",
            path
        );
        // The third twin is a formula, pinned textually: RECIP_BITS =
        // FRAC_BITS + ceil_log2(N) + CENTROID_YMAX_SLACK_BITS.
        let src = std::fs::read_to_string(path).unwrap();
        assert!(
            src.lines().any(|l| {
                l.contains("FRAC_BITS")
                    && l.contains("CENTROID_YMAX_SLACK_BITS")
                    && l.contains("ceil_log2")
            }),
            "{}: RECIP_BITS must stay FRAC_BITS + ceil_log2(N) + CENTROID_YMAX_SLACK_BITS",
            path
        );
    }

    // The tape truncates the centroid product at a hard-coded width whose
    // TruncRing precondition is what the Rust guard's strict term enforces.
    assert_eq!(parse_named_int_constant(MPC, "PROD_BITS"), 63);
}

// Drift silently rescales every revealed row of Y_π.
#[test]
fn test_norm_scale_matches_python() {
    assert_eq!(parse_python_int_constant("NORM_SCALE"), crate::constants::NORM_SCALE);
    assert_eq!(crate::constants::norm_scale_for_preshift(2), crate::constants::NORM_SCALE);
}

// Drift desynchronises the in-MPC edge-filter comparison width from the
// Rust-side worst-case bound.
#[test]
fn test_agg_cell_bits_matches_python() {
    assert_eq!(parse_python_int_constant("AGG_CELL_BITS"), crate::constants::AGG_CELL_BITS as f64);
}

#[test]
fn test_activity_markers_match_python() {
    // The merge tree uses START/END as sentinels and the edge filter preserves
    // their edges, so the marker ids are load-bearing on both sides.
    let py = std::fs::read_to_string("mp-spdz/mpc_utils/constants.py")
        .expect("mp-spdz/mpc_utils/constants.py must exist - run from project root");

    fn parse_int(py: &str, name: &str) -> u32 {
        let line = py
            .lines()
            .find(|l| l.trim_start().starts_with(name))
            .unwrap_or_else(|| panic!("{} not defined in constants.py", name));
        let rhs = line.split('=').nth(1).unwrap().split('#').next().unwrap().trim();
        rhs.parse::<u32>().unwrap_or_else(|_| panic!("parse {} literal", name))
    }

    assert_eq!(parse_int(&py, "START_ACT"), crate::types::START);
    assert_eq!(parse_int(&py, "END_ACT"), crate::types::END);
    // encoding_trigram.py derives FIRST_USER_ACT = END_ACT + 1.
    assert_eq!(crate::types::FIRST_USER_ACTIVITY, parse_int(&py, "END_ACT") + 1);
}

#[test]
fn test_expand_hash_weights_matches_cpython() {
    // Pinned against CPython `random.Random(seed).randrange(1, 1 << 48)`, the
    // sequence the tape consumed as HASH_W; drift breaks the printed-seed audit.
    let cases: [(u64, [u64; 4]); 4] = [
        (0x46494E47, [55110090862613, 245196863651752, 69057385699961, 133206368475102]),
        (0, [108438666086350, 250584532339885, 11396354577494, 271752282791483]),
        (u64::MAX, [70020945567359, 174162375961153, 128368892936274, 26747403298244]),
        (1234567890123456789, [51375772155106, 166069125574836, 264448707952529, 196074130245037]),
    ];
    for (seed, expected) in cases {
        let w = crate::mpc::expand_hash_weights(seed, 4);
        assert_eq!(w, expected, "seed {seed}");
        assert!(w.iter().all(|&v| (1..(1u64 << 48)).contains(&v)));
    }
}

#[test]
fn test_encoding_flag_ladder_matches_mpc_dispatch() {
    // `shared::encoding_flag` is the integer both extract programs dispatch on;
    // a swapped arm silently runs the wrong encoder.
    fn squash(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    let shared = std::fs::read_to_string("src/pipeline/mpc/shared.rs")
        .expect("src/pipeline/mpc/shared.rs must exist - run from project root");
    let shared_sq = squash(&shared);
    // Flag 2 is a retired slot, kept so the other flags never renumber: no Rust
    // arm may claim it and both .mpc programs refuse it.
    for (variant, flag) in
        [("Bigram", 0), ("Ef", 1), ("Activity", 3), ("Trigram", 4), ("SkipBigram", 5)]
    {
        let arm = format!("Encoding::{}=>{}", variant, flag);
        assert!(shared_sq.contains(&arm), "shared.rs encoding_flag must map {}", arm);
    }
    assert!(!shared_sq.contains("=>2,"), "encoding flag 2 is retired; assign a NEW flag");

    let mapping: [(usize, &str); 4] = [
        (1, "encoding_ef"),
        (3, "encoding_activity"),
        (4, "encoding_trigram"),
        (5, "encoding_skip_bigram"),
    ];
    for file in ["mp-spdz/secure_extract.mpc", "mp-spdz/secure_extract_rsecret.mpc"] {
        let src = std::fs::read_to_string(file)
            .unwrap_or_else(|_| panic!("{} must exist - run from project root", file));
        let lines: Vec<String> = src.lines().map(squash).collect();

        assert!(
            lines.iter().any(|l| l.contains("ENCODING_IS_BIGRAM=(ENCODING==0)")),
            "{}: flag-0 bigram predicate missing",
            file
        );
        assert!(
            lines.iter().any(|l| l.contains("assertENCODING!=2")),
            "{}: the retired flag-2 slot must stay refused",
            file
        );
        assert!(
            lines
                .windows(2)
                .any(|w| w[0] == "else:"
                    && w[1] == "encoding_extract=encoding_bigram.extract_features"),
            "{}: ladder default must select encoding_bigram",
            file
        );

        for (flag, encoder) in mapping {
            let guard = format!("ENCODING=={}", flag);
            let expected = format!("encoding_extract={}.extract_features", encoder);
            let hit = lines
                .windows(2)
                .find(|w| w[0].contains(&guard) && w[1].starts_with("encoding_extract="))
                .unwrap_or_else(|| {
                    panic!("{}: no encoder assignment guarded by ENCODING == {}", file, flag)
                });
            assert_eq!(hit[1], expected, "{}: flag {} must select {}", file, flag, encoder);
        }
    }
}

// --- The published release rule: the edge filter and the dead-end pruning ---

// Fixture shared with bridge/test_discovery.py::TestEdgeFilterRustParity:
// START->a 100, a->b 10, b->c 1, c->END 100.
fn edge_filter_parity_dfg() -> crate::types::ClusterDFG {
    use std::collections::HashMap;
    let mut edges: HashMap<(u32, u32), f64> = HashMap::new();
    edges.insert((START, 3), 100.0);
    edges.insert((3, 4), 10.0);
    edges.insert((4, 5), 1.0);
    edges.insert((5, END), 100.0);
    crate::types::ClusterDFG { cluster_id: 0, size: 4, edges }
}

#[test]
fn test_edge_filter_exempts_start_and_end_edges() {
    // Cutoff = max(2, 0.5 * 10) = 5: the max is over ACTIVITY edges only. Had
    // the START weight 100 entered it, the cutoff would be 50 and a->b dropped.
    let mut dfgs = vec![edge_filter_parity_dfg()];
    p4_aggregate::filter_dfg_edges(&mut dfgs, 2.0, 0.5);
    let edges = &dfgs[0].edges;
    assert!(edges.contains_key(&(3, 4)), "a->b (10 >= 5) must survive");
    assert!(!edges.contains_key(&(4, 5)), "b->c (1 < 5) must be dropped");
    assert_eq!(edges.len(), 3, "start/end + a->b remain");

    // Start and end edges survive any cutoff.
    let mut dfgs = vec![edge_filter_parity_dfg()];
    p4_aggregate::filter_dfg_edges(&mut dfgs, 1000.0, 0.0);
    let edges = &dfgs[0].edges;
    assert!(edges.contains_key(&(START, 3)));
    assert!(edges.contains_key(&(5, END)));
    assert!(!edges.contains_key(&(3, 4)));
    assert!(!edges.contains_key(&(4, 5)));
}

#[test]
fn test_edge_filter_int_args() {
    use crate::mpc::rounds::edge_filter_int_args;
    // The campaign cutoff (2, 1/50) and its exact integer twin.
    assert_eq!(edge_filter_int_args(2.0, 0.02), Some((2, 200)));
    assert_eq!(edge_filter_int_args(0.0, 0.05), Some((0, 500)));
    assert_eq!(edge_filter_int_args(3.5, 0.0), Some((4, 0)));
    // No exact decimal form at scale 10^4, or out of range.
    assert_eq!(edge_filter_int_args(2.0, 0.12345), None);
    assert_eq!(edge_filter_int_args(2.0, 1.5), None);
    assert_eq!(edge_filter_int_args(-1.0, 0.02), None);
    assert_eq!(edge_filter_int_args(3e12, 0.02), None); // >= 2^AGG_CELL_BITS
}

#[test]
fn test_edge_filter_matches_in_mpc_integer_boundary() {
    use std::collections::HashMap;
    // 0.07 * 100 = 7.000000000000001 in f64, so the float rule drops a cell of
    // exactly 7 while the in-MPC rational rule keeps it: the federated graph
    // would lose an edge the plaintext tiers keep.
    assert_eq!(crate::mpc::rounds::edge_filter_int_args(0.0, 0.07), Some((0, 700)));
    let mut edges: HashMap<(u32, u32), f64> = HashMap::new();
    edges.insert((START, 3), 1000.0);
    edges.insert((3, 4), 100.0);
    edges.insert((4, 5), 7.0);
    edges.insert((5, END), 1000.0);
    let mut dfgs = vec![crate::types::ClusterDFG { cluster_id: 0, size: 4, edges }];
    p4_aggregate::filter_dfg_edges(&mut dfgs, 0.0, 0.07);
    assert!(dfgs[0].edges.contains_key(&(4, 5)), "a cell on the exact cutoff must survive");
    // One below the cutoff still goes, under either form.
    let mut dfgs = vec![dfgs[0].clone()];
    dfgs[0].edges.insert((4, 5), 6.0);
    p4_aggregate::filter_dfg_edges(&mut dfgs, 0.0, 0.07);
    assert!(!dfgs[0].edges.contains_key(&(4, 5)));
}

/// a=3, b=4, c=5, d=6, e=7: START->a->b->END is the backbone; c is a start
/// activity that reaches no end, d an end activity that no start reaches, e a
/// successor of a that reaches no end.
fn dead_end_dfg() -> crate::types::ClusterDFG {
    use std::collections::HashMap;
    let mut edges: HashMap<(u32, u32), f64> = HashMap::new();
    edges.insert((START, 3), 5.0);
    edges.insert((3, 4), 5.0);
    edges.insert((4, END), 5.0);
    edges.insert((START, 5), 1.0);
    edges.insert((6, END), 1.0);
    edges.insert((3, 7), 2.0);
    edges.insert((START, END), 1.0);
    crate::types::ClusterDFG { cluster_id: 0, size: 7, edges }
}

#[test]
fn test_prune_dead_ends_keeps_only_start_to_end_paths() {
    let mut dfgs = vec![dead_end_dfg()];
    p4_aggregate::prune_dead_ends(&mut dfgs);
    let mut kept: Vec<(u32, u32)> = dfgs[0].edges.keys().copied().collect();
    kept.sort();
    assert_eq!(kept, vec![(START, END), (START, 3), (3, 4), (4, END)]);
    assert_eq!(dfgs[0].size, 7, "the pruning never touches the size field");
}

#[test]
fn test_prune_dead_ends_is_idempotent_and_keeps_unfiltered_graphs() {
    let mut once = vec![dead_end_dfg()];
    p4_aggregate::prune_dead_ends(&mut once);
    let mut twice = once.clone();
    p4_aggregate::prune_dead_ends(&mut twice);
    assert_eq!(once[0].edges, twice[0].edges);
    // Every activity of an unfiltered graph lies on the path of its own trace.
    let mut full = vec![edge_filter_parity_dfg()];
    let before = full[0].edges.clone();
    p4_aggregate::prune_dead_ends(&mut full);
    assert_eq!(full[0].edges, before);
}

#[test]
fn test_aggregate_compile_args_prune_inside_the_mpc() {
    let args = crate::mpc::rounds::aggregate_compile_args(927, 927, 324, 6, 5_000, 8, 2, 200);
    assert_eq!(
        args.len(),
        10,
        "TOTAL_N N_REVEALED D K SECURE_LABELS LABEL_OFFSET N_THREADS EMC EFN PRUNE"
    );
    assert_eq!(args[4], "1", "SECURE_LABELS=1 on every pipeline run");
    assert_eq!(args[9], "1", "PRUNE=1 on every pipeline run");
    let src = std::fs::read_to_string("mp-spdz/secure_aggregate.mpc").unwrap();
    assert!(
        src.contains("PRUNE          = int(program.args[10])"),
        "the tape reads PRUNE at slot 10"
    );
}

#[test]
fn test_parse_start_total() {
    let out = "=== Step 3c ===\nSTART_TOTAL 927\nROW_START 0\n";
    assert_eq!(crate::mpc::rounds::parse_start_total(out).unwrap(), 927);
    assert!(crate::mpc::rounds::parse_start_total("ROW_START 0\n").is_err());
}

#[test]
fn test_build_dfgs_from_dense_pure_counts() {
    // Round-2 cells are pure counts: the assembly passes them through exactly
    // and emits no edge for a PAD cell.
    let d = 5usize; // PAD=0, START=1, END=2, activities 3-4
    let mut row = vec![0.0f64; d * d];
    row[d + 3] = 12.0; // (START, 3)
    row[3 * d + 4] = 7.0; // (3, 4)
    row[4 * d + 3] = 5.0; // (4, 3)
    row[4 * d + 2] = 12.0; // (4, END)
    row[3] = 9.0; // (PAD, 3)
    row[3 * d] = 4.0; // (3, PAD)

    let dfgs = p4_aggregate::build_dfgs_from_dense(&[row], d, &[12]);
    assert_eq!(dfgs.len(), 1);
    let dfg = &dfgs[0];
    assert_eq!(dfg.size, 12);
    assert_eq!(dfg.edges.len(), 4, "PAD cells emit no edge");
    assert_eq!(dfg.edges[&(START, 3)], 12.0);
    assert_eq!(dfg.edges[&(3, 4)], 7.0);
    assert_eq!(dfg.edges[&(4, 3)], 5.0);
    assert_eq!(dfg.edges[&(4, END)], 12.0);
}

#[test]
fn test_dense_bigram_rows_match_persisted_layout() {
    // The dense rows must land where secure_extract persists them and where
    // plaintext_sequence_reference.py puts them: all at `a*d + b`.
    let traces = vec![make_trace("c1", &[3, 4, 4]), make_trace("c2", &[5])];
    let d = 8usize;
    let (index, raw) = p2_bigrams::extract_plaintext(&traces);
    let dense = p2_bigrams::dense_bigram_rows(&raw, &index, d);

    assert_eq!(dense.len(), 2);
    for row in &dense {
        assert_eq!(row.len(), d * d);
    }
    let s = START as usize;
    let e = END as usize;
    assert_eq!(dense[0][s * d + 3], 1.0);
    assert_eq!(dense[0][3 * d + 4], 1.0);
    assert_eq!(dense[0][4 * d + 4], 1.0);
    assert_eq!(dense[0][4 * d + e], 1.0);
    assert_eq!(dense[0].iter().sum::<f64>(), 4.0);
    assert_eq!(dense[1][s * d + 5], 1.0);
    assert_eq!(dense[1][5 * d + e], 1.0);
    assert_eq!(dense[1].iter().sum::<f64>(), 2.0);
}

// --- Ring-width guards and the compile args that carry them ---

#[test]
fn test_secure_kmeans_argmin_bit_bound() {
    use crate::config::Encoding;
    use crate::pipeline::mpc::encoding_mass_bound_for_test as mass;
    use crate::pipeline::mpc::secure_kmeans_argmin_bits_for_test as bits;
    let m = |e, l| u64::try_from(mass(e, l)).unwrap();

    // Bigram trunc-10 x 2 parties at N=1050, K_PROJ=40: ‖x‖₁ = 21, inside the
    // 63-bit signed-ring budget.
    assert_eq!(m(Encoding::Bigram, 20), 21);
    assert_eq!(bits(40, 1050, 21), 51, "bigram benchmark shape");

    // The dominant term is the distance, which does not depend on N.
    assert_eq!(m(Encoding::Activity, 12), 12);
    // 54, not 53: this cell is centroid-product dominated (distance 47), and
    // that term is the strict signed width of `s_j[d]*r_j`.
    assert_eq!(bits(14, 129_985, 12), 54, "full-N RTFM SL must fit at -R 64");
    assert_eq!(bits(40, 1050, 21), bits(40, 4200, 21));

    // `y_max` is the per-encoding L1 mass, so EF's quadratic pair mass is
    // covered: a merged length a linear proxy would wave through is refused.
    assert_eq!(m(Encoding::Ef, 100), 5151);
    assert!(bits(40, 500, 101) <= 63, "the linear-in-L proxy would have passed");
    assert!(bits(40, 500, m(Encoding::Ef, 100)) > 63, "EF pair mass must be refused");

    // Above the reciprocal's precision headroom the centroid rounding would
    // exceed one ulp: a guaranteed refusal, not a silent coarsening.
    assert_eq!(bits(40, 1050, 1 << 16), 64);

    // A centroid product exactly on the ring bound must be refused: the tape's
    // `Trunc(prod, 63, ..)` adds 2^62 and needs |prod| < 2^62.
    assert_eq!(bits(40, 1 << 26, 21), 64, "product at 2^62 must be refused");
    assert!(bits(40, 1 << 25, 21) <= 63, "one bit below it still fits");

    // Degenerate inputs must not underflow the log.
    assert_eq!(bits(1, 1, 1), 35);
}

#[test]
fn test_merged_len_bound_is_the_sum_of_the_party_lengths() {
    use crate::config::Encoding;
    use crate::pipeline::mpc::encoding_mass_bound_for_test as mass;
    use crate::pipeline::mpc::merged_len_bound_for_test as merged;
    use crate::pipeline::mpc::secure_kmeans_argmin_bits_for_test as bits;
    let m = |e, l| u64::try_from(mass(e, l)).unwrap();

    // Parties of unequal length: the sum, not L * n_parties.
    assert_eq!(merged(&[32, 12], 32, 2), 44);
    assert_eq!(merged(&[48, 48], 48, 2), 96);
    // A pinned L off the alignment pads each party past it; L * n_parties caps.
    assert_eq!(merged(&[12, 12], 10, 2), 20);
    assert_eq!(merged(&[8, 8, 4], 8, 3), 20);

    // BPIC 2013 by role under eventually-follows (N=7545, k_proj=24): refused at
    // L * n_parties = 64, admitted at the per-party lengths 32 and 12.
    assert_eq!(bits(24, 7545, m(Encoding::Ef, 64)), 64);
    assert_eq!(bits(24, 7545, m(Encoding::Ef, merged(&[32, 12], 32, 2))), 62);
}

#[test]
fn test_kmeans_compile_args_carry_the_guard_width() {
    use crate::config::Encoding;
    use crate::mpc::rounds::kmeans_compile_args;
    use crate::pipeline::mpc::encoding_mass_bound_for_test as mass;
    use crate::pipeline::mpc::secure_kmeans_argmin_bits_for_test as bits;

    // ARGMIN_BITS is the ninth compile arg and must be the width the guard
    // used: a narrower tape wraps silently, a wider one wastes the saving.
    let (k_proj, n_cases) = (19usize, 1050usize);
    let y_max = u64::try_from(mass(Encoding::Activity, 56)).unwrap();
    let w = bits(k_proj, n_cases, y_max);
    assert_eq!(w, 52, "Sepsis SL activity pin (merged L=56)");

    let args = kmeans_compile_args(n_cases, k_proj, 6, 5, 8, 379_050, true, w, 10);
    assert_eq!(args.len(), 10, "the tape reads ten compile args");
    assert_eq!(
        args,
        vec!["1050", "19", "6", "5", "8", "1", "379050", "1", "52", "10"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
    assert_eq!(args[8], w.to_string(), "ARGMIN_BITS == the guard's width");
    assert_eq!(args[9], "10", "MIN_CLUSTER is the tenth arg");

    let rtfm = bits(14, 129_985, u64::try_from(mass(Encoding::Activity, 12)).unwrap());
    assert_eq!(rtfm, 54, "centroid-product dominated (strict TruncRing width)");
    assert_eq!(kmeans_compile_args(129_985, 14, 10, 5, 8, 0, false, rtfm, 10)[8], "54");
}

#[test]
fn test_merge_threshold_rides_the_last_tape() {
    use crate::constants::MIN_CLUSTER_SIZE;
    use crate::mpc::rounds::{kmeans_compile_args, sequence_compile_args};

    // The oblivious small-cluster merge runs once, in the last tape of the
    // middle step: secure_kmeans carries MIN_CLUSTER_SIZE as its tenth arg when it is
    // the last tape, 0 when the refinement follows, and the refinement carries
    // it as its eleventh (behind PROFILE=0).
    assert_eq!(MIN_CLUSTER_SIZE, 10, "the RP merge floor the SL merge twins");
    let alone = kmeans_compile_args(1050, 19, 6, 5, 8, 379_050, true, 52, MIN_CLUSTER_SIZE);
    assert_eq!(alone[9], "10");
    let seeded = kmeans_compile_args(1050, 19, 6, 5, 8, 379_050, false, 52, 0);
    assert_eq!(seeded[9], "0", "k-means merges nothing when the refinement follows");
    let seq = sequence_compile_args(1050, 361, 6, 5, 8, 399_000, true, 1, MIN_CLUSTER_SIZE);
    assert_eq!(seq.len(), 11, "PROFILE (tenth) then MIN_CLUSTER (eleventh)");
    assert_eq!(&seq[9..], ["0", "10"]);
    assert_eq!(seq[..9], ["1050", "361", "6", "5", "8", "1", "399000", "1", "1"].map(String::from));
}

#[test]
fn test_relabel_compile_args_carry_the_tail_comparison_width() {
    use crate::mpc::rounds::{relabel_compile_args, relabel_tail_cmp_bits};

    // TAIL_CMP_BITS is the tenth compile arg, the twin of the score bound
    // `81 * k_proj * 2^(32 + p)` plus bits for the difference and the sign.
    assert_eq!(relabel_tail_cmp_bits(19, 2), 48, "Sepsis rsec activity pin");
    // A raised preshift widens the bound one for one; more projected dimensions
    // only through ceil_log2(81*k_proj).
    assert_eq!(relabel_tail_cmp_bits(19, 4), 50);
    assert_eq!(relabel_tail_cmp_bits(64, 2), 50);

    let args =
        relabel_compile_args(1050, 379_050, 8, 6, 401_100, true, Some((19, 380_100, 400_050, 2)));
    assert_eq!(
        args,
        vec!["1050", "379050", "8", "6", "401100", "1", "19", "380100", "400050", "48"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );

    // Without the hybrid tail the tape reads six args.
    assert_eq!(
        relabel_compile_args(1050, 379_050, 8, 6, 401_100, false, None),
        vec!["1050", "379050", "8", "6", "401100", "0"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_rsec_norm_preshift_bounds() {
    use crate::config::Encoding;
    use crate::pipeline::mpc::rsec_norm_preshift_for_test as preshift;
    // Small configs stay at the default preshift (results-identical).
    assert_eq!(preshift(Encoding::Bigram, 6, 6).unwrap(), 2);
    assert_eq!(preshift(Encoding::Ef, 6, 6).unwrap(), 2);
    // Sepsis-scale EF (auto-L=24/party x 2 parties, K_PROJ=80) must auto-raise,
    // and the chosen preshift must actually fit the sfix range.
    let p = preshift(Encoding::Ef, 48, 80).unwrap();
    assert!(p > 2 && p.is_multiple_of(2), "EF at L=48 and K_PROJ=80 needs a raised preshift: {p}");
    let b1: u128 = (48 + 1) * (48 + 2) / 2;
    assert!((80 * b1 * b1) >> p < (1 << 14), "‖y‖² must fit sfix after >> {p}");
    // Bigram at the same scale keeps the default.
    let p_bg = preshift(Encoding::Bigram, 48, 80).unwrap();
    let b1_bg: u128 = 49;
    assert!((80 * b1_bg * b1_bg) >> p_bg < (1 << 14));
    // A cell mass >= 2^14 can never fit the sfix cast: EF at absurd length.
    assert!(preshift(Encoding::Ef, 300, 80).is_err());
}

// --- The projected dimension ---

#[test]
fn test_adaptive_kproj_rule_and_floor() {
    use crate::config::Encoding;
    use crate::constants::{ADAPTIVE_KPROJ_MIN, ADAPTIVE_KPROJ_SAFETY_FACTOR};
    use crate::pipeline::mpc::compute_adaptive_kproj_for_test as kproj;
    use crate::pipeline::mpc::encoding_dim;

    // At d = 13 and per-party sparsity s = 10 the CS bound must be evaluated at
    // the encoding's ambient dim, not at d^2 = 169 for every encoding.
    let k = |e: Encoding| {
        kproj(10.0, encoding_dim(e, 13) as f64, ADAPTIVE_KPROJ_SAFETY_FACTOR, ADAPTIVE_KPROJ_MIN)
    };
    assert_eq!(encoding_dim(Encoding::Bigram, 13), 169);
    assert_eq!(encoding_dim(Encoding::Ef, 13), 169);
    assert_eq!(encoding_dim(Encoding::SkipBigram, 13), 338);
    assert_eq!(encoding_dim(Encoding::Trigram, 13), 1000);
    assert_eq!(encoding_dim(Encoding::Activity, 13), 13);
    assert_eq!(k(Encoding::Bigram), 96);
    assert_eq!(k(Encoding::Ef), 96);
    assert_eq!(k(Encoding::SkipBigram), 119);
    assert_eq!(k(Encoding::Trigram), 156);
    // An ambient dim below the CS floor is degenerate: the fallback returns the
    // floor, which the orchestrator then clamps to enc_dim (identity).
    assert_eq!(k(Encoding::Activity), ADAPTIVE_KPROJ_MIN);

    // s = 0 (no party has a two-event trace) makes the CS bound NaN; the
    // fallback returns the floor instead of underflowing `0usize - 1`.
    assert_eq!(kproj(0.0, 1024.0, 0.85, 20), 20);
    // A finite bound below min_kproj + 1 (here cs_full = 0) must also clamp.
    assert_eq!(kproj(0.5, 1.0, 0.85, 20), 20);
    // Non-degenerate sparsity keeps the CS-driven selection.
    assert!(kproj(30.0, 1024.0, 0.85, 20) > 20);
}

#[test]
fn test_estimate_sparsity_is_encoding_aware() {
    use crate::config::Encoding;
    use crate::mpc::CaseData;
    use crate::pipeline::mpc::estimate_sparsity_for_test as sparsity;
    // Slice a b c b: distinct activities {a,b,c} = 3; bigrams {ab, bc, cb} = 3;
    // distance-2 pairs {ac, bb} = 2 on top of the bigrams; trigrams
    // {abc, bcb} = 2; eventually-follows {ab, ac, bc, bb, cb} = 5.
    let c = CaseData { id: "c1".into(), events: vec![(0, 3), (1, 4), (2, 5), (3, 4)] };
    let one = |e| sparsity(e, std::slice::from_ref(&c));
    assert_eq!(one(Encoding::Activity), 3.0);
    assert_eq!(one(Encoding::Bigram), 3.0);
    assert_eq!(one(Encoding::SkipBigram), 5.0);
    assert_eq!(one(Encoding::Trigram), 2.0);
    assert_eq!(one(Encoding::Ef), 5.0);
}

// --- The suppression threshold k_priv ---

// Trace whose activity sequence is a deterministic function of `variant_id`:
// equal ids give the same variant class, distinct ids different classes.
fn variant_trace(case_idx: usize, variant_id: usize) -> (String, EncodedTrace) {
    let mut acts: Vec<u32> = Vec::new();
    let mut x = variant_id + 1;
    while x > 0 {
        acts.push((x % 16) as u32 + 3);
        x /= 16;
    }
    make_trace(&format!("c{case_idx}"), &acts)
}

// Profile rows at the granularity the MPC fingerprints (encoding_matrix rows).
fn bigram_profiles(traces: &[(String, EncodedTrace)]) -> crate::types::FeatureMatrix {
    crate::pipeline::mpc::encoding_profiles_for_test(crate::config::Encoding::Bigram, traces, 32)
        .expect("bigram profiles")
}

#[test]
fn test_budget_bounded_k_priv_balanced_classes_raises() {
    // 34 classes of ~147 cases each, no singletons: the selection raises k_priv
    // well past k0, since privacy improves at no cost.
    let n = 5000;
    let n_classes = 34;
    let traces: Vec<(String, EncodedTrace)> =
        (0..n).map(|i| variant_trace(i, i % n_classes)).collect();

    let profiles = bigram_profiles(&traces);
    let v = crate::pipeline::mpc::measure_variant_anonymity(&profiles);
    assert_eq!(v.n_cases, n);
    assert_eq!(v.singletons(), 0);
    let min_class = v.class_sizes.iter().copied().min().unwrap();
    assert!(min_class > 5, "every class should be large; min={min_class}");

    let k_priv = crate::pipeline::mpc::select_k_priv_for_test(&profiles, 5, 0.10);
    // No suppression occurs until k exceeds the smallest class, so the budget
    // lets k_priv climb to at least min_class.
    assert!(k_priv >= 5, "k_priv {k_priv} must be at least k0=5");
    assert!(k_priv >= min_class, "balanced classes should raise k_priv: {k_priv}");
}

#[test]
fn test_budget_bounded_k_priv_respects_tau_with_singletons() {
    // 644 singletons in 1050 cases: at k=2 every singleton is suppressed, so
    // the budget cannot be met above k0 and the selector pins k_priv = k0.
    let mut traces: Vec<(String, EncodedTrace)> = Vec::with_capacity(1050);
    for i in 0..644 {
        traces.push(variant_trace(i, i));
    }
    let class_count = 30usize;
    let cases_per_class = 13usize;
    let leftover = 1050 - 644 - class_count * cases_per_class;
    for c in 0..class_count {
        for k in 0..cases_per_class {
            traces.push(variant_trace(644 + c * cases_per_class + k, 100_000 + c));
        }
    }
    for k in 0..leftover {
        traces.push(variant_trace(1000 + k, 100_000));
    }

    let profiles = bigram_profiles(&traces);
    let v = crate::pipeline::mpc::measure_variant_anonymity(&profiles);
    assert_eq!(v.n_cases, 1050);
    assert!(v.singleton_fraction() > 0.20);

    // tau=0.10 cannot be met (singletons alone are 61%), so k_priv stays at k0.
    let k_priv = crate::pipeline::mpc::select_k_priv_for_test(&profiles, 5, 0.10);
    assert_eq!(k_priv, 5, "tau infeasible above k0 -> pin at k0; got {k_priv}");
}

#[test]
fn test_budget_bounded_k_priv_counts_zero_phi_rows_as_suppressed() {
    // Zero-φ rows must count as always suppressed: they L2-normalise to zero
    // and the hybrid reveal forces them out of the anonymous set. Fixture:
    // 5 zero rows + nonzero classes of sizes {6,4,3} (n=18), tau=0.40.
    let mut profiles: crate::types::FeatureMatrix = Vec::new();
    for _ in 0..5 {
        profiles.push(vec![0.0, 0.0, 0.0, 0.0]);
    }
    for _ in 0..6 {
        profiles.push(vec![1.0, 0.0, 0.0, 0.0]);
    }
    for _ in 0..4 {
        profiles.push(vec![0.0, 1.0, 0.0, 0.0]);
    }
    for _ in 0..3 {
        profiles.push(vec![0.0, 0.0, 1.0, 0.0]);
    }
    assert_eq!(profiles.len(), 18);

    // effective_suppressed(k) = 5 zeros + nonzero rows in classes < k, so the
    // selector stops at 3; zeros as one class would have climbed to k_priv=5.
    let k_priv = crate::pipeline::mpc::select_k_priv_for_test(&profiles, 2, 0.40);
    assert_eq!(k_priv, 3, "zero-φ-aware budget should choose k_priv=3; got {k_priv}");
}

#[test]
fn test_profile_class_matches_encoding_granularity() {
    // The two traces are bigram-identical but trigram-distinct, so under the
    // trigram embedding the guard must place them in separate classes:
    // anonymity is enforced at the granularity of what the reveal opens.
    let traces =
        vec![make_trace("eulerA", &[3, 4, 3, 5, 3, 4]), make_trace("eulerB", &[3, 5, 3, 4, 3, 4])];
    assert_eq!(
        crate::pipeline::mpc::measure_variant_anonymity(&bigram_profiles(&traces)).distinct,
        1,
        "the same padded bigram multiset must merge into one class"
    );
    let profiles = crate::pipeline::mpc::encoding_profiles_for_test(
        crate::config::Encoding::Trigram,
        &traces,
        8,
    )
    .expect("trigram profiles");
    let v = crate::pipeline::mpc::measure_variant_anonymity(&profiles);
    assert_eq!(v.distinct, 2, "trigram-distinct traces must form separate classes");
}

// --- Round 1 output, the merge key and the extract-once slab ---

#[test]
fn test_parse_row_output_backbone_derives_from_zero_mask() {
    // No per-row flag is opened: tail rows reveal as all-zero, backbone rows
    // are unit-norm, and the N_BACKBONE count anchors the consistency check.
    let ok = "N_BACKBONE 1\nROW_START 0\nB 0 5\nB 1 -3\nROW_END\n\
              ROW_START 1\nB 0 0\nB 1 0\nROW_END\n";
    let (rows, backbone) = crate::mpc::parse_row_output_backbone(ok, 2, 2).expect("valid output");
    assert_eq!(rows[0], vec![5u64, (-3i64) as u64]);
    assert_eq!(backbone, vec![true, false]);

    // A missing count line can only be dropped stdout: hard error.
    let missing = "ROW_START 0\nB 0 5\nROW_END\n";
    let err = crate::mpc::parse_row_output_backbone(missing, 1, 1);
    assert!(err.is_err(), "missing N_BACKBONE line must hard-error");
    assert!(format!("{}", err.unwrap_err()).contains("N_BACKBONE"));

    // A count disagreeing with the zero mask must fail loud, never silently
    // demote the row into the hybrid tail.
    let mismatch = "N_BACKBONE 2\nROW_START 0\nB 0 5\nROW_END\n\
                    ROW_START 1\nB 0 0\nROW_END\n";
    let err = crate::mpc::parse_row_output_backbone(mismatch, 2, 1);
    assert!(
        matches!(err, Err(crate::mpc::SpdzError::DegenerateReveal { n_nonzero: 1, n_backbone: 2 })),
        "a zero-mask vs N_BACKBONE mismatch must be the typed degenerate-reveal error"
    );
}

#[test]
fn test_merge_sorts_key_on_scaled_seconds() {
    // The MPC merge tree compares `(ts / MS_TO_SECS) * DICT_SIZE + act`, so two
    // same-second events must order by ACTIVITY, not by sub-second milliseconds,
    // or a party's own input breaks the merge tree's sorted precondition.
    use crate::constants::MS_TO_SECS;
    use crate::pipeline::data_loading::{
        integrate_traces, intersect_all_cases, merge_truncated_cases, prepare_cases_for_mpc,
    };
    use crate::types::{ActivityDictionary, PartyState, RawEvent};
    let ev = |case: &str, act: &str, ts: u64| RawEvent {
        case_id: case.to_string(),
        activity: act.to_string(),
        timestamp: ts,
    };
    // Same second, inverted: "b" (the higher activity id) carries the smaller ms.
    let p0 =
        PartyState { events: vec![ev("c1", "b", 1_000_000_500), ev("c1", "a", 1_000_000_900)] };
    let states = vec![p0];
    let cases = intersect_all_cases(&states);
    let dict = ActivityDictionary::from_activities(["a".to_string(), "b".to_string()].into())
        .expect("dictionary");
    let a = dict.encode_activity("a").unwrap();
    let b = dict.encode_activity("b").unwrap();
    assert!(a < b, "the fixture needs a < b to invert the ms order");

    let prepared = prepare_cases_for_mpc(&states[0], &cases, &dict, None);
    let acts: Vec<u32> = prepared[0].events.iter().map(|(_, x)| *x).collect();
    assert_eq!(acts, vec![a, b], "the party input must be sorted under the MPC key");
    assert!(
        prepared[0]
            .events
            .windows(2)
            .all(|w| (w[0].0 / MS_TO_SECS, w[0].1) <= (w[1].0 / MS_TO_SECS, w[1].1)),
        "non-decreasing under (ts / MS_TO_SECS, act)"
    );

    let merged = merge_truncated_cases(&[prepared]);
    let merged_acts: Vec<u32> = merged[0].1.iter().map(|e| e.activity).collect();
    assert_eq!(merged_acts, vec![a, b]);

    let integrated =
        integrate_traces(&states.iter().collect::<Vec<_>>(), &cases, &dict).expect("integrate");
    let pt_acts: Vec<u32> = integrated.traces[0].1.iter().map(|e| e.activity).collect();
    assert_eq!(pt_acts, merged_acts, "the plaintext tier must see the federated order");
}

#[test]
fn test_slab_shape_rejects_misaligned_parties() {
    use crate::mpc::{slab_shape, CaseData};
    let case = |id: &str| CaseData { id: id.to_string(), events: vec![(1000, 3)] };
    let full = vec![case("c1"), case("c2"), case("c3")];
    // A short vector: the party emitted only the cases it holds events for.
    let ragged = vec![full.clone(), vec![case("c1"), case("c3")]];
    assert!(slab_shape(&ragged).is_err());
    // Same length, different case at a slot: the silent-merge hazard.
    let swapped = vec![full.clone(), vec![case("c1"), case("c9"), case("c3")]];
    assert!(slab_shape(&swapped).is_err());
    assert!(slab_shape(&[full.clone(), full]).is_ok());
}

#[test]
fn test_input_payload_digest_separates_same_shaped_runs() {
    // The event CONTENT field of the --reuse-extraction slab key: without it a
    // second vertical split of one log at the same data_dir matches the whole
    // key and tail-reads the wrong slab.
    use crate::mpc::{input_payload_digest, CaseData};
    let ev = |ts: u64, act: u32| (ts, act);
    let base: Vec<Vec<CaseData>> = vec![
        vec![
            CaseData { id: "c1".into(), events: vec![ev(1000, 3), ev(3000, 4)] },
            CaseData { id: "c2".into(), events: vec![ev(2000, 3)] },
        ],
        vec![
            CaseData { id: "c1".into(), events: vec![ev(4000, 5)] },
            CaseData { id: "c2".into(), events: vec![ev(5000, 4)] },
        ],
    ];
    let d = |all: &Vec<Vec<CaseData>>| input_payload_digest(all, 8).unwrap();
    assert_eq!(d(&base), d(&base.clone()), "the same events give the same identity");

    // One timestamp changed; every other key field is identical.
    let mut retimed = base.clone();
    retimed[0][0].events[1] = ev(6000, 4);
    assert_ne!(d(&retimed), d(&base), "a re-exported timestamp is a different slab");

    // The same events split differently across the parties.
    let mut resplit = base.clone();
    resplit[0][1].events.push(ev(5000, 4));
    resplit[1][1].events.clear();
    assert_ne!(d(&resplit), d(&base), "a different vertical split is a different slab");

    // An activity relabelled inside one party's stream moves it too.
    let mut recoded = base.clone();
    recoded[1][0].events[0] = ev(4000, 6);
    assert_ne!(d(&recoded), d(&base));
}

// --- Refusals ---

#[test]
fn test_from_activities_rejects_reserved_names() {
    use crate::types::{ActivityDictionary, RESERVED_ACTIVITY_NAMES};
    // bridge/utils.py::load_dfg_from_json drops every edge touching one of
    // these, so accepting them would delete edges silently.
    for bad in RESERVED_ACTIVITY_NAMES {
        let acts = [bad.to_string(), "A".to_string()].into_iter().collect();
        assert!(
            ActivityDictionary::from_activities(acts).is_err(),
            "the reserved activity name {:?} must be refused",
            bad
        );
    }
    let ok = ["A".to_string(), "B".to_string()].into_iter().collect();
    assert!(ActivityDictionary::from_activities(ok).is_ok());
}

#[test]
fn test_kmeans_is_seed_deterministic_and_rejects_k_above_n() {
    // The outer seeded loop, not linfa's internal restarts, is what makes a run
    // reproducible.
    let features: crate::types::FeatureMatrix =
        vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![5.0, 5.0], vec![5.1, 5.0]];
    let a = p3_clustering::cluster_n_init(&features, 2, 10, 10).expect("k-means");
    let b = p3_clustering::cluster_n_init(&features, 2, 10, 10).expect("k-means");
    assert_eq!(a.assignments, b.assignments, "the same seeds must give the same labels");
    assert_eq!(a.assignments[0], a.assignments[1]);
    assert_eq!(a.assignments[2], a.assignments[3]);
    assert_ne!(a.assignments[0], a.assignments[2]);
    // k > n must be a typed error, not a panic inside linfa.
    let err = p3_clustering::cluster_n_init(&features, 10, 10, 1).unwrap_err();
    assert!(format!("{}", err).contains("exceeds"), "k > n must name the shape: {}", err);
}

#[test]
fn test_validate_mode_flags_accepts_the_recorded_modes() {
    let cfg = |extra: &[&str]| {
        let mut args = vec!["clustering_federated".to_string()];
        args.extend(extra.iter().map(|s| s.to_string()));
        crate::config::PipelineConfig::from_args(&args)
    };
    let sk = cfg(&["--mpc", "--secure-kmeans", "--k", "6"]);
    assert!(crate::pipeline::mpc::validate_mode_flags(&sk).is_ok());
    let seq = cfg(&["--mpc", "--secure-kmeans", "--k", "6", "--secure-clustering", "sequence"]);
    assert!(crate::pipeline::mpc::validate_mode_flags(&seq).is_ok());
    let rsec = cfg(&["--mpc", "--reveal-secret-projection"]);
    assert!(crate::pipeline::mpc::validate_mode_flags(&rsec).is_ok());

    // --secure-kmeans takes k as input: in-box k selection is not offered, and
    // the plaintext auto-k leaks per-dataset statistics.
    assert!(crate::pipeline::mpc::validate_mode_flags(&cfg(&["--mpc", "--secure-kmeans"])).is_err());
    assert!(crate::pipeline::mpc::validate_mode_flags(&cfg(&["--mpc"])).is_err());
    // The secure middle step is a --secure-kmeans-only step; clap rejects the
    // pair on the command line, so build it by hand.
    let mut rp_seq = cfg(&["--mpc", "--reveal-secret-projection"]);
    rp_seq.secure_clustering = crate::config::SecureClustering::Sequence;
    assert!(crate::pipeline::mpc::validate_mode_flags(&rp_seq).is_err());
}

#[test]
fn test_run_round2_secure_refuses_inexact_edge_cutoff() {
    // Disabling the in-MPC pre-reveal filter would open every aggregated cell,
    // so a cutoff with no exact integer form is refused. The refusal resolves
    // before any side effect, so the bogus SPDZ path is never touched.
    let cfg = crate::mpc::SpdzConfig {
        spdz_path: std::path::PathBuf::from("/nonexistent/mp-spdz"),
        ..crate::mpc::SpdzConfig::default()
    };
    let err = crate::mpc::run_round2_secure(&cfg, 10, 10, 4, 2, 0, 2, Some((2.0, 0.12345)))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no exact integer form"), "{err}");
    assert!(err.contains("refusing to widen the reveal"), "{err}");
}

#[test]
fn test_validate_k_against_cases() {
    use crate::pipeline::mpc::validate_k_against_cases;
    assert!(validate_k_against_cases(6, 1050, "agreed cases").is_ok());
    assert!(validate_k_against_cases(1050, 1050, "agreed cases").is_ok());
    // k > N: secure_kmeans.mpc strides by N/k = 0 and seeds every centroid
    // from row 0.
    let err = validate_k_against_cases(1051, 1050, "agreed cases").unwrap_err().to_string();
    assert!(err.contains("exceeds the 1050 agreed cases"), "{err}");
    let err2 = validate_k_against_cases(12, 9, "revealed backbone rows").unwrap_err().to_string();
    assert!(err2.contains("revealed backbone rows"), "{err2}");
    assert!(validate_k_against_cases(0, 1050, "agreed cases").is_err());
}

/// The truth matrix `scripts/attacks/attack_on_mpc_output.py --truth` scores
/// against: the raw encoded rows in the integrated case order, counts as
/// integers, and the case ids in that same order beside them.
#[test]
fn test_export_encoded_features_keeps_case_order_and_counts() {
    let traces = vec![make_trace("c2", &[3, 4, 5]), make_trace("c1", &[3, 3, 4])];
    let d = 6;
    let act = p2_bigrams::extract_activity(&traces, d);
    let dir = std::env::temp_dir().join(format!("fedtrac_encoded_dump_{}", std::process::id()));
    crate::pipeline::export::export_encoded_features(&act, &traces, &dir).expect("export");
    let csv = std::fs::read_to_string(dir.join("features_encoded.csv")).expect("matrix file");
    let ids = std::fs::read_to_string(dir.join("features_encoded_cases.txt")).expect("ids file");
    std::fs::remove_dir_all(&dir).ok();
    let rows: Vec<Vec<i64>> = csv
        .lines()
        .map(|l| l.split(',').map(|v| v.parse().expect("integer count")).collect())
        .collect();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.len() == d));
    assert_eq!(&rows[0][3..6], &[1, 1, 1], "row 0 is c2 = 3,4,5 once each");
    assert_eq!(&rows[1][3..6], &[2, 1, 0], "row 1 is c1 = 3 twice, 4 once");
    assert_eq!(ids.lines().collect::<Vec<_>>(), vec!["c2", "c1"], "ids in the integrated order");
}
