#!/usr/bin/env python3
"""The whole chain over persistence: secure_extract, the clustering, secure_aggregate, against a twin of every step."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_common as hc
from harness_common import (
    clear_persistence,
    compile_mpc,
    prune_dense,
    run_mpc,
    write_nparty_inputs,
)
from test_secure_extract import (
    extract_activity_plaintext,
    extract_bigrams_plaintext,
    generate_achlioptas_matrix,
    merge_events_plaintext,
    parse_row_output,
    prepare_nparty_data,
    retruncate_per_party,
    write_projection_matrix,
)
from test_secure_kmeans import parse_labels

from scripts.plaintext_kmeans_reference import lloyd_labels
from scripts.plaintext_sequence_reference import score_margins, sequence_labels

# Iterations of the Markov-chain hard-EM, in the tape and in the twin alike.
SEQ_MAX_ITER = 5
# Tie tolerance: the twin scores in float log2 and the tape in fixed point, so a label may flip at a tiny margin.
SEQ_MARGIN_TOL = 0.05

def aggregate_plaintext(histograms, labels, k, dict_size):
    """Clear twin of secure_aggregate.mpc's secret-label branch: per-cluster DFG accumulation and sizes."""
    d_sq = dict_size * dict_size
    dfg = [[0] * d_sq for _ in range(k)]
    sizes = [0] * k

    for hist, lab in zip(histograms, labels):
        sizes[lab] += 1
        for d in range(d_sq):
            dfg[lab][d] += hist[d]

    return dfg, sizes


START_ACT, END_ACT = 1, 2  # twin of mpc_utils/constants.py


def filter_dfg_plaintext(dfg, dict_size, min_count, frac_num, frac_den=10_000):
    """Integer twin of secure_aggregate.mpc Step 3b and p4_aggregate.rs::filter_dfg_edges."""
    d = dict_size
    out = [list(row) for row in dfg]
    act_cells = [f for f in range(d * d)
                 if f // d not in (0, START_ACT) and f % d not in (0, END_ACT)]
    for row in out:
        mx = max((row[f] for f in act_cells), default=0)
        for f in act_cells:
            w = row[f]
            if not (w >= min_count and w * frac_den >= frac_num * mx):
                row[f] = 0
    return out


def derive_sizes_from_dfg(mpc_dfg, dict_size: int) -> list[int]:
    """Cluster size from the DFG START-row sum, as src/mpc/rounds.rs::run_round2_secure derives it."""
    start_base = START_ACT * dict_size
    return [sum(row[start_base:start_base + dict_size]) for row in mpc_dfg]


ENCODING_LABEL = {0: "bigram", 3: "activity"}


def run_test(spdz_dir: Path, party_files, max_cases, k_proj, k, max_iter, seed,
             partial_lens=None, universe="intersection",
             secure_clustering: str = "kmeans", n_threads: int = 4,
             agg_threads=None, encoding: int = 0, ring=None,
             seq_max_iter: int = SEQ_MAX_ITER, timeout: int = 3600,
             narrow_argmin: bool = False, min_cluster: int = 0):
    """One full chain; agg_threads re-runs only the aggregate stage at each listed thread count, and min_cluster
    runs the oblivious small-cluster merge in the last tape of the chain."""
    assert secure_clustering in ("kmeans", "sequence"), secure_clustering
    assert encoding in ENCODING_LABEL, encoding
    sequence = secure_clustering == "sequence"
    n_parties = len(party_files)
    agg_threads = list(agg_threads) if agg_threads else [n_threads]
    print(f"\n{'='*62}")
    print(
        f"Full Pipeline Test  n_parties={n_parties}  max_cases={max_cases}  "
        f"k={k}  k_proj={k_proj}  max_iter={max_iter}  universe={universe}  "
        f"clustering={secure_clustering}  enc={ENCODING_LABEL[encoding]}  "
        f"threads={n_threads}  agg_threads={agg_threads}  "
        f"ring={ring or hc.RING}  min_cluster={min_cluster}"
        + (f"  PARTIAL_LENS={partial_lens}" if partial_lens else "")
    )
    print(f"{'='*62}")

    # Inputs, shared between the clear reference and the tapes.
    all_encoded, common_ids, dict_size, partial_len, _, _ = prepare_nparty_data(
        party_files, max_cases, universe=universe
    )
    n_cases = len(common_ids)
    d_sq = dict_size * dict_size

    plens_arg = None
    if partial_lens is not None:
        assert len(partial_lens) == n_parties, (partial_lens, n_parties)
        # Re-truncate each party to its own length; the clear reference runs on the same events.
        all_encoded = retruncate_per_party(all_encoded, partial_lens)
        partial_len = max(partial_lens)
        plens_arg = "x".join(str(x) for x in partial_lens)
        write_nparty_inputs(
            spdz_dir, all_encoded, n_parties, n_cases, partial_lens)
    else:
        write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)
    print(f"  n_cases={n_cases}  dict_size={dict_size}  partial_len={partial_len}"
          + (f"  PARTIAL_LENS={plens_arg}" if plens_arg else ""))

    # The projection reads the clustering-encoding row, whose width the tape asserts against expected_enc_dim.
    enc_dim = dict_size if encoding == 3 else d_sq
    r_matrix = generate_achlioptas_matrix(k_proj, enc_dim, seed)
    proj_path = spdz_dir / "Programs" / "Source" / "projection_matrix.txt"
    write_projection_matrix(proj_path, r_matrix)

    # Clear reference: merge, encode, project, cluster, aggregate.
    histograms = []
    projected = []
    for c in range(n_cases):
        party_events = [all_encoded[p][c] for p in range(n_parties)]
        merged = merge_events_plaintext(party_events, dict_size)
        hist = extract_bigrams_plaintext(merged, dict_size)
        histograms.append(hist)
        prof = (extract_activity_plaintext(merged, dict_size) if encoding == 3
                else hist)
        # The projected Y is integer-valued, so the exact clear reference consumes it without rounding.
        projected.append([int(v) for v in (r_matrix @ np.array(prof, dtype=np.int64))])

    # The ring-guard width of this cell, from the same public inputs the Rust gate reads.
    l_total = sum(partial_lens) if partial_lens else partial_len * n_parties
    y_max_bound = hc.encoding_mass_bound(encoding, l_total)
    y_max_actual = max((abs(v) for row in projected for v in row), default=0)
    assert y_max_bound >= y_max_actual, (
        f"encoding_mass_bound({encoding}, {l_total}) = {y_max_bound} is below "
        f"the fixture's actual max |y| = {y_max_actual}")
    argmin_bits = hc.secure_kmeans_argmin_bits(k_proj, n_cases, y_max_bound)

    # The merge runs once, in the last tape of the chain.
    km_labels = lloyd_labels(projected, k=k, max_iter=max_iter,
                             min_cluster=0 if sequence else min_cluster)
    if min_cluster and not sequence:
        km_unmerged = lloyd_labels(projected, k=k, max_iter=max_iter)
        moved = sum(1 for a, b in zip(km_unmerged, km_labels) if a != b)
        print(f"  [REF] k-means labels before the merge={km_unmerged}; "
              f"the merge moves {moved}/{n_cases} cases")
        if moved == 0:
            print("  FAIL fixture: the twin's merge moved nothing")
            return False
    seq_margins = None
    if sequence:
        # The histogram layout is the one the twin produces and secure_extract persists at offset 0.
        expected_labels, seq_models = sequence_labels(
            histograms, k=k, max_iter=seq_max_iter, dh=dict_size,
            init_labels=km_labels, return_models=True, min_cluster=min_cluster,
        )
        if min_cluster:
            seq_unmerged = sequence_labels(
                histograms, k=k, max_iter=seq_max_iter, dh=dict_size,
                init_labels=km_labels)
            moved = sum(1 for a, b in zip(seq_unmerged, expected_labels) if a != b)
            print(f"  [REF] sequence labels before the merge={seq_unmerged}; "
                  f"the merge moves {moved}/{n_cases} cases")
            if moved == 0:
                print("  FAIL fixture: the twin's merge moved nothing")
                return False
        seq_margins = score_margins(histograms, expected_labels, seq_models, dict_size)
        moved = sum(1 for a, b in zip(km_labels, expected_labels) if a != b)
        print(f"  [REF] k-means seed labels={km_labels}")
        print(f"  [REF] sequence labels={expected_labels}  "
              f"(moved {moved}/{n_cases}; min margin {min(seq_margins):.4f} log2)")
    else:
        expected_labels = km_labels
        print(f"  [REF] labels={expected_labels}")
    exp_dfg, exp_sizes = aggregate_plaintext(histograms, expected_labels, k, dict_size)
    print(f"  [REF] sizes={exp_sizes}")

    # Stage 1: secure_extract, persisting Y.
    clear_persistence(spdz_dir, n_parties)
    enc_dim_arg = enc_dim if encoding != 0 else 0
    extract_args = [n_cases, partial_len, dict_size, n_threads, k_proj,
                    n_parties, encoding, 1]
    if plens_arg is not None:
        extract_args += [enc_dim_arg, plens_arg]
    elif enc_dim_arg:
        extract_args += [enc_dim_arg]
    prog = compile_mpc(spdz_dir, "secure_extract", extract_args,
                       n_parties=n_parties, ring=ring)
    _, t_extract = run_mpc(spdz_dir, prog, n_parties=n_parties,
                           timeout=timeout)
    print(f"  [MPC extract] {t_extract:.2f}s")

    # Stage 2: secure_kmeans, reading Y from persistence.
    y_offset = n_cases * d_sq
    label_offset = y_offset + n_cases * k_proj
    kmeans_args = [n_cases, k_proj, k, max_iter, n_threads, 1, y_offset]
    if narrow_argmin:
        # REVEAL_LABELS has to be spelled out to reach ARGMIN_BITS; the persisted one-hot is unaffected.
        kmeans_args += [1, argmin_bits]
    if min_cluster and not sequence:
        if not narrow_argmin:
            kmeans_args += [1, 0]
        kmeans_args += [min_cluster]
    prog = compile_mpc(spdz_dir, "secure_kmeans", kmeans_args,
                       n_parties=n_parties, ring=ring)
    km_stdout, t_kmeans = run_mpc(spdz_dir, prog, n_parties=n_parties,
                                  timeout=timeout)
    print(f"  [MPC kmeans]  {t_kmeans:.2f}s"
          + (f"  ARGMIN_BITS={argmin_bits} (y_max bound {y_max_bound})"
             if narrow_argmin else ""))
    km_narrow_ok = True
    if min_cluster and not sequence:
        if f"Merging clusters below {min_cluster} cases" not in km_stdout:
            print("  FAIL merge: secure_kmeans never announced the merge pass")
            km_narrow_ok = False
        elif parse_labels(km_stdout, n_cases) != km_labels:
            diffs = [(i, e, a) for i, (e, a) in enumerate(
                zip(km_labels, parse_labels(km_stdout, n_cases))) if e != a]
            print(f"  FAIL merge labels: {len(diffs)} mismatch(es) against "
                  f"the merged twin: {diffs[:10]}")
            km_narrow_ok = False
        else:
            print(f"  PASS merge labels: the tape's merged labels match the "
                  f"twin exactly (MIN_CLUSTER={min_cluster})")
    if narrow_argmin:
        if f"SCORE_CMP_BITS={argmin_bits} " not in km_stdout:
            print(f"  FAIL argmin width: the tape did not compare at "
                  f"SCORE_CMP_BITS={argmin_bits}")
            km_narrow_ok = False
        elif argmin_bits >= (ring or hc.RING):
            print(f"  FAIL argmin width: {argmin_bits} does not narrow the "
                  f"{ring or hc.RING}-bit ring, so the cell proves nothing")
            km_narrow_ok = False
        else:
            print(f"  PASS argmin width: E-step compares at {argmin_bits} "
                  f"bits, below the {ring or hc.RING}-bit ring")
        mpc_km_labels = parse_labels(km_stdout, n_cases)
        if mpc_km_labels == km_labels:
            print("  PASS argmin labels: the narrowed tape matches "
                  "plaintext_kmeans_reference exactly")
        else:
            diffs = [(i, e, a) for i, (e, a) in
                     enumerate(zip(km_labels, mpc_km_labels)) if e != a]
            print(f"  FAIL argmin labels: {len(diffs)} mismatch(es) at "
                  f"ARGMIN_BITS={argmin_bits}: {diffs[:10]}")
            km_narrow_ok = False

    # Stage 2b, sequence only: the refinement reads the slab at offset 0 and the one-hot at LABEL_OFFSET, which it
    # rewrites at the same offset, so stage 3 is the same call either way.
    t_seq = 0.0
    mpc_seq_labels = None
    seq_read_seed = False
    if sequence:
        seq_args = [n_cases, d_sq, k, seq_max_iter, n_threads, 1,
                    label_offset, 1, 1]
        if min_cluster:
            seq_args += [0, min_cluster]    # PROFILE, MIN_CLUSTER
        prog = compile_mpc(spdz_dir, "secure_sequence_cluster", seq_args,
                           n_parties=n_parties, ring=ring)
        seq_stdout, t_seq = run_mpc(spdz_dir, prog, n_parties=n_parties,
                                    timeout=timeout)
        print(f"  [MPC seq]     {t_seq:.2f}s")
        if min_cluster and f"Merging clusters below {min_cluster} cases" not in seq_stdout:
            print("  FAIL merge: the refinement never announced the merge pass")
            km_narrow_ok = False
        mpc_seq_labels = parse_labels(seq_stdout, n_cases)
        seq_read_seed = (
            f"initial one-hot labels from persistence offset {label_offset}"
            in seq_stdout
        )
        print(f"  [MPC seq] labels={mpc_seq_labels}")

    # Assertions on the clustering stage; the aggregate reference is rebuilt if a tolerated flip moved a case.
    edge_min_count, edge_frac_num = 2, 200
    exp_dfg_filtered = filter_dfg_plaintext(
        exp_dfg, dict_size, edge_min_count, edge_frac_num)
    ok = km_narrow_ok

    if sequence:
        if seq_read_seed:
            print(f"  PASS seed: sequence stage read the k-means one-hot at "
                  f"offset {label_offset}")
        else:
            print(f"  FAIL seed: no 'initial one-hot labels from persistence "
                  f"offset {label_offset}' line — INIT_MODE=1 did not take effect")
            ok = False

        diffs = [
            (i, e, a, seq_margins[i])
            for i, (e, a) in enumerate(zip(expected_labels, mpc_seq_labels))
            if e != a
        ]
        if not diffs:
            print("  PASS sequence labels: exact match with the plaintext twin "
                  f"(min margin {min(seq_margins):.4f} log2)")
        else:
            hard = [t for t in diffs if t[3] >= SEQ_MARGIN_TOL]
            print(f"  {len(diffs)} label mismatch(es) (idx, twin, mpc, margin): "
                  f"{diffs[:10]}")
            if hard:
                print(f"  FAIL sequence labels: {len(hard)} above the "
                      f"{SEQ_MARGIN_TOL} margin tolerance")
                ok = False
            else:
                print("  PASS sequence labels (tolerated): every mismatch sits "
                      "inside the fixed-point tie margin")
                # Rebuild the DFG and size reference on the tape's labels, which a tolerated flip changed.
                exp_dfg, exp_sizes = aggregate_plaintext(
                    histograms, mpc_seq_labels, k, dict_size)
                exp_dfg_filtered = filter_dfg_plaintext(
                    exp_dfg, dict_size, edge_min_count, edge_frac_num)

    n_zeroed = sum(
        1 for j in range(k) for d in range(d_sq)
        if exp_dfg[j][d] != exp_dfg_filtered[j][d]
    )
    print(f"  [REF] filter zeroes {n_zeroed} sub-cutoff activity cells")

    # Stage 3: secure_aggregate at the pipeline's edge filter and PRUNE=1, re-run per agg_threads entry against the
    # same persistence, since a reordered per-thread accumulator would publish a different DFG from the same labels.
    exp_dfg_filtered = [prune_dense(row, dict_size)[0] for row in exp_dfg_filtered]
    exp_sizes_opened = derive_sizes_from_dfg(exp_dfg_filtered, dict_size)
    for nt in agg_threads:
        agg_args = [n_cases, n_cases, d_sq, k, 1, label_offset,
                    nt, edge_min_count, edge_frac_num, 1]
        prog = compile_mpc(spdz_dir, "secure_aggregate", agg_args,
                           n_parties=n_parties, ring=ring)
        stdout, t_agg = run_mpc(spdz_dir, prog, n_parties=n_parties,
                                timeout=timeout)
        print(f"  [MPC agg]     {t_agg:.2f}s  (N_THREADS={nt})")
        if "MODE: PRODUCTION" not in stdout:
            print("  FAIL banner: secure_aggregate did not print the documented "
                  "'MODE: PRODUCTION' literal at SECURE_LABELS=1")
            ok = False

        # No size is opened: the START rows of a pruned graph count its cases with a kept start activity, and the
        # opened START_TOTAL, over all clusters before the pruning, must be N.
        mpc_dfg = parse_row_output(stdout, k, d_sq)
        mpc_sizes = derive_sizes_from_dfg(mpc_dfg, dict_size)
        start_total = next((int(l.split()[1]) for l in stdout.splitlines()
                            if l.startswith("START_TOTAL ")), None)

        if mpc_sizes == exp_sizes_opened and start_total == sum(exp_sizes) == n_cases:
            print(f"  PASS START rows ({nt} threads): {mpc_sizes}, START_TOTAL {start_total}")
        else:
            print(f"  FAIL START rows ({nt} threads): expected {exp_sizes_opened} and "
                  f"START_TOTAL {n_cases}, got {mpc_sizes} and {start_total}")
            ok = False

        dfg_mismatches = [
            (j, d, exp_dfg_filtered[j][d], mpc_dfg[j][d])
            for j in range(k) for d in range(d_sq)
            if exp_dfg_filtered[j][d] != mpc_dfg[j][d]
        ]
        if not dfg_mismatches:
            print(f"  PASS DFG ({nt} threads): all {k*d_sq} cells match the "
                  "filtered and pruned reference")
        else:
            print(f"  FAIL DFG ({nt} threads): {len(dfg_mismatches)} mismatches, "
                  f"first: {dfg_mismatches[:5]}")
            ok = False

    print(f"  Total MPC: {t_extract + t_kmeans + t_seq:.2f}s + aggregate")
    return ok


def require_memory_scopes() -> None:
    """Pin by source grep that both per-thread accumulators still carry their memory scope."""
    pins = [("secure_aggregate.mpc", 'open_memory_scope("agg_acc")'),
            ("secure_sequence_cluster.mpc", 'open_memory_scope("seq_acc")')]
    for name, literal in pins:
        src = (hc.HERE / name).read_text()
        if literal not in src:
            print(f"ERROR: {name} no longer contains {literal}; the per-thread "
                  f"accumulator scope is gone and updates are lost at more "
                  f"than one thread.", file=sys.stderr)
            sys.exit(1)
    print("  OK memory-scope pin (source grep): agg_acc + seq_acc present")


def main():
    parser = hc.common_parser("Full 3-stage MPC integration test")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--k-proj", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    spdz_dir = hc.setup(args)

    data_dir = PROJECT_ROOT / "data" / "splits" / "sepsis_2party"
    party_files = [data_dir / "Party_A.xes", data_dir / "Party_B.xes"]
    missing = [f for f in party_files if not f.exists()]
    if missing:
        print(f"ERROR: missing party files: {missing}")
        sys.exit(1)
    sepsis_4party = PROJECT_ROOT / "data" / "splits" / "sepsis_4party"
    party_files_4 = [
        sepsis_4party / "party_a_er.xes",
        sepsis_4party / "party_b_lab.xes",
        sepsis_4party / "party_c_emergency.xes",
        sepsis_4party / "party_d_disposition.xes",
    ]
    union_dir = PROJECT_ROOT / "data" / "splits" / "sepsis_2party_erlab_wards"
    union_files = [union_dir / "Party_ER_Lab.xes", union_dir / "Party_Wards.xes"]

    base = dict(max_cases=args.max_cases, k_proj=args.k_proj, k=args.k,
                max_iter=args.max_iter, seed=args.seed)
    results = {}

    def cell(name, *a, **kw):
        if not hc.cell_enabled(name):
            return
        clear_persistence(spdz_dir)
        results[name] = run_test(spdz_dir, *a, **kw)

    require_memory_scopes()

    cell("kmeans", party_files, narrow_argmin=True, **base)

    # The oblivious small-cluster merge in the last tape of each chain, every stage threaded, so the merge's own
    # threaded remap loop is exercised where the accumulator's memory-order trap fired.
    merge_base = dict(base, k=3, n_threads=8)
    cell("merge", party_files, narrow_argmin=True, min_cluster=3, **merge_base)
    cell("merge_sequence", party_files, secure_clustering="sequence",
         min_cluster=3, **merge_base)

    # Non-uniform per-party lengths, against the clear reference built on the same re-truncated events.
    cell("partial_lens", party_files, partial_lens=[8, 4], **base)

    # The union universe on the split whose case sets differ, whose first ids put single-party cases at middle slots.
    if all(f.exists() for f in union_files):
        cell("union", union_files, universe="union", **base)
    else:
        print(f"\nSKIP union universe test: missing {union_files}")

    # Sequence clustering from the k-means partition; k is pinned to 3, since at k=2 that partition is already a
    # fixed point of the Markov EM and the assertion could not tell the two apart.
    seq_base = dict(base, k=3)
    cell("sequence", party_files, secure_clustering="sequence", **seq_base)

    # The activity encoding: the clustering rides the d-wide profile while the published DFG slab stays the bigrams.
    cell("activity", party_files, encoding=3, **base)

    # At -R 128, where every ring-width argument of the helpers is stated for 64 bits.
    cell("ring128", party_files, ring=128, **base)

    # The multi-thread accumulator regression: four parties, the union universe, every stage threaded, and the
    # Round-2 accumulator re-run at eight threads and at one on the same labels.
    if all(f.exists() for f in party_files_4):
        f4 = dict(base, max_cases=120, k=4, k_proj=20)
        cell("f4", party_files_4, partial_lens=[4, 28, 4, 8], universe="union",
             n_threads=8, agg_threads=(8, 1), **f4)
        # The sequence M-step carries the same scope; two iterations give it one threaded pass.
        cell("f4_sequence", party_files_4, partial_lens=[4, 28, 4, 8],
             universe="union", n_threads=8, secure_clustering="sequence",
             seq_max_iter=2, **f4)
    else:
        # A hard exit, not a skip: these cells are the only executed coverage of the accumulator fix.
        missing_4 = [str(f) for f in party_files_4 if not f.exists()]
        if not (hc.cell_enabled("f4") or hc.cell_enabled("f4_sequence")):
            print(f"\nSKIP F4 cells (--only excludes them): missing {missing_4}")
            print()
            for name, ok in results.items():
                print(f"  {name:<14}: {'PASS' if ok else 'FAIL'}")
            return all(results.values()) and bool(results)
        print(
            "ERROR: the 4-party Sepsis split is missing, so the f4 / "
            "f4_sequence cells — the only executed regression for the "
            "multi-thread accumulator fault — cannot run. This harness "
            "refuses to report PASS without them.\n"
            f"  missing: {missing_4}\n"
            "  regenerate with scripts/split_log.py:\n"
            "    .venv/bin/python scripts/split_log.py \\\n"
            "        --xes 'data/raw/Sepsis Cases.xes' \\\n"
            "        --partition-file config/sepsis_4party.json \\\n"
            "        --output-dir data/splits/sepsis_4party",
            file=sys.stderr)
        sys.exit(1)

    print()
    for name, ok in results.items():
        print(f"  {name:<14}: {'PASS' if ok else 'FAIL'}")
    return all(results.values()) and bool(results)


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_pipeline", main))
