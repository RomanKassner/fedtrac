#!/usr/bin/env python3
"""Agreement test for secure_sequence_cluster.mpc against scripts/plaintext_sequence_reference.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_common as hc
from harness_common import clear_persistence, compile_mpc, run_mpc
from test_secure_kmeans import parse_labels, write_party0_input

from scripts.plaintext_sequence_reference import (
    bigram_rows, score_margins, sequence_labels,
)

MARGIN_TOL = 0.05
INIT_SEED = 2


def make_fixture():
    """24 cases of three process types over four activities, shuffled so the stride seeds cut across the types."""
    a, b, c, d = 3, 4, 5, 6
    rng = random.Random(7)
    traces = []
    for _ in range(8):
        t = [a, b, c, d]
        if rng.random() < 0.5:
            t.insert(2, b)
        traces.append(t)
    for _ in range(8):
        t = [a, c, c, d]
        if rng.random() < 0.5:
            t.append(c)
        traces.append(t)
    for _ in range(8):
        t = [b, a, d]
        if rng.random() < 0.5:
            t = [b, a, a, d]
        traces.append(t)
    rng.shuffle(traces)
    dh = 7
    return bigram_rows(traces, dh), dh, 3, 5


def balanced_init(n: int, k: int, seed: int = INIT_SEED) -> list[int]:
    """A balanced but type-blind initial partition."""
    labels = [j for j in range(k) for _ in range(n // k)]
    labels += [j % k for j in range(n - len(labels))]
    random.Random(seed).shuffle(labels)
    return labels


def one_hot(labels: list[int], k: int) -> list[list[int]]:
    return [[1 if j == lab else 0 for j in range(k)] for lab in labels]


def run_case(spdz_dir: Path, X, dh, k, max_iter, init_labels=None,
             min_cluster: int = 0) -> bool:
    """One agreement run: init_labels=None uses the stride seeds, otherwise the supplied partition."""
    n, D = len(X), len(X[0])
    init_mode = 0 if init_labels is None else 1
    print(f"\n{'='*60}")
    print(f"TEST: secure_sequence_cluster agreement (n={n}, D={D}, k={k}, "
          f"iter={max_iter}, INIT_MODE={init_mode}, MIN_CLUSTER={min_cluster})")
    print(f"{'='*60}")

    expected, C = sequence_labels(X, k=k, max_iter=max_iter, dh=dh,
                                  return_models=True, init_labels=init_labels,
                                  min_cluster=min_cluster)
    if min_cluster > 0:
        unmerged = sequence_labels(X, k=k, max_iter=max_iter, dh=dh,
                                   init_labels=init_labels)
        moved = sum(1 for a, b in zip(unmerged, expected) if a != b)
        sizes = [unmerged.count(j) for j in range(k)]
        print(f"  Twin before the merge: {unmerged}  sizes={sizes}")
        if moved == 0:
            print("  FAIL fixture: the twin's merge moved nothing")
            return False
        print(f"  Twin merge moves {moved}/{n} cases")
    margins = score_margins(X, expected, C, dh)
    if init_labels is not None:
        moved = sum(1 for a, b in zip(init_labels, expected) if a != b)
        print(f"  Initial partition: {init_labels}")
        print(f"  Twin moves {moved}/{n} cases off their initial model")
    print(f"  Plaintext labels: {expected}")
    print(f"  Min best-vs-runner-up margin: {min(margins):.4f} log2 units")

    rows = list(X)
    args = [n, D, k, max_iter, hc.THREADS, 0, 0]
    if init_labels is not None:
        # Test mode reads the one-hot rows from party 0 after the count rows.
        rows = rows + one_hot(init_labels, k)
        args = args + [0, 1]        # REVEAL_LABELS, INIT_MODE
    if min_cluster > 0:
        # REVEAL_LABELS, INIT_MODE, PROFILE, MIN_CLUSTER.
        args = args + [0, 0] if init_labels is None else args
        args = args + [0, min_cluster]

    write_party0_input(spdz_dir, rows)
    prog_name = compile_mpc(spdz_dir, "secure_sequence_cluster", args,
                            n_parties=2)
    stdout, wall = run_mpc(spdz_dir, prog_name, n_parties=2)
    actual = parse_labels(stdout, n)
    print(f"  MPC labels:       {actual}")
    print(f"  Wall time: {wall:.2f}s")

    if min_cluster > 0 and f"Merging clusters below {min_cluster} cases" not in stdout:
        print("  FAIL: the tape never announced the merge pass")
        return False
    if init_labels is not None and "initial one-hot labels from party 0" not in stdout:
        print("  FAIL: the box never announced reading the supplied partition "
              "(INIT_MODE=1 did not take effect).")
        return False

    diffs = [(i, e, a, margins[i]) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if not diffs:
        print("  PASS: MPC labels exactly match the plaintext twin.")
        return True
    hard = [t for t in diffs if t[3] >= MARGIN_TOL]
    print(f"  {len(diffs)} mismatch(es) (idx, twin, mpc, margin): {diffs[:10]}")
    if hard:
        print(f"  FAIL: {len(hard)} mismatch(es) above the {MARGIN_TOL} margin tolerance.")
        return False
    print("  PASS (tolerated): every mismatch sits inside the fixed-point tie margin.")
    return True


def run_pipeline_stride_case(spdz_dir: Path, X, dh, k, max_iter) -> bool:
    """The pipeline stride path: count rows from persistence, stride seeds, the one-hot written at LABEL_OFFSET."""
    n, D = len(X), len(X[0])
    print(f"\n{'='*60}")
    print(f"TEST: secure_sequence_cluster PIPELINE stride (n={n}, D={D}, k={k}, "
          f"INPUT_SOURCE=1, INIT_MODE=0)")
    print(f"{'='*60}")

    expected, C = sequence_labels(X, k=k, max_iter=max_iter, dh=dh,
                                  return_models=True)
    margins = score_margins(X, expected, C, dh)
    print(f"  Plaintext labels: {expected}")

    clear_persistence(spdz_dir)
    flat = [v for row in X for v in row]
    hc.write_party0_values(spdz_dir, flat)
    seed_prog = compile_mpc(spdz_dir, "seed_persistence", [len(flat), 0],
                            n_parties=2)
    seed_out, _ = run_mpc(spdz_dir, seed_prog, n_parties=2, timeout=300)
    if "MODE: TESTING" not in seed_out:
        print("  FAIL banner: seed_persistence did not print 'MODE: TESTING'")
        return False

    label_offset = n * D
    args = [n, D, k, max_iter, hc.THREADS, 1, label_offset, 1, 0]
    prog_name = compile_mpc(spdz_dir, "secure_sequence_cluster", args,
                            n_parties=2)
    stdout, wall = run_mpc(spdz_dir, prog_name, n_parties=2, timeout=900)
    actual = parse_labels(stdout, n)
    print(f"  MPC labels:       {actual}")
    print(f"  Wall time: {wall:.2f}s")

    ok = True
    if "MODE: PRODUCTION" not in stdout:
        print("  FAIL: the box did not announce a PRODUCTION mode "
              "(INPUT_SOURCE=1 did not take effect)")
        ok = False
    if f"Persisting labels at offset {label_offset}" not in stdout:
        print(f"  FAIL: no 'Persisting labels at offset {label_offset}' line — "
              "the one-hot was not written back for Round 2")
        ok = False
    if "initial one-hot labels" in stdout:
        print("  FAIL: the box read a supplied partition; INIT_MODE=0 must "
              "use the stride seeds")
        ok = False

    diffs = [(i, e, a, margins[i])
             for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if not diffs:
        print("  PASS: MPC labels exactly match the plaintext twin.")
    else:
        hard = [t for t in diffs if t[3] >= MARGIN_TOL]
        print(f"  {len(diffs)} mismatch(es) (idx, twin, mpc, margin): {diffs[:10]}")
        if hard:
            print(f"  FAIL: {len(hard)} mismatch(es) above the {MARGIN_TOL} "
                  "margin tolerance.")
            ok = False
        else:
            print("  PASS (tolerated): every mismatch sits inside the "
                  "fixed-point tie margin.")
    return ok


def run_test(spdz_dir: Path) -> bool:
    X, dh, k, max_iter = make_fixture()
    results = {}
    if hc.cell_enabled("stride"):
        results["INIT_MODE=0 (stride seeds)"] = run_case(
            spdz_dir, X, dh, k, max_iter)
    if hc.cell_enabled("partition"):
        results["INIT_MODE=1 (supplied partition)"] = run_case(
            spdz_dir, X, dh, k, max_iter, init_labels=balanced_init(len(X), k))
    # The oblivious small-cluster merge.
    if hc.cell_enabled("merge"):
        # The twin ends with one cluster of three cases here, which MIN_CLUSTER=4 moves whole.
        results["MIN_CLUSTER=4 (oblivious merge, k=4)"] = run_case(
            spdz_dir, X, dh, 4, max_iter,
            init_labels=balanced_init(len(X), 4, seed=4), min_cluster=4)
    if hc.cell_enabled("pipeline_stride"):
        results["INPUT_SOURCE=1 + INIT_MODE=0 (pipeline stride)"] = (
            run_pipeline_stride_case(spdz_dir, X, dh, k, max_iter))
    print()
    for name, ok in results.items():
        print(f"  {name:<48}: {'PASS' if ok else 'FAIL'}")
    return all(results.values()) and bool(results)


def main():
    parser = hc.common_parser("Agreement test for secure_sequence_cluster.mpc")
    args = parser.parse_args()
    spdz_dir = hc.setup(args)
    ok = run_test(spdz_dir)
    print("ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_sequence", main))
