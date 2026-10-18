#!/usr/bin/env python3
"""Agreement test for secure_kmeans.mpc: the revealed labels must match scripts/plaintext_kmeans_reference.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_common as hc
from harness_common import compile_mpc, run_mpc

from scripts.plaintext_kmeans_reference import lloyd_labels

# MIN_CLUSTER below which a cluster moves whole; the production value is constants.rs::MIN_CLUSTER_SIZE.
MERGE_MIN_CLUSTER = 4

def make_fixture_three_clusters():
    """Sixteen integer points in four dimensions forming three well-separated clusters."""
    Y = [
        # Cluster A: around origin
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        # Cluster B: around (20, 20, 0, 0)
        [20, 20, 0, 0],
        [21, 19, 1, 0],
        [19, 21, 0, 1],
        [20, 20, 1, 1],
        [22, 20, 0, 0],
        # Cluster C: around (0, 0, 30, 30)
        [0, 0, 30, 30],
        [1, 0, 29, 31],
        [0, 1, 31, 29],
        [0, 0, 30, 31],
        [1, 1, 30, 30],
        [0, 0, 31, 30],
    ]
    k = 3
    max_iter = 5
    return Y, k, max_iter


def make_fixture_small_cluster():
    """The three-cluster fixture plus two outliers at k=4, so the stride seeds leave a cluster below MIN_CLUSTER."""
    Y, _, max_iter = make_fixture_three_clusters()
    Y = [list(r) for r in Y] + [[60, 60, 60, 60], [61, 60, 60, 60]]
    return Y, 4, max_iter


def write_party0_input(spdz_dir: Path, Y: list[list[int]]) -> None:
    """Write the Y matrix as party 0's input."""
    hc.write_party0_rows(spdz_dir, Y)


def parse_labels(stdout: str, n: int) -> list[int]:
    """Parse the LABEL lines of the tape's output."""
    labels = [-1] * n
    for line in stdout.splitlines():
        if line.startswith("LABEL "):
            parts = line.split()
            idx = int(parts[1])
            val = int(parts[2])
            if 0 <= idx < n:
                labels[idx] = val
    missing = [i for i, v in enumerate(labels) if v < 0]
    if missing:
        raise RuntimeError(f"Missing labels for indices {missing}")
    return labels


def run_test(spdz_dir: Path, argmin_bits: int | None = None):
    """argmin_bits=None compares at the ring width; an int passes the narrowed E-step width of a production run."""
    Y, k, max_iter = make_fixture_three_clusters()
    n = len(Y)
    k_proj = len(Y[0])

    print(f"\n{'='*60}")
    print(f"TEST: secure_kmeans agreement (n={n}, k_proj={k_proj}, k={k}, "
          f"iter={max_iter}, ARGMIN_BITS="
          f"{'default (ring width)' if argmin_bits is None else argmin_bits})")
    print(f"{'='*60}")

    expected = lloyd_labels(Y, k=k, max_iter=max_iter)
    print(f"  Plaintext labels: {expected}")

    # Test mode: INPUT_SOURCE=0, Y_OFFSET=0.
    write_party0_input(spdz_dir, Y)
    args = [n, k_proj, k, max_iter, hc.THREADS, 0, 0]
    if argmin_bits is not None:
        # ARGMIN_BITS is the ninth argument, so REVEAL_LABELS has to be spelled out.
        args += [0, int(argmin_bits)]
    prog_name = compile_mpc(spdz_dir, "secure_kmeans", args, n_parties=2)
    stdout, wall = run_mpc(spdz_dir, prog_name, n_parties=2)
    if argmin_bits is not None:
        banner = f"ARGMIN_BITS={argmin_bits} MIN_CLUSTER="
        if banner not in stdout:
            print(f"  FAIL: the tape did not echo '{banner}'")
            return False
        if f"SCORE_CMP_BITS={argmin_bits} " not in stdout:
            print(f"  FAIL: the tape did not compare at SCORE_CMP_BITS="
                  f"{argmin_bits}")
            return False
        print(f"  OK tape compares at SCORE_CMP_BITS={argmin_bits} "
              f"(< the {hc.RING}-bit ring)")

    actual = parse_labels(stdout, n)
    print(f"  MPC labels:       {actual}")
    print(f"  Wall time: {wall:.2f}s")

    if actual == expected:
        print("  PASS: MPC labels exactly match plaintext reference.")
    else:
        diffs = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        print(f"  FAIL: {len(diffs)} mismatch(es): {diffs[:10]}")
        return False

    return True


def run_merge_test(spdz_dir: Path) -> bool:
    """The oblivious small-cluster merge must reproduce the twin's lloyd_labels(..., min_cluster) exactly."""
    Y, k, max_iter = make_fixture_small_cluster()
    n, k_proj = len(Y), len(Y[0])
    y_max = max(abs(v) for row in Y for v in row)
    bits = hc.secure_kmeans_argmin_bits(k_proj, n, y_max)
    print(f"\n{'='*60}")
    print(f"TEST: secure_kmeans oblivious small-cluster merge (n={n}, k={k}, "
          f"MIN_CLUSTER={MERGE_MIN_CLUSTER}, ARGMIN_BITS={bits})")
    print(f"{'='*60}")
    unmerged = lloyd_labels(Y, k=k, max_iter=max_iter)
    expected = lloyd_labels(Y, k=k, max_iter=max_iter,
                            min_cluster=MERGE_MIN_CLUSTER)
    sizes_before = [unmerged.count(j) for j in range(k)]
    sizes_after = [expected.count(j) for j in range(k)]
    print(f"  Twin labels before the merge: {unmerged}  sizes={sizes_before}")
    print(f"  Twin labels after the merge:  {expected}  sizes={sizes_after}")
    moved = sum(1 for a, b in zip(unmerged, expected) if a != b)
    if moved == 0 or not any(0 < s < MERGE_MIN_CLUSTER for s in sizes_before):
        print("  FAIL fixture: the twin's merge moved nothing, so the cell "
              "would prove nothing")
        return False
    if any(0 < s < MERGE_MIN_CLUSTER for s in sizes_after):
        print("  FAIL twin: a cluster below MIN_CLUSTER survives the merge")
        return False

    write_party0_input(spdz_dir, Y)
    args = [n, k_proj, k, max_iter, hc.THREADS, 0, 0, 0, int(bits),
            MERGE_MIN_CLUSTER]
    prog_name = compile_mpc(spdz_dir, "secure_kmeans", args, n_parties=2)
    stdout, wall = run_mpc(spdz_dir, prog_name, n_parties=2)
    if f"MIN_CLUSTER={MERGE_MIN_CLUSTER}" not in stdout:
        print("  FAIL: the tape did not echo MIN_CLUSTER")
        return False
    if f"Merging clusters below {MERGE_MIN_CLUSTER} cases" not in stdout:
        print("  FAIL: the tape never announced the merge pass")
        return False
    actual = parse_labels(stdout, n)
    print(f"  MPC labels:                   {actual}")
    print(f"  Wall time: {wall:.2f}s")
    if actual != expected:
        diffs = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        print(f"  FAIL: {len(diffs)} mismatch(es) after the merge: {diffs[:10]}")
        return False
    print(f"  PASS: the merged labels match the twin exactly ({moved} cases moved).")
    return True


def main():
    parser = hc.common_parser("Agreement test for secure_kmeans.mpc")
    args = parser.parse_args()
    spdz_dir = hc.setup(args)

    Y, k, _ = make_fixture_three_clusters()
    # The fixture is the projected Y, so its largest cell is the y_max the guard bounds.
    y_max = max(abs(v) for row in Y for v in row)
    bits = hc.secure_kmeans_argmin_bits(len(Y[0]), len(Y), y_max)
    print(f"\n  y_max={y_max} -> ARGMIN_BITS={bits} "
          f"(harness_common.secure_kmeans_argmin_bits twin)")
    assert 0 < bits < 64, f"twin returned the refusal sentinel: {bits}"

    results = {}
    # ARGMIN_BITS absent: comparison at the ring width.
    if hc.cell_enabled("default"):
        results["default"] = run_test(spdz_dir)
    # The narrowed width every production run compiles.
    if hc.cell_enabled("argmin_bits"):
        results["argmin_bits"] = run_test(spdz_dir, argmin_bits=bits)
    # The oblivious small-cluster merge.
    if hc.cell_enabled("merge"):
        results["merge"] = run_merge_test(spdz_dir)

    print()
    for name, ok in results.items():
        print(f"  {name:<12}: {'PASS' if ok else 'FAIL'}")
    return all(results.values()) and bool(results)


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_kmeans", main))
