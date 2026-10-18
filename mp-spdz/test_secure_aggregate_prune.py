#!/usr/bin/env python3
"""secure_aggregate.mpc Step 3c, the dead-end pruning, against its clear twin bridge/utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_common as hc  # noqa: E402
from harness_common import (  # noqa: E402
    END_ACT, FIRST_USER_ACT, START_ACT, clear_persistence, compile_mpc, prune_dense, run_mpc,
)
from test_secure_extract import parse_row_output  # noqa: E402
from test_secure_pipeline import filter_dfg_plaintext  # noqa: E402


def padded_bigram_row(trace, d):
    row = [0] * (d * d)
    seq = [START_ACT] + list(trace) + [END_ACT]
    for a, b in zip(seq, seq[1:]):
        row[a * d + b] += 1
    return row


def random_traces(rng, n_cases, n_acts):
    """A common backbone plus rare activities, so the filter at count two leaves dead ends."""
    acts = list(range(FIRST_USER_ACT, FIRST_USER_ACT + n_acts))
    backbone = acts[:3]
    rare = acts[3:]
    traces = []
    for _ in range(n_cases):
        t = list(backbone)
        r = rng.random()
        if r < 0.15:
            t = [int(rng.choice(rare))] + t            # rare start, filtered exit
        elif r < 0.30:
            t = t + [int(rng.choice(rare))]            # rare end, filtered entry
        elif r < 0.45:
            t.insert(2, int(rng.choice(rare)))
        if rng.random() < 0.3:
            t.insert(1, t[1])                          # a repeat (self-loop)
        traces.append(t)
    return traces


def run_cell(spdz_dir, name, n_cases, n_acts, k, seed, edge_min_count, edge_frac_num,
             expect_pruning, n_parties=2):
    rng = np.random.default_rng(seed)
    d = FIRST_USER_ACT + n_acts
    traces = random_traces(rng, n_cases, n_acts)
    labels = [int(x) for x in rng.integers(0, k, size=n_cases)]
    rows = [padded_bigram_row(t, d) for t in traces]

    exp = [[0] * (d * d) for _ in range(k)]
    for row, lab in zip(rows, labels):
        for f, w in enumerate(row):
            exp[lab][f] += w
    exp = filter_dfg_plaintext(exp, d, edge_min_count, edge_frac_num)
    pruned = [prune_dense(row, d) for row in exp]
    exp = [p[0] for p in pruned]
    n_removed = sum(len(p[1]) for p in pruned)

    clear_persistence(spdz_dir)
    hc.write_party0_values(spdz_dir, [v for row in rows for v in row])
    prog = compile_mpc(spdz_dir, "seed_persistence", [n_cases * d * d, 0], n_parties=n_parties)
    run_mpc(spdz_dir, prog, n_parties, timeout=300)

    hc.write_party0_values(spdz_dir, labels)
    args = [n_cases, n_cases, d * d, k, 0, 0, 1, edge_min_count, edge_frac_num, 1]
    prog = compile_mpc(spdz_dir, "secure_aggregate", args, n_parties=n_parties)
    stdout, t = run_mpc(spdz_dir, prog, n_parties, timeout=600)

    ok = True
    total = next((int(l.split()[1]) for l in stdout.splitlines() if l.startswith("START_TOTAL ")), None)
    if total != n_cases:
        print(f"  FAIL {name}: START_TOTAL {total}, expected {n_cases}")
        ok = False
    got = parse_row_output(stdout, k, d * d)
    bad = [(c, f, exp[c][f], got[c][f]) for c in range(k) for f in range(d * d) if exp[c][f] != got[c][f]]
    if bad:
        print(f"  FAIL {name}: {len(bad)} cells differ from the clear filter+pruning, first {bad[:5]}")
        ok = False
    if expect_pruning and n_removed == 0:
        print(f"  FAIL {name}: the reference pruned nothing, so the cell tests nothing")
        ok = False
    if not expect_pruning and n_removed:
        print(f"  FAIL {name}: the reference pruned {n_removed} activities of an unfiltered graph")
        ok = False
    if ok:
        print(f"  PASS {name}: {k * d * d} cells equal the clear twin, {n_removed} activities pruned, "
              f"START_TOTAL {total}, {t:.1f}s")
    return ok


def main():
    parser = hc.common_parser("secure_aggregate Step 3c against the clear pruning")
    args = parser.parse_args()
    spdz_dir = hc.setup(args)
    results = {}
    cells = [
        ("prune_filtered_s1", dict(n_cases=40, n_acts=7, k=3, seed=1, edge_min_count=2,
                                   edge_frac_num=200, expect_pruning=True)),
        ("prune_filtered_s2", dict(n_cases=60, n_acts=9, k=4, seed=2, edge_min_count=2,
                                   edge_frac_num=200, expect_pruning=True)),
        ("prune_filtered_3p", dict(n_cases=40, n_acts=7, k=2, seed=3, edge_min_count=2,
                                   edge_frac_num=200, expect_pruning=True, n_parties=3)),
        ("prune_unfiltered", dict(n_cases=40, n_acts=7, k=3, seed=4, edge_min_count=0,
                                  edge_frac_num=0, expect_pruning=False)),
    ]
    for name, kw in cells:
        if hc.cell_enabled(name):
            results[name] = run_cell(spdz_dir, name, **kw)
    hc.cleanup_all(spdz_dir)
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} cells pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
