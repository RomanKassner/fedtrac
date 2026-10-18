#!/usr/bin/env python3
"""The `sequence` arm of bridge.cluster, run as the Rust core runs it, must reproduce the clear twin
scripts/plaintext_sequence_reference.sequence_labels exactly, or the tier gap moves silently."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.cluster import compact_labels  # noqa: E402
from scripts.plaintext_sequence_reference import (  # noqa: E402
    bigram_rows,
    encode_traces,
    sequence_labels,
)

# Two families over a shared start hub, the separation the Markov models see.
FAMILY_A = [["s", "a1", "a2"], ["s", "a1", "a2", "a1"], ["s", "a1", "a1", "a2"]]
FAMILY_B = [["s", "b1", "b2"], ["s", "b1", "b2", "b1"], ["s", "b1", "b1", "b2"]]
LOG = FAMILY_A + FAMILY_B
K = 2
MAX_ITER = 5


def rows_and_dh():
    encoded, dh, _ = encode_traces(LOG)
    return bigram_rows(encoded, dh), dh


def run_cli(rows, dh, init=None):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "rows.csv"
        # The float formatting the Rust side writes.
        csv.write_text("\n".join(",".join(f"{v:.6f}" for v in r) for r in rows) + "\n")
        cmd = [sys.executable, "-m", "bridge.cluster", "--features", str(csv),
               "--algo", "sequence", "--k", str(K), "--dict-size", str(dh),
               "--max-iter", str(MAX_ITER)]
        if init is not None:
            labels = tmp / "init.txt"
            labels.write_text("\n".join(str(int(v)) for v in init) + "\n")
            cmd += ["--init-labels", str(labels)]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class SequenceArmTests(unittest.TestCase):
    def test_seeded_arm_matches_reference(self):
        rows, dh = rows_and_dh()
        init = [0, 0, 0, 1, 1, 1]
        expected, _ = compact_labels(
            np.asarray(sequence_labels(rows, K, MAX_ITER, dh, init_labels=init))
        )
        got = run_cli(rows, dh, init=init)
        self.assertEqual(got["assignments"], expected.tolist())
        self.assertEqual(got["k_actual"], len(set(expected.tolist())))
        self.assertEqual(got["noise_count"], 0)
        # The seed partition is already the family split, and the models keep it.
        self.assertEqual(got["assignments"][:3], [got["assignments"][0]] * 3)
        self.assertEqual(got["assignments"][3:], [got["assignments"][3]] * 3)

    def test_stride_arm_matches_reference(self):
        rows, dh = rows_and_dh()
        expected, _ = compact_labels(
            np.asarray(sequence_labels(rows, K, MAX_ITER, dh))
        )
        got = run_cli(rows, dh)
        self.assertEqual(got["assignments"], expected.tolist())


if __name__ == "__main__":
    unittest.main()
