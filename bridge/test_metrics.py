#!/usr/bin/env python3
"""The two measures the thesis defines: the entropic relevance of a graph and the fit
measure of the box objective, pinned by hand and against the clear twin of the box."""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bridge.metrics import entropic_relevance, transition_log_likelihood  # noqa: E402
from scripts.plaintext_sequence_reference import (  # noqa: E402
    bigram_rows,
    encode_traces,
    log_weights,
)


class EntropicRelevanceTests(unittest.TestCase):
    """Polyvyanyy, Moffat, Garcia-Banuelos 2020, Definitions IV.1 and IV.2, on a graph
    read as an automaton."""

    # START -> a (4); a -> b (3), a -> c (1); b -> END (3), c -> END (1)
    DFG = {"edges": [{"from": ">", "to": "a", "weight": 4}, {"from": "a", "to": "b", "weight": 3},
                     {"from": "a", "to": "c", "weight": 1}, {"from": "b", "to": "|", "weight": 3},
                     {"from": "c", "to": "|", "weight": 1}]}

    def test_a_log_the_graph_allows_costs_the_mean_code_length_of_its_traces(self):
        r = entropic_relevance([["a", "b"]] * 3 + [["a", "c"]], self.DFG)
        self.assertAlmostEqual(r["rel"], (3 * -math.log2(3 / 4) + -math.log2(1 / 4)) / 4)
        self.assertEqual(r["rho"], 1.0)

    def test_a_trace_the_graph_does_not_allow_is_coded_by_the_background_model(self):
        # <a, d>: the step a -> d has no edge. Three activities a, b, d are scored, so a
        # symbol costs log2(4) bits, and the trace with its end symbol has three symbols.
        # Half of the traces are allowed, so H0 is one bit.
        r = entropic_relevance([["a", "b"], ["a", "d"]], self.DFG)
        self.assertAlmostEqual(r["rel"], 1.0 + (-math.log2(3 / 4) + 3 * math.log2(4)) / 2)
        self.assertEqual(r["rho"], 0.5)

    def test_the_unseen_step_is_not_charged_a_flat_penalty(self):
        r = entropic_relevance([["a", "d"]], self.DFG)
        self.assertAlmostEqual(r["rel"], 3 * math.log2(3))
        self.assertEqual(r["rho"], 0.0)

    def test_empty_traces_are_not_scored(self):
        self.assertIsNone(entropic_relevance([[], []], self.DFG))
        self.assertEqual(entropic_relevance([["a", "b"], []], self.DFG)["traces"], 1)


class FitMeasureConventionTests(unittest.TestCase):
    """Laplace-smoothed log2 per transition, size-weighted, START and END in, no unseen floor."""

    def test_hand_computed_two_cluster_fixture(self):
        # Alphabet {a, b, c} -> d_eff = 4 admissible targets (a, b, c, END).
        # Cluster 0: two copies of [a, b] -> counts >a:2, ab:2, b|:2, rows 2/2/2.
        # Cluster 1: one [a, c]           -> counts >a:1, ac:1, c|:1, rows 1/1/1.
        c0 = [["a", "b"], ["a", "b"]]
        c1 = [["a", "c"]]
        w0 = math.log2(3.0) - math.log2(6.0)      # count 2 + 1, rowsum 2 + 4
        w1 = math.log2(2.0) - math.log2(5.0)      # count 1 + 1, rowsum 1 + 4
        expected = (6 * w0 + 3 * w1) / 9          # 3 transitions per trace
        self.assertAlmostEqual(
            transition_log_likelihood([c0, c1]), expected, places=12)

    def test_start_and_end_transitions_are_counted(self):
        # One single-event trace = exactly two transitions (>a and a|).
        one = transition_log_likelihood([[["a"]]])
        w = math.log2(2.0) - math.log2(3.0)       # d_eff = 2 (a, END)
        self.assertAlmostEqual(one, w, places=12)

    def test_no_unseen_floor(self):
        # A trace under a model that never saw its transitions still gets the smoothed value.
        val = transition_log_likelihood([[["a", "b"]], [["c", "d"]]])
        self.assertGreater(val, -6.0)

    def test_alphabet_is_dictionary_wide(self):
        # d_eff comes from the whole partition, as DH does in the box.
        joint = transition_log_likelihood([[["a", "b"]], [["c", "d"]]])
        alone = transition_log_likelihood([[["a", "b"]]])
        self.assertNotAlmostEqual(joint, alone, places=6)


class TwinParityTests(unittest.TestCase):
    """The same value the objective of scripts/plaintext_sequence_reference.py scores,
    the clear twin of mp-spdz/secure_sequence_cluster.mpc."""

    @staticmethod
    def _twin_value(traces, labels, k):
        enc, dh, _ = encode_traces(traces)
        X = np.asarray(bigram_rows(enc, dh), dtype=np.int64)
        C = np.zeros((k, dh * dh), dtype=np.int64)
        for j in range(k):
            members = np.flatnonzero(np.asarray(labels) == j)
            if members.size:
                C[j] = X[members].sum(axis=0)
        W = log_weights(C, dh)
        total = sum(float(X[i] @ W[labels[i]]) for i in range(len(X)))
        return total / float(X.sum())

    def test_matches_twin_on_random_partition(self):
        rng = random.Random(20260902)
        acts = ["a", "b", "c", "d", "e"]
        traces = [[rng.choice(acts) for _ in range(rng.randint(1, 9))]
                  for _ in range(60)]
        k = 3
        labels = [rng.randrange(k) for _ in range(len(traces))]
        for j in range(k):                       # no empty cluster
            labels[j] = j
        per_cluster = [[t for t, lab in zip(traces, labels) if lab == j]
                       for j in range(k)]
        mine = transition_log_likelihood(per_cluster)
        twin = self._twin_value(traces, labels, k)
        self.assertLess(abs(mine - twin), 1e-9, f"{mine} vs {twin}")


if __name__ == "__main__":
    unittest.main()
