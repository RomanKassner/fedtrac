#!/usr/bin/env python3
"""The edge filter and the dead-end pruning: the two rules the Rust core, the secure
program and the clear twins must agree on."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.discovery import filter_edges  # noqa: E402
from bridge.utils import (  # noqa: E402
    EDGE_MIN_COUNT_MAX,
    edge_filter_int_args,
    edge_filter_keep_fn,
    load_dfg_from_json,
    prune_to_start_end_paths,
)


def filter_envelope(envelope: dict, min_count: float, min_fraction: float):
    """(kept activity edges, start, end) of a published graph, as the pipeline filters it."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cluster_0_dfg.json"
        path.write_text(json.dumps(envelope))
        dfg, start, end, _ = load_dfg_from_json(str(path))
    return filter_edges(dfg, min_count, min_fraction), start, end


class FilterEdgesTests(unittest.TestCase):
    """Twin of src/phases/p4_aggregate.rs::filter_dfg_edges: the integer form of the
    in-MPC pre-reveal pass, `w >= ceil(min_count) and w * 10^4 >= frac_num * max`."""

    # Max weight 100; ("b", "c") sits exactly on the 7% rational boundary.
    DFG = {("a", "b"): 100.0, ("b", "c"): 7.0, ("c", "d"): 6.0}

    def test_rational_boundary_cell_is_kept(self):
        # 7 * 10^4 = 70000 >= 700 * 100 = 70000. The float rule dropped it
        # (0.07 * 100 == 7.000000000000001).
        self.assertIn(("b", "c"), filter_edges(self.DFG, 0, 0.07))

    def test_cell_a_hair_below_the_boundary_is_dropped(self):
        self.assertNotIn(("c", "d"), filter_edges(self.DFG, 0, 0.07))

    def test_cell_clear_of_the_boundary_is_kept_by_both_forms(self):
        # 0.05 * 100 = 5.0 exactly: float and integer forms agree.
        kept = filter_edges(self.DFG, 0, 0.05)
        self.assertEqual(set(kept), set(self.DFG))

    def test_min_count_is_ceiled_as_in_the_integer_form(self):
        kept = filter_edges(self.DFG, 6.5, 0.0)
        self.assertIn(("b", "c"), kept)
        self.assertNotIn(("c", "d"), kept)

    def test_no_filter_returns_the_dfg_unchanged(self):
        self.assertIs(filter_edges(self.DFG, 0, 0.0), self.DFG)

    def test_inexact_fraction_falls_back_to_the_float_cutoff(self):
        # 1/3 has no exact 10^-4 form, so the integer arguments are None.
        self.assertIsNone(edge_filter_int_args(0, 1.0 / 3.0))
        keep = edge_filter_keep_fn([100.0], 0, 1.0 / 3.0)
        self.assertTrue(keep(34.0))
        self.assertFalse(keep(33.0))

    def test_agg_cell_bits_matches_the_width_of_the_secure_program(self):
        text = (ROOT / "mp-spdz" / "mpc_utils" / "constants.py").read_text()
        bits = int(re.search(r"^AGG_CELL_BITS\s*=\s*(\d+)", text, re.M).group(1))
        self.assertEqual(EDGE_MIN_COUNT_MAX, 1 << bits)


class FilterKeepsTheEnvelopeTests(unittest.TestCase):
    """The start and end counts track the cluster size, not the flow: they stay out of
    the cutoff max and are never dropped."""

    ENVELOPE = {"cluster_id": 0, "edges": [
        {"from": ">", "to": "a", "weight": 100},
        {"from": "a", "to": "b", "weight": 10},
        {"from": "b", "to": "c", "weight": 1},
        {"from": "c", "to": "|", "weight": 100},
    ]}

    def test_cutoff_ignores_start_and_end_weights(self):
        # cutoff = max(2, 0.5 * 10) = 5, so a -> b survives; the start weight would make it 50.
        dfg, _, _ = filter_envelope(self.ENVELOPE, 2, 0.5)
        self.assertIn(("a", "b"), dfg)
        self.assertNotIn(("b", "c"), dfg)

    def test_start_and_end_counts_survive_any_cutoff(self):
        _, start, end = filter_envelope(self.ENVELOPE, 1000, 0.0)
        self.assertEqual((start, end), ({"a": 100}, {"c": 100}))


class PruneToStartEndPathsTests(unittest.TestCase):
    """The published graph keeps every start and end count while the edge filter removes
    activity edges, so an activity can stand on no path from a start to an end activity."""

    # a -> b -> c is the path; x only ends (no start reaches it), y only starts (it reaches no end).
    DFG = {("a", "b"): 3.0, ("b", "c"): 3.0, ("x", "b"): 1.0, ("b", "y"): 2.0}
    START = {"a": 3.0, "y": 1.0}
    END = {"c": 3.0, "x": 1.0}

    def test_activities_off_every_start_to_end_path_are_removed_with_their_counts(self):
        dfg, start, end, removed = prune_to_start_end_paths(self.DFG, self.START, self.END)
        self.assertEqual(removed, ["x", "y"])
        self.assertEqual(dfg, {("a", "b"): 3.0, ("b", "c"): 3.0})
        self.assertEqual((start, end), ({"a": 3.0}, {"c": 3.0}))

    def test_a_graph_whose_activities_all_lie_on_a_path_is_returned_whole(self):
        dfg = {("a", "b"): 2.0, ("b", "b"): 1.0, ("b", "c"): 2.0}
        out, start, end, removed = prune_to_start_end_paths(dfg, {"a": 2.0}, {"c": 2.0})
        self.assertEqual((out, start, end, removed), (dfg, {"a": 2.0}, {"c": 2.0}, []))

    def test_an_edge_of_count_zero_carries_no_path(self):
        _, _, _, removed = prune_to_start_end_paths(
            {("a", "b"): 0.0, ("a", "c"): 1.0}, {"a": 1.0}, {"b": 1.0, "c": 1.0})
        self.assertEqual(removed, ["b"])


if __name__ == "__main__":
    unittest.main()
