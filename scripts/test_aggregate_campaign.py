#!/usr/bin/env python3
"""aggregate_campaign.py: the columns and the failure rule, on a minimal campaign dir plus run dirs."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import aggregate_campaign as agg  # noqa: E402
from bridge.metrics import transition_log_likelihood  # noqa: E402


def write_tier(run: Path, tier: str, f_hm: float, *, coverage=1.0,
               dropped=(), conformance=None, k=3) -> None:
    d = run / tier
    d.mkdir(parents=True)
    (d / "evaluation_results.json").write_text(json.dumps({
        "tier": tier,
        "conformance": conformance or {},
        "coverage_hm": {"eval_coverage": coverage, "dropped_clusters": list(dropped)},
        "quality_metrics": {"headline": {"f_hm_all": f_hm}},
        "clusters_hm": [{"size": 1} for _ in range(k)],
    }))
    (d / "cluster_assignments.json").write_text(json.dumps({
        "case_ids": [f"c{i}" for i in range(k)],
        "assignments": list(range(k)),
    }))


def write_run(run: Path, *, pt=0.80, proj=0.78, fed=0.75, coverage=1.0,
              dropped=(), conformance=None, pt_k=3, timing=None) -> None:
    write_tier(run, "plaintext", pt, k=pt_k)
    write_tier(run, "projected", proj)
    write_tier(run, "federated", fed, coverage=coverage, dropped=dropped,
               conformance=conformance)
    (run / "timing.json").write_text(json.dumps(timing or {
        "round1_secure": {"exec_s": 10.0, "compile_s": 1.0},
        "secure_kmeans": {"exec_s": 4.0, "compile_s": 1.0},
        "_generation": 3,
    }))


def write_traces(run: Path, tier: str, clusters: list[list[list[str]]]) -> None:
    """The cluster trace exports the fit column is scored over."""
    d = run / tier
    d.mkdir(parents=True, exist_ok=True)
    for cid, traces in enumerate(clusters):
        (d / f"cluster_{cid}_traces.json").write_text(json.dumps({"traces": traces}))


def campaign(root: Path, manifest: str, summary: str) -> Path:
    camp = root / "campaign"
    camp.mkdir()
    (camp / "manifest.tsv").write_text(manifest)
    (camp / "summary.tsv").write_text(summary)
    return camp


def run_collect(camp: Path):
    summary_rows = agg.load_tsv(str(camp / "summary.tsv"))
    default = agg.meta_case_universe(str(camp))
    return agg.collect(
        agg.manifest_runs(str(camp)), summary_rows,
        agg.manifest_case_universes(str(camp), default), default,
        agg.manifest_encodings(str(camp)),
        agg.manifest_ring_overflow(str(camp), summary_rows))


class RingOverflowColumnTests(unittest.TestCase):

    def test_bypassed_cell_is_marked(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            camp = campaign(
                root,
                f"cell\trun_dir\tflags\tring_overflow\n"
                f"helpdesk_sl_seq\t{run}\t--encoding ef\t1\n",
                "cell\texit\tk\twallclock_s\nhelpdesk_sl_seq\t0\t6\t500\n")
            _, cells, failed = run_collect(camp)
            self.assertEqual(failed, [])
            self.assertEqual(cells[0]["ring_overflow"], "1")
            self.assertEqual(cells[0]["encoding"], "ef")
            self.assertEqual(cells[0]["variant"], "seq")

    def test_no_column_reads_unknown_not_clean(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            camp = campaign(root, f"cell\trun_dir\nsepsis_sl_km\t{run}\n",
                            "cell\texit\nsepsis_sl_km\t0\n")
            _, cells, _ = run_collect(camp)
            self.assertEqual(cells[0]["ring_overflow"], "?")
            self.assertEqual(cells[0]["variant"], "km")   # the k-means twin


class FailureRuleTests(unittest.TestCase):

    def test_manifest_cell_without_summary_row_is_failed(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            camp = campaign(root, f"cell\trun_dir\nsepsis_sl_seq\t{run}\n",
                            "cell\texit\twallclock_s\n")
            _, cells, failed = run_collect(camp)
            self.assertEqual(cells, [])
            self.assertEqual(failed[0]["exit"], "missing")


class IngestedColumnTests(unittest.TestCase):

    def test_coverage_and_conformance_backend(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run, coverage=0.87, dropped=[{"cluster_id": 2}],
                      conformance={"token_replay": True, "fast_eval": True})
            camp = campaign(root, f"cell\trun_dir\nsepsis_sl_seq\t{run}\n",
                            "cell\texit\nsepsis_sl_seq\t0\n")
            _, cells, _ = run_collect(camp)
            row = cells[0]
            self.assertEqual(row["eval_coverage"], 0.87)
            self.assertEqual(row["dropped_clusters"], 1)
            self.assertIs(row["token_replay"], True)
            self.assertIs(row["fast_eval"], True)

    def test_hm_jl_and_mpc_split(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run, pt=0.80, proj=0.78, fed=0.75)
            camp = campaign(root, f"cell\trun_dir\nsepsis_sl_seq\t{run}\n",
                            "cell\texit\nsepsis_sl_seq\t0\n")
            _, cells, _ = run_collect(camp)
            row = cells[0]
            self.assertAlmostEqual(row["jl_cost_f_hm"], -0.02, places=4)
            self.assertAlmostEqual(row["mpc_cost_f_hm"], -0.03, places=4)
            self.assertAlmostEqual(row["delta_f_hm"], -0.05, places=4)
            # The HM split adds up to the HM headline delta; the IM columns
            # (from comparison.json, absent here) stay None.
            self.assertIsNone(row["jl_cost_f_im"])

    def test_baseline_k_is_the_plaintext_tier_k(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run, pt_k=5)
            camp = campaign(root, f"cell\trun_dir\nsepsis_sl_seq\t{run}\n",
                            "cell\texit\tk\nsepsis_sl_seq\t0\t6\n")
            _, cells, _ = run_collect(camp)
            self.assertEqual(cells[0]["baseline_k"], 5)   # != the cell's k=6
            self.assertEqual(cells[0]["k"], "6")


PT_CLUSTERS = [[["a", "b", "c"], ["a", "b", "c"], ["a", "b"]],
               [["d", "e"], ["d", "e", "d", "e"]]]
PROJ_CLUSTERS = [[["a", "b", "c"], ["a", "b"]],
                 [["d", "e"], ["d", "e", "d"], ["d", "e"]]]
FED_CLUSTERS = [[["a", "b"], ["a", "b", "c"]],
                [["d", "e", "d", "e"], ["d", "e"], ["d", "c"]]]


class FitColumnTests(unittest.TestCase):
    """The fit column is the box's own objective, scored by ONE helper."""

    def _cells(self, root: Path, run: Path):
        camp = campaign(root, f"cell\trun_dir\nsepsis_sl_seq\t{run}\n",
                        "cell\texit\nsepsis_sl_seq\t0\n")
        return run_collect(camp)

    def test_per_tier_fit_matches_a_direct_computation(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            for tier, clusters in (("plaintext", PT_CLUSTERS),
                                   ("projected", PROJ_CLUSTERS),
                                   ("federated", FED_CLUSTERS)):
                write_traces(run, tier, clusters)
            long_rows, cells, _ = self._cells(root, run)
            row = cells[0]
            pt = transition_log_likelihood(PT_CLUSTERS)
            proj = transition_log_likelihood(PROJ_CLUSTERS)
            fed = transition_log_likelihood(FED_CLUSTERS)
            self.assertEqual(row["pt_fit_log2"], pt)
            self.assertEqual(row["proj_fit_log2"], proj)
            self.assertEqual(row["fed_fit_log2"], fed)
            self.assertEqual(row["jl_cost_fit_log2"], round(proj - pt, 4))
            self.assertEqual(row["mpc_cost_fit_log2"], round(fed - proj, 4))
            by_tier = {r["tier"]: r["fit_log2"] for r in long_rows}
            self.assertEqual(by_tier, {"plaintext": pt, "projected": proj,
                                       "federated": fed})
            reps = agg.repeats_summary(cells)
            self.assertEqual(reps[0]["fed_fit_log2_mean"], round(fed, 4))
            self.assertEqual(reps[0]["fed_fit_log2_std"], 0.0)

    def test_a_tier_without_traces_scores_none(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            # The sequence algorithm exports no `projected/` traces.
            write_traces(run, "plaintext", PT_CLUSTERS)
            write_traces(run, "federated", FED_CLUSTERS)
            long_rows, cells, _ = self._cells(root, run)
            row = cells[0]
            self.assertIsNone(row["proj_fit_log2"])
            self.assertIsNone(row["jl_cost_fit_log2"])
            self.assertIsNone(row["mpc_cost_fit_log2"])
            self.assertEqual(row["fed_fit_log2"],
                             transition_log_likelihood(FED_CLUSTERS))
            self.assertIsNone({r["tier"]: r["fit_log2"]
                               for r in long_rows}["projected"])

    def test_a_run_with_no_traces_at_all_is_no_crash(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "cell"
            write_run(run)
            long_rows, cells, _ = self._cells(root, run)
            self.assertEqual([cells[0][c] for c in
                              ("pt_fit_log2", "proj_fit_log2", "fed_fit_log2")],
                             [None, None, None])
            self.assertTrue(all(r["fit_log2"] is None for r in long_rows))
            self.assertIsNone(agg.repeats_summary(cells)[0]["fed_fit_log2_mean"])


class TimingSplitTests(unittest.TestCase):

    def test_kselect_probe_rounds_leave_mpc_exec_s(self):
        with TemporaryDirectory() as td:
            run = Path(td) / "run"
            write_run(run, timing={
                "round1_secure": {"exec_s": 10.0},
                "secure_kmeans": {"exec_s": 4.1},
                "secure_kmeans_2": {"exec_s": 30.0},
                "secure_kmeans_3": {"exec_s": 41.9},
                "round2_aggregate": {"exec_s": 5.0},
                "_generation": 2,
            })
            t = agg.timing_columns(str(run))
            self.assertAlmostEqual(t["mpc_exec_s"], 19.1, places=3)
            self.assertAlmostEqual(t["mpc_exec_s_kselect"], 71.9, places=3)
            self.assertEqual(t["generation"], 2)


if __name__ == "__main__":
    unittest.main()
