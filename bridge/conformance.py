#!/usr/bin/env python3
"""Per-cluster conformance of the in-run evaluation, and the single-model baseline."""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

from .discovery import discover_petri_net_hm, discover_tree
from .metrics import structural_complexity
from .report import print_summary_line


class _SoundnessTimeout(Exception):
    pass


def _soundness_alarm(signum, frame):
    raise _SoundnessTimeout()


# pm4py's easy-soundness check is an exhaustive path search that does not terminate on some
# Heuristics Miner nets; `sound` is a diagnostic field, so a bound on it moves no reported number.
SOUNDNESS_TIMEOUT_S = 30


def check_soundness(pnml_path: Path) -> bool | None:
    """Is this a sound WF-net? None = error, or the search exceeded SOUNDNESS_TIMEOUT_S."""
    try:
        from pm4py.objects.petri_net.importer import importer as pnml_importer
        from pm4py.objects.petri_net.utils.check_soundness import (
            check_easy_soundness_of_wfnet,
            check_wfnet,
        )
        net, _, _ = pnml_importer.apply(str(pnml_path))
        if not check_wfnet(net):
            return False
        prev = signal.signal(signal.SIGALRM, _soundness_alarm)
        signal.alarm(SOUNDNESS_TIMEOUT_S)
        try:
            return check_easy_soundness_of_wfnet(net)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
    except _SoundnessTimeout:
        print(f"  WARN: soundness check exceeded {SOUNDNESS_TIMEOUT_S}s on "
              f"{pnml_path}; recording unknown", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  WARN: soundness check failed on {pnml_path}: {e}", file=sys.stderr)
        return None


def run_conformance(tree_path: Path, traces_path: Path,
                    petri_net_path: Path | None = None) -> dict | None:
    """Fitness and precision by token replay, generalization, simplicity and F."""
    from pm4py import convert_to_petri_net
    from pm4py.algo.evaluation.generalization import algorithm as gen_eval
    from pm4py.algo.evaluation.simplicity import algorithm as simpl_eval

    try:
        if petri_net_path:
            from pm4py.objects.petri_net.importer import importer as pnml_importer
            net, im, fm = pnml_importer.apply(str(petri_net_path))
        else:
            from pm4py.objects.process_tree.importer import importer as ptml_importer
            tree = ptml_importer.apply(str(tree_path))
            net, im, fm = convert_to_petri_net(tree)
        from .utils import traces_json_to_event_log
        log = traces_json_to_event_log(str(traces_path))

        from pm4py import fitness_token_based_replay, precision_token_based_replay
        fitness_result = fitness_token_based_replay(log, net, im, fm)
        precision = precision_token_based_replay(log, net, im, fm)
        fitness = (
            fitness_result.get("average_trace_fitness", 0.0)
            if isinstance(fitness_result, dict) else fitness_result
        )

        def _safe(label, fn, *a):
            try:
                return fn(*a)
            except Exception as e:
                print(f"  WARN: {label} eval failed: {e}", file=sys.stderr)
                return None

        generalization = _safe("generalization", gen_eval.apply, log, net, im, fm)
        simplicity = _safe("simplicity", simpl_eval.apply, net)

        f_score = (2 * fitness * precision / (fitness + precision)
                   if (fitness + precision) > 0 else 0.0)
        # Weighted quality over the four dimensions of Buijs et al. 2014; the 10/10/1/1 weights are
        # ours, the paper leaves them to the user. Undefined unless all four were evaluated.
        buijs = (
            (10 * fitness + 10 * precision + generalization + simplicity) / 22.0
            if generalization is not None and simplicity is not None
            else None
        )
    except Exception as e:
        print(f"  ERROR in conformance: {e}")
        return None

    result = {
        "fitness": round(fitness, 4),
        "precision": round(precision, 4),
        "f_score": round(f_score, 4),
    }
    # A failed component stays absent, so the weighted averages skip it instead of folding a
    # sentinel into the reported numbers.
    for k, v in (("generalization", generalization), ("simplicity", simplicity),
                 ("buijs", buijs)):
        if v is not None:
            result[k] = round(v, 4)
    for k in ("fitness", "precision", "generalization", "simplicity", "f_score"):
        tag = k.replace("_", " ").title() if k != "f_score" else "F-Score"
        v = result.get(k)
        print(f"    {tag}: {v:.4f}" if v is not None else f"    {tag}: n/a")
    return result


def collect_cluster_info(
    input_dir: Path,
    dfg_files: list[Path],
) -> list[tuple[str, Path, Path, int]]:
    """(cluster_id, dfg_path, traces_path, size) of every non-empty cluster."""
    cluster_info: list[tuple[str, Path, Path, int]] = []
    for dfg_path in dfg_files:
        cluster_id = dfg_path.stem.replace("cluster_", "").replace("_dfg", "")
        traces_path = input_dir / f"cluster_{cluster_id}_traces.json"
        if not traces_path.exists():
            continue
        with open(traces_path) as f:
            size = json.load(f)["num_traces"]
        if size == 0:
            continue
        cluster_info.append((cluster_id, dfg_path, traces_path, size))
    return cluster_info


def _evaluate_clusters(
    input_dir: Path,
    cluster_info: list[tuple[str, Path, Path, int]],
    *,
    label: str,
    artefact_suffix: str,
    discover_fn,
    with_structural: bool,
) -> tuple[list[dict], dict]:
    """Discover and score every cluster; `dropped_clusters` left the weighted
    averages, `eval_coverage` is the scored share of the cases."""
    is_petri = artefact_suffix.endswith(".pnml")
    results: list[dict] = []
    dropped: list[dict] = []

    def _drop(cluster_id: str, size: int, reason: str) -> None:
        dropped.append({"id": int(cluster_id), "size": size})
        # HM is the headline miner: surface any silently biased-out cluster.
        if label == "HM":
            print(f"  WARNING: HM {reason} for cluster {cluster_id} "
                  f"({size} traces); dropped from weighted averages")

    for cluster_id, dfg_path, traces_path, size in cluster_info:
        print(f"--- Cluster {cluster_id} ({size} traces, {label}) ---")
        artefact = input_dir / f"cluster_{cluster_id}{artefact_suffix}"
        if not discover_fn(dfg_path, artefact):
            _drop(cluster_id, size, "discovery failed")
            continue
        metrics = run_conformance(
            artefact, traces_path,
            petri_net_path=artefact if is_petri else None,
        )
        if metrics:
            metrics["cluster_id"] = int(cluster_id)
            metrics["size"] = size
            if with_structural:
                metrics["sound"] = check_soundness(artefact)
                if (sc := structural_complexity(artefact)):
                    metrics["structure"] = sc
            results.append(metrics)
        else:
            _drop(cluster_id, size, "conformance failed")
        print()

    total = sum(size for _, _, _, size in cluster_info)
    evaluated = sum(r["size"] for r in results)
    coverage = {
        "dropped_clusters": dropped,
        "eval_coverage": 1.0 if total == 0 else round(evaluated / total, 4),
    }
    return results, coverage


def evaluate_clusters_im(input_dir, cluster_info) -> tuple[list[dict], dict]:
    return _evaluate_clusters(
        input_dir, cluster_info,
        label="IM", artefact_suffix="_tree.ptml", with_structural=False,
        discover_fn=discover_tree,
    )


def evaluate_clusters_hm(input_dir, cluster_info) -> tuple[list[dict], dict]:
    return _evaluate_clusters(
        input_dir, cluster_info,
        label="HM", artefact_suffix="_hm.pnml", with_structural=True,
        discover_fn=discover_petri_net_hm,
    )


def run_single_model_baseline(
    input_dir: Path,
    dfg_files: list[Path],
    trace_files: list[Path],
) -> tuple[dict, dict]:
    """IM and HM on the merged graph of the whole log -> (im, hm) metrics."""
    from .discovery import merge_dfgs, merge_traces

    print("=== Single-Model Baseline ===")
    global_dfg_path = input_dir / "global_dfg.json"
    global_traces_path = input_dir / "global_traces.json"
    global_tree_path = input_dir / "global_tree.ptml"

    merge_dfgs(dfg_files, global_dfg_path)
    merge_traces(trace_files, global_traces_path)
    print(f"  Merged {len(dfg_files)} DFGs into global DFG")

    single_model: dict = {}
    print("  Discovering global process tree...")
    if discover_tree(global_dfg_path, global_tree_path):
        print("  Computing conformance (all traces)...")
        single_model = run_conformance(global_tree_path, global_traces_path) or {}

    if single_model:
        print()
        print_summary_line("Single-model (IM)", single_model)

    print("\n=== Global HM Baseline ===")
    global_hm_pnml_path = input_dir / "global_hm.pnml"
    global_hm: dict = {}
    print("  Discovering global Petri net (Heuristics Miner)...")
    if discover_petri_net_hm(global_dfg_path, global_hm_pnml_path):
        print("  Computing conformance (all traces)...")
        global_hm = run_conformance(global_hm_pnml_path, global_traces_path,
                                    petri_net_path=global_hm_pnml_path) or {}

    if global_hm:
        print()
        print_summary_line("Global HM", global_hm)

    return single_model, global_hm
