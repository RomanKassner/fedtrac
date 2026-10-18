#!/usr/bin/env python3
"""Discovery from published graphs: Inductive Miner (PTML) and Heuristics Miner (PNML)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import edge_filter_keep_fn, load_dfg_from_json

if TYPE_CHECKING:
    from pm4py.objects.petri_net.obj import Marking, PetriNet


def compute_activity_occurrences(
    dfg: dict[tuple[str, str], float],
    start_acts: dict[str, float],
    end_acts: dict[str, float],
) -> dict[str, float]:
    """Activity occurrences as max(incoming, outgoing) flow per node."""
    incoming = {}
    outgoing = {}
    for (src, tgt), w in dfg.items():
        outgoing[src] = outgoing.get(src, 0) + w
        incoming[tgt] = incoming.get(tgt, 0) + w
    for act, w in start_acts.items():
        incoming[act] = incoming.get(act, 0) + w
    for act, w in end_acts.items():
        outgoing[act] = outgoing.get(act, 0) + w
    all_acts = set(incoming.keys()) | set(outgoing.keys())
    return {a: max(incoming.get(a, 0), outgoing.get(a, 0)) for a in all_acts}


def filter_edges(
    dfg: dict[tuple[str, str], float],
    min_count: float,
    min_fraction: float,
) -> dict[tuple[str, str], float]:
    """Drop the edges below the cutoff, the twin of src/phases/p4_aggregate.rs::filter_dfg_edges.

    This dfg excludes the start and end cells, so every weight in it enters the max.
    """
    if not dfg or (min_count <= 0 and min_fraction <= 0):
        return dfg
    keep = edge_filter_keep_fn(dfg.values(), min_count, min_fraction)
    return {k: v for k, v in dfg.items() if keep(v)}


def discover_from_dfg(dfg_path: str, output: str | None = None) -> None:
    """Inductive Miner from a published graph; writes a PTML when `output` is given."""
    from pm4py import discover_process_tree_inductive
    from pm4py.objects.dfg.obj import DFG as PM4PyDFG

    dfg, start_acts, end_acts, _raw = load_dfg_from_json(dfg_path)
    if not dfg:
        print(f"Warning: empty DFG in {dfg_path}, skipping", file=sys.stderr)
        return

    dfg_obj = PM4PyDFG(graph=dfg, start_activities=start_acts, end_activities=end_acts)
    tree = discover_process_tree_inductive(dfg_obj, noise_threshold=0.0)

    if output:
        from pm4py.objects.process_tree.exporter import exporter as ptml_exporter
        ptml_exporter.apply(tree, output)
        print(f"Process tree saved to {output}")
    print(f"Process tree: {tree}")


def discover_from_dfg_heuristics(
    dfg_path: str, output: str | None = None,
    dependency_thresh: float = 0.5, and_thresh: float = 0.65, noise_thresh: float = 0.05,
) -> tuple["PetriNet", "Marking", "Marking"] | tuple[None, None, None]:
    """Heuristics Miner from a published graph; writes a PNML when `output` is given."""
    from pm4py.algo.discovery.heuristics.algorithm import apply_dfg

    dfg, start_acts, end_acts, _ = load_dfg_from_json(dfg_path)
    if not dfg:
        print(f"Warning: empty DFG in {dfg_path}, skipping", file=sys.stderr)
        return None, None, None

    activities: set[str] = set()
    for (src, tgt) in dfg:
        activities.add(src)
        activities.add(tgt)
    activities.update(start_acts.keys())
    activities.update(end_acts.keys())
    act_occ = compute_activity_occurrences(dfg, start_acts, end_acts)

    net, im, fm = apply_dfg(
        dfg, activities=activities, activities_occurrences=act_occ,
        start_activities=start_acts, end_activities=end_acts,
        parameters={
            "dependency_thresh": dependency_thresh,
            "and_measure_thresh": and_thresh,
            "min_dfg_occurrences": 1,
            "dfg_pre_cleaning_noise_thresh": noise_thresh,
        },
    )
    if output:
        from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
        pnml_exporter.apply(net, im, output, final_marking=fm)
        print(f"Petri net saved to {output}")
    print(f"Heuristics Miner: {len(net.places)} places, {len(net.transitions)} transitions")
    return net, im, fm


def discover_tree(dfg_path: Path, tree_path: Path) -> bool:
    """Inductive Miner graph -> PTML; True iff the file was written."""
    try:
        discover_from_dfg(str(dfg_path), output=str(tree_path))
    except Exception as e:
        print(f"  ERROR in discovery: {e}")
        return False
    return Path(tree_path).exists()


def discover_petri_net_hm(dfg_path: Path, pnml_path: Path) -> bool:
    """Heuristics Miner graph -> PNML; True iff the file was written."""
    try:
        net, _, _ = discover_from_dfg_heuristics(str(dfg_path), output=str(pnml_path))
    except Exception as e:
        print(f"  ERROR in HM discovery: {e}")
        return False
    return net is not None and Path(pnml_path).exists()


def merge_dfgs(dfg_files: list[Path], output_path: Path) -> Path:
    """Sum the cluster graphs into the graph of the whole log."""
    merged_edges: dict[tuple[str, str], float] = {}
    merged_start: dict[str, float] = {}
    merged_end: dict[str, float] = {}
    total_traces = 0

    for dfg_path in dfg_files:
        with open(dfg_path) as f:
            data = json.load(f)
        total_traces += data.get("num_traces", 0)
        for edge in data["edges"]:
            key = (edge["from"], edge["to"])
            merged_edges[key] = merged_edges.get(key, 0) + edge["weight"]
        for act, w in data.get("start_activities", {}).items():
            merged_start[act] = merged_start.get(act, 0) + w
        for act, w in data.get("end_activities", {}).items():
            merged_end[act] = merged_end.get(act, 0) + w

    activities: set[str] = set()
    edges_list = []
    for (src, tgt), w in sorted(merged_edges.items()):
        edges_list.append({"from": src, "to": tgt, "weight": w})
        if src not in (">", "|"):
            activities.add(src)
        if tgt not in (">", "|"):
            activities.add(tgt)

    merged = {
        "cluster_id": "global",
        "num_traces": total_traces,
        "activities": sorted(activities),
        "edges": edges_list,
        "start_activities": merged_start,
        "end_activities": merged_end,
    }
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return output_path


def merge_traces(trace_files: list[Path], output_path: Path) -> Path:
    all_traces = []
    for tf in trace_files:
        with open(tf) as f:
            data = json.load(f)
        all_traces.extend(data["traces"])
    merged = {"num_traces": len(all_traces), "traces": all_traces}
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return output_path
