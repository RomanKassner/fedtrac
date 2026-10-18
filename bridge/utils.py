"""Shared helpers of the bridge modules: event logs, published graphs, the edge filter."""
from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from pm4py.objects.log.obj import EventLog


def traces_json_to_event_log(traces_json_path: str) -> "EventLog":
    from pm4py.objects.log.obj import Event, EventLog, Trace

    with open(traces_json_path) as f:
        data = json.load(f)

    log = EventLog()
    for i, activity_list in enumerate(data["traces"]):
        trace = Trace()
        trace.attributes["concept:name"] = f"case_{i}"
        for j, act in enumerate(activity_list):
            event = Event()
            event["concept:name"] = act
            event["time:timestamp"] = j
            trace.append(event)
        log.append(trace)
    return log


def prune_to_start_end_paths(
    dfg: dict[tuple[str, str], float],
    start_acts: dict[str, float],
    end_acts: dict[str, float],
) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float], list[str]]:
    """(dfg, start, end, removed) without the activities on no path from a start to an end.

    Clear twin of mp-spdz/secure_aggregate.mpc Step 3c and of
    src/phases/p4_aggregate.rs::prune_dead_ends. The edge filter removes activity edges
    and keeps every start and end count, so a published graph can hold an activity that
    no start activity reaches or that reaches no end activity; this is the soundness
    condition of the directly-follows model.
    """
    succ: dict[str, set[str]] = {}
    pred: dict[str, set[str]] = {}
    for (a, b), w in dfg.items():
        if w > 0:
            succ.setdefault(a, set()).add(b)
            pred.setdefault(b, set()).add(a)

    def reach(seed, step):
        seen, todo = set(seed), list(seed)
        while todo:
            for y in step.get(todo.pop(), ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return seen

    keep = reach([a for a, w in start_acts.items() if w > 0], succ) & reach(
        [a for a, w in end_acts.items() if w > 0], pred)
    activities = set(start_acts) | set(end_acts) | {a for edge in dfg for a in edge}
    return (
        {(a, b): w for (a, b), w in dfg.items() if a in keep and b in keep and w > 0},
        {a: w for a, w in start_acts.items() if a in keep},
        {a: w for a, w in end_acts.items() if a in keep},
        sorted(activities - keep),
    )


def load_dfg_from_json(
    path: str,
) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float], dict]:
    """Load a published graph as (dfg, start_acts, end_acts, raw), marker activities dropped."""
    with open(path) as f:
        data = json.load(f)

    dfg = {}
    skip_acts = {">", "|", "?", ""}
    for edge in data["edges"]:
        src, tgt, w = edge["from"], edge["to"], edge["weight"]
        if src in skip_acts or tgt in skip_acts:
            continue
        dfg[(src, tgt)] = w

    start_activities = {}
    for edge in data["edges"]:
        if edge["from"] == ">":
            start_activities[edge["to"]] = edge["weight"]

    end_activities = {}
    for edge in data["edges"]:
        if edge["to"] == "|":
            end_activities[edge["from"]] = edge["weight"]

    if "start_activities" in data and data["start_activities"]:
        start_activities = {k: v for k, v in data["start_activities"].items()
                           if k not in skip_acts}
    if "end_activities" in data and data["end_activities"]:
        end_activities = {k: v for k, v in data["end_activities"].items()
                         if k not in skip_acts}

    return dfg, start_activities, end_activities, data


def compute_weighted_avg(results: list[dict], keys: tuple[str, ...] = ("fitness", "precision",
                                               "generalization", "simplicity",
                                               "f_score", "buijs")) -> dict[str, float] | None:
    """Size-weighted mean of each key; a key with no value anywhere is omitted."""
    if not results:
        return None
    if sum(r.get("size", 0) for r in results) == 0:
        return None
    out: dict[str, float] = {}
    for key in keys:
        num = 0.0
        den = 0
        for r in results:
            v = r.get(key)
            if v is None:
                continue
            w = r.get("size", 1)
            num += v * w
            den += w
        if den:
            out[key] = round(num / den, 4)
    return out


# 2^AGG_CELL_BITS, the width the .mpc compares EDGE_MIN_COUNT at
# (src/constants.rs, mp-spdz/mpc_utils/constants.py).
EDGE_MIN_COUNT_MAX = 1 << 41


def edge_filter_int_args(min_count: float, min_fraction: float):
    """Integer twin of src/mpc/rounds.rs::edge_filter_int_args: (ceil(min_count), fraction * 10^4),
    None when the fraction has no exact 10^-4 form or the count exceeds the in-MPC width."""
    if not 0.0 <= min_fraction <= 1.0 or min_count < 0.0:
        return None
    # Rust f64::round is half-away-from-zero; Python's round() is half-to-even.
    frac_num = math.floor(float(min_fraction) * 10_000.0 + 0.5)
    if abs(frac_num / 10_000.0 - min_fraction) > 1e-13:
        return None
    mc = math.ceil(float(min_count))
    if mc >= EDGE_MIN_COUNT_MAX:
        return None
    return int(mc), int(frac_num)


def edge_filter_keep_fn(weights, min_count: float, min_fraction: float):
    """Edge-filter predicate `keep(weight) -> bool`, the integer twin of
    src/phases/p4_aggregate.rs::filter_dfg_edges: `w >= mc and w * 10^4 >= frac_num *
    max_weight`, the float cutoff when `edge_filter_int_args` returns None.
    `weights` are activity weights only."""
    max_weight = max(weights, default=0.0)
    args = edge_filter_int_args(min_count, min_fraction)
    if args is None:
        cutoff = max(float(min_count), float(min_fraction) * max_weight)
        return lambda w: w >= cutoff
    mc, frac_num = args
    return lambda w: w >= mc and w * 10_000.0 >= frac_num * max_weight


def load_xes(path: str) -> "pd.DataFrame":
    """Read an XES file into a pm4py data frame."""
    import pm4py

    return pm4py.convert_to_dataframe(pm4py.read_xes(path))
