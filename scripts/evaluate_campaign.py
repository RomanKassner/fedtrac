#!/usr/bin/env python3
"""Model quality of every run of a campaign, by the one instrument the thesis reports.

THE MODEL IS THE PUBLISHED GRAPH ITSELF: a directly follows model in the sense of Leemans, Poppe and Wynn 2019,
sound once every activity lies on a path from a start to an end activity (their Definition 3), which is the pruning.
Per tier folder of a run (plaintext, projected, federated) and per non-empty cluster: the published graph without the
activities on no start-to-end path (bridge.utils.prune_to_start_end_paths), pm4py's Petri net of it (a place per
activity, a transition per edge labelled with the edge's target), fitness and precision by alignments on the cluster's
traces with F as their harmonic mean, the edge and activity counts of the pruned graph, the arc-degree simplicity of
the net, the entropic relevance of the pruned graph to the traces by the published definition
(bridge.metrics.entropic_relevance), and Buijs's token-based generalization of the same net against the same log
(pm4py.algo.evaluation.generalization, variant token_based). The value of a tier is the mean over its non-empty
clusters weighted by their sizes; a cluster without a score is counted in `clusters`, not in `models_scored`, and
leaves the means of the model scores, never those of the graph.

TWO BASELINES per tier folder, computed in the clear from the stored traces the way the clustering in the clear forms
its graphs (the graph of the traces, the protocol's edge filter at count 2 and one fiftieth of the largest count, the
pruning): the whole log as one cluster, and five random partitions of the same cases into clusters of the same sizes
(seeds 1 to 5; De Weerdt et al. 2013 use a fivefold random clustering, Peeperkorn, De Smedt and De Weerdt 2025 a
random row). The sum of the published cluster graphs is no baseline: it is another object for every clustering.

    scripts/evaluate_campaign.py --campaign output/campaign_uniform_<stamp> [--workers 6] [--only REGEX] [--limit-s 1800]

Writes <campaign>/model_quality/<cell>__<tier>.json, one per tier folder, and <campaign>/model_quality.csv, one row per
cell and tier. A result is kept while it carries this instrument's tag and its run directory is the one the manifest
names, so a second call resumes and a repeated cell is evaluated again; any other tag is computed whole. Read only on
the run folders. The evaluation inside a campaign cell (bridge.evaluate, Heuristics Miner, token replay) is another
instrument and is not read here. Six workers by default: the net of a BPIC 2013 folder needs up to 3.4 GB.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

INSTRUMENT = "dfm-2026-10-15"      # the published graph as the model, four measures; another tag is computed again
TIERS = ("plaintext", "projected", "federated")
MIN_EDGE_COUNT, MIN_EDGE_FRACTION = 2.0, 0.02    # src/constants.rs DEFAULT_MIN_EDGE_COUNT, DEFAULT_MIN_EDGE_FRACTION
RANDOM_SEEDS = (1, 2, 3, 4, 5)
MODEL_KEYS = ("fitness", "precision", "f_score", "simplicity", "generalization")   # of the net, over the scored clusters
GRAPH_KEYS = ("edges", "activities", "rel", "rho")                  # of the graph, over every non-empty cluster
TIER_KEYS = (*MODEL_KEYS, *GRAPH_KEYS)
RANDOM_KEYS = ("fitness", "precision", "f_score", "simplicity", "generalization", "edges", "rel")
ADDED_KEYS = ("generalization",)   # appended columns, so every earlier column of model_quality.csv keeps its place
_OLD_TIER_KEYS = tuple(k for k in TIER_KEYS if k not in ADDED_KEYS)
_OLD_RANDOM_KEYS = tuple(k for k in RANDOM_KEYS if k not in ADDED_KEYS)
COLUMNS = ["cell", "tier", "run_dir", "clusters", "cases", "models_scored", "pruned_activities", *_OLD_TIER_KEYS,
           *(f"whole_{k}" for k in (*_OLD_TIER_KEYS, "pruned_activities")),
           *(f"random_{k}_{s}" for k in _OLD_RANDOM_KEYS for s in ("mean", "min", "max")),
           *ADDED_KEYS, *(f"whole_{k}" for k in ADDED_KEYS),
           *(f"random_{k}_{s}" for k in ADDED_KEYS for s in ("mean", "min", "max"))]


class _Limit(Exception):
    pass


def _alarm(signum, frame):
    raise _Limit()


def under_limit(limit_s, fn, *args):
    """fn(*args), or None when it does not return within the limit."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit_s))
    try:
        return fn(*args)
    except _Limit:
        return None
    finally:
        signal.alarm(0)


def token_generalization(net, im, fm, log):
    """Buijs's token-based generalization (Buijs 2014, Section 5.4.4) as pm4py computes it: 1 - (sum of
    1/sqrt(n_t) over the firing transitions + 1 per transition that never fires) / |T|, by token replay."""
    from pm4py.algo.evaluation.generalization import algorithm as generalization
    return generalization.apply(log, net, im, fm, parameters={"show_progress_bar": False})


def cluster_id(path):
    return int(re.search(r"cluster_(\d+)_", Path(path).name)[1])


def graph_from_traces(traces, min_count=MIN_EDGE_COUNT, min_fraction=MIN_EDGE_FRACTION):
    """(dfg, start, end) of the traces under the protocol's edge filter: the graph the clustering in the clear publishes."""
    from bridge.discovery import filter_edges
    dfg, start, end = {}, {}, {}
    for t in traces:
        if not t:
            continue
        start[t[0]] = start.get(t[0], 0.0) + 1.0
        end[t[-1]] = end.get(t[-1], 0.0) + 1.0
        for a, b in zip(t, t[1:]):
            dfg[(a, b)] = dfg.get((a, b), 0.0) + 1.0
    return filter_edges(dfg, min_count, min_fraction), start, end


def envelope(dfg, start, end):
    """The graph in the JSON form of a published graph, the form bridge.metrics.entropic_relevance reads."""
    edges = [{"from": ">", "to": a, "weight": w} for a, w in sorted(start.items())]
    edges += [{"from": a, "to": b, "weight": w} for (a, b), w in sorted(dfg.items())]
    edges += [{"from": a, "to": "|", "weight": w} for a, w in sorted(end.items())]
    return {"edges": edges, "start_activities": dict(start), "end_activities": dict(end)}


def to_event_log(traces):
    from pm4py.objects.log.obj import Event, EventLog, Trace
    log = EventLog()
    for i, acts in enumerate(traces):
        trace = Trace()
        trace.attributes["concept:name"] = f"case_{i}"
        for j, a in enumerate(acts):
            event = Event()
            event["concept:name"] = a
            event["time:timestamp"] = j
            trace.append(event)
        log.append(trace)
    return log


def score_graph(dfg, start, end, traces, limit_s):
    """One graph against its traces: the graph pruned to its start-to-end paths is the model of every measure."""
    import pm4py
    from pm4py.algo.evaluation.simplicity import algorithm as simplicity
    from pm4py.objects.conversion.dfg import converter as dfg_to_net
    from bridge.metrics import entropic_relevance
    from bridge.utils import prune_to_start_end_paths

    traces = [t for t in traces if t]
    out = {"size": len(traces)}
    if not traces:
        return out
    dfg, start, end, removed = prune_to_start_end_paths(dfg, start, end)
    er = entropic_relevance(traces, envelope(dfg, start, end))
    out.update(rel=round(er["rel"], 4), rho=round(er["rho"], 4))
    out["pruned_activities"] = len(removed)
    out["edges"] = len(dfg)
    out["activities"] = len(set(start) | set(end) | {a for edge in dfg for a in edge})
    if not out["activities"]:
        return dict(out, no_score="no model")
    try:
        net, im, fm = dfg_to_net.apply(dfg, parameters={"start_activities": start, "end_activities": end})
    except Exception as e:
        return dict(out, no_score=f"no model: {type(e).__name__}: {str(e)[:80]}")
    log = to_event_log(traces)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit_s))
    try:
        fitness = pm4py.fitness_alignments(log, net, im, fm)["average_trace_fitness"]
        precision = pm4py.precision_alignments(log, net, im, fm)
    except _Limit:
        return dict(out, no_score="no alignment within the limit")
    finally:
        signal.alarm(0)
    out.update(fitness=round(fitness, 4), precision=round(precision, 4),
               f_score=round(2 * fitness * precision / (fitness + precision), 4) if fitness + precision else 0.0,
               simplicity=round(simplicity.apply(net), 4))
    g = under_limit(limit_s, token_generalization, net, im, fm, log)
    out["generalization"] = round(g, 4) if g is not None else None
    if g is None:
        out["no_generalization"] = "no token replay within the limit"
    return out


def published_cluster(dfg_path):
    """(dfg, start, end, published, traces) of one published cluster graph and its stored traces."""
    from bridge.utils import load_dfg_from_json
    dfg, start, end, published = load_dfg_from_json(str(dfg_path))
    with open(Path(dfg_path).with_name(Path(dfg_path).name.replace("_dfg.json", "_traces.json"))) as fh:
        traces = [t for t in json.load(fh)["traces"] if t]
    return dfg, start, end, published, traces


def score_published(dfg_path, limit_s):
    """A published cluster graph against the cluster's traces; also returns the traces for the baselines."""
    dfg, start, end, _published, traces = published_cluster(dfg_path)
    return score_graph(dfg, start, end, traces, limit_s), traces


def score_clear(traces, limit_s):
    """A set of traces scored the way the clustering in the clear publishes it: its filtered graph as the model."""
    dfg, start, end = graph_from_traces(traces)
    return score_graph(dfg, start, end, traces, limit_s)


def random_partition(traces, sizes, seed):
    """The traces shuffled by the seed and cut into parts of the given sizes; every trace lands in exactly one part."""
    if sum(sizes) != len(traces):
        raise ValueError(f"sizes {sizes} do not sum to {len(traces)} traces")
    shuffled = list(traces)
    random.Random(seed).shuffle(shuffled)
    parts, cut = [], 0
    for n in sizes:
        parts.append(shuffled[cut:cut + n])
        cut += n
    return parts


def weighted_mean(rows, key):
    """The mean of the key over the rows that carry it, weighted by their sizes."""
    rows = [c for c in rows if c.get(key) is not None]
    total = sum(c["size"] for c in rows)
    return round(sum(c[key] * c["size"] for c in rows) / total, 4) if total else None


def summarize(clusters):
    """The tier's values: model scores over the scored clusters, graph values over all non-empty ones, each by size."""
    live = [c for c in clusters if c.get("size", 0) > 0]
    scored = [c for c in live if "f_score" in c]
    out = {"clusters": len(live), "cases": sum(c["size"] for c in live), "models_scored": len(scored),
           "pruned_activities": sum(c.get("pruned_activities", 0) for c in live)}
    mean = weighted_mean

    for key in MODEL_KEYS:
        out[key] = mean(scored, key)
    for key in GRAPH_KEYS:
        out[key] = mean(live, key)
    return out


def spread(rows, keys=RANDOM_KEYS):
    """Mean, lowest and highest of each key over the random partitions."""
    out = {"mean": {}, "min": {}, "max": {}}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out["mean"][key] = round(sum(vals) / len(vals), 4) if vals else None
        out["min"][key] = min(vals) if vals else None
        out["max"][key] = max(vals) if vals else None
    return out


def evaluate_folder(tier_dir, limit_s, seeds=RANDOM_SEEDS):
    tier_dir = Path(tier_dir)
    clusters, parts = [], []
    for dfg_path in sorted(tier_dir.glob("cluster_*_dfg.json"), key=cluster_id):
        row, traces = score_published(dfg_path, limit_s)
        row["cluster_id"] = cluster_id(dfg_path)
        clusters.append(row)
        if traces:
            parts.append(traces)
    all_traces = [t for part in parts for t in part]
    sizes = [len(part) for part in parts]
    whole = score_clear(all_traces, limit_s) if all_traces else {"size": 0}
    by_seed = []
    for seed in (seeds if all_traces else ()):
        rows = [score_clear(part, limit_s) for part in random_partition(all_traces, sizes, seed)]
        by_seed.append(dict(summarize(rows), seed=seed))
    return {"instrument": INSTRUMENT, "clusters": clusters, "summary": summarize(clusters), "whole": whole,
            "random": {"seeds": by_seed, **spread(by_seed)}}


def campaign_runs(campaign):
    """{cell: run directory} of the cells whose last run exited 0, later rows of both tables winning."""
    def last(path, keep=lambda r: True):
        rows = {}
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if keep(r):
                    rows[r["cell"]] = r
        return rows
    summary = last(campaign / "summary.tsv")
    manifest = last(campaign / "manifest.tsv", lambda r: r.get("run_dir") and r["run_dir"] != "-")
    return {c: manifest[c]["run_dir"] for c, r in summary.items() if r["exit"] == "0" and c in manifest}


def result_path(campaign, cell, tier):
    return campaign / "model_quality" / f"{cell}__{tier}.json"


def stored_result(path, run_dir):
    """The stored result of this run directory, whatever its tag, or None."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if d.get("run_dir") == os.path.basename(run_dir.rstrip("/")) else None


def is_current(path, run_dir):
    """A result counts while it is this instrument's and its run directory is the one the manifest names."""
    d = stored_result(path, run_dir)
    return bool(d) and d.get("instrument") == INSTRUMENT


def table_rows(campaign, runs):
    rows = []
    for cell, run_dir in sorted(runs.items()):
        for tier in TIERS:
            path = result_path(campaign, cell, tier)
            if not is_current(path, run_dir):
                continue
            with open(path) as fh:
                d = json.load(fh)
            row = {"cell": cell, "tier": tier, "run_dir": d["run_dir"], **d["summary"]}
            whole = d.get("whole") or {}
            row.update({f"whole_{k}": whole.get(k) for k in (*TIER_KEYS, "pruned_activities")})
            rnd = d.get("random") or {}
            for k in RANDOM_KEYS:
                for s in ("mean", "min", "max"):
                    row[f"random_{k}_{s}"] = (rnd.get(s) or {}).get(k)
            rows.append({c: row.get(c) for c in COLUMNS})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="a regex on the cell name")
    ap.add_argument("--limit-s", type=int, default=1800, help="seconds of alignment per model before it counts as unscored")
    ap.add_argument("--folder", type=Path, help="evaluate this one tier folder and print the result (the worker mode)")
    args = ap.parse_args()
    if args.folder:
        print(json.dumps(evaluate_folder(args.folder, args.limit_s)))
        return 0
    if not args.campaign:
        ap.error("--campaign or --folder")
    runs = campaign_runs(args.campaign)
    (args.campaign / "model_quality").mkdir(exist_ok=True)
    jobs = []
    for cell, run_dir in sorted(runs.items()):
        if args.only and not re.search(args.only, cell):
            continue
        base = run_dir if os.path.isabs(run_dir) else str(REPO / run_dir)
        if not os.path.isdir(base):
            base = str(REPO / "output" / "runs" / os.path.basename(run_dir))
        for tier in TIERS:
            path = result_path(args.campaign, cell, tier)
            if glob.glob(os.path.join(base, tier, "cluster_*_dfg.json")) and not is_current(path, run_dir):
                jobs.append((cell, tier, run_dir, os.path.join(base, tier)))

    def run(job):
        cell, tier, run_dir, folder = job
        t0 = time.time()
        cmd = [sys.executable, __file__, "--folder", folder, "--limit-s", str(args.limit_s)]
        p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if p.returncode != 0 or not line.startswith("{"):
            return cell, tier, f"FAILED rc={p.returncode}: {p.stderr.strip()[-300:]}"
        d = json.loads(line)
        d.update(cell=cell, tier=tier, run_dir=os.path.basename(run_dir.rstrip("/")))
        with open(result_path(args.campaign, cell, tier), "w") as fh:
            json.dump(d, fh, indent=1)
        return cell, tier, f"computed, {time.time() - t0:.0f} s"

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for cell, tier, what in pool.map(run, jobs):
            failed += what.startswith("FAILED")
            print(f"{cell} {tier}: {what}", flush=True)
    rows = table_rows(args.campaign, runs)
    with open(args.campaign / "model_quality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    unscored = sum(1 for r in rows if r["models_scored"] != r["clusters"])
    print(f"evaluated {len(jobs)} tier folders, {failed} failed; "
          f"model_quality.csv holds {len(rows)} rows of {len(runs)} cells; "
          f"{unscored} rows with a cluster model that has no score")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
