#!/usr/bin/env python3
"""What other plaintext clustering methods make of the rows Revealed-Projection FedTrac opens: the opened rows are
public, so this simulates the opening per cell in the clear and clusters them by several methods, each scored by the
one instrument the thesis reports (scripts/evaluate_campaign.py, imported here, not reimplemented). The study draws
its own Achlioptas matrix, so its own k-means -- not the run's recorded number -- is the reference.

    scripts/method_study_uniform.py --stages all --workers 6 [--only REGEX]
    scripts/method_study_uniform.py --report            # results.csv and the facts, from what is on disk
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "mp-spdz")):
    if p not in sys.path:
        sys.path.insert(0, p)

CAMPAIGN = REPO / "output" / "campaign_uniform_2026-10-04"
OUT = REPO / "output" / "method_study_uniform_2026-10-09"
# The merged party files of the four splits in the clear (per party {case: [(ms, activity)]} over the pinned case
# list), beside the study's results; --pickle-dir points elsewhere.
PICKLES = OUT / "cases"
# The grid draws and the party-count draws; the split and the case list come from the run's command line in its log.
CELL_RE = re.compile(r"^ms_(sepsis|helpdesk|bpic2013|rtfm)(?:-pc[23]p)?_rp_(activity|bigram|trigram|skipbigram|ef)_s42$")
ENCODING = {"activity": "activity", "bigram": "bigram", "trigram": "trigram",
            "skipbigram": "skip-bigram", "ef": "ef"}
METHODS = ("kmeans", "kmeans-cos", "ward", "gmm-diag", "hdbscan", "consensus", "twin-full-rows")
MIN_CLUSTER_SIZE = 10          # src/constants.rs
PROJECTION_SEED = 42
DENSE_LIMIT = 20_000           # ward and the consensus ensemble hold an n x n matrix of the clustered rows
FAR = 8.0                      # src/pipeline/mpc/reveal_projection.rs::recompute_centroids
SUMMARY_KEYS = ("clusters", "cases", "models_scored", "pruned_activities", "fitness", "precision",
                "f_score", "simplicity", "edges", "activities", "rel", "rho")
COLUMNS = ["cell", "log", "embedding", "k", "k_priv", "opened", "withheld", "cases", "method", "count",
           "precision", "f_score", "fitness", "simplicity", "edges", "activities", "rel", "models_scored",
           "d_f_vs_kmeans", "d_precision_vs_kmeans", "kmeans_f_score", "kmeans_precision",
           "run_f_score", "run_precision", "run_clusters", "ari_to_run", "note"]


# ---------------------------------------------------------------- the runs and their resolved parameters

def recorded_rows():
    """{cell: the federated model_quality.csv row} of the Revealed-Projection draws that ran in full."""
    out = {}
    with open(CAMPAIGN / "model_quality.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["tier"] == "federated" and CELL_RE.match(r["cell"]):
                out[r["cell"]] = r
    return out


def parse_log(cell):
    """The resolved parameters of the run, from the last invocation in its driver log."""
    text = open(CAMPAIGN / f"{cell}.log", errors="replace").read()
    seg = text[text.rfind("Running `"):]

    def last(pattern, cast=str):
        m = re.findall(pattern, seg)
        return cast(m[-1]) if m else None

    cmd = last(r"Running `([^`]*)`")
    anon = re.findall(r"Variant anonymity \(embedding profile\): (\d+) classes / (\d+) cases; "
                      r"singletons (\d+) \([\d.]+%\); min non-singleton class (\d+)", seg)
    reveal = re.findall(r"Hybrid reveal: (\d+)/(\d+) rows in anonymous set", seg)
    return dict(
        cell=cell,
        cmd=cmd,
        split=re.search(r"--data-dir data/splits/(\S+)", cmd)[1],
        caselist=re.search(r"--case-ids config/case_ids/(\S+)", cmd)[1],
        embedding=CELL_RE.match(cell)[2],
        encoding=ENCODING[CELL_RE.match(cell)[2]],
        log_name=CELL_RE.match(cell)[1],
        length_bound=last(r"Dynamic L = (\d+)", int) or last(r"--max-trace-len (\d+)", int),
        # The projected width as the reveal reports it; the [Auto-kproj] line can be a fallback or a clamp.
        k_proj=last(r"shuffled rows, dim=(\d+)", int) or last(r"k_proj (?:\d+ -> )?(\d+)", int)
        or last(r"k_proj = (\d+)", int) or last(r"k_proj=(\d+)", int),
        dict_total=last(r"\((\d+) total with markers\)", int),
        cases=last(r"(\d+) common cases", int),
        k_priv=last(r"k_priv = (\d+) \(selected in-MPC", int),
        k=last(r"\[Auto-k\] k=(\d+)", int),
        classes=int(anon[-1][0]) if anon else None,
        singletons=int(anon[-1][2]) if anon else None,
        min_non_singleton=int(anon[-1][3]) if anon else None,
        opened=int(reveal[-1][0]) if reveal else None,
        skip_projection="Skipping projection" in seg,
        sizes=last(r"Cluster sizes[^\[]*(\[[^\]]*\])"),
    )


# ---------------------------------------------------------------- the clear-text simulation

def merged_cases(params):
    """(case ids, merged event lists, dictionary size, activity names) of the run, in the clear."""
    import test_secure_extract as se
    path = PICKLES / f"cs_{params['split']}__{params['caselist']}.pkl"
    with open(path, "rb") as fh:
        parties, acts = pickle.load(fh)
    aid = {a: 3 + i for i, a in enumerate(acts)}
    name = {v: k for k, v in aid.items()}
    dict_size = 3 + len(acts)
    bound = params["length_bound"]
    ids = sorted(parties[0])
    merged = []
    for case in ids:
        parts = []
        for party in parties:
            events = sorted(((aid[a], ms // 1000) for ms, a in party.get(case, [])),
                            key=lambda e: (e[1], e[0]))[:bound]
            parts.append(events)
        merged.append(se.merge_events_plaintext(parts, dict_size))
    return ids, merged, dict_size, name


def extractor(encoding):
    import test_secure_extract as se
    import test_secure_pipeline_hybrid as hy
    return {"bigram": se.extract_bigrams_plaintext, "ef": se.extract_ef_plaintext,
            "activity": se.extract_activity_plaintext, "trigram": hy.extract_trigram_plaintext,
            "skip-bigram": hy.extract_skip_bigram_plaintext}[encoding]


def feature_dim(encoding, dict_size):
    d = dict_size
    return {"activity": d, "bigram": d * d, "ef": d * d, "trigram": (d - 3) ** 3,
            "skip-bigram": 2 * d * d}[encoding]


def project_and_fingerprint(merged, encoding, dict_size, k_proj, chunk=2000):
    """The projected, unit-normalised rows and, per case, its raw row's fingerprint and whether it is zero; a
    chunk is extracted, fingerprinted and projected, then dropped, so the raw rows are never all held."""
    import numpy as np
    import test_secure_extract as se
    extract = extractor(encoding)
    dim = feature_dim(encoding, dict_size)
    matrix = np.asarray(se.generate_achlioptas_matrix(k_proj, dim, seed=PROJECTION_SEED), dtype=np.int64)
    prints, zero, blocks = [], [], []
    for i in range(0, len(merged), chunk):
        raw = np.asarray([extract(m, dict_size) for m in merged[i:i + chunk]], dtype=np.int64)
        if raw.shape[1] != dim:
            raise SystemExit(f"{encoding}: feature width {raw.shape[1]} is not the expected {dim}")
        for row in raw:
            prints.append(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
        zero.extend(bool(v) for v in (raw == 0).all(axis=1))
        blocks.append((raw @ matrix.T).astype(np.float64))
    projected = np.vstack(blocks)
    norm = np.linalg.norm(projected, axis=1, keepdims=True)
    unit = np.divide(projected, norm, out=np.zeros_like(projected), where=norm > 0)
    return unit, prints, np.asarray(zero, dtype=bool)


def anonymity(prints, zero, k_priv):
    """The class histogram of the raw rows and the anonymous set at k_priv (a zero row is never in it)."""
    import numpy as np
    counts = Counter(prints)
    sizes = list(counts.values())
    backbone = np.asarray([counts[p] >= k_priv and not z for p, z in zip(prints, zero)], dtype=bool)
    return dict(classes=len(counts), singletons=sum(1 for s in sizes if s == 1),
                min_non_singleton=min([s for s in sizes if s >= 2], default=0),
                opened=int(backbone.sum())), backbone


def merge_small(labels, centroids, min_size=MIN_CLUSTER_SIZE):
    """src/pipeline/clustering.rs::merge_small_clusters: every cluster below min_size joins the nearest large one."""
    import numpy as np
    labels = labels.copy()
    k = len(centroids)
    sizes = np.bincount(labels, minlength=k)
    small = [c for c in range(k) if 0 < sizes[c] < min_size]
    large = [c for c in range(k) if sizes[c] >= min_size]
    if not small or not large:
        return labels, 0
    for c in small:
        dist = [float(((centroids[c] - centroids[o]) ** 2).sum()) for o in large]
        nearest = large[int(np.argmin(dist))]
        labels[labels == c] = nearest
        sizes[nearest] += sizes[c]
        sizes[c] = 0
    return labels, len(small)


def centroids_of(rows, labels, k, dim):
    """Post-merge centroids over the clustered rows; an emptied cluster gets the far sentinel, as the MPC does."""
    import numpy as np
    out = np.full((k, dim), FAR, dtype=np.float64)
    for c in range(k):
        members = rows[labels == c]
        if len(members):
            out[c] = members.mean(axis=0)
    return out


def assign_tail(rows, centroids):
    """secure_relabel's tail argmin: squared Euclidean to the nearest centroid, smallest index on a tie."""
    import numpy as np
    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    d2 = ((rows[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) if len(rows) * len(centroids) < 4e7 else None
    if d2 is None:
        out = np.empty(len(rows), dtype=np.int64)
        step = max(1, int(4e7 // max(1, len(centroids))))
        for i in range(0, len(rows), step):
            block = rows[i:i + step]
            out[i:i + step] = np.argmin(((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        return out
    return np.argmin(d2, axis=1)


def cluster_opened(method, rows, k, seed=0):
    """(labels over the clustered rows, the centroids the merge reads, a note). The repository's own method calls."""
    import numpy as np
    from sklearn.cluster import KMeans
    from bridge.cluster import cluster_means, compact_labels, fit, rehome_noise
    if method in ("kmeans", "twin-full-rows"):
        # The harness's own call: 10 restarts, 10 iterations (src/phases/p3_clustering.rs::cluster_n_init,
        # linfa; sklearn here, so the study's k-means is its own reference, not a bit-for-bit twin).
        km = KMeans(n_clusters=k, n_init=10, max_iter=10, random_state=seed).fit(rows)
        return km.labels_.astype(np.int64), km.cluster_centers_, ""
    algo = {"kmeans-cos": "kmeans-cos", "ward": "ward", "gmm-diag": "gmm-diag",
            "hdbscan": "hdbscan", "consensus": "consensus"}[method]
    raw, _ = fit(algo, rows, k, seed)
    rehomed, noise = rehome_noise(rows, raw)
    labels, k_actual = compact_labels(rehomed)
    note = f"noise rows rehomed {noise}; own count {k_actual}" if algo == "hdbscan" else ""
    return labels, cluster_means(rows, labels, k_actual), note


# ---------------------------------------------------------------- the stages

def cache_paths(cell):
    return (OUT / "cache" / f"{cell}.json", OUT / "cache" / f"{cell}_y.npy",
            OUT / "cache" / f"{cell}_backbone.npy", OUT / "cache" / f"{cell}_traces.json.gz")


def published_graph_check(params, ids, traces, run_dir):
    """The study's graphs under the run's OWN labels against its published ones: cells compared and cells differing."""
    from bridge.utils import load_dfg_from_json
    from scripts.evaluate_campaign import graph_from_traces
    base = REPO / (run_dir if run_dir.startswith("output") else f"output/runs/{run_dir}")
    folder = base / "federated"
    stored = json.load(open(folder / "cluster_assignments.json"))
    label = dict(zip(stored["case_ids"], stored["assignments"]))
    if set(label) != set(ids):
        return dict(compared=0, differ=-1, note=f"case sets differ by {len(set(label) ^ set(ids))}")
    groups = defaultdict(list)
    for case, trace in zip(ids, traces):
        groups[label[case]].append(trace)
    compared = differ = 0
    detail = []
    for j, group in sorted(groups.items()):
        dfg, start, end = graph_from_traces(group)
        path = folder / f"cluster_{j}_dfg.json"
        if not path.exists():
            detail.append(f"cluster {j} has {len(group)} cases but no published graph")
            differ += 1
            continue
        pub, pstart, pend, _ = load_dfg_from_json(str(path))
        pub = {k: v for k, v in pub.items() if v > 0}
        pstart = {k: v for k, v in pstart.items() if v > 0}
        pend = {k: v for k, v in pend.items() if v > 0}
        for mine, theirs, what in ((dfg, pub, "pair"), (start, pstart, "start"), (end, pend, "end")):
            keys = set(mine) | set(theirs)
            compared += len(keys)
            bad = [k for k in keys if abs(mine.get(k, 0.0) - theirs.get(k, 0.0)) > 1e-9]
            differ += len(bad)
            if bad:
                detail.append(f"c{j} {what} {[(k, mine.get(k, 0.0), theirs.get(k, 0.0)) for k in bad[:2]]}")
    return dict(compared=compared, differ=differ, note="; ".join(detail[:4]))


def stage_features(cell, recorded):
    import numpy as np
    meta_path, y_path, bb_path, tr_path = cache_paths(cell)
    params = parse_log(cell)
    params["run_dir"] = recorded[cell]["run_dir"]
    ids, merged, dict_size, name = merged_cases(params)
    traces = [[name[act] for _, act in m] for m in merged]
    if len(ids) != params["cases"]:
        raise SystemExit(f"{cell}: {len(ids)} cases in the clear, the log says {params['cases']}")
    unit, prints, zero = project_and_fingerprint(merged, params["encoding"], dict_size, params["k_proj"])
    measured, backbone = anonymity(prints, zero, params["k_priv"])
    checks = dict(
        anonymity=dict(
            classes=[measured["classes"], params["classes"]],
            singletons=[measured["singletons"], params["singletons"]],
            min_non_singleton=[measured["min_non_singleton"], params["min_non_singleton"]],
            opened=[measured["opened"], params["opened"]],
            passed=(measured["classes"] == params["classes"]
                    and measured["singletons"] == params["singletons"]
                    and measured["min_non_singleton"] == params["min_non_singleton"]
                    and measured["opened"] == params["opened"]),
        ),
        graphs=published_graph_check(params, ids, traces, params["run_dir"]),
    )
    checks["graphs"]["passed"] = checks["graphs"]["differ"] == 0 and checks["graphs"]["compared"] > 0
    np.save(y_path, unit.astype(np.float64))
    np.save(bb_path, backbone)
    with gzip.open(tr_path, "wt") as fh:
        json.dump({"case_ids": ids, "traces": traces}, fh)
    meta = dict(params=params, dict_size=dict_size, dim=feature_dim(params["encoding"], dict_size),
                measured=measured, checks=checks, zero_rows=int(zero.sum()))
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return (f"{cell}: anonymity {'ok' if checks['anonymity']['passed'] else 'FAILED'} "
            f"({measured['classes']} classes, {measured['singletons']} singletons, {measured['opened']} opened); "
            f"graphs {checks['graphs']['compared']} cells, {checks['graphs']['differ']} differ")


def stage_labels(cell, method):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    meta_path, y_path, bb_path, _ = cache_paths(cell)
    meta = json.load(open(meta_path))
    params = meta["params"]
    unit = np.load(y_path)
    backbone = np.load(bb_path)
    k = params["k"]
    notes = []
    rows = unit if method == "twin-full-rows" else unit[backbone]
    skipped = None
    if method in ("ward", "consensus") and len(rows) > DENSE_LIMIT:
        skipped = (f"{method} holds an {len(rows)}x{len(rows)} matrix of the clustered rows "
                   f"({len(rows) ** 2 * 8 / 2 ** 30:.0f} GiB), past the study's limit of {DENSE_LIMIT} rows")
    if method == "hdbscan":
        # bridge/cluster.py::fit sets min_cluster_size = max(5, n // max(2k, 10)), from which sklearn's HDBSCAN
        # takes min_samples, so the core distances are an n x min_cluster_size array and can outgrow memory.
        min_size = max(5, len(rows) // max(2 * k, 10) if k > 0 else len(rows) // 20)
        need = len(rows) * min_size * 8 / 2 ** 30
        if need > 2:
            skipped = (f"hdbscan at the harness's min_cluster_size {min_size} needs an "
                       f"{len(rows)} x {min_size} neighbour-distance array ({need:.1f} GiB), past the "
                       f"2 GiB the study allows beside the attack battery")
    if skipped:
        out = dict(cell=cell, method=method, skipped=skipped)
        json.dump(out, open(OUT / "labels" / f"{cell}__{method}.json", "w"))
        return f"{cell} {method}: skipped ({skipped})"
    t0 = time.time()
    labels, centroids, note = cluster_opened(method, rows, k)
    if note:
        notes.append(note)
    labels, merged_away = merge_small(labels, centroids)
    if merged_away:
        notes.append(f"{merged_away} clusters below {MIN_CLUSTER_SIZE} merged")
    if method == "twin-full-rows":
        full = labels
    else:
        full = np.zeros(len(unit), dtype=np.int64)
        full[backbone] = labels
        tail = ~backbone
        if tail.any():
            cents = centroids_of(rows, labels, len(centroids), unit.shape[1])
            full[tail] = assign_tail(unit[tail], cents)
    sizes = Counter(int(v) for v in full)
    stored = json.load(open(REPO / (params["run_dir"] if params["run_dir"].startswith("output")
                                    else f"output/runs/{params['run_dir']}") / "federated"
                            / "cluster_assignments.json"))
    ids = json.load(gzip.open(cache_paths(cell)[3], "rt"))["case_ids"]
    run_label = dict(zip(stored["case_ids"], stored["assignments"]))
    ari = float(adjusted_rand_score([run_label[c] for c in ids], full.tolist()))
    out = dict(cell=cell, method=method, k_requested=k, count=len(sizes), sizes=sorted(sizes.values(), reverse=True),
               opened=int(backbone.sum()), withheld=int((~backbone).sum()), ari_to_run=round(ari, 6),
               seconds=round(time.time() - t0, 1), note="; ".join(notes), labels=[int(v) for v in full])
    json.dump(out, open(OUT / "labels" / f"{cell}__{method}.json", "w"))
    return f"{cell} {method}: count {out['count']} sizes {out['sizes'][:6]} {out['seconds']} s {out['note']}"


def stage_score(cell, method, limit_s):
    from scripts.evaluate_campaign import envelope, graph_from_traces, score_graph, summarize
    labels_path = OUT / "labels" / f"{cell}__{method}.json"
    info = json.load(open(labels_path))
    result_path = OUT / "results" / f"{cell}__{method}.json"
    if info.get("skipped"):
        json.dump(dict(info, clusters=[]), open(result_path, "w"))
        return f"{cell} {method}: skipped"
    traces = json.load(gzip.open(cache_paths(cell)[3], "rt"))["traces"]
    groups = defaultdict(list)
    for trace, label in zip(traces, info["labels"]):
        groups[label].append(trace)
    t0 = time.time()
    rows = []
    for label in sorted(groups):
        group = [t for t in groups[label] if t]
        dfg, start, end = graph_from_traces(group)
        row = score_graph(dfg, start, end, envelope(dfg, start, end), group, limit_s)
        row["cluster_id"] = int(label)
        rows.append(row)
    out = {k: v for k, v in info.items() if k != "labels"}
    out.update(summary=summarize(rows), clusters=rows, score_seconds=round(time.time() - t0, 1))
    json.dump(out, open(result_path, "w"), indent=1)
    s = out["summary"]
    return (f"{cell} {method}: F {s.get('f_score')} precision {s.get('precision')} "
            f"edges {s.get('edges')} rel {s.get('rel')} in {out['score_seconds']} s")


# ---------------------------------------------------------------- the table and the facts

def build_table(cells, recorded):
    rows = []
    for cell in cells:
        meta_path = cache_paths(cell)[0]
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        params, run = meta["params"], recorded[cell]
        ref = None
        ref_path = OUT / "results" / f"{cell}__kmeans.json"
        if ref_path.exists():
            ref = json.load(open(ref_path)).get("summary", {})
        for method in METHODS:
            path = OUT / "results" / f"{cell}__{method}.json"
            if not path.exists():
                continue
            res = json.load(open(path))
            s = res.get("summary", {})
            row = dict(cell=cell, log=params["log_name"], embedding=params["embedding"], k=params["k"],
                       k_priv=params["k_priv"], opened=meta["measured"]["opened"],
                       withheld=params["cases"] - meta["measured"]["opened"], cases=params["cases"],
                       method=method, count=res.get("count"), models_scored=s.get("models_scored"),
                       ari_to_run=res.get("ari_to_run"), note=res.get("skipped") or res.get("note") or "")
            for key in ("precision", "f_score", "fitness", "simplicity", "edges", "activities", "rel"):
                row[key] = s.get(key)
            row["kmeans_f_score"] = (ref or {}).get("f_score")
            row["kmeans_precision"] = (ref or {}).get("precision")
            for a, b, name in (("f_score", "kmeans_f_score", "d_f_vs_kmeans"),
                               ("precision", "kmeans_precision", "d_precision_vs_kmeans")):
                row[name] = (round(row[a] - row[b], 4) if row.get(a) is not None and row.get(b) is not None else None)
            row["run_f_score"] = float(run["f_score"]) if run["f_score"] else None
            row["run_precision"] = float(run["precision"]) if run["precision"] else None
            row["run_clusters"] = int(run["clusters"])
            rows.append({c: row.get(c) for c in COLUMNS})
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def facts(rows, cells):
    """The facts the brief asks for, printed; the caller pastes them into the README."""
    out = []
    by_log = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["f_score"] is not None:
            by_log[r["log"]][r["method"]].append(r)
    out.append("Mean F and precision over the draws, per event log and method (d = to the study's k-means):")
    out.append(f"{'log':<10}{'method':<17}{'draws':>6}{'F':>9}{'precision':>11}{'dF':>9}{'dprec':>9}")
    for log in ("sepsis", "helpdesk", "bpic2013", "rtfm"):
        for method in METHODS:
            got = by_log[log].get(method) or []
            if not got:
                continue
            mean = lambda key: sum(r[key] for r in got if r[key] is not None) / max(
                1, sum(1 for r in got if r[key] is not None))
            dvals = [r["d_f_vs_kmeans"] for r in got if r["d_f_vs_kmeans"] is not None]
            pvals = [r["d_precision_vs_kmeans"] for r in got if r["d_precision_vs_kmeans"] is not None]
            out.append(f"{log:<10}{method:<17}{len(got):>6}{mean('f_score'):>9.4f}{mean('precision'):>11.4f}"
                       f"{(sum(dvals) / len(dvals) if dvals else 0):>9.4f}"
                       f"{(sum(pvals) / len(pvals) if pvals else 0):>9.4f}")
    widest = None
    for cell in cells:
        got = [r for r in rows if r["cell"] == cell and r["f_score"] is not None and r["method"] != "twin-full-rows"]
        if len(got) < 2:
            continue
        lo = min(got, key=lambda r: r["f_score"])
        hi = max(got, key=lambda r: r["f_score"])
        gap = hi["f_score"] - lo["f_score"]
        if widest is None or gap > widest[0]:
            widest = (gap, cell, lo, hi)
    if widest:
        gap, cell, lo, hi = widest
        out.append(f"\nWidest gap between two methods on one run: {gap:.4f} on {cell} "
                   f"({hi['method']} {hi['f_score']} against {lo['method']} {lo['f_score']})")
    past = [r for r in rows if r["d_f_vs_kmeans"] is not None and abs(r["d_f_vs_kmeans"]) > 0.05
            and r["method"] not in ("kmeans",)]
    out.append(f"\nMethods past 0.05 in F from the study's k-means: {len(past)} of "
               f"{sum(1 for r in rows if r['d_f_vs_kmeans'] is not None and r['method'] != 'kmeans')} scored rows")
    for r in sorted(past, key=lambda r: -abs(r["d_f_vs_kmeans"])):
        out.append(f"  {r['cell']:<32}{r['method']:<16}F {r['f_score']} against {r['kmeans_f_score']} "
                   f"({r['d_f_vs_kmeans']:+.4f}), counts {r['count']} / {r['k']}")
    out.append("\nHDBSCAN's chosen counts:")
    for r in rows:
        if r["method"] == "hdbscan":
            out.append(f"  {r['cell']:<32}count {r['count']} (the run's k {r['k']}) {r['note']}")
    out.append("\nThe reference check, the study's k-means against the run's recorded federated f_score:")
    bad = 0
    for r in rows:
        if r["method"] != "kmeans" or r["f_score"] is None or r["run_f_score"] is None:
            continue
        d = r["f_score"] - r["run_f_score"]
        bad += abs(d) > 0.05
        out.append(f"  {r['cell']:<32}study {r['f_score']:.4f} run {r['run_f_score']:.4f} ({d:+.4f})"
                   f"{'  PAST 0.05' if abs(d) > 0.05 else ''}")
    out.append(f"  {bad} of {sum(1 for r in rows if r['method'] == 'kmeans' and r['f_score'] is not None)} "
               f"draws past 0.05")
    out.append("\nThe checks of the simulation, per cell:")
    for cell in cells:
        path = cache_paths(cell)[0]
        if not path.exists():
            continue
        c = json.load(open(path))["checks"]
        out.append(f"  {cell:<32}anonymity {'ok' if c['anonymity']['passed'] else 'FAILED ' + json.dumps(c['anonymity'])}"
                   f"; graphs {c['graphs']['compared']} cells {c['graphs']['differ']} differ {c['graphs']['note']}")
    return "\n".join(out)


# ---------------------------------------------------------------- the driver

def worker(args_list, workers):
    def run(job):
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, *job], cwd=str(REPO), capture_output=True, text=True)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if p.returncode != 0:
            return f"FAILED {' '.join(job)} rc={p.returncode}: {p.stderr.strip()[-400:]}"
        return f"{line}  [{time.time() - t0:.0f} s]"

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for what in pool.map(run, args_list):
            failed += what.startswith("FAILED")
            print(what, flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=None, help="all | features | labels | score")
    ap.add_argument("--only", default=None, help="a regex on the cell name")
    ap.add_argument("--methods", default=None, help="a comma-separated subset of the methods")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-s", type=int, default=1800, help="alignment seconds per model, as the instrument's default")
    ap.add_argument("--report", action="store_true", help="write results.csv and print the facts from what is on disk")
    ap.add_argument("--stage", default=None, help="the worker mode: features | labels | score")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--pickle-dir", type=Path, default=None, help="the merged party files; the default is cases/ beside the results")
    args = ap.parse_args()
    if args.pickle_dir:
        globals()["PICKLES"] = args.pickle_dir
    for sub in ("cache", "labels", "results"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    recorded = recorded_rows()
    cells = sorted(recorded)
    if args.only:
        cells = [c for c in cells if re.search(args.only, c)]
    methods = args.methods.split(",") if args.methods else list(METHODS)

    if args.stage:
        if args.stage == "features":
            print(stage_features(args.cell, recorded))
        elif args.stage == "labels":
            print(stage_labels(args.cell, args.method))
        elif args.stage == "score":
            print(stage_score(args.cell, args.method, args.limit_s))
        else:
            ap.error("--stage features|labels|score")
        return 0

    extra = ["--pickle-dir", str(args.pickle_dir)] if args.pickle_dir else []
    failed = 0
    if args.stages in ("all", "features"):
        failed += worker([["--stage", "features", "--cell", c, *extra] for c in cells], min(args.workers, 3))
    if args.stages in ("all", "labels"):
        failed += worker([["--stage", "labels", "--cell", c, "--method", m, *extra] for c in cells for m in methods],
                         args.workers)
    if args.stages in ("all", "score"):
        failed += worker([["--stage", "score", "--cell", c, "--method", m, "--limit-s", str(args.limit_s), *extra]
                          for c in cells for m in methods], args.workers)
    if args.stages or args.report:
        rows = build_table(sorted(recorded), recorded)
        print(f"\nresults.csv holds {len(rows)} rows over {len({r['cell'] for r in rows})} cells\n")
        print(facts(rows, sorted(recorded)))
    if failed:
        print(f"\n{failed} jobs FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
