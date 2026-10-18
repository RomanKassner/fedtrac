#!/usr/bin/env python3
"""Clustering-quality and conformance metrics of the in-run evaluation."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import entropy as _scipy_entropy


def _warn(tag: str, path: Path, err: BaseException) -> None:
    print(f"[{tag}] {path}: {type(err).__name__}: {err}", file=sys.stderr)


# Expected pm4py / I/O / JSON errors; everything else propagates.
_METRIC_EXPECTED_ERRORS = (
    FileNotFoundError,
    OSError,
    ImportError,
    ValueError,
    KeyError,
    json.JSONDecodeError,
)


def shannon_entropy(freq: dict) -> float:
    """Shannon entropy in bits from a {label: count} dict; 0 for empty input."""
    if not freq:
        return 0.0
    return float(_scipy_entropy(list(freq.values()), base=2))


def edge_counts_from_traces(traces: list[list[str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for trace in traces:
        for i in range(len(trace) - 1):
            edge = (trace[i], trace[i + 1])
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def variant_counts_from_traces(traces: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trace in traces:
        key = "->".join(trace)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compute_entropy_reduction(cluster_traces: list[list[list[str]]]) -> dict:
    """Edge and variant entropy of the whole log against the weighted entropy of the clusters."""
    all_traces = [t for ct in cluster_traces for t in ct]
    n_total = len(all_traces)

    global_edges = edge_counts_from_traces(all_traces)
    h_edge_global = shannon_entropy(global_edges)

    h_edge_clustered = 0.0
    per_cluster_edge = []
    for ct in cluster_traces:
        if not ct:
            per_cluster_edge.append(0.0)
            continue
        edges = edge_counts_from_traces(ct)
        h_c = shannon_entropy(edges)
        per_cluster_edge.append(h_c)
        h_edge_clustered += (len(ct) / n_total) * h_c

    edge_reduction = 1.0 - h_edge_clustered / h_edge_global if h_edge_global > 0 else 0.0

    global_variants = variant_counts_from_traces(all_traces)
    h_var_global = shannon_entropy(global_variants)

    h_var_clustered = 0.0
    per_cluster_var = []
    for ct in cluster_traces:
        if not ct:
            per_cluster_var.append(0.0)
            continue
        variants = variant_counts_from_traces(ct)
        h_c = shannon_entropy(variants)
        per_cluster_var.append(h_c)
        h_var_clustered += (len(ct) / n_total) * h_c

    var_reduction = 1.0 - h_var_clustered / h_var_global if h_var_global > 0 else 0.0

    return {
        "edge_H_global": round(h_edge_global, 4),
        "edge_H_clustered": round(h_edge_clustered, 4),
        "edge_reduction": round(edge_reduction, 4),
        "edge_per_cluster": [round(h, 4) for h in per_cluster_edge],
        "variant_H_global": round(h_var_global, 4),
        "variant_H_clustered": round(h_var_clustered, 4),
        "variant_reduction": round(var_reduction, 4),
        "variant_per_cluster": [round(h, 4) for h in per_cluster_var],
        "n_global_variants": len(global_variants),
    }


def structural_complexity(pnml_path: Path) -> dict | None:
    """|places|, |transitions|, |arcs|, CFC = |arcs| - (|p|+|t|) + 2 (Buijs 2014)."""
    try:
        from pm4py.objects.petri_net.importer import importer as pnml_importer
        net, _, _ = pnml_importer.apply(str(pnml_path))
        n_p = len(net.places)
        n_t = len(net.transitions)
        n_a = len(net.arcs)
        cfc = n_a - (n_p + n_t) + 2
        return {"n_places": n_p, "n_transitions": n_t, "n_arcs": n_a, "cfc": cfc}
    except _METRIC_EXPECTED_ERRORS as e:
        _warn("StructuralComplexity", pnml_path, e)
        return None


# The artificial start and end cells of a published graph (src/types.rs), plus the
# placeholders of older exports.
DFG_START, DFG_END = ">", "|"
MARKER_ACTIVITIES = {DFG_START, DFG_END, "?", ""}

# Laplace pseudo-count alpha of the box objective (mp-spdz/secure_sequence_cluster.mpc
# log_weights, clear twin scripts/plaintext_sequence_reference.py).
SEQ_LAPLACE_ALPHA = 1.0


def transition_model_from_dfg(
    dfg: dict,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """(edge weight, per-source outgoing total) of a published graph, the START row and every
    a -> END edge included."""
    weights = {(e["from"], e["to"]): float(e["weight"]) for e in dfg.get("edges", [])}
    totals: dict[str, float] = {}
    for (src, _), w in weights.items():
        totals[src] = totals.get(src, 0.0) + w
    return weights, totals


def transition_log_likelihood(
    cluster_traces: list[list[list[str]]],
    activities: set[str] | None = None,
) -> float:
    """Mean Laplace-smoothed log2 transition probability, the objective of
    mp-spdz/secure_sequence_cluster.mpc scored in the open over each cluster's
    unfiltered counts C_j: w_j(a,b) = log2(C_j[a,b] + 1) - log2(sum_b' C_j[a,b'] + d_eff),
    d_eff = |activities| + 1, START->first and last->END in, no unseen floor,
    divided by sum_i (|sigma_i| + 1)."""
    if activities is None:
        activities = {a for cluster in cluster_traces for tr in cluster for a in tr}
    d_eff = len(activities) + 1
    total, n_trans = 0.0, 0
    for cluster in cluster_traces:
        counts: Counter = Counter()
        for trace in cluster:
            if not trace:
                continue
            counts[(DFG_START, trace[0])] += 1
            counts.update(zip(trace, trace[1:]))
            counts[(trace[-1], DFG_END)] += 1
        row_sums: dict[str, float] = {}
        for (src, _), c in counts.items():
            row_sums[src] = row_sums.get(src, 0.0) + c
        for (src, tgt), c in counts.items():
            w = (math.log2(counts[(src, tgt)] + SEQ_LAPLACE_ALPHA)
                 - math.log2(row_sums[src] + d_eff))
            total += c * w
            n_trans += c
    return total / n_trans if n_trans else float("nan")


def entropic_relevance_bits(traces: list[list[str]], dfg: dict,
                            unseen_penalty_bits: float = 20.0) -> float | None:
    """Older relevance of the in-run diagnostics: the entropy of the variants plus the
    mean code length, an unseen step costing a flat `unseen_penalty_bits`. Not the
    measure the thesis reports."""
    scored = [t for t in traces if t]
    if not scored:
        return None
    weights, row_sums = transition_model_from_dfg(dfg)
    total_bits = 0.0
    for trace in scored:
        steps = [(DFG_START, trace[0])]
        steps += list(zip(trace, trace[1:]))
        steps.append((trace[-1], DFG_END))
        for a, b in steps:
            row_total = row_sums.get(a, 0.0)
            weight = weights.get((a, b), 0.0)
            if row_total <= 0 or weight <= 0:
                total_bits += unseen_penalty_bits
            else:
                total_bits += -math.log2(weight / row_total)
    variant_H = shannon_entropy(variant_counts_from_traces(scored))
    return variant_H + total_bits / len(scored)


def entropic_relevance(traces: list[list[str]], dfg: dict) -> dict | None:
    """Entropic relevance of a graph to its traces in bits per trace, the measure the thesis reports
    (Polyvyanyy, Moffat, Garcia-Banuelos 2020, Definitions IV.1 and IV.2).

    The graph with its counts is a stochastic automaton: from an activity the next step has the
    probability of its edge count over the sum of the outgoing counts, START->first and last->END
    included, and an allowed trace costs -log2 of its probability. Any other trace costs
    (1 + |t|) log2(1 + |A|) bits, A the activities of the scored traces. rel = H0(rho) + the mean
    cost, rho the share of allowed traces. Lower is better.
    """
    scored = [t for t in traces if t]
    if not scored:
        return None
    weights, row_sums = transition_model_from_dfg(dfg)
    background = math.log2(1 + len({a for t in scored for a in t}))
    total_bits, allowed = 0.0, 0
    for trace in scored:
        steps = [(DFG_START, trace[0]), *zip(trace, trace[1:]), (trace[-1], DFG_END)]
        bits = 0.0
        for a, b in steps:
            row_total, weight = row_sums.get(a, 0.0), weights.get((a, b), 0.0)
            if row_total <= 0 or weight <= 0:
                bits = None
                break
            bits += -math.log2(weight / row_total)
        if bits is None:
            total_bits += (1 + len(trace)) * background
        else:
            total_bits += bits
            allowed += 1
    rho = allowed / len(scored)
    selector = 0.0 if rho in (0.0, 1.0) else -rho * math.log2(rho) - (1 - rho) * math.log2(1 - rho)
    return {"rel": selector + total_bits / len(scored), "rho": rho, "traces": len(scored)}


def compute_entropic_relevance(dfg_path: Path, traces_path: Path,
                                unseen_penalty_bits: float = 20.0) -> float | None:
    """`entropic_relevance_bits` of an exported cluster (its graph JSON and its traces JSON)."""
    try:
        with open(dfg_path) as f:
            dfg_data = json.load(f)
        with open(traces_path) as f:
            traces = json.load(f)["traces"]
        er = entropic_relevance_bits(traces, dfg_data, unseen_penalty_bits)
        return None if er is None else round(er, 4)
    except _METRIC_EXPECTED_ERRORS as e:
        _warn("EntropicRelevance", dfg_path, e)
        return None


# Classification schemes aligned with what bigram features can capture.
VARIANT_SCHEMES = {
    "length": lambda trace: (
        "short" if len(trace) < 15
        else "long" if len(trace) >= 40
        else "mid"
    ),
    "activity_set": lambda trace: (
        "has_A" if "Release A" in trace
        else "has_B" if "Release B" in trace
        else "neither"
    ),
    "loop": lambda trace: (
        "has_loops" if len(trace) != len(set(trace))
        else "no_loops"
    ),
}


def _compute_homogeneity(labels_true: list, labels_pred: list) -> float:
    """Homogeneity score: 1 - H(C|K)/H(C)."""
    if not labels_true:
        return 0.0
    n = len(labels_true)
    h_c = shannon_entropy(dict(Counter(labels_true)))
    if h_c < 1e-9:
        return 1.0
    h_c_given_k = 0.0
    for k_val in set(labels_pred):
        members = [labels_true[i] for i in range(n) if labels_pred[i] == k_val]
        if not members:
            continue
        h_c_given_k += (len(members) / n) * shannon_entropy(dict(Counter(members)))
    return 1.0 - h_c_given_k / h_c


def compute_variant_purity(cluster_traces: list[list[list[str]]]) -> dict:
    """Variant purity under each of the VARIANT_SCHEMES classification schemes."""
    results = {}
    for scheme_name, classify_fn in VARIANT_SCHEMES.items():
        labels_true = []
        labels_pred = []
        for c_id, traces in enumerate(cluster_traces):
            for trace in traces:
                labels_true.append(classify_fn(trace))
                labels_pred.append(c_id)

        if not labels_true:
            results[scheme_name] = {"homogeneity": 0.0, "per_cluster": []}
            continue

        homogeneity = _compute_homogeneity(labels_true, labels_pred)
        n = len(labels_true)

        per_cluster = []
        for c_id in range(len(cluster_traces)):
            members = [labels_true[i] for i in range(n) if labels_pred[i] == c_id]
            dist = dict(Counter(members))
            n_c = len(members)
            dominant_frac = max(dist.values()) / n_c if n_c > 0 else 0
            per_cluster.append({
                "distribution": dist,
                "dominant_fraction": round(dominant_frac, 4),
                "size": n_c,
            })

        results[scheme_name] = {
            "homogeneity": round(homogeneity, 4),
            "global_distribution": dict(Counter(labels_true)),
            "per_cluster": per_cluster,
        }

    return results
