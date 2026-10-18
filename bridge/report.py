#!/usr/bin/env python3
"""Weighted aggregates, quality metrics and result tables of the in-run evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import (
    compute_entropic_relevance,
    compute_entropy_reduction,
    compute_variant_purity,
)
from .utils import compute_weighted_avg

# Below this edge-entropy reduction the clustering is flagged data-bound.
DATA_BOUND_EDGE_RED_THRESHOLD = 0.15


def _gini(sizes: list[int]) -> float:
    """Gini coefficient over the cluster sizes; 0 = uniform, 1 = one dominant cluster."""
    n = len(sizes)
    if n == 0 or sum(sizes) == 0:
        return 0.0
    s = sorted(sizes)
    cum = sum((2 * i - n + 1) * v for i, v in enumerate(s))
    return cum / (n * sum(s))


def print_headline_diagnostics(
    cluster_info: list,
    hm_results: list[dict],
    weighted_avg_hm: dict,
    entropy: dict,
    er_weighted_avg: float | None = None,
) -> dict:
    """Soundness rate, sound-only F, Buijs, relevance, imbalance and the data-bound flag,
    as a JSON-safe dict for evaluation_results.json."""
    sizes = [c.get("size", 0) for c in hm_results] if hm_results else []
    sound_flags = [c.get("sound") for c in hm_results] if hm_results else []
    n_sound = sum(1 for s in sound_flags if s is True)
    n_total = len(sound_flags)
    sound_only = [c for c in hm_results if c.get("sound") is True]
    wavg_sound = compute_weighted_avg(sound_only) or {}

    max_s = max(sizes) if sizes else 0
    min_s = min((s for s in sizes if s > 0), default=0)
    imbalance = (max_s / min_s) if min_s else float("inf") if max_s else 0.0
    gini = _gini(sizes)

    edge_red = entropy.get("edge_reduction", 0.0)
    data_bound = edge_red < DATA_BOUND_EDGE_RED_THRESHOLD

    print("\n=== Headline Diagnostics ===")
    f_all = (weighted_avg_hm or {}).get("f_score", 0.0)
    buijs_all = (weighted_avg_hm or {}).get("buijs", 0.0)
    print(f"  HM F-Score (all):              {f_all:.4f}")
    if n_total:
        if sound_only:
            print(f"  HM F-Score (sound only):       {wavg_sound.get('f_score', 0.0):.4f}  "
                  f"({n_sound}/{n_total} sound)")
        else:
            print(f"  HM F-Score (sound only):       n/a   ({n_sound}/{n_total} sound)")
    print(f"  HM Buijs (10·fit+10·prec+gen+simpl)/22: {buijs_all:.4f}")
    if er_weighted_avg is not None:
        print(f"  Entropic relevance (HM, lower=better): {er_weighted_avg:.2f} bits/trace")
    print(f"  Edge entropy reduction:        {edge_red:>6.1%}"
          f"{'  WARNING data-bound' if data_bound else ''}")
    if sizes:
        print(f"  Cluster size imbalance:        {imbalance:>5.1f}x  "
              f"(Gini={gini:.2f}, sizes={sizes})")
    if data_bound:
        print(f"  Notice: edge_red < {DATA_BOUND_EDGE_RED_THRESHOLD:.0%} — clustering "
              "captures limited flow structure at this k (often data-bound).")

    return {
        "f_hm_all": round(f_all, 4),
        "f_hm_sound_only": round(wavg_sound.get("f_score", 0.0), 4) if sound_only else None,
        "buijs_hm": round(buijs_all, 4),
        "sound_rate": f"{n_sound}/{n_total}" if n_total else None,
        "entropic_relevance_hm": round(er_weighted_avg, 4) if er_weighted_avg is not None else None,
        "edge_reduction": round(edge_red, 4),
        "data_bound": bool(data_bound),
        "imbalance_max_min": round(imbalance, 2) if imbalance != float("inf") else None,
        "imbalance_gini": round(gini, 4),
        "cluster_sizes": sizes,
    }


def print_summary_line(label: str, m: dict) -> None:
    """One line of fitness, precision, generalization, simplicity and F."""
    print(f"  {label}: fitness={m.get('fitness', 0):.4f}"
          f"  precision={m.get('precision', 0):.4f}"
          f"  generalization={m.get('generalization', 0):.4f}"
          f"  simplicity={m.get('simplicity', 0):.4f}"
          f"  F={m.get('f_score', 0):.4f}")


def compute_weighted_averages(
    im_results: list[dict],
    hm_results: list[dict],
) -> tuple[dict, dict]:
    """Size-weighted mean of each miner's per-cluster metrics."""
    wavg_im = compute_weighted_avg(im_results) or {}
    wavg_hm = compute_weighted_avg(hm_results) or {}
    for label, w in (("IM", wavg_im), ("HM", wavg_hm)):
        if w:
            print(f"=== Weighted Averages (clustered, {label}) ===")
            print_summary_line("clustered", w)
            print()
    return wavg_im, wavg_hm


def _print_cluster_table(label: str, results: list[dict], weighted_avg: dict) -> None:
    if not results:
        return
    print(f"\n=== Per-Cluster Results ({label}) ===")
    print(f"{'Cluster':>8} {'Size':>6} {'Fitness':>8} {'Precis.':>8}"
          f" {'General.':>8} {'Simpl.':>8} {'F-Score':>8}")
    for r in results:
        print(f"{r['cluster_id']:>8} {r['size']:>6}"
              f" {r.get('fitness', 0):>8.4f} {r.get('precision', 0):>8.4f}"
              f" {r.get('generalization', 0):>8.4f} {r.get('simplicity', 0):>8.4f}"
              f" {r.get('f_score', 0):>8.4f}")
    print(
        f"{'W.AVG':>8} {'':>6}"
        f" {weighted_avg.get('fitness', 0):>8.4f} {weighted_avg.get('precision', 0):>8.4f}"
        f" {weighted_avg.get('generalization', 0):>8.4f}"
        f" {weighted_avg.get('simplicity', 0):>8.4f}"
        f" {weighted_avg.get('f_score', 0):>8.4f}"
    )


def print_per_cluster_tables(
    im_results: list[dict], hm_results: list[dict],
    weighted_avg: dict, weighted_avg_hm: dict,
    single_model: dict, global_hm: dict,
) -> None:
    """The per-cluster tables and the method comparison."""
    _print_cluster_table("IM", im_results, weighted_avg)
    _print_cluster_table("HM", hm_results, weighted_avg_hm)

    print("\n=== Method Comparison ===")
    print(f"{'Method':<28s} {'Fitness':>8s} {'Precision':>10s} {'F-Score':>8s} {'Buijs':>8s}")
    print("-" * 68)
    for label, w in [
        ("Per-cluster IM", weighted_avg),
        ("Per-cluster HM", weighted_avg_hm),
        ("Global IM (no clustering)", single_model),
        ("Global HM (no clustering)", global_hm),
    ]:
        if w:
            print(f"{label:<28s} {w.get('fitness',0):>8.4f}"
                  f" {w.get('precision',0):>10.4f} {w.get('f_score',0):>8.4f}"
                  f" {w.get('buijs',0):>8.4f}")


def _compute_er(
    cluster_info: list[tuple[str, Path, Path, int]],
) -> tuple[list[dict], float | None]:
    """Per-cluster entropic relevance and its size-weighted mean."""
    per_cluster: list[dict] = []
    total_size = 0
    weighted_sum = 0.0
    for cluster_id, dfg_path, traces_path, size in cluster_info:
        er = compute_entropic_relevance(dfg_path, traces_path)
        per_cluster.append({"cluster_id": int(cluster_id), "size": size, "er": er})
        if er is not None:
            weighted_sum += er * size
            total_size += size
    avg = round(weighted_sum / total_size, 4) if total_size else None
    return per_cluster, avg


def compute_quality_metrics(
    cluster_info: list[tuple[str, Path, Path, int]],
    cluster_hm_results: list[dict],
    weighted_avg_hm: dict,
    global_hm: dict,
) -> dict:
    """Entropy reduction, entropic relevance and variant purity, printed and returned."""
    cluster_traces_data: list[list[list[str]]] = []
    for _, _, traces_path, _ in cluster_info:
        with open(traces_path) as f:
            cluster_traces_data.append(json.load(f)["traces"])
    if not cluster_traces_data:
        return {}

    entropy = compute_entropy_reduction(cluster_traces_data)
    er_per_cluster, er_avg = _compute_er(cluster_info)
    purity_info = compute_variant_purity(cluster_traces_data)

    out: dict = {
        "entropy": entropy,
        "entropic_relevance": {"weighted_avg": er_avg, "per_cluster": er_per_cluster},
        "variant_purity": purity_info,
    }

    _print_core_metrics(entropy, er_per_cluster, er_avg)
    _print_purity_metrics(purity_info)
    _print_summary_tables(cluster_info, cluster_hm_results, weighted_avg_hm, global_hm,
                          entropy, er_per_cluster, er_avg)
    return out


def _print_core_metrics(
    entropy: dict, er_per_cluster: list[dict], er_weighted_avg: float | None,
) -> None:
    print("\n=== Clustering Quality Metrics ===")
    print("\n  Edge (DFG) Entropy:")
    print(f"    H(global)    = {entropy['edge_H_global']:.4f} bits")
    print(f"    H(clustered) = {entropy['edge_H_clustered']:.4f} bits")
    print(f"    Reduction    = {entropy['edge_reduction']:.1%}")
    for i, h in enumerate(entropy["edge_per_cluster"]):
        print(f"    H(cluster {i}) = {h:.4f} bits")

    print(f"\n  Trace Variant Entropy ({entropy['n_global_variants']} global variants):")
    print(f"    H(global)    = {entropy['variant_H_global']:.4f} bits")
    print(f"    H(clustered) = {entropy['variant_H_clustered']:.4f} bits")
    print(f"    Reduction    = {entropy['variant_reduction']:.1%}")
    for i, h in enumerate(entropy["variant_per_cluster"]):
        print(f"    H(cluster {i}) = {h:.4f} bits")

    print("\n  Entropic Relevance (lower = cluster model encodes log more compactly):")
    if er_weighted_avg is not None:
        print(f"    Weighted avg  = {er_weighted_avg:.4f} bits/trace")
    for row in er_per_cluster:
        if row["er"] is not None:
            print(f"    Cluster {row['cluster_id']}     = {row['er']:.4f} bits/trace "
                  f"(size={row['size']})")


def _print_purity_metrics(purity_info: dict | None) -> None:
    if purity_info is None:
        return
    print("\n  Variant Purity (homogeneity by classification scheme):")
    labels = {"length": "Length", "activity_set": "Activity set", "loop": "Loop behavior"}
    best = max(purity_info, key=lambda s: purity_info[s]["homogeneity"])
    for name, data in purity_info.items():
        label = labels.get(name, name)
        marker = " *" if name == best else ""
        print(f"    {label:<16s} homogeneity={data['homogeneity']:.4f}{marker}")
        print(f"      Global: {data.get('global_distribution', {})}")
        for i, pc in enumerate(data.get("per_cluster", [])):
            parts = sorted(pc["distribution"].items(), key=lambda x: -x[1])
            dist_str = ", ".join(f"{k}:{v}" for k, v in parts)
            print(f"      Cluster {i} ({pc['size']:>3d}): "
                  f"dominant={pc['dominant_fraction']:.0%}  [{dist_str}]")


def _print_summary_tables(
    cluster_info: list[tuple[str, Path, Path, int]],
    cluster_hm_results: list[dict],
    weighted_avg_hm: dict,
    global_hm: dict,
    entropy: dict,
    er_per_cluster: list[dict],
    er_weighted_avg: float | None,
) -> None:
    hm_f = weighted_avg_hm.get("f_score", 0) if weighted_avg_hm else 0
    ghm_f = global_hm.get("f_score", 0) if global_hm else 0
    e_red = f"{entropy['edge_reduction']:.1%}"
    v_red = f"{entropy['variant_reduction']:.1%}"
    er_str = f"{er_weighted_avg:.2f}" if er_weighted_avg is not None else "n/a"

    print(f"\n  {'Approach':<28s} {'F-Score':>8s} {'Edge H':>8s} {'Var H':>8s} {'ER':>8s}")
    print(f"  {'-' * 64}")
    print(f"  {'Global HM (no clustering)':<28s} {ghm_f:>8.4f} {'-':>8s} {'-':>8s} {'-':>8s}")
    print(f"  {'Per-cluster HM':<28s} {hm_f:>8.4f} {e_red:>8s} {v_red:>8s} {er_str:>8s}")

    print("\n  Per-cluster detail (HM):")
    print(f"  {'Cluster':>8s} {'Size':>5s} {'Fitness':>8s} {'Precis.':>8s}"
          f" {'F-Score':>8s} {'Edge H':>7s} {'ER':>7s}"
          f" {'|P|':>4s} {'|T|':>4s} {'|A|':>4s} {'CFC':>4s} {'Sound':>6s}")
    print(f"  {'-' * 96}")
    for i, (cluster_id, _, _, size) in enumerate(cluster_info):
        hm_r = next((r for r in cluster_hm_results if r["cluster_id"] == int(cluster_id)), {})
        h_i = entropy["edge_per_cluster"][i] if i < len(entropy["edge_per_cluster"]) else 0
        er_i = er_per_cluster[i].get("er") if i < len(er_per_cluster) else None
        er_cell = f"{er_i:>7.2f}" if er_i is not None else f"{'n/a':>7s}"
        struct = hm_r.get("structure", {}) or {}
        sound = hm_r.get("sound")
        sound_cell = "yes" if sound is True else ("no" if sound is False else "?")
        print(f"  {cluster_id:>8s} {size:>5d} {hm_r.get('fitness',0):>8.4f}"
              f" {hm_r.get('precision',0):>8.4f} {hm_r.get('f_score',0):>8.4f}"
              f" {h_i:>7.4f} {er_cell}"
              f" {struct.get('n_places', 0):>4d} {struct.get('n_transitions', 0):>4d}"
              f" {struct.get('n_arcs', 0):>4d} {struct.get('cfc', 0):>4d}"
              f" {sound_cell:>6s}")
