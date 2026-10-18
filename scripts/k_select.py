#!/usr/bin/env python3
"""Stability-based k-selection via prediction strength (Tibshirani & Walther 2005).

Measures whether cluster structure generalizes across random data splits, so it
is robust to JL projection noise; runs on revealed features at zero MPC cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import normalize

# Below this simulated edge-entropy reduction a k is vetoed regardless of silhouette or PS.
EDGE_RED_VETO_THRESHOLD = 0.10


def edge_entropy_proxy(X: np.ndarray, labels: np.ndarray) -> float:
    """1 - H_clustered / H_global over per-cluster feature-sum distributions: a one-pass proxy for
    `metrics.compute_entropy_reduction`, affordable per candidate k."""
    if X.shape[0] == 0:
        return 0.0
    K = int(labels.max()) + 1 if labels.size else 0
    n_total = len(X)
    abs_X = np.abs(X)
    global_sum = abs_X.sum(axis=0)
    h_global = float(scipy_entropy(global_sum + 1e-12, base=2))
    if h_global <= 0:
        return 0.0
    h_clustered = 0.0
    for c in range(K):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        cluster_sum = abs_X[mask].sum(axis=0)
        h_c = float(scipy_entropy(cluster_sum + 1e-12, base=2))
        h_clustered += (n_c / n_total) * h_c
    return 1.0 - h_clustered / h_global


def load_features(path: Path) -> np.ndarray:
    """Load feature matrix from CSV."""
    X = np.loadtxt(path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def prediction_strength(X, k, n_splits=20, random_state=42):
    """Tibshirani & Walther (2005) prediction strength: (mean_ps, std_ps)."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    strengths = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = n // 2
        idx_train, idx_test = perm[:half], perm[half : 2 * half]
        X_train, X_test = X[idx_train], X[idx_test]

        km_train = KMeans(
            n_clusters=k, n_init=10,
            random_state=int(rng.randint(1_000_000)),
        )
        km_train.fit(X_train)

        km_test = KMeans(
            n_clusters=k, n_init=10,
            random_state=int(rng.randint(1_000_000)),
        )
        labels_test = km_test.fit_predict(X_test)

        dists = pairwise_distances(X_test, km_train.cluster_centers_)
        labels_pred = dists.argmin(axis=1)

        ps_per_cluster = []
        for j in range(k):
            members = np.where(labels_test == j)[0]
            nj = len(members)
            if nj < 2:
                ps_per_cluster.append(0.0)
                continue
            pred_j = labels_pred[members]
            counts = np.bincount(pred_j, minlength=k)
            co_clustered = int(np.sum(counts * (counts - 1)))
            total_pairs = nj * (nj - 1)
            ps_per_cluster.append(co_clustered / total_pairs)

        strengths.append(min(ps_per_cluster) if ps_per_cluster else 0.0)

    return float(np.mean(strengths)), float(np.std(strengths))


def diversity_floor(X, lo=3, hi=8):
    """Minimum k from behavioural diversity (log2 of distinct quantised rows / 16)."""
    quantized = np.round(X, decimals=2)
    n_unique = len(np.unique(quantized, axis=0))
    if n_unique <= 16:
        return lo
    return int(np.clip(np.ceil(np.log2(n_unique / 16)), lo, hi))


def compute_k_max(n_cases, min_floor=50):
    """Admit only counts whose average cluster holds >= min_floor cases; not MIN_CLUSTER_SIZE
    (src/constants.rs), which is the post-clustering floor on each realized cluster."""
    return max(n_cases // min_floor, 2), min_floor


def select_k_stable(X, k_range, ps_relative_drop=0.75, n_splits=20):
    """Largest k where mean PS >= max(ps_max * ps_relative_drop, 0.5);
    CH breaks ties within 1; argmax PS as fallback."""
    X_norm = normalize(X, norm="l2")
    n, d = X_norm.shape

    results = []
    for k in k_range:
        ps_mean, ps_std = prediction_strength(X_norm, k, n_splits=n_splits)

        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_norm)
        ch = calinski_harabasz_score(X_norm, labels)
        sil = silhouette_score(X_norm, labels)
        edge_red = edge_entropy_proxy(X_norm, labels)

        results.append({
            "k": k,
            "ps_mean": round(ps_mean, 4),
            "ps_std": round(ps_std, 4),
            "calinski_harabasz": round(ch, 1),
            "silhouette": round(sil, 4),
            "edge_red_proxy": round(edge_red, 4),
            "edge_red_veto": edge_red < EDGE_RED_VETO_THRESHOLD,
        })

    ps_max = max(r["ps_mean"] for r in results)
    effective_threshold = max(ps_max * ps_relative_drop, 0.5)

    for r in results:
        r["above_threshold"] = r["ps_mean"] >= effective_threshold

    # Silhouette / PS stay the maximand; edge_red is only a structural floor.
    above = [r for r in results if r["above_threshold"] and not r["edge_red_veto"]]

    if above:
        ps_k = max(above, key=lambda r: r["k"])["k"]
    else:
        non_vetoed = [r for r in results if not r["edge_red_veto"]]
        pool = non_vetoed if non_vetoed else results
        ps_k = max(pool, key=lambda r: r["ps_mean"])["k"]

    # Diversity floor: high-variability logs need more clusters even when PS
    # is low. Apply only when PS alone would pick a very conservative k.
    div_k = diversity_floor(X_norm)
    selected_k = max(ps_k, div_k)

    selected_k = min(selected_k, max(k_range))
    selected_k = max(selected_k, min(k_range))

    selected = next(r for r in results if r["k"] == selected_k)

    method = "prediction_strength"
    if selected_k > ps_k:
        method = "diversity_floor"

    return selected["k"], {
        "rows": results,
        "recommended_k": selected["k"],
        "ps": selected["ps_mean"],
        "method": method,
        "ps_threshold": round(effective_threshold, 4),
        "ps_max": round(ps_max, 4),
        "ps_relative_drop": ps_relative_drop,
        "diversity_floor": div_k,
        "ps_selected": ps_k,
        "n_samples": n,
        "n_features": d,
    }


def resolve_k_range(n_samples, args):
    """Determine k range from arguments and data size."""
    if args.k_range:
        return [int(x) for x in args.k_range.split(",")]

    k_min = args.k_min if args.k_min is not None else 3

    if args.k_max is not None:
        k_max = args.k_max
    else:
        k_max, _ = compute_k_max(n_samples)
        k_max = min(k_max, 10)  # cap for runtime

    # Small inputs (e.g. a heavily suppressed backbone under the hybrid
    # reveal) can push k_max below k_min; never return an empty range.
    k_max = max(k_max, k_min)

    return list(range(k_min, k_max + 1))


def print_results(label, result):
    rows = result["rows"]
    rec_k = result["recommended_k"]
    threshold = result["ps_threshold"]
    div_k = result.get("diversity_floor", "?")
    ps_k = result.get("ps_selected", rec_k)

    print(f"\n{'=' * 72}")
    print(f"  {label}  (N={result['n_samples']}, d={result['n_features']}, "
          f"PS threshold={threshold})")
    print(f"  PS selected k={ps_k}, diversity floor k={div_k}")
    print(f"{'=' * 72}")

    print(f"\n{'k':>3}  {'PS Mean':>8}  {'PS Std':>7}  {'':>3}  "
          f"{'CH Index':>10}  {'Silhouette':>10}  {'EdgeRed':>8}")
    print("-" * 72)
    for r in rows:
        if r["k"] == rec_k:
            mark = ">>>"
        elif r.get("edge_red_veto"):
            mark = "VETO"
        elif r["above_threshold"]:
            mark = "  *"
        else:
            mark = "   "
        print(f"{r['k']:>3}  {r['ps_mean']:>8.4f}  {r['ps_std']:>7.4f}  {mark}  "
              f"{r['calinski_harabasz']:>10.1f}  {r['silhouette']:>10.4f}  "
              f"{r.get('edge_red_proxy', 0):>8.4f}")

    above_count = sum(1 for r in rows if r["above_threshold"])
    print(f"\n  {above_count}/{len(rows)} values above PS threshold {threshold}")
    if not any(r["above_threshold"] for r in rows):
        best = max(rows, key=lambda r: r["ps_mean"])
        print(f"  Fallback: k={best['k']} (highest PS={best['ps_mean']:.4f})")
    print(f"  Selected: k={rec_k} (PS={result['ps']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Stability-based k-selection via prediction strength "
                    "(Tibshirani & Walther 2005)")
    parser.add_argument("features", help="Path to features.csv")
    parser.add_argument("--k-range", default=None,
                        help="Comma-separated k values (overrides --k-min/--k-max)")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=None,
                        help="Maximum k (default: computed from n)")
    parser.add_argument("--ps-relative-drop", type=float, default=0.75,
                        help="Relative PS drop factor (floor at 0.5)")
    parser.add_argument("--n-splits", type=int, default=20,
                        help="Number of train/test splits")
    parser.add_argument("--trace-len", type=int, default=None,
                        help="Accepted for orchestrator compatibility; unused "
                             "(k_max is derived from the sample count, not L)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    X = load_features(Path(args.features))
    k_range = resolve_k_range(len(X), args)
    _, result = select_k_stable(X, k_range, args.ps_relative_drop, args.n_splits)
    if args.format == "json":
        print(json.dumps({
            "recommended_k": result["recommended_k"],
            "method": result["method"],
            "ps": result["ps"],
            "ps_threshold": result["ps_threshold"],
            "diversity_floor": result.get("diversity_floor"),
            "ps_selected": result.get("ps_selected"),
            "details": result["rows"],
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
        }))
    else:
        print_results("features", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
