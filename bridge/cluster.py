"""Clustering dispatch for the plaintext + reveal-secret-projection paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


def load_features(path: Path) -> np.ndarray:
    X = np.loadtxt(path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def cluster_means(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Per-cluster mean over rows; missing clusters emit a zero vector."""
    d = X.shape[1]
    means = np.zeros((k, d), dtype=np.float64)
    for j in range(k):
        rows = X[labels == j]
        if len(rows) > 0:
            means[j] = rows.mean(axis=0)
    return means


def rehome_noise(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace HDBSCAN noise (-1) with the nearest cluster; -> (relabeled, noise_count)."""
    noise_mask = labels < 0
    noise_count = int(noise_mask.sum())
    if noise_count == 0:
        return labels, 0
    valid = ~noise_mask
    if valid.sum() == 0:
        # Pathological: every point is noise. Collapse to a single cluster.
        return np.zeros_like(labels), noise_count

    valid_labels = labels[valid]
    uniq = np.unique(valid_labels)
    centroids = np.stack([X[valid][valid_labels == j].mean(axis=0) for j in uniq])
    diffs = X[noise_mask][:, None, :] - centroids[None, :, :]
    nearest_idx = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
    relabeled = labels.copy()
    relabeled[noise_mask] = uniq[nearest_idx]
    return relabeled, noise_count


def compact_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Re-map labels to a contiguous 0..k-1 range. Returns (compacted, k)."""
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    out = np.array([remap[v] for v in labels], dtype=np.int64)
    return out, len(uniq)


def spherical_kmeans(X: np.ndarray, k: int, seed: int,
                     max_iter: int = 100, n_init: int = 10) -> np.ndarray:
    """k-means on the unit sphere (Dhillon & Modha 2001)."""
    n, d = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)

    best_labels = None
    best_obj = -np.inf
    for run in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        C = Xn[idx].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            sim = Xn @ C.T
            new_labels = np.argmax(sim, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                rows = Xn[labels == j]
                if len(rows) > 0:
                    m = rows.mean(axis=0)
                    norm = np.linalg.norm(m)
                    C[j] = m / norm if norm > 1e-12 else Xn[rng.integers(n)]
        # Mean intra-cluster cosine similarity (objective to maximise)
        obj = float((Xn * C[labels]).sum(axis=1).mean())
        if obj > best_obj:
            best_obj = obj
            best_labels = labels.copy()
    return best_labels


def consensus_clustering(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Co-association ensemble (Fred & Jain 2005)."""
    n = len(X)
    rng = np.random.default_rng(seed)

    base_partitions = []
    for s in rng.integers(0, 10_000, size=3):
        base_partitions.append(
            KMeans(n_clusters=k, n_init=5, random_state=int(s)).fit_predict(X)
        )
    base_partitions.append(
        AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    )
    try:
        base_partitions.append(
            GaussianMixture(n_components=k, covariance_type="diag",
                            random_state=seed, max_iter=200).fit_predict(X)
        )
    except (ValueError, np.linalg.LinAlgError):
        pass
    base_partitions.append(spherical_kmeans(X, k, seed, n_init=3))

    co = np.zeros((n, n), dtype=np.float32)
    for p in base_partitions:
        co += (p[:, None] == p[None, :]).astype(np.float32)
    co /= len(base_partitions)

    distance = 1.0 - co
    np.fill_diagonal(distance, 0.0)
    final = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(distance)
    return final


def sequence_clustering(X: np.ndarray, k: int, dict_size: int, max_iter: int,
                        init_labels: Path | None) -> np.ndarray:
    """Markov-chain hard-EM on d² count rows; clear twin of mp-spdz/secure_sequence_cluster.mpc."""
    # Resolves only because the module runs as `python -m bridge.cluster` from the root.
    from scripts.plaintext_sequence_reference import sequence_labels

    if dict_size <= 0:
        raise SystemExit("--algo sequence needs --dict-size")
    counts = np.rint(X).astype(np.int64)
    init = None
    if init_labels is not None:
        init = np.loadtxt(init_labels, dtype=np.int64, ndmin=1)
        if len(init) != len(counts):
            raise SystemExit(
                f"--init-labels has {len(init)} rows for {len(counts)} cases"
            )
    labels = sequence_labels(counts, k, max_iter, dict_size, init_labels=init)
    return np.asarray(labels, dtype=np.int64)


def fit(algo: str, X: np.ndarray, k: int, seed: int, dict_size: int = 0,
        max_iter: int = 5, init_labels: Path | None = None) -> tuple[np.ndarray, int]:
    """(labels, noise_count); a label may be -1 for noise, which the caller re-homes."""
    if algo == "gmm-diag":
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag",
            random_state=seed, max_iter=200,
        )
        return gmm.fit_predict(X), 0
    if algo == "ward":
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return agg.fit_predict(X), 0
    if algo == "hdbscan":
        n = len(X)
        min_size = max(5, n // max(k * 2, 10) if k > 0 else n // 20)
        h = HDBSCAN(min_cluster_size=min_size)
        return h.fit_predict(X), 0
    if algo == "kmeans-cos":
        return spherical_kmeans(X, k, seed), 0
    if algo == "consensus":
        return consensus_clustering(X, k, seed), 0
    if algo == "sequence":
        return sequence_clustering(X, k, dict_size, max_iter, init_labels), 0
    raise SystemExit(f"unknown algo: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--algo", required=True,
                    choices=["gmm-diag", "ward", "hdbscan",
                             "kmeans-cos", "consensus", "sequence"])
    ap.add_argument("--k", type=int, default=0,
                    help="cluster count; HDBSCAN treats it as a hint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dict-size", type=int, default=0,
                    help="sequence: activity-dictionary size (rows are d^2 wide)")
    ap.add_argument("--max-iter", type=int, default=5,
                    help="sequence: hard-EM iterations (fixed, as in MPC)")
    ap.add_argument("--init-labels", type=Path, default=None,
                    help="sequence: initial partition, one integer per line; "
                         "omitted = stride seeds")
    args = ap.parse_args()

    X = load_features(args.features)
    raw_labels, _ = fit(args.algo, X, args.k, args.seed,
                        dict_size=args.dict_size, max_iter=args.max_iter,
                        init_labels=args.init_labels)
    rehomed, noise_count = rehome_noise(X, raw_labels)
    compacted, k_actual = compact_labels(rehomed)
    means = cluster_means(X, compacted, k_actual)

    # An emptied model leaves the tiers evaluated at different k, which nothing else records.
    if args.k and k_actual < args.k:
        print(f"WARNING: {args.algo} returned k_actual={k_actual} < k={args.k} "
              "(a cluster emptied); tiers are not evaluated at the same k",
              file=sys.stderr)
    out = {
        "assignments": compacted.tolist(),
        "representatives": means.tolist(),
        "k_requested": args.k,
        "k_actual": k_actual,
        "noise_count": noise_count,
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
