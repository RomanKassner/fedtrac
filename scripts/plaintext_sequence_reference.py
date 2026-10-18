"""Plaintext twin of mp-spdz/secure_sequence_cluster.mpc: hard-EM over a
mixture of first-order Markov chains on per-case directly-follows count rows,
in float log2 where the box runs fixed point (`score_margins` bounds the gap).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

PAD, START_ACT, END_ACT, FIRST_USER = 0, 1, 2, 3


def structural_cells(dh: int) -> tuple[list[int], list[int]]:
    act_src = [a for a in range(dh) if a not in (PAD, END_ACT)]
    act_tgt = [b for b in range(dh) if b not in (PAD, START_ACT)]
    return act_src, act_tgt


def log_weights(C: np.ndarray, dh: int) -> np.ndarray:
    """k × d̂² matrix of Laplace-smoothed log2 transition weights."""
    k, D = C.shape
    act_src, act_tgt = structural_cells(dh)
    d_eff = len(act_tgt)
    W = np.zeros((k, D), dtype=np.float64)
    for j in range(k):
        Cj = C[j].reshape(dh, dh)
        for a in act_src:
            rs = float(Cj[a, act_tgt].sum()) + d_eff
            lr = math.log2(rs)
            for b in act_tgt:
                W[j, a * dh + b] = math.log2(float(Cj[a, b]) + 1.0) - lr
    return W


def merge_small_clusters_sequence(
    labels: np.ndarray, S: np.ndarray, k: int, min_cluster: int,
) -> np.ndarray:
    """Twin of `secure_sequence_cluster.mpc::merge_small_clusters`."""
    sizes = np.bincount(labels, minlength=k)
    large = [l for l in range(k) if sizes[l] >= min_cluster]
    if not large:
        return labels.copy()
    target = list(range(k))
    for j in range(k):
        if sizes[j] < min_cluster:
            joint = S[labels == j].sum(axis=0) if sizes[j] else np.zeros(k)
            cand = [joint[l] for l in large]
            target[j] = large[int(np.argmax(cand))]
    return np.asarray([target[int(lab)] for lab in labels], dtype=np.int64)


def sequence_labels(
    X: Sequence[Sequence[int]], k: int, max_iter: int, dh: int,
    return_models: bool = False, init_labels: Sequence[int] | None = None,
    min_cluster: int = 0,
):
    X = np.asarray(X, dtype=np.int64)
    n, D = X.shape
    if D != dh * dh:
        raise ValueError(f"row width {D} is not dh^2 for dh={dh}")
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if init_labels is None:
        stride = n // k
        C = np.stack([X[j * stride] for j in range(k)]).astype(np.int64)
    else:
        init = np.asarray(init_labels, dtype=np.int64)
        C = np.zeros((k, D), dtype=np.int64)
        for j in range(k):
            members = np.flatnonzero(init == j)
            if members.size:
                C[j] = X[members].sum(axis=0)
    labels = np.zeros(n, dtype=np.int64)
    for it in range(max_iter):
        W = log_weights(C, dh)
        S = X @ W.T                       # n × k log-likelihoods
        labels = np.argmax(S, axis=1)     # numpy argmax = first index on ties
        if it < max_iter - 1:
            for j in range(k):
                members = np.flatnonzero(labels == j)
                if members.size:
                    C[j] = X[members].sum(axis=0)
    if min_cluster > 0:
        labels = merge_small_clusters_sequence(labels, S, k, min_cluster)
    if return_models:
        return labels.tolist(), C
    return labels.tolist()


def score_margins(X, labels, C, dh) -> list[float]:
    """Per case: best score minus runner-up score (log2 units). Small
    margins are where fixed-point rounding can flip a label."""
    X = np.asarray(X, dtype=np.int64)
    W = log_weights(np.asarray(C), dh)
    S = X @ W.T
    out = []
    for i in range(len(X)):
        s = np.sort(S[i])[::-1]
        out.append(float(s[0] - s[1]) if len(s) > 1 else math.inf)
    return out


def bigram_rows(traces: list[list[int]], dh: int) -> list[list[int]]:
    """Per-case d̂² directly-follows count rows over integer activity ids
    (user ids from FIRST_USER; START→first and last→END included), the
    layout secure_extract persists at offset 0."""
    rows = []
    for tr in traces:
        row = [0] * (dh * dh)
        if tr:
            row[START_ACT * dh + tr[0]] += 1
            for a, b in zip(tr, tr[1:]):
                row[a * dh + b] += 1
            row[tr[-1] * dh + END_ACT] += 1
        rows.append(row)
    return rows


def encode_traces(traces: list[list[str]]) -> tuple[list[list[int]], int, dict[str, int]]:
    """Sorted-name dictionary from FIRST_USER upward (src/types.rs order)."""
    names = sorted({a for tr in traces for a in tr})
    enc = {a: i + FIRST_USER for i, a in enumerate(names)}
    dh = len(names) + FIRST_USER
    return [[enc[a] for a in tr] for tr in traces], dh, enc
