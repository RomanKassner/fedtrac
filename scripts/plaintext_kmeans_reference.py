"""Deterministic Lloyd's k-means: the plaintext specification that
`secure_kmeans.mpc` must reproduce bit-for-bit — integer arithmetic only,
argmin ties to the smallest index, fixed MAX_ITER (stopping early would leak).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction


def argmin_first(values: Sequence[Fraction]) -> int:
    """Argmin with first-index-wins on tie (matches numpy, matches our MPC
    tree_reduce pattern `comp = b < a; swap-to-b else keep-a`)."""
    best_idx = 0
    best_val = values[0]
    for i in range(1, len(values)):
        if values[i] < best_val:
            best_idx = i
            best_val = values[i]
    return best_idx


def lloyd_fractional(
    Y: list[list[int]],
    k: int,
    max_iter: int,
) -> tuple[list[int], list[list[Fraction]]]:
    """Run Lloyd's on integer Y with exact Fraction centroids: the plaintext
    middle step of `--reveal-secret-projection`, which needs real-valued
    centroids. NOT the secure_kmeans twin — `lloyd_integer` is."""
    n = len(Y)
    d = len(Y[0]) if Y else 0
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")

    stride = n // k
    centroids: list[list[Fraction]] = [
        [Fraction(Y[j * stride][dim]) for dim in range(d)] for j in range(k)
    ]
    labels = [0] * n

    for _ in range(max_iter):
        for i in range(n):
            sq_dists = []
            for j in range(k):
                s = Fraction(0)
                for dim in range(d):
                    diff = Fraction(Y[i][dim]) - centroids[j][dim]
                    s += diff * diff
                sq_dists.append(s)
            labels[i] = argmin_first(sq_dists)

        # M-step: centroid := mean of assigned points; empty cluster keeps old
        for j in range(k):
            members = [i for i in range(n) if labels[i] == j]
            if not members:
                continue
            count = Fraction(len(members))
            for dim in range(d):
                s = Fraction(0)
                for i in members:
                    s += Fraction(Y[i][dim])
                centroids[j][dim] = s / count

    return labels, centroids


# Twins of the constants at the top of mp-spdz/secure_kmeans.mpc. Changing
# either side alone breaks the bit-for-bit agreement the harnesses assert.
FRAC_BITS = 16
CENTROID_YMAX_SLACK_BITS = 15


def ceil_log2(x: int) -> int:
    """Twin of Rust's `x.max(1).next_power_of_two().trailing_zeros()` and of
    `secure_kmeans.mpc::_ceil_log2`."""
    return (max(int(x), 1) - 1).bit_length()


def recip_bits(n_cases: int) -> int:
    """`RECIP_BITS` of the tape: the reciprocal `2^RECIP_BITS // n_j` scale."""
    return FRAC_BITS + ceil_log2(n_cases) + CENTROID_YMAX_SLACK_BITS


def merge_small_clusters_integer(
    labels: list[int], chat: list[list[int]], k: int, min_cluster: int,
) -> list[int]:
    """Twin of `secure_kmeans.mpc::merge_small_clusters` and of `clustering.rs::merge_small_clusters`."""
    sizes = [labels.count(j) for j in range(k)]
    large = [l for l in range(k) if sizes[l] >= min_cluster]
    if not large:
        return list(labels)
    target = list(range(k))
    for j in range(k):
        if sizes[j] < min_cluster:
            dists = [sum((a - b) ** 2 for a, b in zip(chat[j], chat[l]))
                     for l in large]
            target[j] = large[argmin_first(dists)]
    return [target[lab] for lab in labels]


def lloyd_integer(
    Y: list[list[int]],
    k: int,
    max_iter: int,
    min_cluster: int = 0,
) -> tuple[list[int], list[list[int]]]:
    """Lloyd's with integer-scaled centroids, the bit-for-bit twin of `secure_kmeans.mpc` (Python `//` and `>>`
    floor toward -inf as MP-SPDZ's restoring division and `comparison.Trunc` do); `chat` is at scale 2^FRAC_BITS."""
    n = len(Y)
    d = len(Y[0]) if Y else 0
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")

    g = recip_bits(n)
    shift = g - FRAC_BITS
    scale = 1 << FRAC_BITS

    stride = n // k
    sums = [list(Y[j * stride]) for j in range(k)]
    counts = [1] * k
    labels = [0] * n
    chat = [[0] * d for _ in range(k)]

    for _ in range(max_iter):
        # Centroid step: exact reciprocal, then an exact arithmetic shift.
        recip = [(1 << g) // counts[j] for j in range(k)]
        chat = [[(sums[j][dim] * recip[j]) >> shift for dim in range(d)]
                for j in range(k)]

        # E-step: direct squared distance against the rounded centroid.
        for i in range(n):
            row = Y[i]
            dists = []
            for j in range(k):
                cj = chat[j]
                s = 0
                for dim in range(d):
                    diff = row[dim] * scale - cj[dim]
                    s += diff * diff
                dists.append(s)
            labels[i] = argmin_first(dists)

        # M-step: exact integer (sums, counts); empty cluster keeps its own.
        new_sums = [[0] * d for _ in range(k)]
        new_counts = [0] * k
        for i in range(n):
            j = labels[i]
            new_counts[j] += 1
            row = Y[i]
            acc = new_sums[j]
            for dim in range(d):
                acc[dim] += row[dim]
        for j in range(k):
            if new_counts[j] > 0:
                sums[j] = new_sums[j]
                counts[j] = new_counts[j]

    if min_cluster > 0:
        # Centroids of the FINAL labeling, as the tape recomputes them.
        recip = [(1 << g) // counts[j] for j in range(k)]
        chat = [[(sums[j][dim] * recip[j]) >> shift for dim in range(d)]
                for j in range(k)]
        labels = merge_small_clusters_integer(labels, chat, k, min_cluster)

    return labels, chat


def lloyd_labels(Y: list[list[int]], k: int, max_iter: int,
                 min_cluster: int = 0) -> list[int]:
    """Labels only, from the secure_kmeans twin."""
    labels, _ = lloyd_integer(Y, k, max_iter, min_cluster=min_cluster)
    return labels


if __name__ == "__main__":
    # Smoke fixture: three obvious clusters in 2D, ordered so stride-based
    # init picks one row from each cluster (N=9, K=3, stride=3 -> rows 0,3,6).
    Y = [
        [0, 0], [1, 0], [0, 1],            # rows 0-2: origin cluster
        [10, 10], [11, 10], [10, 11],      # rows 3-5: (10,10) cluster
        [20, 0], [21, 0], [20, 1],         # rows 6-8: (20,0) cluster
    ]
    labels = lloyd_labels(Y, k=3, max_iter=5)
    print("labels:", labels)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[6] == labels[7] == labels[8]
    assert len({labels[0], labels[3], labels[6]}) == 3
    print("OK")
