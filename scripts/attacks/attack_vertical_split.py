"""Simulator only: an adversary party that knows its own columns of X and the shuffled projected rows tries to recover the columns it does not own."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform


def achlioptas(k, d, rng):
    u = rng.random((k, d))
    R = np.zeros((k, d))
    R[u < 1 / 6] = -1.0
    R[u > 5 / 6] = +1.0
    return R / np.sqrt(k)


def support_metrics(X_true, X_hat, th=0.3):
    mt = X_true > 0
    mh = X_hat > th
    tp = int((mt & mh).sum())
    fp = int((~mt & mh).sum())
    fn = int((mt & ~mh).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r


def vertical_partition(d, sizes, rng):
    """Assign columns to parties at random with the given sizes."""
    perm = rng.permutation(d)
    parties = []
    cursor = 0
    for sz in sizes:
        parties.append(np.sort(perm[cursor:cursor + sz]))
        cursor += sz
    return parties


def align_rows_by_distance(X_known_cols, Y, party_cols, n_known):
    """Match m known cases to rows of Y by pairwise-distance fingerprints over the adversary's own columns."""
    Xk = X_known_cols[:n_known]                     # (m, |party_cols|)
    if Xk.shape[1] < 2:
        return None
    D_X = squareform(pdist(Xk))
    D_Y = squareform(pdist(Y))
    if n_known < 3:
        return None
    X_prof = np.sort(D_X, axis=1)[:, 1:n_known]
    Y_prof = np.sort(D_Y, axis=1)[:, 1:n_known]
    cost = np.linalg.norm(X_prof[:, None, :] - Y_prof[None, :, :], axis=2)
    r_idx, c_idx = linear_sum_assignment(cost)
    return c_idx


def guo_wu_with_mask(X_full, Y, m, party_cols, true_perm):
    """Ceiling: the true alignment of the m known cases, with only the party's columns visible, so R^T comes from a column subset."""
    n, k = Y.shape
    d = X_full.shape[1]
    Xk_partial = np.zeros((m, d))
    Xk_partial[:, party_cols] = X_full[true_perm[:m]][:, party_cols]
    Yk = Y[:m]
    Rt, *_ = lstsq(Xk_partial, Yk, rcond=None)
    Xh, *_ = lstsq(Rt.T, Y.T, rcond=None)
    return Xh.T


def known_pinv_columns_only(X_full, Y, party_cols, true_perm, m):
    """Lower bound: the inversion restricted to the adversary's own columns, which recovers nothing new by construction."""
    n, k = Y.shape
    d = X_full.shape[1]
    Xk = X_full[true_perm[:m]][:, party_cols]   # m x |P|
    Yk = Y[:m]
    if Xk.shape[1] < 1:
        return np.zeros_like(X_full)
    Rt_P, *_ = lstsq(Xk, Yk, rcond=None)
    Xh_P, *_ = lstsq(Rt_P.T, Y.T, rcond=None)
    Xh = np.zeros((n, d))
    Xh[:, party_cols] = Xh_P.T
    return Xh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--partition", default="24,24,80",
                    help="column counts per party, comma-separated (sum <= d)")
    ap.add_argument("--k", type=int, nargs="+", default=[15, 30, 60])
    ap.add_argument("--m", type=int, nargs="+", default=[10, 30, 100],
                    help="known-row counts for the alignment")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    X = np.loadtxt(args.features, delimiter=",")
    n, d = X.shape
    sizes = [int(x) for x in args.partition.split(",")]
    if sum(sizes) > d:
        raise SystemExit(f"partition sum {sum(sizes)} > d={d}")
    if sum(sizes) < d:
        sizes.append(d - sum(sizes))
    parties = vertical_partition(d, sizes, rng)

    print(f"\nDataset: n={n}, d={d}, mean row L0 = {(X>0).sum(1).mean():.1f}")
    print(f"Vertical partition (col counts): {sizes}")
    for i, p in enumerate(parties):
        print(f"  party {i}: |cols|={len(p)}")

    for k_proj in args.k:
        print(f"\n{'='*78}\n  Projection k={k_proj}\n{'='*78}")
        R = achlioptas(k_proj, d, rng)
        Y_unshuf = X @ R.T

        pi = rng.permutation(n)
        Y = Y_unshuf[pi]
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)   # the row normalisation of Round 1
        true_perm = np.argsort(pi)                                    # the ceilings only

        for adv_idx, party_cols in enumerate(parties):
            print(f"\n--- Adversary = party {adv_idx} "
                  f"({len(party_cols)} known cols of {d}) ---")
            unknown_mask = np.ones((n, d), dtype=bool)
            unknown_mask[:, party_cols] = False

            X_known = np.zeros_like(X)
            X_known[:, party_cols] = X[:, party_cols]

            # 1. alignment by distance fingerprints over the party's columns
            for m in args.m:
                if m > n:
                    continue
                X_party_only = X[:, party_cols]
                aligned = align_rows_by_distance(X_party_only, Yn, party_cols, m)
                if aligned is None:
                    continue
                truth = true_perm[:m]
                correct = int((aligned == truth).sum())
                print(f"  align by dist-fp (m={m}): {correct}/{m} "
                      f"({100*correct/m:.1f}%) rows pinned correctly")

            # 2. the ceiling: a perfect alignment and Guo-Wu with the mask
            for m in args.m:
                if m > n:
                    continue
                Xh = guo_wu_with_mask(X, Yn, m, party_cols, true_perm)
                p_all, r_all = support_metrics(X, Xh)
                Xh_priv = Xh.copy()
                X_priv = X.copy()
                p_priv, r_priv = support_metrics(
                    np.where(unknown_mask, X_priv, 0),
                    np.where(unknown_mask, Xh_priv, 0)
                )
                print(f"  Guo-Wu w/mask (perfect align, m={m}): "
                      f"all P={p_all:.2f} R={r_all:.2f}; "
                      f"unknown-cells P={p_priv:.2f} R={r_priv:.2f}")

            # 3. the lower bound: the column-restricted inversion
            Xh = known_pinv_columns_only(X, Yn, party_cols, true_perm, m=min(50, n))
            p_priv, r_priv = support_metrics(
                np.where(unknown_mask, X, 0),
                np.where(unknown_mask, Xh, 0)
            )
            print(f"  column-only inversion (lower bound): "
                  f"unknown-cells P={p_priv:.2f} R={r_priv:.2f}")


if __name__ == "__main__":
    main()
