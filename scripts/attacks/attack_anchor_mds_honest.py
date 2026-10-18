"""Simulator only: reconstructions in which the adversary holds its own column slice alone and must extrapolate to the other cells after an anchored row matching."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def achlioptas(k, d, rng):
    u = rng.random((k, d))
    R = np.zeros((k, d))
    R[u < 1 / 6] = -1.0
    R[u > 5 / 6] = +1.0
    return R / np.sqrt(k)


def support_metrics(X, X_hat, th=0.3):
    mt = X > 0
    mh = X_hat > th
    tp = int((mt & mh).sum())
    fp = int((~mt & mh).sum())
    fn = int((mt & ~mh).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r


def vertical_partition(d, sizes, rng):
    perm = rng.permutation(d)
    parties, cur = [], 0
    for sz in sizes:
        parties.append(np.sort(perm[cur:cur + sz]))
        cur += sz
    return parties


def simulate_round1(X, k, rng):
    R = achlioptas(k, X.shape[1], rng)
    Y = X @ R.T
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = Y / np.where(norms < 1e-12, 1.0, norms)
    pi = rng.permutation(X.shape[0])
    return Y[pi], pi


def match_rows(Y, X_party, max_iter=15):
    """Iterative party-anchored row matching: Y[perm[i]] is matched to case i."""
    n, k = Y.shape
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    for it in range(max_iter):
        T_party, *_ = lstsq(Y[perm], X_party, rcond=None)
        Xp_pred = Y @ T_party
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Xp_pred)
        _, idx = nn.kneighbors(X_party)
        new_perm = idx.ravel()
        if np.array_equal(new_perm, perm):
            break
        perm = new_perm
    return perm, T_party


def attack_trivial_zero(X_full, Y, party_cols):
    """Lower bound: T_full fitted against a zero-padded X_party."""
    n, d = X_full.shape
    X_party = X_full[:, party_cols]
    perm, _ = match_rows(Y, X_party)
    X_pad = np.zeros_like(X_full)
    X_pad[:, party_cols] = X_party
    T_full, *_ = lstsq(Y[perm], X_pad, rcond=None)
    X_hat = Y[perm] @ T_full
    return X_hat


def attack_distance_mds(X_full, Y, party_cols):
    """Distance-difference MDS, which recovers the geometry of the other columns in an unknown rotation."""
    n, d = X_full.shape
    X_party = X_full[:, party_cols]
    perm, _ = match_rows(Y, X_party)

    inv_perm = np.argsort(perm)
    Y_unshuf = Y[inv_perm]
    D_Y_sq = squareform(pdist(Y_unshuf)) ** 2
    D_party_sq = squareform(pdist(X_party)) ** 2
    D_other_sq = np.maximum(D_Y_sq - D_party_sq, 0.0)
    D_other = np.sqrt(D_other_sq)

    other_cols = np.array([c for c in range(d) if c not in party_cols])
    n_other = len(other_cols)
    mds = MDS(
        n_components=min(n - 1, n_other),
        dissimilarity="precomputed", random_state=0, n_init=1,
        max_iter=200, normalized_stress="auto",
    )
    X_other_emb = mds.fit_transform(D_other)

    if X_other_emb.shape[1] < n_other:
        pad = np.zeros((n, n_other))
        pad[:, :X_other_emb.shape[1]] = X_other_emb
        X_other_emb = pad

    X_hat = np.zeros_like(X_full)
    X_hat[:, party_cols] = X_party
    X_hat[:, other_cols] = X_other_emb[:, :n_other]   # no oracle for the rotation: the slots take the embedding as it is
    return X_hat


def attack_dict_learn(X_full, Y, party_cols, n_atoms=None):
    """Sparse-coding recovery on Y, without aligning the dictionary to X."""
    n, d = X_full.shape
    n_atoms = n_atoms or min(d, Y.shape[1] * 4)
    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms, alpha=1.0, max_iter=50, random_state=0,
        transform_algorithm="lasso_cd", transform_alpha=1.0,
        positive_code=True, positive_dict=False,
        fit_algorithm="cd",
    )
    S = dl.fit_transform(Y)
    if S.shape[1] >= d:
        return S[:, :d]
    out = np.zeros((n, d))
    out[:, :S.shape[1]] = S
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[15, 30])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = np.loadtxt(args.features, delimiter=",")
    n0, d = X.shape
    if args.max_n is not None and n0 > args.max_n:
        idx = rng.choice(n0, args.max_n, replace=False)
        X = X[idx]
    n = X.shape[0]

    sizes = [int(x) for x in args.partition.split(",")]
    if sum(sizes) > d:
        raise SystemExit(f"partition sum {sum(sizes)} > d={d}")
    if sum(sizes) < d:
        sizes.append(d - sum(sizes))
    parties = vertical_partition(d, sizes, rng)

    label = args.label or args.features.parent.parent.name
    print(f"\n=== {label} (HONEST attack; no X_full leak) ===")
    print(f"n={n}, d={d}, partition={sizes}\n")

    print(f"{'k':>3}  {'attack':<18} {'party':<8} "
          f"{'P_unk':>6} {'R_unk':>6}")
    print("-" * 70)

    for k in args.k:
        Y, _ = simulate_round1(X, k, rng)
        for adv_idx, party_cols in enumerate(parties):
            unknown = np.ones((n, d), dtype=bool)
            unknown[:, party_cols] = False

            for attack_name, attack_fn in [
                ("trivial-zero", attack_trivial_zero),
                ("dist-MDS", attack_distance_mds),
                ("dict-learn", attack_dict_learn),
            ]:
                X_hat = attack_fn(X, Y, party_cols)
                p_unk, r_unk = support_metrics(
                    np.where(unknown, X, 0),
                    np.where(unknown, X_hat, 0),
                )
                print(f"{k:>3}  {attack_name:<18} party{adv_idx:<3} "
                      f"{p_unk:>6.2f} {r_unk:>6.2f}")
            print()


if __name__ == "__main__":
    main()
