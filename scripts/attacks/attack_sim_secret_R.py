"""Simulator only: re-projects a run's plaintext features with a fresh Achlioptas R and attacks the projected rows."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq, pinv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import FastICA, MiniBatchDictionaryLearning
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")


def achlioptas(k, d, rng):
    u = rng.random((k, d))
    R = np.zeros((k, d), dtype=np.float64)
    R[u < 1.0 / 6.0] = -1.0
    R[u > 5.0 / 6.0] = +1.0
    return R / np.sqrt(k)


def support_metrics(X, X_hat, thresh=0.3):
    mask_true = X > 0
    mask_hat = X_hat > thresh
    tp = int((mask_true & mask_hat).sum())
    fp = int((~mask_true & mask_hat).sum())
    fn = int((mask_true & ~mask_hat).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r


def _best_align(X_true, X_hat):
    """Hungarian-permute and sign-flip the columns of X_hat onto X_true by |correlation|."""
    n, d = X_true.shape
    _, d_hat = X_hat.shape
    cols = min(d, d_hat)
    mu_t, sd_t = X_true.mean(0, keepdims=True), X_true.std(0, keepdims=True) + 1e-9
    mu_h, sd_h = X_hat.mean(0, keepdims=True), X_hat.std(0, keepdims=True) + 1e-9
    tn = (X_true - mu_t) / sd_t
    hn = (X_hat - mu_h) / sd_h
    C = np.abs(tn[:, :cols].T @ hn[:, :cols]) / n
    row_ind, col_ind = linear_sum_assignment(-C)
    aligned = np.zeros_like(X_true[:, :cols])
    for i, j in zip(row_ind, col_ind):
        vec = X_hat[:, j]
        sign = np.sign(np.corrcoef(X_true[:, i], vec)[0, 1] or 1e-9)
        v_std = vec.std() + 1e-9
        t_std = X_true[:, i].std() + 1e-9
        aligned[:, i] = sign * (t_std / v_std) * (vec - vec.mean()) + X_true[:, i].mean()
    mean_corr = float(np.mean([C[i, j] for i, j in zip(row_ind, col_ind)]))
    return aligned, mean_corr


def attack_pinv_random_R(X, Y, R_shape, seed=123):
    """No-prior baseline: guess an R and pseudoinvert."""
    rng = np.random.default_rng(seed)
    R_guess = achlioptas(*R_shape, rng)
    return Y @ pinv(R_guess).T


def attack_known_sample(X, Y, m):
    """Guo-Wu: solve X_known R^T = Y_known on m known rows, then invert."""
    Xk, Yk = X[:m], Y[:m]
    Rt, *_ = lstsq(Xk, Yk, rcond=None)
    Xh, *_ = lstsq(Rt.T, Y.T, rcond=None)
    return Xh.T


def attack_ica(X, Y, k):
    d = X.shape[1]
    nc = min(k, d)
    try:
        ica = FastICA(n_components=nc, random_state=0, max_iter=800,
                      whiten="unit-variance")
        S = ica.fit_transform(Y)
    except Exception:
        return None
    aligned, corr = _best_align(X[:, :nc], S)
    padded = np.zeros_like(X)
    padded[:, :nc] = aligned
    return padded, corr


def attack_dict_learn(X, Y, k, atoms=None):
    atoms = atoms or min(k, X.shape[1])
    try:
        dl = MiniBatchDictionaryLearning(
            n_components=atoms, alpha=1.0, n_iter=50, random_state=0,
            transform_algorithm="lasso_lars", transform_alpha=1.0,
            positive_code=True, positive_dict=False,
        )
        S = dl.fit_transform(Y)
    except Exception:
        return None
    aligned, corr = _best_align(X[:, :atoms], S)
    padded = np.zeros_like(X)
    padded[:, :atoms] = aligned
    return padded, corr


def attack_distance_mds(X, Y):
    D = squareform(pdist(Y))
    try:
        mds = MDS(
            n_components=min(D.shape[0] - 1, X.shape[1]),
            dissimilarity="precomputed",
            random_state=0, n_init=1, max_iter=200,
            normalized_stress="auto",
        )
        Xh = mds.fit_transform(D)
    except Exception:
        return None
    aligned, corr = _best_align(X, Xh)
    if aligned.shape[1] < X.shape[1]:
        pad = np.zeros_like(X)
        pad[:, : aligned.shape[1]] = aligned
        aligned = pad
    return aligned, corr


def evaluate(X, R_shape_list, known_samples=(5, 40, 100, 140), seed=42,
             match_protocol=True):
    """The battery over a list of k; match_protocol normalises the rows of Y as Round 1 publishes them, and the secret shuffle is omitted because these metrics are row-permutation invariant."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    out = []
    for k in R_shape_list:
        R = achlioptas(k, d, rng)
        Y = X @ R.T
        if match_protocol:
            row_norms = np.linalg.norm(Y, axis=1, keepdims=True)
            Y = Y / np.where(row_norms < 1e-12, 1.0, row_norms)

        Xh = attack_pinv_random_R(X, Y, (k, d))
        out.append((k, "pinv-random-R", -1, *support_metrics(X, Xh)))

        for m in known_samples:
            if m > n:
                continue
            Xh = attack_known_sample(X, Y, m)
            out.append((k, "known-sample", m, *support_metrics(X, Xh)))

        r = attack_ica(X, Y, k)
        if r is not None:
            Xh, corr = r
            out.append((k, "FastICA", -1, *support_metrics(X, Xh)))

        r = attack_dict_learn(X, Y, k)
        if r is not None:
            Xh, corr = r
            out.append((k, "MBDL", -1, *support_metrics(X, Xh)))

        if k <= 40:
            r = attack_distance_mds(X, Y)
            if r is not None:
                Xh, corr = r
                out.append((k, "dist+MDS", -1, *support_metrics(X, Xh)))
    return out


def print_table(rows, label):
    print(f"\n== {label} ==")
    print(f"{'k':>3}  {'attack':<14}  {'m':>4}  {'P':>5}  {'R':>5}")
    for k, atk, m, p, r in rows:
        m_str = "-" if m < 0 else str(m)
        print(f"{k:>3}  {atk:<14}  {m_str:>4}  {p:>5.2f}  {r:>5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path,
                    help="features.csv, comma-separated, n x d")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--known", type=int, nargs="+",
                    default=[5, 40, 100, 140])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-protocol-match", action="store_true",
                    help="skip the row normalisation (raw Y)")
    args = ap.parse_args()

    X = np.loadtxt(args.features, delimiter=",")
    rows = evaluate(X, args.k, tuple(args.known), seed=args.seed,
                    match_protocol=not args.no_protocol_match)
    print_table(rows, args.label or str(args.features))


if __name__ == "__main__":
    main()
