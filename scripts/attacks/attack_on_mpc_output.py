#!/usr/bin/env python3
"""Attacks on the opened `federated/features.csv` of a `--reveal-secret-projection` run, scored against the adversary's oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq, pinv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import FastICA
from sklearn.manifold import MDS


def achlioptas(k, d, rng):
    u = rng.random((k, d))
    R = np.zeros((k, d))
    R[u < 1 / 6] = -1
    R[u > 5 / 6] = 1
    return R / np.sqrt(k)


def pad_rows(Xh, n_rows):
    """Zero-pad a reconstruction to the reference row count, so the withheld rows count as misses."""
    if Xh.shape[0] >= n_rows:
        return Xh[:n_rows]
    out = np.zeros((n_rows, Xh.shape[1]))
    out[: Xh.shape[0]] = Xh
    return out


def metrics(X, Xh, th=0.3):
    mt = X > 0
    mh = Xh > th
    tp = int((mt & mh).sum())
    fp = int((~mt & mh).sum())
    fn = int((mt & ~mh).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r


def best_column_align(X_true, X_hat):
    """Hungarian-permute the columns of X_hat onto X_true by |correlation|, with sign flip and scale-mean match."""
    n, d = X_true.shape
    d_hat = X_hat.shape[1]
    cols = min(d, d_hat)
    Xn = (X_true - X_true.mean(0, keepdims=True)) / (X_true.std(0, keepdims=True) + 1e-9)
    Hn = (X_hat - X_hat.mean(0, keepdims=True)) / (X_hat.std(0, keepdims=True) + 1e-9)
    C = np.abs(Xn[:, :cols].T @ Hn[:, :cols]) / n
    r, c = linear_sum_assignment(-C)
    out = np.zeros_like(X_true[:, :cols])
    for i, j in zip(r, c):
        v = X_hat[:, j]
        sign = np.sign(np.corrcoef(X_true[:, i], v)[0, 1] or 1e-9)
        v_std = v.std() + 1e-9
        t_std = X_true[:, i].std() + 1e-9
        out[:, i] = sign * (t_std / v_std) * (v - v.mean()) + X_true[:, i].mean()
    if cols < d:
        pad = np.zeros_like(X_true)
        pad[:, :cols] = out
        out = pad
    return out


def attack_pinv_random(X, Y, seed=123):
    """No-prior baseline: guess an R and pseudoinvert."""
    rng = np.random.default_rng(seed)
    R = achlioptas(Y.shape[1], X.shape[1], rng)
    Xh = Y @ pinv(R).T
    return Xh


def attack_known_norm_match(X, Y, m):
    """Known-sample attack pairing rows by L2 norm, which the row normalisation of the opened rows makes near-random."""
    Xk = X[:m]
    x_norms = np.linalg.norm(Xk, axis=1)
    y_norms = np.linalg.norm(Y, axis=1)
    C = np.abs(x_norms[:, None] - y_norms[None, :])
    r_idx, c_idx = linear_sum_assignment(C)
    Y_paired = Y[c_idx]
    ridge = 1e-6
    A = Xk.T @ Xk + ridge * np.eye(Xk.shape[1])
    B = Xk.T @ Y_paired
    Mt = np.linalg.solve(A, B)
    Xh, *_ = lstsq(Mt.T, Y.T, rcond=None)
    return Xh.T


def nearest_distance_profiles(Y, m, chunk_rows=None):
    """np.sort(squareform(pdist(Y)), axis=1)[:, 1:m] without the N x N matrix, which does not fit at five-digit N."""
    n = Y.shape[0]
    cols = min(m, n)
    if chunk_rows is None:
        chunk_rows = max(1, int(6e7 // max(n, 1)))   # ~0.5 GB of f64 per chunk
    prof = np.empty((n, cols - 1), dtype=np.float64)
    for lo in range(0, n, chunk_rows):
        D = cdist(Y[lo:lo + chunk_rows], Y)
        part = np.partition(D, cols - 1, axis=1)[:, :cols] if cols < n else D
        prof[lo:lo + chunk_rows] = np.sort(part, axis=1)[:, 1:cols]
    return prof


def attack_known_distance_match(X, Y, m):
    """Pairwise-distance profile matching for the m known rows -> Guo-Wu."""
    Xk = X[:m]
    D_X = squareform(pdist(Xk))
    X_prof = np.sort(D_X, axis=1)[:, 1:m]
    Y_prof = nearest_distance_profiles(Y, m)
    C = cdist(X_prof, Y_prof)
    r_idx, c_idx = linear_sum_assignment(C)
    Y_paired = Y[c_idx]
    ridge = 1e-6
    A = Xk.T @ Xk + ridge * np.eye(Xk.shape[1])
    B = Xk.T @ Y_paired
    Mt = np.linalg.solve(A, B)
    Xh, *_ = lstsq(Mt.T, Y.T, rcond=None)
    return Xh.T, int((c_idx[:m] == np.arange(m)).sum())


def attack_mds(X, Y):
    """Distance matrix, MDS, column alignment."""
    D = squareform(pdist(Y))
    mds = MDS(
        n_components=min(D.shape[0] - 1, X.shape[1]),
        dissimilarity="precomputed",
        random_state=0, n_init=1, max_iter=200, normalized_stress="auto",
    )
    Xh = mds.fit_transform(D)
    return best_column_align(X, pad_rows(Xh, X.shape[0]))


def attack_ica(X, Y):
    nc = min(Y.shape[1], X.shape[1])
    ica = FastICA(n_components=nc, random_state=0, max_iter=800, whiten="unit-variance")
    try:
        S = ica.fit_transform(Y)
    except (ValueError, FloatingPointError):
        return np.zeros_like(X)                       # no finite mixing on degenerate Y: a failed attack
    if not np.all(np.isfinite(S)):
        return np.zeros_like(X)
    return best_column_align(X, pad_rows(S, X.shape[0]))


def sorted_quantile(a, q):
    """np.quantile(a, q) ('linear' method) on an already sorted a, without the re-partition per q."""
    a = np.asarray(a, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n = a.shape[0]
    virt = (n - 1) * q
    prev = np.floor(virt).astype(np.intp)
    nxt = prev + 1
    above = virt >= n - 1
    below = virt < 0
    prev[above] = n - 1; nxt[above] = n - 1
    prev[below] = 0; nxt[below] = 0
    gamma = virt - prev
    x, y = a[prev], a[nxt]
    diff = y - x
    out = x + diff * gamma
    hi = gamma >= 0.5
    out[hi] = y[hi] - diff[hi] * (1 - gamma[hi])
    return out


def jl_distance_fidelity(X, Y):
    """Pairwise-distance multisets of X and Y, each rescaled to unit median: the distance leak the projection accepts."""
    dx = np.sort(pdist(X))
    dy = np.sort(pdist(Y))
    if len(dx) != len(dy):
        q = np.linspace(0, 1, min(len(dx), len(dy)))   # fewer opened rows than reference rows: matched quantiles
        dx = sorted_quantile(dx, q)
        dy = sorted_quantile(dy, q)
    dx = dx / (np.median(dx) + 1e-9)
    dy = dy / (np.median(dy) + 1e-9)
    rel = np.abs(dx - dy) / np.maximum(dx, 1e-9)
    return float(np.median(rel)), float((rel < 0.25).mean())


def anonymous_set(X, k_priv):
    """Row indices of X that Round 1 opens at k_priv: nonzero rows whose exact value at least k_priv rows share."""
    keys = [tuple(r) for r in np.asarray(X)]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return np.array([i for i, k in enumerate(keys) if counts[k] >= k_priv and any(v != 0 for v in k)],
                    dtype=np.intp)


def oracle_metrics(X_all, A, Xh, th=0.3, max_rows=3000, seed=0):
    """P and R of a reconstruction in the opened (secretly permuted) order under the row pairing that minimises the support mismatch: an upper bound, since the permutation is never opened."""
    T = np.asarray(X_all)[A] > 0
    H = np.asarray(Xh) > th
    n_t, n_h = T.shape[0], H.shape[0]
    rng = np.random.default_rng(seed)
    if n_t > max_rows:
        T = T[rng.choice(n_t, max_rows, replace=False)]
    if n_h > max_rows:
        H = H[rng.choice(n_h, max_rows, replace=False)]
    if T.shape[1] != H.shape[1]:
        cols = min(T.shape[1], H.shape[1])
        T, H = T[:, :cols], H[:, :cols]
    h = H.sum(1).astype(np.float64)
    t = T.sum(1).astype(np.float64)
    inter = H.astype(np.float64) @ T.astype(np.float64).T
    cost = h[:, None] + t[None, :] - 2.0 * inter                    # support mismatches per pair
    r_idx, c_idx = linear_sum_assignment(cost)
    tp = float(inter[r_idx, c_idx].sum())
    fp = float(h[r_idx].sum() - tp)
    fn_open = float(t[c_idx].sum() - tp)
    p = tp / max(1.0, tp + fp)
    r_open = tp / max(1.0, tp + fn_open)
    s_open = float((np.asarray(X_all)[A] > 0).sum())
    s_all = float((np.asarray(X_all) > 0).sum())
    r = r_open * (s_open / max(1.0, s_all))
    return p, r, int(len(r_idx))


def attack_frequency_match(X_all, Y, drop_zero=True):
    """Frequency matching: opened classes of identical rows ranked by size against the log's profiles ranked by count (oracle knowledge, so an upper bound), a tie broken lexicographically on both sides."""
    X_all = np.asarray(X_all, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    classes, class_of_row, sizes = np.unique(Y, axis=0, return_inverse=True, return_counts=True)
    class_of_row = np.asarray(class_of_row).reshape(-1)
    profiles, counts = np.unique(X_all, axis=0, return_counts=True)
    if drop_zero:
        nz = np.any(profiles != 0, axis=1)
        profiles, counts = profiles[nz], counts[nz]
    c_rank = np.argsort(-sizes, kind="stable")        # np.unique is lexicographic; stable keeps it in a tie
    p_rank = np.argsort(-counts, kind="stable")
    match = np.full(len(classes), -1, dtype=np.intp)
    m = min(len(c_rank), len(p_rank))
    match[c_rank[:m]] = p_rank[:m]
    padded = np.vstack([profiles, np.zeros((1, X_all.shape[1]))])
    Xh = padded[match[class_of_row]]                  # -1 picks the zero pad row
    return Xh, dict(classes=classes, sizes=sizes, class_of_row=class_of_row,
                    profiles=profiles, counts=counts, match=match)


def frequency_match_scores(info, k_priv, true_profile=None):
    """Class accuracy and exact-row recall of a frequency match; without true_profile a class in a size tie counts as a miss, so both are a floor."""
    sizes, counts, match = info["sizes"], info["counts"], info["match"]
    opened_counts = np.sort(counts[counts >= k_priv])[::-1]
    sizes_ok = bool(np.array_equal(np.sort(sizes)[::-1], opened_counts))
    size_ties = {int(s): int(t) for s, t in zip(*np.unique(sizes, return_counts=True))}
    if true_profile is None:
        true_profile = np.full(len(sizes), -1, dtype=np.intp)
        for c, s in enumerate(sizes):
            cand = np.flatnonzero(counts == s)
            if size_ties[int(s)] == 1 and len(cand) == 1:
                true_profile[c] = cand[0]
    true_profile = np.asarray(true_profile)
    hit = (true_profile >= 0) & (match == true_profile)
    tied = np.array([size_ties[int(s)] > 1 for s in sizes])
    return dict(classes=int(len(sizes)), tied_classes=int(tied.sum()), tied_rows=int(sizes[tied].sum()),
                sizes_ok=sizes_ok, hits=int(hit.sum()), hit_rows=int(sizes[hit].sum()),
                class_acc=float(hit.mean()) if len(sizes) else 0.0,
                row_acc=float(sizes[hit].sum() / max(1, sizes.sum())))


def tie_rule_expectation(info, k_proj, draws=200, seed=0, decimals=6):
    """Expected hits of the lexicographic tie rule over R drawn as secure_extract_rsecret.mpc Step 2b draws it (uniform {-1,+1}), in antithetic pairs (R, -R), which reverse every tie."""
    sizes, counts, profiles = info["sizes"], info["counts"], info["profiles"]
    groups = [int(s) for s, t in zip(*np.unique(sizes, return_counts=True)) if t > 1]
    if not groups:
        return 0.0, 0.0
    members = {s: profiles[counts == s] for s in groups}
    rng = np.random.default_rng(seed)
    pairs = max(1, draws // 2)
    hits = np.zeros(len(groups))
    for _ in range(pairs):
        R = rng.choice([-1.0, 1.0], size=(k_proj, profiles.shape[1]))
        for sign in (1.0, -1.0):
            for g, s in enumerate(groups):
                Z = members[s] @ (sign * R).T
                Yg = np.round(Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12), decimals)
                order = np.lexsort(Yg.T[::-1])
                hits[g] += int((order == np.arange(len(order))).sum())
    hits /= 2 * pairs
    return float(hits.sum()), float((hits * np.array(groups)).sum())


def frequency_match_report(X_all, A, Y, k_priv, oracle_max_rows=3000, draws=200):
    """The frequency match scored by oracle_metrics, with its class accuracy and exact-row recall, certain and expected over R."""
    Xh, info = attack_frequency_match(X_all, Y)
    p, r, rows = oracle_metrics(X_all, A, Xh, max_rows=oracle_max_rows)
    sc = frequency_match_scores(info, k_priv)
    e_cls, e_rows = tie_rule_expectation(info, Y.shape[1], draws=draws)
    n_rows = int(info["sizes"].sum())
    out = dict(P=p, R=r, oracle_rows=rows, **sc,
               class_acc_exp=(sc["hits"] + e_cls) / max(1, sc["classes"]),
               row_acc_exp=(sc["hit_rows"] + e_rows) / max(1, n_rows), draws=2 * max(1, draws // 2))
    lines = [f"frequency match: {sc['classes']} opened classes over {n_rows} opened rows, class sizes equal the "
             f"counts of the opened profiles: {'OK' if sc['sizes_ok'] else 'MISMATCH'}; {sc['tied_classes']} classes "
             f"({sc['tied_rows']} rows) in size ties",
             f"frequency match: class_acc = {out['class_acc']:.4f}, row_acc = {out['row_acc']:.4f} (certain: a tied "
             f"class counts as a miss); expected over R (lexicographic tie rule, {out['draws']} antithetic draws): "
             f"class_acc_exp = {out['class_acc_exp']:.4f}, row_acc_exp = {out['row_acc_exp']:.4f}",
             f"frequency match: oracle-aligned P = {p:.4f}, R = {r:.4f} on {rows} rows (the rebuilt multiset equals the "
             f"true one whenever the sizes match, so the oracle scores it 1 on the opened rows below the row cap)"]
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--m-list", type=int, nargs="+",
                    default=[5, 20, 50])
    ap.add_argument("--truth", type=Path, default=None,
                    help="features_encoded.csv of the cell's own encoding (clustering_federated "
                         "--truncate-baseline --dump-encoded-features); with it every attack is scored "
                         "with the oracle row alignment on the anonymous set of --k-priv")
    ap.add_argument("--k-priv", type=int, default=None, help="the anonymity threshold the run selected")
    ap.add_argument("--oracle-max-rows", type=int, default=3000,
                    help="row cap of the oracle assignment (both sides sampled above it)")
    ap.add_argument("--quad-max-rows", type=int, default=0,
                    help="row cap for the distance-quadratic steps, JL fidelity and distance+MDS (0 = no cap)")
    args = ap.parse_args()

    fed = args.run / "federated" / "features.csv"
    pt = args.run / "plaintext" / "features.csv"
    if not fed.exists() or not pt.exists():
        print(f"Missing features.csv in {args.run}. Run with --evaluate-full.")
        return

    Y = np.loadtxt(fed, delimiter=",")
    oracle = args.truth is not None
    if oracle:
        if args.k_priv is None:
            raise SystemExit("--truth needs --k-priv (the threshold the run selected)")
        X_all = np.loadtxt(args.truth, delimiter=",")
        A = anonymous_set(X_all, args.k_priv)
        X = X_all[A]
    else:
        X = np.loadtxt(pt, delimiter=",")
    n_total = Y.shape[0]
    nonzero = np.linalg.norm(Y, axis=1) > 1e-9        # a row below k_priv is masked to zero before the reveal
    Y = Y[nonzero]
    n, k = Y.shape
    _, d = X.shape
    print(f"\n{'=' * 72}")
    print(f"  Run: {args.run.name}")
    mean_l0 = int((X > 0).sum(1).mean())
    if oracle:
        print(f"  X (truth {args.truth.name}, k_priv={args.k_priv}): all rows {X_all.shape[0]} d={d}, "
              f"anonymous set |A|={len(A)} against {int(nonzero.sum())} opened rows -> "
              f"{'OK' if len(A) == int(nonzero.sum()) else 'MISMATCH'}; scoring = ORACLE ROW ALIGNMENT "
              f"(max {args.oracle_max_rows} rows)")
    else:
        print(f"  X (plaintext reference): n={X.shape[0]} d={d}, mean row L0={mean_l0}")
        print("  scoring = positional (row i of the reconstruction against row i of the reference; "
              "NOT the opened row's case, see --truth)")
    print(f"  Y (revealed, shuffled, normalised): n={n} of {n_total} rows non-zero "
          f"({n_total - n} suppressed-tail rows masked), k={k}, "
          f"L2 norm range [{np.linalg.norm(Y,axis=1).min():.3f}, "
          f"{np.linalg.norm(Y,axis=1).max():.3f}]")
    print(f"{'=' * 72}\n")

    # The JL-fidelity and MDS steps are quadratic in rows; above --quad-max-rows they run on an announced subsample.
    qmax = args.quad_max_rows
    X_quad, Y_quad = X, Y
    if qmax and (X.shape[0] > qmax or Y.shape[0] > qmax):
        rng = np.random.default_rng(0)
        if Y.shape[0] > qmax:
            Y_quad = Y[rng.choice(Y.shape[0], qmax, replace=False)]
        if X.shape[0] > qmax:
            X_quad = X[rng.choice(X.shape[0], qmax, replace=False)]
        print(f"[quad-subsample] JL-fidelity and distance+MDS use "
              f"{Y_quad.shape[0]} of {Y.shape[0]} revealed rows and "
              f"{X_quad.shape[0]} of {X.shape[0]} reference rows "
              f"(--quad-max-rows {qmax}); all other attacks run at full N\n")

    med_rel, ok_frac = jl_distance_fidelity(X_quad, Y_quad)
    print(f"JL distance fidelity: median rel-diff = {med_rel:.2f}, "
          f"within-25% = {ok_frac:.1%}\n")

    results = []

    def score(Xh):
        if oracle:
            p, r, rows = oracle_metrics(X_all, A, Xh, max_rows=args.oracle_max_rows)
            return p, r, f"oracle-aligned on {rows} rows"
        p, r = metrics(X, pad_rows(Xh, X.shape[0]))
        return p, r, None

    Xh = attack_pinv_random(X, Y)
    p, r, det = score(Xh)
    results.append(("no-prior pinv", "-", p, r, det))

    if oracle:
        print("FastICA and distance+MDS are not scored on the opened rows: their column alignment "
              "needs the row pairing; the geometry bound is upper_bound__sim_secret_R at the run's width\n")
    else:
        Xh = attack_ica(X, Y)
        p, r = metrics(X, Xh)
        results.append(("FastICA", "-", p, r, None))

        Xh = attack_mds(X_quad, Y_quad)
        p, r = metrics(X_quad, Xh)
        results.append(("distance+MDS", "-", p, r, None))

    for m in args.m_list:
        if m > n:
            continue
        Xh = attack_known_norm_match(X, Y, m)
        p, r, det = score(Xh)
        results.append(("known-sample (norm match)", str(m), p, r, det))

    for m in args.m_list:                             # pairwise-distance matching: the strongest try at the shuffle
        if m > n:
            continue
        Xh, correct = attack_known_distance_match(X, Y, m)
        p, r, det = score(Xh)
        results.append(("known-sample (dist match)", str(m), p, r,
                        det if oracle else f"{correct}/{m} paired"))

    freq_lines = []                                   # frequency matching needs the truth's frequency table
    if oracle:
        fm, freq_lines = frequency_match_report(X_all, A, Y, args.k_priv, oracle_max_rows=args.oracle_max_rows)
        results.append(("frequency match (oracle table)", "-", fm["P"], fm["R"],
                        f"oracle-aligned on {fm['oracle_rows']} rows"))

    print(f"{'attack':<30} {'m':>4}  {'P':>5}  {'R':>5}  detail")
    print("-" * 72)
    for name, m, p, r, det in results:
        print(f"{name:<30} {m:>4}  {p:>5.2f}  {r:>5.2f}  {det or ''}")
    for line in freq_lines:
        print(line)


if __name__ == "__main__":
    main()
