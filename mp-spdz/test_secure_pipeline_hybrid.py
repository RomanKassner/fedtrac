#!/usr/bin/env python3
"""The --reveal-secret-projection chain: the suppressing extract, the backbone clustering, relabel, aggregate."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import harness_common as hc  # noqa: E402
import test_secure_extract as se  # noqa: E402
from harness_common import clear_persistence  # noqa: E402,F401

from scripts.plaintext_kmeans_reference import argmin_first, lloyd_fractional  # noqa: E402

_constants_path = HERE / "mpc_utils" / "constants.py"
_spec = importlib.util.spec_from_file_location("hybrid_constants", _constants_path)
_const_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_const_mod)
NORM_SCALE = int(_const_mod.NORM_SCALE)


def parse_rows_with_backbone(stdout: str, n_cases: int, k_proj: int):
    """Twin of src/mpc/parse.rs::parse_row_output_backbone: the backbone comes from the zero mask, a tail row being
    masked to zero and a backbone row unit-norm, and must agree with the opened count."""
    rows = []
    current = None
    n_backbone = None
    for line in stdout.splitlines():
        if line.strip().startswith("N_BACKBONE "):
            n_backbone = int(line.split()[1])
        elif line.startswith("ROW_START"):
            current = [0] * k_proj
        elif line.startswith("ROW_END"):
            if current is not None:
                rows.append(current)
                current = None
        elif current is not None and line.startswith("B "):
            parts = line[2:].split()
            if len(parts) == 2:
                idx, val = int(parts[0]), int(parts[1])
                if idx < k_proj:
                    current[idx] = val
    assert len(rows) == n_cases, f"Expected {n_cases} rows, got {len(rows)}"
    assert n_backbone is not None, "N_BACKBONE line missing from rsecret stdout"
    backbone = [1 if any(v != 0 for v in row) else 0 for row in rows]
    assert sum(backbone) == n_backbone, (
        f"zero-mask backbone count {sum(backbone)} != opened N_BACKBONE {n_backbone}")
    return rows, backbone


def derive_sizes_from_dfg(mpc_dfg, dict_size: int) -> list[int]:
    """Cluster size from the DFG START-row sum, as src/mpc/rounds.rs::run_round2_secure derives it."""
    start_base = 1 * dict_size
    return [sum(row[start_base:start_base + dict_size]) for row in mpc_dfg]


def to_signed(raw: list[list[int]]) -> np.ndarray:
    """Reinterpret the opened unsigned ring elements as signed, as the Rust parser does."""
    return np.array(
        [[(int(v) - 2**64) if int(v) >= 2**63 else int(v) for v in row] for row in raw],
        dtype=np.int64,
    )


START_ACT, END_ACT, FIRST_USER_ACT = 1, 2, 3  # twin of mpc_utils/constants.py


def dense_enc_dim(encoding: int, dict_size: int) -> int:
    """Twin of mpc_utils.expected_enc_dim: the width each encoding compiles at."""
    if encoding == 3:
        return dict_size
    if encoding == 4:
        du = dict_size - FIRST_USER_ACT
        return du * du * du
    if encoding == 5:
        return 2 * dict_size * dict_size
    return dict_size * dict_size


def extract_trigram_plaintext(merged, dict_size: int) -> list[int]:
    """Clear twin of mpc_utils/encoding_trigram.py; a trace of fewer than three real events gives an all-zero row."""
    du = dict_size - FIRST_USER_ACT
    out = [0] * (du * du * du)
    acts = [act - FIRST_USER_ACT for _, act in merged]
    for k in range(len(acts) - 2):
        a, b, c = acts[k], acts[k + 1], acts[k + 2]
        out[a * du * du + b * du + c] += 1
    return out


def extract_skip_bigram_plaintext(merged, dict_size: int) -> list[int]:
    """Clear twin of mpc_utils/encoding_skip_bigram.py: block 0 the direct pairs, block 1 the distance-2 pairs."""
    d = dict_size
    block = d * d
    out = [0] * (2 * block)
    acts = [act for _, act in merged]
    n = len(acts)
    if n == 0:
        return out
    out[START_ACT * d + acts[0]] += 1
    for k in range(n - 1):
        out[acts[k] * d + acts[k + 1]] += 1
    out[acts[-1] * d + END_ACT] += 1
    if n >= 2:
        out[block + START_ACT * d + acts[1]] += 1
        out[block + acts[-2] * d + END_ACT] += 1
    else:
        out[block + START_ACT * d + END_ACT] += 1
    for k in range(n - 2):
        out[block + acts[k] * d + acts[k + 2]] += 1
    return out


def ground_truth_backbone(profiles: list[list[int]], k_priv: int) -> list[bool]:
    """Clear twin of secure_extract_rsecret.mpc Step 2c-anon: multiplicity at least k_priv and a nonzero profile."""
    keys = [tuple(h) for h in profiles]
    mult = Counter(keys)
    return [mult[key] >= k_priv and any(v != 0 for v in prof)
            for key, prof in zip(keys, profiles)]


def nearest_centroid_int(row: list[int], centroids_int: list[list[int]]) -> int:
    """Clear twin of secure_relabel.mpc's tail assignment, in the same scaled space."""
    sq = []
    for c in centroids_int:
        s = 0
        for a, b in zip(row, c):
            d = a - b
            s += d * d
        sq.append(s)
    return argmin_first(sq)


def aggregate_plaintext(histograms, labels, k, dict_size):
    d_sq = dict_size * dict_size
    dfg = [[0] * d_sq for _ in range(k)]
    sizes = [0] * k
    for hist, lab in zip(histograms, labels):
        sizes[lab] += 1
        for d in range(d_sq):
            dfg[lab][d] += hist[d]
    return dfg, sizes


def write_hash_weights(spdz_dir, prog_name, enc_dim, seed=0x46494E47):
    """The fingerprint hash weights as runtime public input; twin of src/mpc/runtime.rs::expand_hash_weights."""
    import random
    rng = random.Random(seed)
    pdir = Path(spdz_dir) / "Programs" / "Public-Input"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / prog_name, "w") as f:
        for _ in range(enc_dim):
            f.write(f"{rng.randrange(1, 1 << 48)}\n")


def write_relabel_inputs(spdz_dir, n_parties, labels, centroids_int, tail_idx):
    """Party-0 input in the order src/mpc/rounds.rs::run_relabel writes it: labels, centroids, tail count, indices."""
    values = list(labels)
    for row in centroids_int:
        values.extend(row)
    values.append(len(tail_idx))
    values.extend(tail_idx)
    hc.write_party0_values(spdz_dir, values)


def truncate_encoded(all_encoded, trunc):
    """Truncate each party to this many real events and re-pad, so prefixes collide and the backbone is non-empty."""
    PAD = (se.PAD_ACT, 0)
    out = []
    for party in all_encoded:
        new_party = []
        for case in party:
            real = [e for e in case if e[0] != se.PAD_ACT][:trunc]
            real += [PAD] * (trunc - len(real))
            new_party.append(real)
        out.append(new_party)
    return out, trunc


def build_trigram_zero_phi_fixture():
    """Two-party fixture with zero-phi rows under the trigram encoding: a one-event trace has no consecutive triple,
    so suppression must force it into the tail while it still reaches Round 2 by nearest centroid."""
    dict_size = 6
    seqs = (
        [[3, 4, 5]] * 6 +   # one trigram class per sequence
        [[4, 5, 3]] * 4 +
        [[5, 3, 4]] * 2 +
        [[3, 5, 4]] * 1 +   # a nonzero singleton, suppressed
        [[3]] * 3           # zero-phi under the trigram encoding
    )
    partial_len = max(len(s) for s in seqs)
    n_cases = len(seqs)
    common_ids = [f"c{i}" for i in range(n_cases)]
    party_a, party_b = [], []
    for s in seqs:
        # Strictly increasing timestamps, so the merge order is the sequence order.
        a = [(act, t + 1) for t, act in enumerate(s)]
        a += [(se.PAD_ACT, se.PAD_TS)] * (partial_len - len(a))
        party_a.append(a)
        party_b.append([(se.PAD_ACT, se.PAD_TS)] * partial_len)
    return [party_a, party_b], common_ids, dict_size, partial_len


def build_union_coverage_fixture():
    """Two-party fixture with ragged coverage in both directions: an all-PAD block sits mid-selection for either
    party, and class membership stays a property of the merged content alone."""
    dict_size = 6
    seqs = (
        [[3, 4, 5]] * 6 +
        [[4, 5, 3]] * 4 +
        [[5, 3, 4]] * 2 +
        [[3, 5, 4]] * 1 +   # the only suppressed row at K_PRIV=2
        [[3]] * 3
    )
    partial_len = max(len(s) for s in seqs)
    common_ids = [f"u{i}" for i in range(len(seqs))]
    party_a, party_b = [], []
    for i, s in enumerate(seqs):
        # The timestamps make the merge order the sequence order however the events are dealt out.
        events = [(act, t + 1) for t, act in enumerate(s)]
        mode = i % 4
        if mode == 0:
            a, b = events, []
        elif mode == 1:
            a, b = [], events
        elif mode == 2:
            a, b = events[:1], events[1:]
        else:
            a = [e for j, e in enumerate(events) if j % 2 == 0]
            b = [e for j, e in enumerate(events) if j % 2 == 1]
        pad = [(se.PAD_ACT, se.PAD_TS)]
        party_a.append(a + pad * (partial_len - len(a)))
        party_b.append(b + pad * (partial_len - len(b)))
    return [party_a, party_b], common_ids, dict_size, partial_len


def build_large_tail_fixture(n_short: int = 66):
    """Trigram fixture whose suppressed tail exceeds secure_relabel.mpc's TAIL_PARALLEL, filling one unrolled batch
    and a remainder; the tail is all zero-phi, so its secret Y is zero too and the reference stays exact."""
    dict_size = 6
    seqs = [[3, 4, 5]] * 6 + [[4, 5, 3]] * 4 + [[3]] * n_short
    partial_len = max(len(s) for s in seqs)
    common_ids = [f"t{i}" for i in range(len(seqs))]
    party_a, party_b = [], []
    for s in seqs:
        a = [(act, t + 1) for t, act in enumerate(s)]
        a += [(se.PAD_ACT, se.PAD_TS)] * (partial_len - len(a))
        party_a.append(a)
        party_b.append([(se.PAD_ACT, se.PAD_TS)] * partial_len)
    return [party_a, party_b], common_ids, dict_size, partial_len


def run_pipeline(spdz_dir, party_files, max_cases, k_proj, k, max_iter, k_priv, trunc,
                 encoding=0, enc_dim=None, prebuilt=None, universe="intersection",
                 tail_cmp_bits=None, relabel_timeout=300):
    n_parties = len(party_files)
    enc_label = {0: "bigram", 4: "trigram", 5: "skipbigram"}.get(encoding, str(encoding))
    print(f"\n{'='*64}\nHYBRID PIPELINE  enc={enc_label}  K_PRIV={k_priv}  k={k}  "
          f"k_proj={k_proj}  max_cases={max_cases}  trunc={trunc}  "
          f"universe={universe}\n{'='*64}")

    if prebuilt is not None:
        all_encoded, common_ids, dict_size, partial_len = prebuilt
    else:
        all_encoded, common_ids, dict_size, partial_len, _, _ = se.prepare_nparty_data(
            party_files, max_cases, universe=universe
        )
        if trunc:
            all_encoded, partial_len = truncate_encoded(all_encoded, trunc)
    n_cases = len(common_ids)
    d_sq = dict_size * dict_size
    if enc_dim is None:
        enc_dim = dense_enc_dim(encoding, dict_size)
    print(f"  n_cases={n_cases}  dict_size={dict_size}  partial_len={partial_len}  "
          f"enc_dim={enc_dim}")
    assert n_cases >= k, f"need n_cases ({n_cases}) >= k ({k})"

    se.write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)

    # The DFG slab is always the bigrams, while the fingerprint uses the chosen encoding's profile rows.
    histograms = []
    profiles = []
    for c in range(n_cases):
        merged = se.merge_events_plaintext(
            [all_encoded[p][c] for p in range(n_parties)], dict_size)
        histograms.append(se.extract_bigrams_plaintext(merged, dict_size))
        if encoding == 4:
            profiles.append(extract_trigram_plaintext(merged, dict_size))
        elif encoding == 5:
            profiles.append(extract_skip_bigram_plaintext(merged, dict_size))
        else:
            profiles.append(histograms[-1])

    gt_backbone = ground_truth_backbone(profiles, k_priv)
    n_bb_gt = sum(gt_backbone)
    n_zero_phi = sum(1 for p in profiles if not any(v != 0 for v in p))
    print(f"  ground-truth backbone: {n_bb_gt}/{n_cases} "
          f"({n_bb_gt/n_cases*100:.1f}%)  suppressed={n_cases-n_bb_gt}  "
          f"zero-φ={n_zero_phi}")

    # Stage 1: the suppressing extract. ENC_DIM 0 keeps the bigram default; another encoding passes its exact width.
    clear_persistence(spdz_dir, n_parties)
    enc_dim_arg = enc_dim if encoding != 0 else 0
    extract_args = [n_cases, partial_len, dict_size, 4, k_proj, n_parties,
                    encoding, enc_dim_arg, k_priv]
    prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", extract_args,
                          n_parties=n_parties)
    write_hash_weights(spdz_dir, prog, enc_dim)
    stdout, t1 = se.run_mpc(spdz_dir, prog, n_parties, timeout=600)
    raw, bb_flags = parse_rows_with_backbone(stdout, n_cases, k_proj)
    y_scaled = to_signed(raw)
    print(f"  [extract] {t1:.1f}s  revealed backbone rows={sum(bb_flags)}")

    # The backbone count must match the ground truth; the positions are shuffled.
    assert sum(bb_flags) == n_bb_gt, (
        f"backbone count {sum(bb_flags)} != ground-truth {n_bb_gt}")

    # Every suppressed row must be opened as all-zero.
    for i, bb in enumerate(bb_flags):
        if not bb:
            assert not y_scaled[i].any(), f"suppressed row {i} not zeroed: {y_scaled[i]}"
    print("  OK backbone count matches; tail rows masked to zero")

    # The clustering in the clear, on the backbone rows only.
    bb_idx = [i for i, b in enumerate(bb_flags) if b]
    Y_bb = [list(map(int, y_scaled[i])) for i in bb_idx]
    bb_labels, centroids_frac = lloyd_fractional(Y_bb, k=k, max_iter=max_iter)
    # The centroids are rounded in the scaled space, as the Rust side rounds them.
    centroids_int = [[round(c) for c in row] for row in centroids_frac]

    # One label per shuffled position: a backbone row carries its own, a tail position a sentinel.
    shuffled_labels = [0] * n_cases
    for pos, lab in zip(bb_idx, bb_labels):
        shuffled_labels[pos] = lab

    # The clear reference for the tail: the nearest centroid in the same scaled space.
    ref_shuffled_labels = list(shuffled_labels)
    for i in range(n_cases):
        if not bb_flags[i]:
            ref_shuffled_labels[i] = nearest_centroid_int(
                list(map(int, y_scaled[i])), centroids_int)

    # Stage 2: the relabel, with the tail assignment.
    pi_offset = n_cases * d_sq
    y_offset = pi_offset + n_cases
    bb_offset = y_offset + n_cases * k_proj
    label_offset = bb_offset + n_cases

    tail_idx = [i for i, b in enumerate(bb_flags) if not b]
    write_relabel_inputs(spdz_dir, n_parties, shuffled_labels, centroids_int, tail_idx)
    # TAIL_CMP_BITS is the narrowed tail-score width rounds.rs::relabel_tail_cmp_bits passes; absent is the ring.
    relabel_args = [n_cases, pi_offset, 4, k, label_offset, 1, k_proj, y_offset, bb_offset]
    if tail_cmp_bits is not None:
        relabel_args.append(int(tail_cmp_bits))
    prog = se.compile_mpc(spdz_dir, "secure_relabel", relabel_args,
                          n_parties=n_parties)
    stdout2, t2 = se.run_mpc(spdz_dir, prog, n_parties, timeout=relabel_timeout)
    print(f"  [relabel] {t2:.1f}s  n_tail={len(tail_idx)}"
          + (f"  TAIL_CMP_BITS={tail_cmp_bits}" if tail_cmp_bits is not None
             else "  TAIL_CMP_BITS=default (ring width)"))
    if tail_cmp_bits is not None:
        assert f"TAIL_CMP_BITS={int(tail_cmp_bits)}\n" in stdout2, (
            f"secure_relabel did not compare at TAIL_CMP_BITS={tail_cmp_bits}")
        assert int(tail_cmp_bits) < hc.RING, (
            f"TAIL_CMP_BITS={tail_cmp_bits} does not narrow the {hc.RING}-bit "
            "ring, so the cell proves nothing")

    # The opened per-case labels, in original case order.
    mpc_case_labels = [None] * n_cases
    for line in stdout2.splitlines():
        if line.startswith("ASSIGN "):
            _, i, c = line.split()
            mpc_case_labels[int(i)] = int(c)
    assert all(v is not None for v in mpc_case_labels), "missing ASSIGN lines"

    # Stage 3: the aggregate with the edge filter off, so the opened cells are the raw sums the reference computes.
    agg_args = [n_cases, n_cases, d_sq, k, 1, label_offset]
    prog = se.compile_mpc(spdz_dir, "secure_aggregate", agg_args,
                          n_parties=n_parties)
    stdout3, t3 = se.run_mpc(spdz_dir, prog, n_parties, timeout=300)
    mpc_dfg = se.parse_row_output(stdout3, k, d_sq)
    mpc_sizes = derive_sizes_from_dfg(mpc_dfg, dict_size)
    print(f"  [aggregate] {t3:.1f}s  sizes={mpc_sizes} (START-row derived)")

    # The comparison is on the published aggregate, whose sizes and cells are order-free sums; a nonzero-phi tail row
    # is ambiguous under the secret R and may move one unit of mass between two clusters.
    n_tail = sum(1 for b in bb_flags if not b)
    n_ambiguous = max(0, n_tail - n_zero_phi) if n_tail else 0
    ref_size_dist = sorted(Counter(ref_shuffled_labels).get(j, 0) for j in range(k))
    mpc_size_dist = sorted(mpc_sizes)
    dist_l1 = sum(abs(a - b) for a, b in zip(ref_size_dist, mpc_size_dist))
    assert dist_l1 <= 2 * n_ambiguous, (
        f"size distribution mismatch beyond the {n_ambiguous} R-ambiguous "
        f"tail rows: ref {ref_size_dist} vs mpc {mpc_size_dist}")

    # The reference for what the secret-label aggregate must publish, from the tape's own un-shuffled labels.
    ref_dfg, ref_sizes = aggregate_plaintext(histograms, mpc_case_labels, k, dict_size)
    assert ref_sizes == mpc_sizes, f"sizes: ref {ref_sizes} vs mpc {mpc_sizes}"
    dfg_mismatch = [(j, d) for j in range(k) for d in range(d_sq)
                    if ref_dfg[j][d] != mpc_dfg[j][d]]
    assert not dfg_mismatch, f"DFG mismatch at {dfg_mismatch[:5]}"
    print(f"  OK aggregate: {k*d_sq} DFG cells + sizes match the all-case "
          f"(backbone+tail) reference")

    # The tail labels as multisets, a per-row check needing the secret permutation.
    multiset_diff = sum(
        (Counter(mpc_case_labels) - Counter(ref_shuffled_labels)).values())
    assert multiset_diff <= n_ambiguous, (
        f"published label multiset differs from the hybrid reference in "
        f"{multiset_diff} rows (> {n_ambiguous} R-ambiguous tail rows)")
    print(f"  OK tail hybrid-assignment label multiset matches reference "
          f"(<= {n_ambiguous} R-ambiguous rows tolerated)")

    return {
        "n_cases": n_cases, "n_backbone": sum(bb_flags), "n_zero_phi": n_zero_phi,
        "sizes": mpc_sizes, "dfg": mpc_dfg, "labels": sorted(mpc_case_labels),
        "all_backbone": all(bb_flags),
    }


def budget_bounded_k_priv(profiles, k0: int, tau: float) -> int:
    """Clear twin of the ADAPTIVE selection: the largest k whose suppressed count stays within the budget."""
    from collections import Counter
    zero = [not any(v != 0 for v in p) for p in profiles]
    n_zero = sum(zero)
    sizes = list(Counter(tuple(p) for p, z in zip(profiles, zero) if not z).values())
    n = len(profiles)
    tau_count = int(tau * 10000) * n // 10000
    chosen = k0
    for k in range(k0, n + 1):
        if n_zero + sum(s for s in sizes if s < k) <= tau_count:
            chosen = k
    return chosen


def run_adaptive_extract_check(spdz_dir, party_files, max_cases, k_proj, trunc,
                               k0=2, tau=0.5, encoding=0, enc_dim=None, prebuilt=None,
                               universe="intersection"):
    """The in-MPC binary search must select the clear reference's threshold and enforce it on the backbone."""
    n_parties = len(party_files)
    enc_label = {0: "bigram", 4: "trigram", 5: "skipbigram"}.get(encoding, str(encoding))
    print(f"\n{'='*64}\nADAPTIVE K_PRIV SELECTION  enc={enc_label}  k0={k0}  "
          f"tau={tau}  universe={universe}\n{'='*64}")
    if prebuilt is not None:
        all_encoded, common_ids, dict_size, partial_len = prebuilt
    else:
        all_encoded, common_ids, dict_size, partial_len, _, _ = se.prepare_nparty_data(
            party_files, max_cases, universe=universe
        )
        if trunc:
            all_encoded, partial_len = truncate_encoded(all_encoded, trunc)
    n_cases = len(common_ids)
    if enc_dim is None:
        enc_dim = dense_enc_dim(encoding, dict_size)
    se.write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)

    profiles = []
    for c in range(n_cases):
        merged = se.merge_events_plaintext(
            [all_encoded[p][c] for p in range(n_parties)], dict_size)
        if encoding == 4:
            profiles.append(extract_trigram_plaintext(merged, dict_size))
        elif encoding == 5:
            profiles.append(extract_skip_bigram_plaintext(merged, dict_size))
        else:
            profiles.append(se.extract_bigrams_plaintext(merged, dict_size))
    expected = budget_bounded_k_priv(profiles, k0, tau)
    gt_backbone = ground_truth_backbone(profiles, expected)
    n_zero_phi = sum(1 for p in profiles if not any(v != 0 for v in p))
    eff = (n_cases - sum(gt_backbone)) / n_cases
    tau_count = int(tau * 10000) * n_cases // 10000
    print(f"  expected k_priv={expected}, ground-truth backbone "
          f"{sum(gt_backbone)}/{n_cases}, zero-φ={n_zero_phi}, "
          f"effective suppressed={eff*100:.1f}% (budget floor(tau*n)={tau_count})")
    # Suppression stays within the budget unless it is pinned at the floor, where over-budget costs utility only.
    assert (n_cases - sum(gt_backbone)) <= tau_count or expected == k0, (
        f"effective suppression {n_cases - sum(gt_backbone)} > budget {tau_count} "
        f"at non-floor k_priv={expected}")

    clear_persistence(spdz_dir, n_parties)
    enc_dim_arg = enc_dim if encoding != 0 else 0
    extract_args = [n_cases, partial_len, dict_size, 4, k_proj, n_parties,
                    encoding, enc_dim_arg, k0, int(tau * 10000), 1]
    prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", extract_args,
                          n_parties=n_parties)
    write_hash_weights(spdz_dir, prog, enc_dim)
    stdout, t1 = se.run_mpc(spdz_dir, prog, n_parties, timeout=600)
    selected = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("K_PRIV_SELECTED "):
            selected = int(line.split()[1])
            break
    _, bb_flags = parse_rows_with_backbone(stdout, n_cases, k_proj)
    print(f"  [extract] {t1:.1f}s  K_PRIV_SELECTED={selected}  "
          f"backbone={sum(bb_flags)}")
    assert selected == expected, (
        f"in-MPC selection {selected} != plaintext reference {expected}")
    assert sum(bb_flags) == sum(gt_backbone), (
        f"backbone count {sum(bb_flags)} != ground truth at selected "
        f"threshold {sum(gt_backbone)}")
    print("  OK adaptive selection matches reference (effective tau budget); "
          "backbone enforced at it")


def main():
    ap = hc.common_parser("Hybrid-reveal (rsec) integration test")
    ap.add_argument("--max-cases", type=int, default=16)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--k-proj", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--k-priv", type=int, default=2)
    ap.add_argument("--trunc", type=int, default=3,
                    help="truncate to this PARTIAL_LEN to force profile collisions")
    args = ap.parse_args()
    spdz_dir = hc.setup(args)

    data_dir = PROJECT_ROOT / "data" / "splits" / "sepsis_2party"
    party_files = [data_dir / "Party_A.xes", data_dir / "Party_B.xes"]
    missing = [f for f in party_files if not f.exists()]
    if missing:
        print(f"ERROR: missing party files: {missing}", file=sys.stderr)
        sys.exit(1)

    # The narrowed tail-comparison width a pipeline run passes, at the .mpc's own NORM_PRESHIFT default.
    tail_bits = hc.relabel_tail_cmp_bits(args.k_proj)
    print(f"\n  TAIL_CMP_BITS twin (k_proj={args.k_proj}, "
          f"NORM_PRESHIFT={hc.NORM_PRESHIFT_DEFAULT}) = {tail_bits}")

    # The suppressing run, where a short L must leave some rows suppressed.
    hybrid = run_pipeline(spdz_dir, party_files, args.max_cases, args.k_proj,
                          args.k, args.max_iter, k_priv=args.k_priv, trunc=args.trunc,
                          tail_cmp_bits=tail_bits)
    assert hybrid["n_backbone"] < hybrid["n_cases"], (
        "test fixture produced no suppression; lower --trunc or raise --k-priv")

    # K_PRIV=1: every row is anonymous.
    base = run_pipeline(spdz_dir, party_files, args.max_cases, args.k_proj,
                        args.k, args.max_iter, k_priv=1, trunc=args.trunc)
    assert base["all_backbone"], "K_PRIV=1 must reveal every row (no suppression)"
    assert base["n_backbone"] == base["n_cases"], "K_PRIV=1 backbone != all cases"
    print("\n  OK K_PRIV=1 regression: no suppression, every row revealed")

    # The skip-bigram encoding against its dense clear twin.
    skip = run_pipeline(spdz_dir, party_files, args.max_cases, args.k_proj,
                        args.k, args.max_iter, k_priv=args.k_priv,
                        trunc=args.trunc, encoding=5)
    assert skip["n_backbone"] < skip["n_cases"], (
        "skip-bigram fixture produced no suppression; profiles diverge from "
        "the dense 2d² twin")
    print("\n  OK skip-bigram: backbone/tail vs the dense 2d² twin, "
          "all-case DFG/sizes match")

    # The adaptive threshold selection against the clear reference.
    run_adaptive_extract_check(spdz_dir, party_files, args.max_cases,
                               args.k_proj, args.trunc)

    # The trigram encoding, where a short trace gives an all-zero profile that must be suppressed and still reach
    # Round 2 through the nearest-centroid assignment.
    fixture = build_trigram_zero_phi_fixture()
    dict_size = fixture[2]
    du_cubed = dense_enc_dim(4, dict_size)
    party_stub = [Path("A"), Path("B")]  # only its length is read with a prebuilt fixture
    tri = run_pipeline(spdz_dir, party_stub, args.max_cases, k_proj=6, k=2,
                       max_iter=args.max_iter, k_priv=2, trunc=0,
                       encoding=4, enc_dim=du_cubed, prebuilt=fixture)
    assert tri["n_zero_phi"] > 0, "trigram fixture produced no zero-φ rows"
    assert tri["n_backbone"] < tri["n_cases"], "trigram fixture: no suppression"
    assert tri["n_backbone"] == tri["n_cases"] - 4, (
        f"expected 4 suppressed (1 singleton + 3 zero-φ), got "
        f"{tri['n_cases'] - tri['n_backbone']}")
    print("\n  OK trigram zero-φ: zero-φ rows suppressed (bb=0), N_BACKBONE == "
          "nonzero revealed rows, all-case DFG/sizes match (tail reaches Round 2)")

    # The adaptive selection on that fixture must count the zero-phi rows as suppressed.
    run_adaptive_extract_check(spdz_dir, party_stub, args.max_cases, k_proj=6,
                               trunc=0, k0=2, tau=0.40, encoding=4,
                               enc_dim=du_cubed, prebuilt=fixture)

    # A tail larger than TAIL_PARALLEL, so the unrolled tail-argmin block fills once and leaves a remainder.
    big = build_large_tail_fixture()
    big_tail_bits = hc.relabel_tail_cmp_bits(6)
    bigtail = run_pipeline(spdz_dir, party_stub, args.max_cases, k_proj=6, k=2,
                           max_iter=args.max_iter, k_priv=2, trunc=0,
                           encoding=4, enc_dim=du_cubed, prebuilt=big,
                           tail_cmp_bits=big_tail_bits, relabel_timeout=900)
    n_tail_big = bigtail["n_cases"] - bigtail["n_backbone"]
    assert n_tail_big > 64, (
        f"large-tail fixture produced only {n_tail_big} tail rows; the "
        "TAIL_PARALLEL=64 block still never fills")
    assert bigtail["n_zero_phi"] == n_tail_big, (
        "large-tail fixture must have an all-zero-φ tail so the reference "
        f"stays exact: {bigtail['n_zero_phi']} zero-φ vs {n_tail_big} tail")
    print(f"\n  OK large tail: n_tail={n_tail_big} > TAIL_PARALLEL=64 at "
          f"TAIL_CMP_BITS={big_tail_bits}, all-case DFG/sizes match")

    # The union universe on real data: the split whose case sets differ, with single-party cases at middle slots.
    union_dir = PROJECT_ROOT / "data" / "splits" / "sepsis_2party_erlab_wards"
    union_files = [union_dir / "Party_ER_Lab.xes", union_dir / "Party_Wards.xes"]
    if all(f.exists() for f in union_files):
        uni = run_pipeline(spdz_dir, union_files, args.max_cases, args.k_proj,
                           args.k, args.max_iter, k_priv=args.k_priv,
                           trunc=args.trunc, universe="union")
        assert uni["n_backbone"] < uni["n_cases"], (
            "union fixture produced no suppression; lower --trunc or raise --k-priv")
        print("\n  OK union universe (real split): backbone/tail and all-case "
              "DFG/sizes match the union-population reference")
        run_adaptive_extract_check(spdz_dir, union_files, args.max_cases,
                                   args.k_proj, args.trunc, universe="union")
    else:
        print(f"\nSKIP union universe test: missing {union_files}")

    # Union coverage in both directions, where only the lone singleton may be suppressed.
    ufix = build_union_coverage_fixture()
    uni2 = run_pipeline(spdz_dir, party_stub, args.max_cases, k_proj=6, k=2,
                        max_iter=args.max_iter, k_priv=2, trunc=0,
                        prebuilt=ufix, universe="union")
    assert uni2["n_zero_phi"] == 0, (
        f"union coverage fixture: no row may be zero-φ, got {uni2['n_zero_phi']}")
    assert uni2["n_backbone"] == uni2["n_cases"] - 1, (
        f"union coverage fixture: expected exactly 1 suppressed (the [3,5,4] "
        f"singleton), got {uni2['n_cases'] - uni2['n_backbone']}")
    print("\n  OK union coverage both directions: all-PAD blocks at mid-selection "
          "slots for either party, fingerprint classes follow the merged content")

    print("\nALL PASS")
    return True


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_pipeline_hybrid", main))
