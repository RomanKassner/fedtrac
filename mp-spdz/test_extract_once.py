#!/usr/bin/env python3
"""Validation gate for --reuse-extraction: one run's core slab must feed both a k-means tail and a secret-R tail."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import harness_common as hc  # noqa: E402
import test_secure_extract as se  # noqa: E402
import test_secure_pipeline_hybrid as sph  # noqa: E402

from scripts.plaintext_kmeans_reference import lloyd_fractional  # noqa: E402


# Twin of src/mpc/extract_reuse.rs::enc_slab_offset: the slab rides above every mode's persistence write range.
SLAB_MAX_CLUSTERS = 64


def enc_slab_offset(n_cases: int, d_sq: int, enc_dim: int) -> int:
    return n_cases * (d_sq + enc_dim + 2 * (SLAB_MAX_CLUSTERS + 1))


def sk_extract_kmeans_aggregate(spdz_dir, n_cases, partial_len, dict_size, k_proj, k,
                                max_iter, n_parties, extract_args):
    """One k-means chain: extract, cluster, aggregate; the caller decides whether persistence was cleared."""
    d_sq = dict_size * dict_size
    prog = se.compile_mpc(spdz_dir, "secure_extract", extract_args,
                          n_parties=n_parties)
    se.run_mpc(spdz_dir, prog, n_parties=n_parties)

    y_offset = n_cases * d_sq
    prog = se.compile_mpc(spdz_dir, "secure_kmeans",
                          [n_cases, k_proj, k, max_iter, hc.THREADS, 1, y_offset],
                          n_parties=n_parties)
    se.run_mpc(spdz_dir, prog, n_parties=n_parties)

    label_offset = y_offset + n_cases * k_proj
    prog = se.compile_mpc(spdz_dir, "secure_aggregate",
                          [n_cases, n_cases, d_sq, k, 1, label_offset],
                          n_parties=n_parties)
    stdout, _ = se.run_mpc(spdz_dir, prog, n_parties=n_parties)
    dfg = se.parse_row_output(stdout, k, d_sq)
    sizes = sph.derive_sizes_from_dfg(dfg, dict_size)
    return dfg, sizes


def sk_monolithic_args(n_cases, partial_len, dict_size, k_proj, n_parties):
    # Without the CORE_MODE arguments: ENCODING=0, SECURE_Y=1.
    return [n_cases, partial_len, dict_size, 4, k_proj, n_parties, 0, 1]


def sk_reuse_args(n_cases, partial_len, dict_size, k_proj, n_parties, enc_off, core_mode,
                  persist_enc):
    # The reuse argument layout of src/mpc/rounds.rs::run_round1_secure.
    d_sq = dict_size * dict_size
    return [n_cases, partial_len, dict_size, 4, k_proj, n_parties, 0, 1,
            d_sq, "0", core_mode, persist_enc, enc_off, 0]


def test_sk_bit_identity(spdz_dir, party_files, max_cases, k_proj, k, max_iter, seed):
    print(f"\n{'='*64}\nTEST 1: SK bit-identity (monolithic vs full+PERSIST -> tail)\n{'='*64}")
    n_parties = len(party_files)
    all_encoded, common_ids, dict_size, partial_len, _, _ = se.prepare_nparty_data(
        party_files, max_cases)
    n_cases = len(common_ids)
    d_sq = dict_size * dict_size
    enc_off = enc_slab_offset(n_cases, d_sq, d_sq)
    print(f"  n_cases={n_cases} dict_size={dict_size} k={k} k_proj={k_proj} "
          f"enc_off={enc_off}")

    # Shared inputs and a public R, so the whole chain is deterministic.
    se.write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)
    r_matrix = se.generate_achlioptas_matrix(k_proj, d_sq, seed)
    se.write_projection_matrix(
        spdz_dir / "Programs" / "Source" / "projection_matrix.txt", r_matrix)

    # (a) One monolithic run.
    sph.clear_persistence(spdz_dir, n_parties)
    dfg_mono, sizes_mono = sk_extract_kmeans_aggregate(
        spdz_dir, n_cases, partial_len, dict_size, k_proj, k, max_iter, n_parties,
        sk_monolithic_args(n_cases, partial_len, dict_size, k_proj, n_parties))
    print(f"  [monolithic]      sizes={sizes_mono}")

    # (b) A full run with PERSIST_ENC creates the slab; persistence is not cleared before the tail.
    sph.clear_persistence(spdz_dir, n_parties)
    full_prog = se.compile_mpc(spdz_dir, "secure_extract",
                               sk_reuse_args(n_cases, partial_len, dict_size, k_proj,
                                             n_parties, enc_off, 0, 1),
                               n_parties=n_parties)
    se.run_mpc(spdz_dir, full_prog, n_parties=n_parties)
    print("  [full+PERSIST]    slab persisted (bigram@0, encoding@enc_off)")

    # (c) The tail run reads the slab, re-projects and persists Y.
    dfg_tail, sizes_tail = sk_extract_kmeans_aggregate(
        spdz_dir, n_cases, partial_len, dict_size, k_proj, k, max_iter, n_parties,
        sk_reuse_args(n_cases, partial_len, dict_size, k_proj, n_parties, enc_off, 2, 0))
    print(f"  [tail_only SK]    sizes={sizes_tail}")

    assert sizes_tail == sizes_mono, f"sizes differ: mono={sizes_mono} tail={sizes_tail}"
    mism = [(j, d) for j in range(k) for d in range(d_sq) if dfg_mono[j][d] != dfg_tail[j][d]]
    assert not mism, f"DFG differs at {mism[:5]} (monolithic vs tail_only)"
    print(f"  OK SK bit-identity: {k*d_sq} DFG cells + sizes IDENTICAL")
    return dfg_mono


def rsec_prepare(spdz_dir, party_files, max_cases, trunc, k_priv):
    """Truncate, write the inputs, and return the ground-truth backbone and the per-case bigram histograms."""
    n_parties = len(party_files)
    all_encoded, common_ids, dict_size, partial_len, _, _ = se.prepare_nparty_data(
        party_files, max_cases)
    all_encoded, partial_len = sph.truncate_encoded(all_encoded, trunc)
    n_cases = len(common_ids)
    se.write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)

    histograms, profiles = [], []
    for c in range(n_cases):
        merged = se.merge_events_plaintext(
            [all_encoded[p][c] for p in range(n_parties)], dict_size)
        h = se.extract_bigrams_plaintext(merged, dict_size)
        histograms.append(h)
        profiles.append(h)  # at the bigram encoding the profile is the histogram
    gt_backbone = sph.ground_truth_backbone(profiles, k_priv)
    return n_cases, dict_size, partial_len, histograms, gt_backbone


def rsec_tail_after_slab(spdz_dir, n_parties, n_cases, dict_size, partial_len, k_proj, k,
                         max_iter, k_priv, histograms, gt_backbone, label):
    """Run the secret-R tail on an already-persisted slab through relabel and aggregate, against the reference."""
    d_sq = dict_size * dict_size
    enc_off = enc_slab_offset(n_cases, d_sq, d_sq)
    # The reuse argument layout of src/mpc/rounds.rs::run_round1_rsecret.
    tail_args = [n_cases, partial_len, dict_size, 4, k_proj, n_parties, 0, 0,
                 k_priv, 2000, 0, 2, "0", 2, 0, enc_off, 0]
    prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", tail_args,
                          n_parties=n_parties)
    sph.write_hash_weights(spdz_dir, prog, d_sq)
    stdout, t1 = se.run_mpc(spdz_dir, prog, n_parties, timeout=600)
    raw, bb_flags = sph.parse_rows_with_backbone(stdout, n_cases, k_proj)
    y_scaled = sph.to_signed(raw)
    n_bb_gt = sum(gt_backbone)
    print(f"  [{label} tail rsec] {t1:.1f}s  backbone={sum(bb_flags)}/{n_cases} "
          f"(gt {n_bb_gt})")

    assert sum(bb_flags) == n_bb_gt, f"backbone {sum(bb_flags)} != gt {n_bb_gt}"
    for i, bb in enumerate(bb_flags):
        if not bb:
            assert not y_scaled[i].any(), f"suppressed row {i} not zeroed"

    # Cluster the backbone rows; the suppressed tail is assigned by nearest centroid.
    bb_idx = [i for i, b in enumerate(bb_flags) if b]
    Y_bb = [list(map(int, y_scaled[i])) for i in bb_idx]
    bb_labels, centroids_frac = lloyd_fractional(Y_bb, k=k, max_iter=max_iter)
    centroids_int = [[round(c) for c in row] for row in centroids_frac]
    shuffled_labels = [0] * n_cases
    for pos, lab in zip(bb_idx, bb_labels):
        shuffled_labels[pos] = lab

    pi_offset = n_cases * d_sq
    y_offset = pi_offset + n_cases
    bb_offset = y_offset + n_cases * k_proj
    label_offset = bb_offset + n_cases
    tail_idx = [i for i, b in enumerate(bb_flags) if not b]
    sph.write_relabel_inputs(spdz_dir, n_parties, shuffled_labels, centroids_int, tail_idx)
    relabel_args = [n_cases, pi_offset, 4, k, label_offset, 1, k_proj, y_offset, bb_offset]
    prog = se.compile_mpc(spdz_dir, "secure_relabel", relabel_args,
                          n_parties=n_parties)
    stdout2, _ = se.run_mpc(spdz_dir, prog, n_parties, timeout=300)
    mpc_case_labels = [None] * n_cases
    for line in stdout2.splitlines():
        if line.startswith("ASSIGN "):
            _, i, c = line.split()
            mpc_case_labels[int(i)] = int(c)
    assert all(v is not None for v in mpc_case_labels), "missing ASSIGN lines"

    agg_args = [n_cases, n_cases, d_sq, k, 1, label_offset]
    prog = se.compile_mpc(spdz_dir, "secure_aggregate", agg_args,
                          n_parties=n_parties)
    stdout3, _ = se.run_mpc(spdz_dir, prog, n_parties, timeout=300)
    mpc_dfg = se.parse_row_output(stdout3, k, d_sq)
    mpc_sizes = sph.derive_sizes_from_dfg(mpc_dfg, dict_size)

    ref_dfg, ref_sizes = sph.aggregate_plaintext(histograms, mpc_case_labels, k, dict_size)
    assert ref_sizes == mpc_sizes, f"sizes: ref {ref_sizes} vs mpc {mpc_sizes}"
    dfg_mismatch = [(j, d) for j in range(k) for d in range(d_sq)
                    if ref_dfg[j][d] != mpc_dfg[j][d]]
    assert not dfg_mismatch, f"DFG mismatch at {dfg_mismatch[:5]}"
    print(f"  OK {label}: backbone==gt, tail masked, {k*d_sq} DFG cells + sizes "
          "== all-case (backbone+tail) reference")
    return sum(bb_flags)


def test_rsec_correctness(spdz_dir, party_files, max_cases, k_proj, k, max_iter, trunc, k_priv):
    print(f"\n{'='*64}\nTEST 2: rsec correctness (full+PERSIST -> tail_only)\n{'='*64}")
    n_parties = len(party_files)
    n_cases, dict_size, partial_len, histograms, gt_backbone = rsec_prepare(
        spdz_dir, party_files, max_cases, trunc, k_priv)
    d_sq = dict_size * dict_size
    enc_off = enc_slab_offset(n_cases, d_sq, d_sq)
    assert sum(gt_backbone) < n_cases, "fixture has no suppression; lower --trunc"
    print(f"  n_cases={n_cases} dict_size={dict_size} k_priv={k_priv} "
          f"gt_backbone={sum(gt_backbone)} enc_off={enc_off}")

    # A full run with PERSIST_ENC creates the slab; its own revealed Y is discarded.
    sph.clear_persistence(spdz_dir, n_parties)
    full_args = [n_cases, partial_len, dict_size, 4, k_proj, n_parties, 0, 0,
                 k_priv, 2000, 0, 2, "0", 0, 1, enc_off, 0]
    full_prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", full_args,
                               n_parties=n_parties)
    sph.write_hash_weights(spdz_dir, full_prog, d_sq)
    se.run_mpc(spdz_dir, full_prog, n_parties, timeout=600)
    print("  [full+PERSIST rsec] slab persisted")

    # The tail run samples its own R and runs the hybrid chain on the slab.
    rsec_tail_after_slab(spdz_dir, n_parties, n_cases, dict_size, partial_len, k_proj, k,
                         max_iter, k_priv, histograms, gt_backbone, "test2")


def test_cross_mode(spdz_dir, party_files, max_cases, k_proj, k, max_iter, seed, trunc, k_priv):
    print(f"\n{'='*64}\nTEST 3: cross-mode (ONE slab -> SK tail + rsec tail)\n{'='*64}")
    n_parties = len(party_files)
    n_cases, dict_size, partial_len, histograms, gt_backbone = rsec_prepare(
        spdz_dir, party_files, max_cases, trunc, k_priv)
    d_sq = dict_size * dict_size
    enc_off = enc_slab_offset(n_cases, d_sq, d_sq)
    print(f"  n_cases={n_cases} dict_size={dict_size} enc_off={enc_off}")

    # A public R for the k-means side.
    r_matrix = se.generate_achlioptas_matrix(k_proj, d_sq, seed)
    se.write_projection_matrix(
        spdz_dir / "Programs" / "Source" / "projection_matrix.txt", r_matrix)

    # (a) A monolithic reference on the truncated data.
    sph.clear_persistence(spdz_dir, n_parties)
    dfg_ref, _ = sk_extract_kmeans_aggregate(
        spdz_dir, n_cases, partial_len, dict_size, k_proj, k, max_iter, n_parties,
        sk_monolithic_args(n_cases, partial_len, dict_size, k_proj, n_parties))

    # (b) One slab, which both tails read, so persistence is not cleared again.
    sph.clear_persistence(spdz_dir, n_parties)
    full_prog = se.compile_mpc(spdz_dir, "secure_extract",
                               sk_reuse_args(n_cases, partial_len, dict_size, k_proj,
                                             n_parties, enc_off, 0, 1),
                               n_parties=n_parties)
    se.run_mpc(spdz_dir, full_prog, n_parties=n_parties)
    print("  [full+PERSIST SK] one shared slab persisted")

    # (c) The k-means tail on the shared slab must reproduce the monolithic DFG.
    dfg_sk_tail, _ = sk_extract_kmeans_aggregate(
        spdz_dir, n_cases, partial_len, dict_size, k_proj, k, max_iter, n_parties,
        sk_reuse_args(n_cases, partial_len, dict_size, k_proj, n_parties, enc_off, 2, 0))
    mism = [(j, d) for j in range(k) for d in range(d_sq)
            if dfg_ref[j][d] != dfg_sk_tail[j][d]]
    assert not mism, f"cross-mode SK tail DFG differs from monolithic at {mism[:5]}"
    print("  OK SK tail DFG == monolithic reference (read the shared slab)")

    # (d) The secret-R tail on the same slab, which the k-means tail rewrote identically.
    rsec_tail_after_slab(spdz_dir, n_parties, n_cases, dict_size, partial_len, k_proj, k,
                         max_iter, k_priv, histograms, gt_backbone, "test3")
    print("  OK cross-mode: one slab fed BOTH an SK tail and an rsec tail")


def main():
    ap = hc.common_parser("Validation gate for --reuse-extraction")
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--k-proj", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-priv", type=int, default=2)
    ap.add_argument("--trunc", type=int, default=3)
    args = ap.parse_args()
    spdz_dir = hc.setup(args)

    data_dir = PROJECT_ROOT / "data" / "splits" / "sepsis_2party"
    party_files = [data_dir / "Party_A.xes", data_dir / "Party_B.xes"]
    missing = [f for f in party_files if not f.exists()]
    if missing:
        print(f"ERROR: missing party files: {missing}", file=sys.stderr)
        sys.exit(1)

    test_sk_bit_identity(spdz_dir, party_files, args.max_cases, args.k_proj, args.k,
                         args.max_iter, args.seed)
    test_rsec_correctness(spdz_dir, party_files, args.max_cases, args.k_proj, args.k,
                          args.max_iter, args.trunc, args.k_priv)
    test_cross_mode(spdz_dir, party_files, args.max_cases, args.k_proj, args.k,
                    args.max_iter, args.seed, args.trunc, args.k_priv)

    print("\nALL PASS")
    return True


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_extract_once", main))
