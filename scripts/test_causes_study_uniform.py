"""causes_study_uniform.py: the replay of the secure specification in the clear -- the stride seeding, the tape's
rounded centroid, the small-cluster merge, the clear algorithm against sklearn, the run's projection matrix, and the
row the table is built from."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from causes_study_uniform import (DEFAULT_VARIANTS, VARIANT, achlioptas, assign, build_table,  # noqa: E402
                                  centroid_matrix, kmeans_labels, lloyd, merge_small, random_rows, recip_bits,
                                  stride_rows)
from plaintext_kmeans_reference import lloyd_integer, merge_small_clusters_integer  # noqa: E402


def fixture(n, d, seed):
    return np.random.default_rng(seed).integers(-40, 40, size=(n, d)).astype(np.int64)


# ---------------------------------------------------------------- the seeding

def test_stride_seeding_is_the_tapes_init_centroids():
    # secure_kmeans.mpc::init_centroids: stride = N // K, seed j = row j * stride.
    assert stride_rows(12, 3) == [0, 4, 8]
    assert stride_rows(13, 3) == [0, 4, 8]
    assert stride_rows(1184, 6) == [0, 197, 394, 591, 788, 985]
    assert max(stride_rows(100, 7)) < 100


def test_ten_random_seedings_are_distinct_rows_and_reproducible():
    first = random_rows(1184, 6)
    assert len(first) == 10 and all(len(set(rows)) == 6 for rows in first)
    assert all(0 <= r < 1184 for rows in first for r in rows)
    assert random_rows(1184, 6) == first                      # random.Random(1)
    assert first[0] != first[1]


# ---------------------------------------------------------------- the tape's arithmetic

def test_rounded_centroid_equals_the_hand_computation_of_the_tape():
    # chat_j[d] = floor(s_j[d] * floor(2^g / n_j) / 2^(g - 16)) with g = 16 + ceil_log2(N) + 15.
    n_cases = 1000
    g = recip_bits(n_cases)
    assert g == 16 + 10 + 15                                  # ceil_log2(1000) = 10
    sums = np.array([[7, -7, 0], [1000, -3, 12345]], dtype=np.int64)
    counts = np.array([3, 7], dtype=np.int64)
    got = centroid_matrix(sums, counts, True, g)
    by_hand = [[(int(s) * ((1 << g) // int(c))) >> (g - 16) for s in row] for row, c in zip(sums, counts)]
    assert got.tolist() == by_hand
    # Three values written out: 7/3 at 16 fractional bits floors to 152917, -7/3 to -152918, 1000/7 to 9362285.
    assert by_hand[0][0] == 152917 and by_hand[0][1] == -152918
    assert by_hand[1][0] == 9362285
    assert abs(by_hand[0][0] / 2 ** 16 - 7 / 3) < 2 ** -15
    # The exact centroid is the plain mean, no shift.
    assert centroid_matrix(sums, counts, False, g).tolist() == [[7 / 3, -7 / 3, 0.0], [1000 / 7, -3 / 7, 12345 / 7]]


def test_the_rounded_centroid_floors_toward_minus_infinity_like_the_tapes_trunc():
    g = recip_bits(16)
    for s in (-5, -1, 1, 5):
        got = int(centroid_matrix(np.array([[s]], dtype=np.int64), np.array([3], dtype=np.int64), True, g)[0][0])
        assert got == (s * ((1 << g) // 3)) >> (g - 16)
        assert got <= s * 2 ** 16 / 3


def test_the_e_step_breaks_a_tie_on_the_first_index():
    Y = np.array([[0]], dtype=np.int64)
    chat = np.array([[-65536], [65536], [-65536]], dtype=np.int64)     # equidistant at scale 2^16
    assert assign(Y, chat, True).tolist() == [0]
    assert assign(np.array([[0.0]]), np.array([[-1.0], [1.0]]), False).tolist() == [0]


def test_the_e_step_refuses_a_distance_past_the_tapes_ring_bound():
    # A value at or above 2^62 is a run parameters.rs::secure_kmeans_argmin_bits refuses, so the replay refuses it
    # rather than wrapping the ring in silence.
    big = np.array([[2 ** 24, 2 ** 24]], dtype=np.int64)        # y * 2^16 = 2^40 per coordinate
    with pytest.raises(OverflowError):
        assign(big, np.array([[-(2 ** 40), -(2 ** 40)]], dtype=np.int64), True)
    # The same shape inside the bound goes through, and the distance is the plain integer one.
    small = np.array([[3, 4]], dtype=np.int64)
    chat = np.array([[0, 0]], dtype=np.int64)
    assert assign(small, chat, True).tolist() == [0]


def test_lloyd_reproduces_the_plaintext_twin_of_secure_kmeans_bit_for_bit():
    for seed in range(8):
        Y = fixture(40 + seed, 5, 100 + seed)
        for iters in (1, 3, 5, 10):
            want = lloyd_integer(Y.tolist(), 3, iters)[0]
            got, _, _ = lloyd(Y, 3, iters, stride_rows(len(Y), 3), True)
            assert got.tolist() == want, (seed, iters)


def test_an_emptied_cluster_keeps_its_previous_centroid_as_the_m_step_does():
    # Two points, three clusters: cluster 2 is seeded on row 1 and emptied at once, and its centroid must stay a
    # valid argmin target rather than collapse to the origin (secure_kmeans.mpc::m_step).
    Y = np.array([[0], [1000], [1000]], dtype=np.int64)
    labels, chat, _ = lloyd(Y, 3, 5, [0, 1, 2], True)
    assert labels.tolist() == lloyd_integer(Y.tolist(), 3, 5)[0]
    assert int(chat[2][0]) != 0


# ---------------------------------------------------------------- the merge

def test_merge_moves_every_small_cluster_to_the_nearest_large_one():
    labels = np.array([0] * 12 + [1] * 3 + [2] * 11, dtype=np.int64)
    chat = np.array([[0], [30], [100]], dtype=np.int64)
    got = merge_small(labels, chat, 3, 10)
    assert got.tolist() == [0] * 15 + [2] * 11                # the three cases of cluster 1 join cluster 0
    assert got.tolist() == merge_small_clusters_integer(labels.tolist(), chat.tolist(), 3, 10)


def test_nothing_moves_when_no_cluster_reaches_the_threshold():
    labels = np.array([0] * 4 + [1] * 5, dtype=np.int64)
    chat = np.array([[0], [30]], dtype=np.int64)
    assert merge_small(labels, chat, 2, 10).tolist() == labels.tolist()


def test_the_merge_ties_to_the_first_index():
    labels = np.array([0] * 10 + [1] * 3 + [2] * 10, dtype=np.int64)
    chat = np.array([[0], [50], [100]], dtype=np.int64)       # cluster 1 is equidistant from 0 and 2
    assert set(merge_small(labels, chat, 3, 10).tolist()) == {0, 2}
    assert merge_small(labels, chat, 3, 10).tolist()[10] == 0


# ---------------------------------------------------------------- the clear algorithm

def test_the_clear_variant_is_lloyds_and_equals_sklearn_from_the_same_seeding():
    from sklearn.cluster import KMeans
    rows = np.random.default_rng(3).normal(size=(200, 4))
    init = stride_rows(200, 4)
    mine, _, wcss = lloyd(rows, 4, 50, init, False)
    theirs = KMeans(4, init=rows[init], n_init=1, max_iter=50, tol=0).fit(rows)
    assert mine.tolist() == theirs.labels_.tolist()
    assert wcss == pytest.approx(theirs.inertia_, rel=1e-9)


def test_v5_keeps_the_lowest_wcss_of_the_ten_seedings_and_merges_nothing():
    rows = np.random.default_rng(5).normal(size=(120, 3))
    labels, _, tag = kmeans_labels(rows, 4, VARIANT["v5_clear_algo"], 0)
    best = min(lloyd(rows, 4, 10, pick, False)[2] for pick in random_rows(120, 4))
    assert lloyd(rows, 4, 10, random_rows(120, 4)[int(tag[1:])], False)[2] == pytest.approx(best)
    assert sorted(set(labels.tolist())) == list(range(len({*labels.tolist()})))


def test_every_default_variant_names_one_change_against_the_specification():
    spec = VARIANT["v0_spec"]
    assert spec == dict(seeds="stride", iters=5, rounding=True, merge=True, bound="run", normalise=False)
    changed = {v: [k for k in spec if VARIANT[v][k] != spec[k]] for v in DEFAULT_VARIANTS}
    assert changed["v0_spec"] == []
    assert [changed[v] for v in ("v1_ten_seeds", "v2_ten_iter", "v3_no_merge", "v4_no_rounding", "v6_no_bound")] == \
           [["seeds"], ["iters"], ["merge"], ["rounding"], ["bound"]]
    assert set(changed["v5_clear_algo"]) == {"seeds", "iters", "merge", "rounding"}
    assert set(changed["v7_clear_normalised"]) == {"seeds", "iters", "merge", "rounding", "normalise"}
    assert all(VARIANT[f"v1_seed{j}"]["seeds"] == ("one", j) for j in range(10))


# ---------------------------------------------------------------- the run's projection matrix

RECORDED_MATRICES = sorted((REPO / "output" / "runs").glob("*/projection_matrix.txt"))


@pytest.mark.skipif(not RECORDED_MATRICES, reason="no projection_matrix.txt survives in output/runs")
def test_achlioptas_matches_the_recorded_projection_matrix():
    # src/pipeline/mpc/mod.rs writes AchlioptasMatrix::generate(k_proj, enc_dim, 42) beside a Secret-Labels run;
    # the identity passthrough has a 1 on its diagonal and is not a draw.
    checked = 0
    for path in RECORDED_MATRICES:
        lines = path.read_text().split("\n")
        k_proj, d = (int(v) for v in lines[0].split())
        ref = np.array([[int(v) for v in line.split()] for line in lines[1:1 + k_proj]], dtype=np.int8)
        if k_proj == d and (np.diag(ref) == 1).all():
            continue
        assert achlioptas(k_proj, d).tolist() == ref.tolist(), path
        checked += 1
    assert checked, "every surviving matrix is the identity passthrough"


def test_achlioptas_is_sparse_signed_and_reproducible():
    r = achlioptas(25, 64)
    assert r.shape == (25, 64) and set(np.unique(r)) <= {-1, 0, 1}
    assert r.tolist() == achlioptas(25, 64).tolist()
    assert achlioptas(25, 64, seed=43).tolist() != r.tolist()
    big = achlioptas(40, 400)
    assert 0.6 < (big == 0).mean() < 0.73                     # 2/3 zeros by construction


# ---------------------------------------------------------------- the table

def test_build_table_pairs_every_variant_with_v0_and_with_the_recorded_values(tmp_path, monkeypatch):
    import causes_study_uniform as cs
    monkeypatch.setattr(cs, "OUT", tmp_path)
    for sub in ("cache", "labels", "results"):
        (tmp_path / sub).mkdir()
    meta = dict(params=dict(log_name="sepsis", embedding="bigram", method="km", cases=927, k=6, notes=[]),
                dict_size=19, dim=361, dh=19,
                recorded={"federated": {"f_score": "0.6816", "fitness": "0.97", "precision": "0.53",
                                        "clusters": "6"},
                          "plaintext": {"f_score": "0.8000", "fitness": "0.99", "precision": "0.68",
                                        "clusters": "6"}})
    (tmp_path / "cache" / "sepsis-bigram_sl_km.json").write_text(json.dumps(meta))
    for variant, f in (("v0_spec", 0.6816), ("v5_clear_algo", 0.7900)):
        (tmp_path / "results" / f"sepsis-bigram_sl_km__{variant}.json").write_text(json.dumps(dict(
            cell="sepsis-bigram_sl_km", variant=variant, count=6, sizes=[500, 427], seed_used="stride",
            ari_to_run=1.0, note="",
            summary=dict(clusters=6, cases=927, models_scored=6, fitness=0.97, precision=0.53, f_score=f,
                         simplicity=0.5, edges=33.0, activities=16, rel=20.0))))
    rows = cs.build_table(["sepsis-bigram_sl_km"], ["v0_spec", "v5_clear_algo"])
    assert [r["variant"] for r in rows] == ["v0_spec", "v5_clear_algo"]
    by = {r["variant"]: r for r in rows}
    assert by["v0_spec"]["d_f_vs_v0"] == 0.0
    assert by["v5_clear_algo"]["d_f_vs_v0"] == 0.1084
    assert by["v5_clear_algo"]["d_f_vs_plaintext"] == -0.01
    assert by["v0_spec"]["v0_minus_run_f"] == 0.0 and by["v5_clear_algo"]["v0_minus_run_f"] == 0.0
    assert by["v0_spec"]["run_plaintext_f_score"] == 0.8
    assert by["v0_spec"]["sizes"] == "500;427"
    assert (tmp_path / "results.csv").exists()
    assert list(csv_header(tmp_path / "results.csv")) == cs.COLUMNS


def csv_header(path):
    import csv
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def test_a_variant_without_a_result_leaves_the_table_without_a_row(tmp_path, monkeypatch):
    import causes_study_uniform as cs
    monkeypatch.setattr(cs, "OUT", tmp_path)
    for sub in ("cache", "labels", "results"):
        (tmp_path / sub).mkdir()
    assert cs.build_table(["nothing_here"], list(DEFAULT_VARIANTS)) == []
