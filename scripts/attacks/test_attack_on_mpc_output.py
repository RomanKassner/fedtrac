"""The scoring rules of the attack battery: the guess, the oracle row alignment and the frequency match."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attack_on_mpc_output as a  # noqa: E402
from run_attack_battery import frequency_guessing_baseline  # noqa: E402


class TestFrequencyGuessingBaseline(unittest.TestCase):
    """The guess predicts the globally most frequent cells for every case and is scored like an attack."""

    def test_predicts_the_top_cells_for_every_case(self):
        X = np.zeros((10, 6))
        X[:, 1] = 1.0                       # present in every case
        X[:4, 4] = 1.0
        X[:2, 0] = 1.0
        p, r, top = frequency_guessing_baseline(X, top_k=2)
        self.assertEqual(top, [1, 4])
        self.assertAlmostEqual(p, 14 / 20)   # 10 + 4 hits of the 20 predicted cells
        self.assertAlmostEqual(r, 14 / 16)   # of the 16 nonzero cells


class TestOracleRowAlignment(unittest.TestCase):
    """The opened rows stand in the secret permutation's order: `anonymous_set` rebuilds what Round 1 opens
    and `oracle_metrics` scores a reconstruction in that order under the best row pairing."""

    def setUp(self):
        rng = np.random.default_rng(11)
        base = rng.integers(0, 3, size=(9, 12)).astype(float)      # nine distinct profiles
        base[0] = 0                                                # a zero row: withheld regardless
        mult = [4, 5, 3, 2, 1, 6, 1, 2, 3]                         # class sizes
        rows = [base[i] for i, m in enumerate(mult) for _ in range(m)]
        self.X = np.array(rows)                                    # 27 cases in case order
        self.k_priv = 3

    def test_anonymous_set_is_the_classes_at_or_above_the_threshold(self):
        A = a.anonymous_set(self.X, self.k_priv)
        # classes of size >= 3 that are nonzero: mult 5, 3, 6, 3 -> 17 rows; the zero class (4) is out
        self.assertEqual(len(A), 17)
        keys = [tuple(r) for r in self.X[A]]
        self.assertTrue(all(keys.count(k) >= self.k_priv for k in keys))
        self.assertTrue(all(any(v != 0 for v in k) for k in keys))

    def test_perfect_reconstruction_in_permuted_order_scores_one(self):
        A = a.anonymous_set(self.X, self.k_priv)
        rng = np.random.default_rng(3)
        Xh = self.X[A][rng.permutation(len(A))]                  # exact rows, opened order
        p, r, rows = a.oracle_metrics(self.X, A, Xh)
        self.assertEqual(rows, len(A))
        self.assertAlmostEqual(p, 1.0)
        s_open = (self.X[A] > 0).sum(); s_all = (self.X > 0).sum()
        self.assertAlmostEqual(r, s_open / s_all)                 # the withheld rows count as misses

    def test_positional_scoring_is_below_the_oracle_on_the_same_reconstruction(self):
        A = a.anonymous_set(self.X, self.k_priv)
        rng = np.random.default_rng(5)
        perm = rng.permutation(len(A))
        Xh = self.X[A][perm]
        p_pos, r_pos = a.metrics(self.X[A], Xh)                   # row i against row i
        p_or, r_or, _ = a.oracle_metrics(self.X, A, Xh)
        self.assertLess(p_pos, p_or)
        self.assertLess(r_pos, r_or * (self.X > 0).sum() / (self.X[A] > 0).sum() + 1e-9)

    def test_empty_reconstruction_scores_zero(self):
        A = a.anonymous_set(self.X, self.k_priv)
        p, r, _ = a.oracle_metrics(self.X, A, np.zeros((len(A), 12)))
        self.assertEqual((p, r), (0.0, 0.0))

    def test_sampling_above_the_cap_is_announced_and_bounded(self):
        # Above the cap both sides are sampled independently, so the score is an
        # approximation; the row count says so and the score stays a valid ratio.
        A = a.anonymous_set(self.X, self.k_priv)
        rng = np.random.default_rng(9)
        Xh = self.X[A][rng.permutation(len(A))]
        p, r, rows = a.oracle_metrics(self.X, A, Xh, max_rows=10)
        self.assertEqual(rows, 10)
        self.assertTrue(0.0 <= p <= 1.0 and 0.0 <= r <= 1.0)


class TestFrequencyMatch(unittest.TestCase):
    """The opened classes are ranked by size and matched with the profiles of the merged log ranked by count
    (the zero profile dropped); a tie keeps the lexicographic order on both sides."""

    def test_rank_match_withholding_and_tie_rule(self):
        prof = np.array([[3, 1, 0, 0, 2, 0],     # a: 10 cases
                         [0, 2, 1, 0, 0, 1],     # b: 6 cases, lexicographically before c
                         [1, 0, 0, 2, 0, 1],     # c: 6 cases, ties with b
                         [0, 0, 3, 1, 0, 0],     # d: 3 cases
                         [1, 1, 1, 0, 0, 0],     # e: 2 cases, withheld at k_priv 3
                         [0, 0, 0, 1, 1, 1]],    # f: 1 case, withheld
                        dtype=float)
        mult = [10, 6, 6, 3, 2, 1]
        rows = [prof[i] for i, m in enumerate(mult) for _ in range(m)] + [np.zeros(6)] * 8   # a zero profile of 8
        rng = np.random.default_rng(4)
        X = np.array(rows)[rng.permutation(36)]                          # case order
        A = a.anonymous_set(X, 3)
        self.assertEqual(len(A), 25)                                       # a, b, c, d; e, f and the zero rows withheld
        perm = rng.permutation(len(A))                                     # the secret shuffle
        R = rng.choice([-1.0, 1.0], size=(5, 6))

        def opened(sign):
            Z = X[A][perm] @ (sign * R).T
            return np.round(Z / np.linalg.norm(Z, axis=1, keepdims=True), 6)

        for sign in (1.0, -1.0):
            Y = opened(sign)
            Xh, info = a.attack_frequency_match(X, Y)
            self.assertEqual(sorted(info["sizes"].tolist()), [3, 6, 6, 10])
            truth = X[A][perm]
            for t in (prof[0], prof[3]):                                   # unique sizes: always right
                on = np.all(truth == t, axis=1)
                np.testing.assert_array_equal(Xh[on], truth[on])
            # the tie: the lexicographically first opened class of size 6 gets b, the first profile
            six = np.flatnonzero(info["sizes"] == 6)
            first = six[0]                                                 # classes are lexicographic
            self.assertTrue(np.array_equal(info["profiles"][info["match"][first]], prof[1]))
            b_first = bool(np.all(truth[info["class_of_row"] == first] == prof[1]))
            # the realised pairing, known here from the permutation
            true_profile = np.array([int(np.flatnonzero(np.all(info["profiles"] == truth[info["class_of_row"] == c][0],
                                                               axis=1))[0]) for c in range(len(info["sizes"]))])
            real = a.frequency_match_scores(info, 3, true_profile)
            self.assertEqual(real["class_acc"], 1.0 if b_first else 0.5)
            if sign == 1.0:
                b_first_plus = b_first
            else:
                self.assertNotEqual(b_first, b_first_plus)                 # -R reverses the tie
            # what the record proves: the two unique sizes; the tie counts as a miss
            sc = a.frequency_match_scores(info, 3)
            self.assertTrue(sc["sizes_ok"])
            self.assertEqual((sc["classes"], sc["tied_classes"], sc["hits"]), (4, 2, 2))
            self.assertAlmostEqual(sc["class_acc"], 0.5)
            self.assertAlmostEqual(sc["row_acc"], 13 / 25)
            # expectation over R: antithetic pairs make a tie of two exactly one half
            self.assertEqual(a.tie_rule_expectation(info, 5, draws=20), (1.0, 6.0))
            # the oracle alignment cannot see the tie: the rebuilt multiset is the true one
            p, r, n = a.oracle_metrics(X, A, Xh)
            self.assertEqual((p, n), (1.0, 25))
            self.assertAlmostEqual(r, (X[A] > 0).sum() / (X > 0).sum())
        # the zero profile (8 cases) is withheld whatever its count; kept, it takes rank two
        Xh0, info0 = a.attack_frequency_match(X, opened(1.0), drop_zero=False)
        self.assertEqual(int(np.all(Xh0 == 0, axis=1).sum()), 6)


if __name__ == "__main__":
    unittest.main()
