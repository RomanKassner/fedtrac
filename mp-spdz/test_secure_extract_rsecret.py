#!/usr/bin/env python3
"""Correctness test for secure_extract_rsecret.mpc and secure_relabel.mpc: shape, geometry, un-shuffle, suppression."""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# constants.py is loaded directly, since mpc_utils/__init__.py imports MP-SPDZ's Compiler.
import importlib.util  # noqa: E402

import harness_common as hc  # noqa: E402
import test_secure_extract as se  # noqa: E402

_constants_path = HERE / "mpc_utils" / "constants.py"
_spec = importlib.util.spec_from_file_location("rsecret_constants", _constants_path)
_const_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_const_mod)
NORM_SCALE = _const_mod.NORM_SCALE


def parse_assign_output(stdout, n):
    """Parse the ASSIGN lines of secure_relabel's output."""
    labels = [None] * n
    for line in stdout.splitlines():
        if line.startswith("ASSIGN "):
            parts = line.split()
            if len(parts) == 3:
                j, c = int(parts[1]), int(parts[2])
                if 0 <= j < n:
                    labels[j] = c
    missing = [i for i, v in enumerate(labels) if v is None]
    if missing:
        raise AssertionError(f"secure_relabel missing {len(missing)} ASSIGN lines: {missing[:5]}")
    return labels


def multiset_distance_close(a: np.ndarray, b: np.ndarray, rel_tol=0.25):
    """Compare two row sets' pairwise distance multisets, ranked."""
    from scipy.spatial.distance import pdist
    da = np.sort(pdist(a))
    db = np.sort(pdist(b))
    if len(da) != len(db):
        return float("inf"), 0.0
    denom = np.maximum(da, 1e-9)
    rel = np.abs(da - db) / denom
    return float(rel.max()), float((rel < rel_tol).mean())


def test_runlength_scan_schedule(spdz_dir, n=2200):
    """Compile-only: in each prefix-doubling scan array every load is scheduled after the previous round's stores."""
    print(f"\n{'=' * 60}\nTEST: run-length scan schedule (compile-only, N={n})\n"
          f"{'=' * 60}")
    # A Helpdesk-shaped activity cell.
    args = [n, 8, 17, 8, 17, 2, 3, 17, 5, 2000, 1, 2, 0]
    prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", args,
                          n_parties=2)

    sys.path.insert(0, str(spdz_dir))
    import os
    cwd = os.getcwd()
    os.chdir(spdz_dir)
    try:
        from Compiler.program import Tape

        def to_int(a):
            try:
                return int(a)
            except (TypeError, ValueError):
                try:
                    return int(a.i)
                except AttributeError:
                    return None

        strides = []
        d = 1
        while d < n:
            strides.append(d)
            d *= 2
        size_to_stride = {n - d: d for d in strides}
        ops = []
        for i, inst in enumerate(Tape.read_instructions(f"{prog}-0")):
            nm = inst.type.__name__
            if nm not in ("ldms", "stms"):
                continue
            if inst.size != n and inst.size not in size_to_stride:
                continue
            addr = to_int(inst.args[1])
            if addr is None:
                continue
            ops.append((i, nm, inst.size, addr))
        bases = sorted({a for _, _, s, a in ops if s == n})

        def resolve(size, addr):
            if size == n:
                return addr
            d = size_to_stride[size]
            cands = [b for b in bases if addr == b or addr == b + d]
            return cands[0] if len(cands) == 1 else None

        per_array = {}
        for idx, kind, size, addr in ops:
            b = resolve(size, addr)
            if b is None:
                continue
            stride = 0 if size == n else size_to_stride[size]
            per_array.setdefault(b, []).append((idx, kind, stride))

        violations = []
        scan_arrays = 0
        for b, arr in per_array.items():
            if len({s for _, _, s in arr if s >= 1}) < 5:
                continue
            scan_arrays += 1
            max_store = {}
            for idx, kind, stride in arr:
                if kind == "stms" and stride >= 1:
                    max_store[stride] = max(max_store.get(stride, -1), idx)
            for idx, kind, stride in arr:
                if kind == "ldms" and stride >= 2:
                    prev = stride // 2
                    if prev in max_store and idx < max_store[prev]:
                        violations.append((b, idx, stride, max_store[prev]))
    finally:
        os.chdir(cwd)

    assert scan_arrays >= 4, (
        f"expected the four scan arrays (cntf/coefa/sel/rl), found "
        f"{scan_arrays} - the size heuristic no longer matches the tape"
    )
    assert not violations, (
        f"{len(violations)} hoisted scan loads (round-d load scheduled before "
        f"round-d/2 stores): {violations[:5]} - the allocator reordered the "
        f"run-length scan again; re-pin with break_point/protect_memory"
    )
    print(f"  OK schedule: {scan_arrays} scan arrays, 0 hoisted loads")


def _write_p0_values(spdz_dir, values, n_parties=2):
    """One integer per line as party-0 input."""
    hc.write_party0_values(spdz_dir, values)


def _seed_persistence(spdz_dir, values, offset, n_parties=2):
    """Persist known values at a persistence offset via the seed fixture."""
    _write_p0_values(spdz_dir, values, n_parties)
    prog = se.compile_mpc(spdz_dir, "seed_persistence", [len(values), offset],
                          n_parties=n_parties)
    se.run_mpc(spdz_dir, prog, n_parties, timeout=120)


def _run_mpc_expect_crash(spdz_dir, prog_name, n_parties, needle, timeout=120):
    """Run a program that must crash, asserting a nonzero exit and the expected message."""
    import subprocess
    procs = []
    for p in range(n_parties):
        cmd = [str(Path(spdz_dir) / "semi2k-party.x"),
               "-p", str(p), "-N", str(n_parties), prog_name]
        procs.append(subprocess.Popen(cmd, cwd=spdz_dir,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE))
    outs, rcs = [], []
    for proc in procs:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            for pp in procs:
                pp.kill()
            raise
        outs.append(stdout.decode() + stderr.decode())
        rcs.append(proc.returncode)
    combined = "\n".join(outs)
    assert any(rc != 0 for rc in rcs), (
        f"expected the perfect-key guard to crash the run, exits={rcs}")
    assert needle in combined, (
        f"guard crashed without the expected message {needle!r}:\n"
        + combined[:800])


def test_relabel_reveal_sort_identity(spdz_dir, n):
    """A known pi pins the scatter direction of the un-shuffle, which permutation-ness alone cannot."""
    print(f"\n{'=' * 60}\nTEST: relabel reveal_sort identity (known pi, N={n})\n"
          f"{'=' * 60}")
    assert n > 1000, "shape must exceed the allocator's vector budget"
    rng = np.random.default_rng(20260823)
    pi = rng.permutation(n)
    labels = rng.integers(0, 7, size=n)

    _seed_persistence(spdz_dir, pi, 0)
    _write_p0_values(spdz_dir, labels)
    # K=0 forces the ASSIGN reveal; pi was seeded at offset 0.
    prog = se.compile_mpc(spdz_dir, "secure_relabel", [n, 0, 4, 0, 0, 1],
                          n_parties=2)
    stdout, _ = se.run_mpc(spdz_dir, prog, 2, timeout=300)
    l_original = parse_assign_output(stdout, n)
    bad = [i for i in range(n) if l_original[pi[i]] != labels[i]]
    assert not bad, (
        f"un-shuffle scatter broke value identity at {len(bad)} of {n} "
        f"positions (first: i={bad[0]}, pi[i]={pi[bad[0]]}, "
        f"expected {labels[bad[0]]}, got {l_original[pi[bad[0]]]})")
    print(f"  OK l[pi[i]] == labels[i] for all {n} positions")


def test_relabel_perfect_key_guard(spdz_dir, n=64):
    """The guard must crash on a corrupted pi: a duplicate key or an out-of-range key."""
    print(f"\n{'=' * 60}\nTEST: perfect-key guard crashes (N={n})\n{'=' * 60}")
    rng = np.random.default_rng(7)
    labels = list(range(n))

    pi_dup = list(rng.permutation(n))
    pi_dup[1] = pi_dup[0]
    _seed_persistence(spdz_dir, pi_dup, 0)
    _write_p0_values(spdz_dir, labels)
    prog = se.compile_mpc(spdz_dir, "secure_relabel", [n, 0, 4, 0, 0, 1],
                          n_parties=2)
    _run_mpc_expect_crash(spdz_dir, prog, 2, "FATAL reveal_sort")
    print("  OK duplicate key crashed with the FATAL banner")

    pi_range = list(rng.permutation(n))
    pi_range[0] = n + 7
    _seed_persistence(spdz_dir, pi_range, 0)
    _write_p0_values(spdz_dir, labels)
    _run_mpc_expect_crash(spdz_dir, prog, 2, "FATAL reveal_sort")
    print("  OK out-of-range key crashed with the FATAL banner")


def test_shuffle_instruction_counts(spdz_dir):
    """Compile-only: the exact shuffle-instruction counts, which a radix fallback at either sort site would raise."""
    print(f"\n{'=' * 60}\nTEST: shuffle instruction counts (compile-only)\n"
          f"{'=' * 60}")
    relabel = se.compile_mpc(spdz_dir, "secure_relabel", [1500, 0, 4, 0, 0, 1],
                             compile_flags=[])
    rsec_args = [64, 4, 6, 4, 5, 2, 0, 0, 5, 2000, 0, 2]
    rsec_radix = se.compile_mpc(spdz_dir, "secure_extract_rsecret", rsec_args,
                                compile_flags=[])
    counts_radix = _count_shuffles(spdz_dir, rsec_radix)
    rsec_batcher = se.compile_mpc(spdz_dir, "secure_extract_rsecret",
                                  rsec_args, compile_flags=["-Z", "2"])
    counts_batcher = _count_shuffles(spdz_dir, rsec_batcher)
    counts_relabel = _count_shuffles(spdz_dir, relabel)

    assert counts_relabel == {"gensecshuffle": 1, "applyshuffle": 2,
                              "delshuffle": 1}, f"secure_relabel: {counts_relabel}"
    assert counts_radix == {"gensecshuffle": 4, "applyshuffle": 7,
                            "delshuffle": 4, "secshuffle": 1}, \
        f"rsecret no-Z (radix fallback): {counts_radix}"
    assert counts_batcher == {"gensecshuffle": 1, "applyshuffle": 2,
                              "delshuffle": 1, "secshuffle": 1}, \
        f"rsecret -Z 2 (Batcher): {counts_batcher}"
    print("  OK relabel 1/2/1; rsecret radix 4/7/4+1; rsecret Batcher 1/2/1+1")


def test_cascade_cost(spdz_dir):
    """Compile-only: the exact integer-triple cost of the pruned demux cascades, which an output test cannot see."""
    print(f"\n{'=' * 60}\nTEST: demux cascade cost (compile-only)\n{'=' * 60}")
    import re
    import subprocess

    def triple_count(args):
        se.compile_mpc(spdz_dir, "secure_extract_rsecret", args,
                       compile_flags=[])
        out = subprocess.run(
            ["python3", "./compile.py", "-R", "64", "secure_extract_rsecret"]
            + [str(a) for a in args],
            cwd=spdz_dir, capture_output=True, text=True, timeout=600).stdout
        m = re.search(r"(\d+) integer triples", out)
        assert m, "compile.py printed no integer-triple requirement"
        return int(m.group(1))

    # Trigram at d=9, so the user cube is 216.
    got = triple_count([4, 4, 9, 4, 5, 2, 4, 216, 1, 2000, 0, 2])
    assert got == 29062, (
        f"trigram tape requires {got} integer triples, pinned 29062 — a "
        "cascade/dim regression, or a deliberate cost change that must "
        "re-pin this value")
    # Skip-bigram at d=6, so the width is 72.
    got = triple_count([4, 4, 6, 4, 5, 2, 5, 72, 1, 2000, 0, 2])
    assert got == 39138, (
        f"skip-bigram tape requires {got} integer triples, pinned 39138 — "
        "a cascade regression, or a deliberate cost change that must "
        "re-pin this value")
    print("  OK trigram 29062, skip-bigram 39138 integer triples (pruned)")


def _count_shuffles(spdz_dir, prog):
    import os
    sys.path.insert(0, str(spdz_dir))
    cwd = os.getcwd()
    os.chdir(spdz_dir)
    try:
        from Compiler.program import Tape
        c = {}
        for inst in Tape.read_instructions(f"{prog}-0"):
            nm = inst.type.__name__
            if "shuffle" in nm.lower():
                c[nm] = c.get(nm, 0) + 1
        return c
    finally:
        os.chdir(cwd)


def test_backmap_identity(spdz_dir, n=2048, k_proj=5):
    """The revealed class multiplicities pin the back-map: a wrong scatter would move them."""
    print(f"\n{'=' * 60}\nTEST: back-map identity (known classes, N={n})\n"
          f"{'=' * 60}")
    assert n > 1000, "shape must exceed the allocator's vector budget"
    dict_size = 6
    partial_len = 4
    tmpl_a = [(3, 1), (4, 2), (5, 3)]          # n-9 cases, anonymous
    tmpl_b = [(3, 1), (5, 2), (4, 3)]          # 5 cases, anonymous
    tmpl_c = [(4, 1), (3, 2)]                  # 3 cases, suppressed
    tmpl_d = [(5, 1), (5, 2), (4, 3), (3, 4)]  # a singleton, suppressed

    def pad(t):
        return t + [(0, 0)] * (partial_len - len(t))
    cases = ([pad(tmpl_a)] * (n - 9) + [pad(tmpl_b)] * 5
             + [pad(tmpl_c)] * 3 + [pad(tmpl_d)])
    empty = pad([])
    all_encoded = [cases, [empty for _ in cases]]

    se.write_nparty_inputs(spdz_dir, all_encoded, 2, n, partial_len)
    prog = se.compile_mpc(
        spdz_dir, "secure_extract_rsecret",
        [n, partial_len, dict_size, 4, k_proj, 2, 0, 0, 5, 2000, 0, 2],
        n_parties=2)
    # The fingerprint weights ride the runtime Public-Input channel.
    import test_secure_pipeline_hybrid as hyb
    hyb.write_hash_weights(spdz_dir, prog, dict_size * dict_size)
    stdout, _ = se.run_mpc(spdz_dir, prog, 2, timeout=1800)

    assert "K_PRIV_SELECTED 5" in stdout, "pinned k_priv must be 5"
    nb = [int(x.split()[1]) for x in stdout.splitlines()
          if x.startswith("N_BACKBONE")]
    assert nb == [n - 4], f"N_BACKBONE {nb} != {n - 4}"

    raw = np.array(se.parse_row_output(stdout, n, k_proj), dtype=np.uint64)
    y = np.where(raw >= 2**63, raw.astype(np.float64) - 2**64,
                 raw.astype(np.float64)) / NORM_SCALE
    nz = y[np.abs(y).sum(axis=1) > 1e-6]
    assert len(nz) == n - 4, f"{len(nz)} nonzero rows, expected {n - 4}"
    # Group by near-equality: one profile gives one unit row, distinct profiles distinct rows under the secret R.
    groups = []
    for row in nz:
        for g in groups:
            if np.abs(row - g[0]).max() < 1e-3:
                g[1] += 1
                break
        else:
            groups.append([row, 1])
    sizes = sorted(g[1] for g in groups)
    assert sizes == [5, n - 9], (
        f"revealed class multiplicities {sizes} != [5, {n - 9}]: the "
        "back-map assigned backbone bits to the wrong cases")
    print(f"  OK backbone = two classes with multiplicities {sizes}, "
          f"{n - 4} rows revealed, 4 suppressed")


def test_shape_and_relabel(spdz_dir, party_files, max_cases=10, k_proj=4):
    """The extract and relabel round-trip on tiny data."""
    print(f"\n{'=' * 60}\nTEST: shape + relabel round-trip\n{'=' * 60}")
    print(f"  max_cases={max_cases}, k_proj={k_proj}")

    all_encoded, common_ids, dict_size, partial_len, _, _ = (
        se.prepare_nparty_data(party_files, max_cases)
    )
    n_cases = len(common_ids)
    n_parties = len(party_files)

    se.write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_len)
    extract_args = [n_cases, partial_len, dict_size, 4, k_proj, n_parties, 0, 0]
    prog = se.compile_mpc(spdz_dir, "secure_extract_rsecret", extract_args,
                          n_parties=n_parties)
    stdout, _ = se.run_mpc(spdz_dir, prog, n_parties, timeout=600)

    # A negative cell is opened as an unsigned ring element; each cell is the unit value times NORM_SCALE.
    raw = np.array(se.parse_row_output(stdout, n_cases, k_proj), dtype=np.int64)
    y_shuffled = np.array(
        [[(int(v) - 2**64) if int(v) >= 2**63 else int(v) for v in row] for row in raw],
        dtype=np.float64,
    ) / NORM_SCALE
    assert y_shuffled.shape == (n_cases, k_proj), \
        f"Shape mismatch: got {y_shuffled.shape}, expected {(n_cases, k_proj)}"
    print(f"  OK shape: Y_shuffled is {y_shuffled.shape}")

    # The clear reference is the unprojected X, R being secret; the projection preserves the pairwise distances.
    d_sq = dict_size * dict_size
    X_rows = []
    for c in range(n_cases):
        party_events = [all_encoded[p][c] for p in range(n_parties)]
        merged = se.merge_events_plaintext(party_events, dict_size)
        X_rows.append(se.extract_bigrams_plaintext(merged, dict_size))
    X = np.array(X_rows, dtype=np.float64)
    print(f"  X shape: {X.shape}, nnz/row mean: {(X > 0).sum(1).mean():.1f}")

    # Every row has unit norm modulo quantisation, which catches a wrong NORM_PRESHIFT or a mishandled sign.
    max_abs = float(np.abs(y_shuffled).max())
    norms = np.linalg.norm(y_shuffled, axis=1)
    print(
        f"  Y cell range: max|Y|={max_abs:.4f}, "
        f"row-norm mean={norms.mean():.4f} std={norms.std():.4f}"
    )
    assert max_abs < 2.0, (
        f"Y cells exceed unit-norm bound ({max_abs} >= 2); MPC normalisation "
        "miscompiled or NORM_SCALE drift between Python and MPC."
    )
    nonzero_rows = (X > 0).any(axis=1)
    if nonzero_rows.any():
        active_norms = norms[nonzero_rows]
        # The tolerance is well above the quantisation error and still catches real drift.
        assert np.all(np.abs(active_norms - 1.0) < 0.05), (
            f"Active rows are not unit-norm: norms={active_norms.tolist()}"
        )
    print("  OK Y rows have unit L2 norm within quantisation tolerance")

    # Feeding the identity as the shuffled labels, the un-shuffled labels must be a permutation.
    pi_offset = n_cases * d_sq
    labels_input = list(range(n_cases))
    with open(Path(spdz_dir) / "Player-Data" / "Input-P0-0", "w") as f:
        for v in labels_input:
            f.write(f"{v}\n")
    for p in range(1, n_parties):
        (Path(spdz_dir) / "Player-Data" / f"Input-P{p}-0").write_text("")

    # K=0 writes no one-hot and REVEAL_LABELS=1 opens the labels for the check; a pipeline run does neither.
    relabel_args = [n_cases, pi_offset, 4, 0, 0, 1]
    prog2 = se.compile_mpc(spdz_dir, "secure_relabel", relabel_args,
                           n_parties=n_parties)
    stdout2, _ = se.run_mpc(spdz_dir, prog2, n_parties, timeout=120)

    l_original = parse_assign_output(stdout2, n_cases)
    # Permutation-ness is the strongest external check, pi itself being secret in a pipeline run.
    assert sorted(l_original) == labels_input, (
        f"secure_relabel did not produce a permutation: "
        f"sorted={sorted(l_original)}, expected={labels_input}"
    )
    print(f"  OK secure_relabel produced a valid permutation of [0, {n_cases})")

    print("\n  All assertions passed.")


def main():
    ap = hc.common_parser("Correctness test for secure_extract_rsecret.mpc")
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--k-proj", type=int, default=4)
    args = ap.parse_args()
    args.spdz_dir = hc.setup(args)

    data_dir = HERE.parent / "data" / "splits" / "sepsis_2party"
    party_files = [data_dir / "Party_A.xes", data_dir / "Party_B.xes"]
    if not all(p.exists() for p in party_files):
        print(f"Sepsis split files missing under {data_dir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    test_runlength_scan_schedule(args.spdz_dir)
    test_shuffle_instruction_counts(args.spdz_dir)
    test_cascade_cost(args.spdz_dir)
    test_shape_and_relabel(
        args.spdz_dir, party_files, max_cases=args.max_cases, k_proj=args.k_proj,
    )
    test_relabel_reveal_sort_identity(args.spdz_dir, n=2048)
    test_relabel_reveal_sort_identity(args.spdz_dir, n=1500)
    test_relabel_perfect_key_guard(args.spdz_dir)
    # Both fingerprint-sort paths above the allocator budget: a power of two and the out-of-range branch.
    test_backmap_identity(args.spdz_dir, n=2048)
    test_backmap_identity(args.spdz_dir, n=1500)
    print(f"\nTotal: {time.time() - t0:.1f}s")
    return True


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_extract_rsecret", main))
