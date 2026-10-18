#!/usr/bin/env python3
"""End-to-end test for secure_extract.mpc: every projected row must equal its clear twin exactly."""

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "bridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pm4py

import harness_common as hc  # noqa: E402
# Re-exported, since the other harnesses import these names from here.
from harness_common import (  # noqa: E402,F401
    END_ACT,
    FIRST_USER_ACT,
    PAD_ACT,
    PAD_TS,
    START_ACT,
    TRACE_LEN_ALIGNMENT,
    clear_persistence,
    compile_mpc,
    derive_partial_lens,
    retruncate_per_party,
    run_mpc,
    write_nparty_inputs,
)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def load_xes_cases(path):
    """Load an XES file as case id -> [(timestamp, activity), ...]."""
    df = pm4py.read_xes(str(path))
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    cases = defaultdict(list)
    for _, row in df.iterrows():
        cid = str(row["case:concept:name"])
        ts = int(row["time:timestamp"].timestamp() * 1000)
        act = str(row["concept:name"])
        cases[cid].append((ts, act))
    return dict(cases)


def build_dictionary(all_party_cases):
    """Build the activity dictionary from every party's data."""
    activities = set()
    for party_cases in all_party_cases:
        for events in party_cases.values():
            for _, act in events:
                activities.add(act)
    sorted_acts = sorted(activities)
    # PAD, START and END are reserved, so the user activities start above them.
    act_to_id = {}
    for i, act in enumerate(sorted_acts):
        act_to_id[act] = FIRST_USER_ACT + i
    return act_to_id, FIRST_USER_ACT + len(sorted_acts)


def encode_cases(party_cases, common_ids, act_to_id, partial_len, min_ts):
    """Encode one party's cases as [(activity id, relative timestamp), ...]."""
    encoded = []
    for cid in common_ids:
        events = party_cases.get(cid, [])
        enc = []
        for ts, act in events:
            aid = act_to_id.get(act, PAD_ACT)
            ts_rel = max(0, (ts - min_ts) // 1000)
            enc.append((aid, ts_rel))
        # The composite key of the merge tree: sorting on the timestamp alone would break its sorted-input invariant.
        enc.sort(key=lambda x: (x[1], x[0]))
        enc = enc[:partial_len]
        while len(enc) < partial_len:
            enc.append((PAD_ACT, PAD_TS))
        encoded.append(enc)
    return encoded


def merge_events_plaintext(all_party_events, dict_size):
    """Merge every party's events by the composite key, breaking timestamp ties as the secure merge does."""
    merged = []
    for events in all_party_events:
        for act, ts in events:
            if act != PAD_ACT:
                merged.append((ts * dict_size + act, ts, act))
    merged.sort()
    return [(ts, act) for _, ts, act in merged]


def extract_bigrams_plaintext(merged, dict_size):
    """The d^2 bigram histogram of the merged events."""
    d_sq = dict_size * dict_size
    hist = [0] * d_sq
    if not merged:
        return hist

    acts = [act for _, act in merged]

    hist[START_ACT * dict_size + acts[0]] += 1

    for i in range(len(acts) - 1):
        hist[acts[i] * dict_size + acts[i+1]] += 1

    hist[acts[-1] * dict_size + END_ACT] += 1

    return hist


def extract_ef_plaintext(merged, dict_size):
    """The d^2 eventually-follows counts: how often b appears before a."""
    d_sq = dict_size * dict_size
    ef = [0] * d_sq
    if not merged:
        return ef

    acts = [START_ACT] + [act for _, act in merged] + [END_ACT]

    prefix = [0] * dict_size
    for act in acts:
        for b in range(dict_size):
            ef[b * dict_size + act] += prefix[b]
        prefix[act] += 1

    return ef


def extract_activity_plaintext(merged, dict_size):
    """Bag-of-activities twin of mpc_utils/encoding_activity.py and p2_bigrams.rs::extract_activity."""
    hist = [0] * dict_size
    for _, act in merged:
        if 0 <= act < dict_size:
            hist[act] += 1
    return hist


def project_bigrams(hist, r_matrix):
    """Project a histogram through the sparse matrix."""
    return r_matrix @ np.array(hist, dtype=np.float64)


def generate_achlioptas_matrix(k_proj, d_sq, seed=42):
    """A sparse Achlioptas matrix over {-1, 0, +1} with probabilities 1/6, 2/3, 1/6."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 6, size=(k_proj, d_sq))
    matrix = np.zeros((k_proj, d_sq), dtype=np.int32)
    matrix[raw == 0] = -1
    matrix[raw == 5] = 1
    return matrix


def write_projection_matrix(path, matrix):
    """Write the projection matrix in the format the secure program reads."""
    k_proj, d_sq = matrix.shape
    with open(path, 'w') as f:
        f.write(f"{k_proj} {d_sq}\n")
        for row in matrix:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def cosine_similarity(a, b):
    """Cosine similarity of two vectors."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if norm_a < 1e-12 and norm_b < 1e-12 else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def parse_row_output(stdout, n_cases, k_proj):
    """Parse the ROW_START, B and ROW_END lines of the tape's output into a matrix."""
    rows = []
    current = None
    for line in stdout.splitlines():
        if line.startswith("ROW_START"):
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
    return rows


def prepare_nparty_data(party_files, max_cases, act_to_id=None, dict_size=None,
                        universe="intersection"):
    """Load and encode the n-party data; the union universe admits every case any party holds, as all-PAD slices."""
    all_party_cases = [load_xes_cases(f) for f in party_files]

    assert universe in ("intersection", "union"), universe
    common_ids = set(all_party_cases[0].keys())
    for pc in all_party_cases[1:]:
        if universe == "union":
            common_ids |= set(pc.keys())
        else:
            common_ids &= set(pc.keys())
    common_ids = sorted(common_ids)[:max_cases]
    print(f"  {len(common_ids)} {universe} cases")
    if universe == "union":
        # Coverage of the selection: an all-PAD party block and a partial slot in the middle are the shapes at risk.
        cov = [sum(1 for pc in all_party_cases if cid in pc) for cid in common_ids]
        partial_pos = [i for i, c in enumerate(cov) if c < len(all_party_cases)]
        solo_pos = [i for i, c in enumerate(cov) if c == 1]
        print(f"  coverage: {len(partial_pos)} partial slots at {partial_pos[:12]}"
              f"{'...' if len(partial_pos) > 12 else ''}, "
              f"{len(solo_pos)} single-party slots at {solo_pos[:12]}"
              f"{'...' if len(solo_pos) > 12 else ''}, "
              f"min={min(cov, default=0)}/{len(all_party_cases)}")

    if act_to_id is None:
        act_to_id, dict_size = build_dictionary(all_party_cases)
    print(f"  dict_size={dict_size}")

    min_ts = float('inf')
    max_plen = 0
    for pc in all_party_cases:
        for cid in common_ids:
            events = pc.get(cid, [])
            for ts, _ in events:
                if ts > 0:
                    min_ts = min(min_ts, ts)
            max_plen = max(max_plen, len(events))
    if min_ts == float('inf'):
        min_ts = 0
    partial_len = max_plen
    print(f"  partial_len={partial_len}")

    n_parties = len(party_files)
    all_encoded = []
    for p in range(n_parties):
        enc = encode_cases(all_party_cases[p], common_ids, act_to_id, partial_len, min_ts)
        all_encoded.append(enc)

    return all_encoded, common_ids, dict_size, partial_len, min_ts, act_to_id


def build_empty_case_fixture():
    """Two-party synthetic fixture whose last two cases are all-PAD, where the EF row must stay all-zero."""
    dict_size = 6
    seqs = [[3, 4, 5], [3, 4, 5], [4, 5, 3], [4, 5, 3], [5, 3, 4],
            [3, 5], [4, 3], [5, 4, 3], [3], [], []]
    partial_len = 3
    common_ids = [f"e{i}" for i in range(len(seqs))]
    party_a, party_b = [], []
    pad = (PAD_ACT, PAD_TS)
    for s_ in seqs:
        # Alternating slots exercise the merge tree; an empty sequence leaves both party blocks all-PAD.
        a = [(act, t + 1) for t, act in enumerate(s_) if t % 2 == 0]
        b = [(act, t + 1) for t, act in enumerate(s_) if t % 2 == 1]
        party_a.append(a + [pad] * (partial_len - len(a)))
        party_b.append(b + [pad] * (partial_len - len(b)))
    return [party_a, party_b], common_ids, dict_size, partial_len


def compute_plaintext_baseline(all_encoded, common_ids, dict_size, r_matrix, encoding="bigram"):
    """Merge, encode and project each case in the clear."""
    n_cases = len(common_ids)
    n_parties = len(all_encoded)

    extract_fn = {
        "ef": extract_ef_plaintext,
        "activity": extract_activity_plaintext,
    }.get(encoding, extract_bigrams_plaintext)

    baselines_proj = []
    for c in range(n_cases):
        party_events = [all_encoded[p][c] for p in range(n_parties)]
        merged = merge_events_plaintext(party_events, dict_size)
        hist = extract_fn(merged, dict_size)
        baselines_proj.append(project_bigrams(hist, r_matrix))

    return baselines_proj


def run_test_nparty(spdz_dir, party_files, max_cases, k_proj, seed, label,
                    encoding="bigram", universe="intersection", n_threads=8,
                    partial_lens=None, compile_flags=None, prebuilt=None):
    """One cell: the secure extract against the clear baseline; partial_lens "auto" derives the per-party lengths."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    n_parties = len(prebuilt[0]) if prebuilt else len(party_files)
    encoding_flag = {"bigram": 0, "ef": 1, "activity": 3}[encoding]
    print(f"  n_parties={n_parties}, max_cases={max_cases}, k_proj={k_proj}, "
          f"encoding={encoding}, n_threads={n_threads}")

    if prebuilt is not None:
        all_encoded, common_ids, dict_size, partial_len = prebuilt
    else:
        all_encoded, common_ids, dict_size, partial_len, _, _ = \
            prepare_nparty_data(party_files, max_cases, universe=universe)
    n_cases = len(common_ids)

    plens_arg = None
    if partial_lens is not None:
        if partial_lens == "auto":
            partial_lens = derive_partial_lens(all_encoded)
        assert len(partial_lens) == n_parties, (partial_lens, n_parties)
        all_encoded = retruncate_per_party(all_encoded, partial_lens)
        partial_len = max(partial_lens)
        plens_arg = "x".join(str(x) for x in partial_lens)
        print(f"  PARTIAL_LENS={plens_arg}")

    # The projection reads the clustering-encoding row, whose width is that encoding's own.
    enc_dim = dict_size if encoding == "activity" else dict_size * dict_size

    # One R for both sides: Rust draws it with ChaCha8, a numpy generator stands in here.
    r_matrix = generate_achlioptas_matrix(k_proj, enc_dim, seed)

    proj_path = Path(spdz_dir) / "Programs" / "Source" / "projection_matrix.txt"
    write_projection_matrix(proj_path, r_matrix)

    # Write the party inputs.
    write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases,
                        partial_lens if plens_arg else partial_len)

    print(f"  Computing plaintext {encoding} baseline...")
    baselines_proj = compute_plaintext_baseline(
        all_encoded, common_ids, dict_size, r_matrix, encoding=encoding
    )

    enc_dim_arg = 0 if encoding_flag in (0, 1) else enc_dim
    args = [n_cases, partial_len, dict_size, n_threads, k_proj, n_parties,
            encoding_flag]
    if plens_arg:
        # SECURE_Y filler; ENC_DIM 0 means the d^2 default.
        args += [0, enc_dim_arg, plens_arg]
    elif enc_dim_arg:
        args += [0, enc_dim_arg]
    prog_name = compile_mpc(spdz_dir, "secure_extract", args,
                            compile_flags=compile_flags, n_parties=n_parties)
    stdout, wall_time = run_mpc(spdz_dir, prog_name, n_parties)

    mpc_projected = parse_row_output(stdout, n_cases, k_proj)

    # Cosine is reported only; the assertion below is exact equality.
    similarities = [
        cosine_similarity(mpc_projected[c], baselines_proj[c]) for c in range(n_cases)
    ]
    min_sim = min(similarities)
    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)

    round_info = ""
    for line in stdout.splitlines():
        if "round" in line.lower() or "communication" in line.lower():
            round_info += f"  {line}\n"

    print(f"\n  RESULTS for {label}:")
    print(f"  {'-'*50}")
    print(f"  Cases: {n_cases}")
    print(f"  Parties: {n_parties}")
    print(f"  Cosine similarity: min={min_sim:.6f} avg={avg_sim:.6f} max={max_sim:.6f}")
    print(f"  Wall time: {wall_time:.1f}s")
    if round_info:
        print(f"  MPC info:\n{round_info}")

    n_padded = next_pow2(n_parties)
    n_levels = int(math.log2(n_padded))
    party_slots = next_pow2(partial_len)
    total_slots = party_slots * n_padded
    print("\n  Merge tree structure:")
    print(f"    N_PARTIES_PADDED={n_padded}, N_LEVELS={n_levels}")
    print(f"    PARTY_SLOTS={party_slots}, TOTAL_SLOTS={total_slots}")
    for lv in range(n_levels):
        block = party_slots * (2 ** (lv + 1))
        n_merges = total_slots // block
        half = block // 2
        print(f"    Level {lv}: {n_merges} merge(s) of 2x{half} -> {block}")

    # Every projected cell is an integer, so exact equality catches a row a cosine bound would not.
    mismatched = [c for c in range(n_cases)
                  if [int(v) for v in mpc_projected[c]]
                  != [int(round(v)) for v in baselines_proj[c]]]
    lost = [c for c in mismatched
            if not any(mpc_projected[c]) and any(baselines_proj[c])]
    ok = not mismatched
    if not ok:
        print(f"\n  FAIL: {len(mismatched)}/{n_cases} rows differ from the "
              f"plaintext reference ({len(lost)} of them all-zero = case loss); "
              f"first mismatches {mismatched[:12]}")
    else:
        print(f"\n  PASS: all {n_cases} rows exactly equal the reference "
              f"(min cosine {min_sim:.6f})")

    return ok, wall_time, similarities, mpc_projected


def main():
    parser = hc.common_parser("Test secure_extract.mpc")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--k-proj", type=int, default=55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-threads", type=int, default=8)
    args = parser.parse_args()
    spdz_dir = hc.setup(args)

    data_dir = PROJECT_ROOT / "data"
    sepsis_3party = data_dir / "splits" / "3party"
    sepsis_4party = data_dir / "splits" / "sepsis_4party"
    party_files_3 = [
        sepsis_3party / "party_a_er.xes",
        sepsis_3party / "party_b_lab.xes",
        sepsis_3party / "party_c_treatment.xes",
    ]
    party_files_4 = [
        sepsis_4party / "party_a_er.xes",
        sepsis_4party / "party_b_lab.xes",
        sepsis_4party / "party_c_emergency.xes",
        sepsis_4party / "party_d_disposition.xes",
    ]

    results = {}

    def cell(name, *a, **kw):
        if not hc.cell_enabled(name):
            return
        ok, wall, sims, _ = run_test_nparty(spdz_dir, *a, **kw)
        results[name] = {"ok": ok, "time": wall, "min_sim": min(sims)}

    cell("bigram", party_files_3, args.max_cases, args.k_proj, args.seed,
         f"3-party Sepsis ({args.max_cases} cases)", n_threads=args.n_threads)

    cell("ef", party_files_3, args.max_cases, args.k_proj, args.seed,
         f"3-party Sepsis EF encoding ({args.max_cases} cases)",
         encoding="ef", n_threads=args.n_threads)

    # The activity encoding narrows enc_dim to d, which the projection matrix and the width assertion both follow.
    cell("activity", party_files_3, args.max_cases, 8, args.seed,
         f"3-party Sepsis ACTIVITY encoding ({args.max_cases} cases)",
         encoding="activity", n_threads=args.n_threads)

    # EF over all-PAD cases: without the any_real gate every empty case would carry one profile.
    if hc.cell_enabled("ef_empty"):
        fixture = build_empty_case_fixture()
        ok, wall, sims, rows = run_test_nparty(
            spdz_dir, [None, None], len(fixture[1]), 6, args.seed,
            "2-party synthetic EF with ALL-PAD cases",
            encoding="ef", n_threads=args.n_threads, prebuilt=fixture)
        empty_rows = [9, 10]
        nonzero = [c for c in empty_rows if any(rows[c])]
        if nonzero:
            print(f"  FAIL ef_empty: all-PAD case(s) {nonzero} carry a nonzero "
                  "EF profile (the START seed is not gated on any_real)")
            ok = False
        else:
            print(f"  PASS ef_empty: all-PAD cases {empty_rows} encode to the "
                  "all-zero EF row, as extract_ef does in Rust")
        results["ef_empty"] = {"ok": ok, "time": wall, "min_sim": min(sims)}

    # The union universe, at the shortest prefix that has both a mid-selection partial slot and a one-party case.
    union_cases = max(args.max_cases, 24)
    cell("union", party_files_3, union_cases, args.k_proj, args.seed,
         f"3-party Sepsis UNION universe ({union_cases} cases)",
         universe="union", n_threads=args.n_threads)

    # Four parties, the union universe at N=120 and per-party lengths; test_secure_pipeline.py pins the Round-2 half.
    if all(f.exists() for f in party_files_4):
        cell("f4", party_files_4, 120, args.k_proj, args.seed,
             f"4-party Sepsis UNION N=120 PARTIAL_LENS, {args.n_threads} threads",
             universe="union", n_threads=args.n_threads, partial_lens="auto")
    else:
        print(f"\nSKIP F4 extract cell: missing {party_files_4}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, r in results.items():
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {status}: {name} time={r['time']:.1f}s "
              f"min_sim={r['min_sim']:.6f}")
        if not r["ok"]:
            all_pass = False
    print("\nAll tests PASSED!" if all_pass else "\nSome tests FAILED!")
    return all_pass and bool(results)


if __name__ == "__main__":
    sys.exit(hc.run_harness("test_secure_extract", main))
