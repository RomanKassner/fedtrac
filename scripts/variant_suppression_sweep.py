#!/usr/bin/env python3
"""Variant-suppression / privacy-utility curve for --reveal-secret-projection: per k, how many cases and distinct
variants sit in classes smaller than k -- the calibration input for budget-bounded k_priv (largest k with
suppressed <= tau).

Two input modes: one or more XES logs, each read on its own with the per-case trace truncated to --trunc; or
--split-dir DIR, the party files Party_*.xes of one federation split, where the per-case trace is the merge of the
party slices and --trunc bounds each party's slice before the merge, as the protocol does. The merge key is
(timestamp, activity name) over the alphabetically sorted union of the party vocabularies (src/types.rs), the id
order of the in-MPC composite key `ts * DICT_SIZE + act` (src/mpc/input_encoding.rs).
"""
import argparse
import glob
import os
from collections import Counter

import pm4py


def profile(acts: tuple, unit: str):
    """The anonymity class of one case: 'seq' the ordered activity tuple (stricter than the reveal), 'bigram' and
    'activity' the count profiles that are the disclosure units of Y_pi under those encodings."""
    if unit == "bigram":
        return frozenset(Counter(zip(acts, acts[1:])).items())
    if unit == "activity":
        return frozenset(Counter(acts).items())
    return acts


def _case_events(xes_path: str, case_ids: set | None) -> dict:
    """case id -> [(timestamp, activity)], sorted by the merge key."""
    log = pm4py.read_xes(xes_path)
    stamps = log["time:timestamp"].astype("int64").tolist()
    cids = log["case:concept:name"].astype(str).tolist()
    acts = log["concept:name"].astype(str).tolist()
    out: dict[str, list] = {}
    for cid, ts, act in zip(cids, stamps, acts):
        if case_ids is not None and cid not in case_ids:
            continue
        out.setdefault(cid, []).append((ts, act))
    for ev in out.values():
        ev.sort()
    return out


def case_fingerprints(xes_path: str, trunc: int, unit: str,
                      case_ids: set | None = None) -> list:
    """Per-case anonymity fingerprint over the trace truncated to `trunc` (0 = full, the worst case), restricted
    to `case_ids` where given."""
    log = pm4py.read_xes(xes_path)
    out = []
    for cid, case in log.groupby("case:concept:name", sort=False):
        if case_ids is not None and str(cid) not in case_ids:
            continue
        acts = tuple(case.sort_values("time:timestamp")["concept:name"].tolist())
        if trunc:
            acts = acts[:trunc]
        out.append(profile(acts, unit))
    return out


def split_fingerprints(party_paths: list[str], trunc: int, unit: str,
                       case_ids: set | None = None) -> tuple[list, int]:
    """(fingerprints, cases present in every party file) over the merged trace of a split: each party's slice is
    truncated to `trunc` events (the protocol's per-party bound L), then the slices merge on (second, activity)."""
    per_party = [_case_events(p, case_ids) for p in party_paths]
    shared = set(per_party[0])
    for d in per_party[1:]:
        shared &= set(d)
    merged: dict[str, list] = {}
    for d in per_party:
        for cid, ev in d.items():
            if case_ids is not None and cid not in case_ids:
                continue
            merged.setdefault(cid, []).extend(ev[:trunc] if trunc else ev)
    out = []
    for cid in sorted(merged):
        ev = sorted(merged[cid])
        out.append(profile(tuple(a for _, a in ev), unit))
    return out, len(shared)


def suppression_curve(seqs: list[tuple], ks: list[int]) -> dict:
    """For each k: (cases_suppressed, frac_cases, variants_suppressed, frac_variants)."""
    n = len(seqs)
    class_sizes = Counter(seqs)            # variant -> count
    sizes = list(class_sizes.values())
    n_variants = len(sizes)
    out = {}
    for k in ks:
        cases_sup = sum(s for s in sizes if s < k)
        var_sup = sum(1 for s in sizes if s < k)
        out[k] = {
            "cases_suppressed": cases_sup,
            "frac_cases": cases_sup / n if n else 0.0,
            "variants_suppressed": var_sup,
            "frac_variants": var_sup / n_variants if n_variants else 0.0,
        }
    return {"n_cases": n, "n_variants": n_variants,
            "singletons": sum(1 for s in sizes if s == 1), "curve": out}


def budget_bounded_k(curve: dict, taus: list[float], k_floor: int, k_max: int) -> dict:
    """Largest k in [k_floor, k_max] with frac_cases <= tau (>= k_floor always)."""
    out = {}
    for tau in taus:
        best = k_floor
        for k in range(k_floor, k_max + 1):
            fc = curve["curve"].get(k, {}).get("frac_cases", 1.0)
            if fc <= tau:
                best = k
        out[tau] = {"k": best,
                    "frac_cases_at_k": curve["curve"].get(best, {}).get("frac_cases", 0.0)}
    return out


def _report(tag: str, res: dict, ks: list[int], taus: list[float],
            k_floor: int, k_max: int) -> None:
    print(f"\n=== {tag} ===")
    print(f"  cases={res['n_cases']}  variants={res['n_variants']}  "
          f"singletons={res['singletons']} "
          f"({res['singletons']/res['n_cases']*100:.1f}% of cases, "
          f"{res['singletons']/res['n_variants']*100:.1f}% of variants)")
    print("  k :   cases_supp (frac)   variants_supp (frac)")
    for k in ks:
        c = res["curve"][k]
        print(f"  {k:>2}: {c['cases_suppressed']:>7} "
              f"({c['frac_cases']*100:5.1f}%)   "
              f"{c['variants_suppressed']:>7} ({c['frac_variants']*100:5.1f}%)")
    bb = budget_bounded_k(res, taus, k_floor, k_max)
    print("  budget-bounded k:")
    for tau, v in bb.items():
        print(f"    tau<={tau*100:.0f}% of cases -> k={v['k']} "
              f"(drops {v['frac_cases_at_k']*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--split-dir", default=None,
                    help="a federation split directory holding Party_*.xes; the "
                         "per-case trace is the merge of the party slices and "
                         "--trunc bounds each party slice before the merge, as "
                         "the protocol does")
    ap.add_argument("--trunc", type=int, nargs="+", default=[0],
                    help="truncation lengths L to test (0 = full length)")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[2, 3, 5, 8, 10, 15, 20])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--k-floor", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--unit", choices=["seq", "bigram", "activity"], default="seq",
                    help="anonymity unit: ordered sequence, bigram profile or "
                         "activity profile")
    ap.add_argument("--case-ids", default=None,
                    help="restrict to the case ids in this file (one per line), "
                         "the pinned campaign subsample")
    args = ap.parse_args()

    if not args.logs and not args.split_dir:
        ap.error("give one or more XES logs, or --split-dir")

    case_ids = None
    if args.case_ids:
        with open(args.case_ids) as fh:
            case_ids = {ln.strip() for ln in fh if ln.strip()}

    if args.split_dir:
        parties = sorted(glob.glob(os.path.join(args.split_dir, "Party_*.xes")))
        if not parties:
            ap.error(f"no Party_*.xes under {args.split_dir}")
        name = os.path.basename(args.split_dir.rstrip("/"))
        print(f"# split {args.split_dir}: {len(parties)} parties "
              f"({', '.join(os.path.basename(p) for p in parties)})")
        if case_ids is not None:
            print(f"# pinned to {len(case_ids)} case ids from {args.case_ids}")
        for trunc in args.trunc:
            seqs, shared = split_fingerprints(parties, trunc, args.unit, case_ids)
            res = suppression_curve(seqs, args.ks)
            tag = (f"{name} merged (L={trunc or 'full'} per party, "
                   f"unit={args.unit})")
            print(f"\n# cases present in every party file: {shared}")
            _report(tag, res, args.ks, args.taus, args.k_floor, args.k_max)

    for path in args.logs:
        name = path.split("/")[-1]
        for trunc in args.trunc:
            seqs = case_fingerprints(path, trunc, args.unit, case_ids)
            res = suppression_curve(seqs, args.ks)
            tag = f"{name} (L={trunc or 'full'}, unit={args.unit})"
            _report(tag, res, args.ks, args.taus, args.k_floor, args.k_max)


if __name__ == "__main__":
    main()
