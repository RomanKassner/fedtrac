#!/usr/bin/env python3
"""Consistency check of a uniform-campaign directory (scripts/run_uniform_campaign.sh), from its summary.tsv, its
clock.tsv and the logs of its cells.

Within one combination (event log, party count, embedding) every run reads the same cases at the same length bound,
and every Revealed-Projection draw withholds the same rows, since the threshold is selected on the secret class
sizes. A refusal hits every run of a combination or none; a combination whose runs partly failed is reported as
incomplete. A time is read only from a run whose clock row (scripts/clock_probe.py) lies within the bound and whose
first round is the one its design names; void.tsv (cell, since_utc, reason) names runs whose times do not count for
a reason no file of the run records, and those are listed without failing the check.

    scripts/check_uniform_campaign.py output/campaign_uniform_<stamp> [--strict-family b0]
    scripts/check_uniform_campaign.py output/campaign_uniform_<stamp> --is-done <cell>     exit 0 done, 1 not done

Exit 1 on an inconsistency, or when a cell of --strict-family did not exit 0. Incomplete combinations do not fail it.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clock_probe import gain_bound, out_of_bound, read_rows  # noqa: E402

RP_RE = re.compile(r"^ms_(?P<log>.+)_rp_(?P<enc>[a-z]+)_s(?P<seed>\d+)$")
SL_RE = re.compile(r"^(?P<log>.+)-(?P<enc>[a-z]+)_sl_(?P<variant>km|seq)$")
RT_RE = re.compile(r"^rt(?P<log>[A-Za-z0-9]+)_sl_(?P<enc>[a-z]+)_(?P<np>\d+)p$")
ENC_RE = re.compile(r"^enc_(?P<log>.+)_sl_(?P<enc>[a-z]+)$")


def combination(cell):
    """cell name -> (combination key, kind) or (None, None)."""
    m = RP_RE.match(cell)
    if m:
        return (m["log"], m["enc"]), "rp"
    m = SL_RE.match(cell)
    if m:
        return (m["log"], m["enc"]), "sl"
    m = RT_RE.match(cell)
    if m:
        return (f"{m['log'].lower()}-pc{m['np']}p", m["enc"]), "sl"
    m = ENC_RE.match(cell)
    if m:
        return (m["log"], m["enc"]), "sl"
    return None, None


def load(summary):
    """The last row per cell: a resumed campaign appends, later rows win."""
    rows = {}
    with open(summary, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[r["cell"]] = r
    return rows


def check(rows, strict_family=None):
    findings, incomplete, groups = [], [], defaultdict(list)
    for cell, r in rows.items():
        key, kind = combination(cell)
        if key is None:
            findings.append(f"{cell}: name outside the grammar of the campaign")
            continue
        groups[key].append((kind, r))
        if strict_family and r["family"] == strict_family and r["exit"] != "0":
            findings.append(f"{cell}: exit {r['exit']} in family {strict_family}, where no refusal is expected")
    report = []
    for key in sorted(groups):
        runs = groups[key]
        ok = [(k, r) for k, r in runs if r["exit"] == "0"]
        bad = [(k, r) for k, r in runs if r["exit"] != "0"]
        name = f"{key[0]} {key[1]}"
        for field, what in (("n", "population"), ("L", "length bound")):
            vals = {r[field] for _, r in ok}
            if len(vals) > 1:
                findings.append(f"{name}: {what} differs between runs: {sorted(vals)}")
        draws = [r for k, r in ok if k == "rp"]
        for field, what in (("k_priv", "anonymity threshold"), ("suppressed", "withheld rows")):
            vals = {r[field] for r in draws}
            if len(vals) > 1:
                findings.append(f"{name}: {what} differs between the Revealed-Projection draws: {sorted(vals)}")
        for kind, label in (("rp", "Revealed-Projection"), ("sl", "Secret-Labels")):
            exits = {r["exit"] == "0" for k, r in runs if k == kind}
            if len(exits) > 1:
                incomplete.append(f"{name}: some {label} runs exited 0 and some did not")
        ks = sorted({r["k"] for r in draws})
        report.append(f"{name:28s} ok {len(ok):2d} failed {len(bad):2d}  N {'/'.join(sorted({r['n'] for _, r in ok})) or '-':>7s}"
                      f"  L {'/'.join(sorted({r['L'] for _, r in ok})) or '-':>3s}  draws {len(draws)}"
                      f"  k {'+'.join(ks) or '-'}  withheld {'/'.join(sorted({r['suppressed'] for r in draws})) or '-'}")
    return findings, incomplete, report


def first_round(log):
    """How a run got its first round, from its log: "builder" (computed and stored), "stored" (read from the
    store) or "own" (computed, nothing stored); None without a log."""
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                if "--reuse-extraction: creating core slab" in line:
                    return "builder"
                if "--reuse-extraction: tail-only" in line:
                    return "stored"
    except OSError:
        return None
    return "own"


def designed_first_round(cell, r):
    """The first round the design gives a cell; None when summary.tsv does not say whether it reuses one."""
    if r.get("reuse") is None:
        return None
    if r["reuse"] != "reuse":
        return "own"
    m = SL_RE.match(cell)
    return "builder" if RT_RE.match(cell) or (m and m["variant"] == "km") else "stored"


def last_clock_rows(campaign):
    """{cell: its clock row with the latest start}."""
    rows = {}
    for r in read_rows(campaign / "clock.tsv"):
        if r["cell"] not in rows or r["start_utc"] >= rows[r["cell"]]["start_utc"]:
            rows[r["cell"]] = r
    return rows


def load_void(campaign):
    """{cell: (since_utc, reason)} from void.tsv; the latest entry of a cell holds."""
    void = {}
    try:
        with open(campaign / "void.tsv", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["cell"] not in void or r["since_utc"] >= void[r["cell"]][0]:
                    void[r["cell"]] = (r["since_utc"], r.get("reason") or "")
    except OSError:
        pass
    return void


def time_state(cell, clock, void, bound):
    """None when the times of the cell's last run count, else the reason why they do not."""
    c = clock.get(cell)
    if c is None:
        return "no clock row"
    if cell in void and c["start_utc"] <= void[cell][0]:
        return f"void: {void[cell][1]}"
    if out_of_bound(c, bound):
        return f"the monotonic clock gained {c['mono_gain_pct']} % on {c['ref']}"
    return None


def times(rows, campaign):
    """(runs whose times do not count, runs whose first round is not the designed one), exit-0 cells only."""
    clock, void, bound = last_clock_rows(campaign), load_void(campaign), gain_bound()
    void_times, off_design = [], []
    for cell, r in rows.items():
        if r["exit"] != "0":
            continue
        state = time_state(cell, clock, void, bound)
        if state:
            void_times.append(f"{cell}: {state}")
        want, got = designed_first_round(cell, r), first_round(campaign / f"{cell}.log")
        if want and got and want != got:
            off_design.append(f"{cell}: first round {got}, designed {want}")
    return void_times, off_design


def is_done(rows, campaign, cell):
    """(done, reason): done when the last run of the cell exited 0 and its times count."""
    r = rows.get(cell)
    if r is None or r["exit"] != "0":
        return False, ""
    state = time_state(cell, last_clock_rows(campaign), load_void(campaign), gain_bound())
    return state is None, state or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--strict-family", help="a family in which every cell must exit 0")
    ap.add_argument("--is-done", metavar="CELL", help="exit 0 when the cell is done, 1 when it has to run")
    args = ap.parse_args()
    summary = args.campaign / "summary.tsv"
    if not summary.exists():
        sys.exit(f"no summary.tsv in {args.campaign}")
    rows = load(summary)
    if args.is_done:
        done, reason = is_done(rows, args.campaign, args.is_done)
        if reason:
            print(f"exit 0 on record, its times do not count: {reason}")
        sys.exit(0 if done else 1)
    findings, incomplete, report = check(rows, args.strict_family)
    void_times, off_design = times(rows, args.campaign)
    print("\n".join(report))
    if incomplete:
        print("\nINCOMPLETE (a relaunch runs the failed cells again):")
        print("\n".join(f"  {f}" for f in incomplete))
    if void_times:
        print(f"\nTIMES DO NOT COUNT in {len(void_times)} runs (a relaunch of their families runs them again):")
        print("\n".join(f"  {f}" for f in void_times))
    if off_design:
        print(f"\n!!! FIRST ROUND NOT AS DESIGNED in {len(off_design)} runs (their times are not those of their kind):")
        print("\n".join(f"  {f}" for f in off_design))
    if findings:
        print("\n!!! INCONSISTENT:")
        print("\n".join(f"  {f}" for f in findings))
        sys.exit(1)
    print("\nconsistent: every combination reads one population at one length bound, and its draws withhold the same rows")
    if not void_times and not off_design:
        print("every run that exited 0 has a clock row within the bound and the first round of its design")


if __name__ == "__main__":
    main()
