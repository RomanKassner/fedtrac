#!/usr/bin/env python3
"""Cross-party share of directly-follows pairs: a pair of consecutive events of one case is intra-party when both
activities belong to one party and cross-party otherwise.

Over the raw logs and the committed partition configurations (the default), by scripts/split_log.py::
analyze_cross_party_bigrams, the function that writes a split directory's split_stats.json:

    .venv/bin/python scripts/cross_party_share.py [--config config] [--raw data/raw] [--note "<sentence>"]

Or over the merged traces of one recorded run, with the partition that run was split by:

    .venv/bin/python scripts/cross_party_share.py --traces <run>/federated/global_traces.json \\
        --partition config/sepsis_2party.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bridge"))
from split_log import analyze_cross_party_bigrams, load_partition  # noqa: E402

# raw log per configuration-name prefix, in the order the table lists the logs
LOGS = [
    ("sepsis", "Sepsis Cases.xes", "Sepsis"),
    ("helpdesk", "helpdesk.xes", "Helpdesk"),
    ("bpic2013", "bpic2013_incidents.xes", "BPIC 2013"),
    ("road_traffic", "road_traffic_fine_management.xes", "RTFM"),
]


def share(cross: int, total: int) -> str:
    pct = 100.0 * cross / total if total else float("nan")
    return ("%.2f %%" if pct < 0.1 else "%.1f %%") % pct


def traces_share(traces_path: Path, partition_path: Path) -> tuple[int, int]:
    """(pairs, cross-party pairs) over the merged traces of a recorded run."""
    traces = json.loads(traces_path.read_text())["traces"]
    party = {act: name
             for name, acts in load_partition(str(partition_path)).items()
             for act in acts}
    total = cross = 0
    for trace in traces:
        for a, b in zip(trace, trace[1:]):
            total += 1
            cross += party.get(a) != party.get(b)
    return total, cross


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--note", default=None,
                    help="one provenance sentence printed into the header")
    ap.add_argument("--traces", type=Path, default=None,
                    help="a run's federated/global_traces.json instead of the raw logs")
    ap.add_argument("--partition", type=Path, default=None,
                    help="the config/*.json the run was split by (with --traces)")
    args = ap.parse_args()

    if args.traces:
        if not args.partition:
            ap.error("--traces needs --partition")
        total, cross = traces_share(args.traces, args.partition)
        print("pairs %d  cross-party %d  share %s"
              % (total, cross, share(cross, total)))
        return 0

    from utils import load_xes

    configs = sorted(Path(args.config).glob("*.json"))
    print("# Cross-party share of directly-follows pairs per committed "
          "vertical partition; produced %s by" % dt.date.today().isoformat())
    print("#   .venv/bin/python scripts/cross_party_share.py --config %s "
          "--raw %s%s" % (args.config, args.raw,
                          (" --note '...'" if args.note else "")))
    print("# A property of the raw log and the partition configuration alone "
          "(no campaign run enters): every consecutive event pair of every")
    print("#   case is intra-party when both activities belong to one party "
          "and cross-party otherwise; computed by")
    print("#   scripts/split_log.py::analyze_cross_party_bigrams over "
          "data/raw/<log>.xes (pm4py) and the party -> activity lists of")
    print("#   config/<split>.json, the same call that writes "
          "data/splits/<dir>/split_stats.json when a split is generated.")
    if args.note:
        print("# " + args.note)
    print("%-26s %-9s %-8s %-12s %s" % ("split", "log", "pairs",
                                         "cross_party", "share"))
    for prefix, fname, label in LOGS:
        mine = [p for p in configs if p.stem.startswith(prefix)]
        if not mine:
            continue
        df = load_xes(str(Path(args.raw) / fname))
        for p in mine:
            st = analyze_cross_party_bigrams(df, load_partition(str(p)))
            extra = ("  (unassigned pairs: %d)" % st["unassigned"]
                     if st["unassigned"] else "")
            print("%-26s %-9s %-8d %-12d %s%s" % (
                p.stem, label, st["total_bigrams"], st["cross_party"],
                share(st["cross_party"], st["total_bigrams"]), extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
