#!/usr/bin/env python3
"""N-party activity-level log splitter with cross-party bigram analysis."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge"))


def load_partition(path: str) -> dict[str, list[str]]:
    with open(path) as f:
        raw = json.load(f)
    # Keys starting with "_" are annotations (e.g. "_comment"), not parties.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def validate_partition(partition: dict[str, list[str]], all_activities: set[str]):
    assigned = []
    for party, activities in partition.items():
        assigned.extend(activities)

    assigned_set = set(assigned)
    if len(assigned) != len(assigned_set):
        dupes = [a for a in assigned if assigned.count(a) > 1]
        raise ValueError(f"Duplicate activities in partition: {set(dupes)}")

    missing = all_activities - assigned_set
    if missing:
        print(f"WARNING: Activities not assigned to any party: {missing}")

    extra = assigned_set - all_activities
    if extra:
        print(f"WARNING: Partition contains activities not in log: {extra}")


def split_by_activity(df: pd.DataFrame, partition: dict[str, list[str]], output_dir: Path):
    import pm4py

    output_dir.mkdir(parents=True, exist_ok=True)

    for party_name, activities in partition.items():
        activity_set = set(activities)
        party_df = df[df["concept:name"].isin(activity_set)].copy()

        out_path = output_dir / f"{party_name}.xes"
        if len(party_df) > 0:
            pm4py.write_xes(party_df, str(out_path))
        else:
            from pm4py.objects.log.exporter.xes import exporter as xes_exporter
            from pm4py.objects.log.obj import EventLog
            xes_exporter.apply(EventLog(), str(out_path))

        print(f"  {party_name}: {len(party_df)} events, "
              f"{party_df['case:concept:name'].nunique() if len(party_df) > 0 else 0} cases "
              f"-> {out_path}")


def analyze_cross_party_bigrams(df: pd.DataFrame, partition: dict[str, list[str]]) -> dict:
    act_to_party = {}
    for party, activities in partition.items():
        for act in activities:
            act_to_party[act] = party

    df = df.sort_values(["case:concept:name", "time:timestamp"])

    intra_count = 0
    cross_count = 0
    cross_types = defaultdict(int)
    unassigned_count = 0

    for _case_id, group in df.groupby("case:concept:name", sort=False):
        acts = group["concept:name"].tolist()
        for i in range(len(acts) - 1):
            a, b = acts[i], acts[i + 1]
            pa = act_to_party.get(a)
            pb = act_to_party.get(b)
            if pa is None or pb is None:
                unassigned_count += 1
                continue
            if pa == pb:
                intra_count += 1
            else:
                cross_count += 1
                key = tuple(sorted([pa, pb]))
                cross_types[key] += 1

    total = intra_count + cross_count
    return {
        "total_bigrams": total,
        "intra_party": intra_count,
        "cross_party": cross_count,
        "cross_fraction": cross_count / total if total > 0 else 0.0,
        "cross_by_party_pair": dict(cross_types),
        "unassigned": unassigned_count,
    }


def print_statistics(df: pd.DataFrame, partition: dict[str, list[str]], bigram_stats: dict):
    print("\n=== Per-Party Statistics ===")
    print(f"{'Party':<25} {'Events':>8} {'Cases':>8} {'Activities':>12}")
    print("-" * 60)

    for party, activities in partition.items():
        activity_set = set(activities)
        party_df = df[df["concept:name"].isin(activity_set)]
        n_events = len(party_df)
        n_cases = party_df["case:concept:name"].nunique() if n_events > 0 else 0
        print(f"{party:<25} {n_events:>8} {n_cases:>8} {len(activities):>12}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {len(df):>8} {df['case:concept:name'].nunique():>8}")

    print("\n=== Bigram Analysis ===")
    print(f"Total bigrams:      {bigram_stats['total_bigrams']}")
    print(f"Intra-party:        {bigram_stats['intra_party']}")
    print(f"Cross-party:        {bigram_stats['cross_party']}")
    print(f"Cross-party fraction: {bigram_stats['cross_fraction']:.4f}")

    if bigram_stats['cross_by_party_pair']:
        print("\nCross-party bigrams by party pair:")
        for (pa, pb), count in sorted(bigram_stats['cross_by_party_pair'].items()):
            print(f"  {pa} <-> {pb}: {count}")

    if bigram_stats['unassigned'] > 0:
        print(f"\nUnassigned: {bigram_stats['unassigned']}")


def main():
    parser = argparse.ArgumentParser(description="N-Party Activity-Level Log Splitter")
    parser.add_argument("--xes", required=True, help="Input XES file")
    parser.add_argument("--partition-file", required=True, help="Partition JSON config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    from utils import load_xes

    print(f"Loading {args.xes}...")
    df = load_xes(args.xes)
    all_activities = set(df["concept:name"].unique())
    print(f"Loaded {len(df)} events, {df['case:concept:name'].nunique()} cases, "
          f"{len(all_activities)} activities")

    partition = load_partition(args.partition_file)
    print(f"\nPartition: {len(partition)} parties")
    for party, acts in partition.items():
        print(f"  {party}: {acts}")

    validate_partition(partition, all_activities)

    print("\nSplitting log...")
    split_by_activity(df, partition, Path(args.output_dir))

    bigram_stats = analyze_cross_party_bigrams(df, partition)
    print_statistics(df, partition, bigram_stats)

    stats_path = Path(args.output_dir) / "split_stats.json"
    with open(stats_path, "w") as f:
        serializable = {**bigram_stats}
        serializable["cross_by_party_pair"] = {
            f"{k[0]} <-> {k[1]}": v for k, v in bigram_stats["cross_by_party_pair"].items()
        }
        json.dump(serializable, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
