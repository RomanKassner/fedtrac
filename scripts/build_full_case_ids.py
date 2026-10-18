#!/usr/bin/env python3
"""Build config/case_ids/<log>_full.txt: the case ids shared by every party file of a split directory, or -- with --union -- held by any of them."""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def trace_ids(xes_path: Path) -> set[str]:
    ids = set()
    for _, elem in ET.iterparse(xes_path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "trace":
            for child in elem:
                if (child.tag.rsplit("}", 1)[-1] == "string"
                        and child.get("key") == "concept:name"):
                    ids.add(child.get("value"))
                    break
            elem.clear()
    return ids


def party_files(split_dir: Path, parties: int | None) -> list[Path]:
    """The party files whose cases intersect: all .xes by default, or the N the
    loader would read at --n-parties N — a three-party split read at two parties
    shares more cases than its full intersection, so it needs the restriction."""
    candidates = sorted(split_dir.glob("*.xes"))
    if not candidates:
        sys.exit(f"no .xes files in {split_dir}")
    if parties is None:
        return candidates
    if parties == 2:
        a, b = split_dir / "Party_A.xes", split_dir / "Party_B.xes"
        if a.exists() and b.exists():
            return [a, b]
    else:
        numbered = [split_dir / f"party_{i}.xes" for i in range(parties)]
        if all(p.exists() for p in numbered):
            return numbered
    if len(candidates) < parties:
        sys.exit(f"need {parties} .xes files in {split_dir}, "
                 f"found {len(candidates)}")
    return candidates[:parties]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path,
                        help="split directory containing the party .xes files")
    parser.add_argument("out", type=Path,
                        help="output list, e.g. config/case_ids/sepsis_full.txt")
    parser.add_argument("--parties", type=int, default=None,
                        help="intersect only the party files the loader reads "
                             "at --n-parties N (default: every .xes file)")
    parser.add_argument("--union", action="store_true",
                        help="agree on the union instead: every case any "
                             "party holds")
    args = parser.parse_args()

    shared: set[str] | None = None
    files = party_files(args.split_dir, args.parties)
    for pf in files:
        ids = trace_ids(pf)
        print(f"{pf.name}: {len(ids)} cases")
        if shared is None:
            shared = ids
        else:
            shared = shared | ids if args.union else shared & ids
    assert shared is not None
    args.out.write_text("".join(f"{cid}\n" for cid in sorted(shared)))
    print(f"{args.out}: {len(shared)} "
          f"{'union' if args.union else 'shared'} cases "
          f"across {len(files)} parties")


if __name__ == "__main__":
    main()
