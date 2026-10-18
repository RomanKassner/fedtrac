#!/usr/bin/env python3
"""Render every published cluster graph of every run under output/runs as an SVG, beside the whole-log graph the
model-quality instrument scores.

    .venv/bin/python scripts/render_graphs.py --runs output/runs --out output/models [--tier federated]
"""
import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import graphviz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.models_figure_uniform import cluster_graph, whole_log_graph  # noqa: E402


def render(graph, out: Path) -> None:
    """One graph object -> an SVG; ">" is the start marker and "|" the end marker."""
    g = graphviz.Digraph("dfg", format="svg")
    g.node_attr.update({"shape": "box", "fontname": "Helvetica", "fontsize": "11"})
    g.edge_attr.update({"fontname": "Helvetica", "fontsize": "9"})
    g.node("start", "", shape="circle", width="0.2", style="filled", fillcolor="black")
    g.node("end", "", shape="doublecircle", width="0.16")
    ids = {">": "start", "|": "end"}
    for i, act in enumerate(graph["activities"]):
        ids[act] = f"a{i}"
        g.node(ids[act], act)
    for e in graph["edges"]:
        g.edge(ids[e["from"]], ids[e["to"]], label=f" {int(e['weight'])}")
    g.render(str(out), cleanup=True)


def render_run(run: Path, tier: str, out: Path) -> int:
    tier_dir = run / tier
    paths = sorted(tier_dir.glob("cluster_*_dfg.json"))
    if not paths:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        j = int(path.name.split("_")[1])
        render(cluster_graph(tier_dir, j), out / f"cluster_{j}")
    render(whole_log_graph(tier_dir, len(paths))[0], out / "whole_log")
    return len(paths) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=REPO / "output" / "runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", default="federated")
    args = ap.parse_args()

    for entry in sorted(args.runs.iterdir()):
        if entry.is_dir():
            print(f"{entry.name}: {render_run(entry, args.tier, args.out / entry.name)} SVGs")
        elif entry.name.endswith(".tar.xz"):
            name = entry.name[: -len(".tar.xz")]
            with tempfile.TemporaryDirectory() as td:
                with tarfile.open(entry) as tf:
                    tf.extractall(td, filter="data")
                root = Path(td) / name
                root = root if root.is_dir() else Path(td)
                print(f"{name}: {render_run(root, args.tier, args.out / name)} SVGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
