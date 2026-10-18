#!/usr/bin/env python3
"""Figure C of the evaluation: the whole log filtered once beside published cluster graphs.

The whole-log drawing is scripts/evaluate_campaign.py's ``whole_*`` baseline, imported from it
(graph_from_traces -> bridge.discovery.filter_edges -> bridge.utils.prune_to_start_end_paths), so it is the
object the reported baseline was scored on; a published cluster graph is drawn unchanged, unpruned, as a party
receives it. Every drawn number is archived beside the PDFs. The drawing parameters take one value or one per
drawing (whole log first), and --height-cm only reports the width and the label size at that height.

    .venv/bin/python scripts/models_figure_uniform.py \
        --run output/runs/uni_rtfm-bigram_sl_km_2026-10-06_08-17-35 --clusters 3 5 \
        --out-dir <dir> --wrap 9 --fontsize 14 --label-fontsize 12 --ranksep 0.70 0.16 0.70 --nodesep 0.10
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import graphviz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bridge.utils import prune_to_start_end_paths                     # noqa: E402
from scripts.evaluate_campaign import graph_from_traces, envelope     # noqa: E402

PT_PER_CM = 72.0 / 2.54


def draw(graph, out, wrap, fontsize, label_fontsize, ranksep, nodesep, fontname="Helvetica"):
    """One graph object -> a PDF; ``from``/``to`` use ">" for the start marker and "|" for the end marker."""
    g = graphviz.Digraph("dfg", format="pdf")
    g.graph_attr.update({"rankdir": "TB", "nodesep": f"{nodesep}", "ranksep": f"{ranksep}",
                         "bgcolor": "white", "margin": "0"})
    g.node_attr.update({"fontname": fontname, "fontsize": f"{fontsize}", "shape": "box", "height": "0.2",
                        "margin": "0.04,0.02", "penwidth": "0.6"})
    g.edge_attr.update({"fontname": fontname, "fontsize": f"{label_fontsize}", "arrowsize": "0.5"})
    ids = {">": "start", "|": "end"}
    g.node("start", "", shape="circle", width="0.18", fixedsize="true", style="filled", fillcolor="black")
    g.node("end", "", shape="doublecircle", width="0.14", fixedsize="true")
    for i, act in enumerate(graph["activities"]):
        ids[act] = f"a{i}"
        # break_long_words=False: an activity name is never cut inside a word
        g.node(ids[act], "\\n".join(textwrap.wrap(act, wrap, break_long_words=False)))
    top = max(e["weight"] for e in graph["edges"])
    for e in graph["edges"]:
        w = e["weight"]
        g.edge(ids[e["from"]], ids[e["to"]], label=f" {int(w)}", penwidth=f"{0.5 + 2.0 * w / top:.2f}")
    g.render(str(out), cleanup=True)
    return top


def page_size(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size"):
            parts = line.split()
            return float(parts[2]), float(parts[4])
    raise RuntimeError(f"no page size in pdfinfo {pdf}")


def describe(name, graph, extra=""):
    """The counts and every drawn edge with its count, for the caption."""
    acts = graph["activities"]
    inner = [e for e in graph["edges"] if e["from"] != ">" and e["to"] != "|"]
    starts = [e for e in graph["edges"] if e["from"] == ">"]
    ends = [e for e in graph["edges"] if e["to"] == "|"]
    lines = [f"[{name}] {len(acts)} activities, {len(inner)} activity edges, "
             f"{len(starts)} start and {len(ends)} end edges, {len(graph['edges'])} drawn edges{extra}",
             f"[{name}] activities: {', '.join(acts)}"]
    for e in sorted(graph["edges"], key=lambda e: (-e["weight"], e["from"], e["to"])):
        lines.append(f"[{name}]   {e['from']} -> {e['to']}: {int(e['weight'])}")
    return lines


def whole_log_graph(tier_dir, n_clusters):
    """The whole-log baseline as a graph object, with its traces and its removals."""
    traces, sizes = [], []
    for j in range(n_clusters):
        part = [t for t in json.load(open(tier_dir / f"cluster_{j}_traces.json"))["traces"] if t]
        sizes.append(len(part))
        traces.extend(part)
    dfg, start, end = graph_from_traces(traces)
    filtered = len(dfg)
    dfg, start, end, removed = prune_to_start_end_paths(dfg, start, end)
    graph = envelope(dfg, start, end)
    graph["activities"] = sorted(set(start) | set(end) | {a for edge in dfg for a in edge})
    return graph, traces, sizes, removed, filtered


def cluster_graph(tier_dir, j):
    d = json.load(open(tier_dir / f"cluster_{j}_dfg.json"))
    return {"activities": sorted(d["activities"]), "edges": d["edges"], "num_traces": int(d["num_traces"])}


def per_drawing(values, n, what):
    if len(values) == 1:
        return list(values) * n
    if len(values) != n:
        raise SystemExit(f"--{what} takes 1 or {n} values, got {len(values)}")
    return list(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run directory under output/runs")
    ap.add_argument("--tier", default="federated")
    ap.add_argument("--clusters", type=int, nargs="+", required=True, help="the clusters whose published graph is drawn")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wrap", type=int, nargs="+", default=[9], help="characters per label line")
    ap.add_argument("--fontsize", type=float, nargs="+", default=[14], help="activity label size inside the drawing")
    ap.add_argument("--label-fontsize", type=float, nargs="+", default=[12], help="edge count size inside the drawing")
    ap.add_argument("--ranksep", type=float, nargs="+", default=[0.70], help="inches between ranks: the aspect lever")
    ap.add_argument("--nodesep", type=float, nargs="+", default=[0.10], help="inches between nodes of one rank")
    ap.add_argument("--fontname", default="Helvetica",
                    help="the font of the labels, by its fontconfig name")
    ap.add_argument("--height-cm", type=float, default=12.0, help="the height the figure includes each drawing at")
    ap.add_argument("--all-clusters", action="store_true",
                    help="also draw every other cluster of the tier, with the same parameters")
    args = ap.parse_args()

    tier_dir = args.run / args.tier
    n_clusters = len(sorted(tier_dir.glob("cluster_*_dfg.json")))
    names = ["whole_log"] + [f"cluster_{j}" for j in args.clusters]
    extra_ids = [j for j in range(n_clusters) if j not in args.clusters] if args.all_clusters else []
    k = len(names)
    wrap = per_drawing(args.wrap, k, "wrap")
    fontsize = per_drawing(args.fontsize, k, "fontsize")
    label_fontsize = per_drawing(args.label_fontsize, k, "label-fontsize")
    ranksep = per_drawing(args.ranksep, k, "ranksep")
    nodesep = per_drawing(args.nodesep, k, "nodesep")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# figure C (models_rtfm_uniform), drawn by scripts/models_figure_uniform.py on "
             f"{args.run.name}, tier {args.tier}, {n_clusters} clusters",
             "# the whole-log graph is scripts/evaluate_campaign.py's whole_* baseline (graph_from_traces -> "
             "bridge.discovery.filter_edges -> bridge.utils.prune_to_start_end_paths), the cluster graphs are the "
             "published files, unchanged"]

    graphs = {}
    graph, traces, sizes, removed, filtered = whole_log_graph(tier_dir, n_clusters)
    graphs["whole_log"] = graph
    lines += describe("whole_log", graph,
                      f"; {len(traces)} cases over {n_clusters} clusters of sizes {sizes}; the edge filter left "
                      f"{filtered} activity edges and the pruning removed {len(removed)} activities: "
                      f"{', '.join(removed) if removed else '-'}")
    for j in list(args.clusters) + extra_ids:
        cg = cluster_graph(tier_dir, j)
        graphs[f"cluster_{j}"] = cg
        lines += describe(f"cluster_{j}", cg, f"; {cg['num_traces']} cases")

    total = 0.0
    for i, name in enumerate(names + [f"cluster_{j}" for j in extra_ids]):
        p = min(i, k - 1)
        top = draw(graphs[name], args.out_dir / name, wrap[p], fontsize[p], label_fontsize[p], ranksep[p], nodesep[p],
                   args.fontname)
        w, h = page_size(args.out_dir / f"{name}.pdf")
        at = args.height_cm * w / h
        label_pt = label_fontsize[p] * args.height_cm * PT_PER_CM / h
        if i < k:
            total += at
        lines.append(f"[{name}] drawn to {name}.pdf, largest count {int(top)}; page {w:.0f}x{h:.0f} pt, "
                     f"{at:.2f} cm wide at {args.height_cm:g} cm height, edge counts at {label_pt:.1f} pt "
                     f"(wrap {wrap[p]}, font {fontsize[p]:g}/{label_fontsize[p]:g}, ranksep {ranksep[p]:g}, "
                     f"nodesep {nodesep[p]:g})")
    lines.append(f"[layout] the {k} drawings of the figure take {total:.2f} cm of the 14.7 cm text width at "
                 f"{args.height_cm:g} cm height; subfigure shares "
                 + ", ".join(f"{name} {args.height_cm * page_size(args.out_dir / (name + '.pdf'))[0] / page_size(args.out_dir / (name + '.pdf'))[1] / 14.7:.3f}"
                             for name in names))

    record = args.out_dir / "models_2026-10-09.txt"
    record.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"# archived {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
