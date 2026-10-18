#!/usr/bin/env python3
"""Emit the data files and the appendix tables of the uniform campaign's
evaluation figures A (eval_gap), B (eval_quality) and D (eval_cost) from the
campaign's model_quality.csv and agg_cells.csv.

    .venv/bin/python scripts/uniform_figures.py \
        --campaign output/campaign_uniform_2026-10-04 --out <dir> [--facts <tsv>]
"""

from __future__ import annotations

import argparse
import re
import csv
import json
import math
import statistics
import sys
from pathlib import Path

STAMP = "2026-10-09"
INSTRUMENT = "dfm-2026-10-15"   # the relevance is scored on the pruned graph

# key, display name, population, the camel-case stem of the party-count cells, the party-count population.
LOGS = [
    ("sepsis", "Sepsis", 927, "Sepsis", 925),
    ("helpdesk", "Helpdesk", 4577, "Helpdesk", 4221),
    ("bpic2013", "BPIC 2013", 7545, "Bpic2013", 4507),
    ("rtfm", "RTFM", 129985, "Rtfm", 1184),
]

# Top to bottom in figure A, left to right in figure D.
EMBEDDINGS = ["activity", "bigram", "trigram", "skipbigram", "ef"]
EMB_LABEL = {
    "activity": "activity",
    "bigram": "bigram",
    "trigram": "trigram",
    "skipbigram": "skip-bigram",
    "ef": "eventually-follows",
}

SEEDS = ["42", "1337", "7", "101", "202"]  # s42 is the draw run in full
FULL_SEED = "42"

# Refused at configuration time: no run of any instantiation exists.
REFUSED = {("sepsis", "ef")}

INSTANTIATIONS = [
    ("sl_km", "Secret-Labels, $k$-means"),
    ("sl_seq", "Secret-Labels, refinement"),
    ("rp", "Revealed-Projection"),
]

LOSS_BOUND = -0.05

# Figure A dodges the three marks of a row by this much in y.
DODGE_Y = 0.22


# --- readers --------------------------------------------------------------

def read_model_quality(campaign: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with (campaign / "model_quality.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["cell"], row["tier"])
            assert key not in rows, f"duplicate model_quality row {key}"
            rows[key] = row
    return rows


def read_agg(campaign: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (campaign / "agg_cells.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["cell"]] = row
    return rows


def read_facts(path: Path) -> dict[str, str]:
    facts: dict[str, str] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            facts[row["object"]] = row["value"]
    return facts


# --- one run --------------------------------------------------------------

class Run:
    """One scored run: its federated tier, its twin in the clear, the two
    baselines, its time and its byte count."""

    def __init__(self, cell: str, mq: dict, agg: dict, cases: int,
                 campaign: Path | None = None):
        self.cell = cell
        self.campaign = campaign
        fed = mq.get((cell, "federated"))
        clear = mq.get((cell, "plaintext"))
        assert fed is not None, f"{cell}: no federated tier in model_quality.csv"
        assert clear is not None, f"{cell}: no plaintext tier (no twin)"
        # the bits that name the cluster of a trace, per tier (see size_entropy)
        self.h: dict[str, float] = {}
        if campaign is not None:
            for tier in ("federated", "plaintext"):
                path = campaign / "model_quality" / f"{cell}__{tier}.json"
                assert path.exists(), f"{path} is missing"
                self.h["fed" if tier == "federated" else "clear"] = size_entropy(path)
                tag = json.loads(path.read_text())["instrument"]
                assert tag == INSTRUMENT, (
                    f"{cell} {tier}: instrument {tag}, expected {INSTRUMENT}")
        self.fed = fed
        self.clear = clear
        assert int(fed["cases"]) == cases, (
            f"{cell}: {fed['cases']} cases, the designed population is {cases}")
        assert int(clear["cases"]) == cases, f"{cell}: twin on another population"
        acell = agg.get(cell)
        assert acell is not None, f"{cell}: no row in agg_cells.csv"
        self.agg = acell
        # k is the resolved cluster count; a cluster with no case has no graph,
        # so a tier may publish fewer graphs than k.
        self.k = int(acell["k"])
        self.graphs = int(fed["clusters"])
        assert self.graphs <= self.k, (
            f"{cell}: {self.graphs} published graphs for k = {self.k}")
        assert int(clear["clusters"]) == self.k, (
            f"{cell}: the twin has {clear['clusters']} clusters, k is {self.k}")
        self.cases = cases

    # the drawn quantities
    def f(self, tier: str = "fed") -> float:
        return float((self.fed if tier == "fed" else self.clear)["f_score"])

    def precision(self, tier: str = "fed") -> float:
        return float((self.fed if tier == "fed" else self.clear)["precision"])

    def edges(self, tier: str = "fed") -> float:
        return float((self.fed if tier == "fed" else self.clear)["edges"])

    def rel(self, tier: str = "fed") -> float:
        return float((self.fed if tier == "fed" else self.clear)["rel"])

    def rel_coded(self, tier: str = "fed") -> float:
        """Size-weighted mean relevance of the graphs plus the bits that name the
        cluster of a trace; the whole log is one graph and needs none."""
        key = "fed" if tier == "fed" else "clear"
        assert key in self.h, f"{self.cell}: the cluster sizes need the campaign's model_quality folder"
        return self.rel(tier) + self.h[key]

    def random_rel_coded(self, kind: str) -> float:
        """The random partitions have the federated sizes and pay the same bits."""
        assert "fed" in self.h, f"{self.cell}: the cluster sizes need the campaign's model_quality folder"
        return self.random("rel", kind) + self.h["fed"]

    @property
    def gap(self) -> float:
        return self.f("fed") - self.f("clear")

    def whole(self, col: str) -> float:
        return float(self.fed["whole_" + col])

    def random(self, col: str, kind: str) -> float:
        return float(self.fed[f"random_{col}_{kind}"])

    # times and bytes
    def wall(self) -> float:
        return float(self.agg["wallclock_s"])

    def time(self) -> float:
        """The time of the framework itself: the secure programs from the first
        input to the published graphs, compilation included, plus the local
        clustering of a Revealed-Projection draw."""
        assert self.campaign is not None, f"{self.cell}: the time needs the campaign's cell logs"
        text = (self.campaign / f"{self.cell}.log").read_text(errors="ignore")
        def seconds(pattern):
            hits = re.findall(pattern + r" ([\d.]+)(ms|s)\b", text, re.M)
            return [float(v) / (1000.0 if unit == "ms" else 1.0) for v, unit in hits]
        mpc = seconds(r"^MPC completed in")
        assert mpc, f"{self.cell}: no 'MPC completed in' line in its log"
        local = seconds(r"^\[Local\] clustering completed in")
        return mpc[-1] + (local[-1] if local else 0.0)

    def round1(self) -> float:
        return float(self.agg["round1_exec_s"])

    def sent(self) -> float:
        return float(self.agg["data_sent_mb"])

    def check_facts(self, facts: dict[str, str] | None) -> None:
        """Every drawn per-run number against a facts table, when one is given."""
        if facts is None:
            return
        pairs = [
            ("fed_precision", self.precision("fed")),
            ("pt_precision", self.precision("clear")),
            ("whole_precision", self.whole("precision")),
            ("fed_f_score", self.f("fed")),
            ("pt_f_score", self.f("clear")),
            ("gap_f_score", self.gap),
            ("fed_edges", self.edges("fed")),
            ("whole_edges", self.whole("edges")),
            ("fed_rel", self.rel("fed")),
            ("whole_rel", self.whole("rel")),
        ]
        for name, value in pairs:
            key = f"{name} of {self.cell}"
            assert key in facts, f"facts file has no row {key!r}"
            recorded = float(facts[key])
            assert abs(recorded - value) < 5e-4, (
                f"{key}: facts {recorded} != campaign {value}")


def mean(values) -> float:
    return statistics.fmean(values)


def size_entropy(path: Path) -> float:
    """The entropy of a tier's cluster sizes in bits: the selector of Polyvyanyy,
    Moffat and Garcia-Banuelos (2020) taken over the clusters."""
    sizes = [int(c["size"]) for c in json.loads(path.read_text())["clusters"] if int(c.get("size") or 0) > 0]
    n = sum(sizes)
    assert n > 0, f"{path}: no case in any cluster"
    return -sum(s / n * math.log2(s / n) for s in sizes)


# --- the grid and the party-count family ---------------------------------

def grid_cell(log: str, emb: str, inst: str, seed: str | None = None) -> str:
    if inst == "sl_km":
        return f"{log}-{emb}_sl_km"
    if inst == "sl_seq":
        return f"{log}-{emb}_sl_seq"
    return f"ms_{log}_rp_{emb}_s{seed}"


def party_cell(log: str, camel: str, inst: str, parties: int,
               seed: str | None = None) -> str:
    if inst == "sl_km":
        return f"rt{camel}_sl_bigram_{parties}p"
    return f"ms_{log}-pc{parties}p_rp_bigram_s{seed}"


def collect(campaign: Path, facts_path: Path | None):
    mq = read_model_quality(campaign)
    agg = read_agg(campaign)
    facts = read_facts(facts_path) if facts_path is not None else None

    grid: dict[tuple[str, str, str], list[Run]] = {}
    for log, _name, cases, _camel, _pc in LOGS:
        for emb in EMBEDDINGS:
            if (log, emb) in REFUSED:
                for inst, _ in INSTANTIATIONS:
                    for seed in (SEEDS if inst == "rp" else [None]):
                        cell = grid_cell(log, emb, inst, seed)
                        assert (cell, "federated") not in mq, (
                            f"{cell} is scored but the combination is refused")
                        assert cell not in agg, f"{cell} is in agg_cells.csv"
                continue
            for inst, _ in INSTANTIATIONS:
                seeds = SEEDS if inst == "rp" else [None]
                runs = [Run(grid_cell(log, emb, inst, s), mq, agg, cases,
                            campaign) for s in seeds]
                for run in runs:
                    run.check_facts(facts)
                grid[(log, emb, inst)] = runs

    n_runs = sum(len(v) for v in grid.values())
    assert n_runs == 133, f"the grid has {n_runs} runs, expected 133"
    all_gaps = [r.gap for runs in grid.values() for r in runs]
    if facts is not None:
        assert facts["runs of the grid (four event logs, five embeddings, "
                     "three instantiations)"] == "133"
        assert abs(min(all_gaps) - float(
            facts["largest gap below the twin in F"])) < 5e-4
        assert abs(max(all_gaps) - float(
            facts["largest gap above the twin in F"])) < 5e-4
    assert not [g for g in all_gaps if g < LOSS_BOUND], (
        "a grid run beyond the loss bound")

    parties: dict[tuple[str, int, str], list[Run]] = {}
    for log, _name, _cases, camel, pc_cases in LOGS:
        for n_parties in (2, 3):
            for inst, _ in INSTANTIATIONS:
                if inst == "sl_seq":
                    continue  # the party-count family has no refinement runs
                seeds = SEEDS if inst == "rp" else [None]
                runs = [Run(party_cell(log, camel, inst, n_parties, s),
                            mq, agg, pc_cases, campaign) for s in seeds]
                for run in runs:
                    run.check_facts(facts)
                parties[(log, n_parties, inst)] = runs

    n_party = sum(len(v) for v in parties.values())
    assert n_party == 48, f"the party-count family has {n_party} runs, expected 48"
    return grid, parties


# --- writers --------------------------------------------------------------

def header(lines: list[str], what: str, cmd: str) -> None:
    lines.append(f"% {what}")
    lines.append(f"% Producing command: {cmd}")
    lines.append(f"% Campaign: output/campaign_uniform_{'2026-10-04'}, "
                 f"instrument {INSTRUMENT} (model_quality.csv); times and")
    lines.append("%   bytes from agg_cells.csv.  Asserted against the campaign's")
    lines.append("%   tables and, when --facts is given, against that facts table.")


def write(path: Path, lines: list[str], note: str, written: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    written.append(f"  {path}  ({note})")


def write_gap(out: Path, grid, cmd: str, written: list[str]) -> dict:
    summary = {}
    for log, name, cases, _camel, _pc in LOGS:
        lines: list[str] = []
        header(lines, f"Figure A, panel {name}: the gap in F of a run to its twin "
                      "in the clear", cmd)
        lines.append("% One row per embedding, yrow 4 (activity) down to 0")
        lines.append("%   (eventually-follows).  sl_km / sl_seq = the gap of the")
        lines.append("%   Secret-Labels k-means / refinement run; rp_mean = the mean")
        lines.append("%   gap of the five Revealed-Projection draws, rp_min / rp_max")
        lines.append("%   the lowest and the highest of the five, rp_em / rp_ep the")
        lines.append("%   same as a bar around the mean.  nan = refused at")
        lines.append("%   configuration time (no run exists).  y_km / y_seq / y_rp")
        lines.append(f"%   dodge the three marks of a row by {DODGE_Y:+.2f} so they do not")
        lines.append("%   cover one another.")
        lines.append("yrow embedding sl_km sl_seq rp_mean rp_em rp_ep rp_min rp_max "
                     "y_km y_seq y_rp")
        panel = {}
        for emb in EMBEDDINGS:
            yrow = 4 - EMBEDDINGS.index(emb)
            ys = (f"{yrow + DODGE_Y:.2f} {yrow:.2f} {yrow - DODGE_Y:.2f}")
            if (log, emb) in REFUSED:
                lines.append(f"{yrow} {emb} nan nan nan nan nan nan nan {ys}")
                panel[emb] = None
                continue
            km = grid[(log, emb, "sl_km")][0].gap
            seq = grid[(log, emb, "sl_seq")][0].gap
            draws = [r.gap for r in grid[(log, emb, "rp")]]
            m, lo, hi = mean(draws), min(draws), max(draws)
            lines.append(f"{yrow} {emb} {km:+.4f} {seq:+.4f} {m:+.4f} "
                         f"{m - lo:.4f} {hi - m:.4f} {lo:+.4f} {hi:+.4f} {ys}")
            panel[emb] = (km, seq, m, lo, hi)
        write(out / f"eval_gap_{log}_{STAMP}.dat", lines,
              f"{name}, 5 rows", written)
        summary[log] = panel
    return summary


QUALITY_COLS = [("precision", "precision"), ("edges", "edges"), ("rel", "rel")]
DODGE = {"fedtrac": -0.24, "clear": -0.08, "whole": 0.08, "random": 0.24}


def write_quality(out: Path, grid, cmd: str, written: list[str]) -> dict:
    per_log_runs = {}
    for log, _name, _cases, _camel, _pc in LOGS:
        per_log_runs[log] = [r for (lg, _e, _i), runs in grid.items()
                             if lg == log for r in runs]
    summary = {}
    for category in ("fedtrac", "clear", "whole", "random"):
        lines: list[str] = []
        header(lines, f"Figure B, the {category} marks: the published graphs "
                      "against the three references", cmd)
        lines.append("% x = the event log (1 Sepsis, 2 Helpdesk, 3 BPIC 2013,")
        lines.append(f"%   4 RTFM), dodged by {DODGE[category]:+.2f} so the four")
        lines.append("%   categories of a log stand side by side.  For every panel")
        lines.append("%   a value with the bar around it as mean, mean-min and")
        lines.append("%   max-mean (the bar is zero where one value exists).")
        lines.append("%   rel of a clustering (fedtrac, clear, random) adds the bits")
        lines.append("%   that name the cluster of a trace, the entropy of its")
        lines.append("%   cluster sizes; the whole log is one graph and adds none.")
        lines.append("x log prec prec_em prec_ep edges edges_em edges_ep "
                     "rel rel_em rel_ep")
        for idx, (log, name, _cases, _camel, _pc) in enumerate(LOGS, start=1):
            runs = per_log_runs[log]
            fields = []
            for col, _label in QUALITY_COLS:
                if category in ("fedtrac", "clear"):
                    tier = "fed" if category == "fedtrac" else "clear"
                    vals = [{"precision": r.precision(tier),
                             "edges": r.edges(tier),
                             "rel": r.rel_coded(tier)}[col] for r in runs]
                    m, lo, hi = mean(vals), min(vals), max(vals)
                elif category == "whole":
                    vals = {r.whole(col) for r in runs}
                    assert len(vals) == 1, (
                        f"{log}: the whole log's {col} is not one value: {vals}")
                    m = lo = hi = vals.pop()
                elif col == "rel":
                    m = mean(r.random_rel_coded("mean") for r in runs)
                    lo = min(r.random_rel_coded("min") for r in runs)
                    hi = max(r.random_rel_coded("max") for r in runs)
                else:
                    m = mean(r.random(col, "mean") for r in runs)
                    lo = min(r.random(col, "min") for r in runs)
                    hi = max(r.random(col, "max") for r in runs)
                fields.append((m, m - lo, hi - m))
                summary.setdefault(log, {})[(category, col)] = (m, lo, hi)
            flat = " ".join(f"{v:.4f}" for triple in fields for v in triple)
            lines.append(f"{idx + DODGE[category]:.2f} {log} {flat}")
        write(out / f"eval_quality_{category}_{STAMP}.dat", lines,
              "4 rows, one per event log", written)
    return summary


def write_cost(out: Path, grid, parties, cmd: str, written: list[str]) -> dict:
    summary = {"grid": {}, "parties": {}}
    for log, name, _cases, _camel, _pc in LOGS:
        lines: list[str] = []
        header(lines, f"Figure D, panel {name}: what a run costs", cmd)
        lines.append("% x = the embedding (1 activity, 2 bigram, 3 trigram,")
        lines.append("%   4 skip-bigram, 5 eventually-follows).  sl_time / rp_time")
        lines.append("%   = the time of the framework in seconds (Run.time: the secure")
        lines.append("%   programs with their compilation, plus the local clustering of")
        lines.append("%   a draw; without the tiers in the clear and the in-run")
        lines.append("%   evaluation of the cell) for the")
        lines.append("%   Secret-Labels k-means run and for the Revealed-Projection")
        lines.append(f"%   draw run in full (s{FULL_SEED}; the other four draws tail its")
        lines.append("%   stored first round, so only this one is a full run's time).")
        lines.append("%   sl_sent / rp_sent = the data one party sends [MB],")
        lines.append("%   sl_round1 / rp_round1 = the first round's secure time [s].")
        lines.append("%   nan = refused at configuration time.")
        lines.append("x embedding sl_time rp_time sl_sent rp_sent sl_round1 rp_round1")
        panel = {}
        for idx, emb in enumerate(EMBEDDINGS, start=1):
            if (log, emb) in REFUSED:
                lines.append(f"{idx} {emb} nan nan nan nan nan nan")
                panel[emb] = None
                continue
            sl = grid[(log, emb, "sl_km")][0]
            rp = next(r for r in grid[(log, emb, "rp")]
                      if r.cell.endswith(f"_s{FULL_SEED}"))
            lines.append(f"{idx} {emb} {sl.time():.0f} {rp.time():.0f} "
                         f"{sl.sent():.1f} {rp.sent():.1f} "
                         f"{sl.round1():.1f} {rp.round1():.1f}")
            panel[emb] = (sl.time(), rp.time(), sl.sent(), rp.sent())
        write(out / f"eval_cost_{log}_{STAMP}.dat", lines,
              f"{name}, 5 rows", written)
        summary["grid"][log] = panel

    for log, name, _cases, camel, _pc in LOGS:
        lines = []
        header(lines, f"Figure D, the parties panel, {name} under bigram", cmd)
        lines.append("% x = the number of parties.  sl_time = the Secret-Labels")
        lines.append(f"%   k-means run rt{camel}_sl_bigram_<n>p, rp_time = the")
        lines.append("%   Revealed-Projection draw run in full")
        lines.append(f"%   ms_{log}-pc<n>p_rp_bigram_s{FULL_SEED}.  sl_sent / rp_sent the")
        lines.append("%   data one party sends [MB], sl_round1 / rp_round1 the first")
        lines.append("%   round's secure time [s].")
        lines.append("parties sl_time rp_time sl_sent rp_sent sl_round1 rp_round1")
        panel = {}
        for n_parties in (2, 3):
            sl = parties[(log, n_parties, "sl_km")][0]
            rp = next(r for r in parties[(log, n_parties, "rp")]
                      if r.cell.endswith(f"_s{FULL_SEED}"))
            lines.append(f"{n_parties} {sl.time():.0f} {rp.time():.0f} "
                         f"{sl.sent():.1f} {rp.sent():.1f} "
                         f"{sl.round1():.1f} {rp.round1():.1f}")
            panel[n_parties] = (sl.time(), rp.time())
        write(out / f"eval_cost_parties_{log}_{STAMP}.dat", lines,
              f"{name}, 2 rows", written)
        summary["parties"][log] = panel
    return summary


# --- the two appendix tables ---------------------------------------------

def k_label(runs: list[Run]) -> str:
    ks = sorted({r.k for r in runs})
    return str(ks[0]) if len(ks) == 1 else f"{ks[0]} to {ks[-1]}"


def row_values(runs: list[Run]) -> list[str]:
    """precision ours/clear, F ours/clear, edges ours/whole, relevance
    ours/whole -- the mean over the runs of the combination."""
    return [
        f"{mean(r.precision('fed') for r in runs):.3f}",
        f"{mean(r.precision('clear') for r in runs):.3f}",
        f"{mean(r.f('fed') for r in runs):.3f}",
        f"{mean(r.f('clear') for r in runs):.3f}",
        f"{mean(r.edges('fed') for r in runs):.1f}",
        f"{mean(r.whole('edges') for r in runs):.1f}",
        f"{mean(r.rel_coded('fed') for r in runs):.3f}",
        f"{mean(r.whole('rel') for r in runs):.3f}",
    ]


TABLE_HEAD = r"""% Appendix table: every combination of the grid of campaign_uniform_2026-10-04
% under the model-quality instrument %INSTRUMENT%.  Written by
%   %CMD%
% \input by the appendix; no table wrapper, only the tabular.
% One row per combination of event log, embedding and instantiation; the event
% log heads its block with the population all its runs share.  A
% Revealed-Projection row is the mean over its five draws, and k is the
% resolved cluster count, a range where the draws differ.  "FedTrac" is the
% published graph of the federated run, "clear" its twin computed in the clear
% on the same cases, "whole" the whole log filtered once.  Relevance is the
% entropic relevance of the graph to the traces in bits per trace (lower is
% better), for FedTrac plus the bits that name the cluster of a trace (the
% entropy of the cluster sizes; the whole log is one graph and adds none);
% edges are the edges of one published graph.
% Set at \scriptsize with the reference sub-headers rotated: at \footnotesize
% and with the event log as a twelfth column the tabular measures 20.4 cm
% against a text width of 14.7 cm.  The sizes are set inside a group, so they
% end with the tabular.
\begingroup
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.9}
\newcommand{\rothead}[1]{\rotatebox{90}{#1}}
\begin{tabular}{@{}l l c @{\hspace{5pt}} r r @{\hspace{5pt}} r r @{\hspace{5pt}} r r @{\hspace{5pt}} r r@{}}
\toprule
 & & & \multicolumn{2}{c}{precision} & \multicolumn{2}{c}{$F$}
 & \multicolumn{2}{c}{edges} & \multicolumn{2}{c}{relevance} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
embedding & instantiation & $k$
 & \rothead{FedTrac} & \rothead{clear} & \rothead{FedTrac} & \rothead{clear}
 & \rothead{FedTrac} & \rothead{whole} & \rothead{FedTrac} & \rothead{whole} \\
\midrule
"""

TABLE_FOOT = r"""\bottomrule
\end{tabular}
\endgroup
"""


def write_table_grid(out: Path, grid, cmd: str, written: list[str]) -> int:
    body: list[str] = []
    n_rows = 0
    for li, (log, name, cases, _camel, _pc) in enumerate(LOGS):
        if li:
            body.append(r"\addlinespace")
        body.append(r"\multicolumn{11}{@{}l}{\itshape " + name
                    + r"\quad " + f"{cases:,}".replace(",", "{,}")
                    + r" cases} \\")
        for emb in EMBEDDINGS:
            if (log, emb) in REFUSED:
                body.append(f"{EMB_LABEL[emb]} & "
                            r"\multicolumn{10}{l}{refused at configuration "
                            r"time} \\")
                continue
            first_emb = True
            for inst, inst_label in INSTANTIATIONS:
                runs = grid[(log, emb, inst)]
                cells = [
                    EMB_LABEL[emb] if first_emb else "",
                    inst_label,
                    k_label(runs),
                ] + row_values(runs)
                body.append(" & ".join(cells) + r" \\")
                n_rows += 1
                first_emb = False
    assert n_rows == 57, f"the grid table has {n_rows} rows, expected 57"
    text = (TABLE_HEAD.replace("%CMD%", cmd).replace("%INSTRUMENT%", INSTRUMENT)
            + "\n".join(body) + "\n" + TABLE_FOOT)
    path = out / f"uniform_table_grid_{STAMP}.tex"
    path.write_text(text)
    written.append(f"  {path}  ({n_rows} combinations)")
    return n_rows


PARTY_HEAD = r"""% Appendix table: the party-count family of campaign_uniform_2026-10-04 under
% the model-quality instrument %INSTRUMENT%.  Written by
%   %CMD%
% \input by the appendix; no table wrapper, only the tabular.
% Every run of this family uses the bigram embedding, and the four populations
% are smaller than the ones of the grid table, so the two tables do not
% compare.  The event log heads its block with the population its runs share.
% A Revealed-Projection row is the mean over its five draws and its time is
% the one draw that runs in full (seed 42); the other four tail its stored
% first round.  "time" is the time of the framework (Run.time): the secure
% programs with their compilation, plus the local clustering of a draw.  Columns
% as in the grid table.
\begingroup
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.9}
\newcommand{\rothead}[1]{\rotatebox{90}{#1}}
\begin{tabular}{@{}c l c @{\hspace{5pt}} r r @{\hspace{5pt}} r r @{\hspace{5pt}} r r @{\hspace{5pt}} r r @{\hspace{5pt}} r@{}}
\toprule
 & & & \multicolumn{2}{c}{precision} & \multicolumn{2}{c}{$F$}
 & \multicolumn{2}{c}{edges} & \multicolumn{2}{c}{relevance} & \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
parties & instantiation & $k$
 & \rothead{FedTrac} & \rothead{clear} & \rothead{FedTrac} & \rothead{clear}
 & \rothead{FedTrac} & \rothead{whole} & \rothead{FedTrac} & \rothead{whole}
 & \rothead{time [s]} \\
\midrule
"""


def write_table_parties(out: Path, parties, cmd: str, written: list[str]) -> int:
    body: list[str] = []
    n_rows = 0
    for li, (log, name, _cases, _camel, pc_cases) in enumerate(LOGS):
        if li:
            body.append(r"\addlinespace")
        body.append(r"\multicolumn{12}{@{}l}{\itshape " + name
                    + r"\quad " + f"{pc_cases:,}".replace(",", "{,}")
                    + r" cases, bigram} \\")
        for n_parties in (2, 3):
            first_p = True
            for inst, inst_label in INSTANTIATIONS:
                if inst == "sl_seq":
                    continue
                runs = parties[(log, n_parties, inst)]
                full = next((r for r in runs
                             if r.cell.endswith(f"_s{FULL_SEED}")), runs[0])
                cells = [
                    str(n_parties) if first_p else "",
                    inst_label,
                    k_label(runs),
                ] + row_values(runs) + [f"{full.time():.0f}"]
                body.append(" & ".join(cells) + r" \\")
                n_rows += 1
                first_p = False
    assert n_rows == 16, f"the parties table has {n_rows} rows, expected 16"
    text = (PARTY_HEAD.replace("%CMD%", cmd).replace("%INSTRUMENT%", INSTRUMENT)
            + "\n".join(body) + "\n" + TABLE_FOOT)
    path = out / f"uniform_table_parties_{STAMP}.tex"
    path.write_text(text)
    written.append(f"  {path}  ({n_rows} rows)")
    return n_rows


# --- main -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, type=Path,
                    help="output/campaign_uniform_2026-10-04")
    ap.add_argument("--out", required=True, type=Path,
                    help="directory for the .dat and .tex files")
    ap.add_argument("--facts", type=Path, default=None,
                    help="a facts table to assert every drawn number against")
    args = ap.parse_args(argv)

    facts_path = args.facts
    cmd = (f".venv/bin/python scripts/uniform_figures.py"
           f" --campaign {args.campaign} --out {args.out}")

    grid, parties = collect(args.campaign, facts_path)

    written: list[str] = []
    gap = write_gap(args.out, grid, cmd, written)
    quality = write_quality(args.out, grid, cmd, written)
    cost = write_cost(args.out, grid, parties, cmd, written)
    write_table_grid(args.out, grid, cmd, written)
    write_table_parties(args.out, parties, cmd, written)

    checked = (f", checked against {facts_path.name}"
               if facts_path is not None else "")
    print(f"uniform_figures.py: {len(written)} files from {args.campaign} "
          f"(instrument {INSTRUMENT}){checked}")
    for line in written:
        print(line)

    print("\nFigure A, the gap in F to the twin (federated minus clear):")
    for log, name, cases, _c, _p in LOGS:
        parts = []
        for emb in EMBEDDINGS:
            v = gap[log][emb]
            if v is None:
                parts.append(f"{EMB_LABEL[emb]} refused")
                continue
            km, seq, m, lo, hi = v
            parts.append(f"{EMB_LABEL[emb]} km {km:+.3f} ref {seq:+.3f} "
                         f"draws {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  {name} (N={cases}): " + "; ".join(parts))

    print("\nFigure B, the published graphs against the three references:")
    for log, name, _cases, _c, _p in LOGS:
        for col, label in (("precision", "precision"), ("edges", "edges"),
                           ("rel", "relevance [bits]")):
            bits = []
            for cat in ("fedtrac", "clear", "whole", "random"):
                m, lo, hi = quality[log][(cat, col)]
                bits.append(f"{cat} {m:.3f} [{lo:.3f},{hi:.3f}]")
            print(f"  {name} {label}: " + "  ".join(bits))

    print("\nThe relevance with the bits that name the cluster: runs with fewer bits than the whole log")
    for log, name, _cases, _c, _p in LOGS:
        runs = [r for (lg, _e, _i), rs in grid.items() if lg == log for r in rs]
        fewer = sum(1 for r in runs if r.rel_coded("fed") < r.whole("rel"))
        raw = sum(1 for r in runs if r.rel("fed") < r.whole("rel"))
        print(f"  {name}: {fewer} of {len(runs)} (without the naming bits {raw})")

    print("\nFigure D, wall-clock time per run [s] "
          "(Secret-Labels k-means / Revealed-Projection in full):")
    for log, name, _cases, _c, _p in LOGS:
        parts = []
        for emb in EMBEDDINGS:
            v = cost["grid"][log][emb]
            if v is None:
                parts.append(f"{EMB_LABEL[emb]} refused")
            else:
                parts.append(f"{EMB_LABEL[emb]} {v[0]:.0f}/{v[1]:.0f}")
        print(f"  {name}: " + "; ".join(parts))
    print("  parties (2 / 3):")
    for log, name, _cases, _c, _p in LOGS:
        p = cost["parties"][log]
        print(f"    {name}: Secret-Labels {p[2][0]:.0f} -> {p[3][0]:.0f}, "
              f"Revealed-Projection {p[2][1]:.0f} -> {p[3][1]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
