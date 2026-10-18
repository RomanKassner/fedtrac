#!/usr/bin/env python3
"""Roll the per-run benchmark JSON of a campaign up into thesis-citable tables."""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.metrics import transition_log_likelihood  # noqa: E402

TIERS = ("plaintext", "projected", "federated")
REPEAT_RE = re.compile(r"_r(\d+)$")
D4_MS_RE = re.compile(
    r"^ms_(?P<log>.+)_(?P<mode>sl|rp)_(?P<enc>[a-z]+)_s(?P<seed>\d+)$"
)
D4_ENC_RE = re.compile(r"^enc_(?P<log>.+)_(?P<mode>sl|rp)_(?P<enc>[a-z]+)$")
D4_RT_RE = re.compile(
    r"^rt(?P<log>[A-Za-z0-9]+)_(?P<mode>sl|rp)_(?P<enc>[a-z]+)_(?P<suffix>(?:org)?\d+p)$"
)
D4_MODE = {"sl": "sk", "rp": "rsec"}
# Sequence-campaign names carry the middle step, not the encoding (read from
# the manifest flags instead); `km` and `kauto` are different middle steps.
SEQ_RE = re.compile(r"^(?P<log>.+)_sl(?:_(?P<variant>seq_stride|seq|km|kauto))?$")
UNIVERSE_RE = re.compile(r"--case-universe[=\s]+(\w+)")
ENCODING_RE = re.compile(r"--encoding[=\s]+([a-z-]+)")
HEADLINE_FIELDS = (
    "f_hm_all", "f_hm_sound_only", "buijs_hm", "sound_rate",
    "entropic_relevance_hm", "edge_reduction", "imbalance_gini", "data_bound",
)
TIMING_FIELDS = ("mpc_compile_s", "mpc_exec_s", "mpc_exec_s_kselect",
                 "round1_exec_s", "round1_restarts", "round1_restart_exec_s",
                 "data_sent_mb", "global_data_sent_mb",
                 "rounds_total", "generation", "party_flags")
# Per-candidate Lloyd's probes of the retired in-box `--k auto`; their cost is
# summed apart from the pinned-k rounds.
KSELECT_ROUND_RE = re.compile(r"^secure_kmeans_\d+$")
# A Round 1 the protocol's restart rule discarded: excluded from every round
# column and summed into `round1_restart_exec_s`.
DEGENERATE_ROUND_RE = re.compile(r"^round1_rsecret_degenerate(_\d+)?$")


def load_partition(tier_dir: Path):
    """(traces, labels, k, ids->label) from a tier dir; k is the label range."""
    files = sorted(glob.glob(str(tier_dir / "cluster_*_traces.json")),
                   key=lambda p: int(re.search(r"cluster_(\d+)_traces", p).group(1)))
    traces, labels = [], []
    for f in files:
        cid = int(re.search(r"cluster_(\d+)_traces", f).group(1))
        for tr in json.load(open(f))["traces"]:
            traces.append(tr)
            labels.append(cid)
    a = json.load(open(tier_dir / "cluster_assignments.json"))
    by_id = dict(zip(a["case_ids"], a["assignments"]))
    k = max(len(files), (max(labels) + 1) if labels else 0)
    return traces, labels, k, by_id


def group_by_label(traces, labels, k) -> list[list[list[str]]]:
    per_cluster = [[] for _ in range(k)]
    for tr, lab in zip(traces, labels):
        per_cluster[lab].append(tr)
    return per_cluster


def fit_log2(traces, labels, k) -> float:
    """The fit column: the Laplace-smoothed log2 objective per transition,
    weighted, START/END included, unfiltered, no floor."""
    return transition_log_likelihood(group_by_label(traces, labels, k))


def parse_cell(cell):
    """cell name -> (log, mode, encoding-or-None, repeat-or-None, variant)."""
    m = D4_MS_RE.match(cell)
    if m:
        return (f"ms:{m['log']}", D4_MODE[m['mode']], m['enc'],
                int(m['seed']), "")
    m = D4_ENC_RE.match(cell)
    if m:
        return f"enc:{m['log']}", D4_MODE[m['mode']], m['enc'], None, ""
    m = D4_RT_RE.match(cell)
    if m:
        return (f"rt:{m['log'].lower()}-{m['suffix']}", D4_MODE[m['mode']],
                m['enc'], None, "")
    m = SEQ_RE.match(cell)
    if m:
        return (m['log'], D4_MODE['sl'], None, None,
                (m['variant'] or "").replace("_", "-"))
    repeat = None
    m = REPEAT_RE.search(cell)
    if m:
        repeat = int(m.group(1))
        cell = cell[: m.start()]
    mode = "rsec" if "_rsec_" in cell else ("sk" if "_sk_" in cell else "?")
    sep = f"_{mode}_"
    if sep in cell:
        log, encoding = cell.split(sep, 1)
    else:
        log, encoding = cell, "?"
    return log, mode, encoding, repeat, ""


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_tsv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def manifest_runs(campaign_dir):
    """{cell: run_dir} from manifest.tsv; later rows win (re-runs)."""
    out = {}
    for row in load_tsv(os.path.join(campaign_dir, "manifest.tsv")):
        if row.get("run_dir") and row["run_dir"] != "-":
            out[row["cell"]] = row["run_dir"]
    return out


def meta_case_universe(campaign_dir):
    """Campaign-level `case_universe:` from campaign_meta.txt; last line wins."""
    val = None
    try:
        with open(os.path.join(campaign_dir, "campaign_meta.txt")) as fh:
            for line in fh:
                if line.startswith("case_universe:"):
                    val = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return val or "intersection"


def manifest_case_universes(campaign_dir, default):
    """{cell: case_universe}; union and intersection cells must never pool."""
    out = {}
    for row in load_tsv(os.path.join(campaign_dir, "manifest.tsv")):
        if not row.get("cell"):
            continue
        m = UNIVERSE_RE.search(row.get("flags") or "")
        out[row["cell"]] = row.get("case_universe") or (m.group(1) if m else default)
    return out


def manifest_encodings(campaign_dir):
    """{cell: encoding} from the manifest `flags` column."""
    out = {}
    for row in load_tsv(os.path.join(campaign_dir, "manifest.tsv")):
        m = ENCODING_RE.search(row.get("flags") or "")
        if row.get("cell") and m:
            out[row["cell"]] = m.group(1)
    return out


def manifest_ring_overflow(campaign_dir, summary_rows):
    """{cell: "0"|"1"|"?"}: did the cell bypass the secure_kmeans argmin
    ring-width guard? Only the driver's column records it, so `?` is not `0`."""
    out = {}
    for row in load_tsv(os.path.join(campaign_dir, "manifest.tsv")) + list(summary_rows):
        cell = row.get("cell")
        v = row.get("ring_overflow")
        if not cell or v in (None, ""):
            continue
        out.setdefault(cell, "1" if str(v).strip() in ("1", "true", "True", "yes") else "0")
    return out


def baseline_k(run_dir):
    """The k the plaintext baseline tier actually clustered at."""
    a = load_json(os.path.join(run_dir, "plaintext", "cluster_assignments.json"))
    if a and a.get("assignments"):
        return len(set(a["assignments"]))
    d = load_json(os.path.join(run_dir, "plaintext", "evaluation_results.json"))
    if d and d.get("clusters_hm"):
        return len(d["clusters_hm"])
    return None


def harness_wallclock(meta):
    """Measured harness wall time for a cell: `wallclock_s`, else `wall_s`."""
    return meta.get("wallclock_s") or meta.get("wall_s")


def tier_headline(run_dir, tier):
    d = load_json(os.path.join(run_dir, tier, "evaluation_results.json"))
    if not d:
        return None
    return (d.get("quality_metrics") or {}).get("headline") or {}


def tier_fit_log2(run_dir, tier):
    """Mean transition log-probability over a tier's exported cluster traces;
    None (never 0) when the tier exported none."""
    tier_dir = os.path.join(run_dir, tier)
    if not glob.glob(os.path.join(tier_dir, "cluster_*_traces.json")):
        return None
    try:
        traces, labels, k, _ = load_partition(Path(tier_dir))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return fit_log2(traces, labels, k) if traces else None


def timing_columns(run_dir):
    """Flatten timing.json into campaign-table columns (None-safe)."""
    t = load_json(os.path.join(run_dir, "timing.json")) or {}
    rounds = [v for k, v in t.items()
              if isinstance(v, dict) and not KSELECT_ROUND_RE.match(k)
              and not DEGENERATE_ROUND_RE.match(k)]
    kselect = [v for k, v in t.items()
               if isinstance(v, dict) and KSELECT_ROUND_RE.match(k)]
    degenerate = [v for k, v in t.items()
                  if isinstance(v, dict) and DEGENERATE_ROUND_RE.match(k)]
    if not rounds and not kselect:
        return dict.fromkeys(TIMING_FIELDS)
    def s(key, src=None):
        vals = [r.get(key) for r in (rounds if src is None else src)
                if r.get(key) is not None]
        return round(sum(vals), 1) if vals else None
    r1 = next((v for k, v in t.items()
               if k.startswith("round1") and not DEGENERATE_ROUND_RE.match(k)), {})
    # Times compare only within one optimization generation and one runtime-flag
    # setting, so both are columns.
    flags = t.get("_party_flags")
    return {
        "mpc_compile_s": s("compile_s"),
        "mpc_exec_s": s("exec_s"),
        "mpc_exec_s_kselect": s("exec_s", kselect),
        "round1_exec_s": round(r1.get("exec_s"), 1) if r1.get("exec_s") else None,
        "round1_restarts": t.get("_round1_restarts"),
        "round1_restart_exec_s": s("exec_s", degenerate),
        "data_sent_mb": s("data_sent_mb"),
        "global_data_sent_mb": s("global_data_sent_mb"),
        "rounds_total": s("rounds"),
        "generation": t.get("_generation", 1),
        "party_flags": " ".join(flags) if isinstance(flags, list) else None,
    }


def collect(runs, summary_rows, universes=None, default_universe="intersection",
            encodings=None, ring_overflows=None):
    summary = {r["cell"]: r for r in summary_rows}
    universes = universes or {}
    encodings = encodings or {}
    ring_overflows = ring_overflows or {}
    long_rows, cell_rows, failed = [], [], []
    for cell, run_dir in sorted(runs.items()):
        log, mode, encoding, repeat, variant = parse_cell(cell)
        if mode == "?":
            print(f"WARNING: cell {cell!r} matches no known cell-name grammar; "
                  "it is in NO table (fix the name or the grammar)",
                  file=sys.stderr)
            continue
        if encoding is None:
            encoding = encodings.get(cell, "?")
        meta = summary.get(cell)
        # A manifest cell with no summary row is a failure, not a success with
        # empty columns.
        if meta is None:
            failed.append({"cell": cell, "exit": "missing", "wallclock_s": None,
                           "run_dir": run_dir})
            continue
        if meta.get("exit") not in (None, "", "0"):
            failed.append({"cell": cell, "exit": meta.get("exit"),
                           "wallclock_s": harness_wallclock(meta),
                           "run_dir": run_dir})
            continue
        comp = load_json(os.path.join(run_dir, "plaintext", "comparison.json")) or {}
        present = [t for t in TIERS if os.path.isdir(os.path.join(run_dir, t))]
        fits = {t: tier_fit_log2(run_dir, t) for t in present}
        for tier in present:
            h = tier_headline(run_dir, tier)
            if h is None:
                continue
            row = {"cell": cell, "log": log, "mode": mode, "encoding": encoding,
                   "variant": variant, "repeat": repeat, "tier": tier}
            for f in HEADLINE_FIELDS:
                row[f] = h.get(f)
            row["fit_log2"] = fits.get(tier)
            long_rows.append(row)

        fed = tier_headline(run_dir, "federated") or {}
        pt = tier_headline(run_dir, "plaintext") or {}
        # *_f_hm is the HeuristicsMiner headline, *_f_im an Inductive-Miner
        # weighted-average delta; a missing tier metric is None, never 0.
        pvf = comp.get("plaintext_vs_federated", {})
        pvp = comp.get("plaintext_vs_projected", {})
        pvf_proj = comp.get("projected_vs_federated", {})
        proj = tier_headline(run_dir, "projected") or {}
        def _dhm(a, b):
            try:
                return round(float(b.get("f_hm_all")) - float(a.get("f_hm_all")), 4)
            except (TypeError, ValueError):
                return None
        delta_f_hm = _dhm(pt, fed)
        # HM twins of the JL / MPC split of dF.
        jl_cost_f_hm = _dhm(pt, proj)
        mpc_cost_f_hm = _dhm(proj, fed)
        def _dfit(a, b):
            return None if a is None or b is None else round(b - a, 4)
        fed_eval = load_json(
            os.path.join(run_dir, "federated", "evaluation_results.json")) or {}
        wavg_hm = fed_eval.get("weighted_avg_hm") or {}
        glob_hm = fed_eval.get("global_hm") or {}
        def _delta(field):
            try:
                return round(float(wavg_hm[field]) - float(glob_hm[field]), 4)
            except (TypeError, ValueError, KeyError):
                return None
        conf = fed_eval.get("conformance") or {}
        cov = fed_eval.get("coverage_hm") or {}
        purity = (fed_eval.get("quality_metrics") or {}).get("variant_purity") or {}
        def _homog(scheme):
            try:
                return round(float(purity[scheme]["homogeneity"]), 4)
            except (TypeError, ValueError, KeyError):
                return None
        cell_rows.append({
            "cell": cell, "log": log, "mode": mode, "encoding": encoding,
            "variant": variant, "repeat": repeat,
            "k": meta.get("k"), "k_priv": meta.get("k_priv"),
            "suppressed": meta.get("suppressed"),
            "case_universe": meta.get("case_universe")
                             or universes.get(cell, default_universe),
            "wallclock_s": harness_wallclock(meta),
            **timing_columns(run_dir),
            "pt_f_hm": pt.get("f_hm_all"),
            "fed_f_hm": fed.get("f_hm_all"),
            "fed_f_sound": fed.get("f_hm_sound_only"),
            "delta_f_hm": delta_f_hm,
            "delta_f_im": pvf.get("f_score"),
            "jl_cost_f_hm": jl_cost_f_hm,
            "mpc_cost_f_hm": mpc_cost_f_hm,
            "jl_cost_f_im": pvp.get("f_score"),
            "mpc_cost_f_im": pvf_proj.get("f_score"),
            "pt_fit_log2": fits.get("plaintext"),
            "proj_fit_log2": fits.get("projected"),
            "fed_fit_log2": fits.get("federated"),
            "jl_cost_fit_log2": _dfit(fits.get("plaintext"), fits.get("projected")),
            "mpc_cost_fit_log2": _dfit(fits.get("projected"), fits.get("federated")),
            "fed_er": fed.get("entropic_relevance_hm"),
            "fed_edge_red": fed.get("edge_reduction"),
            "fed_imbalance_gini": fed.get("imbalance_gini"),
            "ari": comp.get("ari"), "nmi": comp.get("nmi"),
            "global_dfg_l1": comp.get("global_dfg_l1"),
            "delta_f_global": _delta("f_score"),
            "delta_simpl_global": _delta("simplicity"),
            "purity_length": _homog("length"),
            "purity_activity_set": _homog("activity_set"),
            "purity_loop": _homog("loop"),
            "tiers_present": "+".join(present),
            "baseline_k": baseline_k(run_dir),
            "ring_overflow": ring_overflows.get(cell, "?"),
            "token_replay": conf.get("token_replay"),
            "fast_eval": conf.get("fast_eval"),
            "eval_coverage": cov.get("eval_coverage"),
            "dropped_clusters": len(cov.get("dropped_clusters") or []),
            "run_dir": os.path.basename(run_dir),
        })
    return long_rows, cell_rows, failed


def repeats_summary(cell_rows):
    """Group repeats -> mean +/- std of federated F and fit; a group mixing
    resolved k is flagged, its mean conflating two variances."""
    groups = {}
    for r in cell_rows:
        key = (r["log"], r["mode"], r["encoding"], r.get("variant", ""))
        try:
            f = float(r["fed_f_hm"])
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, []).append((f, r.get("k"), r.get("fed_fit_log2")))
    out = []
    for (log, mode, enc, variant), vals in sorted(groups.items()):
        fs = [f for f, _, _ in vals]
        ks = sorted({str(k) for _, k, _ in vals if k not in (None, "", "-")})
        fits = [v for _, _, v in vals if v is not None]
        out.append({
            "log": log, "mode": mode, "encoding": enc, "variant": variant,
            "n": len(fs),
            "fed_f_mean": round(statistics.mean(fs), 4),
            "fed_f_std": round(statistics.stdev(fs), 4) if len(fs) > 1 else 0.0,
            "fed_fit_log2_mean": round(statistics.mean(fits), 4) if fits else None,
            "fed_fit_log2_std": (round(statistics.stdev(fits), 4)
                                 if len(fits) > 1 else (0.0 if fits else None)),
            "k_set": "+".join(ks) if ks else "-",
            "k_mixed": len(ks) > 1,
        })
        if len(ks) > 1:
            print(f"WARNING: repeat group {log}/{mode}/{enc}/{variant} mixes k in "
                  f"{{{','.join(ks)}}} — mean conflates k-selection variance",
                  file=sys.stderr)
    return out


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--" if v in (None, "") else str(v)


def cell_order(row):
    return (row["log"], row["mode"], row["encoding"], row.get("variant", ""),
            row["repeat"] or 0)


def latex_table(cell_rows):
    lines = [
        r"\begin{tabular}{lllrrrrrr}", r"\toprule",
        r"Log & Enc & Mode & $k$ & PT $F$ & Fed $F$ & $\Delta F$ & ER & edge-red \\",
        r"\midrule",
    ]
    for r in sorted(cell_rows, key=cell_order):
        enc = r["encoding"].replace("_", r"\_")
        lines.append(" & ".join([
            r["log"].replace("_", r"\_"), enc, r["mode"],
            fmt(r["k"], 0), fmt(r["pt_f_hm"]), fmt(r["fed_f_hm"]),
            fmt(r["delta_f_hm"]), fmt(r["fed_er"], 1), fmt(r["fed_edge_red"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def print_console(cell_rows, reps, failed):
    cols = ["log", "encoding", "mode", "k", "baseline_k", "ring_overflow",
            "k_priv", "suppressed", "pt_f_hm", "fed_f_hm", "delta_f_hm",
            "jl_cost_f_hm", "mpc_cost_f_hm", "fed_fit_log2",
            "eval_coverage", "token_replay",
            "fed_er", "ari", "nmi", "mpc_exec_s", "global_data_sent_mb"]
    # Flags stay verbatim: "1.000" / "True" would read as measurements.
    raw = {"ring_overflow", "token_replay"}
    def cell(r, c):
        if c in raw:
            v = r.get(c)
            return "--" if v is None else str(v)
        return fmt(r.get(c), 1 if c in ("fed_er", "mpc_exec_s", "global_data_sent_mb")
                   else 3)
    w = {c: max(len(c), *(len(cell(r, c)) for r in cell_rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in sorted(cell_rows, key=cell_order):
        print("  ".join(cell(r, c).ljust(w[c]) for c in cols))
    print("\n--- repeat groups: federated F and fit mean +/- std (k-aware) ---")
    for g in reps:
        flag = "  [MIXED k!]" if g["k_mixed"] else ""
        fit = ("--" if g["fed_fit_log2_mean"] is None else
               f"{g['fed_fit_log2_mean']:.4f} +/- {g['fed_fit_log2_std']:.4f}")
        print(f"  {g['log']:<14} {g['mode']:<5} {g['encoding']:<12} n={g['n']} "
              f"k={g['k_set']:<5} F = {g['fed_f_mean']:.4f} +/- {g['fed_f_std']:.4f}"
              f"  fit = {fit}{flag}")
    if failed:
        print("\n--- failed cells (excluded from all tables) ---")
        for f in failed:
            print(f"  {f['cell']}: exit={f['exit']} ({f['wallclock_s']}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True,
                    help="campaign dir with manifest.tsv + summary.tsv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    campaign = args.campaign
    if not os.path.isdir(campaign):
        print(f"No campaign dir at {campaign}", file=sys.stderr)
        return 1
    summary_rows = load_tsv(os.path.join(campaign, "summary.tsv"))

    runs = manifest_runs(campaign)
    if not runs:
        print(f"{campaign}/manifest.tsv missing or empty.", file=sys.stderr)
        return 1

    default_universe = meta_case_universe(campaign)
    long_rows, cell_rows, failed = collect(
        runs, summary_rows, manifest_case_universes(campaign, default_universe),
        default_universe, manifest_encodings(campaign),
        manifest_ring_overflow(campaign, summary_rows))
    if not cell_rows:
        print("No successful cells found", file=sys.stderr)
        return 1
    reps = repeats_summary(cell_rows)

    out_dir = args.out_dir or campaign
    os.makedirs(out_dir, exist_ok=True)
    long_fields = ["cell", "log", "mode", "encoding", "variant", "repeat", "tier",
                   *HEADLINE_FIELDS, "fit_log2"]
    write_csv(os.path.join(out_dir, "agg_long.csv"), long_rows, long_fields)
    write_csv(os.path.join(out_dir, "agg_cells.csv"), cell_rows,
              list(cell_rows[0].keys()))
    write_csv(os.path.join(out_dir, "agg_repeats.csv"), reps,
              ["log", "mode", "encoding", "variant", "n", "fed_f_mean",
               "fed_f_std", "fed_fit_log2_mean", "fed_fit_log2_std",
               "k_set", "k_mixed"])
    write_csv(os.path.join(out_dir, "agg_failed.csv"), failed,
              ["cell", "exit", "wallclock_s", "run_dir"])
    if args.latex:
        with open(os.path.join(out_dir, "agg_headline.tex"), "w") as fh:
            fh.write(latex_table(cell_rows) + "\n")

    print(f"campaign: {campaign}  cells: {len(cell_rows)} ok, {len(failed)} failed, "
          f"tiers x cells: {len(long_rows)}\n")
    print_console(cell_rows, reps, failed)
    print(f"\nwrote: {out_dir}/agg_{{long,cells,repeats,failed}}.csv"
          + ("  + agg_headline.tex" if args.latex else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
