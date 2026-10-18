#!/usr/bin/env python3
"""Rescores an attack battery in the clear against each cell's own encoded matrix and writes the recovery table, the figure data and the two appendix tables."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_attack_battery import ROW_RE, frequency_guessing_baseline  # noqa: E402

BIN = REPO / "target/release/clustering_federated"
KEEP_FLAGS = {"--data-dir", "--case-ids", "--encoding", "--n-parties", "--max-trace-len"}
M_LIST = [5, 20, 50]
COLS = (["cell", "source", "encoding", "N", "opened", "anonymous_set", "align_ok", "k_priv", "k_proj",
         "null_P", "null_R", "null_cols", "jl_med", "jl_w25", "pinv_P", "pinv_R"]
        + [f"{a}_m{m}_{x}" for a in ("norm", "dist") for m in M_LIST for x in ("P", "R")]
        + ["oracle_rows", "sim_ica_P", "sim_ica_R", "sim_mds_P", "sim_mds_R", "sim_k", "run_dir", "truth", "log"]
        + ["freq_P", "freq_R", "freq_class_acc", "freq_row_acc", "freq_class_acc_exp", "freq_row_acc_exp",
           "freq_classes", "freq_tied_classes", "freq_tied_rows", "freq_sizes_ok", "freq_oracle_rows"])

LOGS = ("sepsis", "helpdesk", "bpic2013", "rtfm")
EMBEDDINGS = ("activity", "bigram", "trigram", "skipbigram", "ef")
LOG_TITLE = {"sepsis": "Sepsis", "helpdesk": "Helpdesk", "bpic2013": "BPIC 2013", "rtfm": "RTFM"}
EMB_TITLE = {"activity": "activity", "bigram": "bigram", "ef": "EF", "skipbigram": "skip-bigram",
             "trigram": "trigram"}


def run(cmd, log_file, cwd=REPO):
    with open(log_file, "w") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
        f.flush()
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def write(path, table):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for c in sorted(table):
            w.writerow({k: table[c].get(k, "") for k in COLS})


def parse_command(log_path):
    """(the run's command line, its output directory) from a campaign log."""
    txt = open(log_path, errors="replace").read(30000)
    m = re.search(r"Running `([^`]*)`", txt)
    o = re.search(r"Output: (\S+)", txt)
    return (m.group(1) if m else None), (o.group(1) if o else None)


def kept_args(command):
    """The flags of KEEP_FLAGS with their values: the cell's data, case set, encoding and party count."""
    toks = command.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in KEEP_FLAGS:
            out += [t, toks[i + 1]]
            i += 2
        else:
            i += 1
    return out


def parse_scorer_log(path):
    """The alignment check, the projection width, the JL fidelity and the P/R rows of attack_on_mpc_output."""
    txt = open(path, errors="replace").read()
    row = {}
    m = re.search(r"anonymous set \|A\|=(\d+) against (\d+) opened rows -> (OK|MISMATCH)", txt)
    if m:
        row["anonymous_set"], row["opened"], row["align_ok"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"k=(\d+), L2 norm", txt)
    if m:
        row["k_proj"] = m.group(1)
    m = re.search(r"median rel-diff = (\d+\.\d+), within-25% = (\d+\.\d+)%", txt)
    if m:
        row["jl_med"], row["jl_w25"] = m.group(1), m.group(2)
    for line in txt.splitlines():
        r = ROW_RE.match(line.strip())
        if not r:
            continue
        name, mm, pp, rr, det = r.group("name").strip(), r.group("m"), r.group("p"), r.group("r"), r.group("detail")
        if name == "no-prior pinv":
            row["pinv_P"], row["pinv_R"] = pp, rr
        elif name.startswith("known-sample (norm match)"):
            row[f"norm_m{mm}_P"], row[f"norm_m{mm}_R"] = pp, rr
        elif name.startswith("known-sample (dist match)"):
            row[f"dist_m{mm}_P"], row[f"dist_m{mm}_R"] = pp, rr
        o = re.search(r"oracle-aligned on (\d+) rows", det or "")
        if o:
            row["oracle_rows"] = o.group(1)
    return row


def score_frequency(row, cdir, oracle_max_rows):
    """Frequency matching on the recorded opened rows and the cell's truth dump; nothing is rerun."""
    from attack_on_mpc_output import anonymous_set, frequency_match_report
    X_all = np.loadtxt(REPO / row["truth"], delimiter=",")
    Y = np.loadtxt(REPO / row["run_dir"] / "federated/features.csv", delimiter=",")
    Y = Y[np.linalg.norm(Y, axis=1) > 1e-9]                     # the opened rows (withheld rows are zero)
    kp = int(row["k_priv"])
    A = anonymous_set(X_all, kp)
    fm, lines = frequency_match_report(X_all, A, Y, kp, oracle_max_rows=oracle_max_rows)
    head = (f"cell {row['cell']}: truth {row['truth']} ({X_all.shape[0]} rows, d={X_all.shape[1]}), opened rows "
            f"{row['run_dir']}/federated/features.csv ({Y.shape[0]} nonzero, k={Y.shape[1]}), k_priv={kp}, "
            f"|A|={len(A)} -> {'OK' if len(A) == Y.shape[0] else 'MISMATCH'}")
    with open(cdir / "frequency_match.log", "w") as f:
        f.write("\n".join([head] + lines) + "\n")
    return {"freq_P": f"{fm['P']:.4f}", "freq_R": f"{fm['R']:.4f}", "freq_class_acc": f"{fm['class_acc']:.4f}",
            "freq_row_acc": f"{fm['row_acc']:.4f}", "freq_class_acc_exp": f"{fm['class_acc_exp']:.4f}",
            "freq_row_acc_exp": f"{fm['row_acc_exp']:.4f}", "freq_classes": str(fm["classes"]),
            "freq_tied_classes": str(fm["tied_classes"]), "freq_tied_rows": str(fm["tied_rows"]),
            "freq_sizes_ok": ("OK" if fm["sizes_ok"] and len(A) == Y.shape[0] else "MISMATCH"),
            "freq_oracle_rows": str(fm["oracle_rows"])}


def battery_cells(battery: Path):
    """(cell, run_dir) of every row of the battery's runs.tsv, in file order."""
    out = []
    with open(battery / "runs.tsv") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out


def k_priv_of(command, log_path):
    """The anonymity threshold of the run, from its command line or from the threshold its log reports."""
    m = re.search(r"--variant-k-anonymity (\d+)", command or "")
    if m:
        return int(m.group(1))
    txt = open(log_path, errors="replace").read()
    m = re.search(r"Budget-bounded k_priv = (\d+) \(selected in-MPC", txt)
    if m:
        return int(m.group(1))
    m = re.search(r"Budget-bounded k_priv = (\d+) ", txt)
    return int(m.group(1)) if m else None


def sim_geometry_at(battery: Path, cell, k_proj):
    """FastICA and dist+MDS of this battery's own no-shuffle simulation at the run's width."""
    p = battery / cell / "upper_bound__sim_secret_R.log"
    if not p.exists():
        return {}
    out = {}
    for line in open(p, errors="replace"):
        m = re.match(r"\s*(\d+)\s+(FastICA|dist\+MDS)\s+-\s+(\d+\.\d+)\s+(\d+\.\d+)", line)
        if m and int(m.group(1)) == k_proj:
            key = "sim_ica" if m.group(2) == "FastICA" else "sim_mds"
            out[key + "_P"], out[key + "_R"] = m.group(3), m.group(4)
            out["sim_k"] = m.group(1)
    return out


def quality_f(quality_csv: Path):
    """cell -> the F of its federated tier, from the campaign's model_quality.csv."""
    out = {}
    if not quality_csv.exists():
        return out
    for r in csv.DictReader(open(quality_csv)):
        if r.get("tier") == "federated" and r.get("f_score"):
            out[r["cell"]] = r["f_score"]
    return out


def rescore(args):
    table = {}
    tab_path = args.out / "recovery_table_rescored.csv"
    if tab_path.exists():
        for r in csv.DictReader(open(tab_path)):
            table[r["cell"]] = r
    cells = battery_cells(args.battery)
    names = [c for c in cells if not args.only or c[0] in set(args.only)]
    print(f"{len(names)} cells of {args.battery} -> {args.out}")
    for i, (cell, run_dir) in enumerate(names, 1):
        if not (args.battery / cell / "battery_summary.json").exists():
            print(f"[{i}/{len(names)}] {cell}: battery not finished, skipped")
            continue
        if cell in table and table[cell].get("pinv_P") and not args.redo_freq:
            print(f"[{i}/{len(names)}] {cell}: done, skipped")
            continue
        cdir = args.out / cell
        cdir.mkdir(parents=True, exist_ok=True)
        row = dict.fromkeys(COLS, "")
        row.update(cell=cell, source=args.battery.name, run_dir=run_dir)
        lp = args.campaign / f"{cell}.log"
        if not lp.exists():
            row["log"] = f"NO CAMPAIGN LOG {lp}"
            table[cell] = row; write(tab_path, table); print(f"[{i}] {cell}: no campaign log"); continue
        command, _ = parse_command(lp)
        if not command or not (REPO / run_dir / "federated/features.csv").exists():
            row["log"] = f"NO COMMAND OR OPENED MATRIX ({run_dir})"
            table[cell] = row; write(tab_path, table); print(f"[{i}] {cell}: {row['log']}"); continue
        kp = k_priv_of(command, lp) or 1
        enc = re.search(r"--encoding (\S+)", command)
        row.update(encoding=enc.group(1) if enc else "", k_priv=str(kp))
        # 1. the truth matrix: Phase 0-2 rerun in the clear, truncated as the run truncated
        dump_log = cdir / "dump_encoded_features.log"
        rc = run([BIN] + kept_args(command)
                 + ["--truncate-baseline", "--dump-encoded-features", "--run-name", f"rescore_uni_{cell}"],
                 dump_log)
        m = re.search(r"Output: (\S+)", open(dump_log, errors="replace").read())
        truth = REPO / m.group(1) / "features_encoded.csv" if m else None
        if rc != 0 or not truth or not truth.exists():
            row["log"] = f"DUMP FAILED rc={rc}"
            table[cell] = row; write(tab_path, table); print(f"[{i}] {cell}: dump failed"); continue
        row["truth"] = str(truth.relative_to(REPO))
        # 2. frequency guessing on the cell's own matrix
        X = np.loadtxt(truth, delimiter=",")
        p, r, top = frequency_guessing_baseline(X)
        row.update(null_P=f"{p:.4f}", null_R=f"{r:.4f}", null_cols=" ".join(map(str, top)), N=str(X.shape[0]))
        # 3. the attacks with the oracle row alignment
        slog = cdir / "real_envelope_oracle.log"
        rc = run([sys.executable, HERE / "attack_on_mpc_output.py", "--run", REPO / run_dir, "--truth", truth,
                  "--k-priv", str(kp), "--m-list", *map(str, M_LIST), "--quad-max-rows", str(args.quad_max_rows),
                  "--oracle-max-rows", str(args.oracle_max_rows)], slog)
        row.update(parse_scorer_log(slog))
        row["log"] = str(slog.relative_to(REPO)) + ("" if rc == 0 else f" (rc={rc})")
        # 4. the geometry bound of this battery's simulation
        if row.get("k_proj"):
            row.update(sim_geometry_at(args.battery, cell, int(row["k_proj"])))
        # 5. frequency matching on the opened rows
        row.update(score_frequency(row, cdir, args.oracle_max_rows))
        table[cell] = row
        write(tab_path, table)
        print(f"[{i}/{len(names)}] {cell}: |A|={row.get('anonymous_set')} opened={row.get('opened')} "
              f"{row.get('align_ok')} guess {row['null_P']}/{row['null_R']} pinv {row.get('pinv_P')}/"
              f"{row.get('pinv_R')} dist5 {row.get('dist_m5_P')}/{row.get('dist_m5_R')}", flush=True)
    return table


def load_table(out: Path):
    p = out / "recovery_table_rescored.csv"
    if not p.exists():
        return {}
    return {r["cell"]: r for r in csv.DictReader(open(p)) if r.get("pinv_P")}


def num(n):
    return f"${int(n):,}$".replace(",", "{,}")


def pr(r, a, d=2):
    if r.get(a + "_P") in (None, ""):
        return "---"
    return f"${float(r[a + '_P']):.{d}f}$/${float(r[a + '_R']):.{d}f}$"


def fmatch(r):
    if r.get("freq_row_acc") in (None, ""):
        return "---"
    return f"${float(r['freq_row_acc']):.2f}$ (${float(r['freq_row_acc_exp']):.2f}$)"


def emit(args, rows):
    """The figure data file and the two appendix table sources."""
    fed_f = quality_f(args.campaign / "model_quality.csv")
    out = args.emit
    out.mkdir(parents=True, exist_ok=True)
    dat = out / "attack_battery_uniform_2026-10-09.dat"
    L = ["# generated by scripts/attacks/rescore_battery_uniform.py from "
         f"{args.out}/recovery_table_rescored.csv",
         f"# battery {args.battery}, campaign {args.campaign}; every value an oracle-aligned upper bound",
         "# panel: log cell N opened k_priv null_P null_R pinv_P pinv_R d5_P d5_R d20_P d20_R "
         "freq_row_acc freq_row_acc_exp fedF   (the bigram Revealed-Projection run of each event log)"]
    panels = [(lg, f"ms_{lg}_rp_bigram_s42") for lg in LOGS]
    missing = []
    for lg, cell in panels:
        r = rows.get(cell)
        if not r:
            missing.append(cell); continue
        L.append(" ".join([lg, cell, r["N"], r["opened"], r["k_priv"], r["null_P"], r["null_R"],
                           r["pinv_P"], r["pinv_R"], r["dist_m5_P"], r["dist_m5_R"],
                           r.get("dist_m20_P", "-"), r.get("dist_m20_R", "-"),
                           r.get("freq_row_acc", "-"), r.get("freq_row_acc_exp", "-"),
                           fed_f.get(cell, "-")]))
    L.append("# grid: log embedding cell N opened k_priv k_proj null_P null_R pinv_P pinv_R d5_P d5_R "
             "d20_P d20_R freq_row_acc freq_row_acc_exp fedF   (every Revealed-Projection cell of the grid)")
    for lg in LOGS:
        for e in EMBEDDINGS:
            cell = f"ms_{lg}_rp_{e}_s42"
            r = rows.get(cell)
            if not r:
                continue
            L.append(" ".join(["grid", lg, e, cell, r["N"], r["opened"], r["k_priv"], r.get("k_proj", "-"),
                               r["null_P"], r["null_R"], r["pinv_P"], r["pinv_R"],
                               r["dist_m5_P"], r["dist_m5_R"], r.get("dist_m20_P", "-"),
                               r.get("dist_m20_R", "-"), r.get("freq_row_acc", "-"),
                               r.get("freq_row_acc_exp", "-"), fed_f.get(cell, "-")]))
    dat.write_text("\n".join(L) + "\n")
    print("wrote", dat, len(L), "lines" + (f"   MISSING: {missing}" if missing else ""))

    GEN = ["% GENERATED by scripts/attacks/rescore_battery_uniform.py from the rescored recovery table of "
           f"{args.battery.name}", "% -- do not edit by hand."]
    tab = out / "attack_table_uniform_2026-10-09.tex"
    T = GEN + ["\\footnotesize \\setlength{\\tabcolsep}{3pt}",
               "\\begin{tabular}{@{}l l r r r l l l l l@{}}", "\\toprule",
               "Log & Embedding & Opened & $k_\\mathrm{priv}$ & $k_\\mathrm{proj}$ & Guess & Alone & "
               "\\shortstack[l]{Five\\\\known} & \\shortstack[l]{Twenty\\\\known} & "
               "\\shortstack[l]{Freq.\\\\match} \\\\", "\\midrule"]
    first = True
    for lg in LOGS:
        brs = [(rows[f"ms_{lg}_rp_{e}_s42"], e) for e in EMBEDDINGS if f"ms_{lg}_rp_{e}_s42" in rows]
        if not brs:
            continue
        if not first:
            T.append("\\addlinespace[2pt]")
        first = False
        for i, (r, e) in enumerate(brs):
            lab = LOG_TITLE[lg] if i == 0 else (num(r["N"]) + " cases" if i == 1 else "")
            T.append(f"{lab:18s} & {EMB_TITLE[e]:11s} & {num(r['opened']):11s} & {num(r['k_priv'])} & "
                     f"${r.get('k_proj', '')}$ & {pr(r, 'null', 3)} & {pr(r, 'pinv')} & "
                     f"{pr(r, 'dist_m5')} & {pr(r, 'dist_m20')} & {fmatch(r)} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    tab.write_text("\n".join(T) + "\n")
    print("wrote", tab, sum(1 for l in T if l.endswith("\\\\")) - 1, "rows")

    tabsim = out / "attack_table_sim_uniform_2026-10-09.tex"
    S = GEN + ["\\footnotesize \\setlength{\\tabcolsep}{4pt}",
               "\\begin{tabular}{@{}l l r l l l@{}}", "\\toprule",
               "Log & Embedding & $k_\\mathrm{proj}$ & Guess & FastICA & Dist.+MDS \\\\", "\\midrule"]
    first = True
    for lg in LOGS:
        brs = [(rows[f"ms_{lg}_rp_{e}_s42"], e) for e in EMBEDDINGS
               if f"ms_{lg}_rp_{e}_s42" in rows and rows[f"ms_{lg}_rp_{e}_s42"].get("sim_ica_P")]
        if not brs:
            continue
        if not first:
            S.append("\\addlinespace[2pt]")
        first = False
        bgcell = f"ms_{lg}_rp_bigram_s42"
        bj = args.battery / bgcell / "battery_summary.json"
        guess = "---"
        if bj.exists():
            b = json.load(open(bj))["frequency_guessing_baseline"]
            guess = f"${b['P']:.3f}$/${b['R']:.3f}$"
        for i, (r, e) in enumerate(brs):
            lab = LOG_TITLE[lg] if i == 0 else (num(r["N"]) + " cases" if i == 1 else "")
            S.append(f"{lab:18s} & {EMB_TITLE[e]:11s} & ${r['k_proj']}$ & {guess} & "
                     f"{pr(r, 'sim_ica')} & {pr(r, 'sim_mds')} \\\\")
    S += ["\\bottomrule", "\\end{tabular}"]
    tabsim.write_text("\n".join(S) + "\n")
    print("wrote", tabsim, sum(1 for l in S if l.endswith("\\\\")) - 1, "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path, default=REPO / "output/attack_battery_uniform_2026-10-09")
    ap.add_argument("--campaign", type=Path, default=REPO / "output/campaign_uniform_2026-10-04")
    ap.add_argument("--out", type=Path, default=REPO / "output/attack_battery_rescored_uniform_2026-10-09")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quad-max-rows", type=int, default=3000)
    ap.add_argument("--oracle-max-rows", type=int, default=3000)
    ap.add_argument("--redo-freq", action="store_true")
    ap.add_argument("--emit", type=Path, default=None,
                    help="also write the figure data file and the two table sources into this directory")
    ap.add_argument("--emit-only", action="store_true", help="only --emit, rescore nothing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.emit_only:
        rescore(args)
    if args.emit:
        emit(args, load_table(args.out))


if __name__ == "__main__":
    main()
