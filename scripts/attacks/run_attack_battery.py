#!/usr/bin/env python3
"""One reproducible run of the full `--reveal-secret-projection` attack battery."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Pinned before `import numpy` and exported to every child: the ill-conditioned least squares shifts with the thread count.
BLAS_THREADS = os.environ.get("BATTERY_BLAS_THREADS", "1")
BLAS_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS")
for _v in BLAS_ENV_VARS:
    os.environ[_v] = BLAS_THREADS

import numpy as np  # noqa: E402

ATTACK_DIR = Path(__file__).resolve().parent
PY = sys.executable
ROW_RE = re.compile(                      # a line of attack_on_mpc_output's results table: name m P R [detail]
    r"^(?P<name>.+?)\s+(?P<m>-|\d+)\s+(?P<p>\d+\.\d{2})\s+"
    r"(?P<r>\d+\.\d{2})\s*(?P<detail>.*)$"
)


def frequency_guessing_baseline(X, top_k=5):
    """The guess: the top_k globally most frequent feature cells predicted for every case, reading no opened row."""
    mt = X > 0
    top = np.argsort(-mt.sum(0))[:top_k]
    mh = np.zeros_like(mt)
    mh[:, top] = True
    tp = int((mt & mh).sum())
    fp = int((~mt & mh).sum())
    fn = int((mt & ~mh).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, [int(t) for t in top]


def revealed_nonzero_rows(run: Path):
    """(#non-zero revealed rows, #total) for a run's federated Y, or None."""
    fed = run / "federated" / "features.csv"
    pt = run / "plaintext" / "features.csv"
    if not fed.exists() or not pt.exists():
        return None
    Y = np.loadtxt(fed, delimiter=",")
    if Y.ndim != 2:
        return None
    nz = int((np.linalg.norm(Y, axis=1) > 1e-9).sum())
    return nz, Y.shape[0]


def autodetect_run():
    """The most recent run with opened rows, preferring one in which the withholding fired."""
    cands = sorted(set(glob.glob("output/runs/*rsec*") + glob.glob("output/runs/*reval*")),
                   key=os.path.getmtime, reverse=True)
    fallback = None
    for c in cands:
        run = Path(c)
        info = revealed_nonzero_rows(run)
        if not info or info[0] == 0:
            continue
        nz, total = info
        if nz < total:           # the withholding fired
            return run
        fallback = fallback or run
    return fallback


def run_attack(name, argv, report, out_dir):
    """Run one attack into <name>.log and the combined report."""
    banner = f"\n{'#' * 78}\n# {name}\n# $ {' '.join(str(a) for a in argv)}\n{'#' * 78}\n"
    print(banner.rstrip())
    proc = subprocess.run([PY, *map(str, argv)], capture_output=True, text=True)
    body = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    (out_dir / f"{name}.log").write_text(body)
    report.write(banner)
    report.write(body)
    if proc.returncode != 0:
        report.write(f"\n[!] {name} exited {proc.returncode}\n")
        print(f"[!] {name} exited {proc.returncode}")
    return proc.stdout


def parse_real_envelope(stdout):
    """The P/R rows and the JL fidelity line of attack_on_mpc_output's output."""
    rows, jl = [], None
    in_table = False
    for line in stdout.splitlines():
        if line.startswith("JL distance fidelity"):
            jl = line.strip()
        if line.lstrip().startswith("attack") and " P " in f" {line} ":
            in_table = True
            continue
        if in_table:
            if set(line.strip()) <= {"-"} or not line.strip():
                continue
            m = ROW_RE.match(line)
            if m:
                rows.append({
                    "attack": m.group("name").strip(),
                    "m": m.group("m"),
                    "P": float(m.group("p")),
                    "R": float(m.group("r")),
                    "detail": m.group("detail").strip() or None,
                })
    return rows, jl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: the latest)")
    ap.add_argument("--out", type=Path, default=None,
                    help="report dir (default: output/attack_battery_<run>)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m-list", type=int, nargs="+", default=[5, 20, 50],
                    help="known-row budgets for the real-envelope attack")
    ap.add_argument("--skip-sim", action="store_true",
                    help="run only the real-envelope attack (skip the upper bound)")
    ap.add_argument("--quad-max-rows", type=int, default=0,
                    help="forwarded to attack_on_mpc_output as its row cap (0 = no cap)")
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parents[2])

    run = args.run or autodetect_run()
    if run is None:
        print("No rsec run with non-trivial revealed Y found. Pass --run, or make "
              "one with: cargo run --release -- --mpc --reveal-secret-projection "
              "--evaluate-full --encoding activity --data-dir data/splits/sepsis_2party",
              file=sys.stderr)
        return 1
    info = revealed_nonzero_rows(run)
    if not info or info[0] == 0:
        print(f"{run} has no non-zero revealed rows (all suppressed?).", file=sys.stderr)
        return 1
    nz, total = info
    X = np.loadtxt(run / "plaintext" / "features.csv", delimiter=",")
    Y = np.loadtxt(run / "federated" / "features.csv", delimiter=",")
    d, k_proj = X.shape[1], Y.shape[1]
    feats = run / "plaintext" / "features.csv"

    out_dir = args.out or Path(f"output/attack_battery_{run.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fg_p, fg_r, fg_cols = frequency_guessing_baseline(X)
    header = (f"Attack battery on {run.name}\n"
              f"  revealed Y: {nz}/{total} rows non-zero ({total - nz} suppressed-tail "
              f"masked), k_proj={k_proj}\n"
              f"  X (plaintext oracle): n={X.shape[0]} d={d}\n"
              f"  seed={args.seed}  m-list={args.m_list}  "
              f"blas_threads={BLAS_THREADS}\n"
              f"  frequency-guessing baseline (top-5 cells for every case, no revealed "
              f"row read): P={fg_p:.2f} R={fg_r:.2f} cols={fg_cols}\n")
    print(header)
    report = (out_dir / "report.txt").open("w")
    report.write(header)

    # REAL ENVELOPE: the hybrid-suppression-aware attack on the actual run.
    real_out = run_attack(
        "real_envelope__attack_on_mpc_output",
        [ATTACK_DIR / "attack_on_mpc_output.py", "--run", run,
         "--m-list", *args.m_list]
        + (["--quad-max-rows", str(args.quad_max_rows)]
           if args.quad_max_rows else []),
        report, out_dir)
    real_rows, jl = parse_real_envelope(real_out)

    # NO-SUPPRESSION UPPER BOUND: self-simulating attacks on the same X.
    if not args.skip_sim:
        # d-aware 3-party split (remainder bucket forms party 3); k swept around
        # the run's actual k_proj so the bound is comparable to the real envelope.
        part = f"{d // 3},{d // 3}"
        sim_k = sorted({k_proj, min(2 * k_proj, d), max(10, k_proj // 2)})
        sim_k_str = [str(x) for x in sim_k]
        run_attack(
            "upper_bound__sim_secret_R",
            [ATTACK_DIR / "attack_sim_secret_R.py", "--features", feats,
             "--k", *sim_k_str, "--seed", args.seed],
            report, out_dir)
        run_attack(
            "upper_bound__vertical_split",
            [ATTACK_DIR / "attack_vertical_split.py", "--features", feats,
             "--partition", part, "--k", *sim_k_str, "--seed", args.seed],
            report, out_dir)
        run_attack(
            "upper_bound__anchor_mds_honest",
            [ATTACK_DIR / "attack_anchor_mds_honest.py", "--features", feats,
             "--partition", part, "--k", *sim_k_str, "--seed", args.seed],
            report, out_dir)

    report.close()

    summary = {
        "run": run.name,
        "revealed_nonzero_rows": nz,
        "total_rows": total,
        "suppressed_tail_rows": total - nz,
        "k_proj": k_proj,
        "d": d,
        "seed": args.seed,
        "blas_threads": BLAS_THREADS,
        "jl_distance_fidelity": jl,
        "frequency_guessing_baseline": {
            "top_k": 5, "P": round(fg_p, 4), "R": round(fg_r, 4),
            "cols": fg_cols,
            "note": ("null model: the top-5 globally most frequent feature cells "
                     "predicted for every case, scored like the attacks (micro P/R "
                     "vs X > 0); reads no revealed row"),
        },
        "real_envelope": real_rows,
        "note": ("real_envelope = hybrid-suppression-aware attack on the actual "
                 "revealed Y; P/R are the citable post-hardening leakage. The "
                 "FastICA/distance+MDS rows are oracle-aligned upper bounds "
                 "(best_column_align uses X_true) measuring the accepted "
                 "distance-leak floor, not honest attacks. The m/m 'paired' "
                 "detail is semantically broken post-suppression (indices shift "
                 "after the masked-row drop) -- read P/R, not the paired count."),
    }
    (out_dir / "battery_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}\nREAL ENVELOPE (citable):")
    for r in real_rows:
        print(f"  {r['attack']:<30} m={r['m']:>3}  P={r['P']:.2f}  R={r['R']:.2f}")
    if jl:
        print(f"  {jl}")
    print(f"\nwrote: {out_dir}/report.txt, battery_summary.json, per-attack logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
