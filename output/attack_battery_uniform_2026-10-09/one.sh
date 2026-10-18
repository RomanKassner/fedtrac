#!/bin/bash
# one battery run: $1 cell, $2 run dir, $3 cases, $4 flag (none or --skip-sim); an atomic lock per run so two drivers never run one cell twice
cd /home/roman/projects/federated_clustering
out=output/attack_battery_uniform_2026-10-09
if [ -s "$out/$1/battery_summary.json" ]; then echo "[skip] $1 (done)"; exit 0; fi
if ! mkdir "$out/$1.lock" 2>/dev/null; then echo "[busy] $1 (another lane)"; exit 0; fi
[ "$4" = "none" ] && f="" || f="$4"
t0=$(date +%s)
echo "[run] $1 <- $2 ($(date '+%F %T')) flags: --quad-max-rows 3000 $f"
.venv/bin/python scripts/attacks/run_attack_battery.py --run "$2" --out "$out/$1" --quad-max-rows 3000 $f > "$out/$1.console" 2>&1
rc=$?
echo "[done] $1 rc=$rc $(( $(date +%s) - t0 )) s"
