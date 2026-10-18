# FedTrac

Trace clustering across organizations inside a secure multi-party computation. Each party holds the events of its own
activities. The parties jointly cluster the shared cases and publish one directly-follows graph per cluster, in two
instantiations:

- **Secret-Labels**: feature extraction, projection, k-means (optionally refined by a Markov-chain step) and aggregation
  run inside the computation; only the per-cluster graphs are opened.
- **Revealed-Projection**: the computation opens the projected, shuffled, unit-length rows of the cases whose feature
  profile is shared by at least k_priv cases; the parties cluster those rows in the clear; the computation labels the
  withheld rows and aggregates.

A plaintext mode runs the same pipeline on the merged log as the reference. This repository is the code and the recorded
evidence of the master's thesis "FedTrac" (RWTH Aachen, 2026).

## Layout

```
src/        Rust core: public parameters, rounds, published graphs, run folders (binary clustering_federated)
mp-spdz/    the secure programs for MP-SPDZ (secure_extract, secure_extract_rsecret, secure_kmeans,
            secure_sequence_cluster, secure_relabel, secure_aggregate), mpc_utils/, the eight test programs
bridge/     Python: clustering in the clear, the in-run evaluation, the graph and relevance measures
scripts/    the campaign driver, the model-quality instrument, the attack battery and its rescoring,
            the figure data, the log splitter
config/     the splits by role (party -> activities) and the pinned case populations (config/case_ids/)
data/       splits/<split>.tar.xz: the per-party sublogs every recorded run read (the raw logs are not shipped)
output/     the recorded evidence (below)
```

## Setup

- Rust 1.93.1 (`rust-toolchain.toml`; rustup installs it). `cargo build --release`.
- Python 3.12 in a venv at the repository root, which the core finds itself:
  `python3.12 -m venv .venv && .venv/bin/pip install -r bridge/requirements.freeze.txt pytest`.
  The freeze file is the recorded environment (pm4py 2.7.22.2, numpy 1.26.4, scipy 1.17.1, pandas 3.0.2,
  scikit-learn 1.8.0); `bridge/requirements.txt` gives the ranges.
- MP-SPDZ at commit `bf7f8f4` (tag v0.4.2): `make -j8 tldr && make -j8 semi2k-party.x`; `export MP_SPDZ_DIR=~/MP-SPDZ`
  (the core's default as well). The core copies the programs into `Programs/Source/` and compiles them on each run.
- graphviz (`dot`) for the graph drawings.

## Event logs

Four public logs from 4TU.ResearchData, placed under `data/raw/` with these names (sha256 of the unpacked file):

| Log | DOI | Deposit file | Local file | sha256 |
|---|---|---|---|---|
| Sepsis Cases | 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460 | `Sepsis Cases - Event Log.xes.gz` | `Sepsis Cases.xes` | `b501d2063678b9471e3cb1be04d961f924ae5be6ecc8154701bb57a24c7ae0ab` |
| Road Traffic Fine Management | 10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5 | `Road_Traffic_Fine_Management_Process.xes.gz` | `road_traffic_fine_management.xes` | `5fd307c2742db76dfa788f73dca6b47a80e78672da5cd7df146745d338f82261` |
| BPI Challenge 2013 incidents | 10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee | `BPI_Challenge_2013_incidents.xes.gz` | `bpic2013_incidents.xes` | `240a370cfca2a8b5def0ef722672f469c01812a554f000e602bc9a5bcbc2314e` |
| Helpdesk | 10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb | `finale.csv` | `helpdesk.xes` | `69365836bb9ae6c1a824484b502e6c5624ec14f12b4b427b10869be671fae2c7` |

Helpdesk ships as a CSV; `helpdesk.xes` is its pm4py conversion (columns renamed to `case:concept:name`, `concept:name`,
`org:resource`, `time:timestamp`; timestamps parsed as `%Y/%m/%d %H:%M:%S.%f`). In BPIC 2013 `concept:name` is the
incident status, so the alphabet is its four values.

The per-party sublogs of every recorded run are shipped: `cd data/splits && for f in *.tar.xz; do tar -xJf $f; done`.
A split is cut from a raw log by `scripts/split_log.py --xes <raw log> --partition-file config/<split>.json --output-dir data/splits/<dir>`.
The case populations the runs cluster are pinned in `config/case_ids/` (`scripts/build_full_case_ids.py` rebuilds a list).

## Running

```bash
# the clustering in the clear
cargo run --release -- --data-dir data/splits/sepsis_2party --case-ids config/case_ids/sepsis_2party_full.txt --encoding bigram --k 6 --run-name sepsis_pt --evaluate-full
# Secret-Labels (add --secure-clustering sequence for the refinement)
cargo run --release -- --mpc --secure-kmeans --k 6 --data-dir data/splits/sepsis_2party --case-ids config/case_ids/sepsis_2party_full.txt --encoding bigram --run-name sepsis_sl --evaluate-full
# Revealed-Projection (k by prediction strength on the opened rows)
cargo run --release -- --mpc --reveal-secret-projection --projection-seed 42 --data-dir data/splits/sepsis_2party --case-ids config/case_ids/sepsis_2party_full.txt --encoding bigram --run-name sepsis_rp --evaluate-full
```

Embeddings: `activity`, `bigram`, `trigram`, `skip-bigram`, `ef`; `--n-parties 3` reads a three-party split; `--help`
lists every flag. A run writes `output/runs/<name>_<timestamp>/` with one folder per tier: `plaintext` for the clustering in the
clear; a secure run writes `federated` and `timing.json`, and under `--evaluate-full` also `plaintext` and `projected`.
A tier folder holds the published graphs `cluster_<j>_dfg.json`, the per-case labels `cluster_assignments.json`, the
traces of each cluster and, for the opened rows of a Revealed-Projection run, `features.csv`. `--evaluate-full` opens the per-case labels for the evaluation; a deployment does not pass it. The
recorded runs used `semi2k-party.x` over Z_2^64, compile flags `-Z <parties> -D --budget 4000`, party flags
`-o softspoken=6` and 8 threads (`CLUSTERING_MPC_COMPILE_FLAGS`, `CLUSTERING_MPC_PARTY_FLAGS`, `CLUSTERING_MPC_THREADS`).

## Tests

Unpack the splits first (above).

```bash
cargo test --release
.venv/bin/python -m pytest bridge scripts
for t in secure_extract secure_extract_rsecret secure_kmeans secure_sequence extract_once secure_aggregate_prune \
         secure_pipeline secure_pipeline_hybrid; do .venv/bin/python mp-spdz/test_$t.py --spdz-dir $MP_SPDZ_DIR || break; done
```

The eight MP-SPDZ test programs run the secure programs as party processes against their counterparts in the clear
(`scripts/plaintext_kmeans_reference.py`, `scripts/plaintext_sequence_reference.py`, `bridge/`): six check one program
each, `secure_pipeline` and `secure_pipeline_hybrid` run the whole chain and assert the published graphs and cluster
sizes bit for bit. They read the splits `sepsis_2party`, `sepsis_4party`, `3party` and `sepsis_2party_erlab_wards`.

## The recorded evidence

`output/campaign_uniform_2026-10-04/` is the campaign the thesis reports (2026-10-04 to 2026-10-11, one release binary):
`manifest.tsv` (command line, commit, protocol and flags per cell), `summary.tsv` (times and bytes), `clock.tsv` (the
clock check of every cell), `void.tsv` (cells whose times do not count), `<cell>.log`, `code_state.txt`,
`campaign_meta.txt`, `agg_cells.csv` and `agg_long.csv` (`scripts/aggregate_campaign.py`), and `model_quality.csv`
with `model_quality/<cell>__<tier>.json` (`scripts/evaluate_campaign.py`, instrument `dfm-2026-10-15`).

`output/runs/<run>.tar.xz` holds every run folder a reported number rests on, as the run wrote it: the 242 scored runs
and the 165 feature dumps the attacks were rescored against. `cd output/runs && for f in *.tar.xz; do tar -xJf $f; done`
restores them.

`output/attack_battery_uniform_2026-10-09/` is the attack battery (`scripts/attacks/run_attack_battery.py`, one folder per
attacked run) and `output/attack_battery_rescored_uniform_2026-10-09/` its rescoring with the oracle row alignment
(`scripts/attacks/rescore_battery_uniform.py`; `recovery_table_rescored.csv`). `output/causes_study_2026-10-10/` and
`output/method_study_uniform_2026-10-09/` hold the results of `scripts/causes_study_uniform.py` (the replay of the secure
clustering in the clear) and `scripts/method_study_uniform.py` (other clustering methods on the opened rows).

`output/models/<run>/` holds every published cluster graph of the 242 scored runs and the whole-log graph the
instrument scores, drawn as SVG by `scripts/render_graphs.py --runs output/runs --out output/models`.

## Reproducing the numbers

Every script runs as `.venv/bin/python <script>` from the repository root, on the unpacked runs. The commits that
`manifest.tsv` and `code_state.txt` name belong to the development history; this repository is that code at its final
state with the unused parts removed, and the instrument below reproduces every recorded value except pm4py's
token-based generalization, which moves in the third decimal between two scorings.

| what | command |
|---|---|
| model quality of every tier folder: the pruned published graph as the model, fitness and precision by alignments, relevance, the whole-log and random references | `scripts/evaluate_campaign.py --campaign output/campaign_uniform_2026-10-04` (hours; one tier folder: `--folder output/runs/<run>/federated`, minutes) |
| times, bytes and the refinement scores per cell | `scripts/aggregate_campaign.py --campaign output/campaign_uniform_2026-10-04` |
| the data of the evaluation figures and of the grid and party tables | `scripts/uniform_figures.py --campaign output/campaign_uniform_2026-10-04 --out <dir>` |
| the attack tables from the recorded rescoring | `scripts/attacks/rescore_battery_uniform.py --emit-only --emit <dir>` |
| the rescoring itself (rebuilds the feature dumps with the binary) | `scripts/attacks/rescore_battery_uniform.py` |
| the replay of the secure clustering in the clear | `scripts/causes_study_uniform.py --stages all` |
| the other clustering methods on the opened rows | `scripts/method_study_uniform.py --stages all` |
| the withholding curve | `scripts/variant_suppression_sweep.py --split-dir data/splits/<dir> --case-ids config/case_ids/<list> --unit bigram` |
| the cross-party share of a population | `scripts/cross_party_share.py --traces output/runs/<run>/federated/global_traces.json --partition config/<split>.json` (the split the run read, `manifest.tsv`) |
| the graph drawings of one run at figure size | `scripts/models_figure_uniform.py --run output/runs/<run> --clusters 3 5 --out-dir <dir>` |

A whole campaign is
`scripts/launch_uniform_campaign.sh <family>...` (one systemd user service over `scripts/run_uniform_campaign.sh`, whose
cell table is the campaign's design); the full-log RTFM cells under eventually-follows and skip-bigram compile with a
working set of about 69 GB and need memory and swap of at least 75 GB.

## Disclosure

Secret-Labels opens the per-cluster graphs after the edge filter and the pruning inside the computation, and nothing
else; the cluster sizes follow from the start cells. Revealed-Projection additionally opens the projected, shuffled,
unit-length rows of the cases above the withholding threshold, the threshold and the number of opened rows. The
projection matrix, the shuffle and the labels of the withheld rows stay secret. In the evaluation every party runs as a
process on one host.
