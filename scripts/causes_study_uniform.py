#!/usr/bin/env python3
"""The causes of the gap between the secure Secret-Labels clustering and the clustering in the clear: the secure
specification replayed in the clear per run, one difference removed at a time, every partition scored by the one
instrument the thesis reports (scripts/evaluate_campaign.py, imported here, not reimplemented).

    scripts/causes_study_uniform.py --stages all --workers 8 [--only REGEX] [--variants v0_spec,v2_ten_iter]
    scripts/causes_study_uniform.py --report            # results.csv and the facts, from what is on disk
"""
import argparse
import csv
import glob
import gzip
import json
import pickle
import random
import re
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "mp-spdz"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAMPAIGN = REPO / "output" / "campaign_uniform_2026-10-04"
OUT = REPO / "output" / "causes_study_2026-10-10"
CASES = OUT / "cases"
# The merged party files of method_study_uniform.py, same naming and same builder; read when they are there.
BORROWED_CASES = REPO / "output" / "method_study_uniform_2026-10-09" / "cases"

# The 38 Secret-Labels grid cells (four logs x five encodings x {km, seq} minus the two refused sepsis-ef cells).
GRID_RE = re.compile(r"^(sepsis|helpdesk|bpic2013|rtfm)-(activity|bigram|trigram|skipbigram|ef)_sl_(km|seq)$")
# The party-count family, k-means only: `rt<Log>_sl_<emb>_<n>p`.
PC_RE = re.compile(r"^rt(Sepsis|Helpdesk|Bpic2013|Rtfm)_sl_(activity|bigram|trigram|skipbigram|ef)_([23])p$")
ENCODING = {"activity": "activity", "bigram": "bigram", "trigram": "trigram",
            "skipbigram": "skip-bigram", "ef": "ef"}

MIN_CLUSTER_SIZE = 10            # src/constants.rs::MIN_CLUSTER_SIZE
CLEAR_MAX_ITER = 10              # src/constants.rs::DEFAULT_MAX_ITERATIONS
CLEAR_N_INIT = 10                # src/constants.rs::DEFAULT_N_INIT
SEQ_MAX_ITER = 5                 # src/constants.rs::SECURE_SEQUENCE_MAX_ITER, the same in both tiers
PROJECTION_SEED = 42             # src/constants.rs::DEFAULT_PROJECTION_SEED
FRAC_BITS = 16                   # mp-spdz/secure_kmeans.mpc, scripts/plaintext_kmeans_reference.py
SEED_DRAW = 1                    # random.Random(1) for the random seedings
PROD_LIMIT = 1 << 62             # secure_kmeans.mpc: |s_j[d] * r_j| must stay strictly below 2^62

# seeds: "stride" | "ten" | ("one", j); iters: of the k-means (of the initialisation on a sequence run);
# bound: "run" = the rows the run read, truncated at its length bound; "full" = the rows of the untruncated traces.
SPEC = dict(seeds="stride", iters=5, rounding=True, merge=True, bound="run", normalise=False)
VARIANT = {
    "v0_spec":             dict(SPEC),
    "v1_ten_seeds":        dict(SPEC, seeds="ten"),
    "v2_ten_iter":         dict(SPEC, iters=CLEAR_MAX_ITER),
    "v3_no_merge":         dict(SPEC, merge=False),
    "v4_no_rounding":      dict(SPEC, rounding=False),
    "v5_clear_algo":       dict(SPEC, seeds="ten", iters=CLEAR_MAX_ITER, rounding=False, merge=False),
    "v6_no_bound":         dict(SPEC, bound="full"),
    "v7_clear_normalised": dict(SPEC, seeds="ten", iters=CLEAR_MAX_ITER, rounding=False, merge=False,
                                normalise=True),
}
for _j in range(CLEAR_N_INIT):
    VARIANT[f"v1_seed{_j}"] = dict(SPEC, seeds=("one", _j))
DEFAULT_VARIANTS = ("v0_spec", "v1_ten_seeds", "v2_ten_iter", "v3_no_merge", "v4_no_rounding",
                    "v5_clear_algo", "v6_no_bound", "v7_clear_normalised")

COLUMNS = ["cell", "log", "embedding", "method", "variant", "cases", "k", "clusters", "sizes",
           "fitness", "precision", "f_score", "simplicity", "generalization", "edges", "activities", "rel",
           "models_scored", "run_f_score", "run_fitness", "run_precision", "run_plaintext_f_score",
           "d_f_vs_v0", "d_f_vs_plaintext", "v0_minus_run_f", "ari_to_run", "seed_used", "note"]


# ---------------------------------------------------------------- the run's projection matrix

def _pcg_seed(seed):
    """rand_core 0.6 SeedableRng::seed_from_u64: eight PCG-XSH-RR 64/32 outputs as the 32-byte ChaCha key."""
    mul, inc, m32, m64 = 6364136223846793005, 11634580027462260723, 0xffffffff, (1 << 64) - 1
    state, out = seed & m64, b""
    for _ in range(8):
        state = (state * mul + inc) & m64
        x = ((((state >> 18) ^ state) >> 27)) & m32
        r = state >> 59
        out += struct.pack("<I", ((x >> r) | (x << ((32 - r) & 31))) & m32)
    return out


def _chacha8_words(key, n):
    """The first n output words of ChaCha8 in counter mode from a 32-byte key (rand_chacha 0.3's stream)."""
    m32 = 0xffffffff

    def rotl(v, c):
        return ((v << c) | (v >> (32 - c))) & m32

    def quarter(s, a, b, c, d):
        s[a] = (s[a] + s[b]) & m32; s[d] = rotl(s[d] ^ s[a], 16)
        s[c] = (s[c] + s[d]) & m32; s[b] = rotl(s[b] ^ s[c], 12)
        s[a] = (s[a] + s[b]) & m32; s[d] = rotl(s[d] ^ s[a], 8)
        s[c] = (s[c] + s[d]) & m32; s[b] = rotl(s[b] ^ s[c], 7)

    k = list(struct.unpack("<8I", key))
    out, counter = [], 0
    while len(out) < n:
        start = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] + k + [counter & m32, counter >> 32, 0, 0]
        w = start[:]
        for _ in range(4):                      # 8 rounds
            quarter(w, 0, 4, 8, 12); quarter(w, 1, 5, 9, 13); quarter(w, 2, 6, 10, 14); quarter(w, 3, 7, 11, 15)
            quarter(w, 0, 5, 10, 15); quarter(w, 1, 6, 11, 12); quarter(w, 2, 7, 8, 13); quarter(w, 3, 4, 9, 14)
        out.extend((w[i] + start[i]) & m32 for i in range(16))
        counter += 1
    return out[:n]


def achlioptas(k_proj, d, seed=PROJECTION_SEED):
    """Twin of src/projection.rs::AchlioptasMatrix::generate: ChaCha8 from seed_from_u64, rand 0.8's f64 draw
    (the top 53 bits of a u64 scaled by 2^-53), -1 below 1/6, +1 above 5/6, 0 between."""
    import numpy as np
    n = k_proj * d
    w = np.asarray(_chacha8_words(_pcg_seed(seed), 2 * n), dtype=np.uint64)
    u = ((w[1::2] << np.uint64(32)) | w[0::2]) >> np.uint64(11)
    coin = u.astype(np.float64) * (1.0 / (1 << 53))
    out = np.zeros(n, dtype=np.int8)
    out[coin < 1.0 / 6.0] = -1
    out[coin > 5.0 / 6.0] = 1
    return out.reshape(k_proj, d)


# ---------------------------------------------------------------- the runs and their resolved parameters

def recorded_rows():
    """{(cell, tier): the model_quality.csv row}; a torn read is retried, the file being rewritten whole."""
    for attempt in range(6):
        out = {}
        try:
            with open(CAMPAIGN / "model_quality.csv", newline="") as fh:
                for r in csv.DictReader(fh):
                    if GRID_RE.match(r["cell"]) or PC_RE.match(r["cell"]):
                        out[(r["cell"], r["tier"])] = r
            if out:
                return out
        except (OSError, csv.Error):
            pass
        time.sleep(2 + attempt)
    raise SystemExit("model_quality.csv holds no row of a Secret-Labels cell")


def run_dirs():
    """{cell: run directory} of the campaign, from the manifest the instrument reads."""
    from evaluate_campaign import campaign_runs
    return campaign_runs(CAMPAIGN)


def replayable(cells):
    """The cells the campaign has a finished run for; a designed refusal has a log and no run to replay."""
    try:
        have = run_dirs()
    except (OSError, KeyError):
        return list(cells), []
    return [c for c in cells if c in have], [c for c in cells if c not in have]


def grid_cells():
    """The Secret-Labels grid cells with a finished run, from the campaign directory alone."""
    return replayable(sorted(p.stem for p in CAMPAIGN.glob("*.log") if GRID_RE.match(p.stem)))[0]


def refused_cells():
    """The grid cells whose run did not finish (the two designed sepsis-ef refusals of the campaign)."""
    return replayable(sorted(p.stem for p in CAMPAIGN.glob("*.log") if GRID_RE.match(p.stem)))[1]


def all_cells():
    return replayable(sorted(p.stem for p in CAMPAIGN.glob("*.log")
                             if GRID_RE.match(p.stem) or PC_RE.match(p.stem)))[0]


def parse_log(cell):
    """The resolved parameters of the run, from the last invocation in its driver log."""
    text = open(CAMPAIGN / f"{cell}.log", errors="replace").read()
    seg = text[text.rfind("Running `"):]

    def last(pattern, cast=str):
        m = re.findall(pattern, seg)
        return cast(m[-1]) if m else None

    cmd = last(r"Running `([^`]*)`")
    grid = GRID_RE.match(cell)
    if grid:
        log_name, emb, method = grid[1], grid[2], grid[3]
    else:
        pc = PC_RE.match(cell)
        log_name, emb, method = pc[1].lower(), pc[2], "km"
    km = last(r"Compiling secure_kmeans with args (\[[^\]]*\])")
    seq = last(r"Compiling secure_sequence_cluster with args (\[[^\]]*\])")
    km = [int(v) for v in json.loads(km)] if km else None
    seq = [int(v) for v in json.loads(seq)] if seq else None
    notes = []
    if km is None:
        notes.append("secure_kmeans compile args not in the log (bytecode cache hit); the tape's defaults assumed")
    if method == "seq" and seq is None:
        notes.append("secure_sequence_cluster compile args not in the log; the tape's defaults assumed")
    if km and km[3] != 5:
        notes.append(f"the run's secure_kmeans MAX_ITER is {km[3]}, not 5")
    params = dict(
        cell=cell, cmd=cmd, log_name=log_name, embedding=emb, encoding=ENCODING[emb], method=method,
        split=re.search(r"--data-dir data/splits/(\S+)", cmd)[1],
        caselist=re.search(r"--case-ids config/case_ids/(\S+)", cmd)[1],
        length_bound=last(r"Dynamic L = (\d+)", int) or last(r"--max-trace-len (\d+)", int),
        k_proj=last(r"k_proj=(\d+)", int) or last(r"k_proj = (\d+)", int),
        dict_total=last(r"\((\d+) total with markers\)", int),
        cases=last(r"(\d+) common cases", int),
        k=(km[2] if km else None) or last(r"--k (\d+)", int) or last(r"\[Auto-k\] k=(\d+)", int),
        skip_projection="Skipping projection" in seg,
        km_args=km, seq_args=seq, notes=notes,
        # The iteration count and the merge threshold of each secure step, as its compile args name them.
        km_max_iter=km[3] if km else 5,
        km_min_cluster=km[9] if km and len(km) > 9 else (0 if method == "seq" else MIN_CLUSTER_SIZE),
        seq_max_iter=seq[3] if seq else SEQ_MAX_ITER,
        seq_min_cluster=seq[10] if seq and len(seq) > 10 else MIN_CLUSTER_SIZE,
        sizes=last(r"Cluster sizes[^\[]*(\[[^\]]*\])"),
    )
    return params


def cases_pickle(params):
    """The merged party file of the run's (split, pinned case list), built if neither directory holds it."""
    name = f"cs_{params['split']}__{params['caselist']}.pkl"
    for base in (BORROWED_CASES, CASES):
        if (base / name).exists():
            return base / name
    CASES.mkdir(parents=True, exist_ok=True)
    build_cases(params["split"], params["caselist"], CASES / name)
    return CASES / name


def build_cases(split, caselist, path):
    """Per party {case: [(ms, activity)]} over the pinned cases every party holds, plus the sorted activity
    names: the twin of Phase 0 (src/main.rs)."""
    import xml.etree.ElementTree as ET
    from datetime import datetime
    pin = {line.strip() for line in open(REPO / "config" / "case_ids" / caselist) if line.strip()}

    def load(xes):
        cases = {}
        for _, el in ET.iterparse(xes, events=("end",)):
            if el.tag.split("}")[-1] != "trace":
                continue
            cid, events = None, []
            for child in el:
                tag = child.tag.split("}")[-1]
                if tag == "string" and child.get("key") == "concept:name":
                    cid = child.get("value")
                elif tag == "event":
                    act = stamp = None
                    for x in child:
                        if x.get("key") == "concept:name":
                            act = x.get("value")
                        elif x.get("key") == "time:timestamp":
                            stamp = x.get("value")
                    ms = max(0, int(round(datetime.fromisoformat(stamp).timestamp() * 1000))) if stamp else 0
                    events.append((ms, act))
            if cid in pin:
                cases.setdefault(cid, []).extend(events)
            el.clear()
        return cases

    files = sorted(glob.glob(str(REPO / "data" / "splits" / split / "*.xes")))
    parties = [load(f) for f in files]
    common = set(pin)
    for p in parties:
        common &= set(p)
    parties = [{c: p[c] for c in common} for p in parties]
    acts = sorted({a for p in parties for evs in p.values() for _, a in evs})
    with open(path, "wb") as fh:
        pickle.dump((parties, acts), fh)
    return len(common), len(acts)


# ---------------------------------------------------------------- the rows the secure clustering reads

def projected_rows(merged, params, dict_size):
    """The integer rows secure_kmeans reads: the encoding's counts times the run's own projection matrix."""
    import numpy as np
    import method_study_uniform as ms
    extract = ms.extractor(params["encoding"])
    dim = ms.feature_dim(params["encoding"], dict_size)
    k_proj = params["k_proj"]
    matrix = None
    if params["skip_projection"]:
        if k_proj != dim:
            raise SystemExit(f"{params['cell']}: the log skips the projection but k_proj {k_proj} is not the "
                             f"encoding width {dim}")
    else:
        matrix = achlioptas(k_proj, dim).astype(np.int64)
    blocks, mass = [], 0
    for i in range(0, len(merged), 2000):
        raw = np.asarray([extract(m, dict_size) for m in merged[i:i + 2000]], dtype=np.int64)
        if raw.shape[1] != dim:
            raise SystemExit(f"{params['encoding']}: feature width {raw.shape[1]} is not the expected {dim}")
        mass = max(mass, int(np.abs(raw).sum(axis=1).max(initial=0)))
        blocks.append(raw if matrix is None else raw @ matrix.T)
    return np.vstack(blocks).astype(np.int64), dim, mass


def bigram_slab(merged, dict_size):
    """The d^2 directly-follows count rows the sequence clustering reads (the slab secure_extract persists at 0)."""
    import numpy as np
    import test_secure_extract as se
    out = np.zeros((len(merged), dict_size * dict_size), dtype=np.int32)
    for i in range(0, len(merged), 2000):
        out[i:i + 2000] = [se.extract_bigrams_plaintext(m, dict_size) for m in merged[i:i + 2000]]
    return out


# ---------------------------------------------------------------- Lloyd's loop, the tape's arithmetic

def recip_bits(n_cases):
    """RECIP_BITS of secure_kmeans.mpc: FRAC_BITS + ceil_log2(N) + CENTROID_YMAX_SLACK_BITS."""
    from plaintext_kmeans_reference import recip_bits as twin
    return twin(n_cases)


def stride_rows(n, k):
    """secure_kmeans.mpc::init_centroids: seed j is row j * (N // K)."""
    stride = n // k if k > 0 else 1
    return [j * stride for j in range(k)]


def random_rows(n, k, draws=CLEAR_N_INIT, seed=SEED_DRAW):
    """The `draws` random seedings, k distinct rows each, from random.Random(seed)."""
    rng = random.Random(seed)
    return [rng.sample(range(n), k) for _ in range(draws)]


def centroid_matrix(sums, counts, rounding, g):
    """The rounded centroid of the tape (floor(s_j * floor(2^g / n_j) / 2^(g-f))) or the exact mean."""
    import numpy as np
    if not rounding:
        return sums.astype(np.float64) / counts[:, None]
    recip = ((1 << g) // counts).astype(np.int64)
    prod = sums.astype(np.int64) * recip[:, None]
    if int(np.abs(prod).max(initial=0)) >= PROD_LIMIT:
        raise OverflowError("the centroid product leaves the tape's 2^62 bound; the run would have been refused")
    return prod >> np.int64(g - FRAC_BITS)


def assign(Y, chat, rounding, chunk=20000):
    """e_step: the nearest centroid by squared Euclidean distance, first index on a tie. The tape computes the
    distance in the ring, so a value at or above 2^62 is a run the pipeline refuses
    (`parameters.rs::secure_kmeans_argmin_bits`); a worst case that could leave int64 is summed in float64 beside
    it, where a wrap by 2^64 shows."""
    import numpy as np
    k = len(chat)
    out = np.empty(len(Y), dtype=np.int64)
    scale = 1 << FRAC_BITS
    for i in range(0, len(Y), chunk):
        block = Y[i:i + chunk]
        block = block.astype(np.int64) * scale if rounding else block.astype(np.float64)
        d2 = np.empty((len(block), k), dtype=np.int64 if rounding else np.float64)
        for j in range(k):
            diff = block - chat[j]
            if not rounding:
                d2[:, j] = (diff * diff).sum(axis=1)
                continue
            reach = float(np.abs(diff).max(initial=0))
            if reach ** 2 >= PROD_LIMIT:
                raise OverflowError("one squared coordinate of the E-step leaves the tape's 2^62 bound")
            squares = diff * diff
            total = squares.sum(axis=1)
            if Y.shape[1] * reach ** 2 >= PROD_LIMIT:          # the sum could leave int64: check it did not
                exact = squares.astype(np.float64).sum(axis=1)
                if float(np.abs(exact - total.astype(np.float64)).max(initial=0)) > 2.0 ** 40:
                    raise OverflowError("the E-step distance leaves the tape's 2^62 bound")
            if int(total.max(initial=0)) >= PROD_LIMIT:
                raise OverflowError("the E-step distance leaves the tape's 2^62 bound")
            d2[:, j] = total
        out[i:i + chunk] = np.argmin(d2, axis=1)
    return out


def lloyd(Y, k, iters, init_rows, rounding):
    """(labels, the centroids of the final labeling, the WCSS) of Lloyd's as secure_kmeans.mpc runs it; an empty
    cluster keeps its previous (sums, count), so its centroid stays a valid argmin target."""
    import numpy as np
    n = len(Y)
    g = recip_bits(n)
    sums = Y[init_rows].copy()
    counts = np.ones(k, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        labels = assign(Y, centroid_matrix(sums, counts, rounding, g), rounding)
        fresh = np.bincount(labels, minlength=k)
        for j in range(k):
            if fresh[j] > 0:
                sums[j] = Y[labels == j].sum(axis=0)
                counts[j] = int(fresh[j])
    chat = centroid_matrix(sums, counts, rounding, g)
    value = chat.astype(np.float64) / (1 << FRAC_BITS) if rounding else chat
    wcss = float(((Y.astype(np.float64) - value[labels]) ** 2).sum())
    return labels, chat, wcss


def merge_small(labels, chat, k, min_cluster):
    """scripts/plaintext_kmeans_reference.py::merge_small_clusters_integer, the twin of the tape's oblivious merge."""
    import numpy as np
    from plaintext_kmeans_reference import merge_small_clusters_integer
    merged = merge_small_clusters_integer([int(v) for v in labels], [list(row) for row in chat], k, min_cluster)
    return np.asarray(merged, dtype=np.int64)


def kmeans_labels(Y, k, spec, min_cluster):
    """(labels, the centroids, the note) of one variant's k-means on the rows Y."""
    import numpy as np
    seeds = spec["seeds"]
    if seeds == "stride":
        picks, tags = [stride_rows(len(Y), k)], ["stride"]
    elif seeds == "ten":
        picks, tags = random_rows(len(Y), k), [f"r{j}" for j in range(CLEAR_N_INIT)]
    else:
        j = seeds[1]
        picks, tags = [random_rows(len(Y), k)[j]], [f"r{j}"]
    best = None
    for tag, rows in zip(tags, picks):
        labels, chat, wcss = lloyd(Y, k, spec["iters"], rows, spec["rounding"])
        if best is None or wcss < best[2]:
            best = (labels, chat, wcss, tag)
    labels, chat, _, tag = best
    if min_cluster > 0:
        labels = merge_small(labels, chat, k, min_cluster)
    return labels, chat, tag


def variant_labels(variant, Y, slab, params, dh):
    """(labels over the cases, the seeding kept, a note) of one variant of one run."""
    import numpy as np
    spec = VARIANT[variant]
    k = params["k"]
    rows = Y
    if spec["normalise"]:
        norm = np.linalg.norm(Y.astype(np.float64), axis=1, keepdims=True)
        rows = np.divide(Y.astype(np.float64), norm, out=np.zeros(Y.shape, dtype=np.float64), where=norm > 0)
    if params["method"] == "km":
        labels, _, tag = kmeans_labels(rows, k, spec, params["km_min_cluster"] if spec["merge"] else 0)
        return labels, tag, ""
    from plaintext_sequence_reference import sequence_labels
    init, _, tag = kmeans_labels(rows, k, spec, params["km_min_cluster"])     # 0 in the sequence pipeline
    labels = sequence_labels(slab, k, params["seq_max_iter"], dh, init_labels=[int(v) for v in init],
                             min_cluster=params["seq_min_cluster"] if spec["merge"] else 0)
    return np.asarray(labels, dtype=np.int64), tag, f"the sequence hard-EM at {params['seq_max_iter']} iterations"


# ---------------------------------------------------------------- the stages

def cache_paths(cell):
    return (OUT / "cache" / f"{cell}.json", OUT / "cache" / f"{cell}_y.npy",
            OUT / "cache" / f"{cell}_slab.npy", OUT / "cache" / f"{cell}_traces.json.gz",
            OUT / "cache" / f"{cell}_y_full.npy", OUT / "cache" / f"{cell}_slab_full.npy")


def merged_traces(params, bound):
    """(case ids, the merged event lists, the dictionary size, the activity names) at a per-party length bound."""
    import method_study_uniform as ms
    ms.PICKLES = cases_pickle(params).parent
    return ms.merged_cases(dict(params, length_bound=bound))


def run_folder(run_dir):
    return REPO / (run_dir if run_dir.startswith("output") else f"output/runs/{run_dir}")


def stored_labels(run_dir, ids):
    """The run's own federated assignments, in the study's case order."""
    with open(run_folder(run_dir) / "federated" / "cluster_assignments.json") as fh:
        stored = json.load(fh)
    label = dict(zip(stored["case_ids"], stored["assignments"]))
    if set(label) != set(ids):
        return None
    return [label[c] for c in ids]


def label_check(mine, theirs):
    """V0 against the stored assignments: identical, the cases differing after the best relabel, the ARI."""
    from sklearn.metrics import adjusted_rand_score
    if theirs is None:
        return dict(identical=False, differ=-1, ari=None, passed=False, note="case sets differ")
    pairs = Counter(zip(mine, theirs))
    remap = {}
    for (a, b), _ in pairs.most_common():
        if a not in remap and b not in remap.values():
            remap[a] = b
    differ = sum(remap.get(a, -1) != b for a, b in zip(mine, theirs))
    return dict(identical=list(mine) == list(theirs), differ=int(differ),
                ari=round(float(adjusted_rand_score(theirs, list(mine))), 6),
                passed=list(mine) == list(theirs), note="")


def instrument_hash():
    import hashlib
    return hashlib.md5((REPO / "scripts" / "evaluate_campaign.py").read_bytes()).hexdigest()[:12]


def stage_features(cell):
    import numpy as np
    import method_study_uniform as ms
    recorded = recorded_rows()
    meta_path, y_path, slab_path, tr_path, y_full_path, slab_full_path = cache_paths(cell)
    params = parse_log(cell)
    params["run_dir"] = run_dirs()[cell]
    # The features and the published graph are built from the events the parties truncate at the run's bound; the
    # traces the run exports beside the graph, and that the instrument scores the graph against, are untruncated.
    ids, merged, dict_size, name = merged_traces(params, params["length_bound"])
    traces = [[name[act] for _, act in m] for m in merged]
    ids_full, merged_full, _, _ = merged_traces(params, 10 ** 9)
    if ids_full != ids:
        raise SystemExit(f"{cell}: the case order differs between the two trace builds")
    full = [[name[act] for _, act in m] for m in merged_full]
    if len(ids) != params["cases"]:
        raise SystemExit(f"{cell}: {len(ids)} cases in the clear, the log says {params['cases']}")
    if params["dict_total"] and dict_size != params["dict_total"]:
        raise SystemExit(f"{cell}: dictionary {dict_size} in the clear, the log says {params['dict_total']}")
    Y, dim, mass = projected_rows(merged, params, dict_size)
    np.save(y_path, Y)
    # The same rows without the length bound, for the variant that removes it (V6).
    Y_full, _, mass_full = projected_rows(merged_full, params, dict_size)
    np.save(y_full_path, Y_full)
    if params["method"] == "seq":
        np.save(slab_path, bigram_slab(merged, dict_size))
        np.save(slab_full_path, bigram_slab(merged_full, dict_size))
    with gzip.open(tr_path, "wt") as fh:
        json.dump({"case_ids": ids, "traces": traces, "full_traces": full}, fh)
    checks = dict(graphs=ms.published_graph_check(params, ids, traces, params["run_dir"]),
                  exported_traces=exported_traces_check(params["run_dir"], ids, full))
    checks["graphs"]["passed"] = checks["graphs"]["differ"] == 0 and checks["graphs"]["compared"] > 0
    cut = sum(1 for a, b in zip(traces, full) if a != b)
    meta = dict(params=params, dict_size=dict_size, dim=dim, dh=dict_size, l1_mass=mass,
                l1_mass_full=mass_full, checks=checks, truncated_cases=cut,
                recorded={t: {k: recorded[(cell, t)][k] for k in ("f_score", "fitness", "precision", "clusters")}
                          for t in ("federated", "plaintext") if (cell, t) in recorded})
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return (f"{cell}: {len(ids)} cases ({cut} truncated at L={params['length_bound']}), dict {dict_size}, "
            f"width {dim} -> {Y.shape[1]} "
            f"({'identity' if params['skip_projection'] else 'Achlioptas seed 42'}), L1 mass <= {mass}; "
            f"graphs {checks['graphs']['compared']} cells, {checks['graphs']['differ']} differ; "
            f"exported traces {checks['exported_traces']['note'] or 'equal'}")


def exported_traces_check(run_dir, ids, full):
    """The untruncated traces of the study against the traces the run exported beside its published graphs."""
    folder = run_folder(run_dir) / "federated"
    with open(folder / "cluster_assignments.json") as fh:
        stored = json.load(fh)
    label = dict(zip(stored["case_ids"], stored["assignments"]))
    if set(label) != set(ids):
        return dict(compared=0, differ=-1, note="case sets differ")
    groups = defaultdict(Counter)
    for case, trace in zip(ids, full):
        groups[label[case]][tuple(trace)] += 1
    compared = differ = 0
    for j, want in sorted(groups.items()):
        path = folder / f"cluster_{j}_traces.json"
        if not path.exists():
            differ += sum(want.values())
            continue
        with open(path) as fh:
            got = Counter(tuple(t) for t in json.load(fh)["traces"])
        compared += sum(want.values())
        differ += sum((want - got).values()) + sum((got - want).values())
    return dict(compared=compared, differ=differ, passed=differ == 0 and compared > 0,
                note="" if differ == 0 and compared else f"{differ} of {compared} traces differ")


def stage_labels(cell, variant):
    import numpy as np
    meta_path, y_path, slab_path, tr_path, y_full_path, slab_full_path = cache_paths(cell)
    meta = json.load(open(meta_path))
    params = meta["params"]
    full = VARIANT[variant]["bound"] == "full"
    Y = np.load(y_full_path if full else y_path)
    slab = None
    if params["method"] == "seq":
        slab = np.load(slab_full_path if full else slab_path).astype(np.int64)
    t0 = time.time()
    labels, tag, note = variant_labels(variant, Y, slab, params, meta["dh"])
    if full:
        note = "; ".join(x for x in [note, "the rows of the untruncated traces"] if x)
    sizes = Counter(int(v) for v in labels)
    ids = json.load(gzip.open(tr_path, "rt"))["case_ids"]
    theirs = stored_labels(params["run_dir"], ids)
    check = label_check([int(v) for v in labels], theirs)
    out = dict(cell=cell, variant=variant, k_requested=params["k"], count=len(sizes),
               sizes=sorted(sizes.values(), reverse=True), seed_used=tag, note=note,
               ari_to_run=check["ari"], labels_check=check, seconds=round(time.time() - t0, 1),
               labels=[int(v) for v in labels])
    with open(OUT / "labels" / f"{cell}__{variant}.json", "w") as fh:
        json.dump(out, fh)
    return (f"{cell} {variant}: {len(sizes)} clusters, sizes {out['sizes'][:7]}, seed {tag}, "
            f"ARI to the run {check['ari']}, identical {check['identical']} in {out['seconds']} s")


def stage_score(cell, variant, limit_s):
    from evaluate_campaign import envelope, graph_from_traces, score_graph, summarize
    info = json.load(open(OUT / "labels" / f"{cell}__{variant}.json"))
    cached = json.load(gzip.open(cache_paths(cell)[3], "rt"))
    groups = defaultdict(lambda: ([], []))
    for trace, full, label in zip(cached["traces"], cached["full_traces"], info["labels"]):
        groups[label][0].append(trace)
        groups[label][1].append(full)
    t0, rows = time.time(), []
    for label in sorted(groups):
        cut, full = groups[label]
        # The graph is the one the protocol publishes (the truncated events, the edge filter); the log it is scored
        # against is the one the run exports (untruncated), exactly as the instrument scores a tier folder.
        dfg, start, end = graph_from_traces([t for t in cut if t])
        row = score_graph(dfg, start, end, envelope(dfg, start, end), [t for t in full if t], limit_s)
        row["cluster_id"] = int(label)
        rows.append(row)
    out = {k: v for k, v in info.items() if k != "labels"}
    out.update(summary=summarize(rows), clusters=rows, score_seconds=round(time.time() - t0, 1),
               instrument=instrument_hash())
    with open(OUT / "results" / f"{cell}__{variant}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    s = out["summary"]
    return (f"{cell} {variant}: F {s.get('f_score')} fitness {s.get('fitness')} precision {s.get('precision')} "
            f"edges {s.get('edges')} in {out['score_seconds']} s")


# ---------------------------------------------------------------- the table and the facts

def build_table(cells, variants):
    rows = []
    for cell in cells:
        meta_path = cache_paths(cell)[0]
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        params, rec = meta["params"], meta.get("recorded", {})
        base = None
        base_path = OUT / "results" / f"{cell}__v0_spec.json"
        if base_path.exists():
            base = json.load(open(base_path)).get("summary", {})
        for variant in variants:
            path = OUT / "results" / f"{cell}__{variant}.json"
            if not path.exists():
                continue
            res = json.load(open(path))
            s = res.get("summary", {})
            row = dict(cell=cell, log=params["log_name"], embedding=params["embedding"], method=params["method"],
                       variant=variant, cases=params["cases"], k=params["k"], clusters=res.get("count"),
                       sizes=";".join(str(v) for v in res.get("sizes", [])), models_scored=s.get("models_scored"),
                       ari_to_run=res.get("ari_to_run"), seed_used=res.get("seed_used"),
                       note="; ".join(x for x in [res.get("note"), *params.get("notes", [])] if x))
            for key in ("fitness", "precision", "f_score", "simplicity", "generalization", "edges",
                        "activities", "rel"):
                row[key] = s.get(key)
            row["run_f_score"] = _f(rec.get("federated", {}).get("f_score"))
            row["run_fitness"] = _f(rec.get("federated", {}).get("fitness"))
            row["run_precision"] = _f(rec.get("federated", {}).get("precision"))
            row["run_plaintext_f_score"] = _f(rec.get("plaintext", {}).get("f_score"))
            row["d_f_vs_v0"] = _d(row["f_score"], (base or {}).get("f_score"))
            row["d_f_vs_plaintext"] = _d(row["f_score"], row["run_plaintext_f_score"])
            row["v0_minus_run_f"] = _d((base or {}).get("f_score"), row["run_f_score"])
            rows.append({c: row.get(c) for c in COLUMNS})
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _f(value):
    return float(value) if value not in (None, "", "-") else None


def _d(a, b):
    return round(a - b, 4) if a is not None and b is not None else None


def _span(values):
    return (sum(values) / len(values), min(values), max(values)) if values else (None, None, None)


def facts(rows, cells, variants):
    """THE FACTS, each under the command that produced it."""
    out = []
    cmd = "scripts/causes_study_uniform.py --report"
    by = defaultdict(list)
    for r in rows:
        if not GRID_RE.match(r["cell"]):      # the grid alone; the party-count cells get their own section
            continue
        by[(r["method"], r["variant"], r["log"])].append(r)
        by[(r["method"], r["variant"], "all")].append(r)

    for method, title in (("km", "the Secret-Labels k-means grid (sl_km)"),
                          ("seq", "the Secret-Labels sequence grid (sl_seq)")):
        out.append(f"\n### F of each variant minus F of V0, {title}   [{cmd}]")
        out.append(f"{'variant':<22}{'log':<10}{'runs':>5}{'mean F':>9}{'mean dF':>10}{'lowest':>9}{'highest':>9}")
        for variant in variants:
            wrote = False
            for log in ("sepsis", "helpdesk", "bpic2013", "rtfm", "all"):
                got = [r for r in by[(method, variant, log)] if r["d_f_vs_v0"] is not None]
                if not got:
                    continue
                mean_f, _, _ = _span([r["f_score"] for r in got])
                mean_d, lo, hi = _span([r["d_f_vs_v0"] for r in got])
                out.append(f"{variant:<22}{log:<10}{len(got):>5}{mean_f:>9.4f}{mean_d:>+10.4f}"
                           f"{lo:>+9.4f}{hi:>+9.4f}")
                wrote = True
            if wrote:
                out.append("")
        out.append(f"### F of each variant minus the recorded plaintext F (the clustering in the clear), {title}"
                   f"   [{cmd}]")
        out.append(f"{'variant':<22}{'log':<10}{'runs':>5}{'mean d':>9}{'lowest':>9}{'highest':>9}"
                   f"{'within 0.02':>13}{'at or above':>12}")
        for variant in variants:
            wrote = False
            for log in ("sepsis", "helpdesk", "bpic2013", "rtfm", "all"):
                got = [r for r in by[(method, variant, log)] if r["d_f_vs_plaintext"] is not None]
                if not got:
                    continue
                mean_d, lo, hi = _span([r["d_f_vs_plaintext"] for r in got])
                near = sum(1 for r in got if abs(r["d_f_vs_plaintext"]) <= 0.02)
                above = sum(1 for r in got if r["d_f_vs_plaintext"] >= 0)
                out.append(f"{variant:<22}{log:<10}{len(got):>5}{mean_d:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
                           f"{near:>13}{above:>12}")
                wrote = True
            if wrote:
                out.append("")

    out.append(f"### The gap the study explains: V0 (the recorded secure F) minus the recorded F of the clear "
               f"reference   [{cmd}]")
    out.append(f"{'grid':<8}{'runs':>5}{'mean':>9}{'median':>9}{'lowest':>9}{'highest':>9}{'|gap|<=0.02':>13}"
               f"{'secure >= clear':>17}")
    for method in ("km", "seq", "both"):
        got = [r for r in by[(method, "v0_spec", "all")] if r["d_f_vs_plaintext"] is not None] if method != "both" \
            else [r for r in rows if GRID_RE.match(r["cell"]) and r["variant"] == "v0_spec"
                  and r["d_f_vs_plaintext"] is not None]
        if not got:
            continue
        d = sorted(r["d_f_vs_plaintext"] for r in got)
        mean, lo, hi = _span(d)
        out.append(f"{method:<8}{len(d):>5}{mean:>+9.4f}{d[len(d) // 2]:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
                   f"{sum(1 for x in d if abs(x) <= 0.02):>13}{sum(1 for x in d if x >= 0):>17}")

    out.append(f"\n### The widest single-run effect of each variant over the whole grid   [{cmd}]")
    out.append(f"{'variant':<22}{'runs':>5}{'F equal to V0':>15}{'past 0.02':>11}  widest")
    grid_rows = [r for r in rows if GRID_RE.match(r["cell"])]
    for variant in variants:
        got = [r for r in grid_rows if r["variant"] == variant and r["d_f_vs_v0"] is not None]
        if not got:
            continue
        worst = max(got, key=lambda r: abs(r["d_f_vs_v0"]))
        out.append(f"{variant:<22}{len(got):>5}{sum(1 for r in got if r['d_f_vs_v0'] == 0):>15}"
                   f"{sum(1 for r in got if abs(r['d_f_vs_v0']) > 0.02):>11}  "
                   f"{worst['d_f_vs_v0']:+.4f} on {worst['cell']} (F {worst['f_score']:.4f})")

    outside = sorted({r["cell"] for r in rows if not GRID_RE.match(r["cell"])})
    if outside:
        out.append(f"\n### Cells outside the grid (the first check)   [{cmd} --only '<cell>']")
        out.append(f"{'cell':<26}{'variant':<16}{'F':>9}{'fitness':>9}{'precision':>11}{'edges':>8}"
                   f"{'recorded F':>12}{'clusters':>9}")
        for cell in outside:
            for r in [x for x in rows if x["cell"] == cell]:
                if r["f_score"] is None:
                    continue
                out.append(f"{cell:<26}{r['variant']:<16}{r['f_score']:>9.4f}{r['fitness']:>9.4f}"
                           f"{r['precision']:>11.4f}{r['edges']:>8.2f}"
                           f"{(r['run_f_score'] or 0):>12.4f}{r['clusters']:>9}")

    out.append(f"\n### V0 against the recorded federated F of the run   [{cmd}]")
    v0 = [r for r in rows if r["variant"] == "v0_spec" and r["f_score"] is not None and r["run_f_score"] is not None]
    diffs = [(abs(r["v0_minus_run_f"]), r) for r in v0]
    mean_abs = sum(d for d, _ in diffs) / len(diffs) if diffs else None
    out.append(f"  {len(v0)} runs replayed; mean |V0 - recorded| {mean_abs:.4f}" if diffs else "  no run scored")
    for d, r in sorted(diffs, key=lambda x: -x[0])[:8]:
        out.append(f"  {r['cell']:<28}V0 {r['f_score']:.4f} recorded {r['run_f_score']:.4f} "
                   f"({r['v0_minus_run_f']:+.4f})")
    out.append(f"  equal to four decimals in {sum(1 for d, _ in diffs if d == 0)} of {len(diffs)} runs; "
               f"past 0.0100 in {sum(1 for d, _ in diffs if d > 0.01)}")

    out.append(f"\n### The checks of the replay, per cell   [{cmd}]")
    for cell in cells:
        path = cache_paths(cell)[0]
        if not path.exists():
            out.append(f"  {cell:<28}no cache (the features stage did not run)")
            continue
        meta = json.load(open(path))
        c = meta["checks"]
        lab = {}
        lp = OUT / "labels" / f"{cell}__v0_spec.json"
        if lp.exists():
            lab = json.load(open(lp)).get("labels_check", {})
        out.append(f"  {cell:<28}graphs {c['graphs']['compared']:>4} cells {c['graphs']['differ']} differ"
                   f"{' ' + c['graphs']['note'] if c['graphs']['note'] else ''}"
                   f" | exported traces {c.get('exported_traces', {}).get('differ')} of "
                   f"{c.get('exported_traces', {}).get('compared')} differ"
                   f" | {meta.get('truncated_cases')} cases truncated"
                   f" | V0 labels identical {lab.get('identical')} differ {lab.get('differ')} "
                   f"ARI {lab.get('ari')}")

    stamps = {}
    for cell in cells:
        for variant in variants:
            p = OUT / "results" / f"{cell}__{variant}.json"
            if p.exists():
                stamps.setdefault(json.load(open(p)).get("instrument"), []).append(f"{cell}__{variant}")
    out.append(f"\n### The instrument   [{cmd}]")
    for h, got in sorted(stamps.items(), key=lambda kv: -len(kv[1])):
        out.append(f"  scripts/evaluate_campaign.py md5 {h}: {len(got)} results")
    missing = [(c, v) for c in cells for v in variants if not (OUT / "results" / f"{c}__{v}.json").exists()]
    out.append(f"\n### Missing   [{cmd}]\n  {len(missing)} of {len(cells) * len(variants)} (cell, variant) pairs "
               f"have no result")
    for c, v in missing[:40]:
        out.append(f"  {c} {v}")
    refused = refused_cells()
    out.append(f"  {len(refused)} grid cells are not replayed at all, since the campaign has no finished run of "
               f"them: {', '.join(refused) or 'none'}")
    return "\n".join(out)


# ---------------------------------------------------------------- the driver

def worker(jobs, workers):
    def run(job):
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, *job], cwd=str(REPO), capture_output=True, text=True)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if p.returncode != 0:
            return f"FAILED {' '.join(job)} rc={p.returncode}: {p.stderr.strip()[-400:]}"
        return f"{line}  [{time.time() - t0:.0f} s]"

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for what in pool.map(run, jobs):
            failed += what.startswith("FAILED")
            print(what, flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=None, help="all | cases | features | labels | score")
    ap.add_argument("--only", default=None, help="a regex on the cell name")
    ap.add_argument("--variants", default=None, help="a comma-separated subset of the variants")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-s", type=int, default=1800, help="alignment seconds per model, the instrument's default")
    ap.add_argument("--report", action="store_true", help="write results.csv and print the facts from what is on disk")
    ap.add_argument("--stage", default=None, help="the worker mode: cases | features | labels | score")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    for sub in ("cases", "cache", "labels", "results"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    cells = all_cells() if args.only else grid_cells()
    if args.only:
        cells = [c for c in cells if re.search(args.only, c)]
    variants = args.variants.split(",") if args.variants else list(DEFAULT_VARIANTS)
    for v in variants:
        if v not in VARIANT:
            ap.error(f"unknown variant {v}; known: {', '.join(sorted(VARIANT))}")

    if args.stage:
        if args.stage == "cases":
            n, d = build_cases(*args.cell.split("::"), CASES / f"cs_{args.cell.replace('::', '__')}.pkl")
            print(f"{args.cell}: {n} common cases, {d} activities")
        elif args.stage == "features":
            print(stage_features(args.cell))
        elif args.stage == "labels":
            print(stage_labels(args.cell, args.variant))
        elif args.stage == "score":
            print(stage_score(args.cell, args.variant, args.limit_s))
        else:
            ap.error("--stage cases|features|labels|score")
        return 0

    failed = 0
    if args.stages in ("all", "cases"):
        want = {f"{p['split']}::{p['caselist']}" for p in (parse_log(c) for c in cells)}
        todo = [w for w in sorted(want)
                if not any((b / f"cs_{w.replace('::', '__')}.pkl").exists() for b in (BORROWED_CASES, CASES))]
        print(f"{len(want)} (split, case list) pairs; {len(todo)} to build", flush=True)
        failed += worker([["--stage", "cases", "--cell", w] for w in todo], min(args.workers, 4))
    if args.stages in ("all", "features"):
        failed += worker([["--stage", "features", "--cell", c] for c in cells], min(args.workers, 4))
    if args.stages in ("all", "labels"):
        failed += worker([["--stage", "labels", "--cell", c, "--variant", v] for c in cells for v in variants],
                         min(args.workers, 4))
    if args.stages in ("all", "score"):
        failed += worker([["--stage", "score", "--cell", c, "--variant", v, "--limit-s", str(args.limit_s)]
                          for c in cells for v in variants], args.workers)
    if args.stages or args.report:
        rows = build_table(sorted(set(all_cells()) | set(cells)), list(VARIANT))
        print(f"\nresults.csv holds {len(rows)} rows over {len({r['cell'] for r in rows})} cells")
        print(facts(rows, cells, variants))
    if failed:
        print(f"\n{failed} jobs FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
