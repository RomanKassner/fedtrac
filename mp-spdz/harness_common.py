"""Shared scaffolding for the eight mp-spdz/test_*.py harnesses: inputs, persistence, compile, run, the CLI."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

# The tape's own constants, imported from the clear twin rather than re-declared.
from scripts.plaintext_kmeans_reference import (  # noqa: E402
    CENTROID_YMAX_SLACK_BITS as SK_CENTROID_YMAX_SLACK_BITS,
    FRAC_BITS as SK_FRAC_BITS,
    ceil_log2 as ceil_log2,
)

# Player-Data holds Input-P{0..7}-0; every writer truncates all of them, so no harness reads another's leftovers.
MAX_PARTIES = 8

# Process-wide knobs, set once by setup() from the common CLI.
RING = 64
NO_SPLIT = False
EXTRA_COMPILE_FLAGS: list[str] | None = None
THREADS = 4
ONLY: set[str] | None = None
SPDZ_DIR: Path | None = None
PROFILE = False

# Per program, the position of the PROFILE compile argument and each .mpc's own defaults for the optional arguments
# before it, so padding a short cell up to it adds only timers.
PROFILE_ARG = {
    "secure_extract": (15, {8: 0, 9: 0, 10: "0", 11: 0, 12: 0, 13: 0, 14: 0}),
    "secure_extract_rsecret": (18, {8: 0, 9: 1, 10: 2000, 11: 0, 12: 2,
                                    13: "0", 14: 0, 15: 0, 16: 0, 17: 0}),
    "secure_sequence_cluster": (10, {8: 0, 9: 0}),
}

# Shortest argument list any cell passes per program; _check_profile_tables asserts the padding covers it at import.
PROFILE_MIN_ARGS = {
    "secure_extract": 7,
    "secure_extract_rsecret": 7,
    "secure_sequence_cluster": 7,
}


def _check_profile_tables():
    for program, (index, defaults) in PROFILE_ARG.items():
        needed = set(range(PROFILE_MIN_ARGS[program] + 1, index))
        missing = sorted(needed - set(defaults))
        assert not missing, (
            f"PROFILE_ARG[{program!r}] has no padding default for position(s) "
            f"{missing}; `--profile` would raise KeyError on the shortest "
            f"call site ({PROFILE_MIN_ARGS[program]} args)")


_check_profile_tables()


def with_profile(program, args):
    """args plus a trailing PROFILE=1 when --profile is on."""
    if not PROFILE or program not in PROFILE_ARG:
        return args
    index, defaults = PROFILE_ARG[program]
    out = list(args)
    if len(out) >= index:
        # The call site already spells the PROFILE slot: switch it on in place.
        out[index - 1] = 1
        return out
    for pos in range(len(out) + 1, index):
        if pos not in defaults:
            raise RuntimeError(
                f"--profile: PROFILE_ARG[{program!r}] has no default for "
                f"position {pos} (call site passed {len(args)} args)")
        out.append(defaults[pos])
    return out + [1]


# secure_extract_rsecret.mpc's own NORM_PRESHIFT default, which every harness cell runs at.
NORM_PRESHIFT_DEFAULT = 2


def encoding_mass_bound(encoding: int, l_total: int) -> int:
    """Twin of parameters.rs::encoding_mass_bound: the worst-case L1 mass of one merged trace, per ENCODING flag."""
    l = int(l_total)
    if encoding == 0:
        return l + 1
    if encoding == 1:
        return (l + 1) * (l + 2) // 2
    if encoding == 3:
        return max(l, 1)
    if encoding == 4:
        return max(l - 2, 1)
    if encoding == 5:
        return 2 * l + 1
    raise ValueError(f"unknown ENCODING flag {encoding}")


def secure_kmeans_argmin_bits(k_proj: int, n_cases: int, y_max: int) -> int:
    """Twin of parameters.rs::secure_kmeans_argmin_bits, the E-step comparison width; 64 is the refusal sentinel."""
    y_bits = ceil_log2(y_max)
    if y_bits > SK_CENTROID_YMAX_SLACK_BITS:
        return 64
    recip_bits = SK_FRAC_BITS + ceil_log2(n_cases) + SK_CENTROID_YMAX_SLACK_BITS
    distance_bits = ceil_log2(k_proj) + 2 * (y_bits + SK_FRAC_BITS) + 3
    # +2: the attainable magnitude needs y_bits + recip_bits bits plus one, and the sign one more.
    centroid_product_bits = y_bits + recip_bits + 2
    return max(distance_bits, centroid_product_bits)


def relabel_tail_cmp_bits(k_proj: int, norm_preshift: int = NORM_PRESHIFT_DEFAULT) -> int:
    """Twin of rounds.rs::relabel_tail_cmp_bits, the TAIL_CMP_BITS of secure_relabel.mpc."""
    return ceil_log2(81 * int(k_proj)) + 32 + int(norm_preshift) + 3


# Pin both width twins against the Rust values at one Sepsis shape, so a drift on either side breaks this at import.
assert encoding_mass_bound(3, 56) == 56
assert secure_kmeans_argmin_bits(19, 1050, 56) == 52, (
    "secure_kmeans_argmin_bits twin disagrees with the Rust value 52 at the "
    "Sepsis pin (N=1050, k_proj=19, y_max=56)")


def player_dir(spdz_dir) -> Path:
    d = Path(spdz_dir) / "Player-Data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def truncate_party_inputs(spdz_dir) -> None:
    """Empty every Input-P*-0, so no harness's party data stays readable by the next."""
    pdir = player_dir(spdz_dir)
    for p in range(MAX_PARTIES):
        (pdir / f"Input-P{p}-0").write_text("")


def write_party_inputs(spdz_dir, texts) -> None:
    """The one input writer: truncate all party streams, then write texts[p] as party p's input."""
    truncate_party_inputs(spdz_dir)
    pdir = player_dir(spdz_dir)
    for p, text in enumerate(texts):
        (pdir / f"Input-P{p}-0").write_text(text)


def write_party0_values(spdz_dir, values, one_per_line: bool = True) -> None:
    """Party 0 supplies values; every other party supplies nothing."""
    if one_per_line:
        text = "".join(f"{int(v)}\n" for v in values)
    else:
        text = " ".join(str(int(v)) for v in values) + "\n"
    write_party_inputs(spdz_dir, [text])


def write_party0_rows(spdz_dir, rows) -> None:
    """Party 0 supplies a row-major integer matrix."""
    write_party0_values(spdz_dir, [v for row in rows for v in row],
                        one_per_line=False)


# Activity encoding; twin of src/types.rs and mpc_utils/constants.py.
PAD_ACT = 0
PAD_TS = 0
START_ACT = 1
END_ACT = 2
FIRST_USER_ACT = 3
TRACE_LEN_ALIGNMENT = 4  # twin of src/constants.rs::TRACE_LEN_ALIGNMENT


def write_nparty_inputs(spdz_dir, all_encoded, n_parties, n_cases, partial_lens):
    """Write the n-party (activity, timestamp) fixture in the format of Rust's write_inputs_nparty."""
    lens = ([int(partial_lens)] * n_parties if isinstance(partial_lens, int)
            else [int(x) for x in partial_lens])
    assert len(lens) == n_parties, (lens, n_parties)
    texts = []
    for p in range(n_parties):
        parts = []
        for case_events in all_encoded[p]:
            for act, ts in case_events:
                parts.append(f"{act} {ts} ")
        for _ in range(len(all_encoded[p]), n_cases):
            parts.append("0 0 " * lens[p])
        parts.append("\n")
        texts.append("".join(parts))
    write_party_inputs(spdz_dir, texts)


def derive_partial_lens(all_encoded, alignment=TRACE_LEN_ALIGNMENT):
    """Per-party L_p: the longest real slice, rounded up to the disclosed alignment."""
    lens = []
    for party in all_encoded:
        real = max((sum(1 for act, _ in case if act != PAD_ACT) for case in party),
                   default=0)
        lens.append(max(alignment, -(-real // alignment) * alignment))
    return lens


def retruncate_per_party(all_encoded, partial_lens):
    """Re-truncate and pad each party to its own L_p, so the clear twin runs on the events the tape reads."""
    out = []
    for party, lp in zip(all_encoded, partial_lens):
        new_party = []
        for case in party:
            real = [e for e in case if e[0] != PAD_ACT][:lp]
            real += [(PAD_ACT, PAD_TS)] * (lp - len(real))
            new_party.append(real)
        out.append(new_party)
    return out


def clear_persistence(spdz_dir, n_parties: int = MAX_PARTIES) -> None:
    """Drop stale transactions, memory dumps and the slab sidecar, the set input_encoding.rs::SecretsCleanup wipes."""
    spdz_dir = Path(spdz_dir)
    pdir = spdz_dir / "Persistence"
    pdir.mkdir(parents=True, exist_ok=True)
    for p in range(max(n_parties, MAX_PARTIES)):
        f = pdir / f"Transactions-P{p}.data"
        if f.exists():
            f.unlink()
    pl = player_dir(spdz_dir)
    for f in pl.glob("Memory-*-P*"):
        f.unlink()
    slab = pl / "extract_slab.json"
    if slab.exists():
        slab.unlink()


def cleanup_all(spdz_dir) -> None:
    """Full post-run wipe of the party streams and the persistence."""
    try:
        truncate_party_inputs(spdz_dir)
        clear_persistence(spdz_dir)
    except OSError as exc:  # never mask the real failure
        print(f"  [cleanup] WARNING: {exc}")


def resolve_compile_flags(compile_flags, n_parties):
    """Per-cell compile flags: explicit compile_flags wins, then --compile-flags or the environment, else -Z."""
    if compile_flags is not None:
        return [str(f) for f in compile_flags]
    if EXTRA_COMPILE_FLAGS:
        return list(EXTRA_COMPILE_FLAGS)
    env = os.environ.get("TEST_MPC_COMPILE_FLAGS", "").split()
    if env:
        return env
    if n_parties and not NO_SPLIT:
        return ["-Z", str(int(n_parties))]
    return []


def _dep_paths(src_mpc):
    """Every file the tape depends on: the .mpc and all of mpc_utils/, whose kernels are inlined into the bytecode."""
    deps = [src_mpc] if src_mpc.exists() else []
    utils_src = HERE / "mpc_utils"
    if utils_src.is_dir():
        deps += sorted(utils_src.glob("*.py"))
    return deps


PROJECTION_MATRIX = "projection_matrix.txt"


def _dep_digest(spdz_dir, src_mpc, deps):
    """sha256 over the dependency set's content, as src/mpc/runtime.rs does; the projection matrix joins it."""
    h = hashlib.sha256()
    for p in deps:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    proj = Path(spdz_dir) / "Programs" / "Source" / PROJECTION_MATRIX
    if proj.exists() and src_mpc.exists() and PROJECTION_MATRIX in src_mpc.read_text():
        h.update(PROJECTION_MATRIX.encode())
        h.update(proj.read_bytes())
    return h.hexdigest()


def prune_dense(row, d):
    """Clear twin of secure_aggregate.mpc Step 3c on one dense d*d row, via bridge/utils.py."""
    dfg, start, end = {}, {}, {}
    for f, w in enumerate(row):
        if w == 0:
            continue
        u, v = divmod(f, d)
        if u == START_ACT and v == END_ACT:
            continue
        if u == START_ACT:
            start[v] = w
        elif v == END_ACT:
            end[u] = w
        else:
            dfg[(u, v)] = w
    from bridge.utils import prune_to_start_end_paths
    dfg, start, end, removed = prune_to_start_end_paths(dfg, start, end)
    out = [0] * (d * d)
    out[START_ACT * d + END_ACT] = row[START_ACT * d + END_ACT]
    for (u, v), w in dfg.items():
        out[u * d + v] = w
    for u, w in start.items():
        out[START_ACT * d + u] = w
    for u, w in end.items():
        out[u * d + END_ACT] = w
    return out, removed


def compile_mpc(spdz_dir, program, args, compile_flags=None, n_parties=None,
                ring=None):
    """Compile a secure program and return the name MP-SPDZ runs; the bytecode cache is keyed on ring, flags and a
    digest of the whole dependency set, so an mpc_utils-only change cannot leave a harness on a stale tape."""
    flags = resolve_compile_flags(compile_flags, n_parties)
    args = with_profile(program, args)
    ring = RING if ring is None else ring
    prog_name = f"{program}-{'-'.join(str(a) for a in args)}" if args else program
    flag_str = f"-R {ring} " + " ".join(flags)

    src_dir = HERE
    dst_dir = Path(spdz_dir) / "Programs" / "Source"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_mpc = src_dir / f"{program}.mpc"
    if src_mpc.exists():
        shutil.copy2(src_mpc, dst_dir / f"{program}.mpc")

    utils_src = src_dir / "mpc_utils"
    utils_dst = dst_dir / "mpc_utils"
    if utils_src.is_dir():
        if utils_dst.exists():
            shutil.rmtree(utils_dst)
        shutil.copytree(utils_src, utils_dst)

    bc_path = Path(spdz_dir) / "Programs" / "Bytecode" / f"{prog_name}-0.bc"
    flag_sidecar = Path(spdz_dir) / "Programs" / "Bytecode" / f"{prog_name}.testflags"
    deps = _dep_paths(src_mpc)
    # Newest mtime over the whole dependency set; the digest catches a content change that left the mtimes alone.
    dep_mtime = max((p.stat().st_mtime for p in deps), default=0.0)
    bc_mtime = bc_path.stat().st_mtime if bc_path.exists() else 0
    stamp = f"{flag_str}\n{_dep_digest(spdz_dir, src_mpc, deps)}"
    cached_flags = flag_sidecar.read_text() if flag_sidecar.exists() else ""

    if not bc_path.exists() or dep_mtime > bc_mtime or cached_flags != stamp:
        print(f"  Compiling {prog_name} [{flag_str}]...")
        cmd = (["python3", "./compile.py", "-R", str(ring)] + flags
               + [program] + [str(a) for a in args])
        result = subprocess.run(cmd, cwd=spdz_dir, capture_output=True,
                                text=True, timeout=900)
        if result.returncode != 0:
            print(f"  COMPILE FAILED:\n{result.stderr}")
            raise RuntimeError(f"Compilation failed: {result.stderr[:500]}")
        flag_sidecar.write_text(stamp)
        print(f"  Compiled {prog_name}")
    else:
        print(f"  Using cached bytecode for {prog_name}")

    return prog_name


def run_mpc(spdz_dir, prog_name, n_parties, timeout=600, binary=None):
    """Run prog_name with n parties and return (party-0 stdout, wall seconds)."""
    print(f"  Running {prog_name} with {n_parties} parties...")
    start = time.time()

    party_flags = os.environ.get("TEST_MPC_PARTY_FLAGS", "").split()
    binary = binary or "semi2k-party.x"
    procs = []
    for p in range(n_parties):
        cmd = [str(Path(spdz_dir) / binary),
               "-p", str(p), "-N", str(n_parties), prog_name] + party_flags
        procs.append(subprocess.Popen(cmd, cwd=spdz_dir,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE))

    results = []
    for proc in procs:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            results.append((proc.returncode, stdout.decode(), stderr.decode()))
        except subprocess.TimeoutExpired:
            for pp in procs:
                pp.kill()
            raise RuntimeError(f"MPC timed out after {timeout}s")

    elapsed = time.time() - start
    print(f"  MPC completed in {elapsed:.1f}s")

    if PROFILE:
        # The timer summary goes to stderr, the per-timer lines to stdout; party 0 is the one the harnesses read.
        for stream in (results[0][1], results[0][2]):
            for line in stream.splitlines():
                if line.startswith("Time") and " = " in line:
                    print(f"  [profile] {line}")

    for p, (rc, _, stderr) in enumerate(results):
        if rc != 0:
            print(f"  Party {p} failed (exit {rc}): {stderr[:300]}")
            raise RuntimeError(f"Party {p} failed: {stderr[:300]}")

    return results[0][1], elapsed


def common_parser(description: str) -> argparse.ArgumentParser:
    """The CLI every harness shares; the caller adds its own options to it."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--spdz-dir", default=os.path.expanduser("~/MP-SPDZ"))
    ap.add_argument("--threads", type=int, default=4,
                    help="compile-time N_THREADS for cells that take it")
    ap.add_argument("--ring", type=int, default=64,
                    help="compile.py -R width")
    ap.add_argument("--compile-flags", default="",
                    help="replace the default -Z <n_parties> with these flags")
    ap.add_argument("--no-split", action="store_true",
                    help="compile without -Z, the no-split fallback")
    ap.add_argument("--only", default="",
                    help="comma-separated cell names to run (default: all)")
    ap.add_argument("--profile", action="store_true",
                    help="compile the timer-carrying programs with PROFILE=1 "
                         "and echo party 0's timer lines")
    return ap


def setup(args) -> Path:
    """Apply the common CLI to the process-wide knobs and check the binary."""
    global RING, NO_SPLIT, EXTRA_COMPILE_FLAGS, THREADS, ONLY, SPDZ_DIR, PROFILE
    RING = int(getattr(args, "ring", 64))
    NO_SPLIT = bool(getattr(args, "no_split", False))
    EXTRA_COMPILE_FLAGS = (getattr(args, "compile_flags", "") or "").split() or None
    THREADS = int(getattr(args, "threads", 4))
    PROFILE = bool(getattr(args, "profile", False))
    only = (getattr(args, "only", "") or "").strip()
    ONLY = {c.strip() for c in only.split(",") if c.strip()} or None

    SPDZ_DIR = spdz_dir = Path(args.spdz_dir)
    if not (spdz_dir / "semi2k-party.x").exists():
        print(f"ERROR: semi2k-party.x not found in {spdz_dir}", file=sys.stderr)
        sys.exit(1)
    return spdz_dir


def cell_enabled(name: str) -> bool:
    """True unless --only was given and excludes this cell."""
    return ONLY is None or name in ONLY


def run_harness(name: str, fn, spdz_dir=None) -> int:
    """Run a harness body, always cleaning MP-SPDZ afterwards, and print the measured wall time."""
    t0 = time.time()
    status = "FAIL"
    try:
        ok = fn()
        status = "PASS" if ok else "FAIL"
        return 0 if ok else 1
    finally:
        cleanup_all(spdz_dir or SPDZ_DIR or Path.home() / "MP-SPDZ")
        print(f"\n[harness] {name}: {status} in {time.time() - t0:.1f}s "
              f"(ring {RING}, split {'off' if NO_SPLIT else 'on'})")
