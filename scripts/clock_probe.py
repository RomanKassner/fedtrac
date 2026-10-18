#!/usr/bin/env python3
"""Clock record of a campaign cell (scripts/run_uniform_campaign.sh): the monotonic clock against real time.

The binary times every round with the monotonic clock, and under WSL2 that clock can run fast for hours while the
host holds the realtime clock to its own by stepping it back, so a round time is read with the gain of its cell.

    clock_probe.py probe                  the clocks now, one line: realtime monotonic monotonic_raw ntp_offset
                                          cpu_all cpu_own
    clock_probe.py row <cell> "<probe>"   the cell's row of clock.tsv, from the probe taken before it and one taken now
    clock_probe.py header                 the header of clock.tsv
    clock_probe.py upgrade <clock.tsv>    rewrite a file of an earlier header under the current one ("-" where a
                                          column did not exist)
    clock_probe.py selftest [seconds]     two probes that many seconds apart (default 30); exit 3 when the monotonic
                                          clock left the reference, exit 4 when no NTP server answered

ntp_offset is the realtime clock minus the time of an NTP server (CLOCK_NTP_HOST, default ntp.ubuntu.com; "-" asks
none) and "-" when no answer came. The reference of a cell is its realtime span corrected by the change of that offset
("ntp"), or the realtime span alone when an offset is missing ("realtime"). `row` exits 3 when the monotonic span
differs from the reference by more than CLOCK_GAIN_BOUND_PCT (default 0.5) and by more than GAIN_FLOOR_S.

cpu_all is the CPU time every process of the machine has used since boot (/proc/stat without idle and iowait),
cpu_own that of the control group the probe runs in, so cpu_other_s is the CPU time spent outside a campaign run as
a user service. Started by hand the probe shares its group with the login shell and the column then misses what else
that shell started.
"""
import argparse
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime, timezone

COLUMNS = ["cell", "start_utc", "real_s", "mono_s", "raw_s", "ntp_off_start_s", "ntp_off_end_s", "ref", "ref_s",
           "mono_gain_pct", "cpu_all_s", "cpu_own_s", "cpu_other_s"]
NTP_EPOCH = 2208988800  # seconds from 1900 to 1970
GAIN_FLOOR_S = 0.25     # below this difference a gain is the noise of two NTP answers, not a clock
GAIN_BOUND_PCT = 0.5


def gain_bound():
    return float(os.environ.get("CLOCK_GAIN_BOUND_PCT", GAIN_BOUND_PCT))


def _ntp_time(data, at):
    sec, frac = struct.unpack("!II", data[at:at + 8])
    return sec - NTP_EPOCH + frac / 2**32


def _ntp_query(host, timeout):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            t1 = time.time()
            s.sendto(b"\x1b" + 47 * b"\0", (host, 123))
            data, _ = s.recvfrom(64)
            t4 = time.time()
    except OSError:
        return None
    if len(data) < 48 or data[1] == 0:  # stratum 0: the server refuses or is unsynchronised
        return None
    return ((t1 - _ntp_time(data, 32)) + (t4 - _ntp_time(data, 40))) / 2


def ntp_offset(host, timeout=2.0):
    """Realtime clock minus the server's time in seconds; None without a usable answer."""
    # In a thread, because the name lookup is not bound by the socket timeout and a cell must not wait for it.
    answer = []
    t = threading.Thread(target=lambda: answer.append(_ntp_query(host, timeout)), daemon=True)
    t.start()
    t.join(timeout + 1.0)
    return answer[0] if answer else None


def cpu_all(stat="/proc/stat"):
    """CPU seconds all processes have used since boot (no idle, no iowait); None when unreadable."""
    try:
        with open(stat) as fh:
            f = fh.readline().split()
        # user nice system idle iowait irq softirq steal: guest time is already inside user and nice
        return sum(int(f[i]) for i in (1, 2, 3, 6, 7, 8)) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def cpu_own(cgroup="/proc/self/cgroup", root="/sys/fs/cgroup"):
    """CPU seconds of the control group this process runs in; None without a version-2 group."""
    try:
        with open(cgroup) as fh:
            path = next(line.strip().split("::", 1)[1] for line in fh if line.startswith("0::"))
        with open(os.path.join(root, path.lstrip("/"), "cpu.stat")) as fh:
            return next(int(line.split()[1]) for line in fh if line.startswith("usage_usec")) / 1e6
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def probe(host):
    """(realtime, monotonic, monotonic_raw, ntp_offset, cpu_all, cpu_own); the clocks are read right after the
    answer, the last three are None when unavailable."""
    off = ntp_offset(host) if host and host != "-" else None
    return (time.clock_gettime(time.CLOCK_REALTIME), time.clock_gettime(time.CLOCK_MONOTONIC),
            time.clock_gettime(time.CLOCK_MONOTONIC_RAW), off, cpu_all(), cpu_own())


def fmt(p):
    p = tuple(p) + (None,) * (6 - len(p))
    return " ".join(f"{v:.6f}" for v in p[:3]) + " " + " ".join("-" if v is None else f"{v:.6f}" for v in p[3:6])


def parse(line):
    """A probe line; one of four fields (before the CPU columns) reads as one without CPU times."""
    parts = line.split()
    parts += ["-"] * (6 - len(parts))
    return (*map(float, parts[:3]), *(None if v == "-" else float(v) for v in parts[3:6]))


def record(cell, start, end):
    """The clock.tsv row of a cell from the probes before and after it."""
    start, end = (tuple(p) + (None,) * (6 - len(p)) for p in (start, end))
    real, mono, raw = (e - s for s, e in zip(start[:3], end[:3]))
    if start[3] is not None and end[3] is not None:
        ref, ref_s = "ntp", real - (end[3] - start[3])
    else:
        ref, ref_s = "realtime", real
    gain = 100.0 * (mono - ref_s) / ref_s if ref_s > 0 else 0.0
    span = lambda i: None if start[i] is None or end[i] is None else end[i] - start[i]  # noqa: E731
    all_s, own_s = span(4), span(5)
    other_s = None if all_s is None or own_s is None else max(all_s - own_s, 0.0)
    off = lambda v, digits=3: "-" if v is None else f"{v:.{digits}f}"  # noqa: E731
    return {"cell": cell, "start_utc": datetime.fromtimestamp(start[0], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "real_s": f"{real:.3f}", "mono_s": f"{mono:.3f}", "raw_s": f"{raw:.3f}",
            "ntp_off_start_s": off(start[3]), "ntp_off_end_s": off(end[3]), "ref": ref, "ref_s": f"{ref_s:.3f}",
            "mono_gain_pct": f"{gain:.2f}", "cpu_all_s": off(all_s, 1), "cpu_own_s": off(own_s, 1),
            "cpu_other_s": off(other_s, 1)}


def out_of_bound(row, bound_pct):
    """True when the monotonic span left the reference by more than the bound and the floor."""
    diff = abs(float(row["mono_s"]) - float(row["ref_s"]))
    return diff > GAIN_FLOOR_S and abs(float(row["mono_gain_pct"])) > bound_pct


def read_rows(path):
    """The rows of a clock.tsv as dicts over COLUMNS, whichever header the file was written under."""
    rows = []
    try:
        with open(path) as fh:
            lines = [line.rstrip("\n") for line in fh if line.strip()]
    except OSError:
        return rows
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        got = dict(zip(header, line.split("\t")))
        rows.append({c: got.get(c, "-") for c in COLUMNS})
    return rows


def upgrade(path):
    """Rewrite `path` under the current header; a file that already carries it is left alone."""
    try:
        with open(path) as fh:
            first = fh.readline().rstrip("\n")
    except OSError:
        return False
    if first == "\t".join(COLUMNS):
        return False
    rows = read_rows(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in COLUMNS) + "\n")
    os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("header")
    r = sub.add_parser("row")
    r.add_argument("cell")
    r.add_argument("start", help="the line `probe` printed before the cell")
    u = sub.add_parser("upgrade")
    u.add_argument("path")
    s = sub.add_parser("selftest")
    s.add_argument("seconds", nargs="?", type=float, default=30.0)
    args = ap.parse_args()
    host = os.environ.get("CLOCK_NTP_HOST", "ntp.ubuntu.com")
    if args.cmd == "header":
        print("\t".join(COLUMNS))
    elif args.cmd == "probe":
        print(fmt(probe(host)))
    elif args.cmd == "upgrade":
        print(f"{args.path}: {'rewritten under the current header' if upgrade(args.path) else 'unchanged'}")
    elif args.cmd == "selftest":
        start = probe(host)
        time.sleep(args.seconds)
        row = record("selftest", start, probe(host))
        print(f"monotonic {row['mono_gain_pct']} % against {row['ref']} over {row['ref_s']} s")
        if out_of_bound(row, gain_bound()):
            sys.exit(3)
        if row["ref"] != "ntp":
            sys.exit(4)
    else:
        row = record(args.cell, parse(args.start), probe(host))
        print("\t".join(row[c] for c in COLUMNS))
        if out_of_bound(row, gain_bound()):
            sys.exit(3)


if __name__ == "__main__":
    main()
