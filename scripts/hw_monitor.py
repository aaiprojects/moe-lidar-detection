#!/usr/bin/env python3
"""Sample hardware state to a crash-durable log.

This machine (DGX Spark / GB10) has hard-reset four times during notebook runs
without leaving a single diagnostic behind: no OOM kill, no kernel panic, no GPU
Xid, no thermal trip. The journal simply stops mid-boot. When the kernel never
gets to react, the only way to learn anything is to have already written the
evidence to disk.

So every sample is followed by flush + fsync. That is deliberately more
expensive than buffered I/O, and it is the entire point: a hard power event can
then lose at most the final sample, instead of everything since the last 4 KB
page filled.

Usage:
    python scripts/hw_monitor.py &                     # 2 s samples, default log
    python scripts/hw_monitor.py --interval 1
    python scripts/hw_monitor.py --log /tmp/hw.csv

Read it back after a crash with:
    python scripts/hw_monitor.py --summary
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO / "outputs" / "hw_monitor.csv"

FIELDNAMES = [
    "ts",
    "event",
    "label",
    "mem_used_gb",
    "mem_avail_gb",
    "swap_used_gb",
    "soc_temp_c",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_util_pct",
    "gpu_mem_used_gb",
    "top_proc_rss_gb",
    "top_proc_name",
]


def read_meminfo() -> dict[str, float]:
    """Return MemTotal/MemAvailable/SwapTotal/SwapFree in GB from /proc/meminfo."""
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: dict[str, float] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in wanted:
                values[key] = int(rest.split()[0]) / 1e6  # kB -> GB
                if len(values) == len(wanted):
                    break
    return values


def read_soc_temp() -> float | None:
    """Hottest ACPI thermal zone, in Celsius.

    These zones read 74-77 C even at idle on this box, so the absolute number
    matters less than how far it climbs under sustained load.
    """
    temps: list[float] = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            temps.append(int((zone / "temp").read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


def read_gpu(nvidia_smi: str | None) -> dict[str, float | None]:
    """Query the GPU via nvidia-smi. Fields unsupported on GB10 come back None."""
    blank = {
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_util_pct": None,
        "gpu_mem_used_gb": None,
    }
    if nvidia_smi is None:
        return blank

    query = "temperature.gpu,power.draw.average,utilization.gpu,memory.used"
    try:
        out = subprocess.run(
            [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return blank
    if out.returncode != 0 or not out.stdout.strip():
        return blank

    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

    def num(raw: str, scale: float = 1.0) -> float | None:
        try:
            return round(float(raw) * scale, 2)
        except ValueError:  # "N/A" / "[Not Supported]"
            return None

    if len(parts) < 4:
        return blank
    return {
        "gpu_temp_c": num(parts[0]),
        "gpu_power_w": num(parts[1]),
        "gpu_util_pct": num(parts[2]),
        "gpu_mem_used_gb": num(parts[3], 1 / 1024.0),  # MiB -> GB
    }


def read_top_process() -> tuple[float | None, str | None]:
    """Largest-RSS process, to attribute memory growth to the notebook kernel."""
    best_rss = -1
    best_name = None
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "statm").read_text().split()
            rss_pages = int(fields[1])
            if rss_pages > best_rss:
                best_rss = rss_pages
                best_name = (proc / "comm").read_text().strip()
        except (OSError, ValueError, IndexError):
            continue  # process exited while we were reading it
    if best_rss < 0:
        return None, None
    return round(best_rss * os.sysconf("SC_PAGE_SIZE") / 1e9, 2), best_name


def build_row(event: str = "sample", label: str = "") -> dict:
    mem = read_meminfo()
    gpu = read_gpu(shutil.which("nvidia-smi"))
    rss, name = read_top_process()
    swap_used = mem.get("SwapTotal", 0.0) - mem.get("SwapFree", 0.0)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "label": label,
        "mem_used_gb": round(mem.get("MemTotal", 0.0) - mem.get("MemAvailable", 0.0), 2),
        "mem_avail_gb": round(mem.get("MemAvailable", 0.0), 2),
        "swap_used_gb": round(swap_used, 2),
        "soc_temp_c": read_soc_temp(),
        "top_proc_rss_gb": rss,
        "top_proc_name": name,
        **gpu,
    }


def append_row(log_path: Path, row: dict) -> None:
    """Append one row, then force it to physical storage."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def mark(label: str, log_path: Path = DEFAULT_LOG) -> None:
    """Record a named checkpoint, so a crash can be tied to a pipeline step.

    Safe to call from a notebook: monitoring must never be the thing that breaks
    the run it is monitoring, so all errors are swallowed.
    """
    try:
        append_row(Path(log_path), build_row(event="marker", label=label))
    except Exception:
        pass


def summarize(log_path: Path, tail: int = 25) -> int:
    if not log_path.exists():
        print(f"No log at {log_path}", file=sys.stderr)
        return 1
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{log_path} is empty", file=sys.stderr)
        return 1

    def peak(field: str):
        vals = [(float(r[field]), r["ts"]) for r in rows if r.get(field)]
        return max(vals) if vals else None

    print(f"{len(rows)} samples in {log_path}")
    print(f"first {rows[0]['ts']}  last {rows[-1]['ts']}\n")
    for field in ("mem_used_gb", "swap_used_gb", "soc_temp_c", "gpu_temp_c", "gpu_power_w"):
        hit = peak(field)
        if hit:
            print(f"  peak {field:<16} {hit[0]:>8.2f}   at {hit[1]}")

    print(f"\nLast {tail} rows (the final one is where it died):\n")
    show = ["ts", "event", "mem_used_gb", "soc_temp_c", "gpu_temp_c", "gpu_power_w", "label"]
    print("  " + "  ".join(f"{c:>12}" for c in show))
    for r in rows[-tail:]:
        print("  " + "  ".join(f"{(r.get(c) or ''):>12}" for c in show))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", action="store_true", help="print the existing log and exit")
    parser.add_argument("--mark", metavar="LABEL", help="write one marker row and exit")
    args = parser.parse_args()

    if args.summary:
        return summarize(args.log)
    if args.mark:
        mark(args.mark, args.log)
        return 0

    print(f"Sampling every {args.interval}s -> {args.log} (Ctrl-C to stop)")
    append_row(args.log, build_row(event="monitor_start", label=f"interval={args.interval}s"))
    try:
        while True:
            append_row(args.log, build_row())
            time.sleep(args.interval)
    except KeyboardInterrupt:
        append_row(args.log, build_row(event="monitor_stop"))
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
