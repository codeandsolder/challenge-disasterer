#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# ///
"""Compare one or more hardware-matrix run directories by identical benchmark keys."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PAIRED_RE = re.compile(
    r"^PAIRED_HASHX difficulty=(?P<difficulty>\d+) implementation=(?P<implementation>\S+) "
    r"n=(?P<n>\d+) mean_solver_ms=(?P<mean_solver_ms>[0-9.]+)$"
)
BENCH_RE = re.compile(
    r"^BENCH algorithm=(?P<algorithm>\S+) solver=(?P<solver>\S+) backend=(?P<backend>\S+) "
    r"implementation=(?P<implementation>\S+) difficulty=(?P<difficulty>\d+) n=(?P<n>\d+) "
    r"avg_answer_ms=(?P<avg_answer_ms>[0-9.]+) effective_h_s=(?P<effective_h_s>[0-9.]+) "
    r"effective_mh_s=(?P<effective_mh_s>[0-9.]+) mean_solver_ms=(?P<mean_solver_ms>[0-9.]+)$"
)


def load(run: Path):
    meta = json.loads((run / "metadata.json").read_text())
    rows = []
    for path in sorted(run.glob("solver-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "phase" not in row:
                row["phase"] = "throughput"
            if "target_condition_ok" not in row:
                row["target_condition_ok"] = True
            rows.append(row)
    paired = []
    benches = []
    anubis_log = run / "live-anubis.log"
    if anubis_log.is_file():
        for line in anubis_log.read_text(errors="replace").splitlines():
            if match := PAIRED_RE.match(line):
                item = match.groupdict()
                paired.append({
                    "difficulty": int(item["difficulty"]),
                    "implementation": item["implementation"],
                    "samples": int(item["n"]),
                    "mean_solver_ms": float(item["mean_solver_ms"]),
                })
            elif match := BENCH_RE.match(line):
                item = match.groupdict()
                benches.append({
                    "algorithm": item["algorithm"],
                    "solver": item["solver"],
                    "backend": item["backend"],
                    "implementation": None if item["implementation"] == "-" else item["implementation"],
                    "difficulty": int(item["difficulty"]),
                    "samples": int(item["n"]),
                    "avg_answer_ms": float(item["avg_answer_ms"]),
                    "effective_h_s": float(item["effective_h_s"]),
                    "mean_solver_ms": float(item["mean_solver_ms"]),
                })
    return meta, rows, paired, benches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    metas = {}
    paired_hashx = []
    anubis_benchmarks = []
    for run in args.runs:
        meta, rows, paired, benches = load(run)
        host = meta["hostname"]
        metas[host] = meta
        paired_hashx.extend({"host": host, **item} for item in paired)
        anubis_benchmarks.extend({"host": host, **item} for item in benches)
        for row in rows:
            key = (row["phase"], row["protocol"], row["variant"], row["implementation"], row["accelerator"], host, row.get("gpu_adapter"))
            grouped[key].append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        phase, protocol, variant, implementation, accelerator, host, gpu_adapter = key
        valid = all(row["target_condition_ok"] for row in rows)
        summary.append({
            "phase": phase,
            "protocol": protocol,
            "variant": variant,
            "implementation": implementation,
            "accelerator": accelerator,
            "host": host,
            "gpu_adapter": gpu_adapter,
            "samples": len(rows),
            "all_target_conditions_valid": valid,
            "median_solve_ms": statistics.median(row["solve_ns"] for row in rows) / 1e6,
            "median_work_per_second": statistics.median(row["work_per_second"] for row in rows),
            "median_attempted_work": statistics.median(row["attempted_work"] for row in rows),
        })
    print(json.dumps({
        "hosts": metas,
        "results": summary,
        "paired_hashx": paired_hashx,
        "anubis_benchmarks": anubis_benchmarks,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
