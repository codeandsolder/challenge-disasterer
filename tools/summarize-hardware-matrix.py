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
    for anubis_log in sorted(run.glob("live-anubis*.log")):
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
    baselines = []
    baseline_path = run / "preexisting-baseline.jsonl"
    if baseline_path.is_file():
        baselines = [json.loads(line) for line in baseline_path.read_text().splitlines() if line.strip()]
    return meta, rows, paired, benches, baselines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    metas = {}
    paired_hashx = []
    anubis_benchmarks = []
    baseline_rows = []
    for run in args.runs:
        meta, rows, paired, benches, baselines = load(run)
        host = meta["hostname"]
        metas[host] = meta
        paired_hashx.extend({"host": host, **item} for item in paired)
        anubis_benchmarks.extend({"host": host, **item} for item in benches)
        baseline_rows.extend({"host": host, **item} for item in baselines)
        for row in rows:
            work_budget = row["attempted_work"] if row["phase"] == "throughput" else None
            key = (
                row["phase"], row["protocol"], row["variant"], row["implementation"],
                row["accelerator"], host, row.get("gpu_adapter"), work_budget,
            )
            grouped[key].append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        phase, protocol, variant, implementation, accelerator, host, gpu_adapter, work_budget = key
        valid = all(row["target_condition_ok"] for row in rows)
        summary.append({
            "phase": phase,
            "protocol": protocol,
            "variant": variant,
            "implementation": implementation,
            "accelerator": accelerator,
            "host": host,
            "gpu_adapter": gpu_adapter,
            "work_budget": work_budget,
            "samples": len(rows),
            "all_target_conditions_valid": valid,
            "median_solve_ms": statistics.median(row["solve_ns"] for row in rows) / 1e6,
            "median_work_per_second": statistics.median(row["work_per_second"] for row in rows),
            "median_attempted_work": statistics.median(row["attempted_work"] for row in rows),
        })
    baseline_grouped = defaultdict(list)
    for row in baseline_rows:
        baseline_grouped[(row["host"], row["baseline"], row["protocol"], row["variant"], row["implementation"])].append(row)
    baseline_summary = []
    for key, rows in sorted(baseline_grouped.items()):
        host, baseline, protocol, variant, implementation = key
        baseline_summary.append({
            "host": host, "baseline": baseline, "protocol": protocol, "variant": variant,
            "implementation": implementation, "samples": len(rows),
            "median_work_per_second": statistics.median(row["work_per_second"] for row in rows),
            "median_solve_ms": statistics.median(row["solve_ns"] for row in rows) / 1e6,
            "median_attempted_work": statistics.median(row["attempted_work"] for row in rows),
        })

    comparisons = []
    legacy_by_key = {(r["host"], r["protocol"], r["variant"]): r for r in baseline_summary if r["baseline"] != "current"}
    current_auto = {(r["host"], r["protocol"], r["variant"]): r for r in baseline_summary if r["baseline"] == "current"}
    for key, legacy in sorted(legacy_by_key.items()):
        host, protocol, variant = key
        matched = current_auto.get(key)
        candidates = [
            r for r in summary
            if r["host"] == host
            and r["phase"] == "throughput"
            and r["protocol"] == protocol
            and r["variant"] == variant
            and r["accelerator"] == "cpu"
            and r["median_attempted_work"] == legacy["median_attempted_work"]
        ]
        best = max(candidates, key=lambda r: r["median_work_per_second"], default=None)
        if matched is not None and matched["median_attempted_work"] != legacy["median_attempted_work"]:
            raise ValueError(
                f"historical exact-work mismatch for {host}/{protocol}/{variant}: "
                f"current={matched['median_attempted_work']} legacy={legacy['median_attempted_work']}"
            )
        comparisons.append({
            "host": host, "protocol": protocol, "variant": variant,
            "legacy_baseline": legacy["baseline"],
            "legacy_median_work_per_second": legacy["median_work_per_second"],
            "matched_current_auto_exact_work_ratio": None if matched is None else matched["median_work_per_second"] / legacy["median_work_per_second"],
            "best_current_exact_work_implementation": None if best is None else best["implementation"],
            "best_current_exact_work_ratio": None if best is None else best["median_work_per_second"] / legacy["median_work_per_second"],
        })

    print(json.dumps({
        "hosts": metas,
        "results": summary,
        "preexisting_baselines": baseline_summary,
        "preexisting_comparison": comparisons,
        "paired_hashx": paired_hashx,
        "anubis_benchmarks": anubis_benchmarks,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
