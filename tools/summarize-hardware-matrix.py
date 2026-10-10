#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# ///
"""Compare one or more hardware-matrix run directories by identical benchmark keys."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(run: Path):
    meta = json.loads((run / "metadata.json").read_text())
    rows = []
    for path in sorted(run.glob("solver-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return meta, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    metas = {}
    for run in args.runs:
        meta, rows = load(run)
        host = meta["hostname"]
        metas[host] = meta
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
    print(json.dumps({"hosts": metas, "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
