#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# ///
"""Run the local challenge-disasterer solver + authoritative-verifier matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ("altcha", "capjs", "cerberus", "goaway", "mcaptcha", "anubis")


def run(argv: list[str], *, env: dict[str, str] | None = None, stdout=None, stderr=None) -> subprocess.CompletedProcess:
    print("+", shlex.join(argv), file=sys.stderr, flush=True)
    return subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)


def text(argv: list[str]) -> str:
    result = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.strip()


def metadata() -> dict:
    return {
        "schema": 1,
        "timestamp_unix_ms": time.time_ns() // 1_000_000,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
                if line.startswith("model name") and ":" in line
            ),
            "unknown",
        ),
        "git_sha": text(["git", "rev-parse", "HEAD"]),
        "git_status": text(["git", "status", "--short"]),
        "rustc": text(["rustc", "+nightly-2026-10-08", "-Vv"]),
        "kernel": text(["uname", "-a"]),
        "vulkan": text(["bash", "-lc", "vulkaninfo --summary 2>/dev/null | rg 'deviceName|driverName|driverInfo' || true"]),
        "nvidia": text(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader 2>/dev/null || true"]),
        "wgp_adapter_name": os.environ.get("WGPU_ADAPTER_NAME"),
    }


def cpu_flags() -> set[str]:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.startswith("flags") and ":" in line:
            return set(line.split(":", 1)[1].split())
    return set()


def classic_implementations(protocol: str, all_implementations: bool, gpu: bool) -> list[str]:
    if not all_implementations:
        return ["auto"]
    flags = cpu_flags()
    implementations = ["auto", "safe"]
    if "avx2" in flags:
        implementations.append("avx2")
    if protocol in {"goaway", "capjs", "mcaptcha"} and "sha_ni" in flags:
        implementations.append("sha-ni")
    if "avx512f" in flags:
        implementations.append("avx512")
    if gpu:
        implementations.append("wgpu")
    return implementations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--correctness-limit", type=int, default=50_000_000)
    parser.add_argument("--throughput-limit", type=int, default=1_000_000)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--all-implementations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run authoritative classic-protocol verifiers against every CPU implementation available on this host.",
    )
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = metadata()
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    statuses: dict[str, int] = {}
    for phase, limit in (("correctness", args.correctness_limit), ("throughput", args.throughput_limit)):
        matrix_stdout = (out / f"solver-{phase}.jsonl").open("wb")
        matrix_stderr = (out / f"solver-{phase}.stderr.log").open("wb")
        matrix = run(
            [
                "cargo", "+nightly-2026-10-08", "run", "--locked", "--release",
                "--manifest-path", "tools/solver-matrix/Cargo.toml", "--",
                "--samples", str(args.samples), "--limit", str(limit),
                "--mode", phase,
                "--gpu", str(args.gpu).lower(), "--commit", meta["git_sha"],
            ],
            stdout=matrix_stdout,
            stderr=matrix_stderr,
        )
        statuses[f"solver-{phase}"] = matrix.returncode
        matrix_stdout.close()
        matrix_stderr.close()

    if args.live:
        for name in LIVE:
            script = ROOT / "ci" / f"live-{name}-e2e.sh"
            if name == "anubis":
                cases = [None]
            else:
                cases = classic_implementations(name, args.all_implementations, args.gpu)
            for implementation in cases:
                suffix = name if implementation is None else f"{name}-{implementation}"
                env = os.environ.copy()
                if implementation is not None:
                    env["CHALLENGE_IMPLEMENTATION"] = implementation
                with (out / f"live-{suffix}.log").open("wb") as log:
                    result = run([str(script)], env=env, stdout=log, stderr=subprocess.STDOUT)
                statuses[f"live-{suffix}"] = result.returncode

    manifest = {"metadata": meta, "statuses": statuses}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failures = {name: code for name, code in statuses.items() if code != 0}
    if failures:
        print(json.dumps({"failed": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
