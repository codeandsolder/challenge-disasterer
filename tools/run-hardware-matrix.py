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
LIVE = ("altcha", "capjs", "cerberus", "goaway", "haphash", "mcaptcha", "anubis")
ANUBIS_GPU_QUANTUM = 8192
PREEXISTING_BASELINE_QUANTUM = 160_000


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
        "rustc": text(["rustc", "-Vv"]),
        "kernel": text(["uname", "-a"]),
        "vulkan": text(["bash", "-lc", "vulkaninfo --summary 2>/dev/null | rg 'deviceName|driverName|driverInfo' || true"]),
        "nvidia": text(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader 2>/dev/null || true"]),
        "tools": {
            "go": text(["bash", "-lc", "go version 2>/dev/null || true"]),
            "node": text(["bash", "-lc", "node --version 2>/dev/null || true"]),
            "haproxy": text(["bash", "-lc", "haproxy -v 2>/dev/null | head -n1 || true"]),
            "chromium": text(["bash", "-lc", "chromium --version 2>/dev/null || true"]),
            "sccache": text(["bash", "-lc", "sccache --version 2>/dev/null || true"]),
        },
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
    if protocol in {"goaway", "capjs", "mcaptcha", "haphash"} and "sha_ni" in flags:
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
    parser.add_argument("--throughput-limit", type=int, default=1_048_576)
    parser.add_argument(
        "--argon2-limit",
        type=int,
        default=64,
        help="Exact Argon2id candidate budget per deterministic matrix sample.",
    )
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preexisting-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preexisting-limit",
        type=int,
        default=1_120_000,
        help="Common fixed-work budget aligned to the historical pow-buster solver quanta.",
    )
    parser.add_argument("--browser-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--haphash-debian-live", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--all-implementations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run authoritative classic-protocol verifiers against every CPU implementation available on this host.",
    )
    args = parser.parse_args()
    if not args.cpu and not args.gpu:
        parser.error("at least one of --cpu or --gpu must be enabled")
    if args.gpu and args.throughput_limit % ANUBIS_GPU_QUANTUM != 0:
        parser.error(
            f"--throughput-limit must be a multiple of {ANUBIS_GPU_QUANTUM} when --gpu is enabled"
        )

    if args.preexisting_baseline and args.preexisting_limit % PREEXISTING_BASELINE_QUANTUM != 0:
        parser.error(
            f"--preexisting-limit must be a multiple of {PREEXISTING_BASELINE_QUANTUM}"
        )

    meta = metadata()
    if args.require_clean and meta["git_status"]:
        parser.error(f"refusing retained benchmark from dirty tree:\n{meta['git_status']}")
    meta["parameters"] = {
        "samples": args.samples,
        "correctness_limit": args.correctness_limit,
        "throughput_limit": args.throughput_limit,
        "argon2_limit": args.argon2_limit,
        "cpu": args.cpu,
        "gpu": args.gpu,
        "live": args.live,
        "preexisting_baseline": args.preexisting_baseline,
        "external_baselines": args.external_baselines,
        "preexisting_limit": args.preexisting_limit,
        "browser_reference": args.browser_reference,
        "haphash_debian_live": args.haphash_debian_live,
        "all_implementations": args.all_implementations,
        "require_clean": args.require_clean,
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    statuses: dict[str, int] = {}
    matrix_bin = ROOT / "target/release/challenge-disasterer-solver-matrix"
    matrix_build = run([
        "cargo", "build", "--locked", "--release",
        "--manifest-path", "tools/solver-matrix/Cargo.toml",
    ])
    statuses["build-solver-matrix"] = matrix_build.returncode

    gpu_adapters: list[str] = []
    if matrix_build.returncode == 0 and args.gpu:
        discovered = subprocess.run(
            [str(matrix_bin), "--list-gpu-adapters"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        statuses["gpu-adapter-discovery"] = discovered.returncode
        if discovered.returncode == 0:
            gpu_adapters = json.loads(discovered.stdout)
            if not gpu_adapters:
                statuses["gpu-adapter-discovery"] = 1
    meta["gpu_adapters"] = gpu_adapters
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    if matrix_build.returncode == 0:
        for phase, limit in (("correctness", args.correctness_limit), ("throughput", args.throughput_limit)):
            with (out / f"solver-{phase}.jsonl").open("wb") as matrix_stdout, \
                 (out / f"solver-{phase}.stderr.log").open("wb") as matrix_stderr:
                common = [
                    str(matrix_bin),
                    "--samples", str(args.samples), "--limit", str(limit),
                    "--argon2-limit", str(args.argon2_limit),
                    "--mode", phase, "--commit", meta["git_sha"],
                ]
                if args.cpu:
                    cpu = run(
                        [*common, "--cpu", "true", "--gpu", "false"],
                        stdout=matrix_stdout,
                        stderr=matrix_stderr,
                    )
                    statuses[f"solver-{phase}-cpu"] = cpu.returncode
                for index, adapter in enumerate(gpu_adapters):
                    env = os.environ.copy()
                    env["WGPU_ADAPTER_NAME"] = adapter
                    gpu = run(
                        [*common, "--cpu", "false", "--gpu", "true"],
                        env=env,
                        stdout=matrix_stdout,
                        stderr=matrix_stderr,
                    )
                    statuses[f"solver-{phase}-gpu-{index}"] = gpu.returncode

    if args.gpu and gpu_adapters:
        specialized_stdout = out / "solver-specialized-anubis-gpu.jsonl"
        specialized_stderr = out / "solver-specialized-anubis-gpu.stderr.log"
        build = run(["cargo", "build", "--locked", "--release"])
        statuses["build-specialized-anubis-gpu"] = build.returncode
        if build.returncode == 0:
            specialized_codes = []
            with specialized_stdout.open("wb") as stdout, specialized_stderr.open("wb") as stderr:
                for adapter in gpu_adapters:
                    env = os.environ.copy()
                    env["WGPU_ADAPTER_NAME"] = adapter
                    for sample in range(args.samples):
                        specialized = run(
                            [
                                str(ROOT / "target/release/challenge-disasterer"),
                                "--benchmark-anubis-gpu-hashes", str(args.throughput_limit),
                                "--benchmark-sample", str(sample),
                            ],
                            env=env,
                            stdout=stdout,
                            stderr=stderr,
                        )
                        specialized_codes.append(specialized.returncode)
            statuses["solver-specialized-anubis-gpu"] = next(
                (code for code in specialized_codes if code != 0), 0
            )

    if args.cpu and args.preexisting_baseline:
        baseline_manifest = ROOT / "tools" / "preexisting-baseline" / "Cargo.toml"
        baseline_out = out / "preexisting-baseline.jsonl"
        baseline_err = out / "preexisting-baseline.stderr.log"
        baseline_env = os.environ.copy()
        # The historical comparator is a nested Cargo workspace but path-depends on
        # the in-tree solver. Canonical-path sccache needs their common source root
        # declared explicitly so both workspaces stay inside one canonical build root.
        baseline_env["SCCACHE_CANONICAL_BUILD_ROOT"] = str(ROOT)
        build = run([
            "cargo", "build", "--locked", "--release",
            "--manifest-path", str(baseline_manifest),
        ], env=baseline_env)
        statuses["build-preexisting-baseline"] = build.returncode
        if build.returncode == 0:
            baseline_bin = ROOT / "tools" / "preexisting-baseline" / "target" / "release" / "challenge-disasterer-preexisting-baseline"
            with baseline_out.open("wb") as stdout, baseline_err.open("wb") as stderr:
                result = run([str(baseline_bin), "--samples", str(args.samples), "--limit", str(args.preexisting_limit)], stdout=stdout, stderr=stderr)
            statuses["preexisting-baseline"] = result.returncode

    if args.external_baselines:
        external_out = out / "external-baselines.jsonl"
        external_err = out / "external-baselines.stderr.log"
        external_cmd = [
            "uv", "run", str(ROOT / "tools" / "external-baselines.py"),
            "--samples", str(args.samples),
            "--hashx-limit", str(args.throughput_limit),
            "--argon2-limit", str(args.argon2_limit),
            "--opencl-limit", str(args.throughput_limit),
            "--cpu" if args.cpu else "--no-cpu",
            "--gpu" if args.gpu else "--no-gpu",
        ]
        with external_out.open("wb") as stdout, external_err.open("wb") as stderr:
            result = run(external_cmd, stdout=stdout, stderr=stderr)
        statuses["external-baselines"] = result.returncode

    if args.live:
        for name in LIVE:
            script = ROOT / "ci" / f"live-{name}-e2e.sh"
            if name == "anubis":
                anubis_args = [
                    "--fuzz",
                    "--fuzz-repeats", str(max(1, min(args.samples, 4))),
                    "--benchmark",
                    "--benchmark-samples", str(args.samples),
                ]
                browser_reference_pending = args.browser_reference
                if args.cpu:
                    cpu_args = [
                        *anubis_args,
                        "--browser-reference", str(browser_reference_pending).lower(),
                    ]
                    with (out / "live-anubis-cpu.log").open("wb") as log:
                        result = run([str(script), *cpu_args], stdout=log, stderr=subprocess.STDOUT)
                    statuses["live-anubis-cpu"] = result.returncode
                    browser_reference_pending = False
                if args.gpu and gpu_adapters:
                    for index, adapter in enumerate(gpu_adapters):
                        env = os.environ.copy()
                        env["WGPU_ADAPTER_NAME"] = adapter
                        gpu_args = [
                            *anubis_args,
                            "--browser-reference", str(browser_reference_pending).lower(),
                            "--solver-gpu",
                            "--gpu-only",
                            "--exercise-all-gpu-adapters",
                            "false",
                        ]
                        with (out / f"live-anubis-gpu-{index}.log").open("wb") as log:
                            result = run(
                                [str(script), *gpu_args],
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        statuses[f"live-anubis-gpu-{index}"] = result.returncode
                        browser_reference_pending = False
                continue

            cases = classic_implementations(name, args.all_implementations, bool(gpu_adapters))
            upstream_variants = (
                (("current", None), ("v0.4.6", "v0.4.6"), ("v0.4.5", "v0.4.5"))
                if name == "cerberus"
                else (("current", None),)
            )
            for upstream_label, upstream_ref in upstream_variants:
                for implementation in cases:
                    adapter_cases = [(None, None)]
                    if implementation == "wgpu":
                        adapter_cases = [(index, adapter) for index, adapter in enumerate(gpu_adapters)]
                    for adapter_index, adapter in adapter_cases:
                        variant_suffix = "" if upstream_label == "current" else f"-{upstream_label}"
                        adapter_suffix = "" if adapter_index is None else f"-gpu-{adapter_index}"
                        suffix = f"{name}{variant_suffix}-{implementation}{adapter_suffix}"
                        env = os.environ.copy()
                        env["CHALLENGE_IMPLEMENTATION"] = implementation
                        if adapter is not None:
                            env["WGPU_ADAPTER_NAME"] = adapter
                        if upstream_ref is not None:
                            env["CERBERUS_REF"] = upstream_ref
                        if name == "haphash" and args.haphash_debian_live:
                            env["HAPHASH_DEBIAN_LIVE"] = "1"
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
