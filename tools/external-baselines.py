#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2",
#   "pyopencl>=2025.2",
# ]
# ///
"""Benchmark pinned independent PoW implementations as external performance ceilings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyopencl as cl

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "external-baselines"

SOURCES = {
    "anubis_offload": (
        "https://github.com/DavidBuchanan314/anubis_offload.git",
        "3c06b948b8704a81f7aed46be716e35e489aef24",
    ),
    "hashx": (
        "https://github.com/tevador/hashx.git",
        "08babdf4f41b0b8991d1fa94914c7c6902de0cb6",
    ),
    "argon2": (
        "https://github.com/P-H-C/phc-winner-argon2.git",
        "f57e61e19229e23c4445b85494dbf7c07de721cb",
    ),
}

STEPS_PER_TASK = 0x100


def run(argv: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(argv), file=sys.stderr, flush=True)
    return subprocess.run(
        argv,
        cwd=cwd or ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )


def ensure_checkout(name: str) -> Path:
    url, rev = SOURCES[name]
    dst = CACHE / "src" / name
    if not (dst / ".git").is_dir():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", rev], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "--detach", "--force", rev], check=True)
    actual = subprocess.check_output(["git", "-C", str(dst), "rev-parse", "HEAD"], text=True).strip()
    if actual != rev:
        raise RuntimeError(f"{name}: expected {rev}, got {actual}")
    return dst


def emit(*, baseline: str, protocol: str, variant: str, implementation: str,
         accelerator: str, sample: int, attempted: int, ns: int,
         gpu_adapter: str | None = None, comparison_scope: str = "primitive-ceiling") -> None:
    print(json.dumps({
        "schema": 1,
        "baseline": baseline,
        "protocol": protocol,
        "variant": variant,
        "implementation": implementation,
        "accelerator": accelerator,
        "gpu_adapter": gpu_adapter,
        "sample": sample,
        "attempted_work": attempted,
        "solve_ns": ns,
        "work_per_second": attempted * 1e9 / ns,
        "comparison_scope": comparison_scope,
    }, separators=(",", ":")), flush=True)


def build_hashx(src: Path) -> Path:
    build = CACHE / "build" / "hashx"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "-j"], check=True)
    return build / "hashx-bench"


HASHX_RATE = re.compile(r"^([0-9.]+) hashes/sec\.$", re.MULTILINE)


def bench_hashx(samples: int, limit: int) -> None:
    src = ensure_checkout("hashx")
    binary = build_hashx(src)
    rev = SOURCES["hashx"][1][:12]
    for sample in range(samples):
        result = run(
            [str(binary), "--start", str(1000 + sample), "--seeds", "1",
             "--nonces", str(limit), "--threads", "1", "--diff", "2147483647"],
            capture=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        match = HASHX_RATE.search(result.stdout)
        if not match:
            raise RuntimeError(f"unable to parse hashx-bench output:\n{result.stdout}")
        rate = float(match.group(1))
        ns = max(1, round(limit * 1e9 / rate))
        emit(
            baseline=f"tevador/hashx@{rev}",
            protocol="anubis",
            variant="hashx-v1-primitive",
            implementation="compiled-jit",
            accelerator="cpu",
            sample=sample,
            attempted=limit,
            ns=ns,
        )


ARGON2_ADAPTER = r"""
#include <argon2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t ns_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) abort();
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const uint32_t sample = (uint32_t)strtoul(argv[1], NULL, 10);
    const uint64_t limit = strtoull(argv[2], NULL, 10);
    uint8_t pwd[48];
    memset(pwd, 0x42, sizeof(pwd));
    memcpy(pwd, &sample, sizeof(sample));
    pwd[47] = 0x80;
    uint8_t out[32], salt[8];
    volatile uint8_t sink = 0;
    uint64_t start = ns_now();
    for (uint64_t nonce = 0; nonce < limit; ++nonce) {
        memcpy(salt, &nonce, sizeof(salt));
        int rc = argon2_hash(2, 19 * 1024, 1,
            pwd, sizeof(pwd), salt, sizeof(salt),
            out, sizeof(out), NULL, 0, Argon2_id, ARGON2_VERSION_13);
        if (rc != ARGON2_OK) {
            fprintf(stderr, "argon2_hash: %s\n", argon2_error_message(rc));
            return 1;
        }
        sink ^= out[31];
    }
    uint64_t elapsed = ns_now() - start;
    printf("%llu %u\n", (unsigned long long)elapsed, (unsigned)sink);
    return 0;
}
"""


def build_argon2(src: Path) -> Path:
    build = CACHE / "build" / "argon2"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "-C", str(src), "-j", "libargon2.a"], check=True)
    adapter = build / "argon2-fixed.c"
    adapter.write_text(ARGON2_ADAPTER)
    binary = build / "argon2-fixed"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O3", "-march=native",
        "-I", str(src / "include"), str(adapter), str(src / "libargon2.a"),
        "-pthread", "-o", str(binary),
    ], check=True)
    return binary


def bench_argon2(samples: int, limit: int) -> None:
    src = ensure_checkout("argon2")
    binary = build_argon2(src)
    rev = SOURCES["argon2"][1][:12]
    for sample in range(samples):
        result = run([str(binary), str(sample), str(limit)], capture=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        ns = int(result.stdout.split()[0])
        emit(
            baseline=f"P-H-C/phc-winner-argon2@{rev}",
            protocol="anubis",
            variant="wasm-argon2id-1.28",
            implementation="reference-c-native",
            accelerator="cpu",
            sample=sample,
            attempted=limit,
            ns=ns,
            comparison_scope="exact-candidate",
        )


def load_sha256_prefix(src: Path):
    spec = importlib.util.spec_from_file_location("anubis_offload_sha256", src / "sha256.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load anubis_offload sha256.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sha256_prefix


def opencl_gpu_devices() -> list[cl.Device]:
    devices: list[cl.Device] = []
    for platform in cl.get_platforms():
        try:
            devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
        except cl.LogicError:
            pass
    return devices


def bench_anubis_offload(samples: int, limit: int) -> None:
    if limit % STEPS_PER_TASK:
        raise ValueError(f"OpenCL limit must be a multiple of {STEPS_PER_TASK}")
    src = ensure_checkout("anubis_offload")
    prefix_state = load_sha256_prefix(src)
    kernel_src = (src / "sha256.cl").read_text()
    challenge = b"A" * 64
    initial_h = np.array(prefix_state(challenge), dtype=np.uint32)
    rev = SOURCES["anubis_offload"][1][:12]
    devices = opencl_gpu_devices()
    if not devices:
        print("external-baselines: no OpenCL GPU devices; skipping anubis_offload", file=sys.stderr)
        return
    global_size = limit // STEPS_PER_TASK
    for device in devices:
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, kernel_src).build()
        kernel = cl.Kernel(program, "mine")
        mf = cl.mem_flags
        found = np.array([1], dtype=np.uint32)
        nonce = np.array([0], dtype=np.uint64)
        result_h = np.zeros(8, dtype=np.uint32)
        found_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found)
        nonce_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nonce)
        result_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_h)
        initial_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_h)
        kernel.set_args(
            found_buf, nonce_buf, result_buf, initial_buf,
            np.uint64(0), np.uint32(len(challenge)), np.uint32(0xFFFFFFFF), np.uint32(0xFFFFFFFF),
        )
        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), None)
        queue.finish()
        for sample in range(samples):
            start = time.perf_counter_ns()
            cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), None)
            queue.finish()
            ns = time.perf_counter_ns() - start
            emit(
                baseline=f"DavidBuchanan314/anubis_offload@{rev}",
                protocol="anubis",
                variant="fast-decimal-sha256",
                implementation="opencl-prefix-final-block",
                accelerator="gpu",
                gpu_adapter=f"{device.platform.name}: {device.name}",
                sample=sample,
                attempted=limit,
                ns=ns,
                comparison_scope="protocol-kernel-ceiling",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--hashx-limit", type=int, default=1_048_576)
    parser.add_argument("--argon2-limit", type=int, default=64)
    parser.add_argument("--opencl-limit", type=int, default=1_048_576)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        bench_hashx(args.samples, args.hashx_limit)
        bench_argon2(args.samples, args.argon2_limit)
    if args.gpu:
        bench_anubis_offload(args.samples, args.opencl_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
