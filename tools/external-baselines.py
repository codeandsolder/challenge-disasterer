#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2",
#   "argon2-cffi>=25",
#   "pyopencl>=2025.2",
# ]
# ///
"""Benchmark pinned independent PoW implementations as external references and ceilings."""

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
from argon2.low_level import Type as Argon2Type, hash_secret_raw
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
    "hashwx": (
        "https://github.com/tevador/hashwx.git",
        "74b567a31276a4c5d7cb232a5b7639467f49f961",
    ),
    "tor_hashx": (
        "https://gitlab.torproject.org/tpo/core/tor.git",
        "a279bd94de1093563bbc32841880c34eb4c0bf69",
    ),
    "arti_hashx": (
        "https://gitlab.torproject.org/tpo/core/arti.git",
        "213443e39d4ea93dee12cdda8c610c3cc6ac7f9f",
    ),
    "argon2": (
        "https://github.com/P-H-C/phc-winner-argon2.git",
        "f57e61e19229e23c4445b85494dbf7c07de721cb",
    ),
}

STEPS_PER_TASK = 0x100
OPENCL_WARMUP_RUNS = 3


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
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dst)], check=True, stdout=sys.stderr)
        sparse = {
            "tor_hashx": "src/ext/equix",
            "arti_hashx": "crates/hashx",
        }.get(name)
        if sparse is not None:
            subprocess.run(["git", "-C", str(dst), "sparse-checkout", "set", sparse], check=True, stdout=sys.stderr)
    subprocess.run(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", rev], check=True, stdout=sys.stderr)
    subprocess.run(["git", "-C", str(dst), "checkout", "--detach", "--force", rev], check=True, stdout=sys.stderr)
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


HASHX_VECTOR_HEX = "aebdd50aa67c93afb82a4c534603b65e46decd584c55161c526ebc099415ccf1"

HASHX_ADAPTER = r"""
#include <hashx.h>
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

static hashx_ctx *make_ctx(int compiled) {
#ifdef TOR_HASHX
    hashx_ctx *ctx = hashx_alloc(compiled ? HASHX_TYPE_COMPILED : HASHX_TYPE_INTERPRETED);
    if (!ctx) return NULL;
    static const char seed[] = "This is a test";
    if (hashx_make(ctx, seed, sizeof(seed)) != HASHX_OK) {
        hashx_free(ctx);
        return NULL;
    }
#else
    hashx_ctx *ctx = hashx_alloc(compiled ? HASHX_COMPILED : HASHX_INTERPRETED);
    if (ctx == HASHX_NOTSUPP || !ctx) return NULL;
    static const char seed[] = "This is a test";
    if (!hashx_make(ctx, seed, sizeof(seed))) {
        hashx_free(ctx);
        return NULL;
    }
#endif
    return ctx;
}

static int exec_hash(hashx_ctx *ctx, uint64_t input, uint8_t out[HASHX_SIZE]) {
#ifdef TOR_HASHX
    return hashx_exec(ctx, input, out) == HASHX_OK;
#else
    hashx_exec(ctx, input, out);
    return 1;
#endif
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    int compiled = strcmp(argv[1], "compiled") == 0;
    uint64_t limit = strtoull(argv[2], NULL, 10);
    hashx_ctx *ctx = make_ctx(compiled);
    if (!ctx) return 3;

    uint8_t check[HASHX_SIZE];
    if (!exec_hash(ctx, 123456, check)) return 4;
    for (unsigned i = 0; i < HASHX_SIZE; ++i) printf("%02x", check[i]);
    printf("\n");

    uint8_t out[HASHX_SIZE];
    volatile uint8_t sink = 0;
    uint64_t start = ns_now();
    for (uint64_t input = 0; input < limit; ++input) {
        if (!exec_hash(ctx, input, out)) return 5;
        sink ^= out[0];
    }
    uint64_t elapsed = ns_now() - start;
    printf("%llu %u\n", (unsigned long long)elapsed, (unsigned)sink);
    hashx_free(ctx);
    return 0;
}
"""


def _build_c_hashx(name: str, src: Path, *, tor: bool) -> Path:
    hashx_src = src / "src/ext/equix/hashx" if tor else src
    build = CACHE / "build" / name
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", str(hashx_src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "-j", "--target", "hashx_static"],
        check=True,
        stdout=sys.stderr,
    )
    adapter = build / "hashx-fixed.c"
    adapter.write_text(HASHX_ADAPTER)
    binary = build / "hashx-fixed"
    args = [os.environ.get("CC", "cc"), "-O3", "-DHASHX_STATIC"]
    if tor:
        args.append("-DTOR_HASHX=1")
    args.extend([
        "-I", str(hashx_src / "include"),
        str(adapter), str(build / "libhashx.a"), "-o", str(binary),
    ])
    subprocess.run(args, check=True, stdout=sys.stderr)
    return binary


ARTI_HASHX_ADAPTER = r'''#![forbid(unsafe_code)]
use hashx::{HashXBuilder, RuntimeOption};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let runtime = match args.next().as_deref() {
        Some("compiled") => RuntimeOption::CompileOnly,
        Some("interpreted") => RuntimeOption::InterpretOnly,
        _ => std::process::exit(2),
    };
    let limit: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| std::process::exit(2));
    let hash = HashXBuilder::new().runtime(runtime).build(b"This is a test\0")
        .unwrap_or_else(|_| std::process::exit(3));
    let check = hash.hash_to_bytes(123456);
    for byte in check { print!("{byte:02x}"); }
    println!();
    let started = Instant::now();
    let mut sink = 0_u8;
    for input in 0..limit {
        sink ^= hash.hash_to_bytes(input)[0];
    }
    println!("{} {sink}", started.elapsed().as_nanos());
}
'''


def build_arti_hashx(src: Path) -> Path:
    build = CACHE / "build" / "arti_hashx"
    crate_dir = build / "adapter"
    (crate_dir / "src").mkdir(parents=True, exist_ok=True)
    (crate_dir / "Cargo.toml").write_text(
        '[package]\nname="arti-hashx-fixed"\nversion="0.0.0"\nedition="2024"\n'
        f'[dependencies]\nhashx={{path={json.dumps(str(src / "crates/hashx"))}}}\n\n[workspace]\n'
    )
    (crate_dir / "src/main.rs").write_text(ARTI_HASHX_ADAPTER)
    target = build / "target"
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(crate_dir / "Cargo.toml"), "--target-dir", str(target)],
        check=True,
        stdout=sys.stderr,
    )
    return target / "release" / "arti-hashx-fixed"


def _check_and_bench_hashx(binary: Path, runtime: str, limit: int) -> int:
    result = run([str(binary), runtime, str(limit)], capture=True)
    if result.returncode:
        raise RuntimeError(result.stderr or f"{binary}: exit {result.returncode}")
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError(f"unexpected HashX adapter output: {result.stdout!r}")
    if lines[0] != HASHX_VECTOR_HEX:
        raise RuntimeError(
            f"HashX correctness mismatch for {binary} {runtime}: {lines[0]} != {HASHX_VECTOR_HEX}"
        )
    return int(lines[1].split()[0])


def bench_hashx(samples: int, limit: int) -> None:
    implementations = []

    tevador_src = ensure_checkout("hashx")
    implementations.append((
        f"tevador/hashx@{SOURCES['hashx'][1][:12]}",
        _build_c_hashx("hashx", tevador_src, tor=False),
        ("compiled", "interpreted"),
        "c",
    ))

    tor_src = ensure_checkout("tor_hashx")
    implementations.append((
        f"tor/src/ext/equix@{SOURCES['tor_hashx'][1][:12]}",
        _build_c_hashx("tor_hashx", tor_src, tor=True),
        ("compiled", "interpreted"),
        "c-tor",
    ))

    arti_src = ensure_checkout("arti_hashx")
    implementations.append((
        f"tor/arti-hashx@{SOURCES['arti_hashx'][1][:12]}",
        build_arti_hashx(arti_src),
        ("compiled", "interpreted"),
        "rust-arti",
    ))

    # Every timing row is gated by the same upstream test vector. Also run the
    # vector once even when samples=0, so this command doubles as a compatibility check.
    for baseline, binary, runtimes, family in implementations:
        for runtime in runtimes:
            _check_and_bench_hashx(binary, runtime, 1)
            for sample in range(samples):
                ns = _check_and_bench_hashx(binary, runtime, limit)
                emit(
                    baseline=baseline,
                    protocol="anubis",
                    variant="hashx-v1-primitive",
                    implementation=f"{family}-{runtime}",
                    accelerator="cpu",
                    sample=sample,
                    attempted=limit,
                    ns=ns,
                )


HASHWX_RATE = re.compile(r"^([0-9.]+) hashes/sec\.$", re.MULTILINE)


def build_hashwx(src: Path) -> Path:
    build = CACHE / "build" / "hashwx"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "-j", "--target", "hashwx-tests", "hashwx-bench"],
        check=True,
        stdout=sys.stderr,
    )
    tests = run([str(build / "hashwx-tests")], capture=True)
    if tests.returncode:
        raise RuntimeError(tests.stderr or tests.stdout)
    return build / "hashwx-bench"


def bench_hashwx(samples: int, limit: int, nonces_per_seed: int) -> None:
    if limit <= 0 or nonces_per_seed <= 0 or limit % nonces_per_seed:
        raise ValueError("HashWX limit must be a positive multiple of nonces-per-seed")
    src = ensure_checkout("hashwx")
    binary = build_hashwx(src)
    rev = SOURCES["hashwx"][1][:12]
    seeds = limit // nonces_per_seed
    for runtime, extra in [("compiled-jit", []), ("interpreted", ["--interpret"])]:
        for sample in range(samples):
            result = run(
                [
                    str(binary),
                    "--start", str(1000 + sample * seeds),
                    "--seeds", str(seeds),
                    "--nonces", str(nonces_per_seed),
                    "--threads", "1",
                    "--diff", str(2_147_483_647),
                    *extra,
                ],
                capture=True,
            )
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            match = HASHWX_RATE.search(result.stdout)
            if not match:
                raise RuntimeError(f"unable to parse hashwx-bench output:\n{result.stdout}")
            rate = float(match.group(1))
            ns = max(1, round(limit * 1e9 / rate))
            emit(
                baseline=f"tevador/hashwx@{rev}",
                protocol="capjs",
                variant="hashwx-format2",
                implementation=runtime,
                accelerator="cpu",
                sample=sample,
                attempted=limit,
                ns=ns,
                comparison_scope="primitive-ceiling",
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
    uint8_t out[32], salt[8] = {0};
    int check_rc = argon2_hash(2, 19 * 1024, 1,
        pwd, sizeof(pwd), salt, sizeof(salt),
        out, sizeof(out), NULL, 0, Argon2_id, ARGON2_VERSION_13);
    if (check_rc != ARGON2_OK) return 1;
    for (unsigned i = 0; i < sizeof(out); ++i) printf("%02x", out[i]);
    printf("\n");
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


def x86_64_microarch_level() -> str | None:
    if os.uname().machine not in {"x86_64", "amd64"}:
        return None
    flags = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags.update(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    if "avx512f" in flags:
        return "x86-64-v4"
    if "avx2" in flags:
        return "x86-64-v3"
    return "x86-64-v2"


def build_argon2(src: Path) -> Path:
    build = CACHE / "build" / "argon2"
    build.mkdir(parents=True, exist_ok=True)
    opttarget = x86_64_microarch_level()
    make_env = os.environ.copy()
    make_args = ["make", "-C", str(src)]
    if opttarget is not None:
        make_args.append(f"OPTTARGET={opttarget}")
    # The source selection (ref.c vs opt.c) is decided at make parse time, so
    # clean before rebuilding with a possibly different deterministic target.
    subprocess.run(make_args + ["clean"], check=True, stdout=sys.stderr, env=make_env)
    subprocess.run(make_args + ["-j", "libargon2.a"], check=True, stdout=sys.stderr, env=make_env)
    adapter = build / "argon2-fixed.c"
    adapter.write_text(ARGON2_ADAPTER)
    binary = build / "argon2-fixed"
    cc_args = [os.environ.get("CC", "cc"), "-O3"]
    if opttarget is not None:
        cc_args.append(f"-march={opttarget}")
    cc_args.extend([
        "-I", str(src / "include"), str(adapter), str(src / "libargon2.a"),
        "-pthread", "-o", str(binary),
    ])
    subprocess.run(cc_args, check=True, stdout=sys.stderr)
    return binary


def bench_argon2(samples: int, limit: int) -> None:
    src = ensure_checkout("argon2")
    binary = build_argon2(src)
    rev = SOURCES["argon2"][1][:12]
    for sample in range(samples):
        result = run([str(binary), str(sample), str(limit)], capture=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        lines = result.stdout.strip().splitlines()
        if len(lines) != 2:
            raise RuntimeError(f"unexpected Argon2 adapter output: {result.stdout!r}")
        pwd = bytearray([0x42] * 48)
        pwd[:4] = sample.to_bytes(4, sys.byteorder)
        pwd[47] = 0x80
        expected = hash_secret_raw(
            secret=bytes(pwd),
            salt=(0).to_bytes(8, sys.byteorder),
            time_cost=2,
            memory_cost=19 * 1024,
            parallelism=1,
            hash_len=32,
            type=Argon2Type.ID,
            version=19,
        ).hex()
        if lines[0] != expected:
            raise RuntimeError(f"Argon2 correctness mismatch: {lines[0]} != {expected}")
        ns = int(lines[1].split()[0])
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
        for _ in range(OPENCL_WARMUP_RUNS):
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
                comparison_scope="protocol-kernel-reference",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--hashx-limit", type=int, default=1_048_576)
    parser.add_argument("--hashwx-limit", type=int, default=1_048_576)
    parser.add_argument("--hashwx-nonces-per-seed", type=int, default=65_536)
    parser.add_argument("--argon2-limit", type=int, default=64)
    parser.add_argument("--opencl-limit", type=int, default=67_108_864)
    parser.add_argument(
        "--cpu-family",
        action="append",
        choices=("hashx", "hashwx", "argon2"),
        help="Run only selected CPU baseline families; repeat to select multiple.",
    )
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        cpu_families = set(args.cpu_family or ("hashx", "hashwx", "argon2"))
        if "hashx" in cpu_families:
            bench_hashx(args.samples, args.hashx_limit)
        if "hashwx" in cpu_families:
            bench_hashwx(args.samples, args.hashwx_limit, args.hashwx_nonces_per_seed)
        if "argon2" in cpu_families:
            bench_argon2(args.samples, args.argon2_limit)
    if args.gpu:
        bench_anubis_offload(args.samples, args.opencl_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
