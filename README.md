# challenge-disasterer

A small local service for routing proof-of-work / anti-bot challenge solvers to an available backend.

Current solvers: **Anubis SHA-256 PoW** (CPU/SHA-NI and GPU) and **Anubis HashX PoW** (upstream native JIT on CPU plus GPU when the `gpu` feature is enabled). The API remains solver-generic so more challenge families can be added without changing callers.

## API

```json
POST /solve
{
  "solver": "anubis",
  "challenge": {
    "data": "<Anubis randomData>",
    "difficulty": 6
  }
}
```

Response:

```json
{
  "solver": "anubis",
  "backend": "gpu",
  "answer": { "nonce": "...", "hash": "..." },
  "queue_ms": 0.1,
  "solve_ms": 53.0,
  "total_ms": 53.1
}
```

`GET /stats` returns available solvers, startup calibration, queued/active GPU work, CPU capacity, calibrated expected time-to-solution for a new request at every Anubis difficulty, rolling median solve time by difficulty, and aggregate counters. It is intended to be cached periodically by a task router rather than queried for every task.

## Backend policy

- The `gpu` Cargo feature is enabled by default. Build with `--no-default-features` for a CPU-only binary.
- A normal build attempts one persistent native `wgpu` device/pipeline at startup. `--gpu=false` disables it at runtime.
- When the GPU is enabled, operational, and the challenge shape is GPU-compatible, requests stay on the GPU even if they queue. There is deliberately no CPU spillover scheduler.
- CPU is used only when GPU support is disabled/not compiled, GPU initialization fails, the challenge shape is not GPU-compatible, or an actual GPU solve fails.
- CPU concurrency is bounded by `--cpu-workers`.
- Both available backends are calibrated at startup. `/stats` bases the new-request ETA on the active backend; GPU ETA includes expected work already queued or executing.

The Anubis GPU kernel uses a SHA-256 midstate, fixed-width octal-as-decimal nonces, fully unrolled rounds, 64×128 workgroups, and cooperative cancellation every 16 hashes.

## Run

```sh
cargo run --release -- --listen 127.0.0.1:17417
```

CPU-only build:

```sh
cargo run --release --no-default-features -- --listen 127.0.0.1:17417
```

The repository currently pins `nightly-2026-10-08`; Anubis assets are built with Rust 1.99 in the live-compatibility workflow.

## Live Anubis compatibility test

`tools/anubis-e2e` is an end-to-end compatibility harness. It does not duplicate
Anubis verification logic. Instead it starts the actual Anubis binary built from
the upstream revision under test, lets that server issue a challenge, sends the
issued challenge through challenge-disasterer's HTTP API, submits the returned
proof to Anubis's real `pass-challenge` endpoint, and only passes after the
resulting authorization cookie reaches a protected test upstream.

The case syntax is `ANUBIS_ALGORITHM:DISASTERER_SOLVER:DIFFICULTY`. Individual
cases can target an exact backend, for example:

```sh
cargo run --release --manifest-path tools/anubis-e2e/Cargo.toml -- \
  --anubis-bin target/anubis-live-bin \
  --solver-bin target/release/challenge-disasterer \
  --case fast:anubis:2
```

`ci/live-anubis-e2e.sh` can clone/build `TecharoHQ/anubis` main and run the same
check locally. `ANUBIS_REF` selects another upstream ref; `ANUBIS_SRC`,
`ANUBIS_BIN`, `ANUBIS_SKIP_BUILD`, and `ANUBIS_UPDATE` allow CI or local tooling
to provide an existing checkout/build.

`.github/workflows/live-anubis.yml` deliberately checks out **current upstream
Anubis main** on every run (and once daily), builds its generated web/WASM
assets, and exercises the solver against it. This is intended to catch upstream
wire-format and validation changes rather than merely proving that our own
local verifier still agrees with itself.

Adding another solver family only needs another case once challenge-disasterer
exposes it; HashX now uses the same harness contract as SHA via cases such as
`hashx:anubis-hashx:16`.

For a from-source local Anubis build, its current asset pipeline additionally
requires Node 22+, `brotli`, `zstd`, and the Rust `wasm32-unknown-unknown`
target. The GitHub workflow installs these explicitly; the local shell wrapper
fails early with a clear message when the external compressors or Node version
are missing.


### Fuzz matrix and benchmark

The same wrapper can exercise a matrix of fresh challenges rather than one fixed
case. `--fuzz` currently covers `fast` and deprecated `slow` through the explicit
CPU and GPU SHA-256 backends, the normal automatic SHA route, and HashX through
the GPU backend. The matrix spans cheap boundary cases through representative
higher difficulties. Every fuzz trial obtains new random challenge data from the
running Anubis HEAD, submits a deliberately corrupted proof that **must** be
rejected, then requires the real proof to reach the protected upstream.

```sh
cargo build --release --features gpu
cargo build --release --manifest-path tools/anubis-e2e/Cargo.toml

tools/anubis-e2e/target/release/challenge-disasterer-anubis-e2e \
  --anubis-bin target/anubis-live-bin \
  --solver-bin target/release/challenge-disasterer \
  --solver-gpu \
  --fuzz --fuzz-repeats 4 \
  --benchmark --benchmark-samples 12
```

`--benchmark` reports two complementary metrics:

- `avg_answer_ms`: mean complete challenge-page -> solve -> Anubis verification ->
  protected-upstream latency per accepted answer.
- `effective_h_s` / `effective_mh_s`: geometric expected work for the issued
  difficulty divided by solver time. This is the useful hash-rate estimate for
  stochastic PoW solves; `mean_solver_ms` keeps backend setup/compile overhead
  visible alongside the end-to-end answer latency.

When `--browser-reference=true` (the default), the benchmark also launches real
headless Chromium against the same Anubis process. This measures the **stock
Anubis HEAD client code** rather than a reimplementation: `fast` uses the shipped
WebCrypto/JS worker, while `hashx`, `sha256`, and `argon2id` use the shipped WASM
workers. Those browser proofs must also be accepted by the same live server.

The HashX GPU backend specializes by adapter. On NVIDIA SM75/Turing it emits the bounded 512-op HashX program directly as SASS into an embedded fixed-layout cubin shell and loads it with the CUDA Driver API. Code generation is tens of microseconds and the driver module path is warmed during backend initialization, so measured fresh-challenge latency beats the persistent Vulkan interpreter even at difficulty 1; the Vulkan u32 interpreter remains the zero-compiler fallback if CUDA/SASS initialization fails. The CUDA dependency is dynamically loaded and does **not** require a CUDA toolkit at build time. Other NVIDIA architectures retain the portable Vulkan paths. AMD/RADV continues to use the persistent passthrough SPIR-V kernel with extended 64-bit multiplies through difficulty 20, with challenge-specialized WGSL above that range.

The SM75 emitter is deliberately narrow rather than a general assembler: typed Rust encoders cover only the instruction forms HashX needs, then patch a 128 KiB `.text.kern` section in a prebuilt cubin shell. The shell contains dead padding after the real `EXIT`; an 8,192-program generation fuzz reached 4,068 used instruction words versus 8,192 available. Turing instruction-encoding details are derived from the MIT-licensed TuringAs project; see `THIRD_PARTY_NOTICES.md`.

### HashX CPU backend

`anubis-hashx-jit` uses HashX 0.8's upstream native compiler directly. For each fresh challenge HashX emits its 512-op program with `dynasmrt` into an executable mapping; the eight virtual registers live in real x86-64 registers (`r8`..`r15`) and operations become direct `imul`/`mul`/`xor`/`sub`/`lea` plus the generated branch. Worker threads share the immutable compiled program and search disjoint nonce ranges.

On the Ryzen 7 4700U this beat the experimental auto-vectorized interpreter decisively: in a 64-sample live Anubis d18 comparison with two CPU workers, upstream JIT measured **28.01 ms/accepted answer and 14.73 MH/s**, versus **37.60 ms and 9.29 MH/s** for the auto-vectorized interpreter. A separate fixed-work upstream benchmark scaled to **25.42 MH/s at 8 threads** on the concurrently loaded laptop. The custom SIMD interpreter has therefore been retired from active code; its experiments remain in git history.
