# challenge-disasterer

A small local service for routing proof-of-work / anti-bot challenge solvers to an available backend.

Current solvers: **Anubis SHA-256 PoW** (CPU/SHA-NI and GPU) and **Anubis HashX PoW** (auto-vectorized CPU plus GPU when the `gpu` feature is enabled). The API remains solver-generic so more challenge families can be added without changing callers.

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

The HashX GPU integration uses the persistent fixed-block passthrough SPIR-V kernel with extended 64-bit multiplies through difficulty 20, avoiding per-challenge shader compilation on the latency-sensitive range. Higher difficulties currently retain the challenge-specialized fallback. The benchmark includes all backend setup in `mean_solver_ms`, so future native GFX9 specialization can be compared without changing the correctness or reporting contract.

### Auto-vectorized HashX CPU backend

`anubis-hashx-simd` is the explicit CPU HashX backend. It keeps a **256-nonce
structure-of-arrays batch** to amortize the one runtime opcode dispatch over many
hashes, but deliberately does not encode 256 nonces as one SIMD value. The hot
per-op lane loops are ordinary branchless Rust and LLVM chooses the vectorization
factor and interleave for the target CPU.

On the Ryzen 7 4700U (`-C target-cpu=native`) LLVM lowers those loops to AVX2
YMM operations: four `u64` lanes per hardware vector, including `vpmuludq`
expansions for HashX's 64-bit multiplies. This separates the **interpreter batch
size** from the **hardware SIMD width** and avoids the register-spill explosion
seen with explicit `Simd<u64, 16/32>` values.

Fixed-work `perf stat` testing over 32,768 hashes found the batch-size knee at
roughly 256 nonces: 128 = 26.68M cycles, 256 = 25.56M, 512 = 25.85M, and
1024 = 26.88M. The backend spreads disjoint 256-nonce batches across the
configured CPU worker pool and scalar-validates a winning nonce with upstream
HashX before returning it.

The upstream HashX dynasm JIT remains faster per core on AVX2 because it removes
runtime opcode dispatch entirely and Zen 2 lacks AVX-512DQ-style packed 64-bit
integer multiplication. The CPU vector backend stays explicit so callers and CI
can benchmark it independently; it is also available in `--no-default-features`
CPU-only builds.
