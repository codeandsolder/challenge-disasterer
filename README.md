# challenge-disasterer

A small local service for routing proof-of-work / anti-bot challenge solvers to an available backend.

Current solver: **Anubis SHA-256 PoW**. The API is intentionally solver-generic so more challenge families can be added without changing callers.

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

The case syntax is `ANUBIS_ALGORITHM:DISASTERER_SOLVER:DIFFICULTY`. The current
compatibility case is the legacy/current `fast` Anubis SHA-256 protocol:

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

Adding a solver family only needs another case once challenge-disasterer exposes
it. For example, a future HashX backend can be exercised with a case such as
`hashx:anubis-hashx:16` without changing the harness protocol.

For a from-source local Anubis build, its current asset pipeline additionally
requires Node 22+, `brotli`, `zstd`, and the Rust `wasm32-unknown-unknown`
target. The GitHub workflow installs these explicitly; the local shell wrapper
fails early with a clear message when the external compressors or Node version
are missing.
