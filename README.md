# challenge-disasterer

A tiny local service for routing proof-of-work / anti-bot challenge solvers to the fastest available backend.

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

`GET /stats` returns the available solvers, startup calibration, GPU work ahead, CPU capacity, calibrated expected time-to-solution for a new request at every Anubis difficulty, rolling median solve time by difficulty, and aggregate counters. It is designed to be cached periodically by a task router rather than queried on every task.

## Scheduling

- GPU is the preferred backend and owns one persistent native `wgpu` device/pipeline.
- The Anubis GPU kernel uses a SHA-256 midstate, fixed-width octal-as-decimal nonces, fully unrolled rounds, 64×128 workgroups, and cooperative cancellation every 16 hashes.
- Queue accounting is in **expected hashes**, not number of jobs.
- If a request waits more than `--cpu-spillover-ms` (100 ms by default) *before the GPU starts it*, it is cancelled from the GPU queue and moved to CPU when a CPU slot is available.
- GPU failures and challenge shapes unsupported by the GPU fall back to CPU.
- CPU concurrency is bounded by `--cpu-fallback-workers`.
- Both GPU and CPU are calibrated with fixed-work, effectively-unsolvable d16 samples at startup.

The current CPU backend uses upstream `pow-buster` runtime dispatch. The locally benchmarked SHA-NI octal patch is intentionally not hidden in this repository as an untracked machine-local dependency; it should land as a proper dependency/fork before being used here.

## Run

```sh
cargo run --release -- --listen 127.0.0.1:17417
```

Rust 1.99 is the development baseline.
