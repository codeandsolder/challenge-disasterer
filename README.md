# challenge-disasterer

A small local service for routing proof-of-work / anti-bot challenge solvers to an available backend.

Current solvers: **Anubis fast/slow SHA-256**, **Anubis WASM SHA-256**, **Anubis Argon2id**, **Anubis HashX**, **Cerberus**, **go-away js-pow-sha256**, **Cap.js**, **mCaptcha**, **haphash** (upstream HAProxy and current Debian Varnish prefix forms), and **ALTCHA v2** (SHA-256 and PBKDF2/SHA-256). The former `pow-buster-imp` solver engine now lives in-tree as `crates/pow-core`; there is no runtime or Git dependency on the old repository. Classic protocols expose concrete CPU implementations (`safe`, AVX2, SHA-NI where applicable, AVX-512 when present, plus `auto`) and one persistent generalized WGPU context. Anubis keeps its specialized SHA GPU path, while HashX keeps upstream CPU JIT plus every advertised portable/native GPU implementation. `/solve` returns protocol-native answer JSON so callers do not need a second conversion layer.

## API

```json
POST /solve
{
  "solver": "anubis",
  "algorithm": "fast",
  "batch_size_hint": 1,
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

`GET /stats` returns available solvers plus machine-readable `solver_capabilities` (algorithms, difficulty domain, whether an endpoint is itself a router, and concrete implementations), startup calibration, queued/active GPU work, CPU capacity, rolling timing data, and aggregate counters. This capability description is also the discovery contract used by the routing calibrator and by optional external solver services.

`batch_size_hint` defaults to 1 and tells calibrated routes how much concurrent work the caller is about to submit. Explicit solver names continue to force their family. `anubis` is the automatic fast/slow SHA route; `anubis-sha256` and `anubis-argon2id` implement the current Anubis WASM challenge methods on CPU; `anubis-hashx` is the explicit HashX GPU family; `anubis-hashx-jit` forces the upstream CPU JIT; and `anubis-hashx-auto` is the calibrated CPU/GPU/external HashX route.

## Backend policy

- The `gpu` Cargo feature is enabled by default. Build with `--no-default-features` for a CPU-only binary.
- A normal build attempts one persistent native `wgpu` device/pipeline at startup. `--gpu=false` disables it at runtime.
- Without a routing-calibration file, the historical SHA policy remains: a compatible operational GPU is preferred and CPU is the fallback. HashX auto similarly prefers its GPU family when available and otherwise uses the upstream CPU JIT.
- With `--routing-calibration <report.json>`, the automatic `anubis` and `anubis-hashx-auto` endpoints choose the fastest compatible **concrete** calibrated route for the requested algorithm, difficulty, and nearest calibrated batch size. Local GPU rows are usable only on the adapter they were measured on; CPU and configured external providers remain eligible.
- Explicit endpoints (`anubis-cpu`, `anubis-gpu`, `anubis-sha256`, `anubis-argon2id`, `anubis-hashx`, `anubis-hashx-jit`) retain force semantics and bypass the cross-family calibrated router.
- CPU concurrency is bounded by `--cpu-workers`. Argon2id search fanout can be overridden with `--argon2-workers`; by default it is capped at the smaller of the configured CPU budget, available logical CPUs, and detected physical cores so SMT does not oversubscribe the memory-hard search.
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

The repository pins its development toolchain in `rust-toolchain.toml`; Anubis assets are built with Rust 1.99 in the live-compatibility workflow.

## Reproducible hardware matrix

`tools/run-hardware-matrix.py` is the canonical cross-hardware validation/benchmark entry point. It writes immutable run directories containing host/toolchain/GPU metadata, JSONL direct-solver measurements, authoritative verifier logs, and a manifest of every gate. `tools/summarize-hardware-matrix.py` compares multiple run directories by identical protocol/backend keys while keeping distinct exact throughput work budgets separate, and also extracts the paired HashX/Anubis benchmarks.

The same run also executes `tools/preexisting-baseline`, pinned to the original `eternal-flame-AD/pow-buster@d359adb` ancestor. The isolated baseline workspace uses the same ThinLTO / single-codegen-unit release profile and 64-bit comparison mode as the main solver matrix on both old and current dependencies, then runs alternating old/new samples with identical message bytes and an aligned 1,120,000-candidate budget for every solver family the old implementation had in common with the current core (Anubis decimal and WASM SHA-256 shape, haphash, mCaptcha, Cap.js, go-away, and both Cerberus formats). The separate aligned budget is intentional: historical Cerberus decimal executes work in coarse batches and overshoots arbitrary limits such as 1,048,576; the harness fails rather than calling unequal attempted work “exact”. The summarizer reports matched `auto` old/new ratios and reports a fastest-current CPU ratio only when that current row used exactly the same attempted-work budget as the historical baseline; unlike work budgets are never normalized into a pretend apples-to-apples result. Protocols added later, such as ALTCHA and HashX, are compared against their authoritative upstream/browser references instead of inventing a nonexistent old baseline.

On multi-GPU hosts the runner enumerates every non-CPU Vulkan adapter, runs CPU rows once, and explicitly forces each GPU adapter for the generalized WGPU matrix, specialized Anubis SHA kernel, and live WGPU protocol gates. Adapter identity is part of every GPU result key, so an iGPU and dGPU can never be silently pooled into one number.

The direct matrix deliberately separates two questions that must not be conflated:

- **correctness:** normal protocol targets; every returned proof is checked against the protocol condition and the live harness subsequently submits it to the authoritative implementation;
- **throughput:** SHA-family solvers use a fixed candidate budget while preserving the protocol comparison path: MASK uses an exact 64-bit mask match, GT uses `> 2^64-2`, and LT uses `< 1`. Each has about a 2^-64 accidental-hit probability per candidate, so every implementation performs exactly the requested work without benchmarking a different comparison path or giving the optimizer a provably impossible predicate. Any accidental hit or attempted-work mismatch fails the run. Argon2id uses its own exact `--argon2-limit` candidate budget because one memory-hard Argon2 candidate is not comparable to one SHA candidate.
  When GPU benchmarking is enabled the budget must be a multiple of 8192, the specialized Anubis kernel's natural work quantum; the runner defaults to 1,048,576 candidates and rejects misaligned budgets rather than rounding them silently.

HashX uses the live Anubis harness's paired benchmark rather than a duplicate synthetic verifier. CPU JIT, automatic routing, and every GPU implementation solve the **same freshly issued challenge** in rotating order; the harness requires identical nonce/hash across implementations before reporting timing. This covers `wgsl-specialized`, the adapter-specific persistent Vulkan backend, and `cuda-sm75-sass` when available.

Current Anubis also ships WASM `sha256` and `argon2id` challenge algorithms. challenge-disasterer exposes both as native solver routes: `anubis-sha256` reuses the in-tree binary SHA-256 backends (`safe`, `auto`, AVX2, AVX-512 where available, and generalized WGPU), while `anubis-argon2id` uses Argon2 0.6.0 with parameters and nonce-byte rules matching upstream. The real-Anubis harness submits both proof types back to the current upstream verifier, and the stock-browser benchmark remains as an independent reference.

Example:

```sh
uv run tools/run-hardware-matrix.py   --output target/hardware-matrix/$(hostname)-$(date +%Y%m%d-%H%M%S)   --samples 5 --throughput-limit 1048576 --gpu --live

uv run tools/summarize-hardware-matrix.py   target/hardware-matrix/sf314-42-*   target/hardware-matrix/cachyasus-*
```

Use `--no-cpu --no-live` for direct GPU-only collection on a host whose CPU is busy, or `--no-gpu --no-live` for direct CPU-only collection. The `--cpu`/`--gpu` selectors control direct solver rows; live verifier runs are a separate validation phase and may exercise CPU routing. At least one direct resource class must remain enabled. Retained runs reject a dirty Git tree by default (`--no-require-clean` is for throwaway development smoke only), and the run metadata records every workload/validation parameter alongside the exact commit and hardware identity.

## Authoritative protocol compatibility tests

Protocol compatibility is tested against the implementation that actually issues or verifies each proof rather than against a duplicate verifier in this repository:

- **Anubis:** `ci/live-anubis-e2e.sh` starts the current upstream Anubis server, obtains a live challenge, submits the disasterer proof to Anubis, and requires the resulting authorization cookie to reach a protected upstream.
- **Cerberus:** `ci/live-cerberus-e2e.sh` builds the official Cerberus Caddy plugin, obtains a real challenge page, submits the proof to its real `/answer` endpoint, and requires its approval cookie to pass the protected route. The hardware runner executes current upstream plus `v0.4.6` and `v0.4.5` for every available backend, covering the binary-format boundary and the last decimal-format release against their own authoritative verifiers.
- **go-away:** `ci/live-goaway-e2e.sh` runs the current upstream `js-pow-sha256` WASM runtime and uses its own `MakeChallenge` and `VerifyChallenge` ABI.
- **Cap.js:** `ci/live-capjs-e2e.sh` uses current `capjs-core` `generateChallenge` and `validateChallenge`.
- **ALTCHA v2:** `ci/live-altcha-e2e.sh` uses current `altcha-lib` `createChallenge` and `verifySolution` for both supported algorithms.
- **mCaptcha:** `ci/live-mcaptcha-e2e.sh` validates the native work response with the `mcaptcha_pow_sha256` crate used by mCaptcha itself.
- **haphash:** `ci/live-haphash-e2e.sh` clones current `dgl/haphash`, serves its actual `challenge.html`, and submits proofs to the real HAProxy SHA-256 ACL verifier. Hardware runs additionally exercise Debian's live Varnish-derived deployment, whose current wire shape is `SHA256(pow_challenge_cookie_value + ";" + decimal_nonce)` at difficulty 2; the harness deliberately fails if that page shape changes rather than inheriting the obsolete historical client prefix.

Every non-Anubis harness also submits a deliberately corrupted proof and requires the authoritative verifier to reject it. `.github/workflows/live-protocols.yml` runs both `auto` and `safe` against every classic authoritative protocol; Cerberus additionally crosses both implementations with current upstream, `v0.4.6`, and `v0.4.5`. Hardware-specific AVX2, SHA-NI, and WGPU lanes remain in the real-hardware matrix instead of assuming hosted-runner features. The CI jobs remain independent so one expensive upstream build does not serialize the rest of the matrix.

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
`hashx:anubis-hashx:16`. An optional fourth case field asserts the concrete
implementation selected by normal routing, for example
`hashx:anubis-hashx:18:cuda-sm75-sass`. The same assertion can be applied to the
generated HashX fuzz/benchmark matrix with `--expect-hashx-gpu-implementation`,
so a benchmark cannot silently fall back to a different GPU path.

With GPU testing enabled, the harness asks the solver binary to enumerate every
usable Vulkan `shaderInt64` adapter, starts a separately pinned solver instance
for each one, then discovers that adapter's `hashx_gpu_implementations` from
`/stats`. By default it explicitly forces and validates **every concrete HashX
implementation that can run on every available GPU**. Forced tests do not fall
back: a broken native/persistent/specialized path fails as itself. During
benchmarking, CPU JIT, automatic routing, and every advertised GPU implementation
solve the exact same fresh HashX challenges in a rotating execution order. The
harness requires identical minimum nonce/hash and prints `ROUTING` comparisons
between the automatic route and the fastest concrete GPU path. The default paired
crossover sweep is d1, d8, d16, d18, and d20..d24. Set
`--exercise-all-gpu-adapters=false` or
`--exercise-all-hashx-implementations=false` only when intentionally narrowing a
local run.

For a from-source local Anubis build, its current asset pipeline additionally
requires Node 22+, `brotli`, `zstd`, and the Rust `wasm32-unknown-unknown`
target. The GitHub workflow installs these explicitly; the local shell wrapper
fails early with a clear message when the external compressors or Node version
are missing.


### Routing calibration

The same live-Anubis harness can produce the routing table consumed by the
service. Calibration discovers every solver/capability advertised by `/stats`,
expands every supported difficulty whose geometric expected work is below the
configured ceiling, and separately force-tests every concrete implementation.
For each parameter point it measures concurrent batches of **1, 10, and 100**
fresh independently issued challenges by default. Each workload is repeated
three times by default to keep a lucky one-solution PoW from deciding routing.

Every measured solution still completes the full live path: Anubis issues it,
the candidate solver solves it, and Anubis accepts the returned proof before the
sample counts. Batch makespan times only the concurrent solver requests; proof
validation runs serially immediately afterward. This is both the routing-relevant
measurement and necessary for current Anubis, whose WASM verifier is non-reentrant
under overlapping validations. Router endpoints are benchmarked for diagnostics
but are excluded from winner selection, so a pre-existing router cannot certify
itself.

```sh
tools/anubis-e2e/target/release/challenge-disasterer-anubis-e2e \
  --anubis-bin target/anubis-live-bin \
  --solver-bin target/release/challenge-disasterer \
  --solver-gpu \
  --calibrate-routing \
  --calibration-batch-sizes 1,10,100 \
  --calibration-rounds 3 \
  --calibration-max-expected-work 16777216 \
  --calibration-output routing.json
```

`--calibration-max-expected-work` is the practical-domain guard. With the
default value it exhaustively covers current fast/slow difficulties through d6
and binary-work algorithms such as HashX through d24; raise it when calibrating
larger real workloads rather than making a default calibration accidentally run
100-way d64 work.

Additional solver services can participate without being compiled into
challenge-disasterer. They only need the same small HTTP discovery/solve
contract (`GET /stats`, `POST /solve`) and should advertise concrete
`solver_capabilities`. Add one or more with:

```sh
  --external-solver remote=http://10.0.0.2:17417
```

The report records the provider URL and includes external concrete backends in
winner selection. challenge-disasterer can then consume it directly:

```sh
target/release/challenge-disasterer \
  --routing-calibration routing.json \
  --listen 127.0.0.1:17417
```

External answers are not trusted merely because calibration selected them:
automatic routes validate them locally before returning them. HashX is checked
against the upstream HashX implementation and SHA answers against the local
Anubis verifier. Concrete routes are kept in calibrated fastest-first order; if
an external provider or local backend fails, the router tries the next calibrated
candidate before falling back to the legacy policy.

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

`--benchmark` reports the normal end-to-end metrics below and, when a GPU is
available, paired `PAIRED_HASHX` / `ROUTING` results for every advertised HashX
implementation. A `ROUTING status=SUBOPTIMAL` line means automatic routing
selected a different GPU implementation and was more than 25% slower than the paired winner. If auto
selected the same implementation, timing noise cannot mislabel the routing
choice. The output also reports `best_overall`, including CPU JIT, so low-work
cases where GPU dispatch overhead dominates remain visible. Routing performance
is diagnostic rather than a noisy CI failure.

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

The HashX GPU backend has one shared validated `GpuProgram` representation (SipHash key plus the fixed 512 typed HashX operations). Portable Vulkan, challenge-specialized WGSL, and architecture-native backends all consume that same program. Native NVIDIA selection resolves the wgpu-selected CUDA device once, inspects its compute capability, and dispatches to an architecture encoder; SM75/Turing is currently implemented and later SM generations can be added as sibling backends without duplicating HashX parsing, validation, device matching, or solve orchestration.

On SM75/Turing the native backend emits the program directly as SASS into an embedded fixed-layout cubin shell and loads it with the CUDA Driver API. Code generation is tens of microseconds and the driver module path is warmed during backend initialization, so measured fresh-challenge latency beats the persistent Vulkan interpreter even at difficulty 1; the NVIDIA u32 Vulkan interpreter remains the runtime fallback if CUDA/native specialization is unavailable. The CUDA dependency is dynamically loaded and does **not** require a CUDA toolkit at build time. AMD/RADV continues to use the working `vulkan-extmul-persistent` passthrough-SPIR-V path through difficulty 20, with `wgsl-specialized` above that range.

HashX GPU solve responses retain `"backend":"gpu"` for API compatibility and add an `implementation` field identifying the actual path (`cuda-sm75-sass`, `vulkan-extmul-persistent`, `vulkan-nvidia-u32-persistent`, or `wgsl-specialized`). The verifier and benchmark can require this value explicitly, which is used for cross-architecture validation rather than inferring the backend from adapter logs.

The SM75 emitter is deliberately narrow rather than a general assembler: typed Rust encoders cover only the instruction forms HashX needs, then patch a 128 KiB `.text.kern` section in a prebuilt cubin shell. The shell contains dead padding after the real `EXIT`; an 8,192-program generation fuzz reached 4,068 used instruction words versus 8,192 available. Turing instruction-encoding details are derived from the MIT-licensed TuringAs project; see `THIRD_PARTY_NOTICES.md`.

### HashX CPU backend

`anubis-hashx-jit` uses HashX 0.8's upstream native compiler directly. For each fresh challenge HashX emits its 512-op program with `dynasmrt` into an executable mapping; the eight virtual registers live in real x86-64 registers (`r8`..`r15`) and operations become direct `imul`/`mul`/`xor`/`sub`/`lea` plus the generated branch. Worker threads share the immutable compiled program and search disjoint nonce ranges.

On the Ryzen 7 4700U this beat the experimental auto-vectorized interpreter decisively: in a 64-sample live Anubis d18 comparison with two CPU workers, upstream JIT measured **28.01 ms/accepted answer and 14.73 MH/s**, versus **37.60 ms and 9.29 MH/s** for the auto-vectorized interpreter. A separate fixed-work upstream benchmark scaled to **25.42 MH/s at 8 threads** on the concurrently loaded laptop. The custom SIMD interpreter has therefore been retired from active code; its experiments remain in git history.
