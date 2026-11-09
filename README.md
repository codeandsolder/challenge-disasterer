# challenge-disasterer

A small local service for routing proof-of-work / anti-bot challenge solvers to an available backend.

The project scope is deliberately consumer-facing: client-work gates in front of otherwise cheap public content are in scope; resource-proportional admission controls for intrinsically expensive operations (for example, model inference) are not a target.

Current solvers: **Anubis fast/slow SHA-256**, **Anubis WASM SHA-256**, **Anubis Argon2id**, **Anubis HashX**, **Cerberus**, **go-away js-pow-sha256**, **Cap.js** (legacy SHA-256 plus format-2 SHA-256, RSW, HashWX, and mixed PoW challenges), **mCaptcha**, **haphash** (upstream HAProxy and current Debian Varnish prefix forms), **Linear.pub SHA-1**, and **ALTCHA** (still-deployable v1 SHA-1/SHA-256/SHA-512 plus current v2 SHA-256/SHA-384/SHA-512, PBKDF2/SHA-256/SHA-384/SHA-512, Argon2id, and Scrypt). Non-PoW browser/instrumentation challenges are intentionally out of scope. The former `pow-buster-imp` solver engine, protocol adapters and specialized Anubis Argon2 implementation now live in the [reusable challenge-solvers workspace](https://github.com/codeandsolder/challenge-solvers), consumed by this service as pinned Git packages. The service no longer vendors these solver sources. Portable SIMD is required on the pinned nightly toolchain; the optional `gpu` feature controls GPU support independently. Explicit scalar implementations remain available for hardware fallback and performance comparisons. Classic protocols expose only the concrete implementations that apply to each algorithm rather than pretending heterogeneous families share one backend set. Anubis keeps its specialized SHA GPU path, while HashX uses the audited `hashwx-rs` native JIT plus every advertised portable/native GPU implementation. `/solve` returns protocol-native answer JSON so callers do not need a second conversion layer.

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

`GET /stats` returns available solvers plus machine-readable `solver_capabilities` (algorithms, difficulty domain, whether an endpoint is itself a router, selectable implementations, and where relevant the counter encoding/domain), startup calibration, queued/active GPU work, CPU capacity, rolling timing data, and aggregate counters. The `general.median_solve_ms_by_difficulty` values are medians of backend execution time (excluding queue wait and HTTP overhead) over the latest 256 finite, nonnegative successful Anubis timings for each difficulty; an even sample count uses the mean of its two middle values. This capability description is also the discovery contract used by the routing calibrator and by optional external solver services. ALTCHA v2 advertises `uint32` and `string` counter modes separately. SHA-256 WGPU supports arbitrary-length binary salt and nonce using a once-per-challenge CPU-computed SHA-256 chaining state, with at most two counter-dependent final blocks on GPU. PBKDF2-SHA-256 WGPU retains its specialized 16-byte salt/nonce HMAC path and handles other lengths with general HMAC: nonce||counter passwords longer than 64 bytes are SHA-256 hashed before HMAC, and salt||INT(1) is padded across arbitrary SHA-256 blocks. Both input formats use CPU verification of the winning digest and respect device storage limits. PBKDF2 can also derive keys longer than SHA-256's 32-byte output block: GPU checks any full-length hexadecimal prefix across consecutive PBKDF2 blocks only after earlier block prefixes match, then independently derives and returns the full requested key on CPU for the winner. Both support independently verified hex prefixes up to the 32-byte SHA-256 output and the complete JavaScript-safe string-counter range. On GPUs advertising native `SHADER_INT64`, SHA-384 and SHA-512 use an independently verified 80-round u64 compression kernel, and PBKDF2-HMAC-SHA-384/512 use the same kernel with full-length HMAC key preprocessing, arbitrary-length salt/nonce and multiple PBKDF2 output blocks. All four support both ALTCHA v2 counter encodings, arbitrary valid derived key lengths within GPU device bounds, and exact hexadecimal prefixes including odd nibbles. Their GPU winners are rederived and verified using the CPU RustCrypto implementation. GPUs without `SHADER_INT64` still run the existing SHA-256 kernels; SHA-384/512 automatically fall back to CPU. Memory-hard KDFs remain on CPU by default. Upstream `altcha-lib` remains the independent E2E/benchmark oracle and is not linked into the production service.

`batch_size_hint` defaults to 1 and tells calibrated routes how much concurrent work the caller is about to submit. Explicit solver names continue to force their family. `anubis` is the automatic fast/slow SHA route; `anubis-sha256` and `anubis-argon2id` implement the current Anubis WASM methods (SHA-256 prefers GPU by default; Argon2id uses CPU); `anubis-hashx` is the explicit HashX GPU family; `anubis-hashx-jit` forces the `hashwx-rs` native CPU JIT; and `anubis-hashx-auto` is the calibrated CPU/GPU/external HashX route.

## Backend policy

- The `gpu` Cargo feature is enabled by default. `--no-default-features` builds an optimized portable-SIMD CPU-only service using the pinned nightly toolchain; it does not disable CPU acceleration.
- A normal build attempts one persistent native `wgpu` device/pipeline at startup. `--gpu=false` disables it at runtime.
- **Default GPU hash placement:** `--gpu-policy prefer-gpu` (default) sends supported SHA-256 and Cerberus BLAKE3 work to an operational GPU **even while another request is using it**; jobs wait for the shared GPU slot rather than consuming CPU cores. The choice is about overall machine resource use, not the lowest single-request latency. Classic Anubis fast/slow, Anubis WASM SHA-256, ALTCHA SHA-256/SHA-384/SHA-512 and their PBKDF2 variants, GoAway, haphash, mCaptcha, SHA-only Cap.js, and Cerberus BLAKE3 are eligible. ALTCHA eligibility uses its existing parser; unsupported shapes, algorithms or device failures fall back to CPU. Memory-hard KDFs are not automatically GPU-routed.
- **Configurable GPU hash placement:** `--gpu-policy idle-only` uses GPU if no other job from this service owns its shared execution slot, otherwise CPU; `--gpu-policy cpu-only` keeps automatic hash work on CPU, while explicit GPU requests remain available. `--gpu=false` disables GPU backends altogether. A shared one-slot semaphore serializes this service's specialized Anubis, general WGPU and HashX jobs. It manages **our job queues**, not unrelated applications' GPU utilization. The selected GPU policy appears in `GET /stats` under `config.gpu_policy`.
- **Calibration:** `--routing-calibration` ranks local CPU/remote fallbacks and estimates latency, but it does not override GPU hash placement. Explicit `anubis-cpu`, `anubis-gpu`, `implementation=wgpu` and CPU-specific implementations retain their force semantics.
- CPU concurrency is bounded by `--cpu-workers`. `--argon2-workers` is the upper bound for the Anubis and ALTCHA independent Argon2id nonce search. ALTCHA defaults to adaptive fanout based on the memory cost, passes, hexadecimal prefix work and live spare CPU permits; non-Argon2 protocols stay single-worker. Use `--altcha-worker-policy fixed` to force the configured maximum for reproducible benchmark comparisons. Each active ALTCHA worker holds one CPU permit and a memory reservation rounded to whole MiB. On Linux, the default total memory budget is half the available host/cgroup RAM at startup, or override it with `--argon2-memory-budget-mib`. A single Argon2 arena exceeding the entire budget produces an immediate error rather than waiting forever. Other platforms without detectable memory can set an explicit budget. The solve response provides `nonce_workers` and the stats endpoint reports policy and available memory permits.
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

The repository requires the pinned nightly toolchain in `rust-toolchain.toml`. The independent upstream Anubis assets still use their own Rust 1.99 build in the live-compatibility workflow.

## Reproducible hardware matrix

`tools/ci/run-hardware-matrix.py` is the canonical cross-hardware validation/benchmark entry point. It writes immutable run directories containing host/toolchain/GPU metadata, JSONL direct-solver measurements, authoritative verifier logs, and a manifest of every gate. `tools/ci/summarize-hardware-matrix.py` compares multiple run directories by identical protocol/backend keys while keeping distinct exact throughput work budgets separate, and also extracts the paired HashX/Anubis benchmarks.

The same run also executes `tools/preexisting-baseline`, pinned to the original `eternal-flame-AD/pow-buster@d359adb` ancestor. The isolated baseline workspace uses the same ThinLTO / single-codegen-unit release profile and 64-bit comparison mode as the main solver matrix on both old and current dependencies, then runs alternating old/new samples with identical message bytes and an aligned 1,120,000-candidate budget for every solver family the old implementation had in common with the current core (Anubis decimal and WASM SHA-256 shape, haphash, mCaptcha, Cap.js, go-away, and both Cerberus formats). The separate aligned budget is intentional: historical Cerberus decimal executes work in coarse batches and overshoots arbitrary limits such as 1,048,576; the harness fails rather than calling unequal attempted work “exact”. The summarizer reports matched `auto` old/new ratios and reports a fastest-current CPU ratio only when that current row used exactly the same attempted-work budget as the historical baseline; unlike work budgets are never normalized into a pretend apples-to-apples result. Protocols added later, such as ALTCHA and HashX, are compared against their authoritative upstream/browser references instead of inventing a nonexistent old baseline.

The historical paired runner also accepts repeatable `--protocol` selections (`anubis`, `haphash`, `mcaptcha`, `capjs`, `go-away`, `cerberus`) to isolate a comparison without heating the CPU with unrelated protocols. Omit the selector for the complete historical overlap. Its legacy Cerberus work-quantum alignment restriction applies only when Cerberus is selected. For example, `cargo +nightly-2026-11-03 run --locked --release --manifest-path tools/preexisting-baseline/Cargo.toml -- --protocol go-away --samples 5 --limit 1120000` measures just the original and current go-away solvers with alternating order. The independent baseline's lockfile and legacy API are verified in the release-only CI tier, avoiding another historical-harness regression without adding another ordinary PR gate.


On multi-GPU hosts the runner enumerates every non-CPU Vulkan adapter, runs CPU rows once, and explicitly forces each GPU adapter for the generalized WGPU matrix, specialized Anubis SHA kernel, and live WGPU protocol gates. Adapter identity is part of every GPU result key, so an iGPU and dGPU can never be silently pooled into one number.

For a targeted investigation, the direct Rust solver matrix accepts repeatable `--protocol` selectors instead of running every protocol. Omit them for the complete matrix; use e.g. `--protocol go-away` for only go-away, or `--protocol anubis --protocol haphash` for both families. Selection is enforced for both CPU and GPU rows; Linear.pub has CPU rows only. An isolated, exact-work go-away run can use:

```sh
cargo +nightly-2026-11-03 run --locked --release \
  --manifest-path tools/solver-matrix/Cargo.toml -- \
  --protocol go-away --cpu true --gpu false \
  --mode throughput --samples 5 --limit 1120000
```

The direct matrix deliberately separates two questions that must not be conflated:

- **correctness:** normal protocol targets; every returned proof is checked against the protocol condition and the live harness subsequently submits it to the authoritative implementation;
- **throughput:** SHA-family solvers use a fixed candidate budget while preserving the protocol comparison path: MASK uses an exact 64-bit mask match, GT uses `> 2^64-2`, and LT uses `< 1`. Each has about a 2^-64 accidental-hit probability per candidate, so every implementation performs exactly the requested work without benchmarking a different comparison path or giving the optimizer a provably impossible predicate. Any accidental hit or attempted-work mismatch fails the run. Argon2id uses its own exact `--argon2-limit` candidate budget because one memory-hard Argon2 candidate is not comparable to one SHA candidate.
  When GPU benchmarking is enabled the budget must be a multiple of 8192, the specialized Anubis kernel's natural work quantum; the runner defaults to 1,048,576 candidates and rejects misaligned budgets rather than rounding them silently.

HashX uses the live Anubis harness's paired benchmark rather than a duplicate synthetic verifier. CPU JIT, automatic routing, and every GPU implementation solve the **same freshly issued challenge** in rotating order; the harness requires identical nonce/hash across implementations before reporting timing. This covers `wgsl-specialized`, the adapter-specific persistent Vulkan backend, and `cuda-sm75-sass` when available.

Cap.js HashWX is benchmarked at the protocol's production reuse window of 65,536 nonces per generated function. The direct matrix uses the production `hashwx-rs` native JIT and charges each per-seed program generation/compilation to the measured work. `tools/ci/external-baselines.py` separately pins tevador HashWX and measures its native compiled JIT and interpreter on identical work budgets; these are labeled primitive ceilings rather than independent algorithm implementations.

Cap.js format-2 SHA-256 has its own CPU and WGPU rows rather than being hidden behind the legacy format-1 SHA primitive. RSW is measured at Cap's production default 2048-bit modulus / 75,000 sequential squarings: the direct matrix measures the shipping Rust `BigUint` path, while the external matrix runs the exact same modulus/input through the pinned official Cap worker (`BigInt`) and GNU GMP. The worker is the protocol reference; GMP is an independent native ceiling.

Linear.pub has a deterministic `seed || decimal_nonce` SHA-1 row matching the deployed cookie gate. An OpenSSL 48-byte one-block SHA-1 microbenchmark is retained alongside it as a primitive ceiling only; it intentionally does not pretend that raw SHA-1 includes decimal formatting or protocol overhead.

mCaptcha now also benchmarks the pinned current official `mCaptcha/mcaptcha_pow_sha256` crate on the exact salt + bincode-string + decimal-nonce protocol layout, alongside the historical pow-buster comparison.

ALTCHA has deterministic exact-work rows for every still-supported v1 digest (SHA-1/SHA-256/SHA-512), every current v2 algorithm (SHA-256/SHA-384/SHA-512, PBKDF2/SHA-256/SHA-384/SHA-512, Argon2id, and scrypt), in both `uint32` and `string` counter modes. The benchmark constructs a challenge whose prefix corresponds to a known counter, requires challenge-disasterer and the current official `altcha-lib` solver to find that exact counter, and verifies our answer with the official verifier. This produces apples-to-apples service and current-upstream reference rows instead of inferring work from expected difficulty. With multiple ALTCHA Argon2id nonce workers, attempted_nonces reports
completed candidate hashes and may exceed the returned counter plus one due to
speculative work. The benchmark uses actual work for throughput and still
reports the minimal matching counter. Workers are charged against the same CPU
permit budget as Anubis Argon2id, and each holds one full memoryCost KiB arena.

An opt-in memory-hard benchmark profile is available by setting ALTCHA_BENCH_PRODUCTION_ARGON2=1 before running the ALTCHA benchmark (optionally set ALTCHA_BENCH_ONLY_ARGON2=1 to isolate this algorithm): it adds 64/128 MiB, three-pass, p=1 cases in both counter encodings, and one 64 MiB, three-pass, p=2 case. Each added case deliberately places its solution at counter 7 (eight mandatory lowest-counter checks), although parallel nonce workers may perform additional speculative Argon2 hashes, which are counted independently in the output. The benchmark reports actual fanout using nonce_workers; use alternating comparable host runs for performance decisions.

Current Anubis also ships WASM `sha256` and `argon2id` challenge algorithms. challenge-disasterer exposes both as native solver routes: `anubis-sha256` reuses the external `challenge-solvers` binary SHA-256 backends (`safe`, `auto`, AVX2, AVX-512 where available, and generalized WGPU), while `anubis-argon2id` uses our specialized portable-SIMD Argon2id engine with parameters and nonce-byte rules matching upstream. The real-Anubis harness submits both proof types back to the current upstream verifier, and the stock-browser benchmark remains as an independent reference.

Example:

```sh
uv run tools/ci/run-hardware-matrix.py   --output target/hardware-matrix/$(hostname)-$(date +%Y%m%d-%H%M%S)   --samples 5 --throughput-limit 1048576 --gpu --live

uv run tools/ci/summarize-hardware-matrix.py   target/hardware-matrix/sf314-42-*   target/hardware-matrix/cachyasus-*
```

Use `--no-cpu --no-live` for direct GPU-only collection on a host whose CPU is busy, or `--no-gpu --no-live` for direct CPU-only collection. The `--cpu`/`--gpu` selectors control direct solver rows; live verifier runs are a separate validation phase and may exercise CPU routing. At least one direct resource class must remain enabled. Retained runs reject a dirty Git tree by default (`--no-require-clean` is for throwaway development smoke only), and the run metadata records every workload/validation parameter alongside the exact commit and hardware identity.

## Authoritative protocol compatibility tests

Protocol compatibility is tested against the implementation that actually issues or verifies each proof rather than against a duplicate verifier in this repository:

Full live-protocol compatibility runs daily, on published releases, or by
manual **Live protocol compatibility** workflow dispatch against a selected
branch. Routine pull requests do not spawn 25 upstream verifier runners.
The selected exact revision is built once with canonical sccache and that
release binary is distributed to all independent protocol checks.

The seven non-Anubis script-driven live-protocol harnesses require `CHALLENGE_SOLVER_BINARY`
to name an already-built challenge-disasterer executable; they do not compile it.
Hosted CI builds one CPU-only release executable with canonical sccache and
distributes that exact-commit artifact to all 25 independent live-protocol
matrix cases. For local runs, explicitly supply the path to an existing
CPU/GPU build as appropriate. Relative paths are resolved before the
harness changes its working directory.

- **Anubis:** `ci/live-anubis-e2e.sh` starts the current upstream Anubis server, obtains a live challenge, submits the disasterer proof to Anubis, and requires the resulting authorization cookie to reach a protected upstream.
- **Cerberus:** `ci/live-cerberus-e2e.sh` builds the official Cerberus Caddy plugin, obtains a real challenge page, submits the proof to its real `/answer` endpoint, and requires its approval cookie to pass the protected route. The hardware runner executes current upstream plus `v0.4.9` through `v0.4.5` for every available backend, covering the recent compatibility range and the decimal/binary-format boundary against each release's own authoritative verifier.
- **go-away:** `ci/live-goaway-e2e.sh` runs the current upstream `js-pow-sha256` WASM runtime and uses its own `MakeChallenge` and `VerifyChallenge` ABI. The verifier's meaningful leading-zero domain is the full SHA-256 width, difficulty `0..=256`, and every CPU/SIMD/SHA-NI/WGPU backend preserves those full-digest semantics. The ordinary `1..=64` range keeps the cheaper first-64-bit compare internally; larger values use the same accelerated hashing kernels with a full-digest predicate. Difficulties above 256 are unsatisfiable and rejected.
- **Cap.js:** `ci/live-capjs-e2e.sh` uses current `capjs-core` `generateChallenge` and `validateChallenge` for legacy SHA-256 and format-2 SHA-256, RSW, HashWX, and mixed PoW challenges. HashWX production uses the audited `hashwx-rs` native JIT; pinned tevador native JIT and interpreter builds remain independent external benchmark references.
- **ALTCHA:** `ci/live-altcha-e2e.sh` uses current `altcha-lib` generation and verification across all current v2 KDFs plus still-supported v1 SHA-1/SHA-256/SHA-512. V2 decimal-string counter mode is searched through the full JavaScript safe-integer domain instead of inheriting the official Rust crate's `u32` solver bound.
- **mCaptcha:** `ci/live-mcaptcha-e2e.sh` validates the native work response with the `mcaptcha_pow_sha256` crate used by mCaptcha itself.
- **Linear.pub:** `ci/live-linear-e2e.sh` fetches the live site-wide SHA-1 gate, solves the issued 40-hex seed through `/solve`, requires a deliberately invalid cookie to remain blocked, and requires the returned `yX3` cookie to reach the public storefront. The CPU solver uses RustCrypto SHA-1; on GPUs with baseline WGPU support the accelerated solver performs all 80 SHA-1 rounds for the complete valid 40-ASCII-hex-seed plus decimal-u32 nonce domain, verifying every GPU solution independently on CPU. Automatic GPU-first placement remains configurable with the global GPU policy; no shader-int64 feature is required.
- **haphash:** `ci/live-haphash-e2e.sh` clones current `dgl/haphash`, serves its actual `challenge.html`, and submits proofs to the real HAProxy SHA-256 ACL verifier. HAProxy verifies leading-zero hex nibbles across the full digest, and every CPU/SIMD/SHA-NI/WGPU backend covers the verifier's `0..=64`-nibble domain; `1..=16` retains the cheaper first-64-bit compare internally and larger values use a full-digest predicate. Hosted CI and the hardware matrix both run normal difficulty 4 plus verifier-valid difficulty 0, with the latter deliberately exercising the full-digest backend branch. The stock browser loop is only operational for positive difficulty (its zero-difficulty loop never returns even though HAProxy accepts every proof), so the stock deployable client range is `1..=64`. Values above 64 are unsatisfiable. Hardware runs additionally exercise Debian's live Varnish-derived deployment, whose current wire shape is `SHA256(pow_challenge_cookie_value + ";" + decimal_nonce)` at difficulty 2; the harness deliberately fails if that page shape changes rather than inheriting the obsolete historical client prefix.

Every non-Anubis harness also submits a deliberately corrupted proof and requires the authoritative verifier to reject it. The live-protocol workflow runs both auto and safe where a protocol exposes both; Linear.pub exposes the RustCrypto auto implementation and its native WGPU accelerator. Cerberus additionally crosses both implementations with current upstream, v0.4.6, and v0.4.5. Hardware-specific AVX2, SHA-NI, and WGPU lanes remain in the real-hardware matrix instead of assuming hosted-runner features. The CI jobs remain independent so one expensive upstream build does not serialize the rest of the matrix.

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
with the same audited `hashwx-rs` program/verifier used by the local solver, and
SHA answers against the local Anubis verifier. Concrete routes are ranked by estimated batch makespan from their measured
concurrency points; if an external provider or local backend fails, the router tries the
next compatible calibrated candidate (including other measured batch sizes)
before falling back to the legacy policy.

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

The SM75 emitter is deliberately narrow rather than a general assembler: typed Rust encoders cover only the instruction forms HashX needs, then patch a 128 KiB `.text.kern` section in a prebuilt cubin shell. The shell contains dead padding after the real `EXIT`; an 8,192-program generation fuzz reached 4,068 used instruction words versus 8,192 available. Turing instruction-encoding details are derived from the MIT-licensed TuringAs project; see the [third-party notices in the owning `challenge-hashx` crate](https://github.com/codeandsolder/challenge-solvers/blob/main/crates/challenge-hashx/THIRD_PARTY_NOTICES.md).

### HashX CPU backend

`anubis-hashx-jit` uses the native HashX compiler in `hashwx-rs`. Each fresh challenge is expanded once into an immutable `HashXProgram` containing the exact 512-op instruction stream and register key; the CPU JIT and accelerator backends consume that same authoritative program. On x86-64, `dynasmrt` emits the program into an executable mapping with the eight virtual registers kept in real registers and the VM operations lowered to direct integer instructions. Worker threads share the immutable compiled program and search disjoint nonce ranges. No backend reconstructs HashX IR from debug text.

On the Ryzen 7 4700U this beat the experimental auto-vectorized interpreter decisively: in a 64-sample live Anubis d18 comparison with two CPU workers, upstream JIT measured **28.01 ms/accepted answer and 14.73 MH/s**, versus **37.60 ms and 9.29 MH/s** for the auto-vectorized interpreter. A separate fixed-work upstream benchmark scaled to **25.42 MH/s at 8 threads** on the concurrently loaded laptop. The custom SIMD interpreter has therefore been retired from active code; its experiments remain in git history.
