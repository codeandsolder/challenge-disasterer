# Solver core provenance

`challenge-pow-core` is the in-tree protocol PoW engine for `challenge-disasterer`.

It was imported on 2026-10-10 from `codeandsolder/pow-buster-imp` commit
`fa7ff0b29f6737a75fba49766a81951caae92d62`, which descends from
`eternal-flame-AD/pow-buster`. The imported code remains Apache-2.0; see
`LICENSE` in this directory.

Included: message builders, protocol adapters, SHA-256/BLAKE3 CPU backends,
runtime CPU dispatch, ALTCHA/PBKDF2 support, Cerberus decimal and 0.4.6+
binary formats, and the generalized WebGPU solver backend.

Deliberately excluded: the old standalone HTTP server/client, browser addon,
static UI, CLI, and WASM FFI surfaces. `challenge-disasterer` owns service,
routing, live-protocol verification, benchmarking, and deployment now.

The rejected portable-SIMD / generic CPU Gray-order experiment was not imported:
it regressed Renoir cycle counts in prior measurements. This is distinct from the
SHA-NI reduced-radix/Gray optimization at `a633d96`: that commit is an ancestor of
the imported `fa7ff0b` lineage, and its optimized SHA-NI path plus regression tests
remain in the current core. A later old-branch-vs-unified fixed-work A/B was
essentially identical because both sides already contained that optimization.

The independent challenge-disasterer GPU Gray-order experiment was tested in seven
alternating 64M-candidate pairs on both Renoir and SM75, then integrated: it
improved the specialized Anubis kernel by about 1.7% on Renoir and 30.6% on RTX
2060/SM75.

Post-import reconciliation also includes the one core fix after `fa7ff0b` in
`pow-buster-imp`: the missing scalar BLAKE3 helper import in AVX-512 tests from
`b2b3437`. Later commits in that repository are CI/WASM packaging changes or the
merge that created this in-tree core and do not add solver functionality.

For performance comparison with preexisting work, the independent benchmark
under `tools/preexisting-baseline` pins the original upstream ancestor
`eternal-flame-AD/pow-buster@d359adb3a692ba6be27fc5c1d5efdbbd7439434d`.
It is intentionally a separate Cargo workspace so the historical git dependency
never becomes a runtime or workspace dependency of `challenge-disasterer`.
