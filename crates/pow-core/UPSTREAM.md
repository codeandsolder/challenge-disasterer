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

The rejected portable-SIMD / CPU Gray-order experiment was not imported: it
regressed Renoir cycle counts in prior measurements. The independent
challenge-disasterer GPU Gray-order experiment remains a benchmark candidate
until it is tested on both Renoir and SM75 hardware.
