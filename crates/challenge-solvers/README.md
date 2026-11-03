# challenge-solvers

Feature-configurable proof-of-work solvers used by `challenge-disasterer`.

The crate intentionally separates three layers:

1. shared message, SHA-256, BLAKE3, and portable-SIMD kernels;
2. architecture dispatch and stateful solver wrappers;
3. optional challenge/protocol modules.

The challenge modules are independent Cargo features: `altcha`, `anubis`, `capjs`,
`cerberus`, `goaway`, and `mcaptcha`. `all-challenges` enables all six. Enabling or
disabling a protocol does not create a separate hashing/SIMD implementation: all
protocols reuse the same solver core.

On x86-64, the portable wide backend selects 16 logical `u32` lanes behind an
AVX-512F target-feature boundary, 8 lanes behind AVX2, and then falls through to
SHA-NI or scalar implementations where applicable. The AVX2 and AVX-512 variants
share one state-machine implementation and one set of portable-SIMD kernels; the
architecture-specific code is limited to runtime feature detection and tiny target
feature thunks.

WebAssembly currently keeps its SIMD128 facade while using the same portable kernels
for the supported solver families. Further consolidation should happen by extending
the shared backend contract rather than copying challenge logic.

See `UPSTREAM.md` for provenance and retained-license details.
