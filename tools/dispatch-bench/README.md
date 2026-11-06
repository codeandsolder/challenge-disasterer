# Controlled CPU dispatch and code-generation benchmark

Standalone experimental tool. The historical preexisting-baseline reference is
unchanged. The fixtures deliberately follow the current half of that reference
as of application revision 0797d4a. Shared compiled fixture code with that
control would itself affect the whole-program optimization under investigation.

Build three separate *release executables* from the same source and pinned
solver, saving a copy of each executable before changing its feature set:

    cargo +nightly-2026-11-03 build --locked --release --manifest-path tools/dispatch-bench/Cargo.toml --no-default-features --features typed
    cargo +nightly-2026-11-03 build --locked --release --manifest-path tools/dispatch-bench/Cargo.toml --no-default-features --features erased
    cargo +nightly-2026-11-03 build --locked --release --manifest-path tools/dispatch-bench/Cargo.toml --features erased

The third command compiles both typed and erased into one executable. All
three use ThinLTO with one codegen unit. No build changes production kernels.

Run each executable with --protocol go-away --samples 5 --limit 1120000,
alternating order at the same CPU affinity. Typed is labeled auto, erased
auto-erased. The erased case uses a boxed trait and an out-of-line dispatch
boundary to avoid trivial call-site devirtualization.

Only compare the standalone experimental variants as *codegen conditions*.
A mixed-mode binary may optimize even its typed hot path differently; don't
misinterpret an across-binary difference as direct virtual-call overhead.
Inspect the optimized hot-path instructions and benchmark other protocols
before changing production solver code. Run standalone locked unit tests
for each feature configuration to verify CLI behavior.
