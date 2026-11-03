#[cfg(target_arch = "wasm32")]
mod wasm_bench {
    use std::time::Instant;

    use challenge_solvers::{
        message::{
            CerberusBinaryMessage, CerberusDecimalMessage, CerberusMessage, DecimalMessage,
            DoubleBlockMessage, GoAwayMessage, SingleBlockMessage,
        },
        solver::{SOLVE_TYPE_MASK, Solver, simd128},
    };
    use serde_json::json;

    #[expect(
        clippy::cast_precision_loss,
        reason = "benchmark throughput is approximate floating-point telemetry"
    )]
    fn measure<S: Solver>(name: &str, mut solver: S, limit: u64) {
        solver.set_limit(limit);
        let started = Instant::now();
        let solution = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
        let elapsed = started.elapsed();
        let attempted = solver.attempted_nonces();
        assert!(
            solution.is_none(),
            "unexpected exact-64-bit benchmark solution in {name}"
        );
        assert_eq!(
            attempted, limit,
            "{name} did not exhaust the fixed-work budget"
        );
        let ns = elapsed.as_nanos();
        let rate = attempted as f64 / elapsed.as_secs_f64();
        println!(
            "{}",
            json!({
                "schema": 1,
                "implementation": "simd128",
                "family": name,
                "attempted_work": attempted,
                "solve_ns": ns,
                "work_per_second": rate,
            })
        );
    }

    pub fn run() -> Result<(), String> {
        let limit = match std::env::args().nth(1) {
            Some(value) => value
                .parse::<u64>()
                .map_err(|error| format!("invalid work limit {value:?}: {error}"))?,
            None => 1_120_000,
        };
        if limit == 0 {
            return Err(String::from("work limit must be positive"));
        }

        let salt = [b'a'; 64];
        let single = SingleBlockMessage::new(&salt, 0)
            .ok_or_else(|| String::from("single-block benchmark shape is invalid"))?;
        measure(
            "sha256-single-block",
            simd128::SingleBlockSolver::from(single),
            limit,
        );

        let double = (0..salt.len())
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .ok_or_else(|| String::from("no double-block benchmark shape found"))?;
        measure(
            "sha256-double-block",
            simd128::DoubleBlockSolver::from(double),
            limit,
        );

        measure(
            "sha256-decimal",
            simd128::DecimalSolver::from(
                DecimalMessage::new(&[b'd'; 96], 0)
                    .ok_or_else(|| String::from("decimal benchmark shape is invalid"))?,
            ),
            limit,
        );

        measure(
            "go-away",
            simd128::GoAwaySolver::from(GoAwayMessage::new_bytes(&[0x42; 32], 0)),
            limit,
        );

        measure(
            "cerberus-binary",
            simd128::CerberusSolver::from(CerberusMessage::Binary(CerberusBinaryMessage::new(
                b"cerberus-binary-benchmark",
                0,
            ))),
            limit,
        );

        measure(
            "cerberus-decimal",
            simd128::CerberusSolver::from(CerberusMessage::Decimal(
                CerberusDecimalMessage::new(b"cerberus-decimal-benchmark|", 0)
                    .ok_or_else(|| String::from("Cerberus decimal benchmark shape is invalid"))?,
            )),
            limit,
        );
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn main() -> Result<(), String> {
    wasm_bench::run()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("challenge-disasterer-wasm-simd128-bench is a wasm32-only benchmark tool");
}
