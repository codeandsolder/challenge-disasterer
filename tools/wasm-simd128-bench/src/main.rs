#[cfg(target_arch = "wasm32")]
mod wasm_bench {
    use std::time::Instant;

    use pow_core::{
        message::{
            CerberusBinaryMessage, CerberusDecimalMessage, CerberusMessage, DecimalMessage,
            DoubleBlockMessage, GoAwayMessage, SingleBlockMessage,
        },
        solver::{SOLVE_TYPE_MASK, Solver, simd128},
    };
    use serde_json::json;

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

    pub fn run() {
        let limit = std::env::args()
            .nth(1)
            .map(|value| {
                value
                    .parse::<u64>()
                    .expect("work limit must be a positive integer")
            })
            .unwrap_or(1_120_000);
        assert!(limit > 0, "work limit must be positive");

        let salt = [b'a'; 64];
        measure(
            "sha256-single-block",
            simd128::SingleBlockSolver::from(
                SingleBlockMessage::new(&salt, 0).expect("single-block benchmark shape"),
            ),
            limit,
        );

        let double = (0..salt.len())
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .expect("double-block benchmark shape");
        measure(
            "sha256-double-block",
            simd128::DoubleBlockSolver::from(double),
            limit,
        );

        measure(
            "sha256-decimal",
            simd128::DecimalSolver::from(
                DecimalMessage::new(&[b'd'; 96], 0).expect("decimal benchmark shape"),
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
                    .expect("Cerberus decimal benchmark shape"),
            )),
            limit,
        );
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bench::run();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("challenge-disasterer-wasm-simd128-bench is a wasm32-only benchmark tool");
}
