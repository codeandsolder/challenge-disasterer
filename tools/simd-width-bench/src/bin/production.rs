use std::time::Instant;

use challenge_solvers::{
    message::SingleBlockMessage,
    solver::{SOLVE_TYPE_MASK, Solver},
};
use serde_json::json;

const DEFAULT_LIMIT: u64 = 8_960_000;
const DEFAULT_SAMPLES: u32 = 7;

fn run<S>(name: &str, sample: u32, limit: u64)
where
    S: Solver<Output = [u32; 8]> + From<SingleBlockMessage>,
{
    let salt = [b'a'; 64];
    let mut solver =
        S::from(SingleBlockMessage::new(&salt, 0).expect("single-block benchmark fixture"));
    solver.set_limit(limit);
    let started = Instant::now();
    let result = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
    let elapsed = started.elapsed();
    assert!(result.is_none());
    assert_eq!(solver.attempted_nonces(), limit);
    println!(
        "{}",
        json!({
            "implementation": name,
            "sample": sample,
            "attempted_work": limit,
            "solve_ns": elapsed.as_nanos(),
            "work_per_second": limit as f64 / elapsed.as_secs_f64(),
        })
    );
}

fn main() {
    let limit = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("limit must be an integer"))
        .unwrap_or(DEFAULT_LIMIT);
    let samples = std::env::args()
        .nth(2)
        .map(|value| value.parse().expect("samples must be an integer"))
        .unwrap_or(DEFAULT_SAMPLES);

    for sample in 0..samples {
        if std::arch::is_x86_feature_detected!("avx2") {
            run::<challenge_solvers::solver::x86::Avx2SingleBlockSolver>(
                "avx2-portable",
                sample,
                limit,
            );
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            run::<challenge_solvers::solver::x86::Avx512SingleBlockSolver>(
                "avx512-portable",
                sample,
                limit,
            );
        }
    }
}
