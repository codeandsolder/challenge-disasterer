use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use challenge_solvers::{
    message::SingleBlockMessage,
    solver::{SOLVE_TYPE_MASK, Solver},
};
use serde_json::json;

const DEFAULT_LIMIT: u64 = 8_960_000;
const DEFAULT_SAMPLES: u32 = 7;

fn work_per_second(attempted: u64, elapsed: Duration) -> Option<u128> {
    let nanos = elapsed.as_nanos();
    (u128::from(attempted) * 1_000_000_000).checked_div(nanos)
}

fn single_block_fixture() -> Result<SingleBlockMessage> {
    let salt = [b'a'; 64];
    SingleBlockMessage::new(&salt, 0).context("single-block benchmark fixture must fit one block")
}

fn run<S>(name: &str, sample: u32, limit: u64) -> Result<()>
where
    S: Solver<Output = [u32; 8]> + From<SingleBlockMessage>,
{
    let mut solver = S::from(single_block_fixture()?);
    solver.set_limit(limit);
    let started = Instant::now();
    let result = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
    let elapsed = started.elapsed();
    ensure!(result.is_none(), "benchmark unexpectedly found a solution");
    ensure!(
        solver.attempted_nonces() == limit,
        "solver attempted {} candidates instead of {limit}",
        solver.attempted_nonces()
    );
    println!(
        "{}",
        json!({
            "implementation": name,
            "sample": sample,
            "attempted_work": limit,
            "solve_ns": elapsed.as_nanos(),
            "work_per_second": work_per_second(limit, elapsed),
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let limit = match args.next() {
        Some(value) => value.parse::<u64>().context("limit must be an integer")?,
        None => DEFAULT_LIMIT,
    };
    let samples = match args.next() {
        Some(value) => value.parse::<u32>().context("samples must be an integer")?,
        None => DEFAULT_SAMPLES,
    };

    for sample in 0..samples {
        if std::arch::is_x86_feature_detected!("avx2") {
            run::<challenge_solvers::solver::x86::Avx2SingleBlockSolver>(
                "avx2-portable",
                sample,
                limit,
            )?;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            run::<challenge_solvers::solver::x86::Avx512SingleBlockSolver>(
                "avx512-portable",
                sample,
                limit,
            )?;
        }
    }

    Ok(())
}
