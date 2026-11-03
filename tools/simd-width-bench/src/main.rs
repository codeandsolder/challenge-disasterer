use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use challenge_solvers::{
    message::{GoAwayMessage, SingleBlockMessage},
    solver::{SOLVE_TYPE_MASK, Solver, width_bench},
};
use serde_json::json;

const DEFAULT_LIMIT: u64 = 8_960_000;
const DEFAULT_SAMPLES: u32 = 7;

fn work_per_second(attempted: u64, elapsed: Duration) -> Option<u128> {
    let nanos = elapsed.as_nanos();
    (u128::from(attempted) * 1_000_000_000).checked_div(nanos)
}

fn emit(name: &str, family: &str, sample: u32, attempted: u64, elapsed: Duration) {
    println!(
        "{}",
        json!({
            "implementation": name,
            "family": family,
            "sample": sample,
            "attempted_work": attempted,
            "solve_ns": elapsed.as_nanos(),
            "work_per_second": work_per_second(attempted, elapsed),
        })
    );
}

fn single_block_fixture() -> Result<SingleBlockMessage> {
    let salt = [b'a'; 64];
    SingleBlockMessage::new(&salt, 0).context("single-block benchmark fixture must fit one block")
}

fn portable_single<const LANES: usize>(sample: u32, limit: u64) -> Result<()> {
    let mut msg = single_block_fixture()?;
    let mut attempted = 0;
    let started = Instant::now();
    let result = width_bench::solve_single_block::<LANES, SOLVE_TYPE_MASK>(
        &mut msg,
        &mut attempted,
        limit,
        0,
        u64::MAX,
    );
    let elapsed = started.elapsed();
    ensure!(result.is_none(), "benchmark unexpectedly found a solution");
    ensure!(
        attempted == limit,
        "solver attempted {attempted} candidates instead of {limit}"
    );
    emit(
        &format!("portable-{LANES}"),
        "single-block",
        sample,
        attempted,
        elapsed,
    );
    Ok(())
}

fn auto_single(sample: u32, limit: u64) -> Result<()> {
    let mut msg = single_block_fixture()?;
    let mut attempted = 0;
    let started = Instant::now();
    let result = width_bench::solve_single_block_auto::<SOLVE_TYPE_MASK>(
        &mut msg,
        &mut attempted,
        limit,
        0,
        u64::MAX,
    );
    let elapsed = started.elapsed();
    ensure!(result.is_none(), "benchmark unexpectedly found a solution");
    ensure!(
        attempted == limit,
        "solver attempted {attempted} candidates instead of {limit}"
    );
    emit("auto-generic", "single-block", sample, attempted, elapsed);
    Ok(())
}

fn auto_goaway(sample: u32, limit: u64) -> Result<()> {
    let msg = GoAwayMessage::new_bytes(&[0x42; 32], 0);
    let mut attempted = 0;
    let started = Instant::now();
    let result =
        width_bench::solve_goaway_auto::<SOLVE_TYPE_MASK>(&msg, &mut attempted, limit, 0, u64::MAX);
    let elapsed = started.elapsed();
    ensure!(result.is_none(), "benchmark unexpectedly found a solution");
    ensure!(
        attempted == limit,
        "solver attempted {attempted} candidates instead of {limit}"
    );
    emit("auto-generic", "go-away", sample, attempted, elapsed);
    Ok(())
}

fn portable_goaway<const LANES: usize>(sample: u32, limit: u64) -> Result<()> {
    let msg = GoAwayMessage::new_bytes(&[0x42; 32], 0);
    let mut attempted = 0;
    let started = Instant::now();
    let result = width_bench::solve_goaway::<LANES, SOLVE_TYPE_MASK>(
        &msg,
        &mut attempted,
        limit,
        0,
        u64::MAX,
    );
    let elapsed = started.elapsed();
    ensure!(result.is_none(), "benchmark unexpectedly found a solution");
    ensure!(
        attempted == limit,
        "solver attempted {attempted} candidates instead of {limit}"
    );
    emit(
        &format!("portable-{LANES}"),
        "go-away",
        sample,
        attempted,
        elapsed,
    );
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_single(sample: u32, limit: u64) -> Result<()> {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return Ok(());
    }
    let mut solver =
        challenge_solvers::solver::x86::Avx2SingleBlockSolver::from(single_block_fixture()?);
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
    emit("avx2-wrapper", "single-block", sample, limit, elapsed);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn avx512_single(sample: u32, limit: u64) -> Result<()> {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return Ok(());
    }
    let mut solver =
        challenge_solvers::solver::x86::Avx512SingleBlockSolver::from(single_block_fixture()?);
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
    emit("avx512-wrapper", "single-block", sample, limit, elapsed);
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
    ensure!(limit % 16 == 0, "limit must be divisible by 16");

    for sample in 0..samples {
        // Rotate order to reduce thermal/frequency bias.
        match sample % 3 {
            0 => {
                portable_single::<4>(sample, limit)?;
                portable_single::<8>(sample, limit)?;
                portable_single::<16>(sample, limit)?;
            }
            1 => {
                portable_single::<16>(sample, limit)?;
                portable_single::<4>(sample, limit)?;
                portable_single::<8>(sample, limit)?;
            }
            _ => {
                portable_single::<8>(sample, limit)?;
                portable_single::<16>(sample, limit)?;
                portable_single::<4>(sample, limit)?;
            }
        }
        auto_single(sample, limit)?;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        avx2_single(sample, limit)?;
        #[cfg(target_arch = "x86_64")]
        avx512_single(sample, limit)?;

        auto_goaway(sample, limit)?;
        portable_goaway::<4>(sample, limit)?;
        portable_goaway::<8>(sample, limit)?;
        portable_goaway::<16>(sample, limit)?;
    }

    Ok(())
}
