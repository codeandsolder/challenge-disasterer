use std::time::Instant;

use pow_core::{
    message::{GoAwayMessage, SingleBlockMessage},
    solver::{SOLVE_TYPE_MASK, Solver, width_bench},
};
use serde_json::json;

fn emit(name: &str, family: &str, sample: u32, attempted: u64, elapsed: std::time::Duration) {
    println!(
        "{}",
        json!({
            "implementation": name,
            "family": family,
            "sample": sample,
            "attempted_work": attempted,
            "solve_ns": elapsed.as_nanos(),
            "work_per_second": attempted as f64 / elapsed.as_secs_f64(),
        })
    );
}

fn portable_single<const LANES: usize>(sample: u32, limit: u64) {
    let salt = [b'a'; 64];
    let mut msg = SingleBlockMessage::new(&salt, 0).expect("single-block benchmark shape");
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
    assert!(result.is_none());
    assert_eq!(attempted, limit);
    emit(
        &format!("portable-{LANES}"),
        "single-block",
        sample,
        attempted,
        elapsed,
    );
}

fn auto_single(sample: u32, limit: u64) {
    let salt = [b'a'; 64];
    let mut msg = SingleBlockMessage::new(&salt, 0).expect("single-block benchmark shape");
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
    assert!(result.is_none());
    assert_eq!(attempted, limit);
    emit("auto-generic", "single-block", sample, attempted, elapsed);
}

fn auto_goaway(sample: u32, limit: u64) {
    let msg = GoAwayMessage::new_bytes(&[0x42; 32], 0);
    let mut attempted = 0;
    let started = Instant::now();
    let result =
        width_bench::solve_goaway_auto::<SOLVE_TYPE_MASK>(&msg, &mut attempted, limit, 0, u64::MAX);
    let elapsed = started.elapsed();
    assert!(result.is_none());
    assert_eq!(attempted, limit);
    emit("auto-generic", "go-away", sample, attempted, elapsed);
}

fn portable_goaway<const LANES: usize>(sample: u32, limit: u64) {
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
    assert!(result.is_none());
    assert_eq!(attempted, limit);
    emit(
        &format!("portable-{LANES}"),
        "go-away",
        sample,
        attempted,
        elapsed,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_single(sample: u32, limit: u64) {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let salt = [b'a'; 64];
    let mut solver = pow_core::solver::avx2::SingleBlockSolver::from(
        SingleBlockMessage::new(&salt, 0).expect("single-block benchmark shape"),
    );
    solver.set_limit(limit);
    let started = Instant::now();
    let result = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
    let elapsed = started.elapsed();
    assert!(result.is_none());
    assert_eq!(solver.attempted_nonces(), limit);
    emit("avx2-wrapper", "single-block", sample, limit, elapsed);
}

#[cfg(target_arch = "x86_64")]
fn avx512_single(sample: u32, limit: u64) {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return;
    }
    let salt = [b'a'; 64];
    let mut solver = pow_core::solver::avx512::SingleBlockSolver::from(
        SingleBlockMessage::new(&salt, 0).expect("single-block benchmark shape"),
    );
    solver.set_limit(limit);
    let started = Instant::now();
    let result = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
    let elapsed = started.elapsed();
    assert!(result.is_none());
    assert_eq!(solver.attempted_nonces(), limit);
    emit("avx512-wrapper", "single-block", sample, limit, elapsed);
}

fn main() {
    let limit = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_960_000u64);
    let samples = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7u32);
    assert_eq!(limit % 16, 0, "limit must be divisible by 16");

    for sample in 0..samples {
        // Rotate order to reduce thermal/frequency bias.
        match sample % 3 {
            0 => {
                portable_single::<4>(sample, limit);
                portable_single::<8>(sample, limit);
                portable_single::<16>(sample, limit);
            }
            1 => {
                portable_single::<16>(sample, limit);
                portable_single::<4>(sample, limit);
                portable_single::<8>(sample, limit);
            }
            _ => {
                portable_single::<8>(sample, limit);
                portable_single::<16>(sample, limit);
                portable_single::<4>(sample, limit);
            }
        }
        auto_single(sample, limit);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        avx2_single(sample, limit);
        #[cfg(target_arch = "x86_64")]
        avx512_single(sample, limit);

        auto_goaway(sample, limit);
        portable_goaway::<4>(sample, limit);
        portable_goaway::<8>(sample, limit);
        portable_goaway::<16>(sample, limit);
    }
}
