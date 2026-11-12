#![cfg_attr(feature = "gpu", recursion_limit = "256")]

mod cpu;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(test)]
mod tests;

use std::{
    num::{NonZeroU8, NonZeroU32, NonZeroU64},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result};
use argon2::Argon2;
use challenge_solvers::{
    build_mcaptcha_prefix, compute_mask_cerberus, compute_mask_goaway, compute_mask_zero_nibbles,
    compute_target_mcaptcha,
    message::{
        AltchaMessage, BinaryMessage, CapJSEmitter, CerberusBinaryMessage, CerberusDecimalMessage,
        CerberusMessage, DecimalMessage, GoAwayMessage,
    },
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LT, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use clap::{Parser, ValueEnum};
use hashwx::HashWx;
use num_bigint::BigUint;
use serde::Serialize;
use serde_json::{Value, json};
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha512};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Correctness,
    Throughput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Protocol {
    Anubis,
    Haphash,
    Mcaptcha,
    Capjs,
    Linear,
    GoAway,
    Cerberus,
    Altcha,
}

#[derive(Parser, Debug)]
struct Args {
    /// Fresh deterministic challenge instances per protocol/backend.
    #[arg(long, default_value_t = 5)]
    samples: u32,
    /// Hard candidate ceiling per solve.
    #[arg(long, default_value_t = 50_000_000)]
    limit: u64,
    /// Exact candidate ceiling for the memory-hard Anubis Argon2id rows.
    #[arg(long, default_value_t = 64)]
    argon2_limit: u64,
    /// Include CPU implementations in the matrix.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cpu: bool,
    /// Include the generalized WGPU backend selected by `WGPU_ADAPTER_NAME/default` selection.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    gpu: bool,
    /// Print usable GPU adapter names as JSON and exit.
    #[arg(long, default_value_t = false)]
    list_gpu_adapters: bool,
    /// Matrix phase. Throughput preserves each comparison type with a ~2^-64 hit condition and must exhaust exactly `limit` candidates.
    #[arg(long, value_enum, default_value_t = Mode::Correctness)]
    mode: Mode,
    /// Git commit reported in JSONL metadata.
    #[arg(long, default_value = "unknown")]
    commit: String,
    /// Select protocol families to execute (repeat to select multiple); default is all.
    #[arg(long, value_enum)]
    protocol: Vec<Protocol>,
}

impl Args {
    fn includes(&self, candidate: Protocol) -> bool {
        self.protocol.is_empty() || self.protocol.contains(&candidate)
    }

    fn has_gpu_work(&self) -> bool {
        self.protocol.is_empty() || self.protocol.iter().any(|p| *p != Protocol::Linear)
    }
}

#[derive(Serialize)]
struct Record {
    schema: u32,
    phase: &'static str,
    timestamp_unix_ms: u128,
    host: String,
    cpu_model: String,
    commit: String,
    protocol: &'static str,
    variant: &'static str,
    implementation: String,
    accelerator: &'static str,
    sample: u32,
    parameters: Value,
    attempted_work: u64,
    solve_ns: u128,
    work_per_second: f64,
    found: bool,
    nonce: Option<u64>,
    target_condition_ok: bool,
    gpu_adapter: Option<String>,
}

trait DynPow {
    fn set_limit_dyn(&mut self, limit: u64);
    fn attempted(&self) -> u64;
    fn solve_dyn_local(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

impl<T> DynPow for T
where
    T: Solver<Output = [u32; 8]> + SolverDyn<Output = [u32; 8]>,
{
    fn set_limit_dyn(&mut self, limit: u64) {
        self.set_limit(limit);
    }
    fn attempted(&self) -> u64 {
        self.attempted_nonces()
    }
    fn solve_dyn_local(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])> {
        self.solve_dyn(target, ty, mask)
    }
}

struct Meta {
    host: String,
    cpu_model: String,
    commit: String,
}

struct Case<'a> {
    protocol: &'static str,
    variant: &'static str,
    implementation: &'a str,
    accelerator: &'static str,
    sample: u32,
    parameters: Value,
    target: u64,
    mask: u64,
    ty: u8,
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn read_trimmed(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name\t: ").map(str::to_owned))
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn condition_ok(ty: u8, target: u64, mask: u64, hash: [u32; 8]) -> bool {
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    match ty {
        challenge_solvers::solver::SOLVE_TYPE_GT => value > target,
        challenge_solvers::solver::SOLVE_TYPE_LT => value < target,
        challenge_solvers::solver::SOLVE_TYPE_MASK => value & mask == target & mask,
        _ => false,
    }
}

const fn phase_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Correctness => "correctness",
        Mode::Throughput => "throughput",
    }
}

const fn benchmark_condition(mode: Mode, target: u64, mask: u64, ty: u8) -> (u64, u64, u8) {
    match mode {
        Mode::Correctness => (target, mask, ty),
        // Keep the protocol's comparison path while making a hit about 2^-64 likely per
        // candidate. Using GT with u64::MAX is provably impossible and can let LTO erase
        // hash work; changing every case to MASK benchmarks a different hot path.
        Mode::Throughput => match ty {
            SOLVE_TYPE_GT => (u64::MAX - 1, u64::MAX, SOLVE_TYPE_GT),
            SOLVE_TYPE_LT => (1, u64::MAX, SOLVE_TYPE_LT),
            SOLVE_TYPE_MASK => (0, u64::MAX, SOLVE_TYPE_MASK),
            _ => (target, mask, ty),
        },
    }
}

fn emit_cpu(
    meta: &Meta,
    case: Case<'_>,
    mut solver: Box<dyn DynPow>,
    limit: u64,
    mode: Mode,
) -> Result<()> {
    let (target, mask, ty) = benchmark_condition(mode, case.target, case.mask, case.ty);
    solver.set_limit_dyn(limit);
    let started = Instant::now();
    let solution = solver.solve_dyn_local(target, ty, mask);
    let solve_ns = started.elapsed().as_nanos();
    let attempted = solver.attempted();
    let (nonce, ok) = solution.map_or((None, true), |(nonce, hash)| {
        (Some(nonce), condition_ok(ty, target, mask, hash))
    });
    if matches!(mode, Mode::Throughput) {
        anyhow::ensure!(
            solution.is_none(),
            "throughput target unexpectedly found a proof"
        );
        anyhow::ensure!(
            attempted == limit,
            "throughput run attempted {attempted} candidates, expected exactly {limit}"
        );
    }
    emit(&Record {
        schema: 1,
        phase: phase_name(mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: case.protocol,
        variant: case.variant,
        implementation: case.implementation.to_owned(),
        accelerator: case.accelerator,
        sample: case.sample,
        parameters: case.parameters,
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: nonce.is_some(),
        nonce,
        target_condition_ok: ok,
        gpu_adapter: None,
    })
}

fn emit(record: &Record) -> Result<()> {
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark throughput is floating-point telemetry, not an exact integer result"
)]
fn rate(work: u64, ns: u128) -> f64 {
    if ns == 0 {
        0.0
    } else {
        work as f64 * 1e9 / ns as f64
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_gpu_adapters {
        #[cfg(feature = "gpu")]
        {
            println!(
                "{}",
                serde_json::to_string(&challenge_solvers::gpu::GpuContext::adapter_names().await)?
            );
            return Ok(());
        }
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("GPU support is not compiled into this solver-matrix binary");
    }
    let meta = Meta {
        host: read_trimmed("/proc/sys/kernel/hostname"),
        cpu_model: cpu_model(),
        commit: args.commit.clone(),
    };
    if args.cpu {
        cpu::run_cpu_matrix(&meta, &args)?;
    }
    if args.gpu && args.has_gpu_work() {
        #[cfg(feature = "gpu")]
        gpu::run_gpu_matrix(&meta, &args).await?;
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu true requires the solver-matrix gpu feature");
    } else if args.gpu && !args.cpu {
        anyhow::bail!(
            "selected protocols have no GPU benchmarks; enable CPU or select another protocol"
        );
    }
    Ok(())
}
