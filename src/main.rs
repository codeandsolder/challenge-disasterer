#![recursion_limit = "256"]

mod anubis;
mod anubis_wasm;
mod classic_pow;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod hashwx_sm75_cuda;
mod hashx;
mod hashx_cpu_jit;
#[cfg(feature = "gpu")]
mod hashx_gpu;
#[cfg(feature = "gpu")]
mod hashx_gpu_native;
#[cfg(feature = "gpu")]
mod hashx_sm75;
#[cfg(feature = "gpu")]
mod hashx_sm75_cuda;
mod linear_pow;
mod routing;
mod service;
mod stats;

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;
use service::{AppState, Config};
use tracing::info;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:17417")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 4)]
    cpu_workers: usize,
    /// Workers used by one Argon2id search. Defaults to the smaller of the
    /// configured CPU worker budget and the machine's available physical cores.
    #[arg(long)]
    argon2_workers: Option<usize>,
    #[arg(long, default_value_t = 2_000_000)]
    cpu_calibration_hashes: u64,
    #[cfg(feature = "gpu")]
    #[arg(long, default_value_t = 64_000_000)]
    gpu_calibration_hashes: u64,
    #[arg(long, default_value_t = cfg!(feature = "gpu"), action = clap::ArgAction::Set)]
    gpu: bool,
    #[cfg(feature = "gpu")]
    #[arg(long, default_value_t = false, hide = true)]
    list_hashx_gpu_adapters: bool,
    /// Run the specialized Anubis SHA GPU kernel for an exact candidate budget and exit.
    #[cfg(feature = "gpu")]
    #[arg(long, hide = true)]
    benchmark_anubis_gpu_hashes: Option<u64>,
    /// Sample index attached to the specialized Anubis GPU benchmark record.
    #[cfg(feature = "gpu")]
    #[arg(long, hide = true, default_value_t = 0)]
    benchmark_sample: usize,
    /// Calibration JSON produced by tools/anubis-e2e --calibrate-routing.
    #[arg(long)]
    routing_calibration: Option<std::path::PathBuf>,
}

fn default_argon2_workers(
    cpu_workers: usize,
    available_workers: usize,
    physical_workers: usize,
) -> usize {
    cpu_workers
        .min(available_workers)
        .min(physical_workers)
        .max(1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("challenge_disasterer=info".parse()?),
        )
        .init();
    let args = Args::parse();
    anyhow::ensure!(
        args.cpu_workers > 0,
        "cpu-workers must be greater than zero"
    );
    let available_workers = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let default_argon2_workers = default_argon2_workers(
        args.cpu_workers,
        available_workers,
        num_cpus::get_physical(),
    );
    let argon2_workers = args.argon2_workers.unwrap_or(default_argon2_workers);
    anyhow::ensure!(
        argon2_workers > 0,
        "argon2-workers must be greater than zero"
    );
    anyhow::ensure!(
        argon2_workers <= args.cpu_workers,
        "argon2-workers ({argon2_workers}) cannot exceed cpu-workers ({})",
        args.cpu_workers
    );
    #[cfg(feature = "gpu")]
    if args.list_hashx_gpu_adapters {
        println!(
            "{}",
            serde_json::to_string(&hashx_gpu::adapter_names().await)?
        );
        return Ok(());
    }
    #[cfg(feature = "gpu")]
    if let Some(hashes) = args.benchmark_anubis_gpu_hashes {
        anyhow::ensure!(
            hashes > 0,
            "benchmark hash budget must be greater than zero"
        );
        anyhow::ensure!(
            hashes % gpu::GpuSolver::work_quantum() == 0,
            "benchmark hash budget must be a multiple of {}",
            gpu::GpuSolver::work_quantum()
        );
        let mut solver = gpu::GpuSolver::create().await?;
        let adapter = solver.adapter().to_owned();
        let hashes_per_second = solver.calibrate(hashes)?;
        println!(
            "{}",
            serde_json::json!({
                "schema": 1,
                "protocol": "anubis",
                "variant": "specialized-sha256-wgpu",
                "implementation": "specialized-wgpu",
                "accelerator": "gpu",
                "sample": args.benchmark_sample,
                "attempted_work": hashes,
                "work_per_second": hashes_per_second,
                "solve_ns": ((hashes as f64 / hashes_per_second) * 1e9).round() as u128,
                "gpu_adapter": adapter,
            })
        );
        return Ok(());
    }
    let config = Config {
        cpu_workers: args.cpu_workers,
        argon2_workers,
        cpu_calibration_hashes: args.cpu_calibration_hashes,
        #[cfg(feature = "gpu")]
        gpu_calibration_hashes: args.gpu_calibration_hashes,
        gpu_enabled: args.gpu,
        routing_calibration: args.routing_calibration,
    };
    let state = Arc::new(AppState::initialize(config).await?);
    let app = Router::new()
        .route("/solve", post(service::solve))
        .route("/stats", get(service::stats))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .context("bind")?;
    info!(listen = %args.listen, "challenge-disasterer ready");
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::default_argon2_workers;

    #[test]
    fn argon2_defaults_to_physical_cores_on_smt_hosts() {
        assert_eq!(default_argon2_workers(16, 16, 8), 8);
    }

    #[test]
    fn argon2_respects_smaller_configured_cpu_budget() {
        assert_eq!(default_argon2_workers(4, 16, 8), 4);
    }

    #[test]
    fn argon2_respects_smaller_available_parallelism() {
        assert_eq!(default_argon2_workers(16, 4, 8), 4);
    }
}
