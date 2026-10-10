#![recursion_limit = "256"]

mod anubis;
mod classic_pow;
#[cfg(feature = "gpu")]
mod gpu;
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
    /// Calibration JSON produced by tools/anubis-e2e --calibrate-routing.
    #[arg(long)]
    routing_calibration: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("challenge_disasterer=info".parse()?),
        )
        .init();
    let args = Args::parse();
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
