#![recursion_limit = "256"]

mod anubis;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod hashx_gpu;
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
    let config = Config {
        cpu_workers: args.cpu_workers,
        cpu_calibration_hashes: args.cpu_calibration_hashes,
        #[cfg(feature = "gpu")]
        gpu_calibration_hashes: args.gpu_calibration_hashes,
        gpu_enabled: args.gpu,
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
