#![recursion_limit = "256"]

#[cfg(feature = "gpu")]
pub(crate) use challenge_hashx::hashx_gpu;
pub(crate) use challenge_hashx::{hashx, hashx_cpu_jit};
pub(crate) use challenge_protocols::{anubis, anubis_wasm, classic_pow};
#[cfg(feature = "gpu")]
mod gpu;
mod metrics;
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
use service::{AppState, Config, GpuPolicy};
use tracing::info;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:17417")]
    listen: SocketAddr,
    /// Shared CPU worker budget across all requests and solver types.
    #[arg(long, default_value_t = 4)]
    cpu_workers: usize,
    /// Per-request worker limit, defaulting to the total CPU worker budget.
    #[arg(long)]
    cpu_workers_per_request: Option<usize>,
    /// Shared memory budget for all memory-hard KDF arenas, in MiB.
    /// Defaults to half the host/cgroup available RAM on Linux when readable.
    #[arg(long)]
    kdf_memory_budget_mib: Option<usize>,
    #[arg(long, default_value_t = 2_000_000)]
    cpu_calibration_hashes: u64,
    #[cfg(feature = "gpu")]
    #[arg(long, default_value_t = 64_000_000)]
    gpu_calibration_hashes: u64,
    #[arg(long, default_value_t = cfg!(feature = "gpu"), action = clap::ArgAction::Set)]
    gpu: bool,
    /// Automatic SHA-256/BLAKE3 placement: prefer GPU by default; optionally only
    /// use an idle GPU or keep automatic work on CPU.
    #[arg(long, value_enum, default_value_t = GpuPolicy::PreferGpu)]
    gpu_policy: GpuPolicy,
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
    let cpu_workers_per_request = args.cpu_workers_per_request.unwrap_or(args.cpu_workers);
    anyhow::ensure!(
        cpu_workers_per_request > 0 && cpu_workers_per_request <= args.cpu_workers,
        "cpu-workers-per-request must be in 1..=cpu-workers"
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
        let solver = gpu::GpuSolver::create().await?;
        let adapter = solver.adapter().to_owned();
        let hashes_per_second = solver.calibrate(hashes)?;
        let solve_ns = std::time::Duration::try_from_secs_f64(
            metrics::approximate_work_f64(hashes) / hashes_per_second,
        )
        .context("benchmark solve duration is not finite")?
        .as_nanos();
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
                "solve_ns": solve_ns,
                "gpu_adapter": adapter,
            })
        );
        return Ok(());
    }
    let config = Config {
        cpu_workers: args.cpu_workers,
        cpu_workers_per_request,
        kdf_memory_budget_mib: args.kdf_memory_budget_mib,
        cpu_calibration_hashes: args.cpu_calibration_hashes,
        #[cfg(feature = "gpu")]
        gpu_calibration_hashes: args.gpu_calibration_hashes,
        gpu_enabled: args.gpu,
        gpu_policy: args.gpu_policy,
        routing_calibration: args.routing_calibration,
    };
    let app = Router::new()
        .route("/solve", post(service::solve))
        .route("/stats", get(service::stats))
        .with_state(Arc::new(AppState::initialize(config).await?));
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .context("bind")?;
    info!(listen = %args.listen, "challenge-disasterer ready");
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Args, GpuPolicy};
    use clap::Parser;

    #[test]
    fn automatic_gpu_placement_defaults_to_gpu_and_is_configurable() {
        assert!(matches!(
            Args::try_parse_from(["challenge-disasterer"]),
            Ok(Args {
                gpu_policy: GpuPolicy::PreferGpu,
                ..
            })
        ));
        for (value, policy) in [
            ("prefer-gpu", GpuPolicy::PreferGpu),
            ("idle-only", GpuPolicy::IdleOnly),
            ("cpu-only", GpuPolicy::CpuOnly),
        ] {
            assert!(matches!(
                Args::try_parse_from([
                    "challenge-disasterer",
                    "--gpu-policy",
                    value
                ]),
                Ok(Args { gpu_policy, .. }) if gpu_policy == policy
            ));
        }
    }

    #[test]
    fn cpu_worker_limits_parse_independently() {
        let parsed = Args::try_parse_from([
            "challenge-disasterer",
            "--cpu-workers",
            "8",
            "--cpu-workers-per-request",
            "3",
        ]);
        assert!(matches!(
            parsed,
            Ok(Args {
                cpu_workers: 8,
                cpu_workers_per_request: Some(3),
                ..
            })
        ));
        let defaults = Args::try_parse_from(["challenge-disasterer"]);
        assert!(matches!(
            defaults,
            Ok(Args {
                cpu_workers: 4,
                cpu_workers_per_request: None,
                ..
            })
        ));
    }
}
