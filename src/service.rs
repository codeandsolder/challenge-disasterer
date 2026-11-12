#[cfg(feature = "gpu")]
use std::time::Instant;
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicU64},
};

#[cfg(feature = "gpu")]
use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(feature = "gpu")]
use std::sync::atomic::Ordering;
#[cfg(feature = "gpu")]
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn};

use crate::{
    anubis,
    routing::RoutingCalibration,
    stats::{Calibration, StatsStore},
};
#[cfg(feature = "gpu")]
use crate::{
    anubis::Challenge,
    anubis_wasm, classic_pow,
    gpu::GpuSolver,
    hashx,
    hashx_gpu::{GpuResult as HashxGpuResult, GpuSolver as HashxGpuSolver},
};

mod execution;
mod http;
pub use http::solve;
mod kdf_schedule;
mod routing;
mod telemetry;
pub use telemetry::stats;

/// Policy for automatic hash-puzzle work placement, independent of measured
/// CPU/GPU solve latency. Explicit backend requests override this policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GpuPolicy {
    /// Assign supported hash puzzles to GPU, waiting for earlier GPU jobs.
    PreferGpu,
    /// Use GPU only when no in-process GPU job currently owns its slot.
    IdleOnly,
    /// Route automatic hash puzzles to CPU; explicit GPU remains available.
    CpuOnly,
}

impl GpuPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreferGpu => "prefer-gpu",
            Self::IdleOnly => "idle-only",
            Self::CpuOnly => "cpu-only",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub cpu_workers: usize,
    pub cpu_workers_per_request: usize,
    pub kdf_memory_budget_mib: Option<usize>,
    pub cpu_calibration_hashes: u64,
    #[cfg(feature = "gpu")]
    pub gpu_calibration_hashes: u64,
    pub gpu_enabled: bool,
    pub gpu_policy: GpuPolicy,
    pub routing_calibration: Option<PathBuf>,
}

#[cfg(feature = "gpu")]
struct GpuJob {
    _gpu_permit: OwnedSemaphorePermit,
    challenge: Challenge,
    expected_work: u64,
    started: oneshot::Sender<Instant>,
    result: oneshot::Sender<anyhow::Result<anubis::BackendResult>>,
}

#[cfg(feature = "gpu")]
struct HashxGpuJob {
    _gpu_permit: OwnedSemaphorePermit,
    challenge: hashx::Challenge,
    implementation: Option<String>,
    started: oneshot::Sender<Instant>,
    result: oneshot::Sender<anyhow::Result<HashxGpuResult>>,
}

#[cfg(feature = "gpu")]
struct GeneralGpuJob {
    _gpu_permit: OwnedSemaphorePermit,
    solver: String,
    challenge: Value,
    started: oneshot::Sender<Instant>,
    result: oneshot::Sender<anyhow::Result<classic_pow::Result>>,
}

pub struct AppState {
    config: Config,
    calibration: Calibration,
    #[cfg(feature = "gpu")]
    gpu_tx: Option<mpsc::Sender<GpuJob>>,
    #[cfg(feature = "gpu")]
    hashx_gpu_tx: Option<mpsc::Sender<HashxGpuJob>>,
    #[cfg(feature = "gpu")]
    general_gpu_tx: Option<mpsc::Sender<GeneralGpuJob>>,
    gpu_operational: bool,
    hashx_gpu_operational: bool,
    general_gpu_operational: bool,
    #[cfg(feature = "gpu")]
    general_gpu_features: wgpu::Features,
    hashx_gpu_implementations: Vec<&'static str>,
    #[cfg(feature = "gpu")]
    gpu_slots: Arc<Semaphore>,
    gpu_queue_work: Arc<AtomicU64>,
    gpu_queue_depth: Arc<AtomicU64>,
    gpu_active_work: Arc<AtomicU64>,
    cpu_slots: Arc<Semaphore>,
    kdf_memory_slots: Option<Arc<Semaphore>>,
    kdf_memory_capacity_mib: Option<usize>,
    kdf_admission: kdf_schedule::KdfAdmission,
    stats: RwLock<StatsStore>,
    routing: Option<RoutingCalibration>,
    http: reqwest::Client,
}

#[derive(Debug, Serialize)]
pub struct SolveResponse {
    solver: String,
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    implementation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nonce_workers: Option<usize>,
    answer: Value,
    queue_ms: f64,
    solve_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Deserialize)]
struct ExternalSolveResponse {
    backend: String,
    #[serde(default)]
    implementation: Option<String>,
    answer: Value,
    solve_ms: f64,
}

impl SolveResponse {
    fn record_success(&self, stats: &mut StatsStore, difficulty: u8) {
        if self.solver.starts_with("anubis") {
            stats.record(difficulty, &self.backend, self.solve_ms);
        } else {
            stats.record_generic(&self.backend);
        }
    }
}

#[cfg(feature = "gpu")]
async fn solve_general_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    solver: &str,
    challenge: Value,
) -> anyhow::Result<classic_pow::Result> {
    if solver == "anubis-sha256" {
        let challenge: anubis_wasm::Challenge =
            serde_json::from_value(challenge).context("invalid Anubis WASM SHA-256 challenge")?;
        let result = anubis_wasm::solve_sha256_gpu(gpu, &challenge).await?;
        return Ok(classic_pow::Result {
            answer: serde_json::to_value(result.answer)
                .context("serialize Anubis WASM SHA-256 GPU answer")?,
            solve_ms: result.solve_ms,
            implementation: result.implementation,
        });
    }
    classic_pow::gpu::solve(gpu, solver, challenge).await
}

impl AppState {
    /// Whether the running general WGPU context can execute native-u64 shaders.
    /// The status reflects the negotiated device, not the adapter name or
    /// unrelated SHA-256 GPU capability.
    const fn general_gpu_has_native_u64(&self) -> bool {
        if !self.general_gpu_operational {
            return false;
        }
        #[cfg(feature = "gpu")]
        {
            self.general_gpu_features
                .contains(wgpu::Features::SHADER_INT64)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "service initialization keeps calibration and backend bring-up in one ordered transaction"
    )]
    #[cfg_attr(
        feature = "gpu",
        expect(
            clippy::future_not_send,
            reason = "general WGPU initialization intentionally awaits a local non-Send future before any service worker is spawned"
        )
    )]
    pub async fn initialize(config: Config) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.cpu_workers > 0,
            "cpu_workers must be greater than zero"
        );
        anyhow::ensure!(
            config.cpu_workers_per_request > 0
                && config.cpu_workers_per_request <= config.cpu_workers,
            "cpu_workers_per_request must be in 1..=cpu_workers"
        );

        let kdf_memory_capacity_mib = config.kdf_memory_budget_mib.or_else(|| {
            kdf_schedule::available_memory_mib().map(|available| (available / 2).max(1))
        });
        if let Some(capacity) = kdf_memory_capacity_mib {
            anyhow::ensure!(
                capacity > 0 && capacity <= Semaphore::MAX_PERMITS,
                "KDF memory budget must be within 1..={} MiB",
                Semaphore::MAX_PERMITS
            );
            info!(
                capacity_mib = capacity,
                "Shared KDF memory budget configured"
            );
        }

        let cpu_hashes_per_second = tokio::task::spawn_blocking({
            let n = config.cpu_calibration_hashes;
            move || anubis::calibrate_cpu(n)
        })
        .await??;
        info!(
            mh_s = cpu_hashes_per_second / 1e6,
            "CPU calibration complete"
        );

        #[cfg(feature = "gpu")]
        let gpu_slots = Arc::new(Semaphore::new(1));
        let queue_work = Arc::new(AtomicU64::new(0));
        let queue_depth = Arc::new(AtomicU64::new(0));
        let active_work = Arc::new(AtomicU64::new(0));

        #[cfg(feature = "gpu")]
        let (gpu_tx, gpu_rate, gpu_adapter) = if config.gpu_enabled {
            match GpuSolver::create().await {
                Ok(gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    match gpu.calibrate(config.gpu_calibration_hashes) {
                        Ok(rate) => {
                            info!(adapter = %adapter, mh_s = rate / 1e6, "GPU calibration complete");
                            let (tx, mut rx) = mpsc::channel::<GpuJob>(1024);
                            let worker_queue_work = queue_work.clone();
                            let worker_queue_depth = queue_depth.clone();
                            let worker_active_work = active_work.clone();
                            tokio::spawn(async move {
                                while let Some(job) = rx.recv().await {
                                    worker_queue_depth.fetch_sub(1, Ordering::Relaxed);
                                    worker_queue_work
                                        .fetch_sub(job.expected_work, Ordering::Relaxed);
                                    worker_active_work.store(job.expected_work, Ordering::Relaxed);
                                    let _ = job.started.send(Instant::now());
                                    let result = gpu.solve(&job.challenge);
                                    worker_active_work.store(0, Ordering::Relaxed);
                                    let _ = job.result.send(result);
                                }
                            });
                            (Some(tx), Some(rate), Some(adapter))
                        }
                        Err(err) => {
                            warn!(%err, "GPU calibration failed; using CPU");
                            (None, None, Some(adapter))
                        }
                    }
                }
                Err(err) => {
                    warn!(%err, "GPU unavailable; using CPU");
                    (None, None, None)
                }
            }
        } else {
            info!("GPU disabled; using CPU");
            (None, None, None)
        };

        #[cfg(feature = "gpu")]
        let gpu_operational = gpu_tx.is_some();

        #[cfg(not(feature = "gpu"))]
        let (gpu_rate, gpu_adapter, gpu_operational) = {
            if config.gpu_enabled {
                warn!("GPU requested but this binary was built without the gpu feature; using CPU");
            }
            (None, None, false)
        };

        #[cfg(feature = "gpu")]
        let (hashx_gpu_tx, hashx_gpu_operational, hashx_gpu_implementations) = if config.gpu_enabled
        {
            match HashxGpuSolver::create().await {
                Ok(mut gpu) => {
                    let implementations = gpu.implementations();
                    info!(adapter = %gpu.adapter(), ?implementations, "HashX GPU backend ready");
                    let (tx, mut rx) = mpsc::channel::<HashxGpuJob>(128);
                    tokio::spawn(async move {
                        while let Some(job) = rx.recv().await {
                            let _ = job.started.send(Instant::now());
                            let result = gpu.solve_with_implementation(
                                &job.challenge,
                                job.implementation.as_deref(),
                            );
                            let _ = job.result.send(result);
                        }
                    });
                    (Some(tx), true, implementations)
                }
                Err(err) => {
                    warn!(%err, "HashX GPU unavailable");
                    (None, false, Vec::new())
                }
            }
        } else {
            (None, false, Vec::new())
        };

        #[cfg(not(feature = "gpu"))]
        let (hashx_gpu_operational, hashx_gpu_implementations) = (false, Vec::new());

        #[cfg(feature = "gpu")]
        let (general_gpu_tx, general_gpu_operational, general_gpu_features) = if config.gpu_enabled
        {
            match challenge_solvers::gpu::GpuContext::create().await {
                Ok(gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    let enabled_features = gpu.features();
                    match gpu.warm_up().await {
                        Ok(()) => {
                            info!(adapter = %adapter, "general WGPU backend ready");
                            let (tx, mut rx) = mpsc::channel::<GeneralGpuJob>(128);
                            tokio::spawn(async move {
                                while let Some(job) = rx.recv().await {
                                    let _ = job.started.send(Instant::now());
                                    let result =
                                        solve_general_gpu(&gpu, &job.solver, job.challenge).await;
                                    let _ = job.result.send(result);
                                }
                            });
                            (Some(tx), true, enabled_features)
                        }
                        Err(err) => {
                            warn!(%err, "general WGPU warm-up failed");
                            (None, false, wgpu::Features::empty())
                        }
                    }
                }
                Err(err) => {
                    warn!(%err, "general WGPU unavailable");
                    (None, false, wgpu::Features::empty())
                }
            }
        } else {
            (None, false, wgpu::Features::empty())
        };

        #[cfg(not(feature = "gpu"))]
        let general_gpu_operational = false;

        let routing = config
            .routing_calibration
            .as_deref()
            .map(RoutingCalibration::load)
            .transpose()?;
        if let Some(path) = config.routing_calibration.as_deref() {
            info!(path = %path.display(), "routing calibration loaded");
        }

        Ok(Self {
            config: config.clone(),
            calibration: Calibration {
                gpu_hashes_per_second: gpu_rate,
                cpu_hashes_per_second_per_worker: cpu_hashes_per_second,
                gpu_adapter,
            },
            #[cfg(feature = "gpu")]
            gpu_tx,
            #[cfg(feature = "gpu")]
            hashx_gpu_tx,
            #[cfg(feature = "gpu")]
            general_gpu_tx,
            gpu_operational,
            hashx_gpu_operational,
            general_gpu_operational,
            #[cfg(feature = "gpu")]
            general_gpu_features,
            hashx_gpu_implementations,
            #[cfg(feature = "gpu")]
            gpu_slots,
            gpu_queue_work: queue_work,
            gpu_queue_depth: queue_depth,
            gpu_active_work: active_work,
            cpu_slots: Arc::new(Semaphore::new(config.cpu_workers)),
            kdf_memory_slots: kdf_memory_capacity_mib
                .map(|capacity| Arc::new(Semaphore::new(capacity))),
            kdf_memory_capacity_mib,
            kdf_admission: kdf_schedule::KdfAdmission::default(),
            stats: RwLock::new(StatsStore::default()),
            routing,
            http: reqwest::Client::new(),
        })
    }
}

#[cfg(test)]
mod tests;
