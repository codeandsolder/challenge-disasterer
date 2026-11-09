use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicU64},
    time::Instant,
};

#[cfg(feature = "gpu")]
use anyhow::Context;
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(feature = "gpu")]
use std::sync::atomic::Ordering;
#[cfg(feature = "gpu")]
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn};

use crate::{
    anubis::{self, Challenge},
    anubis_wasm, hashx,
    routing::RoutingCalibration,
    stats::{Calibration, StatsStore},
};
#[cfg(feature = "gpu")]
use crate::{
    classic_pow,
    gpu::GpuSolver,
    hashx_gpu::{GpuResult as HashxGpuResult, GpuSolver as HashxGpuSolver},
};

mod argon2_schedule;
mod execution;
pub use argon2_schedule::WorkerPolicy;
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
    pub argon2_workers: usize,
    pub altcha_worker_policy: WorkerPolicy,
    pub argon2_memory_budget_mib: Option<usize>,
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
    general_gpu_native_u64: bool,
    hashx_gpu_implementations: Vec<&'static str>,
    #[cfg(feature = "gpu")]
    gpu_slots: Arc<Semaphore>,
    gpu_queue_work: Arc<AtomicU64>,
    gpu_queue_depth: Arc<AtomicU64>,
    gpu_active_work: Arc<AtomicU64>,
    cpu_slots: Arc<Semaphore>,
    argon2_memory_slots: Option<Arc<Semaphore>>,
    argon2_memory_capacity_mib: Option<usize>,
    stats: RwLock<StatsStore>,
    routing: Option<RoutingCalibration>,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    pub solver: String,
    #[serde(default)]
    pub algorithm: Option<String>,
    #[serde(default)]
    pub implementation: Option<String>,
    #[serde(default = "default_batch_size_hint")]
    pub batch_size_hint: usize,
    pub challenge: Value,
}

const fn default_batch_size_hint() -> usize {
    1
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

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    error: String,
}

#[derive(Debug, Deserialize)]
struct ExternalSolveResponse {
    backend: String,
    #[serde(default)]
    implementation: Option<String>,
    answer: Value,
    solve_ms: f64,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorResponse>)>;

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
            config.argon2_workers > 0 && config.argon2_workers <= config.cpu_workers,
            "argon2_workers must be in 1..=cpu_workers"
        );

        let argon2_memory_capacity_mib = config.argon2_memory_budget_mib.or_else(|| {
            argon2_schedule::available_memory_mib().map(|available| (available / 2).max(1))
        });
        if let Some(capacity) = argon2_memory_capacity_mib {
            anyhow::ensure!(
                capacity > 0 && capacity <= Semaphore::MAX_PERMITS,
                "Argon2 memory budget must be within 1..={} MiB",
                Semaphore::MAX_PERMITS
            );
            info!(
                capacity_mib = capacity,
                "Argon2 shared memory budget configured"
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
        let (general_gpu_tx, general_gpu_operational, general_gpu_native_u64) = if config
            .gpu_enabled
        {
            match challenge_solvers::gpu::GpuContext::create().await {
                Ok(gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    let native_u64 = gpu.features().contains(wgpu::Features::SHADER_INT64);
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
                            (Some(tx), true, native_u64)
                        }
                        Err(err) => {
                            warn!(%err, "general WGPU warm-up failed");
                            (None, false, false)
                        }
                    }
                }
                Err(err) => {
                    warn!(%err, "general WGPU unavailable");
                    (None, false, false)
                }
            }
        } else {
            (None, false, false)
        };

        #[cfg(not(feature = "gpu"))]
        let (general_gpu_operational, general_gpu_native_u64) = (false, false);

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
            general_gpu_native_u64,
            hashx_gpu_implementations,
            #[cfg(feature = "gpu")]
            gpu_slots,
            gpu_queue_work: queue_work,
            gpu_queue_depth: queue_depth,
            gpu_active_work: active_work,
            cpu_slots: Arc::new(Semaphore::new(config.cpu_workers)),
            argon2_memory_slots: argon2_memory_capacity_mib
                .map(|capacity| Arc::new(Semaphore::new(capacity))),
            argon2_memory_capacity_mib,
            stats: RwLock::new(StatsStore::default()),
            routing,
            http: reqwest::Client::new(),
        })
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "HTTP solver dispatch remains explicit so protocol-specific validation and routing are auditable"
)]
pub async fn solve(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<SolveResponse> {
    let queued_at = Instant::now();
    if request.batch_size_hint == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "batch_size_hint must be greater than zero".to_owned(),
            }),
        ));
    }
    let (difficulty, result) = match request.solver.as_str() {
        "anubis" | "anubis-cpu" | "anubis-gpu" => {
            let algorithm = request.algorithm.as_deref().unwrap_or("fast");
            if !matches!(algorithm, "fast" | "slow") {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("unsupported Anubis SHA algorithm: {algorithm}"),
                    }),
                ));
            }
            if request.implementation.is_some() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "implementation override is only supported for anubis-hashx"
                            .to_owned(),
                    }),
                ));
            }
            let challenge: Challenge =
                serde_json::from_value(request.challenge).map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: err.to_string(),
                        }),
                    )
                })?;
            let difficulty = challenge.difficulty;
            let result = match request.solver.as_str() {
                "anubis-cpu" => {
                    challenge.validate().map_err(|err| {
                        (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: err.to_string(),
                            }),
                        )
                    })?;
                    state.run_cpu(challenge, queued_at).await
                }
                "anubis-gpu" => {
                    #[cfg(feature = "gpu")]
                    {
                        state.run_gpu_only(challenge, queued_at).await
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        Err(anyhow::anyhow!("SHA GPU backend not compiled"))
                    }
                }
                _ => {
                    state
                        .route_anubis(algorithm, challenge, request.batch_size_hint, queued_at)
                        .await
                }
            };
            (difficulty, result)
        }
        "anubis-sha256" | "anubis-argon2id" => {
            let expected_algorithm = if request.solver == "anubis-sha256" {
                "sha256"
            } else {
                "argon2id"
            };
            if request
                .algorithm
                .as_deref()
                .is_some_and(|algorithm| algorithm != expected_algorithm)
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!(
                            "{} solver requires algorithm={expected_algorithm}",
                            request.solver
                        ),
                    }),
                ));
            }
            let challenge: anubis_wasm::Challenge = serde_json::from_value(request.challenge)
                .map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: err.to_string(),
                        }),
                    )
                })?;
            challenge.validate().map_err(|err| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: err.to_string(),
                    }),
                )
            })?;
            let difficulty = u8::try_from(challenge.difficulty).unwrap_or(u8::MAX);
            let result = if request.solver == "anubis-sha256"
                && request.implementation.as_deref() == Some("wgpu")
            {
                #[cfg(feature = "gpu")]
                {
                    state.run_anubis_wasm_gpu(challenge, queued_at).await
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow::anyhow!(
                        "Anubis WASM SHA-256 WGPU backend not compiled"
                    ))
                }
            } else {
                state
                    .run_anubis_wasm(
                        request.solver.clone(),
                        challenge,
                        request.implementation.clone(),
                        queued_at,
                    )
                    .await
            };
            (difficulty, result)
        }
        "anubis-hashx" | "anubis-hashx-jit" | "anubis-hashx-auto" => {
            if request
                .algorithm
                .as_deref()
                .is_some_and(|algorithm| algorithm != "hashx")
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "HashX solver requires algorithm=hashx".to_owned(),
                    }),
                ));
            }
            let challenge: hashx::Challenge =
                serde_json::from_value(request.challenge).map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: err.to_string(),
                        }),
                    )
                })?;
            let difficulty = u8::try_from(challenge.difficulty).unwrap_or(u8::MAX);
            let result = match request.solver.as_str() {
                "anubis-hashx-jit" => {
                    if request.implementation.is_some() {
                        Err(anyhow::anyhow!(
                            "implementation override is not supported for anubis-hashx-jit"
                        ))
                    } else {
                        state.run_hashx_jit(challenge, queued_at).await
                    }
                }
                "anubis-hashx-auto" => {
                    if request.implementation.is_some() {
                        Err(anyhow::anyhow!(
                            "implementation override is not supported for anubis-hashx-auto"
                        ))
                    } else {
                        state
                            .route_hashx(challenge, request.batch_size_hint, queued_at)
                            .await
                    }
                }
                _ => {
                    #[cfg(feature = "gpu")]
                    {
                        state
                            .run_hashx_gpu(challenge, request.implementation.clone(), queued_at)
                            .await
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        Err(anyhow::anyhow!("HashX GPU backend not compiled"))
                    }
                }
            };
            (difficulty, result)
        }
        "cerberus" | "goaway" | "capjs" | "mcaptcha" | "haphash" | "altcha" | "linear" => {
            if request.solver == "linear"
                && matches!(
                    request.implementation.as_deref(),
                    Some(implementation) if implementation != "auto"
                )
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Linear.pub exposes only the auto implementation".to_owned(),
                    }),
                ));
            }
            if request.algorithm.is_some() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "algorithm override is not used by this protocol solver".to_owned(),
                    }),
                ));
            }
            let result = if request.implementation.as_deref() == Some("wgpu") {
                #[cfg(feature = "gpu")]
                {
                    state
                        .run_general_gpu(request.solver.clone(), request.challenge, queued_at)
                        .await
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow::anyhow!("general WGPU backend not compiled"))
                }
            } else {
                state
                    .run_classic(
                        request.solver.clone(),
                        request.challenge,
                        request.implementation.clone(),
                        queued_at,
                    )
                    .await
            };
            (0, result)
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("unknown solver: {other}"),
                }),
            ));
        }
    };

    match result {
        Ok(mut response) => {
            response.solver = request.solver.clone();
            let mut stats_guard = state.stats.write().await;
            response.record_success(&mut stats_guard, difficulty);
            drop(stats_guard);
            Ok(Json(response))
        }
        Err(err) => {
            state.stats.write().await.record_failure();
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hardware-only route test: all automatic GPU policies honor the same
    /// GPU ownership, with verified CPU and GPU challenge solutions.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    #[ignore = "requires a physical WGPU adapter"]
    async fn sha256_gpu_policy_idle_busy_queued_and_forced() -> anyhow::Result<()> {
        use std::time::Duration;

        let mut state = AppState::initialize(Config {
            cpu_workers: 2,
            argon2_workers: 1,
            altcha_worker_policy: WorkerPolicy::Adaptive,
            argon2_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            gpu_calibration_hashes: 8_192,
            gpu_enabled: true,
            gpu_policy: GpuPolicy::IdleOnly,
            routing_calibration: None,
        })
        .await?;
        assert!(state.gpu_operational, "specialized SHA GPU must be ready");
        assert!(
            state.general_gpu_operational,
            "general SHA GPU must be ready"
        );
        let classic = anubis::Challenge {
            data: "a".repeat(64),
            difficulty: 1,
        };
        let wasm = anubis_wasm::Challenge {
            data: "02".repeat(64),
            difficulty: 1,
        };
        let idle = state
            .route_anubis("fast", classic.clone(), 1, Instant::now())
            .await?;
        assert_eq!(idle.backend, "gpu");
        let answer: anubis::Answer = serde_json::from_value(idle.answer)?;
        assert!(anubis::verify(&classic, &answer));

        let idle = state
            .try_run_anubis_wasm_gpu(wasm.clone(), Instant::now())
            .await?
            .context("idle general SHA256 GPU route")?;
        assert_eq!(idle.backend, "gpu");
        let answer: anubis::Answer = serde_json::from_value(idle.answer)?;
        assert!(anubis_wasm::verify_sha256(&wasm, &answer));

        let busy = state.gpu_slots.clone().acquire_owned().await?;
        let cpu = tokio::time::timeout(
            Duration::from_secs(10),
            state.route_anubis("fast", classic.clone(), 1, Instant::now()),
        )
        .await??;
        assert_eq!(cpu.backend, "cpu", "must not wait for busy GPU");
        let answer: anubis::Answer = serde_json::from_value(cpu.answer)?;
        assert!(anubis::verify(&classic, &answer));
        assert!(
            state
                .try_run_anubis_wasm_gpu(wasm, Instant::now())
                .await?
                .is_none(),
            "general SHA256 GPU must also see the occupied global slot"
        );
        drop(busy);
        // Default policy favors GPU even while occupied by another request.
        // The semaphore is owned by the GPU job, not released when enqueued.
        state.config.gpu_policy = GpuPolicy::PreferGpu;
        let busy = state.gpu_slots.clone().acquire_owned().await?;
        {
            let pending = state.route_anubis("fast", classic.clone(), 1, Instant::now());
            tokio::pin!(pending);
            assert!(
                tokio::time::timeout(Duration::from_millis(100), pending.as_mut())
                    .await
                    .is_err(),
                "prefer-gpu must wait rather than dispatching to CPU"
            );
            drop(busy);
            let queued = tokio::time::timeout(Duration::from_secs(10), pending.as_mut()).await??;
            assert_eq!(queued.backend, "gpu", "queued GPU must be preferred");
            let answer: anubis::Answer = serde_json::from_value(queued.answer)?;
            assert!(anubis::verify(&classic, &answer));
        }

        // CPU-only is a preference for automatic work, not a prohibition on
        // the explicit GPU endpoint.
        state.config.gpu_policy = GpuPolicy::CpuOnly;
        let cpu = state
            .route_anubis("fast", classic.clone(), 1, Instant::now())
            .await?;
        assert_eq!(cpu.backend, "cpu");
        let forced_gpu = state.run_gpu_only(classic.clone(), Instant::now()).await?;
        assert_eq!(forced_gpu.backend, "gpu");

        state.config.gpu_policy = GpuPolicy::PreferGpu;
        let recovered = state
            .route_anubis("fast", classic, 1, Instant::now())
            .await?;
        assert_eq!(recovered.backend, "gpu", "GPU should be reusable");

        let cerberus = serde_json::json!({
            "challenge": "gpu-policy-cerberus",
            "difficulty": 2,
            "nonce": 3959614364_u64,
            "ts": 1759954402_u64,
            "signature": "testing",
            "version": "0.4.9"
        });
        let solved = state
            .run_classic(
                "cerberus".to_owned(),
                cerberus.clone(),
                None,
                Instant::now(),
            )
            .await?;
        assert_eq!(solved.backend, "gpu", "Cerberus BLAKE3 must prefer GPU");
        assert!(solved.answer["response"].as_str().is_some());
        assert!(solved.answer["solution"].as_str().is_some());
        state.config.gpu_policy = GpuPolicy::CpuOnly;
        let cpu = state
            .run_classic(
                "cerberus".to_owned(),
                cerberus.clone(),
                None,
                Instant::now(),
            )
            .await?;
        assert_eq!(cpu.backend, "cpu");
        let explicit_gpu = state
            .run_general_gpu("cerberus".to_owned(), cerberus, Instant::now())
            .await?;
        assert_eq!(explicit_gpu.backend, "gpu");
        Ok(())
    }

    /// Real service routing through the ALTCHA SHA-384/SHA-512 and PBKDF2
    /// protocol adapter. Compares complete keys to the independent CPU path.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    #[ignore = "requires a physical WGPU adapter with SHADER_INT64"]
    async fn altcha_sha64_auto_gpu_and_policy_overrides_match_cpu() -> anyhow::Result<()> {
        let mut state = AppState::initialize(Config {
            cpu_workers: 2,
            argon2_workers: 1,
            altcha_worker_policy: WorkerPolicy::Adaptive,
            argon2_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            gpu_calibration_hashes: 8_192,
            gpu_enabled: true,
            gpu_policy: GpuPolicy::PreferGpu,
            routing_calibration: None,
        })
        .await?;
        assert!(state.general_gpu_operational);
        for (algorithm, key_length, decimal) in [
            ("SHA-384", 33, false),
            ("SHA-512", 47, true),
            ("PBKDF2/SHA-384", 115, false),
            ("PBKDF2/SHA-512", 149, true),
        ] {
            let mut challenge = serde_json::json!({
                "counterMode": if decimal { "string" } else { "uint32" },
                "parameters": {
                    "algorithm": algorithm,
                    "cost": 2,
                    "keyLength": key_length,
                    "keyPrefix": "",
                    "nonce": "9a".repeat(135),
                    "salt": "7b".repeat(157)
                }
            });
            // An empty prefix makes every nonce valid and lets the GPU
            // return any workgroup's winner. Derive the real key with the
            // CPU reference, then challenge the GPU with its full prefix.
            // This also exercises PBKDF2 matching across output blocks.
            let reference = classic_pow::solve("altcha", challenge.clone(), Some("safe"), 1)?;
            assert_eq!(reference.answer["counter"], 0);
            let key = reference.answer["derivedKey"]
                .as_str()
                .context("CPU ALTCHA answer lacks derivedKey")?;
            challenge["parameters"]["keyPrefix"] = serde_json::json!(key);
            let gpu = state
                .run_classic("altcha".to_owned(), challenge.clone(), None, Instant::now())
                .await?;
            assert_eq!(gpu.backend, "gpu", "{algorithm} must prefer GPU");
            assert_eq!(gpu.answer["counter"], 0);
            assert_eq!(
                gpu.answer["derivedKey"], reference.answer["derivedKey"],
                "{algorithm} derivedKey must exactly match the CPU verifier"
            );
            if algorithm == "PBKDF2/SHA-512" {
                state.config.gpu_policy = GpuPolicy::CpuOnly;
                let cpu = state
                    .run_classic("altcha".to_owned(), challenge.clone(), None, Instant::now())
                    .await?;
                assert_eq!(cpu.backend, "cpu", "cpu-only must override GPU preference");
                assert_eq!(cpu.answer["derivedKey"], reference.answer["derivedKey"]);
                let forced = state
                    .run_general_gpu("altcha".to_owned(), challenge, Instant::now())
                    .await?;
                assert_eq!(forced.backend, "gpu", "explicit WGPU overrides cpu-only");
                assert_eq!(forced.answer["derivedKey"], reference.answer["derivedKey"]);
            }
        }
        Ok(())
    }

    #[test]
    fn response_stats_record_solve_time_not_queue_or_request_time() {
        let mut stats = StatsStore::default();
        let response = SolveResponse {
            solver: "anubis".to_owned(),
            backend: "cpu".to_owned(),
            implementation: None,
            nonce_workers: None,
            answer: serde_json::Value::Null,
            queue_ms: 90.0,
            solve_ms: 3.0,
            total_ms: 100.0,
        };
        response.record_success(&mut stats, 7);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&7], 3.0);

        let other = SolveResponse {
            solver: "linear".to_owned(),
            ..response
        };
        other.record_success(&mut stats, 0);
        assert_eq!(stats.solves_total, 2);
        assert_eq!(stats.solves_cpu, 2);
        assert!(
            !stats
                .snapshot()
                .median_solve_ms_by_difficulty
                .contains_key(&0)
        );
    }
}
