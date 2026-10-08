use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::Context;
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, Semaphore};
#[cfg(feature = "gpu")]
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use crate::{
    anubis::{self, Challenge},
    hashx, hashx_cpu_simd,
    stats::{Calibration, StatsStore},
};
#[cfg(feature = "gpu")]
use crate::{gpu::GpuSolver, hashx_gpu::GpuSolver as HashxGpuSolver};

#[derive(Debug, Clone)]
pub struct Config {
    pub cpu_workers: usize,
    pub cpu_calibration_hashes: u64,
    #[cfg(feature = "gpu")]
    pub gpu_calibration_hashes: u64,
    pub gpu_enabled: bool,
}

#[cfg(feature = "gpu")]
struct GpuJob {
    challenge: Challenge,
    expected_work: u64,
    started: oneshot::Sender<()>,
    result: oneshot::Sender<anyhow::Result<anubis::BackendResult>>,
}

#[cfg(feature = "gpu")]
struct HashxGpuJob {
    challenge: hashx::Challenge,
    result: oneshot::Sender<anyhow::Result<anubis::BackendResult>>,
}

pub struct AppState {
    config: Config,
    calibration: Calibration,
    #[cfg(feature = "gpu")]
    gpu_tx: Option<mpsc::Sender<GpuJob>>,
    #[cfg(feature = "gpu")]
    hashx_gpu_tx: Option<mpsc::Sender<HashxGpuJob>>,
    gpu_operational: bool,
    hashx_gpu_operational: bool,
    gpu_queue_work: Arc<AtomicU64>,
    gpu_queue_depth: Arc<AtomicU64>,
    gpu_active_work: Arc<AtomicU64>,
    cpu_slots: Arc<Semaphore>,
    stats: RwLock<StatsStore>,
}

#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    pub solver: String,
    pub challenge: Value,
}

#[derive(Debug, Serialize)]
pub struct SolveResponse {
    solver: &'static str,
    backend: &'static str,
    answer: anubis::Answer,
    queue_ms: f64,
    solve_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    error: String,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorResponse>)>;

impl AppState {
    pub async fn initialize(config: Config) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.cpu_workers > 0,
            "cpu_workers must be greater than zero"
        );

        let cpu_hashes_per_second = tokio::task::spawn_blocking({
            let n = config.cpu_calibration_hashes;
            move || anubis::calibrate_cpu(n)
        })
        .await??;
        info!(
            mh_s = cpu_hashes_per_second / 1e6,
            "CPU calibration complete"
        );

        let queue_work = Arc::new(AtomicU64::new(0));
        let queue_depth = Arc::new(AtomicU64::new(0));
        let active_work = Arc::new(AtomicU64::new(0));

        #[cfg(feature = "gpu")]
        let (gpu_tx, gpu_rate, gpu_adapter) = if config.gpu_enabled {
            match GpuSolver::create().await {
                Ok(mut gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    match gpu.calibrate(config.gpu_calibration_hashes) {
                        Ok(rate) => {
                            info!(adapter = %adapter, mh_s = rate / 1e6, "GPU calibration complete");
                            let (tx, mut rx) = mpsc::channel::<GpuJob>(1024);
                            let queued_work = queue_work.clone();
                            let queued_depth = queue_depth.clone();
                            let active_work = active_work.clone();
                            tokio::spawn(async move {
                                while let Some(job) = rx.recv().await {
                                    queued_depth.fetch_sub(1, Ordering::Relaxed);
                                    queued_work.fetch_sub(job.expected_work, Ordering::Relaxed);
                                    active_work.store(job.expected_work, Ordering::Relaxed);
                                    let _ = job.started.send(());
                                    let result = gpu.solve(&job.challenge);
                                    active_work.store(0, Ordering::Relaxed);
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
        let (hashx_gpu_tx, hashx_gpu_operational) = if config.gpu_enabled {
            match HashxGpuSolver::create().await {
                Ok(mut gpu) => {
                    info!(adapter = %gpu.adapter(), "HashX GPU backend ready");
                    let (tx, mut rx) = mpsc::channel::<HashxGpuJob>(128);
                    tokio::spawn(async move {
                        while let Some(job) = rx.recv().await {
                            let result = gpu.solve(&job.challenge);
                            let _ = job.result.send(result);
                        }
                    });
                    (Some(tx), true)
                }
                Err(err) => {
                    warn!(%err, "HashX GPU unavailable");
                    (None, false)
                }
            }
        } else {
            (None, false)
        };

        #[cfg(not(feature = "gpu"))]
        let hashx_gpu_operational = false;

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
            gpu_operational,
            hashx_gpu_operational,
            gpu_queue_work: queue_work,
            gpu_queue_depth: queue_depth,
            gpu_active_work: active_work,
            cpu_slots: Arc::new(Semaphore::new(config.cpu_workers)),
            stats: RwLock::new(StatsStore::default()),
        })
    }

    async fn run_cpu(
        &self,
        challenge: Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let permit = self.cpu_slots.clone().acquire_owned().await?;
        let start = Instant::now();
        let queue_ms = start.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            anubis::solve_cpu(&challenge, 4_000_000_000)
        })
        .await??;
        Ok(SolveResponse {
            solver: "anubis",
            backend: "cpu",
            answer: result.answer,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn run_gpu_only(
        &self,
        challenge: Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        anyhow::ensure!(
            challenge.gpu_compatible(),
            "challenge shape is not GPU-compatible"
        );
        let tx = self
            .gpu_tx
            .as_ref()
            .context("SHA GPU backend unavailable")?;
        let expected_work = challenge.expected_work();
        let (started_tx, started_rx) = oneshot::channel();
        let (result_tx, result_rx) = oneshot::channel();
        self.gpu_queue_work
            .fetch_add(expected_work, Ordering::Relaxed);
        self.gpu_queue_depth.fetch_add(1, Ordering::Relaxed);
        if tx
            .send(GpuJob {
                challenge,
                expected_work,
                started: started_tx,
                result: result_tx,
            })
            .await
            .is_err()
        {
            self.gpu_queue_work
                .fetch_sub(expected_work, Ordering::Relaxed);
            self.gpu_queue_depth.fetch_sub(1, Ordering::Relaxed);
            anyhow::bail!("SHA GPU worker unavailable");
        }
        started_rx
            .await
            .map_err(|_| anyhow::anyhow!("SHA GPU worker dropped start signal"))?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("SHA GPU worker dropped result"))??;
        Ok(SolveResponse {
            solver: "anubis-gpu",
            backend: "gpu",
            answer: result.answer,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn run_hashx_simd(
        &self,
        challenge: hashx::Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        let worker_count = self.config.cpu_workers;
        let permits = u32::try_from(worker_count).context("too many CPU workers")?;
        let permit = self.cpu_slots.clone().acquire_many_owned(permits).await?;
        let start = Instant::now();
        let queue_ms = start.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            hashx_cpu_simd::solve(&challenge, worker_count)
        })
        .await??;
        Ok(SolveResponse {
            solver: "anubis-hashx-simd",
            backend: "cpu-simd",
            answer: result.answer,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn run_hashx_gpu(
        &self,
        challenge: hashx::Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        let tx = self
            .hashx_gpu_tx
            .as_ref()
            .context("HashX GPU backend unavailable")?;
        let (result_tx, result_rx) = oneshot::channel();
        tx.send(HashxGpuJob {
            challenge,
            result: result_tx,
        })
        .await
        .map_err(|_| anyhow::anyhow!("HashX GPU worker unavailable"))?;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("HashX GPU worker dropped result"))??;
        Ok(SolveResponse {
            solver: "anubis-hashx",
            backend: "gpu",
            answer: result.answer,
            queue_ms: 0.0,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn route_anubis(&self, challenge: Challenge) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        let queued_at = Instant::now();

        #[cfg(feature = "gpu")]
        if let Some(tx) = self.gpu_tx.as_ref().filter(|_| challenge.gpu_compatible()) {
            let expected_work = challenge.expected_work();
            let (started_tx, started_rx) = oneshot::channel();
            let (result_tx, result_rx) = oneshot::channel();
            self.gpu_queue_work
                .fetch_add(expected_work, Ordering::Relaxed);
            self.gpu_queue_depth.fetch_add(1, Ordering::Relaxed);

            if tx
                .send(GpuJob {
                    challenge: challenge.clone(),
                    expected_work,
                    started: started_tx,
                    result: result_tx,
                })
                .await
                .is_err()
            {
                self.gpu_queue_work
                    .fetch_sub(expected_work, Ordering::Relaxed);
                self.gpu_queue_depth.fetch_sub(1, Ordering::Relaxed);
                warn!("GPU worker unavailable; falling back to CPU");
                return self.run_cpu(challenge, queued_at).await;
            }

            started_rx
                .await
                .map_err(|_| anyhow::anyhow!("GPU worker dropped start signal"))?;
            let started = Instant::now();
            let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
            return match result_rx
                .await
                .map_err(|_| anyhow::anyhow!("GPU worker dropped result"))?
            {
                Ok(result) => Ok(SolveResponse {
                    solver: "anubis",
                    backend: "gpu",
                    answer: result.answer,
                    queue_ms,
                    solve_ms: result.solve_ms,
                    total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
                }),
                Err(err) => {
                    warn!(%err, "GPU solve failed; falling back to CPU");
                    self.run_cpu(challenge, queued_at).await
                }
            };
        }

        self.run_cpu(challenge, queued_at).await
    }
}

pub async fn solve(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<SolveResponse> {
    let queued_at = Instant::now();
    let (difficulty, result) = match request.solver.as_str() {
        "anubis" | "anubis-cpu" | "anubis-gpu" => {
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
                _ => state.route_anubis(challenge).await,
            };
            (difficulty, result)
        }
        "anubis-hashx" | "anubis-hashx-simd" => {
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
            let result = if request.solver == "anubis-hashx-simd" {
                state.run_hashx_simd(challenge, queued_at).await
            } else {
                #[cfg(feature = "gpu")]
                {
                    state.run_hashx_gpu(challenge, queued_at).await
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow::anyhow!("HashX GPU backend not compiled"))
                }
            };
            (difficulty, result)
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
            response.solver = match request.solver.as_str() {
                "anubis-cpu" => "anubis-cpu",
                "anubis-gpu" => "anubis-gpu",
                "anubis-hashx" => "anubis-hashx",
                "anubis-hashx-simd" => "anubis-hashx-simd",
                _ => "anubis",
            };
            state
                .stats
                .write()
                .await
                .record(difficulty, response.backend, response.total_ms);
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

#[derive(Serialize)]
pub struct StatsResponse {
    available_solvers: [&'static str; 5],
    calibration: Calibration,
    config: StatsConfig,
    gpu_queue_depth: u64,
    gpu_queue_expected_work: u64,
    gpu_active_expected_work: u64,
    cpu_workers_available: usize,
    expected_new_solve_ms_by_difficulty: BTreeMap<u8, f64>,
    general: crate::stats::GeneralStats,
}

#[derive(Serialize)]
pub struct StatsConfig {
    cpu_workers: usize,
    gpu_compiled: bool,
    gpu_enabled: bool,
    gpu_operational: bool,
    hashx_gpu_operational: bool,
}

pub async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let queued_work = state.gpu_queue_work.load(Ordering::Relaxed);
    let gpu_queue_depth = state.gpu_queue_depth.load(Ordering::Relaxed);
    let active_work = state.gpu_active_work.load(Ordering::Relaxed);
    let work_ahead = queued_work.saturating_add(active_work);
    let mut expected = BTreeMap::new();
    for difficulty in 1..=16_u8 {
        let work = 16_u64.saturating_pow(u32::from(difficulty)) as f64;
        let eta_ms = if let Some(gpu) = state.calibration.gpu_hashes_per_second {
            (work_ahead as f64 + work) / gpu * 1000.0
        } else {
            work / state.calibration.cpu_hashes_per_second_per_worker * 1000.0
        };
        expected.insert(difficulty, eta_ms);
    }
    let general = state.stats.read().await.snapshot();
    Json(StatsResponse {
        available_solvers: [
            "anubis",
            "anubis-cpu",
            "anubis-gpu",
            "anubis-hashx",
            "anubis-hashx-simd",
        ],
        calibration: state.calibration.clone(),
        config: StatsConfig {
            cpu_workers: state.config.cpu_workers,
            gpu_compiled: cfg!(feature = "gpu"),
            gpu_enabled: state.config.gpu_enabled,
            gpu_operational: state.gpu_operational,
            hashx_gpu_operational: state.hashx_gpu_operational,
        },
        gpu_queue_depth,
        gpu_queue_expected_work: queued_work,
        gpu_active_expected_work: active_work,
        cpu_workers_available: state.cpu_slots.available_permits(),
        expected_new_solve_ms_by_difficulty: expected,
        general,
    })
}
