use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, Semaphore, mpsc, oneshot};
use tracing::{info, warn};

use crate::{
    anubis::{self, Challenge},
    gpu::GpuSolver,
    stats::{Calibration, StatsStore},
};

#[derive(Debug, Clone)]
pub struct Config {
    pub spillover_ms: u64,
    pub cpu_workers: usize,
    pub cpu_calibration_hashes: u64,
    pub gpu_calibration_hashes: u64,
    pub gpu_enabled: bool,
}

struct GpuJob {
    challenge: Challenge,
    expected_work: u64,
    cancelled: Arc<AtomicBool>,
    started: oneshot::Sender<()>,
    result: oneshot::Sender<anyhow::Result<anubis::BackendResult>>,
}

pub struct AppState {
    config: Config,
    calibration: Calibration,
    gpu_tx: Option<mpsc::Sender<GpuJob>>,
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
        let (gpu_tx, gpu_rate, gpu_adapter) = if config.gpu_enabled {
            match GpuSolver::create().await {
                Ok(mut gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    match gpu.calibrate(config.gpu_calibration_hashes) {
                        Ok(rate) => {
                            info!(adapter = %adapter, mh_s = rate / 1e6, "GPU calibration complete");
                            let (tx, mut rx) = mpsc::channel::<GpuJob>(1024);
                            let qw = queue_work.clone();
                            let qd = queue_depth.clone();
                            tokio::spawn(async move {
                                while let Some(job) = rx.recv().await {
                                    qd.fetch_sub(1, Ordering::Relaxed);
                                    qw.fetch_sub(job.expected_work, Ordering::Relaxed);
                                    if job.cancelled.load(Ordering::Acquire) {
                                        let _ = job.result.send(Err(anyhow::anyhow!(
                                            "cancelled before GPU start"
                                        )));
                                        continue;
                                    }
                                    let _ = job.started.send(());
                                    let result = gpu.solve(&job.challenge);
                                    let _ = job.result.send(result);
                                }
                            });
                            (Some(tx), Some(rate), Some(adapter))
                        }
                        Err(err) => {
                            warn!(%err, "GPU calibration failed; CPU-only");
                            (None, None, Some(adapter))
                        }
                    }
                }
                Err(err) => {
                    warn!(%err, "GPU unavailable; CPU-only");
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };
        Ok(Self {
            config: config.clone(),
            calibration: Calibration {
                gpu_hashes_per_second: gpu_rate,
                cpu_hashes_per_second_per_worker: cpu_hashes_per_second,
                gpu_adapter,
            },
            gpu_tx,
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

    async fn route_anubis(&self, challenge: Challenge) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        let queued_at = Instant::now();
        let Some(tx) = self.gpu_tx.as_ref().filter(|_| challenge.gpu_compatible()) else {
            return self.run_cpu(challenge, queued_at).await;
        };
        let expected = challenge.expected_work();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (started_tx, mut started_rx) = oneshot::channel();
        let (result_tx, result_rx) = oneshot::channel();
        self.gpu_queue_work.fetch_add(expected, Ordering::Relaxed);
        self.gpu_queue_depth.fetch_add(1, Ordering::Relaxed);
        if tx
            .send(GpuJob {
                challenge: challenge.clone(),
                expected_work: expected,
                cancelled: cancelled.clone(),
                started: started_tx,
                result: result_tx,
            })
            .await
            .is_err()
        {
            self.gpu_queue_work.fetch_sub(expected, Ordering::Relaxed);
            self.gpu_queue_depth.fetch_sub(1, Ordering::Relaxed);
            return self.run_cpu(challenge, queued_at).await;
        }
        let spill = tokio::time::sleep(std::time::Duration::from_millis(self.config.spillover_ms));
        tokio::pin!(spill);
        tokio::select! {
            started = &mut started_rx => {
                started.map_err(|_| anyhow::anyhow!("GPU worker dropped start signal"))?;
                let start=Instant::now(); let queue_ms=start.duration_since(queued_at).as_secs_f64()*1000.0;
                match result_rx.await.map_err(|_| anyhow::anyhow!("GPU worker dropped result"))? {
                    Ok(result)=>Ok(SolveResponse{solver:"anubis",backend:"gpu",answer:result.answer,queue_ms,solve_ms:result.solve_ms,total_ms:queued_at.elapsed().as_secs_f64()*1000.0}),
                    Err(err)=>{ warn!(%err,"GPU solve failed; falling back to CPU"); self.run_cpu(challenge,queued_at).await }
                }
            }
            () = &mut spill => {
                if started_rx.try_recv().is_ok() {
                    let start=Instant::now(); let queue_ms=start.duration_since(queued_at).as_secs_f64()*1000.0;
                    match result_rx.await.map_err(|_| anyhow::anyhow!("GPU worker dropped result"))? {
                        Ok(result)=>Ok(SolveResponse{solver:"anubis",backend:"gpu",answer:result.answer,queue_ms,solve_ms:result.solve_ms,total_ms:queued_at.elapsed().as_secs_f64()*1000.0}),
                        Err(_)=>self.run_cpu(challenge,queued_at).await,
                    }
                } else if self.cpu_slots.available_permits()>0 {
                    cancelled.store(true, Ordering::Release); self.run_cpu(challenge,queued_at).await
                } else {
                    started_rx.await.map_err(|_| anyhow::anyhow!("GPU worker dropped start signal"))?;
                    let start=Instant::now(); let queue_ms=start.duration_since(queued_at).as_secs_f64()*1000.0;
                    let result=result_rx.await.map_err(|_| anyhow::anyhow!("GPU worker dropped result"))??;
                    Ok(SolveResponse{solver:"anubis",backend:"gpu",answer:result.answer,queue_ms,solve_ms:result.solve_ms,total_ms:queued_at.elapsed().as_secs_f64()*1000.0})
                }
            }
        }
    }
}

pub async fn solve(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SolveRequest>,
) -> ApiResult<SolveResponse> {
    if request.solver != "anubis" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("unknown solver: {}", request.solver),
            }),
        ));
    }
    let challenge: Challenge = serde_json::from_value(request.challenge).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    let difficulty = challenge.difficulty;
    match state.route_anubis(challenge).await {
        Ok(response) => {
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
    available_solvers: [&'static str; 1],
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
    cpu_spillover_ms: u64,
    cpu_fallback_workers: usize,
    gpu_enabled: bool,
}

pub async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let queued_work = state.gpu_queue_work.load(Ordering::Relaxed);
    let gpu_queue_depth = state.gpu_queue_depth.load(Ordering::Relaxed);
    let active_work = state.gpu_active_work.load(Ordering::Relaxed);
    let work_ahead = queued_work.saturating_add(active_work);
    let mut expected = BTreeMap::new();
    for d in 1..=16_u8 {
        let work = 16_u64.saturating_pow(u32::from(d)) as f64;
        let cpu_ms = work / state.calibration.cpu_hashes_per_second_per_worker * 1000.0;
        let eta = if let Some(gpu) = state.calibration.gpu_hashes_per_second {
            let gpu_ms = (work_ahead as f64 + work) / gpu * 1000.0;
            if gpu_ms > state.config.spillover_ms as f64 && state.cpu_slots.available_permits() > 0
            {
                gpu_ms.min(state.config.spillover_ms as f64 + cpu_ms)
            } else {
                gpu_ms
            }
        } else {
            cpu_ms
        };
        expected.insert(d, eta);
    }
    let general = state.stats.read().await.snapshot();
    Json(StatsResponse {
        available_solvers: ["anubis"],
        calibration: state.calibration.clone(),
        config: StatsConfig {
            cpu_spillover_ms: state.config.spillover_ms,
            cpu_fallback_workers: state.config.cpu_workers,
            gpu_enabled: state.config.gpu_enabled,
        },
        gpu_queue_depth,
        gpu_queue_expected_work: queued_work,
        gpu_active_expected_work: active_work,
        cpu_workers_available: state.cpu_slots.available_permits(),
        expected_new_solve_ms_by_difficulty: expected,
        general,
    })
}
