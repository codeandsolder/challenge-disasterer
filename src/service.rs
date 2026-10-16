use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
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
    anubis_wasm, classic_pow, hashx, hashx_cpu_jit,
    routing::{RouteEntry, RoutingCalibration},
    stats::{Calibration, StatsStore},
};
#[cfg(feature = "gpu")]
use crate::{
    gpu::GpuSolver,
    hashx_gpu::{GpuResult as HashxGpuResult, GpuSolver as HashxGpuSolver},
};

#[derive(Debug, Clone)]
pub struct Config {
    pub cpu_workers: usize,
    pub cpu_calibration_hashes: u64,
    #[cfg(feature = "gpu")]
    pub gpu_calibration_hashes: u64,
    pub gpu_enabled: bool,
    pub routing_calibration: Option<PathBuf>,
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
    implementation: Option<String>,
    result: oneshot::Sender<anyhow::Result<HashxGpuResult>>,
}

#[cfg(feature = "gpu")]
struct GeneralGpuJob {
    solver: String,
    challenge: Value,
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
    hashx_gpu_implementations: Vec<&'static str>,
    gpu_queue_work: Arc<AtomicU64>,
    gpu_queue_depth: Arc<AtomicU64>,
    gpu_active_work: Arc<AtomicU64>,
    cpu_slots: Arc<Semaphore>,
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

#[cfg(feature = "gpu")]
async fn solve_general_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
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
    classic_pow::solve_gpu(gpu, solver, challenge).await
}

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
        let (hashx_gpu_tx, hashx_gpu_operational, hashx_gpu_implementations) = if config.gpu_enabled
        {
            match HashxGpuSolver::create().await {
                Ok(mut gpu) => {
                    let implementations = gpu.implementations();
                    info!(adapter = %gpu.adapter(), ?implementations, "HashX GPU backend ready");
                    let (tx, mut rx) = mpsc::channel::<HashxGpuJob>(128);
                    tokio::spawn(async move {
                        while let Some(job) = rx.recv().await {
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
        let (general_gpu_tx, general_gpu_operational) = if config.gpu_enabled {
            match pow_buster::gpu::GpuContext::create().await {
                Ok(mut gpu) => {
                    let adapter = gpu.adapter().to_owned();
                    match gpu.warm_up().await {
                        Ok(()) => {
                            info!(adapter = %adapter, "general WGPU backend ready");
                            let (tx, mut rx) = mpsc::channel::<GeneralGpuJob>(128);
                            tokio::spawn(async move {
                                while let Some(job) = rx.recv().await {
                                    let result =
                                        solve_general_gpu(&mut gpu, &job.solver, job.challenge)
                                            .await;
                                    let _ = job.result.send(result);
                                }
                            });
                            (Some(tx), true)
                        }
                        Err(err) => {
                            warn!(%err, "general WGPU warm-up failed");
                            (None, false)
                        }
                    }
                }
                Err(err) => {
                    warn!(%err, "general WGPU unavailable");
                    (None, false)
                }
            }
        } else {
            (None, false)
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
            hashx_gpu_implementations,
            gpu_queue_work: queue_work,
            gpu_queue_depth: queue_depth,
            gpu_active_work: active_work,
            cpu_slots: Arc::new(Semaphore::new(config.cpu_workers)),
            stats: RwLock::new(StatsStore::default()),
            routing,
            http: reqwest::Client::new(),
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
            solver: "anubis".to_owned(),
            backend: "cpu".to_owned(),
            implementation: None,
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn run_anubis_wasm(
        &self,
        solver: String,
        challenge: anubis_wasm::Challenge,
        implementation: Option<String>,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let worker_count = if solver == "anubis-argon2id" {
            self.config.cpu_workers
        } else {
            1
        };
        let permits = u32::try_from(worker_count).context("too many CPU workers")?;
        let permit = self.cpu_slots.clone().acquire_many_owned(permits).await?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let response_solver = solver.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            match solver.as_str() {
                "anubis-sha256" => anubis_wasm::solve_sha256(&challenge, implementation.as_deref()),
                "anubis-argon2id" => {
                    anubis_wasm::solve_argon2id(&challenge, implementation.as_deref(), worker_count)
                }
                _ => anyhow::bail!("unsupported Anubis WASM solver: {solver}"),
            }
        })
        .await??;
        Ok(SolveResponse {
            solver: response_solver,
            backend: "cpu".to_owned(),
            implementation: Some(result.implementation),
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn run_anubis_wasm_gpu(
        &self,
        challenge: anubis_wasm::Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        self.run_general_gpu(
            "anubis-sha256".to_owned(),
            serde_json::to_value(challenge).context("serialize Anubis WASM SHA-256 challenge")?,
            queued_at,
        )
        .await
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
            solver: "anubis-gpu".to_owned(),
            backend: "gpu".to_owned(),
            implementation: None,
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn run_hashx_jit(
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
            hashx_cpu_jit::solve(&challenge, worker_count)
        })
        .await??;
        Ok(SolveResponse {
            solver: "anubis-hashx-jit".to_owned(),
            backend: "cpu-jit".to_owned(),
            implementation: None,
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn run_hashx_gpu(
        &self,
        challenge: hashx::Challenge,
        implementation: Option<String>,
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
            implementation,
            result: result_tx,
        })
        .await
        .map_err(|_| anyhow::anyhow!("HashX GPU worker unavailable"))?;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("HashX GPU worker dropped result"))??;
        Ok(SolveResponse {
            solver: "anubis-hashx".to_owned(),
            backend: "gpu".to_owned(),
            implementation: Some(result.implementation.to_owned()),
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            queue_ms: 0.0,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn run_classic(
        &self,
        solver: String,
        challenge: Value,
        implementation: Option<String>,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let permit = self.cpu_slots.clone().acquire_owned().await?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let response_solver = solver.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            classic_pow::solve(&solver, challenge, implementation.as_deref())
        })
        .await??;
        Ok(SolveResponse {
            solver: response_solver,
            backend: "cpu".to_owned(),
            implementation: Some(result.implementation),
            answer: result.answer,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn run_general_gpu(
        &self,
        solver: String,
        challenge: Value,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let tx = self
            .general_gpu_tx
            .as_ref()
            .context("general WGPU backend unavailable")?;
        let (result_tx, result_rx) = oneshot::channel();
        tx.send(GeneralGpuJob {
            solver: solver.clone(),
            challenge,
            result: result_tx,
        })
        .await
        .map_err(|_| anyhow::anyhow!("general WGPU worker unavailable"))?;
        let started = Instant::now();
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("general WGPU worker dropped result"))??;
        Ok(SolveResponse {
            solver,
            backend: "gpu".to_owned(),
            implementation: Some(result.implementation),
            answer: result.answer,
            queue_ms: started.duration_since(queued_at).as_secs_f64() * 1000.0,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn calibrated_routes(
        &self,
        algorithm: &str,
        difficulty: u32,
        batch_size_hint: usize,
    ) -> Vec<RouteEntry> {
        self.routing.as_ref().map_or_else(Vec::new, |routing| {
            routing.select_candidates(
                algorithm,
                difficulty,
                batch_size_hint,
                self.calibration.gpu_adapter.as_deref(),
            )
        })
    }

    async fn run_external(
        &self,
        route: &RouteEntry,
        public_solver: &str,
        algorithm: &str,
        challenge: Value,
        batch_size_hint: usize,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let base_url = route
            .provider_url
            .as_deref()
            .context("external calibration route has no provider URL")?;
        let mut request = serde_json::json!({
            "solver": route.solver,
            "algorithm": algorithm,
            "batch_size_hint": batch_size_hint,
            "challenge": challenge,
        });
        if let Some(implementation) = route.implementation.as_deref() {
            request["implementation"] = serde_json::json!(implementation);
        }
        let started = Instant::now();
        let timeout = Duration::from_secs_f64((route.makespan_ms / 1000.0 * 4.0).clamp(5.0, 300.0));
        let response = self
            .http
            .post(format!("{base_url}/solve"))
            .timeout(timeout)
            .json(&request)
            .send()
            .await
            .with_context(|| format!("call external solver {}", route.provider))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .context("read external solver response")?;
        anyhow::ensure!(
            status.is_success(),
            "external solver {} returned {status}: {body}",
            route.provider
        );
        let external: ExternalSolveResponse =
            serde_json::from_str(&body).context("decode external solver response")?;
        let backend = if external.backend.is_empty() {
            format!("external:{}", route.provider)
        } else {
            format!("external:{}:{}", route.provider, external.backend)
        };
        Ok(SolveResponse {
            solver: public_solver.to_owned(),
            backend,
            implementation: external
                .implementation
                .or_else(|| route.implementation.clone()),
            answer: external.answer,
            queue_ms: started.duration_since(queued_at).as_secs_f64() * 1000.0,
            solve_ms: external.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn route_hashx(
        &self,
        challenge: hashx::Challenge,
        batch_size_hint: usize,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        for route in self.calibrated_routes("hashx", challenge.difficulty, batch_size_hint) {
            if route.provider == "local" {
                match route.solver.as_str() {
                    "anubis-hashx-jit" => {
                        match self.run_hashx_jit(challenge.clone(), queued_at).await {
                            Ok(response) => return Ok(response),
                            Err(err) => {
                                warn!(solver = %route.solver, %err, "calibrated local HashX route failed; trying next candidate");
                            }
                        }
                    }
                    "anubis-hashx" =>
                    {
                        #[cfg(feature = "gpu")]
                        if self.hashx_gpu_operational {
                            match self
                                .run_hashx_gpu(
                                    challenge.clone(),
                                    route.implementation.clone(),
                                    queued_at,
                                )
                                .await
                            {
                                Ok(response) => return Ok(response),
                                Err(err) => {
                                    warn!(solver = %route.solver, implementation = ?route.implementation, %err, "calibrated local HashX route failed; trying next candidate");
                                }
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                let challenge_value = serde_json::json!({
                    "data": &challenge.data,
                    "difficulty": challenge.difficulty,
                });
                match self
                    .run_external(
                        &route,
                        "anubis-hashx-auto",
                        "hashx",
                        challenge_value,
                        batch_size_hint,
                        queued_at,
                    )
                    .await
                {
                    Ok(response) => {
                        let verified =
                            serde_json::from_value::<anubis::Answer>(response.answer.clone())
                                .context("decode external HashX answer")
                                .and_then(|answer| hashx::verify(&challenge, &answer));
                        match verified {
                            Ok(true) => return Ok(response),
                            Ok(false) => {
                                warn!(provider = %route.provider, "external HashX solver returned invalid answer; trying next calibrated candidate");
                            }
                            Err(err) => {
                                warn!(provider = %route.provider, %err, "external HashX answer validation failed; trying next calibrated candidate");
                            }
                        }
                    }
                    Err(err) => {
                        warn!(provider = %route.provider, %err, "external HashX solver failed; trying next calibrated candidate");
                    }
                }
            }
        }

        #[cfg(feature = "gpu")]
        if self.hashx_gpu_operational {
            return self.run_hashx_gpu(challenge, None, queued_at).await;
        }
        self.run_hashx_jit(challenge, queued_at).await
    }

    async fn route_anubis(
        &self,
        algorithm: &str,
        challenge: Challenge,
        batch_size_hint: usize,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        for route in
            self.calibrated_routes(algorithm, u32::from(challenge.difficulty), batch_size_hint)
        {
            if route.provider == "local" {
                match route.solver.as_str() {
                    "anubis-cpu" => match self.run_cpu(challenge.clone(), queued_at).await {
                        Ok(response) => return Ok(response),
                        Err(err) => {
                            warn!(solver = %route.solver, %err, "calibrated local Anubis route failed; trying next candidate");
                        }
                    },
                    "anubis-gpu" =>
                    {
                        #[cfg(feature = "gpu")]
                        if self.gpu_operational && challenge.gpu_compatible() {
                            match self.run_gpu_only(challenge.clone(), queued_at).await {
                                Ok(response) => return Ok(response),
                                Err(err) => {
                                    warn!(solver = %route.solver, %err, "calibrated local Anubis route failed; trying next candidate");
                                }
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                let challenge_value = serde_json::json!({
                    "data": &challenge.data,
                    "difficulty": challenge.difficulty,
                });
                match self
                    .run_external(
                        &route,
                        "anubis",
                        algorithm,
                        challenge_value,
                        batch_size_hint,
                        queued_at,
                    )
                    .await
                {
                    Ok(response) => {
                        let verified =
                            serde_json::from_value::<anubis::Answer>(response.answer.clone())
                                .is_ok_and(|answer| anubis::verify(&challenge, &answer));
                        if verified {
                            return Ok(response);
                        }
                        warn!(provider = %route.provider, "external Anubis solver returned invalid answer; trying next calibrated candidate");
                    }
                    Err(err) => {
                        warn!(provider = %route.provider, %err, "external Anubis solver failed; trying next calibrated candidate");
                    }
                }
            }
        }

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
                    solver: "anubis".to_owned(),
                    backend: "gpu".to_owned(),
                    implementation: None,
                    answer: serde_json::to_value(result.answer)
                        .context("serialize solver answer")?,
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
        "cerberus" | "goaway" | "capjs" | "mcaptcha" | "haphash" | "altcha" => {
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
            let mut stats = state.stats.write().await;
            if request.solver.starts_with("anubis") {
                stats.record(difficulty, &response.backend, response.total_ms);
            } else {
                stats.record_generic(&response.backend);
            }
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

#[derive(Debug, Serialize)]
pub struct SolverCapability {
    solver: &'static str,
    algorithms: &'static [&'static str],
    difficulty_min: u32,
    difficulty_max: u32,
    router: bool,
    implementations: Vec<&'static str>,
}

#[derive(Serialize)]
pub struct StatsResponse {
    available_solvers: Vec<&'static str>,
    solver_capabilities: Vec<SolverCapability>,
    calibration: Calibration,
    config: StatsConfig,
    gpu_queue_depth: u64,
    gpu_queue_expected_work: u64,
    gpu_active_expected_work: u64,
    cpu_workers_available: usize,
    expected_new_solve_ms_by_difficulty: BTreeMap<u8, f64>,
    hashx_gpu_implementations: Vec<&'static str>,
    general: crate::stats::GeneralStats,
}

#[derive(Serialize)]
pub struct StatsConfig {
    cpu_workers: usize,
    gpu_compiled: bool,
    gpu_enabled: bool,
    gpu_operational: bool,
    hashx_gpu_operational: bool,
    routing_calibration_loaded: bool,
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
    let mut anubis_sha_implementations = vec!["auto", "safe"];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        anubis_sha_implementations.push("avx2");
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        anubis_sha_implementations.push("avx512");
    }
    if state.general_gpu_operational {
        anubis_sha_implementations.push("wgpu");
    }
    let mut available_solvers = vec![
        "anubis",
        "anubis-cpu",
        "anubis-sha256",
        "anubis-argon2id",
        "anubis-hashx-jit",
        "anubis-hashx-auto",
    ];
    let mut solver_capabilities = vec![
        SolverCapability {
            solver: "anubis",
            algorithms: &["fast", "slow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: true,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-cpu",
            algorithms: &["fast", "slow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-sha256",
            algorithms: &["sha256"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            implementations: anubis_sha_implementations,
        },
        SolverCapability {
            solver: "anubis-argon2id",
            algorithms: &["argon2id"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            implementations: vec!["argon2-0.5.3"],
        },
        SolverCapability {
            solver: "anubis-hashx-jit",
            algorithms: &["hashx"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-hashx-auto",
            algorithms: &["hashx"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: true,
            implementations: Vec::new(),
        },
    ];
    for (solver, algorithms, difficulty_max, sha_ni) in [
        (
            "cerberus",
            &["blake3"] as &'static [&'static str],
            64,
            false,
        ),
        ("goaway", &["sha256"], 32, true),
        ("capjs", &["sha256"], 64, true),
        ("mcaptcha", &["sha256"], 64, true),
        ("haphash", &["sha256"], 16, true),
        ("altcha", &["sha256", "pbkdf2-sha256"], 64, false),
    ] {
        let mut implementations = vec!["auto", "safe"];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                implementations.push("avx2");
            }
            if sha_ni && std::arch::is_x86_feature_detected!("sha") {
                implementations.push("sha-ni");
            }
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            implementations.push("avx512");
        }
        if state.general_gpu_operational {
            implementations.push("wgpu");
        }
        available_solvers.push(solver);
        solver_capabilities.push(SolverCapability {
            solver,
            algorithms,
            difficulty_min: 1,
            difficulty_max,
            router: false,
            implementations,
        });
    }
    if state.gpu_operational {
        available_solvers.push("anubis-gpu");
        solver_capabilities.push(SolverCapability {
            solver: "anubis-gpu",
            algorithms: &["fast", "slow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            implementations: Vec::new(),
        });
    }
    if state.hashx_gpu_operational {
        available_solvers.push("anubis-hashx");
        solver_capabilities.push(SolverCapability {
            solver: "anubis-hashx",
            algorithms: &["hashx"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: true,
            implementations: state.hashx_gpu_implementations.clone(),
        });
    }
    Json(StatsResponse {
        available_solvers,
        solver_capabilities,
        calibration: state.calibration.clone(),
        config: StatsConfig {
            cpu_workers: state.config.cpu_workers,
            gpu_compiled: cfg!(feature = "gpu"),
            gpu_enabled: state.config.gpu_enabled,
            gpu_operational: state.gpu_operational,
            hashx_gpu_operational: state.hashx_gpu_operational,
            routing_calibration_loaded: state.routing.is_some(),
        },
        gpu_queue_depth,
        gpu_queue_expected_work: queued_work,
        gpu_active_expected_work: active_work,
        cpu_workers_available: state.cpu_slots.available_permits(),
        expected_new_solve_ms_by_difficulty: expected,
        hashx_gpu_implementations: state.hashx_gpu_implementations.clone(),
        general,
    })
}
