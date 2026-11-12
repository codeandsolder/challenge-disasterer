//! Local CPU/GPU execution and external-provider calls, independent of HTTP dispatch.

#[cfg(feature = "gpu")]
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::Value;
#[cfg(feature = "gpu")]
use tokio::sync::{OwnedSemaphorePermit, oneshot};

use super::{
    AppState, ExternalSolveResponse, SolveResponse,
    kdf_schedule::{Reservation, altcha_argon2_workers},
};
#[cfg(feature = "gpu")]
use super::{GeneralGpuJob, GpuJob, HashxGpuJob};
use crate::{
    anubis::{self, Challenge},
    anubis_wasm, classic_pow, hashx, hashx_cpu_jit,
    routing::RouteEntry,
};

/// Prefer GPU compute for supported independent-hash puzzles, including
/// SHA-1, SHA-256, SHA-384/512 and Cerberus BLAKE3, regardless of latency crossover. Unsupported
/// shapes and explicitly selected CPU implementations retain CPU routing.
#[cfg(feature = "gpu")]
fn prefers_auto_gpu(solver: &str, challenge: &Value, implementation: Option<&str>) -> bool {
    if implementation.is_some_and(|value| value != "auto") {
        return false;
    }
    match solver {
        "altcha" | "linear" => classic_pow::auto_gpu_expected_work(solver, challenge).is_some(),
        // These classic families' GPU implementations compute SHA-256 work.
        // They validate the actual challenge with their existing parsers;
        // unsupported shapes transparently fall back to CPU.
        "cerberus" | "goaway" | "haphash" | "mcaptcha" => true,
        "capjs" => {
            challenge.get("format").and_then(Value::as_u64) != Some(2)
                || challenge
                    .get("challenges")
                    .and_then(Value::as_array)
                    .is_some_and(|challenges| {
                        // Offload SHA-256 even in mixed format-2 lists. The
                        // existing protocol adapter executes RSW and HashWX
                        // with their native CPU implementations.
                        challenges.iter().any(|entry| {
                            entry.get("protocol").and_then(Value::as_str) == Some("sha256-pow")
                        })
                    })
        }
        _ => false,
    }
}

/// Optional native-u64 GPU requirement for the complete SHA-512 compression
/// family, including its PBKDF2-HMAC modes. Other families can use baseline
/// GPU features, including the existing SHA-256 path.
#[cfg(feature = "gpu")]
fn requires_native_u64(solver: &str, challenge: &Value) -> bool {
    if solver != "altcha" {
        return false;
    }
    challenge
        .pointer("/parameters/algorithm")
        .or_else(|| challenge.get("algorithm"))
        .and_then(Value::as_str)
        .is_some_and(|algorithm| {
            let normalized = algorithm.to_ascii_uppercase();
            let digest = normalized.strip_prefix("PBKDF2/").unwrap_or(&normalized);
            matches!(digest, "SHA-384" | "SHA-512")
        })
}

/// A format-2 Cap.js request with both GPU SHA-256 and CPU-only protocols.
/// Split by protocol, preserving the original response ordering.
#[cfg(feature = "gpu")]
struct MixedCapJs {
    sha: Value,
    cpu: Value,
    sha_positions: Vec<bool>,
}

#[cfg(feature = "gpu")]
fn split_mixed_capjs(challenge: &Value) -> anyhow::Result<Option<MixedCapJs>> {
    if challenge.get("format").and_then(Value::as_u64) != Some(2) {
        return Ok(None);
    }
    // Validate the entire envelope before any SHA-only GPU partition runs.
    // Without this, browser-only instrumentation in the CPU partition could
    // be detected only after a preceding GPU proof-of-work had completed.
    classic_pow::preflight_capjs_format_two(challenge)?;
    let entries = challenge
        .get("challenges")
        .and_then(Value::as_array)
        .context("validated Cap.js format-2 envelope lacks challenges")?;
    let sha_positions: Vec<bool> = entries
        .iter()
        .map(|entry| entry.get("protocol").and_then(Value::as_str) == Some("sha256-pow"))
        .collect();
    if !sha_positions.iter().any(|is_sha| *is_sha) || sha_positions.iter().all(|is_sha| *is_sha) {
        return Ok(None);
    }
    let mut sha = challenge.clone();
    let mut cpu = challenge.clone();
    sha["challenges"] = Value::Array(
        entries
            .iter()
            .zip(&sha_positions)
            .filter(|(_, is_sha)| **is_sha)
            .map(|(entry, _)| entry.clone())
            .collect(),
    );
    cpu["challenges"] = Value::Array(
        entries
            .iter()
            .zip(&sha_positions)
            .filter(|(_, is_sha)| !**is_sha)
            .map(|(entry, _)| entry.clone())
            .collect(),
    );
    Ok(Some(MixedCapJs {
        sha,
        cpu,
        sha_positions,
    }))
}

#[cfg(feature = "gpu")]
fn merge_mixed_capjs(
    mixed: &MixedCapJs,
    gpu: SolveResponse,
    cpu: &SolveResponse,
    queued_at: Instant,
) -> anyhow::Result<SolveResponse> {
    let mut sha_solutions = gpu
        .answer
        .get("solutions")
        .and_then(Value::as_array)
        .context("GPU Cap.js response lacks solutions")?
        .iter();
    let mut cpu_solutions = cpu
        .answer
        .get("solutions")
        .and_then(Value::as_array)
        .context("CPU Cap.js response lacks solutions")?
        .iter();
    let solutions: Vec<Value> = mixed
        .sha_positions
        .iter()
        .map(|is_sha| {
            if *is_sha {
                sha_solutions.next()
            } else {
                cpu_solutions.next()
            }
            .cloned()
            .context("Cap.js partial response lost a subchallenge")
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(
        sha_solutions.next().is_none() && cpu_solutions.next().is_none(),
        "Cap.js partial response had extra subchallenge solutions"
    );
    anyhow::ensure!(
        gpu.answer.get("token") == cpu.answer.get("token"),
        "Cap.js partial responses have mismatched tokens"
    );
    let attempts = gpu
        .answer
        .get("attempted_nonces_total")
        .and_then(Value::as_u64)
        .context("GPU Cap.js response lacks attempted count")?
        .checked_add(
            cpu.answer
                .get("attempted_nonces_total")
                .and_then(Value::as_u64)
                .context("CPU Cap.js response lacks attempted count")?,
        )
        .context("Cap.js combined attempted count overflow")?;
    let mut answer = gpu.answer;
    answer["solutions"] = Value::Array(solutions);
    answer["attempted_nonces_total"] = serde_json::json!(attempts);
    let total_ms = queued_at.elapsed().as_secs_f64() * 1000.0;
    // Both partial responses use the same request start. Subtract the GPU
    // stage's elapsed time before accounting for subsequent CPU queueing.
    let queue_ms = gpu.queue_ms + (cpu.queue_ms - gpu.total_ms).max(0.0);
    let solve_ms = gpu.solve_ms + cpu.solve_ms;
    Ok(SolveResponse {
        solver: "capjs".to_owned(),
        backend: "gpu+cpu".to_owned(),
        implementation: Some(format!(
            "{}+{}",
            gpu.implementation.as_deref().unwrap_or("wgpu"),
            cpu.implementation.as_deref().unwrap_or("cpu")
        )),
        answer,
        nonce_workers: cpu.nonce_workers,
        queue_ms,
        solve_ms,
        total_ms,
    })
}

impl AppState {
    /// Automatic GPU placement favors available GPU capacity, not lowest
    /// isolated request latency. A single shared permit serializes this service's GPU
    /// worker families; the default waits rather than scheduling CPU work.
    #[cfg(feature = "gpu")]
    pub(super) async fn reserve_auto_gpu(&self) -> Option<OwnedSemaphorePermit> {
        match self.config.gpu_policy {
            super::GpuPolicy::PreferGpu => self.gpu_slots.clone().acquire_owned().await.ok(),
            super::GpuPolicy::IdleOnly => self.gpu_slots.clone().try_acquire_owned().ok(),
            super::GpuPolicy::CpuOnly => None,
        }
    }

    pub(super) async fn run_cpu(
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
            nonce_workers: None,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(super) async fn run_anubis_wasm(
        &self,
        solver: String,
        challenge: anubis_wasm::Challenge,
        implementation: Option<String>,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        #[cfg(feature = "gpu")]
        if solver == "anubis-sha256"
            && implementation
                .as_deref()
                .is_none_or(|value| value == "auto")
            && self.general_gpu_operational
        {
            match self
                .try_run_anubis_wasm_gpu(challenge.clone(), queued_at)
                .await
            {
                Ok(Some(response)) => return Ok(response),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(%err, "automatic Anubis SHA-256 GPU failed; routing to CPU");
                }
            }
        }
        let memory_bytes =
            (solver == "anubis-argon2id").then_some(anubis_wasm::ARGON2_MEMORY_BYTES);
        let desired = if memory_bytes.is_some() {
            self.config.cpu_workers_per_request
        } else {
            1
        };
        let reservation = Reservation::acquire(
            &self.cpu_slots,
            self.kdf_memory_slots.as_ref(),
            self.kdf_memory_capacity_mib,
            memory_bytes,
            desired.min(self.config.cpu_workers_per_request),
        )
        .await?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let response_solver = solver.clone();
        let (result, workers) = tokio::task::spawn_blocking(move || {
            let workers = reservation.workers();
            let result = match solver.as_str() {
                "anubis-sha256" => anubis_wasm::solve_sha256(&challenge, implementation.as_deref()),
                "anubis-argon2id" => {
                    anubis_wasm::solve_argon2id(&challenge, implementation.as_deref(), workers)
                }
                _ => anyhow::bail!("unsupported Anubis WASM solver: {solver}"),
            };
            drop(reservation);
            result.map(|value| (value, workers))
        })
        .await??;
        Ok(SolveResponse {
            solver: response_solver,
            backend: "cpu".to_owned(),
            implementation: Some(result.implementation),
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            nonce_workers: memory_bytes.map(|_| workers),
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    pub(super) async fn run_anubis_wasm_gpu(
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

    /// Automatic SHA-256 GPU route. Its wait-vs-fallback behavior comes from
    /// the configured policy; a lost GPU worker still falls back to CPU.
    #[cfg(feature = "gpu")]
    pub(super) async fn try_run_anubis_wasm_gpu(
        &self,
        challenge: anubis_wasm::Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<Option<SolveResponse>> {
        self.dispatch_general_gpu(
            "anubis-sha256".to_owned(),
            serde_json::to_value(challenge).context("serialize Anubis WASM SHA-256 challenge")?,
            queued_at,
            true,
        )
        .await
    }

    #[cfg(feature = "gpu")]
    pub(super) async fn run_gpu_only(
        &self,
        challenge: Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        anyhow::ensure!(
            challenge.gpu_compatible(),
            "challenge shape is not GPU-compatible"
        );
        let permit = self.gpu_slots.clone().acquire_owned().await?;
        self.run_gpu_with_permit(challenge, queued_at, permit).await
    }

    /// Called by the automatic router only after atomically reserving the
    /// idle GPU. Explicit WGPU requests wait for the shared permit instead.
    #[cfg(feature = "gpu")]
    pub(super) async fn run_gpu_with_permit(
        &self,
        challenge: Challenge,
        queued_at: Instant,
        gpu_permit: OwnedSemaphorePermit,
    ) -> anyhow::Result<SolveResponse> {
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
                _gpu_permit: gpu_permit,
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
        let started = started_rx
            .await
            .map_err(|_| anyhow::anyhow!("SHA GPU worker dropped start signal"))?;
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("SHA GPU worker dropped result"))??;
        Ok(SolveResponse {
            solver: "anubis-gpu".to_owned(),
            backend: "gpu".to_owned(),
            implementation: None,
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            nonce_workers: None,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(super) async fn run_hashx_jit(
        &self,
        challenge: hashx::Challenge,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        let reservation = Reservation::acquire(
            &self.cpu_slots,
            None,
            None,
            None,
            self.config.cpu_workers_per_request,
        )
        .await?;
        let start = Instant::now();
        let queue_ms = start.duration_since(queued_at).as_secs_f64() * 1000.0;
        let (result, worker_count) = tokio::task::spawn_blocking(move || {
            let worker_count = reservation.workers();
            let result = hashx_cpu_jit::solve(&challenge, worker_count);
            drop(reservation);
            result.map(|value| (value, worker_count))
        })
        .await??;
        Ok(SolveResponse {
            solver: "anubis-hashx-jit".to_owned(),
            backend: "cpu-jit".to_owned(),
            implementation: None,
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            nonce_workers: Some(worker_count),
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "gpu")]
    pub(super) async fn run_hashx_gpu(
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
        let (started_tx, started_rx) = oneshot::channel();
        let (result_tx, result_rx) = oneshot::channel();
        let gpu_permit = self.gpu_slots.clone().acquire_owned().await?;
        tx.send(HashxGpuJob {
            _gpu_permit: gpu_permit,
            challenge,
            implementation,
            started: started_tx,
            result: result_tx,
        })
        .await
        .map_err(|_| anyhow::anyhow!("HashX GPU worker unavailable"))?;
        let started = started_rx
            .await
            .map_err(|_| anyhow::anyhow!("HashX GPU worker dropped start signal"))?;
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("HashX GPU worker dropped result"))??;
        Ok(SolveResponse {
            solver: "anubis-hashx".to_owned(),
            backend: "gpu".to_owned(),
            implementation: Some(result.implementation.to_owned()),
            answer: serde_json::to_value(result.answer).context("serialize solver answer")?,
            nonce_workers: None,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(super) async fn run_classic(
        &self,
        solver: String,
        challenge: Value,
        implementation: Option<String>,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        #[cfg(feature = "gpu")]
        if self.general_gpu_operational
            && (!requires_native_u64(&solver, &challenge) || self.general_gpu_has_native_u64())
            && prefers_auto_gpu(&solver, &challenge, implementation.as_deref())
        {
            let mixed = if solver == "capjs" {
                split_mixed_capjs(&challenge)?
            } else {
                None
            };
            let gpu_challenge = mixed
                .as_ref()
                .map_or_else(|| challenge.clone(), |parts| parts.sha.clone());
            match self
                .dispatch_general_gpu(solver.clone(), gpu_challenge, queued_at, true)
                .await
            {
                Ok(Some(gpu)) => {
                    if let Some(parts) = mixed {
                        // GPU permit released; CPU-only work never holds the GPU queue.
                        let cpu = self
                            .run_classic_cpu(solver, parts.cpu.clone(), None, queued_at)
                            .await?;
                        return merge_mixed_capjs(&parts, gpu, &cpu, queued_at);
                    }
                    return Ok(gpu);
                }
                Ok(None) => {
                    tracing::debug!(solver, "GPU occupied; routing hash work to CPU");
                }
                Err(err) => {
                    tracing::warn!(solver, %err, "automatic hash WGPU failed; routing to CPU");
                }
            }
        }
        self.run_classic_cpu(solver, challenge, implementation, queued_at)
            .await
    }

    /// Choose Tor v1 Equi-X nonce-search fanout from the client-requested effort.
    ///
    /// Fixed-work benchmarks on Ryzen 4700U/4900HS show that thread startup and
    /// speculative work dominate effort 1-2; above that, wider fanout reduces
    /// latency. These thresholds affect resource allocation, never proof validity.
    /// Resource reservation may return fewer workers if CPU permits are occupied.
    /// Malformed effort values are left to the protocol parser, using one worker
    /// rather than speculatively reserving an arbitrary number of slots.
    pub(super) fn tor_equix_suggested_workers(challenge: &Value, configured: usize) -> usize {
        let effort = challenge
            .get("effort")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let desired = match effort {
            Some(0..=2) | None => 1,
            Some(3..=8) => 2,
            Some(9..=63) => 4,
            Some(64..) => configured,
        };
        desired.min(configured).max(1)
    }

    async fn run_classic_cpu(
        &self,
        solver: String,
        challenge: Value,
        implementation: Option<String>,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let memory_bytes = classic_pow::memory_hard_requirement(&solver, &challenge).transpose()?;
        let altcha_workers =
            altcha_argon2_workers(&solver, &challenge, self.config.cpu_workers_per_request);
        // All memory-hard protocols participate, including Scrypt.
        let in_flight = memory_bytes.map(|_| self.kdf_admission.enter());
        let friendly = solver == "friendlycaptcha";
        let tor_equix = solver == "tor-hs-v1";
        let sequentialread = solver == "pow-bot-deterrent";
        let desired = if friendly {
            classic_pow::friendlycaptcha_suggested_workers(
                &challenge,
                self.config.cpu_workers_per_request,
            )
            .unwrap_or(1)
        } else if tor_equix {
            Self::tor_equix_suggested_workers(&challenge, self.config.cpu_workers_per_request)
        } else if sequentialread {
            classic_pow::pow_bot_deterrent_suggested_workers(
                &challenge,
                self.config.cpu_workers_per_request,
            )?
        } else {
            altcha_workers.unwrap_or(1)
        };
        let reservation = Reservation::acquire(
            &self.cpu_slots,
            self.kdf_memory_slots.as_ref(),
            self.kdf_memory_capacity_mib,
            memory_bytes,
            desired.min(self.config.cpu_workers_per_request),
        )
        .await?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let response_solver = solver.clone();
        let (result, workers) = tokio::task::spawn_blocking(move || {
            let workers = reservation.workers();
            let result = classic_pow::solve(&solver, challenge, implementation.as_deref(), workers);
            // HTTP cancellation must not release resources while
            // the spawned blocking KDF is still executing.
            drop(reservation);
            drop(in_flight);
            (result, workers)
        })
        .await?;
        tracing::debug!(workers, desired, "classic CPU resources reserved");
        let result = result?;
        Ok(SolveResponse {
            solver: response_solver,
            backend: "cpu".to_owned(),
            implementation: Some(result.implementation),
            answer: result.answer,
            nonce_workers: if friendly || tor_equix || sequentialread || altcha_workers.is_some() {
                Some(workers)
            } else {
                None
            },
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Mandatory WGPU route selected explicitly by the client.
    #[cfg(feature = "gpu")]
    pub(super) async fn run_general_gpu(
        &self,
        solver: String,
        challenge: Value,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        let mixed = if solver == "capjs" {
            split_mixed_capjs(&challenge)?
        } else {
            None
        };
        if let Some(parts) = mixed {
            let gpu = self
                .dispatch_general_gpu(solver.clone(), parts.sha.clone(), queued_at, false)
                .await?
                .context("mandatory WGPU dispatch was not enqueued")?;
            // Even explicit WGPU runs CPU-only subchallenges without
            // occupying the GPU worker or its shared permit.
            let cpu = self
                .run_classic_cpu(solver, parts.cpu.clone(), None, queued_at)
                .await?;
            return merge_mixed_capjs(&parts, gpu, &cpu, queued_at);
        }
        self.dispatch_general_gpu(solver, challenge, queued_at, false)
            .await?
            .context("mandatory WGPU dispatch was not enqueued")
    }

    /// Send GPU work using one shared execution permit across all GPU worker
    /// families. Automatic SHA-256 follows the configured GPU preference;
    /// explicit WGPU requests always wait for their turn.
    #[cfg(feature = "gpu")]
    async fn dispatch_general_gpu(
        &self,
        solver: String,
        challenge: Value,
        queued_at: Instant,
        opportunistic: bool,
    ) -> anyhow::Result<Option<SolveResponse>> {
        let tx = self
            .general_gpu_tx
            .as_ref()
            .context("general WGPU backend unavailable")?;
        let (started_tx, started_rx) = oneshot::channel();
        let (result_tx, result_rx) = oneshot::channel();
        let job = GeneralGpuJob {
            _gpu_permit: if opportunistic {
                let Some(permit) = self.reserve_auto_gpu().await else {
                    return Ok(None);
                };
                permit
            } else {
                self.gpu_slots.clone().acquire_owned().await?
            },
            solver: solver.clone(),
            challenge,
            started: started_tx,
            result: result_tx,
        };
        if opportunistic {
            if tx.try_send(job).is_err() {
                return Ok(None);
            }
        } else {
            tx.send(job)
                .await
                .map_err(|_| anyhow::anyhow!("general WGPU worker unavailable"))?;
        }
        let started = started_rx
            .await
            .map_err(|_| anyhow::anyhow!("general WGPU worker dropped start signal"))?;
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let result = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("general WGPU worker dropped result"))??;
        Ok(Some(SolveResponse {
            solver,
            backend: "gpu".to_owned(),
            implementation: Some(result.implementation),
            answer: result.answer,
            nonce_workers: None,
            queue_ms,
            solve_ms: result.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    pub(super) async fn run_external(
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
            nonce_workers: None,
            queue_ms: started.duration_since(queued_at).as_secs_f64() * 1000.0,
            solve_ms: external.solve_ms,
            total_ms: queued_at.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

#[cfg(all(test, feature = "gpu"))]
mod auto_gpu_tests {
    use super::*;

    fn altcha(prefix: &str, algorithm: &str, cost: u32) -> Value {
        serde_json::json!({
            "counterMode": "string",
            "parameters": {
                "algorithm": algorithm,
                "cost": cost,
                "keyLength": 12,
                "keyPrefix": prefix,
                "nonce": "01".repeat(16),
                "salt": "02".repeat(16)
            }
        })
    }

    #[test]
    fn auto_gpu_prefers_all_supported_sha_work_except_explicit_cpu() {
        let cheap = altcha("0", "SHA-256", 2);
        let borderline = altcha("00", "SHA-256", 2);
        let substantial = altcha("000", "SHA-256", 2);
        let expensive_pbkdf2 = altcha("00", "PBKDF2/SHA-256", 16);
        let unsupported = altcha("000000000", "ARGON2ID", 2);
        let sha384 = altcha("000", "SHA-384", 2);
        let sha512 = altcha("000", "SHA-512", 2);
        let hmac384 = altcha("00", "PBKDF2/SHA-384", 2);
        let hmac512 = altcha("00", "PBKDF2/SHA-512", 2);

        assert!(prefers_auto_gpu("altcha", &cheap, None));
        assert!(prefers_auto_gpu("altcha", &borderline, Some("auto")));
        assert!(prefers_auto_gpu("altcha", &substantial, None));
        assert!(prefers_auto_gpu("altcha", &expensive_pbkdf2, Some("auto")));
        assert!(!prefers_auto_gpu("altcha", &expensive_pbkdf2, Some("safe")));
        assert!(!prefers_auto_gpu("altcha", &substantial, Some("sha-ni")));
        assert!(!prefers_auto_gpu("altcha", &unsupported, None));
        assert!(prefers_auto_gpu("altcha", &sha384, None));
        assert!(prefers_auto_gpu("altcha", &sha512, None));
        assert!(prefers_auto_gpu("altcha", &hmac384, None));
        assert!(prefers_auto_gpu("altcha", &hmac512, None));
        for challenge in [&sha384, &sha512, &hmac384, &hmac512] {
            assert!(requires_native_u64("altcha", challenge));
        }
        assert!(!requires_native_u64("altcha", &substantial));
        assert!(!requires_native_u64("altcha", &expensive_pbkdf2));
        assert!(!requires_native_u64("altcha", &unsupported));
        assert!(!requires_native_u64("cerberus", &sha384));
        assert!(prefers_auto_gpu("goaway", &substantial, None));
        assert!(prefers_auto_gpu("mcaptcha", &substantial, Some("auto")));
        assert!(prefers_auto_gpu("haphash", &substantial, None));
        assert!(prefers_auto_gpu("capjs", &substantial, None));
        assert!(prefers_auto_gpu("cerberus", &substantial, None));
        assert!(!prefers_auto_gpu("cerberus", &substantial, Some("safe")));
        let sha256_only = serde_json::json!({
            "format": 2,
            "challenges": [{"protocol": "sha256-pow", "payload": {}}]
        });
        let mixed = serde_json::json!({
            "format": 2,
            "challenges": [
                {"protocol": "sha256-pow", "payload": {}},
                {"protocol": "hashwx", "payload": {}}
            ]
        });
        let mixed_rsw = serde_json::json!({
            "format": 2,
            "challenges": [
                {"protocol": "rsw", "payload": {}},
                {"protocol": "sha256-pow", "payload": {}}
            ]
        });
        let only_hashwx = serde_json::json!({
            "format": 2,
            "challenges": [{"protocol": "hashwx", "payload": {}}]
        });
        let only_rsw = serde_json::json!({
            "format": 2,
            "challenges": [{"protocol": "rsw", "payload": {}}]
        });
        assert!(prefers_auto_gpu("capjs", &sha256_only, None));
        assert!(prefers_auto_gpu("capjs", &mixed, None));
        assert!(prefers_auto_gpu("capjs", &mixed_rsw, None));
        assert!(!prefers_auto_gpu("capjs", &only_hashwx, None));
        assert!(!prefers_auto_gpu("capjs", &only_rsw, None));
        assert!(!prefers_auto_gpu("capjs", &mixed, Some("safe")));
        assert!(!prefers_auto_gpu("goaway", &substantial, Some("sha-ni")));
    }

    #[test]
    fn mixed_capjs_split_and_rejoin_preserves_order_and_attempts() -> anyhow::Result<()> {
        let challenge = serde_json::json!({
            "token": "ordered",
            "format": 2,
            "challenges": [
                {"protocol": "rsw", "payload": {"N": "11", "x": "2", "t": 3}},
                {"protocol": "sha256-pow", "payload": {"salt": "first", "target": "0"}},
                {"protocol": "hashwx", "payload": {"c": "00".repeat(32), "d": 1, "n": 1}},
                {"protocol": "sha256-pow", "payload": {"salt": "second", "target": "0"}}
            ]
        });
        let parts = split_mixed_capjs(&challenge)?.context("mixed Cap.js must split")?;
        assert_eq!(parts.sha["challenges"].as_array().map(Vec::len), Some(2));
        assert_eq!(parts.cpu["challenges"].as_array().map(Vec::len), Some(2));
        assert_eq!(parts.sha["token"], "ordered");
        assert_eq!(parts.cpu["token"], "ordered");
        let response = |answer: Value, backend: &str, name: &str| SolveResponse {
            solver: "capjs".to_owned(),
            backend: backend.to_owned(),
            implementation: Some(name.to_owned()),
            answer,
            nonce_workers: None,
            queue_ms: 0.0,
            solve_ms: 1.0,
            total_ms: 1.0,
        };
        let gpu = response(
            serde_json::json!({
                "token": "ordered",
                "solutions": [{"nonce": 2}, {"nonce": 4}],
                "attempted_nonces_total": 5
            }),
            "gpu",
            "wgpu",
        );
        let cpu = response(
            serde_json::json!({
                "token": "ordered",
                "solutions": [{"y": "1"}, {"nonce": "0"}],
                "attempted_nonces_total": 7
            }),
            "cpu",
            "hashwx-native+rsw",
        );
        let merged = merge_mixed_capjs(&parts, gpu, &cpu, Instant::now())?;
        assert_eq!(merged.backend, "gpu+cpu");
        assert_eq!(
            merged.implementation.as_deref(),
            Some("wgpu+hashwx-native+rsw")
        );
        assert_eq!(
            merged.answer["solutions"],
            serde_json::json!([
                {"y": "1"}, {"nonce": 2}, {"nonce": "0"}, {"nonce": 4}
            ])
        );
        assert_eq!(merged.answer["attempted_nonces_total"], 12);
        assert_eq!(merged.answer["token"], "ordered");
        assert!(
            split_mixed_capjs(&serde_json::json!({
                "token": "single",
                "format": 2,
                "challenges": [{"protocol": "sha256-pow", "payload": {"salt": "only", "target": "0"}}]
            }))?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn mixed_capjs_rejects_unsupported_work_before_gpu_split() -> anyhow::Result<()> {
        let challenge = serde_json::json!({
            "token": "mixed-unsupported",
            "format": 2,
            "challenges": [
                {"protocol": "sha256-pow", "payload": {"salt": "some", "target": "0"}},
                {"protocol": "instrumentation", "payload": {}},
                {"protocol": "rsw", "payload": {"N": "11", "x": "2", "t": 3}}
            ]
        });
        let error = split_mixed_capjs(&challenge)
            .err()
            .context("unsupported format-2 work must fail before GPU splitting")?;
        assert!(error.to_string().contains("browser-environment probe"));
        Ok(())
    }
}
