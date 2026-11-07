//! Local CPU/GPU execution and external-provider calls, independent of HTTP dispatch.

#[cfg(feature = "gpu")]
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::Value;
#[cfg(feature = "gpu")]
use tokio::sync::oneshot;

use super::{
    AppState, ExternalSolveResponse, SolveResponse,
    argon2_schedule::{Reservation, Work},
};
#[cfg(feature = "gpu")]
use super::{GeneralGpuJob, GpuJob, HashxGpuJob};
use crate::{
    anubis::{self, Challenge},
    anubis_wasm, classic_pow, hashx, hashx_cpu_jit,
    routing::RouteEntry,
};

// The Asus RTX 2060 matched-binary measurements put the SHA/PBKDF2
// break-even near a few hundred effective iterations. Use 1024 as the
// initial cross-shape cost score to avoid GPU dispatch on tiny proofs;
// unsupported shapes are filtered by challenge-protocols' own parser.
#[cfg(feature = "gpu")]
const ALTCHA_AUTO_GPU_MIN_WORK: u128 = 1_024;

#[cfg(feature = "gpu")]
fn prefers_auto_gpu(solver: &str, challenge: &Value, implementation: Option<&str>) -> bool {
    (implementation.is_none() || implementation == Some("auto"))
        && classic_pow::auto_gpu_expected_work(solver, challenge)
            .is_some_and(|work| work >= ALTCHA_AUTO_GPU_MIN_WORK)
}

impl AppState {
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
        let worker_count = if solver == "anubis-argon2id" {
            self.config.argon2_workers
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
            nonce_workers: None,
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
            nonce_workers: None,
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
        tx.send(HashxGpuJob {
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
            && prefers_auto_gpu(&solver, &challenge, implementation.as_deref())
        {
            match self
                .dispatch_general_gpu(solver.clone(), challenge.clone(), queued_at, true)
                .await
            {
                Ok(Some(result)) => return Ok(result),
                Ok(None) => {
                    tracing::debug!(solver, "GPU queue at capacity, routing ALTCHA to CPU");
                }
                Err(err) => {
                    tracing::warn!(solver, %err, "automatic ALTCHA WGPU failed; routing to CPU");
                }
            }
        }
        let work = Work::from_challenge(&solver, &challenge);
        let desired = work.map_or(1, |profile| {
            profile.target_workers(
                self.config.argon2_workers,
                self.config.cpu_workers,
                self.config.altcha_worker_policy,
            )
        });
        let reservation = Reservation::acquire(
            &self.cpu_slots,
            self.argon2_memory_slots.as_ref(),
            self.argon2_memory_capacity_mib,
            work,
            desired,
        )
        .await?;
        let started = Instant::now();
        let queue_ms = started.duration_since(queued_at).as_secs_f64() * 1000.0;
        let response_solver = solver.clone();
        let (result, workers) = tokio::task::spawn_blocking(move || {
            let workers = reservation.workers();
            let result = classic_pow::solve(&solver, challenge, implementation.as_deref(), workers);
            // Explicitly release CPU and memory only after the search completes.
            drop(reservation);
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
            nonce_workers: work.map(|_| workers),
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
        self.dispatch_general_gpu(solver, challenge, queued_at, false)
            .await?
            .context("mandatory WGPU dispatch was not enqueued")
    }

    /// Send a general WGPU job without duplicating protocol execution.
    /// Opportunistic ALTCHA auto routing does not wait for *channel* capacity:
    /// when all 128 queued positions are occupied, use CPU instead. Explicit
    /// WGPU requests retain FIFO backpressure and do not silently switch.
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
    fn auto_gpu_prefers_substantial_supported_work_but_not_tiny_or_forced_cpu() {
        let cheap = altcha("0", "SHA-256", 2);
        let borderline = altcha("00", "SHA-256", 2);
        let substantial = altcha("000", "SHA-256", 2);
        let expensive_pbkdf2 = altcha("00", "PBKDF2/SHA-256", 16);
        let unsupported = altcha("000000000", "SHA-512", 2);

        assert!(!prefers_auto_gpu("altcha", &cheap, None));
        assert!(!prefers_auto_gpu("altcha", &borderline, Some("auto")));
        assert!(prefers_auto_gpu("altcha", &substantial, None));
        assert!(prefers_auto_gpu("altcha", &expensive_pbkdf2, Some("auto")));
        assert!(!prefers_auto_gpu("altcha", &expensive_pbkdf2, Some("safe")));
        assert!(!prefers_auto_gpu("altcha", &substantial, Some("sha-ni")));
        assert!(!prefers_auto_gpu("altcha", &unsupported, None));
        assert!(!prefers_auto_gpu("capjs", &substantial, None));
    }
}
