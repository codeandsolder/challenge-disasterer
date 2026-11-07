//! Calibrated route selection and fallback, separate from backend execution.

#[cfg(feature = "gpu")]
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Context;
#[cfg(feature = "gpu")]
use tokio::sync::oneshot;
use tracing::warn;

#[cfg(feature = "gpu")]
use super::GpuJob;
use super::{AppState, SolveResponse};
use crate::{
    anubis::{self, Challenge},
    hashx,
    routing::RouteEntry,
};

impl AppState {
    pub(super) fn calibrated_routes(
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

    pub(super) async fn route_hashx(
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
                            serde_json::from_value::<hashx::Answer>(response.answer.clone())
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

    #[expect(
        clippy::too_many_lines,
        reason = "routing fallback order is intentionally linear and mirrors the calibrated route sequence"
    )]
    pub(super) async fn route_anubis(
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

            let started = started_rx
                .await
                .map_err(|_| anyhow::anyhow!("GPU worker dropped start signal"))?;
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
                    nonce_workers: None,
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
