//! Calibrated route selection and fallback, separate from backend execution.

use std::time::Instant;

use anyhow::Context;
use tracing::warn;

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

    pub(super) async fn route_anubis(
        &self,
        algorithm: &str,
        challenge: Challenge,
        batch_size_hint: usize,
        queued_at: Instant,
    ) -> anyhow::Result<SolveResponse> {
        challenge.validate()?;
        // Spend idle GPU capacity before consuming CPU resources, regardless
        // of which backend has a lower calibrated single-request latency.
        // Reserving the shared permit is atomic across all GPU queues.
        #[cfg(feature = "gpu")]
        if self.gpu_operational && challenge.gpu_compatible() {
            if let Ok(permit) = self.gpu_slots.clone().try_acquire_owned() {
                match self
                    .run_gpu_with_permit(challenge.clone(), queued_at, permit)
                    .await
                {
                    Ok(response) => return Ok(response),
                    Err(err) => {
                        warn!(%err, "idle GPU solve failed; falling back to calibrated routes")
                    }
                }
            }
        }

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
                    // The idle GPU was considered above. A calibration row
                    // must not queue behind busy GPU work and hog the device.
                    "anubis-gpu" => {}
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

        self.run_cpu(challenge, queued_at).await
    }
}
