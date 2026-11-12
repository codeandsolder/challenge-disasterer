//! HTTP request validation, protocol dispatch and response/error mapping.
//!
//! Compute scheduling and solver implementations remain on `AppState` and
//! the shared solver crates; this module owns only the HTTP-facing contract.

use std::{sync::Arc, time::Instant};

use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{AppState, SolveResponse};
use crate::{anubis::Challenge, anubis_wasm, hashx};

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
pub struct ErrorResponse {
    pub(super) error: String,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorResponse>)>;

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
        "cerberus" | "goaway" | "capjs" | "mcaptcha" | "haphash" | "altcha" | "linear"
        | "friendlycaptcha" | "tor-hs-v1" | "pow-bot-deterrent" => {
            if matches!(
                request.solver.as_str(),
                "friendlycaptcha" | "tor-hs-v1" | "pow-bot-deterrent"
            ) && request.implementation.as_deref() == Some("wgpu")
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("{} supports only CPU implementations", request.solver),
                    }),
                ));
            }
            if request.solver == "linear"
                && matches!(
                    request.implementation.as_deref(),
                    Some(implementation) if !matches!(implementation, "auto" | "wgpu")
                )
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Linear.pub supports only auto and wgpu implementations".to_owned(),
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
