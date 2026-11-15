//! HTTP request validation, protocol dispatch and response/error mapping.
//!
//! Compute scheduling and solver implementations remain on `AppState` and
//! the shared solver crates; this module owns only the HTTP-facing contract.

use std::{sync::Arc, time::Instant};

use axum::{
    Extension, Json,
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
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

/// First point at which Axum exposes a /solve request to application code.
/// It precedes body buffering, JSON deserialization, and handler scheduling,
/// but not TCP accept or HTTP header parsing in hyper.
#[derive(Clone, Copy)]
pub struct RequestIngress(Instant);

#[derive(Clone, Copy)]
/// All durations are milliseconds, not absolute timestamps.
struct HandlerStages {
    pre_handler: f64,
    handler: f64,
    queue: Option<f64>,
    solve: Option<f64>,
}

/// Attach the best available HTTP ingress timestamp before Axum extracts JSON.
/// Every /solve response, including an extractor or routing error, gets an
/// end-to-end server-side duration in the standard Server-Timing header.
pub async fn mark_request_ingress(mut request: Request, next: Next) -> Response {
    let received_at = Instant::now();
    request.extensions_mut().insert(RequestIngress(received_at));
    let mut response = next.run(request).await;
    let server_ms = received_at.elapsed().as_secs_f64() * 1_000.0;
    let phases = response.extensions_mut().remove::<HandlerStages>();
    let header = phases.map_or_else(
        || format!("server;dur={server_ms:.3}"),
        |phases| {
            let mut header = format!(
                "pre_handler;dur={:.3}, handler;dur={:.3}, server;dur={server_ms:.3}",
                phases.pre_handler, phases.handler,
            );
            if let (Some(queue_ms), Some(solve_ms)) = (phases.queue, phases.solve) {
                use std::fmt::Write as _;
                let _ = write!(header, ", queue;dur={queue_ms:.3}, solve;dur={solve_ms:.3}");
            }
            header
        },
    );
    if let Ok(value) = HeaderValue::from_str(&header) {
        response.headers_mut().insert("server-timing", value);
    }
    response
}

/// The production HTTP route; the original `solve` remains the one true
/// protocol implementation used directly by the existing unit tests.
/// The timing header does not change successful or error JSON schemas.
pub async fn solve_instrumented(
    State(state): State<Arc<AppState>>,
    Extension(arrival): Extension<RequestIngress>,
    Json(request): Json<SolveRequest>,
) -> Response {
    let pre_handler_ms = arrival.0.elapsed().as_secs_f64() * 1_000.0;
    let handler_start = Instant::now();
    let result = solve(State(state), Json(request)).await;
    let handler_ms = handler_start.elapsed().as_secs_f64() * 1_000.0;
    let (queue_ms, solve_ms) = match &result {
        Ok(Json(reply)) => (Some(reply.queue_ms), Some(reply.solve_ms)),
        Err(_) => (None, None),
    };
    let mut response = result.into_response();
    response.extensions_mut().insert(HandlerStages {
        pre_handler: pre_handler_ms,
        handler: handler_ms,
        queue: queue_ms,
        solve: solve_ms,
    });
    response
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
        "cerberus" | "goaway" | "capjs" | "mcaptcha" | "haphash" | "altcha" | "linear"
        | "friendlycaptcha" | "tor-hs-v1" | "pow-bot-deterrent" => {
            if matches!(request.solver.as_str(), "tor-hs-v1" | "pow-bot-deterrent")
                && request.implementation.as_deref() == Some("wgpu")
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("{} supports only CPU implementations", request.solver),
                    }),
                ));
            }
            if request.solver == "friendlycaptcha"
                && request.implementation.as_deref() == Some("wgpu")
                && !state.general_gpu_has_native_u64()
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Friendly Captcha WGPU requires an active SHADER_INT64 GPU"
                            .to_owned(),
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
