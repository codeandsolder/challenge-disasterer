//! HTTP integration and state/routing regression tests.
//!
//! Deliberately kept separate from service construction and request dispatch.

use super::http::SolveRequest;
use super::*;
use axum::{Json, extract::State, http::StatusCode};

#[tokio::test]
async fn friendlycaptcha_v1_http_dispatch_and_cpu_only_policy() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    // v1, two subpuzzles, difficulty zero, otherwise zeroed 32-byte header.
    const PUZZLE: &str = "testing.AAAAAAAAAAAAAAAAAQACAAAAAAAAAAAAAAAAAAAAAAA=";
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 2,
            cpu_workers_per_request: 1,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    let Json(snapshot) = stats(State(state.clone())).await;
    let advertised = serde_json::to_value(snapshot)?;
    assert!(
        advertised["available_solvers"]
            .as_array()
            .context("available solvers are an array")?
            .iter()
            .any(|entry| entry == "friendlycaptcha")
    );
    let capabilities = advertised["solver_capabilities"]
        .as_array()
        .context("solver capabilities are an array")?;
    let capability = capabilities
        .iter()
        .find(|entry| entry["solver"] == "friendlycaptcha")
        .context("Friendly Captcha capability missing")?;
    assert_eq!(capability["algorithms"], serde_json::json!(["blake2b-256"]));
    assert_eq!(capability["difficulty_min"], 0);
    assert_eq!(capability["difficulty_max"], 255);
    assert_eq!(capability["counter_mode"], "indexed-u32-le");
    assert_eq!(capability["counter_max"], u64::from(u32::MAX));
    assert_eq!(
        capability["implementations"],
        serde_json::json!(["auto", "safe"])
    );

    for implementation in [None, Some("safe".to_owned())] {
        let request = SolveRequest {
            solver: "friendlycaptcha".to_owned(),
            algorithm: None,
            implementation,
            batch_size_hint: 1,
            challenge: serde_json::json!({"puzzle": PUZZLE}),
        };
        let Json(response) =
            solve(State(state.clone()), Json(request))
                .await
                .map_err(|(status, Json(err))| {
                    anyhow::anyhow!(
                        "Friendly Captcha v1 HTTP dispatch returned {status}: {}",
                        err.error
                    )
                })?;
        assert_eq!(response.solver, "friendlycaptcha");
        assert_eq!(response.backend, "cpu");
        assert_eq!(
            response.implementation.as_deref(),
            Some("rustcrypto-blake2b")
        );
        assert_eq!(response.answer["solution_count"], 2);
        assert!(response.answer["attempted_nonces"].as_u64().is_some());
        let encoded = response.answer["response"]
            .as_str()
            .context("missing original Friendly Captcha response")?;
        assert!(encoded.starts_with(&format!("{PUZZLE}.")));
        let fields: Vec<_> = encoded.split('.').collect();
        assert_eq!(fields.len(), 4);
        let solutions = STANDARD.decode(fields[2])?;
        assert!(challenge_protocols::classic_pow::verify_friendlycaptcha_v1(
            PUZZLE, &solutions,
        )?);
    }

    let unsupported = SolveRequest {
        solver: "friendlycaptcha".to_owned(),
        algorithm: None,
        implementation: Some("wgpu".to_owned()),
        batch_size_hint: 1,
        challenge: serde_json::json!({"puzzle": PUZZLE}),
    };
    let result = solve(State(state), Json(unsupported)).await;
    assert!(matches!(result, Err((StatusCode::BAD_REQUEST, _))));
    Ok(())
}

/// End-to-end Friendly v1 GPU routing, forced backend, and CPU policy parity.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical GPU with SHADER_INT64"]
async fn friendlycaptcha_gpu_auto_forced_and_cpu_fallback_http() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let mut header = [0u8; 64];
    header[12] = 1;
    header[14] = 16;
    header[15] = 128;
    header[24..32].copy_from_slice(b"nonce123");
    let puzzle = format!("fixture.{}", STANDARD.encode(header));
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 4,
            cpu_workers_per_request: 4,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            gpu_calibration_hashes: 8_192,
            gpu_enabled: true,
            gpu_policy: GpuPolicy::PreferGpu,
            routing_calibration: None,
        })
        .await?,
    );
    assert!(
        state.general_gpu_has_native_u64(),
        "Friendly v1 needs a Vulkan GPU with SHADER_INT64"
    );
    let Json(snapshot) = stats(State(state.clone())).await;
    let capabilities = serde_json::to_value(snapshot)?;
    let friendly = capabilities["solver_capabilities"]
        .as_array()
        .context("solver_capabilities")?
        .iter()
        .find(|capability| capability["solver"] == "friendlycaptcha")
        .context("Friendly capability missing")?;
    assert_eq!(
        friendly["implementations"],
        serde_json::json!(["auto", "safe", "wgpu"])
    );
    for implementation in [None, Some("wgpu"), Some("safe")] {
        let request = SolveRequest {
            solver: "friendlycaptcha".to_owned(),
            algorithm: None,
            implementation: implementation.map(str::to_owned),
            batch_size_hint: 1,
            challenge: serde_json::json!({"puzzle": puzzle}),
        };
        let Json(response) =
            solve(State(state.clone()), Json(request))
                .await
                .map_err(|(status, Json(err))| {
                    anyhow::anyhow!("Friendly GPU HTTP {status}: {}", err.error)
                })?;
        let expected_backend = if implementation == Some("safe") {
            "cpu"
        } else {
            "gpu"
        };
        assert_eq!(response.backend, expected_backend);
        if expected_backend == "gpu" {
            assert_eq!(response.implementation.as_deref(), Some("wgpu-blake2b-256"));
            assert_eq!(response.nonce_workers, None);
        }
        assert_eq!(response.answer["solution_count"], 16);
        let encoded = response.answer["response"]
            .as_str()
            .context("Friendly response missing")?;
        let parts: Vec<_> = encoded.split('.').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "fixture");
        assert_eq!(
            parts[1],
            puzzle.split_once('.').context("puzzle separator")?.1
        );
        let proofs = STANDARD.decode(parts[2])?;
        assert!(challenge_protocols::classic_pow::verify_friendlycaptcha_v1(
            &puzzle, &proofs
        )?);
        for (bank, proof) in proofs.chunks_exact(8).enumerate() {
            assert_eq!(usize::from(proof[0]), bank);
        }
        assert_eq!(STANDARD.decode(parts[3])?.len(), 3);
        println!(
            "FRIENDLY_HTTP_GPU_RESULT implementation={implementation:?} backend={} total_ms={:.3}",
            response.backend, response.total_ms
        );
    }
    Ok(())
}

/// Independently searched Friendly banks may consume only actually owned CPU permits.
#[tokio::test]
async fn friendlycaptcha_parallel_bank_workers_respect_cpu_permits() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let mut header = [0_u8; 32];
    header[12] = 1;
    header[14] = 16;
    header[15] = 128;
    let puzzle = format!("fixture.{}", STANDARD.encode(header));
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 4,
            cpu_workers_per_request: 4,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    for held_permits in [2_u32, 0_u32] {
        let held = if held_permits == 0 {
            None
        } else {
            Some(
                state
                    .cpu_slots
                    .clone()
                    .acquire_many_owned(held_permits)
                    .await?,
            )
        };
        let available = usize::try_from(4_u32 - held_permits)?;
        assert_eq!(state.cpu_slots.available_permits(), available);
        let request = SolveRequest {
            solver: "friendlycaptcha".to_owned(),
            algorithm: None,
            implementation: None,
            batch_size_hint: 1,
            challenge: serde_json::json!({"puzzle": puzzle}),
        };
        let Json(response) =
            solve(State(state.clone()), Json(request))
                .await
                .map_err(|(status, Json(err))| {
                    anyhow::anyhow!("Friendly CPU reservation returned {status}: {}", err.error)
                })?;
        assert_eq!(response.backend, "cpu");
        assert_eq!(response.nonce_workers, Some(available));
        assert_eq!(
            response.implementation.as_deref(),
            Some("rustcrypto-blake2b")
        );
        assert_eq!(response.answer["solution_count"], 16);
        let response_text = response.answer["response"]
            .as_str()
            .context("missing Friendly Captcha proof")?;
        let proof_bytes = STANDARD.decode(
            response_text
                .split('.')
                .nth(2)
                .context("missing proof section")?,
        )?;
        assert!(challenge_protocols::classic_pow::verify_friendlycaptcha_v1(
            &puzzle,
            &proof_bytes,
        )?);
        assert_eq!(state.cpu_slots.available_permits(), available);
        drop(held);
        assert_eq!(state.cpu_slots.available_permits(), 4);
    }
    Ok(())
}

#[tokio::test]
async fn sequentialread_scrypt_http_route_and_go_reference() -> anyhow::Result<()> {
    use anyhow::Context as _;

    // Exact base64(JSON) challenge from sequentialread's Go data model,
    // with scrypt.Key verification independently generated using x/crypto.
    let challenge = serde_json::Value::String(
        "eyJOIjoxNiwiciI6MSwicCI6MSwia2xlbiI6MTYsImkiOiJBUUlEQkFVR0J3Zz0iLCJkIjoiMGZmZiIsImRsIjo0fQ==".to_owned()
    );
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 4,
            cpu_workers_per_request: 1,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    let before = state.cpu_slots.available_permits();
    let Json(response) = solve(
        State(state.clone()),
        Json(SolveRequest {
            solver: "pow-bot-deterrent".to_owned(),
            algorithm: None,
            implementation: None,
            batch_size_hint: 1,
            challenge: challenge.clone(),
        }),
    )
    .await
    .map_err(|(status, Json(err))| {
        anyhow::anyhow!("sequentialread Scrypt returned {status}: {}", err.error)
    })?;
    assert_eq!(response.solver, "pow-bot-deterrent");
    assert_eq!(response.backend, "cpu");
    assert_eq!(response.implementation.as_deref(), Some("scrypt"));
    assert_eq!(response.nonce_workers, Some(1));
    assert_eq!(response.answer["nonce"], "10");
    assert_eq!(
        response.answer["derived_key"],
        "732d9a0d3e86cd94a3f9bfda94f00620"
    );
    assert_eq!(response.answer["challenge"], challenge);
    assert!(challenge_protocols::classic_pow::verify_pow_bot_deterrent(
        &challenge,
        response.answer["nonce"].as_str().context("nonce")?,
    )?);
    assert_eq!(state.cpu_slots.available_permits(), before);

    let Json(stats) = stats(State(state.clone())).await;
    let advertised = serde_json::to_value(stats)?;
    assert!(
        advertised["available_solvers"]
            .as_array()
            .context("available_solvers")?
            .iter()
            .any(|name| name == "pow-bot-deterrent")
    );
    let capability = advertised["solver_capabilities"]
        .as_array()
        .context("solver_capabilities")?
        .iter()
        .find(|cap| cap["solver"] == "pow-bot-deterrent")
        .context("sequentialread capability")?;
    assert_eq!(capability["algorithms"], serde_json::json!(["scrypt"]));
    assert_eq!(
        capability["implementations"],
        serde_json::json!(["auto", "safe"])
    );

    let Err((status, _)) = solve(
        State(state.clone()),
        Json(SolveRequest {
            solver: "pow-bot-deterrent".to_owned(),
            algorithm: None,
            implementation: Some("wgpu".to_owned()),
            batch_size_hint: 1,
            challenge: challenge.clone(),
        }),
    )
    .await
    else {
        anyhow::bail!("wgpu was erroneously accepted")
    };
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(state.cpu_slots.available_permits(), before);
    Ok(())
}

#[tokio::test]
async fn sequentialread_scrypt_fanout_respects_shared_cpu_and_ram() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    // The real verifier threshold requires about 32 attempts; metadata
    // intentionally advertises difficulty 1 and must not govern routing.
    let encoded = serde_json::json!({
        "N": 4096, "r": 8, "p": 1, "klen": 16,
        "i": STANDARD.encode([1_u8, 2, 3, 4, 5, 6, 7, 8]),
        "d": "07", "dl": 1
    });
    let challenge = serde_json::Value::String(STANDARD.encode(serde_json::to_vec(&encoded)?));
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 4,
            cpu_workers_per_request: 4,
            // N=4096,r=8,p=1 needs slightly over 4 MiB per worker;
            // only two of four suggested workers fit into 12 MiB.
            kdf_memory_budget_mib: Some(12),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    let memory = state
        .kdf_memory_slots
        .as_ref()
        .context("shared memory semaphore")?
        .clone();
    for (held_cpu, expected_workers) in [(0_u32, 2), (3, 1)] {
        let held = if held_cpu > 0 {
            Some(state.cpu_slots.clone().acquire_many_owned(held_cpu).await?)
        } else {
            None
        };
        let starting_memory = memory.available_permits();
        let Json(response) = solve(
            State(state.clone()),
            Json(SolveRequest {
                solver: "pow-bot-deterrent".to_owned(),
                algorithm: None,
                implementation: None,
                batch_size_hint: 1,
                challenge: challenge.clone(),
            }),
        )
        .await
        .map_err(|(status, Json(err))| {
            anyhow::anyhow!(
                "sequentialread parallel solve failed {status}: {}",
                err.error
            )
        })?;
        assert_eq!(response.nonce_workers, Some(expected_workers));
        assert!(challenge_protocols::classic_pow::verify_pow_bot_deterrent(
            &challenge,
            response.answer["nonce"]
                .as_str()
                .context("parallel nonce")?,
        )?);
        assert_eq!(memory.available_permits(), starting_memory);
        assert_eq!(
            state.cpu_slots.available_permits(),
            4 - usize::try_from(held_cpu)?
        );
        drop(held);
    }
    assert_eq!(memory.available_permits(), 12);
    assert_eq!(state.cpu_slots.available_permits(), 4);
    Ok(())
}

#[test]
fn tor_hs_v1_fanout_thresholds_and_invalid_effort() {
    // This policy chooses a target, not a hard reservation. The
    // shared semaphore may reserve fewer actual workers at runtime.
    for (effort, target) in [
        (0_u32, 1_usize),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (8, 2),
        (9, 4),
        (16, 4),
        (63, 4),
        (64, 16),
        (128, 16),
        (u32::MAX, 16),
    ] {
        for configured in [1_usize, 2, 3, 4, 8, 16] {
            let expected = target.min(configured);
            assert_eq!(
                AppState::tor_equix_suggested_workers(
                    &serde_json::json!({"effort": effort}),
                    configured,
                ),
                expected,
                "effort={effort}, configured={configured}"
            );
        }
    }
    // Invalid numeric values are rejected by the Tor protocol parser,
    // not allowed to trigger a large speculative worker reservation.
    for challenge in [
        serde_json::json!({}),
        serde_json::json!({"effort": null}),
        serde_json::json!({"effort": "64"}),
        serde_json::json!({"effort": -1}),
        serde_json::json!({"effort": 3.5}),
        serde_json::json!({"effort": u64::from(u32::MAX) + 1}),
    ] {
        assert_eq!(AppState::tor_equix_suggested_workers(&challenge, 16), 1);
    }
    assert_eq!(
        AppState::tor_equix_suggested_workers(&serde_json::json!({"effort": u32::MAX}), 0,),
        1
    );
}

#[tokio::test]
async fn tor_hs_v1_adaptive_fanout_respects_live_permits() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let id = "11".repeat(32);
    let seed = STANDARD.encode([0xaa_u8; 32]);
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 8,
            cpu_workers_per_request: 8,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    for held_permits in [0_u32, 6_u32, 7_u32] {
        let held = if held_permits == 0 {
            None
        } else {
            Some(
                state
                    .cpu_slots
                    .clone()
                    .acquire_many_owned(held_permits)
                    .await?,
            )
        };
        let free = usize::try_from(8_u32 - held_permits)?;
        for (effort, target) in [
            (1_u32, 1_usize),
            (2, 1),
            (4, 2),
            (8, 2),
            (9, 4),
            (16, 4),
            (63, 4),
            (64, 8),
            (128, 8),
        ] {
            let request = SolveRequest {
                solver: "tor-hs-v1".to_owned(),
                algorithm: None,
                implementation: None,
                batch_size_hint: 1,
                challenge: serde_json::json!({
                    "blinded_id": id,
                    "seed_b64": seed,
                    "effort": effort,
                    "nonce": "55555555555555555555555555555555",
                }),
            };
            let Json(response) = solve(State(state.clone()), Json(request)).await.map_err(
                |(status, Json(err))| {
                    anyhow::anyhow!(
                        "Tor effort {effort} with {free} free slots failed ({status}): {}",
                        err.error
                    )
                },
            )?;
            assert_eq!(response.backend, "cpu");
            assert_eq!(response.solver, "tor-hs-v1");
            assert_eq!(response.nonce_workers, Some(target.min(free)));
            assert_eq!(response.answer["effort"], effort);
            assert_eq!(response.answer["extension_type"], 2);
            let bytes = hex::decode(
                response.answer["extension_body"]
                    .as_str()
                    .context("Tor proof extension_body")?,
            )?;
            assert_eq!(bytes.len(), 41);
            assert!(challenge_protocols::classic_pow::verify_tor_hs_v1(
                &id, &seed, &bytes,
            )?);
            assert_eq!(state.cpu_slots.available_permits(), free);
        }
        drop(held);
        assert_eq!(state.cpu_slots.available_permits(), 8);
    }
    Ok(())
}

#[tokio::test]
async fn tor_hs_v1_cpu_permits_and_c_tor_reference_roundtrip() -> anyhow::Result<()> {
    use anyhow::Context as _;
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let id = "11".repeat(32);
    let seed = STANDARD.encode([0xaa_u8; 32]);
    let state = Arc::new(
        AppState::initialize(Config {
            cpu_workers: 2,
            cpu_workers_per_request: 2,
            kdf_memory_budget_mib: Some(128),
            cpu_calibration_hashes: 8_192,
            #[cfg(feature = "gpu")]
            gpu_calibration_hashes: 8_192,
            gpu_enabled: false,
            gpu_policy: GpuPolicy::CpuOnly,
            routing_calibration: None,
        })
        .await?,
    );
    let Json(snapshot) = stats(State(state.clone())).await;
    let advertised = serde_json::to_value(snapshot)?;
    assert!(
        advertised["available_solvers"]
            .as_array()
            .context("available_solvers")?
            .iter()
            .any(|item| item == "tor-hs-v1")
    );
    let capability = advertised["solver_capabilities"]
        .as_array()
        .context("solver_capabilities")?
        .iter()
        .find(|item| item["solver"] == "tor-hs-v1")
        .context("Tor v1 solver capability")?;
    assert_eq!(
        capability["algorithms"],
        serde_json::json!(["equix", "blake2b-32"])
    );
    assert_eq!(capability["difficulty_max"], u32::MAX);
    assert_eq!(capability["counter_mode"], "nonce-u128-le");
    assert_eq!(
        capability["implementations"],
        serde_json::json!(["auto", "safe"])
    );

    for held_permits in [1_u32, 0_u32] {
        let held = if held_permits == 0 {
            None
        } else {
            Some(state.cpu_slots.clone().acquire_owned().await?)
        };
        let available = usize::try_from(2 - held_permits)?;
        let challenge = serde_json::json!({
            "blinded_id": id,
            "seed_b64": seed,
            "effort": 100,
            "nonce": "55555555555555555555555555555555",
        });
        let request = SolveRequest {
            solver: "tor-hs-v1".to_owned(),
            algorithm: None,
            implementation: None,
            batch_size_hint: 1,
            challenge: challenge.clone(),
        };
        let Json(response) =
            solve(State(state.clone()), Json(request))
                .await
                .map_err(|(status, Json(err))| {
                    anyhow::anyhow!("Tor v1 CPU route failed {status}: {}", err.error)
                })?;
        assert_eq!(response.backend, "cpu");
        assert_eq!(response.solver, "tor-hs-v1");
        assert_eq!(response.nonce_workers, Some(available));
        assert_eq!(response.implementation.as_deref(), Some("equix-hashx"));
        assert_eq!(response.answer["extension_type"], 2);
        assert_eq!(response.answer["effort"], 100);
        let proof = hex::decode(
            response.answer["extension_body"]
                .as_str()
                .context("extension_body")?,
        )?;
        assert_eq!(proof.len(), 41);
        assert!(challenge_protocols::classic_pow::verify_tor_hs_v1(
            &id, &seed, &proof,
        )?);
        assert_eq!(state.cpu_slots.available_permits(), available);
        drop(held);
        assert_eq!(state.cpu_slots.available_permits(), 2);
    }
    let unsupported = SolveRequest {
        solver: "tor-hs-v1".to_owned(),
        algorithm: None,
        implementation: Some("wgpu".to_owned()),
        batch_size_hint: 1,
        challenge: serde_json::json!({
            "blinded_id": id,
            "seed_b64": seed,
            "effort": 1,
            "nonce": "00000000000000000000000000000000",
        }),
    };
    assert!(matches!(
        solve(State(state), Json(unsupported)).await,
        Err((StatusCode::BAD_REQUEST, _))
    ));
    Ok(())
}

/// Hardware-only route test: all automatic GPU policies honor the same
/// GPU ownership, with verified CPU and GPU challenge solutions.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical WGPU adapter"]
async fn sha256_gpu_policy_idle_busy_queued_and_forced() -> anyhow::Result<()> {
    use std::time::Duration;

    let mut state = AppState::initialize(Config {
        cpu_workers: 2,
        cpu_workers_per_request: 1,
        kdf_memory_budget_mib: Some(128),
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

/// Mixed Cap.js format-2 GPU SHA-256 plus CPU HashWX/RSW.
/// The GPU permit must not be retained while running CPU-only subchallenges.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical WGPU adapter"]
async fn capjs_mixed_gpu_sha_with_cpu_hashwx_and_rsw() -> anyhow::Result<()> {
    use sha2::{Digest as _, Sha256};

    let mut state = AppState::initialize(Config {
        cpu_workers: 2,
        cpu_workers_per_request: 1,
        kdf_memory_budget_mib: Some(128),
        cpu_calibration_hashes: 8_192,
        gpu_calibration_hashes: 8_192,
        gpu_enabled: true,
        gpu_policy: GpuPolicy::PreferGpu,
        routing_calibration: None,
    })
    .await?;
    assert!(state.general_gpu_operational);
    let challenge = serde_json::json!({
        "token": "hybrid-cap",
        "format": 2,
        "challenges": [
            {"protocol": "hashwx", "payload": {
                "c": "00".repeat(32), "d": 1, "n": 1
            }},
            {"protocol": "sha256-pow", "payload": {
                "salt": "hybrid-sha-", "target": "0"
            }},
            {"protocol": "rsw", "payload": {
                "N": "11", "x": "2", "t": 3
            }}
        ]
    });
    let verify = |answer: &Value| -> anyhow::Result<()> {
        let solutions = answer["solutions"]
            .as_array()
            .context("Cap.js solutions must be an array")?;
        assert_eq!(solutions.len(), 3);
        assert_eq!(solutions[0]["nonce"], "0");
        assert_eq!(solutions[2]["y"], "1");
        let nonce = solutions[1]["nonce"]
            .as_f64()
            .context("Cap.js SHA nonce must be a number")?;
        let digest = Sha256::digest(format!("hybrid-sha-{nonce}").as_bytes());
        assert_eq!(
            digest[0] >> 4,
            0,
            "SHA-256 prefix must be independently valid"
        );
        Ok(())
    };
    let mixed = state
        .run_classic("capjs".to_owned(), challenge.clone(), None, Instant::now())
        .await?;
    assert_eq!(mixed.backend, "gpu+cpu");
    assert_eq!(
        mixed.implementation.as_deref(),
        Some("wgpu+hashwx-native+rsw")
    );
    verify(&mixed.answer)?;
    let forced = state
        .run_general_gpu("capjs".to_owned(), challenge.clone(), Instant::now())
        .await?;
    assert_eq!(forced.backend, "gpu+cpu");
    verify(&forced.answer)?;

    state.config.gpu_policy = GpuPolicy::CpuOnly;
    let cpu = state
        .run_classic("capjs".to_owned(), challenge, None, Instant::now())
        .await?;
    assert_eq!(cpu.backend, "cpu");
    verify(&cpu.answer)?;
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
        cpu_workers_per_request: 1,
        kdf_memory_budget_mib: Some(128),
        cpu_calibration_hashes: 8_192,
        gpu_calibration_hashes: 8_192,
        gpu_enabled: true,
        gpu_policy: GpuPolicy::PreferGpu,
        routing_calibration: None,
    })
    .await?;
    assert!(state.general_gpu_operational);
    assert!(
        state.general_gpu_has_native_u64(),
        "selected GPU must have SHADER_INT64"
    );
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

#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical WGPU adapter (SHA-512 requires SHADER_INT64)"]
async fn altcha_v1_gpu_first_cpu_only_and_forced_full_digest() -> anyhow::Result<()> {
    use sha1::{Digest as _, Sha1};
    use sha2::{Sha256, Sha512};

    let mut state = AppState::initialize(Config {
        cpu_workers: 2,
        cpu_workers_per_request: 1,
        kdf_memory_budget_mib: Some(128),
        cpu_calibration_hashes: 8_192,
        gpu_calibration_hashes: 8_192,
        gpu_enabled: true,
        gpu_policy: GpuPolicy::PreferGpu,
        routing_calibration: None,
    })
    .await?;
    assert!(state.general_gpu_operational);
    for algorithm in ["SHA-1", "SHA-256", "SHA-512"] {
        if algorithm == "SHA-512" && !state.general_gpu_has_native_u64() {
            continue;
        }
        let long_salt = "x".repeat(121);
        for salt in ["", "Привет-世界", long_salt.as_str()] {
            let nonce = 17_u32;
            let bytes = format!("{salt}{nonce}");
            let digest = match algorithm {
                "SHA-1" => hex::encode(Sha1::digest(bytes.as_bytes())),
                "SHA-256" => hex::encode(Sha256::digest(bytes.as_bytes())),
                "SHA-512" => hex::encode(Sha512::digest(bytes.as_bytes())),
                _ => unreachable!("expected SHA variant"),
            };
            let challenge = serde_json::json!({
                "algorithm":algorithm,
                "salt":salt,
                "signature":"signed",
                "maxnumber":nonce+1,
                "challenge":digest,
            });
            let gpu = state
                .run_classic("altcha".to_owned(), challenge.clone(), None, Instant::now())
                .await?;
            assert_eq!(gpu.backend, "gpu", "{algorithm}");
            assert_eq!(gpu.answer["number"], nonce);
            assert_eq!(gpu.implementation.as_deref(), Some("wgpu"));

            state.config.gpu_policy = GpuPolicy::CpuOnly;
            let cpu = state
                .run_classic("altcha".to_owned(), challenge.clone(), None, Instant::now())
                .await?;
            assert_eq!(cpu.backend, "cpu");
            assert_eq!(cpu.answer["number"], nonce);
            assert_eq!(cpu.implementation.as_deref(), Some("auto"));

            let forced = state
                .run_general_gpu("altcha".to_owned(), challenge, Instant::now())
                .await?;
            assert_eq!(forced.backend, "gpu");
            assert_eq!(forced.answer["number"], nonce);
            state.config.gpu_policy = GpuPolicy::PreferGpu;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical WGPU adapter"]
async fn linear_sha1_gpu_first_cpu_only_and_explicit_override() -> anyhow::Result<()> {
    use sha1::{Digest, Sha1};

    let mut state = AppState::initialize(Config {
        cpu_workers: 2,
        cpu_workers_per_request: 1,
        kdf_memory_budget_mib: Some(128),
        cpu_calibration_hashes: 8_192,
        gpu_calibration_hashes: 8_192,
        gpu_enabled: true,
        gpu_policy: GpuPolicy::PreferGpu,
        routing_calibration: None,
    })
    .await?;
    assert!(state.general_gpu_operational);
    let seed = "72BBE3B063367AB2F323D46D2ADA8A8A29BB5B58";
    let challenge = serde_json::json!({ "seed": seed });
    let verify = |response: &SolveResponse| -> anyhow::Result<()> {
        let nonce = response.answer["nonce"]
            .as_u64()
            .context("Linear GPU nonce must be an integer")?;
        let cookie = format!("yX3={seed}{nonce}");
        assert_eq!(response.answer["cookie"], cookie);
        assert_eq!(response.answer["cookie_name"], "yX3");
        let digest = Sha1::digest(format!("{seed}{nonce}").as_bytes());
        assert_eq!(&digest[7..9], &[0xb0, 0x0b]);
        Ok(())
    };
    let gpu = state
        .run_classic("linear".to_owned(), challenge.clone(), None, Instant::now())
        .await?;
    assert_eq!(gpu.backend, "gpu");
    verify(&gpu)?;

    state.config.gpu_policy = GpuPolicy::CpuOnly;
    let cpu = state
        .run_classic("linear".to_owned(), challenge.clone(), None, Instant::now())
        .await?;
    assert_eq!(cpu.backend, "cpu");
    assert_eq!(cpu.answer["nonce"], 29_224);
    verify(&cpu)?;

    let forced = state
        .run_general_gpu("linear".to_owned(), challenge, Instant::now())
        .await?;
    assert_eq!(forced.backend, "gpu");
    verify(&forced)?;
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
