use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::Ordering},
};

use axum::{Json, extract::State};
use serde::Serialize;

use crate::{anubis_wasm, stats::Calibration};

use super::AppState;

#[derive(Debug, Serialize)]
pub struct SolverCapability {
    solver: &'static str,
    algorithms: &'static [&'static str],
    difficulty_min: u32,
    difficulty_max: u32,
    router: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    counter_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    counter_max: Option<u64>,
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
    argon2_memory_available_mib: Option<usize>,
    expected_new_solve_ms_by_difficulty: BTreeMap<u8, f64>,
    hashx_gpu_implementations: Vec<&'static str>,
    general: crate::stats::GeneralStats,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "queue ETA is approximate telemetry and does not require integer-exact f64 representation"
)]
const fn telemetry_work_as_f64(work: u64) -> f64 {
    work as f64
}

#[derive(Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "serialized stats expose independent backend capability/configuration flags"
)]
pub struct StatsConfig {
    cpu_workers: usize,
    argon2_workers: usize,
    altcha_worker_policy: &'static str,
    argon2_memory_budget_mib: Option<usize>,
    gpu_compiled: bool,
    gpu_enabled: bool,
    gpu_policy: &'static str,
    gpu_operational: bool,
    hashx_gpu_operational: bool,
    routing_calibration_loaded: bool,
}

/// Only list CPU implementations that exist in this build *and* can execute on
/// this host. This is also the route-discovery contract for external clients.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_implementations(sha_ni: bool) -> Vec<&'static str> {
    let mut implementations = vec!["auto", "safe"];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        implementations.push("avx2");
    }
    if sha_ni && std::arch::is_x86_feature_detected!("sha") {
        implementations.push("sha-ni");
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        implementations.push("avx512");
    }
    implementations
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_implementations(_sha_ni: bool) -> Vec<&'static str> {
    vec!["auto", "safe"]
}

#[expect(
    clippy::too_many_lines,
    reason = "stats response construction is a flat serialization map of solver capabilities"
)]
pub async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let queued_work = state.gpu_queue_work.load(Ordering::Relaxed);
    let gpu_queue_depth = state.gpu_queue_depth.load(Ordering::Relaxed);
    let active_work = state.gpu_active_work.load(Ordering::Relaxed);
    let work_ahead = queued_work.saturating_add(active_work);
    let mut expected = BTreeMap::new();
    for difficulty in 1..=16_u8 {
        let work = telemetry_work_as_f64(16_u64.saturating_pow(u32::from(difficulty)));
        let eta_ms = state.calibration.gpu_hashes_per_second.map_or_else(
            || work / state.calibration.cpu_hashes_per_second_per_worker * 1000.0,
            |gpu| (telemetry_work_as_f64(work_ahead) + work) / gpu * 1000.0,
        );
        expected.insert(difficulty, eta_ms);
    }
    let general = state.stats.read().await.snapshot();
    let mut anubis_sha_implementations = cpu_implementations(false);
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
            counter_mode: None,
            counter_max: None,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-cpu",
            algorithms: &["fast", "slow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-sha256",
            algorithms: &["sha256"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: anubis_sha_implementations,
        },
        SolverCapability {
            solver: "anubis-argon2id",
            algorithms: &["argon2id"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: vec![anubis_wasm::ARGON2_IMPLEMENTATION],
        },
        SolverCapability {
            solver: "anubis-hashx-jit",
            algorithms: &["hashx"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: Vec::new(),
        },
        SolverCapability {
            solver: "anubis-hashx-auto",
            algorithms: &["hashx"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: true,
            counter_mode: None,
            counter_max: None,
            implementations: Vec::new(),
        },
    ];
    let classic_implementations = |sha_ni: bool, wgpu: bool| {
        let mut implementations = cpu_implementations(sha_ni);
        if wgpu && state.general_gpu_operational {
            implementations.push("wgpu");
        }
        implementations
    };

    for solver in [
        "cerberus",
        "goaway",
        "capjs",
        "mcaptcha",
        "haphash",
        "altcha",
        "friendlycaptcha",
    ] {
        available_solvers.push(solver);
    }

    let sha64_implementations = || {
        let mut implementations = vec!["auto", "safe"];
        if state.general_gpu_has_native_u64() {
            implementations.push("wgpu");
        }
        implementations
    };

    solver_capabilities.extend([
        SolverCapability {
            solver: "friendlycaptcha",
            algorithms: &["blake2b-256"],
            difficulty_min: 0,
            difficulty_max: 255,
            router: false,
            counter_mode: Some("indexed-u32-le"),
            counter_max: Some(u64::from(u32::MAX)),
            implementations: vec!["auto", "safe"],
        },
        SolverCapability {
            solver: "cerberus",
            algorithms: &["blake3"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: classic_implementations(false, true),
        },
        SolverCapability {
            solver: "goaway",
            algorithms: &["sha256"],
            difficulty_min: 0,
            difficulty_max: 256,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: classic_implementations(true, true),
        },
        SolverCapability {
            solver: "capjs",
            algorithms: &["sha256", "sha256-pow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: classic_implementations(true, true),
        },
        SolverCapability {
            solver: "capjs",
            algorithms: &["rsw"],
            difficulty_min: 1,
            difficulty_max: u32::MAX,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: vec!["auto"],
        },
        SolverCapability {
            solver: "capjs",
            algorithms: &["hashwx"],
            difficulty_min: 1,
            difficulty_max: 1_000_000_000,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: vec!["auto"],
        },
        SolverCapability {
            solver: "mcaptcha",
            algorithms: &["sha256"],
            difficulty_min: 1,
            difficulty_max: u32::MAX,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: classic_implementations(true, true),
        },
        SolverCapability {
            solver: "haphash",
            algorithms: &["sha256"],
            difficulty_min: 0,
            difficulty_max: 64,
            router: false,
            counter_mode: None,
            counter_max: None,
            implementations: classic_implementations(true, true),
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha256", "pbkdf2-sha256"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_operational,
            counter_mode: Some("uint32"),
            counter_max: Some(u64::from(u32::MAX)),
            implementations: classic_implementations(false, true),
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha384", "sha512", "pbkdf2-sha384", "pbkdf2-sha512"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_has_native_u64(),
            counter_mode: Some("uint32"),
            counter_max: Some(u64::from(u32::MAX)),
            implementations: sha64_implementations(),
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["argon2id", "scrypt"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            counter_mode: Some("uint32"),
            counter_max: Some(u64::from(u32::MAX)),
            implementations: vec!["auto", "safe"],
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha256", "pbkdf2-sha256"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_operational,
            counter_mode: Some("string"),
            counter_max: Some(9_007_199_254_740_991),
            implementations: {
                let mut implementations = vec!["auto", "safe"];
                if state.general_gpu_operational {
                    implementations.push("wgpu");
                }
                implementations
            },
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha384", "sha512", "pbkdf2-sha384", "pbkdf2-sha512"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_has_native_u64(),
            counter_mode: Some("string"),
            counter_max: Some(9_007_199_254_740_991),
            implementations: sha64_implementations(),
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["argon2id", "scrypt"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: false,
            counter_mode: Some("string"),
            counter_max: Some(9_007_199_254_740_991),
            implementations: vec!["auto", "safe"],
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha1-v1", "sha256-v1"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_operational,
            counter_mode: None,
            counter_max: Some(u64::from(u32::MAX)),
            implementations: {
                let mut supported = vec!["auto", "safe"];
                if state.general_gpu_operational {
                    supported.push("wgpu");
                }
                supported
            },
        },
        SolverCapability {
            solver: "altcha",
            algorithms: &["sha512-v1"],
            difficulty_min: 1,
            difficulty_max: 64,
            router: state.general_gpu_has_native_u64(),
            counter_mode: None,
            counter_max: Some(u64::from(u32::MAX)),
            implementations: sha64_implementations(),
        },
    ]);
    available_solvers.push("linear");
    solver_capabilities.push(SolverCapability {
        solver: "linear",
        algorithms: &["sha1"],
        difficulty_min: 16,
        difficulty_max: 16,
        router: state.general_gpu_operational,
        counter_mode: None,
        counter_max: None,
        implementations: {
            let mut supported = vec!["auto"];
            if state.general_gpu_operational {
                supported.push("wgpu");
            }
            supported
        },
    });
    if state.gpu_operational {
        available_solvers.push("anubis-gpu");
        solver_capabilities.push(SolverCapability {
            solver: "anubis-gpu",
            algorithms: &["fast", "slow"],
            difficulty_min: 1,
            difficulty_max: 16,
            router: false,
            counter_mode: None,
            counter_max: None,
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
            counter_mode: None,
            counter_max: None,
            implementations: state.hashx_gpu_implementations.clone(),
        });
    }
    Json(StatsResponse {
        available_solvers,
        solver_capabilities,
        calibration: state.calibration.clone(),
        config: StatsConfig {
            cpu_workers: state.config.cpu_workers,
            argon2_workers: state.config.argon2_workers,
            altcha_worker_policy: match state.config.altcha_worker_policy {
                super::WorkerPolicy::Adaptive => "adaptive",
                super::WorkerPolicy::Fixed => "fixed",
            },
            argon2_memory_budget_mib: state.argon2_memory_capacity_mib,
            gpu_compiled: cfg!(feature = "gpu"),
            gpu_enabled: state.config.gpu_enabled,
            gpu_policy: state.config.gpu_policy.as_str(),
            gpu_operational: state.gpu_operational,
            hashx_gpu_operational: state.hashx_gpu_operational,
            routing_calibration_loaded: state.routing.is_some(),
        },
        gpu_queue_depth,
        gpu_queue_expected_work: queued_work,
        gpu_active_expected_work: active_work,
        cpu_workers_available: state.cpu_slots.available_permits(),
        argon2_memory_available_mib: state
            .argon2_memory_slots
            .as_ref()
            .map(|slots| slots.available_permits()),
        expected_new_solve_ms_by_difficulty: expected,
        hashx_gpu_implementations: state.hashx_gpu_implementations.clone(),
        general,
    })
}

#[cfg(test)]
mod tests {
    use super::cpu_implementations;

    #[test]
    fn advertised_cpu_backends_match_compiled_capabilities() {
        let regular = cpu_implementations(false);
        let sha = cpu_implementations(true);
        assert!(regular.starts_with(&["auto", "safe"]));
        assert!(sha.starts_with(&["auto", "safe"]));
        assert!(!regular.contains(&"sha-ni"));
    }
}
