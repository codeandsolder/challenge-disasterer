use std::{num::NonZeroU32, time::Instant};

use anyhow::{Context, anyhow, bail, ensure};
use pow_buster::{
    AltchaSha256Solver, DecimalSolver,
    adapter::{
        capjs::ChallengeDescriptor as CapJsChallenge,
        cerberus::ChallengeDescriptor as CerberusChallenge, goaway::GoAwayConfig,
        mcaptcha::PoWConfig,
    },
    build_mcaptcha_prefix, compute_mask_goaway, compute_target_mcaptcha, extract128_be,
    message::{AltchaMessage, CapJSEmitter, CerberusMessage, DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use serde::Deserialize;
use serde_json::{Value, json};

const DEFAULT_LIMIT: u64 = 4_000_000_000;

#[derive(Debug)]
pub struct Result {
    pub answer: Value,
    pub solve_ms: f64,
    pub implementation: String,
}

#[derive(Debug, Deserialize)]
struct McaptchaRequest {
    config: PoWConfig,
    key: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AltchaEnvelope {
    parameters: AltchaParameters,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AltchaParameters {
    algorithm: String,
    nonce: String,
    salt: String,
    cost: NonZeroU32,
    key_length: NonZeroU32,
    key_prefix: String,
}

pub fn solve(kind: &str, challenge: Value, implementation: Option<&str>) -> anyhow::Result<Result> {
    let implementation = Implementation::parse(implementation.unwrap_or("auto"))?;
    match kind {
        "cerberus" => solve_cerberus(challenge, implementation),
        "goaway" => solve_goaway(challenge, implementation),
        "capjs" => solve_capjs(challenge, implementation),
        "mcaptcha" => solve_mcaptcha(challenge, implementation),
        "altcha" => solve_altcha(challenge, implementation),
        _ => bail!("unsupported classic solver: {kind}"),
    }
}

#[derive(Clone, Copy, Debug)]
enum Implementation {
    Auto,
    Safe,
    Avx2,
    ShaNi,
    Avx512,
}

impl Implementation {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "safe" => Ok(Self::Safe),
            "avx2" => Ok(Self::Avx2),
            "sha-ni" => Ok(Self::ShaNi),
            "avx512" => Ok(Self::Avx512),
            other => bail!("unsupported classic implementation: {other}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Avx2 => "avx2",
            Self::ShaNi => "sha-ni",
            Self::Avx512 => "avx512",
        }
    }
}

trait DynPow {
    fn set_limit(&mut self, limit: u64);
    fn attempted(&self) -> u64;
    fn solve(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

impl<T> DynPow for T
where
    T: Solver<Output = [u32; 8]> + SolverDyn<Output = [u32; 8]>,
{
    fn set_limit(&mut self, limit: u64) {
        Solver::set_limit(self, limit);
    }
    fn attempted(&self) -> u64 {
        Solver::get_attempted_nonces(self)
    }
    fn solve(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])> {
        SolverDyn::solve_dyn(self, target, ty, mask)
    }
}

fn require_x86_feature(implementation: Implementation) -> anyhow::Result<()> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match implementation {
            Implementation::Avx2 if !std::arch::is_x86_feature_detected!("avx2") => {
                bail!("AVX2 unavailable on this CPU")
            }
            Implementation::ShaNi if !std::arch::is_x86_feature_detected!("sha") => {
                bail!("SHA-NI unavailable on this CPU")
            }
            #[cfg(target_arch = "x86_64")]
            Implementation::Avx512 if !std::arch::is_x86_feature_detected!("avx512f") => {
                bail!("AVX-512 unavailable on this CPU")
            }
            _ => Ok(()),
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    match implementation {
        Implementation::Auto | Implementation::Safe => Ok(()),
        _ => bail!("requested x86 implementation on a non-x86 host"),
    }
}

fn decimal_solver(
    implementation: Implementation,
    message: DecimalMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(DecimalSolver::from(message))),
        Implementation::Safe => Ok(Box::new(pow_buster::solver::safe::DecimalSolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(pow_buster::solver::avx2::DecimalSolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(pow_buster::solver::sha_ni::DecimalSolver::from(
            message,
        ))),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(pow_buster::solver::avx512::DecimalSolver::from(
            message,
        ))),
        #[allow(unreachable_patterns)]
        _ => bail!("implementation unavailable on this target"),
    }
}

fn goaway_solver(
    implementation: Implementation,
    message: GoAwayMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(pow_buster::GoAwaySolver::from(message))),
        Implementation::Safe => Ok(Box::new(pow_buster::solver::safe::GoAwaySolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(pow_buster::solver::avx2::GoAwaySolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(pow_buster::solver::sha_ni::GoAwaySolver::from(
            message,
        ))),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(pow_buster::solver::avx512::GoAwaySolver::from(
            message,
        ))),
        #[allow(unreachable_patterns)]
        _ => bail!("implementation unavailable on this target"),
    }
}

fn cerberus_solver(
    implementation: Implementation,
    message: CerberusMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "Cerberus has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(pow_buster::CerberusSolver::from(message))),
        Implementation::Safe => Ok(Box::new(pow_buster::solver::safe::CerberusSolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(pow_buster::solver::avx2::CerberusSolver::from(
            message,
        ))),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(pow_buster::solver::avx512::CerberusSolver::from(
            message,
        ))),
        _ => bail!("implementation unavailable for Cerberus"),
    }
}

fn altcha_solver(
    implementation: Implementation,
    message: AltchaMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "ALTCHA has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(AltchaSha256Solver::from(message))),
        Implementation::Safe => Ok(Box::new(
            pow_buster::solver::safe::AltchaSha256Solver::from(message),
        )),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(
            pow_buster::solver::avx2::AltchaSha256Solver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            pow_buster::solver::avx512::AltchaSha256Solver::from(message),
        )),
        _ => bail!("implementation unavailable for ALTCHA"),
    }
}

fn solve_cerberus(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    let started = Instant::now();
    let mut solution = None;
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = descriptor.build_msg(bank) else {
            break;
        };
        let mut solver = cerberus_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver.solve(0, SOLVE_TYPE_MASK, descriptor.mask());
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let (solution, hash) = solution.context("Cerberus search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let response = encode_words_le(hash);
    Ok(Result {
        answer: json!({
            "solution": solution.to_string(),
            "response": response,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn solve_goaway(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    ensure!(
        config.challenge().len() == 64,
        "go-away challenge must be exactly 64 characters"
    );
    let challenge_bytes: [u8; 64] = config
        .challenge()
        .as_bytes()
        .try_into()
        .context("go-away challenge must be exactly 64 bytes")?;
    let mut message =
        GoAwayMessage::new_hex(&challenge_bytes, 0).context("invalid go-away hex challenge")?;
    let mut attempted_nonces = 0_u64;
    let mut solution = None;
    let started = Instant::now();
    for high_word in 0_u32.. {
        message.set_high_word(high_word);
        let mut solver = goaway_solver(implementation, message.clone())?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver.solve(0, SOLVE_TYPE_MASK, compute_mask_goaway(config.difficulty()));
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let (nonce, hash) = solution.context("go-away search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;

    let token = format!("{}{:016x}", config.challenge(), nonce);
    let id = hash
        .into_iter()
        .take(4)
        .flat_map(u32::to_ne_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    Ok(Result {
        answer: json!({
            "nonce": nonce.to_string(),
            "token": token,
            "id": id,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn solve_capjs(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let rules = *descriptor.rules();
    let token = descriptor.token.clone();
    let emitter = CapJSEmitter::new(token.as_bytes());
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(rules.count);
    for index in 0..rules.count {
        let mut salt = vec![0_u8; rules.salt_length];
        let mut targets = [0_u32; 2];
        emitter.emit(&mut salt, &mut targets, index as u32 + 1);
        let mask = u64::MAX << (64 - u64::from(rules.difficulty) * 4);
        let (message, fixup) =
            DecimalMessage::new_f64(&salt, 0).context("unsupported Cap.js message shape")?;
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, _) = solver
            .solve(target, SOLVE_TYPE_MASK, mask)
            .context("Cap.js search exhausted configured limit")?;
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        solutions.push(fixup.map(|x| x.fixup(nonce)).unwrap_or(nonce as f64));
    }
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let answer = json!({
        "token": token,
        "solutions": solutions,
        "_meta": {
            "elapsed_us": (solve_ms * 1000.0) as u64,
            "attempted_nonces": attempted_nonces,
            "hashrate": if solve_ms > 0.0 { (attempted_nonces as f64 / (solve_ms / 1000.0)) as u64 } else { 0 },
        },
        "attempted_nonces_total": attempted_nonces,
    });
    Ok(Result {
        answer,
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn solve_mcaptcha(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let request: McaptchaRequest =
        serde_json::from_value(challenge).context("invalid mCaptcha request")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt);
    let target = compute_target_mcaptcha(u64::from(request.config.difficulty_factor));
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;

    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(&prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve(target, SOLVE_TYPE_GT, u64::MAX);
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if let Some((nonce, hash)) = result {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({
                    "string": request.config.string,
                    "result": extract128_be(hash).to_string(),
                    "nonce": nonce,
                    "key": request.key,
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms,
                implementation: implementation.name().to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }

    Err(anyhow!("mCaptcha search exhausted configured limit"))
}

fn solve_altcha(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let envelope: AltchaEnvelope =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let parameters = envelope.parameters;
    ensure!(
        matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256"),
        "unsupported ALTCHA algorithm: {}",
        parameters.algorithm
    );
    let nonce = decode_fixed_16("nonce", &parameters.nonce)?;
    let salt = decode_fixed_16("salt", &parameters.salt)?;
    let (target, mask) = prefix_target(&parameters.key_prefix)?;
    let message = AltchaMessage {
        nonce,
        salt,
        cost: parameters.cost,
        pbkdf2: parameters.algorithm == "PBKDF2/SHA-256",
        key_length: parameters.key_length,
    };
    let mut solver = altcha_solver(implementation, message)?;
    solver.set_limit(DEFAULT_LIMIT);
    let started = Instant::now();
    let (counter, key) = solver
        .solve(target, SOLVE_TYPE_MASK, mask)
        .context("ALTCHA search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let key_length = usize::try_from(parameters.key_length.get()).unwrap_or(usize::MAX);
    ensure!(key_length <= 32, "ALTCHA keyLength above 32 is unsupported");
    let derived_key = encode_words_be(key);
    let derived_key = &derived_key[..key_length * 2];

    Ok(Result {
        answer: json!({
            "counter": counter,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted(),
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn decode_fixed_16(label: &str, value: &str) -> anyhow::Result<[u8; 16]> {
    let bytes = hex::decode(value).with_context(|| format!("invalid ALTCHA {label} hex"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("ALTCHA {label} must be exactly 16 bytes"))
}

fn prefix_target(prefix: &str) -> anyhow::Result<(u64, u64)> {
    ensure!(
        !prefix.is_empty() && prefix.len().is_multiple_of(2),
        "ALTCHA keyPrefix must be non-empty whole-byte hex"
    );
    let bytes = hex::decode(prefix).context("invalid ALTCHA keyPrefix hex")?;
    ensure!(
        bytes.len() <= 4,
        "ALTCHA keyPrefix longer than 4 bytes needs the 64-bit comparison build"
    );
    let mut first = [0_u8; 8];
    first[..bytes.len()].copy_from_slice(&bytes);
    let target = u64::from_be_bytes(first);
    let bits = bytes.len() * 8;
    let mask = if bits == 0 {
        0
    } else {
        u64::MAX << (64 - bits)
    };
    Ok((target, mask))
}

fn encode_words_le(words: [u32; 8]) -> String {
    words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn encode_words_be(words: [u32; 8]) -> String {
    words
        .into_iter()
        .flat_map(u32::to_be_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(feature = "gpu")]
pub async fn solve_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    kind: &str,
    challenge: Value,
) -> anyhow::Result<Result> {
    match kind {
        "cerberus" => solve_cerberus_gpu(gpu, challenge).await,
        "goaway" => solve_goaway_gpu(gpu, challenge).await,
        "capjs" => solve_capjs_gpu(gpu, challenge).await,
        "mcaptcha" => solve_mcaptcha_gpu(gpu, challenge).await,
        "altcha" => solve_altcha_gpu(gpu, challenge).await,
        _ => bail!("unsupported classic GPU solver: {kind}"),
    }
}

#[cfg(feature = "gpu")]
async fn solve_cerberus_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    let started = Instant::now();
    let mut solution = None;
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = descriptor.build_msg(bank) else {
            break;
        };
        let mut solver = pow_buster::gpu::CerberusSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, 0, descriptor.mask())
            .await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.get_attempted_nonces());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("Cerberus GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "solution": solution.nonce.to_string(),
            "response": encode_words_le(solution.hash),
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(feature = "gpu")]
async fn solve_goaway_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    let challenge_bytes: [u8; 64] = config
        .challenge()
        .as_bytes()
        .try_into()
        .context("go-away challenge must be exactly 64 bytes")?;
    let mut message =
        GoAwayMessage::new_hex(&challenge_bytes, 0).context("invalid go-away hex challenge")?;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solution = None;
    for high_word in 0_u32.. {
        message.set_high_word(high_word);
        let mut solver = pow_buster::gpu::GoAwaySolver::from(message.clone());
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, 0, compute_mask_goaway(config.difficulty()))
            .await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.get_attempted_nonces());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("go-away GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let token = format!("{}{:016x}", config.challenge(), solution.nonce);
    let id = solution
        .hash
        .into_iter()
        .take(4)
        .flat_map(u32::to_ne_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(Result {
        answer: json!({
            "nonce": solution.nonce.to_string(),
            "token": token,
            "id": id,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(feature = "gpu")]
async fn solve_capjs_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let rules = *descriptor.rules();
    let token = descriptor.token.clone();
    let emitter = CapJSEmitter::new(token.as_bytes());
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(rules.count);
    for index in 0..rules.count {
        let mut salt = vec![0_u8; rules.salt_length];
        let mut targets = [0_u32; 2];
        emitter.emit(&mut salt, &mut targets, index as u32 + 1);
        let mask = u64::MAX << (64 - u64::from(rules.difficulty) * 4);
        let (message, fixup) =
            DecimalMessage::new_f64(&salt, 0).context("unsupported Cap.js message shape")?;
        let mut solver = pow_buster::gpu::DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
            .await?
            .context("Cap.js GPU search exhausted configured limit")?;
        attempted_nonces = attempted_nonces.saturating_add(solver.get_attempted_nonces());
        solutions.push(
            fixup
                .map(|x| x.fixup(solution.nonce))
                .unwrap_or(solution.nonce as f64),
        );
    }
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "token": token,
            "solutions": solutions,
            "_meta": {
                "elapsed_us": (solve_ms * 1000.0) as u64,
                "attempted_nonces": attempted_nonces,
                "hashrate": if solve_ms > 0.0 { (attempted_nonces as f64 / (solve_ms / 1000.0)) as u64 } else { 0 },
            },
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(feature = "gpu")]
async fn solve_mcaptcha_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let request: McaptchaRequest =
        serde_json::from_value(challenge).context("invalid mCaptcha request")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt);
    let target = compute_target_mcaptcha(u64::from(request.config.difficulty_factor));
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(&prefix, bank) else {
            break;
        };
        let mut solver = pow_buster::gpu::DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve::<SOLVE_TYPE_GT>(gpu, target, u64::MAX).await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.get_attempted_nonces());
        if let Some(solution) = result {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({
                    "string": request.config.string,
                    "result": extract128_be(solution.hash).to_string(),
                    "nonce": solution.nonce,
                    "key": request.key,
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms,
                implementation: "wgpu".to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("mCaptcha GPU search exhausted configured limit"))
}

#[cfg(feature = "gpu")]
async fn solve_altcha_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let envelope: AltchaEnvelope =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let parameters = envelope.parameters;
    ensure!(
        matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256"),
        "unsupported ALTCHA algorithm: {}",
        parameters.algorithm
    );
    let nonce = decode_fixed_16("nonce", &parameters.nonce)?;
    let salt = decode_fixed_16("salt", &parameters.salt)?;
    let (target, mask) = prefix_target(&parameters.key_prefix)?;
    let message = AltchaMessage {
        nonce,
        salt,
        cost: parameters.cost,
        pbkdf2: parameters.algorithm == "PBKDF2/SHA-256",
        key_length: parameters.key_length,
    };
    let mut solver = pow_buster::gpu::AltchaSha256Solver::from(message);
    solver.set_limit(DEFAULT_LIMIT);
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let key_length = usize::try_from(parameters.key_length.get()).unwrap_or(usize::MAX);
    ensure!(key_length <= 32, "ALTCHA keyLength above 32 is unsupported");
    let derived_key = encode_words_be(solution.hash);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Result {
        answer: json!({
            "counter": solution.nonce,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.get_attempted_nonces(),
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altcha_prefix_target_places_bytes_at_hash_front() {
        assert_eq!(
            prefix_target("12ab").unwrap(),
            (0x12ab_0000_0000_0000, 0xffff_0000_0000_0000)
        );
    }

    #[test]
    fn fixed_hex_rejects_wrong_length() {
        assert!(decode_fixed_16("nonce", "00").is_err());
    }

    #[test]
    fn protocol_nonce_json_types_match_wire_contracts() {
        let mcaptcha = solve_mcaptcha(
            json!({
                "config": {
                    "string": "schema-test",
                    "difficulty_factor": 1,
                    "salt": "schema-test-salt"
                },
                "key": "schema-test"
            }),
            Implementation::Auto,
        )
        .expect("easy mCaptcha solve");
        assert!(mcaptcha.answer["nonce"].is_u64());

        let goaway = solve_goaway(
            json!({
                "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "difficulty": 1
            }),
            Implementation::Auto,
        )
        .expect("easy go-away solve");
        assert!(goaway.answer["nonce"].is_string());
    }
}
