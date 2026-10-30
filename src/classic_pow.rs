use std::{
    num::{NonZeroU8, NonZeroU32},
    time::Instant,
};

use altcha::{
    Challenge as AltchaChallenge, CounterMode as AltchaCounterMode,
    SolveChallengeOptions as AltchaSolveOptions,
};
use anyhow::{Context, anyhow, bail, ensure};
use argon2::{
    Algorithm as Argon2Algorithm, Argon2, Params as Argon2Params, Version as Argon2Version,
};
use hashwx::HashWx;
use num_bigint::BigUint;
use pbkdf2::pbkdf2_hmac;
use pow_buster::{
    AltchaSha256Solver, DecimalSolver,
    adapter::{
        capjs::ChallengeDescriptor as CapJsChallenge,
        cerberus::ChallengeDescriptor as CerberusChallenge, goaway::GoAwayConfig,
        mcaptcha::PoWConfig,
    },
    build_mcaptcha_prefix, compute_mask_goaway, compute_mask_zero_nibbles, compute_target_mcaptcha,
    extract128_be,
    message::{AltchaMessage, CapJSEmitter, CerberusMessage, DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256, Sha384, Sha512};

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
struct HaphashChallenge {
    prefix: String,
    difficulty: NonZeroU8,
}

#[derive(Debug, Deserialize)]
struct CapV2Envelope {
    token: String,
    format: u8,
    challenges: Vec<CapV2Challenge>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "protocol")]
enum CapV2Challenge {
    #[serde(rename = "sha256-pow")]
    Sha256Pow { payload: CapV2Sha256 },
    #[serde(rename = "rsw")]
    Rsw { payload: CapV2Rsw },
    #[serde(rename = "hashwx")]
    HashWx { payload: CapV2HashWx },
    #[serde(rename = "instrumentation")]
    Instrumentation { payload: Value },
}

#[derive(Debug, Deserialize)]
struct CapV2Sha256 {
    salt: String,
    target: String,
}

#[derive(Debug, Deserialize)]
struct CapV2Rsw {
    #[serde(rename = "N")]
    modulus: String,
    x: String,
    t: u32,
}

#[derive(Debug, Deserialize)]
struct CapV2HashWx {
    c: String,
    d: u64,
    n: u32,
}

pub fn solve(kind: &str, challenge: Value, implementation: Option<&str>) -> anyhow::Result<Result> {
    if kind == "altcha"
        && let Some(label @ ("reference" | "altcha-rust")) = implementation
    {
        return solve_altcha_official_only(challenge, label);
    }
    let implementation = Implementation::parse(implementation.unwrap_or("auto"))?;
    match kind {
        "cerberus" => solve_cerberus(challenge, implementation),
        "goaway" => solve_goaway(challenge, implementation),
        "capjs" => solve_capjs(challenge, implementation),
        "mcaptcha" => solve_mcaptcha(challenge, implementation),
        "haphash" => solve_haphash(challenge, implementation),
        "altcha" => solve_altcha(challenge, implementation),
        "linear" => crate::linear_pow::solve(challenge, implementation),
        _ => bail!("unsupported classic solver: {kind}"),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Implementation {
    Auto,
    Safe,
    Avx2,
    ShaNi,
    Avx512,
}

impl Implementation {
    pub(crate) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "safe" => Ok(Self::Safe),
            "avx2" => Ok(Self::Avx2),
            "sha-ni" => Ok(Self::ShaNi),
            "avx512" => Ok(Self::Avx512),
            other => bail!("unsupported classic implementation: {other}"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Avx2 => "avx2",
            Self::ShaNi => "sha-ni",
            Self::Avx512 => "avx512",
        }
    }
}

pub(crate) trait DynPow {
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
        Solver::attempted_nonces(self)
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

pub(crate) fn binary_solver(
    implementation: Implementation,
    message: pow_buster::message::BinaryMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "binary SHA-256 has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(pow_buster::BinarySolver::from(message))),
        Implementation::Safe => Ok(Box::new(pow_buster::solver::safe::BinarySolver::from(
            message,
        ))),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(pow_buster::solver::avx2::BinarySolver::from(
            message,
        ))),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(pow_buster::solver::avx512::BinarySolver::from(
            message,
        ))),
        _ => bail!("implementation unavailable for binary SHA-256"),
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

fn solve_cap_decimal_cpu(
    prefix: &[u8],
    target: u64,
    mask: u64,
    implementation: Implementation,
) -> anyhow::Result<(f64, u64)> {
    let mut attempted = 0_u64;
    for bank in 0..=u32::MAX {
        let Some((message, fixup)) = DecimalMessage::new_f64(prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(u64::MAX);
        let result = solver.solve(target, SOLVE_TYPE_MASK, mask);
        attempted = attempted.saturating_add(solver.attempted());
        if let Some((nonce, _)) = result {
            let nonce = fixup
                .map(|value| value.fixup(nonce))
                .unwrap_or(nonce as f64);
            return Ok((nonce, attempted));
        }
    }
    bail!("Cap.js SHA-256 search exhausted the lossless JavaScript-number domain")
}

#[cfg(feature = "gpu")]
async fn solve_cap_decimal_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    prefix: &[u8],
    target: u64,
    mask: u64,
) -> anyhow::Result<(f64, u64)> {
    let mut attempted = 0_u64;
    for bank in 0..=u32::MAX {
        let Some((message, fixup)) = DecimalMessage::new_f64(prefix, bank) else {
            break;
        };
        let mut solver = pow_buster::gpu::DecimalSolver::from(message);
        solver.set_limit(u64::MAX);
        let result = solver.solve::<SOLVE_TYPE_MASK>(gpu, target, mask).await?;
        attempted = attempted.saturating_add(solver.attempted_nonces());
        if let Some(solution) = result {
            let nonce = fixup
                .map(|value| value.fixup(solution.nonce))
                .unwrap_or(solution.nonce as f64);
            return Ok((nonce, attempted));
        }
    }
    bail!("Cap.js SHA-256 GPU search exhausted the lossless JavaScript-number domain")
}

fn solve_capjs(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    if challenge.get("format").and_then(Value::as_u64) == Some(2) {
        return solve_capjs_v2(challenge, implementation);
    }
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
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, attempted) = solve_cap_decimal_cpu(&salt, target, mask, implementation)?;
        attempted_nonces = attempted_nonces.saturating_add(attempted);
        solutions.push(nonce);
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

fn cap_v2_prefix(target: &str) -> anyhow::Result<(u64, u64)> {
    ensure!(
        !target.is_empty(),
        "Cap.js SHA-256 target must not be empty"
    );
    ensure!(target.len() <= 16, "Cap.js SHA-256 target exceeds 64 bits");
    ensure!(
        target.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Cap.js SHA-256 target must be hexadecimal"
    );
    let bits = u32::try_from(target.len())?.saturating_mul(4);
    let value = u64::from_str_radix(target, 16).context("invalid Cap.js SHA-256 target")?;
    let shift = 64_u32.saturating_sub(bits);
    let mask = if bits == 64 {
        u64::MAX
    } else {
        u64::MAX << shift
    };
    Ok((value << shift, mask))
}

fn solve_cap_v2_sha256(
    payload: &CapV2Sha256,
    implementation: Implementation,
) -> anyhow::Result<(Value, u64)> {
    let (target, mask) = cap_v2_prefix(&payload.target)?;
    let (nonce, attempted) =
        solve_cap_decimal_cpu(payload.salt.as_bytes(), target, mask, implementation)?;
    Ok((json!({ "nonce": nonce }), attempted))
}

fn solve_cap_v2_rsw(payload: &CapV2Rsw) -> anyhow::Result<Value> {
    ensure!(payload.t > 0, "Cap.js RSW iteration count must be positive");
    let modulus = BigUint::parse_bytes(payload.modulus.as_bytes(), 16)
        .context("invalid Cap.js RSW modulus")?;
    let mut value =
        BigUint::parse_bytes(payload.x.as_bytes(), 16).context("invalid Cap.js RSW input")?;
    ensure!(modulus.bits() > 1, "Cap.js RSW modulus is too small");
    value %= &modulus;
    for _ in 0..payload.t {
        value = (&value * &value) % &modulus;
    }
    Ok(json!({ "y": value.to_str_radix(16) }))
}

fn hashwx_seed(challenge: &[u8; 32], block: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(challenge);
    digest.update(block.to_le_bytes());
    digest.finalize().into()
}

fn solve_cap_v2_hashwx(payload: &CapV2HashWx) -> anyhow::Result<(Value, u64)> {
    ensure!(
        payload.d > 0 && payload.d <= 1_000_000_000,
        "Cap.js HashWX difficulty out of range"
    );
    ensure!(
        (1..=1_048_576).contains(&payload.n),
        "Cap.js HashWX nonces-per-hash out of range"
    );
    let challenge = hex::decode(&payload.c).context("invalid Cap.js HashWX challenge hex")?;
    let challenge: [u8; 32] = challenge
        .try_into()
        .map_err(|_| anyhow!("Cap.js HashWX challenge must be 32 bytes"))?;
    let target = u64::MAX / payload.d;
    let per_seed = u64::from(payload.n);
    let mut current_block = 0_u64;
    let mut hashwx = HashWx::compile(&hashwx_seed(&challenge, current_block))
        .context("compile Cap.js HashWX program")?;
    let mut attempted = 0_u64;
    let mut nonce = 0_u64;
    loop {
        let block = nonce / per_seed;
        if block != current_block {
            hashwx = HashWx::compile(&hashwx_seed(&challenge, block))
                .context("compile Cap.js HashWX program")?;
            current_block = block;
        }
        attempted = attempted.saturating_add(1);
        if hashwx.hash(nonce) <= target {
            return Ok((json!({ "nonce": nonce.to_string() }), attempted));
        }
        if nonce == u64::MAX {
            break;
        }
        nonce += 1;
    }
    bail!("Cap.js HashWX search exhausted the u64 nonce space")
}

fn cap_v2_backend_name(sha_backend: Option<&str>, used_hashwx: bool, used_rsw: bool) -> String {
    let mut backends = Vec::with_capacity(3);
    if let Some(sha_backend) = sha_backend {
        backends.push(sha_backend);
    }
    if used_hashwx {
        backends.push("hashwx-native");
    }
    if used_rsw {
        backends.push("rsw");
    }
    backends.join("+")
}

fn solve_capjs_v2(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let envelope: CapV2Envelope =
        serde_json::from_value(challenge).context("invalid Cap.js format-2 challenge")?;
    ensure!(envelope.format == 2, "invalid Cap.js format-2 marker");
    ensure!(
        !envelope.challenges.is_empty(),
        "Cap.js format-2 challenge list is empty"
    );
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(envelope.challenges.len());
    let mut used_sha256 = false;
    let mut used_hashwx = false;
    let mut used_rsw = false;
    for challenge in &envelope.challenges {
        let solution = match challenge {
            CapV2Challenge::Sha256Pow { payload } => {
                used_sha256 = true;
                let (solution, attempted) = solve_cap_v2_sha256(payload, implementation)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Rsw { payload } => {
                used_rsw = true;
                solve_cap_v2_rsw(payload)?
            }
            CapV2Challenge::HashWx { payload } => {
                used_hashwx = true;
                let (solution, attempted) = solve_cap_v2_hashwx(payload)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Instrumentation { payload } => {
                let _ = payload;
                bail!(
                    "Cap.js instrumentation is a browser-environment probe, not a proof-of-work mode"
                )
            }
        };
        solutions.push(solution);
    }
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let implementation_name = cap_v2_backend_name(
        used_sha256.then_some(implementation.name()),
        used_hashwx,
        used_rsw,
    );
    Ok(Result {
        answer: json!({
            "token": envelope.token,
            "solutions": solutions,
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation_name,
    })
}

fn solve_haphash(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let challenge: HaphashChallenge =
        serde_json::from_value(challenge).context("invalid haphash challenge")?;
    ensure!(
        challenge.difficulty.get() <= 16,
        "haphash difficulty above 16 nibbles is unsupported"
    );
    let prefix = challenge.prefix.as_bytes();
    ensure!(!prefix.is_empty(), "haphash prefix must not be empty");
    let mask = compute_mask_zero_nibbles(challenge.difficulty);
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve(0, SOLVE_TYPE_MASK, mask);
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if let Some((nonce, _)) = result {
            return Ok(Result {
                answer: json!({
                    "nonce": nonce.to_string(),
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms: started.elapsed().as_secs_f64() * 1000.0,
                implementation: implementation.name().to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("haphash search exhausted configured limit"))
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

fn altcha_counter_mode(challenge: &Value) -> anyhow::Result<AltchaCounterMode> {
    match challenge.get("counterMode").and_then(Value::as_str) {
        None | Some("uint32") => Ok(AltchaCounterMode::Uint32),
        Some("string") => Ok(AltchaCounterMode::String),
        Some(other) => bail!("unsupported ALTCHA counterMode: {other}"),
    }
}

fn solve_altcha_official_only(challenge: Value, implementation: &str) -> anyhow::Result<Result> {
    ensure!(
        challenge.get("parameters").is_some(),
        "ALTCHA {implementation} backend supports v2 only"
    );
    let counter_mode = altcha_counter_mode(&challenge)?;
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    solve_altcha_official(&challenge, counter_mode, implementation)
}
fn solve_altcha(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    if challenge.get("parameters").is_none() {
        return solve_altcha_v1(challenge, implementation);
    }
    let counter_mode = altcha_counter_mode(&challenge)?;
    let official: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;

    if counter_mode == AltchaCounterMode::String {
        ensure!(
            matches!(implementation, Implementation::Auto | Implementation::Safe),
            "ALTCHA string-counter mode supports auto/safe implementations only"
        );
        return solve_altcha_generic(&official, counter_mode);
    }
    if let Some(result) = try_solve_altcha_fast(&official, implementation)? {
        return Ok(result);
    }
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "this ALTCHA algorithm supports auto/safe or altcha-rust implementations only"
    );
    solve_altcha_generic(&official, counter_mode)
}

fn try_solve_altcha_fast(
    challenge: &AltchaChallenge,
    implementation: Implementation,
) -> anyhow::Result<Option<Result>> {
    let Some((message, target, mask, key_length)) = altcha_fast_case(challenge)? else {
        return Ok(None);
    };
    let mut solver = altcha_solver(implementation, message)?;
    solver.set_limit(u64::from(u32::MAX) + 1);
    let started = Instant::now();
    let (counter, key) = solver
        .solve(target, SOLVE_TYPE_MASK, mask)
        .context("ALTCHA optimized search exhausted the u32 counter domain")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let derived_key = encode_words_be(key);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Some(Result {
        answer: json!({
            "counter": counter,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted(),
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    }))
}

fn altcha_fast_case(
    challenge: &AltchaChallenge,
) -> anyhow::Result<Option<(AltchaMessage, u64, u64, usize)>> {
    let parameters = &challenge.parameters;
    if !matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256")
        || parameters.cost == 0
        || parameters.key_length == 0
        || parameters.key_length > 32
        || parameters.key_prefix.is_empty()
        || !parameters.key_prefix.len().is_multiple_of(2)
        || parameters.key_prefix.len() > 8
    {
        return Ok(None);
    }
    let nonce: [u8; 16] = match hex::decode(&parameters.nonce)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
    {
        Some(value) => value,
        None => return Ok(None),
    };
    let salt: [u8; 16] = match hex::decode(&parameters.salt)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
    {
        Some(value) => value,
        None => return Ok(None),
    };
    let Some(cost) = NonZeroU32::new(parameters.cost) else {
        return Ok(None);
    };
    let Ok(key_length_u32) = u32::try_from(parameters.key_length) else {
        return Ok(None);
    };
    let Some(key_length) = NonZeroU32::new(key_length_u32) else {
        return Ok(None);
    };
    let (target, mask) = match prefix_target(&parameters.key_prefix) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let message = AltchaMessage {
        nonce,
        salt,
        cost,
        pbkdf2: parameters.algorithm == "PBKDF2/SHA-256",
        key_length,
    };
    Ok(Some((message, target, mask, parameters.key_length)))
}
const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

struct AltchaHexPrefix(Vec<u8>);

impl AltchaHexPrefix {
    fn parse(prefix: &str) -> anyhow::Result<Self> {
        let nibbles = prefix
            .bytes()
            .map(|byte| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                _ => bail!("ALTCHA keyPrefix must be hexadecimal"),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self(nibbles))
    }

    fn matches(&self, derived: &[u8]) -> bool {
        self.0.len() <= derived.len() * 2
            && self.0.iter().enumerate().all(|(index, expected)| {
                let byte = derived[index / 2];
                let actual = if index.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                actual == *expected
            })
    }
}

fn altcha_iterative_sha<D: Digest>(initial: &[u8], iterations: u32, key_length: usize) -> Vec<u8> {
    let mut data = initial.to_vec();
    for _ in 0..iterations.max(1) {
        data = D::digest(&data).to_vec();
    }
    data.truncate(key_length.min(data.len()));
    data
}

fn altcha_derive_key(
    parameters: &altcha::ChallengeParameters,
    salt: &[u8],
    password: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let key_length = parameters.key_length;
    let algorithm = parameters.algorithm.to_ascii_uppercase();
    match algorithm.as_str() {
        "SHA-256" | "SHA-384" | "SHA-512" => {
            let initial: Vec<_> = salt.iter().chain(password).copied().collect();
            match algorithm.as_str() {
                "SHA-384" => Ok(altcha_iterative_sha::<Sha384>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
                "SHA-512" => Ok(altcha_iterative_sha::<Sha512>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
                _ => Ok(altcha_iterative_sha::<Sha256>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
            }
        }
        "PBKDF2/SHA-256" | "PBKDF2/SHA-384" | "PBKDF2/SHA-512" => {
            let mut out = vec![0_u8; key_length];
            match algorithm.as_str() {
                "PBKDF2/SHA-384" => {
                    pbkdf2_hmac::<Sha384>(password, salt, parameters.cost, &mut out)
                }
                "PBKDF2/SHA-512" => {
                    pbkdf2_hmac::<Sha512>(password, salt, parameters.cost, &mut out)
                }
                _ => pbkdf2_hmac::<Sha256>(password, salt, parameters.cost, &mut out),
            }
            Ok(out)
        }
        "SCRYPT" => {
            let n = parameters.cost;
            let log_n = n
                .checked_ilog2()
                .filter(|_| n.is_power_of_two())
                .context("ALTCHA scrypt cost must be a power of two")?;
            let params = scrypt::Params::new(
                u8::try_from(log_n).context("ALTCHA scrypt log2(N) exceeds u8")?,
                parameters.memory_cost.unwrap_or(8),
                parameters.parallelism.unwrap_or(1),
                key_length,
            )
            .context("invalid ALTCHA scrypt parameters")?;
            let mut out = vec![0_u8; key_length];
            scrypt::scrypt(password, salt, &params, &mut out)
                .context("ALTCHA scrypt derivation failed")?;
            Ok(out)
        }
        "ARGON2ID" => {
            let params = Argon2Params::new(
                parameters
                    .memory_cost
                    .context("ALTCHA Argon2id requires memoryCost")?,
                parameters.cost,
                parameters.parallelism.unwrap_or(1),
                Some(key_length),
            )
            .context("invalid ALTCHA Argon2id parameters")?;
            let argon2 = Argon2::new(Argon2Algorithm::Argon2id, Argon2Version::V0x13, params);
            let mut out = vec![0_u8; key_length];
            argon2
                .hash_password_into(password, salt, &mut out)
                .context("ALTCHA Argon2id derivation failed")?;
            Ok(out)
        }
        other => bail!("unsupported ALTCHA algorithm: {other}"),
    }
}

fn altcha_generic_result(
    counter: u64,
    derived: Vec<u8>,
    started: Instant,
    implementation: &str,
) -> Result {
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Result {
        answer: json!({
            "counter": counter,
            "derivedKey": hex::encode(derived),
            "time": solve_ms,
            "attempted_nonces": counter.saturating_add(1),
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    }
}

fn solve_altcha_generic(
    challenge: &AltchaChallenge,
    counter_mode: AltchaCounterMode,
) -> anyhow::Result<Result> {
    let parameters = &challenge.parameters;
    let nonce = hex::decode(&parameters.nonce).context("invalid ALTCHA nonce hex")?;
    let salt = hex::decode(&parameters.salt).context("invalid ALTCHA salt hex")?;
    let prefix = AltchaHexPrefix::parse(&parameters.key_prefix)?;
    let started = Instant::now();
    let mut password = Vec::with_capacity(nonce.len() + 16);

    match counter_mode {
        AltchaCounterMode::Uint32 => {
            for counter in 0..=u32::MAX {
                password.clear();
                password.extend_from_slice(&nonce);
                password.extend_from_slice(&counter.to_be_bytes());
                let derived = altcha_derive_key(parameters, &salt, &password)?;
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        u64::from(counter),
                        derived,
                        started,
                        "altcha-generic",
                    ));
                }
            }
            bail!("ALTCHA uint32-counter search exhausted the u32 domain")
        }
        AltchaCounterMode::String => {
            let mut counter_buffer = itoa::Buffer::new();
            for counter in 0..=JS_MAX_SAFE_INTEGER {
                password.clear();
                password.extend_from_slice(&nonce);
                password.extend_from_slice(counter_buffer.format(counter).as_bytes());
                let derived = altcha_derive_key(parameters, &salt, &password)?;
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        counter,
                        derived,
                        started,
                        "altcha-string",
                    ));
                }
            }
            bail!("ALTCHA string-counter search exhausted Number.MAX_SAFE_INTEGER")
        }
    }
}

fn solve_altcha_official(
    challenge: &AltchaChallenge,
    counter_mode: AltchaCounterMode,
    implementation: &str,
) -> anyhow::Result<Result> {
    let started = Instant::now();
    let mut options = AltchaSolveOptions::new(challenge);
    options.counter_mode = counter_mode;
    options.timeout_ms = 0;
    let solution = altcha::solve_challenge(options)
        .context("ALTCHA official Rust solver failed")?
        .context("ALTCHA search exhausted u32 counter space")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "counter": solution.counter,
            "derivedKey": solution.derived_key,
            "time": solution.time.unwrap_or(solve_ms),
            "attempted_nonces": u64::from(solution.counter) + 1,
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    })
}

#[derive(Debug, Deserialize)]
struct AltchaV1Challenge {
    algorithm: String,
    challenge: String,
    maxnumber: Option<u32>,
    salt: String,
    signature: String,
}

fn solve_altcha_v1(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "ALTCHA v1 supports auto/safe implementations only"
    );
    let challenge: AltchaV1Challenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v1 challenge")?;
    ensure!(
        matches!(
            challenge.algorithm.as_str(),
            "SHA-1" | "SHA-256" | "SHA-512"
        ),
        "unsupported ALTCHA v1 algorithm: {}",
        challenge.algorithm
    );
    ensure!(
        !challenge.signature.is_empty(),
        "ALTCHA v1 signature must not be empty"
    );
    let max = challenge.maxnumber.unwrap_or(1_000_000);
    let started = Instant::now();
    for number in 0..=max {
        let input = format!("{}{number}", challenge.salt);
        let candidate = match challenge.algorithm.as_str() {
            "SHA-1" => {
                use sha1::Digest as _;
                hex::encode(sha1::Sha1::digest(input.as_bytes()))
            }
            "SHA-256" => hex::encode(Sha256::digest(input.as_bytes())),
            "SHA-512" => hex::encode(sha2::Sha512::digest(input.as_bytes())),
            _ => unreachable!(),
        };
        if candidate == challenge.challenge {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({ "number": number, "took": solve_ms }),
                solve_ms,
                implementation: implementation.name().to_owned(),
            });
        }
    }
    bail!("ALTCHA v1 search exhausted maxnumber")
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
        "haphash" => solve_haphash_gpu(gpu, challenge).await,
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
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
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
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
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
async fn solve_capjs_v2_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let envelope: CapV2Envelope =
        serde_json::from_value(challenge).context("invalid Cap.js format-2 challenge")?;
    ensure!(envelope.format == 2, "invalid Cap.js format-2 marker");
    ensure!(
        !envelope.challenges.is_empty(),
        "Cap.js format-2 challenge list is empty"
    );

    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(envelope.challenges.len());
    let mut used_sha256 = false;
    let mut used_hashwx = false;
    let mut used_rsw = false;

    for challenge in &envelope.challenges {
        let solution = match challenge {
            CapV2Challenge::Sha256Pow { payload } => {
                used_sha256 = true;
                let (target, mask) = cap_v2_prefix(&payload.target)?;
                let (nonce, attempted) =
                    solve_cap_decimal_gpu(gpu, payload.salt.as_bytes(), target, mask).await?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                json!({ "nonce": nonce })
            }
            CapV2Challenge::Rsw { payload } => {
                used_rsw = true;
                solve_cap_v2_rsw(payload)?
            }
            CapV2Challenge::HashWx { payload } => {
                used_hashwx = true;
                let (solution, attempted) = solve_cap_v2_hashwx(payload)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Instrumentation { payload } => {
                let _ = payload;
                bail!(
                    "Cap.js instrumentation is a browser-environment probe, not a proof-of-work mode"
                )
            }
        };
        solutions.push(solution);
    }

    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "token": envelope.token,
            "solutions": solutions,
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: cap_v2_backend_name(used_sha256.then_some("wgpu"), used_hashwx, used_rsw),
    })
}

#[cfg(feature = "gpu")]
async fn solve_capjs_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    if challenge.get("format").and_then(Value::as_u64) == Some(2) {
        return solve_capjs_v2_gpu(gpu, challenge).await;
    }
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
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, attempted) = solve_cap_decimal_gpu(gpu, &salt, target, mask).await?;
        attempted_nonces = attempted_nonces.saturating_add(attempted);
        solutions.push(nonce);
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
async fn solve_haphash_gpu(
    gpu: &mut pow_buster::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let challenge: HaphashChallenge =
        serde_json::from_value(challenge).context("invalid haphash challenge")?;
    ensure!(
        challenge.difficulty.get() <= 16,
        "haphash difficulty above 16 nibbles is unsupported"
    );
    let prefix = challenge.prefix.as_bytes();
    ensure!(!prefix.is_empty(), "haphash prefix must not be empty");
    let mask = compute_mask_zero_nibbles(challenge.difficulty);
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(prefix, bank) else {
            break;
        };
        let mut solver = pow_buster::gpu::DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve::<SOLVE_TYPE_MASK>(gpu, 0, mask).await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
        if let Some(solution) = result {
            return Ok(Result {
                answer: json!({
                    "nonce": solution.nonce.to_string(),
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms: started.elapsed().as_secs_f64() * 1000.0,
                implementation: "wgpu".to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("haphash GPU search exhausted configured limit"))
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
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
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
    ensure!(
        altcha_counter_mode(&challenge)? == AltchaCounterMode::Uint32,
        "ALTCHA WGPU supports uint32 counter mode only"
    );
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let (message, target, mask, key_length) = altcha_fast_case(&challenge)?
        .context("ALTCHA challenge shape is unsupported by the WGPU fast path")?;
    let mut solver = pow_buster::gpu::AltchaSha256Solver::from(message);
    solver.set_limit(DEFAULT_LIMIT);
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let derived_key = encode_words_be(solution.hash);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Result {
        answer: json!({
            "counter": solution.nonce,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted_nonces(),
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
    fn haphash_solver_returns_nonce_for_protocol_prefix() {
        let result = solve_haphash(
            json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":1}),
            Implementation::Safe,
        )
        .unwrap();
        assert!(result.answer["nonce"].is_string());
    }

    #[test]
    fn altcha_fast_path_declines_nonstandard_nonce_shape() {
        let challenge = AltchaChallenge {
            parameters: altcha::ChallengeParameters {
                algorithm: "SHA-256".to_owned(),
                cost: 1,
                data: None,
                expires_at: None,
                key_length: 32,
                key_prefix: "00".to_owned(),
                key_signature: None,
                memory_cost: None,
                nonce: "00".to_owned(),
                parallelism: None,
                salt: "00".repeat(16),
            },
            signature: None,
        };
        assert!(
            try_solve_altcha_fast(&challenge, Implementation::Safe)
                .expect("fast-path eligibility check")
                .is_none()
        );
    }

    #[test]
    fn altcha_string_counter_derivation_covers_js_safe_integer_max() {
        let parameters = altcha::ChallengeParameters {
            algorithm: "SHA-256".to_owned(),
            cost: 2,
            data: None,
            expires_at: None,
            key_length: 32,
            key_prefix: "00".to_owned(),
            key_signature: None,
            memory_cost: None,
            nonce: hex::encode((0_u8..16).collect::<Vec<_>>()),
            parallelism: None,
            salt: hex::encode((16_u8..32).collect::<Vec<_>>()),
        };
        let nonce = hex::decode(&parameters.nonce).expect("fixed nonce");
        let salt = hex::decode(&parameters.salt).expect("fixed salt");
        let mut password = nonce;
        password.extend_from_slice(b"9007199254740991");
        let derived = altcha_derive_key(&parameters, &salt, &password)
            .expect("derive max-safe-integer string counter");
        assert_eq!(
            hex::encode(derived),
            "fb46d9456a5caf40eb17da502cdec45ac202cc4f30f50ee667bddb40904be548"
        );
    }

    #[test]
    fn altcha_generic_modes_match_upstream_empty_prefix_and_case_insensitive_algorithm() {
        let challenge = AltchaChallenge {
            parameters: altcha::ChallengeParameters {
                algorithm: "sha-256".to_owned(),
                cost: 1,
                data: None,
                expires_at: None,
                key_length: 32,
                key_prefix: String::new(),
                key_signature: None,
                memory_cost: None,
                nonce: "00".to_owned(),
                parallelism: None,
                salt: "00".repeat(16),
            },
            signature: None,
        };
        for (mode, implementation) in [
            (AltchaCounterMode::Uint32, "altcha-generic"),
            (AltchaCounterMode::String, "altcha-string"),
        ] {
            let result = solve_altcha_generic(&challenge, mode).expect("generic ALTCHA solve");
            assert_eq!(result.answer["counter"], 0);
            assert_eq!(result.implementation, implementation);
        }
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
