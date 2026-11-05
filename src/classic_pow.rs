use std::{
    num::{NonZeroU8, NonZeroU64},
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow, bail, ensure};
use challenge_solvers::{
    DecimalSolver, build_mcaptcha_prefix,
    capjs::ChallengeDescriptor as CapJsChallenge,
    cerberus::ChallengeDescriptor as CerberusChallenge,
    compute_mask_goaway, compute_mask_zero_nibbles, compute_target_mcaptcha, extract128_be,
    goaway::GoAwayConfig,
    mcaptcha::PoWConfig,
    message::{CapJSEmitter, CerberusMessage, DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LEADING_ZERO_BITS, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use hashwx::HashWx;
use num_bigint::BigUint;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod altcha;
#[cfg(feature = "gpu")]
pub mod gpu;

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
    difficulty: u8,
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
    let implementation = Implementation::parse(implementation.unwrap_or("auto"))?;
    match kind {
        "cerberus" => solve_cerberus(challenge, implementation),
        "goaway" => solve_goaway(challenge, implementation),
        "capjs" => solve_capjs(challenge, implementation),
        "mcaptcha" => solve_mcaptcha(challenge, implementation),
        "haphash" => solve_haphash(challenge, implementation),
        "altcha" => altcha::solve(challenge, implementation),
        "linear" => crate::linear_pow::solve(challenge, implementation),
        _ => bail!("unsupported classic solver: {kind}"),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Implementation {
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

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Avx2 => "avx2",
            Self::ShaNi => "sha-ni",
            Self::Avx512 => "avx512",
        }
    }
}

pub trait DynPow {
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
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::DecimalSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2DecimalSolver::from(message),
        )),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(
            challenge_solvers::solver::sha_ni::DecimalSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512DecimalSolver::from(message),
        )),
        #[cfg_attr(
            target_arch = "x86_64",
            expect(
                unreachable_patterns,
                reason = "all solver implementations in this match are available on x86_64"
            )
        )]
        _ => bail!("implementation unavailable on this target"),
    }
}

pub fn binary_solver(
    implementation: Implementation,
    message: challenge_solvers::message::BinaryMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "binary SHA-256 has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(challenge_solvers::BinarySolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::BinarySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2BinarySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512BinarySolver::from(message),
        )),
        _ => bail!("implementation unavailable for binary SHA-256"),
    }
}

fn goaway_solver(
    implementation: Implementation,
    message: GoAwayMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(challenge_solvers::GoAwaySolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::GoAwaySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2GoAwaySolver::from(message),
        )),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(
            challenge_solvers::solver::sha_ni::GoAwaySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512GoAwaySolver::from(message),
        )),
        #[cfg_attr(
            target_arch = "x86_64",
            expect(
                unreachable_patterns,
                reason = "all solver implementations in this match are available on x86_64"
            )
        )]
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
        Implementation::Auto => Ok(Box::new(challenge_solvers::CerberusSolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::CerberusSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2CerberusSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512CerberusSolver::from(message),
        )),
        _ => bail!("implementation unavailable for Cerberus"),
    }
}

fn solve_cerberus(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    ensure!(
        descriptor.difficulty().get() <= 16,
        "Cerberus difficulty above 16 dubits is unsupported by upstream clients"
    );
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

fn finish_goaway(
    config: &GoAwayConfig,
    implementation: &str,
    solution: (u64, [u32; 8]),
    attempted_nonces: u64,
    started: Instant,
) -> Result {
    let (nonce, hash) = solution;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let token = format!("{}{nonce:016x}", config.challenge());
    let id = hex::encode(
        hash.into_iter()
            .take(4)
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    Result {
        answer: json!({
            "nonce": nonce.to_string(),
            "token": token,
            "id": id,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    }
}

fn solve_goaway(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    ensure!(
        config.challenge().len() == 64,
        "go-away challenge must be exactly 64 characters"
    );
    ensure!(
        config.difficulty() <= 256,
        "go-away difficulty above the 256-bit SHA-256 digest width is unsatisfiable"
    );
    let accelerated_mask = u8::try_from(config.difficulty())
        .ok()
        .and_then(NonZeroU8::new)
        .filter(|difficulty| difficulty.get() <= 64)
        .map(compute_mask_goaway);
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
        let mut solver = goaway_solver(implementation, message.clone())?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = if let Some(mask) = accelerated_mask {
            solver.solve(0, SOLVE_TYPE_MASK, mask)
        } else {
            solver.solve(config.difficulty(), SOLVE_TYPE_LEADING_ZERO_BITS, u64::MAX)
        };
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("go-away search exhausted configured limit")?;
    Ok(finish_goaway(
        &config,
        implementation.name(),
        solution,
        attempted_nonces,
        started,
    ))
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
            let nonce = match fixup {
                Some(value) => value.fixup(nonce),
                None => js_safe_integer_as_f64(nonce)?,
            };
            return Ok((nonce, attempted));
        }
    }
    bail!("Cap.js SHA-256 search exhausted the lossless JavaScript-number domain")
}

fn solve_capjs(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    if challenge.get("format").and_then(Value::as_u64) == Some(2) {
        return solve_capjs_v2(challenge, implementation);
    }
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let rules = *descriptor.rules();
    let token = descriptor.token;
    let emitter = CapJSEmitter::new(token.as_bytes());
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(rules.count);
    for index in 0..rules.count {
        let mut salt = vec![0_u8; rules.salt_length];
        let mut targets = [0_u32; 2];
        let subchallenge = u32::try_from(index)
            .context("Cap.js subchallenge index exceeds u32")?
            .checked_add(1)
            .context("Cap.js subchallenge index overflow")?;
        emitter.emit(&mut salt, &mut targets, subchallenge);
        let mask = rules
            .mask()
            .context("Cap.js difficulty exceeds the supported 64-bit target width")?;
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, attempted) = solve_cap_decimal_cpu(&salt, target, mask, implementation)?;
        attempted_nonces = attempted_nonces.saturating_add(attempted);
        solutions.push(nonce);
    }
    let elapsed = started.elapsed();
    let solve_ms = elapsed.as_secs_f64() * 1000.0;
    let (elapsed_us, hashrate) = elapsed_work_metadata(elapsed, attempted_nonces);
    let answer = json!({
        "token": token,
        "solutions": solutions,
        "_meta": {
            "elapsed_us": elapsed_us,
            "attempted_nonces": attempted_nonces,
            "hashrate": hashrate,
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
        challenge.difficulty <= 64,
        "haphash difficulty above the 64-nibble SHA-256 digest width is unsatisfiable"
    );
    let prefix = challenge.prefix.as_bytes();
    ensure!(!prefix.is_empty(), "haphash prefix must not be empty");
    let fast_mask = if challenge.difficulty == 0 {
        Some(0)
    } else {
        NonZeroU8::new(challenge.difficulty)
            .filter(|difficulty| difficulty.get() <= 16)
            .map(compute_mask_zero_nibbles)
    };
    let full_digest_bits = u64::from(challenge.difficulty) * 4;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = if let Some(mask) = fast_mask {
            solver.solve(0, SOLVE_TYPE_MASK, mask)
        } else {
            solver.solve(full_digest_bits, SOLVE_TYPE_LEADING_ZERO_BITS, u64::MAX)
        };
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
    let difficulty_factor = NonZeroU64::new(u64::from(request.config.difficulty_factor))
        .context("mCaptcha difficulty factor must be positive")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt)
        .context("mCaptcha string length exceeds the wire format")?;
    let target = compute_target_mcaptcha(difficulty_factor);
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

const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[expect(
    clippy::cast_precision_loss,
    reason = "value is explicitly constrained to JavaScript's exactly representable integer range"
)]
fn js_safe_integer_as_f64(value: u64) -> anyhow::Result<f64> {
    ensure!(
        value <= JS_MAX_SAFE_INTEGER,
        "Cap.js nonce exceeds JavaScript's maximum safe integer"
    );
    Ok(value as f64)
}

fn elapsed_work_metadata(elapsed: Duration, attempted: u64) -> (u64, u64) {
    let elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    let nanos = elapsed.as_nanos();
    let hashrate = u128::from(attempted)
        .saturating_mul(1_000_000_000)
        .checked_div(nanos)
        .map_or(0, |per_second| {
            u64::try_from(per_second).unwrap_or(u64::MAX)
        });
    (elapsed_us, hashrate)
}

fn encode_words_le(words: [u32; 8]) -> String {
    hex::encode(
        words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_error<T>(result: anyhow::Result<T>, message: &str) -> anyhow::Result<anyhow::Error> {
        result.err().ok_or_else(|| anyhow::anyhow!("{message}"))
    }

    #[test]
    fn haphash_solver_returns_nonce_for_protocol_prefix() -> anyhow::Result<()> {
        let result = solve_haphash(
            json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":1}),
            Implementation::Safe,
        )?;
        assert!(result.answer["nonce"].is_string());
        Ok(())
    }

    #[test]
    fn haphash_full_digest_predicate_covers_zero_through_64_nibbles() {
        let zero = [0_u32; 8];
        assert!(challenge_solvers::hash_has_leading_zero_bits(&zero, 0));
        assert!(challenge_solvers::hash_has_leading_zero_bits(&zero, 256));

        let mut digest = [0_u32; 8];
        digest[2] = 0x0fff_ffff;
        assert!(challenge_solvers::hash_has_leading_zero_bits(&digest, 68));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&digest, 69));

        digest.fill(0);
        digest[7] = 1;
        assert!(challenge_solvers::hash_has_leading_zero_bits(&digest, 255));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&digest, 256));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&zero, 257));
    }

    #[test]
    fn haphash_zero_difficulty_is_trivial_for_accelerated_solver() -> anyhow::Result<()> {
        let result = solve_haphash(
            json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":0}),
            Implementation::Auto,
        )?;
        assert!(result.answer["nonce"].is_string());
        assert_eq!(result.answer["attempted_nonces"], 1);
        Ok(())
    }

    #[test]
    fn haphash_rejects_unsatisfiable_difficulty_above_digest_width() -> anyhow::Result<()> {
        let error = require_error(
            solve_haphash(
                json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":65}),
                Implementation::Safe,
            ),
            "difficulty above 64 nibbles must be rejected",
        )?;
        assert!(error.to_string().contains("64-nibble SHA-256 digest width"));
        Ok(())
    }

    #[test]
    fn altcha_reference_backend_is_not_exposed_by_production() -> anyhow::Result<()> {
        for implementation in ["reference", "altcha-rust"] {
            let error = require_error(
                solve("altcha", json!({"parameters": {}}), Some(implementation)),
                "external ALTCHA reference solver must not be selectable in production",
            )?;
            assert!(
                error
                    .to_string()
                    .contains("unsupported classic implementation")
            );
        }
        Ok(())
    }

    #[test]
    fn cerberus_and_goaway_reject_unrepresentable_difficulties() -> anyhow::Result<()> {
        let cerberus = require_error(
            solve_cerberus(
                json!({
                    "challenge": "domain-test",
                    "difficulty": 17,
                    "nonce": 1,
                    "ts": 1,
                    "signature": "domain-test",
                    "version": "0.4.9"
                }),
                Implementation::Safe,
            ),
            "current Cerberus clients only define up to 16 dubits",
        )?;
        assert!(cerberus.to_string().contains("above 16 dubits"));

        let goaway = require_error(
            solve_goaway(
                json!({
                    "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "difficulty": 257
                }),
                Implementation::Safe,
            ),
            "go-away support must reject unsatisfiable difficulty above 256 bits",
        )?;
        assert!(goaway.to_string().contains("256-bit SHA-256 digest width"));
        Ok(())
    }

    #[test]
    fn mcaptcha_rejects_zero_difficulty_factor() -> anyhow::Result<()> {
        let error = require_error(
            solve_mcaptcha(
                json!({
                    "config": {
                        "string": "schema-test",
                        "difficulty_factor": 0,
                        "salt": "schema-test-salt"
                    },
                    "key": "schema-test"
                }),
                Implementation::Auto,
            ),
            "zero mCaptcha difficulty must be rejected before target calculation",
        )?;
        assert!(error.to_string().contains("must be positive"));
        Ok(())
    }

    #[test]
    fn protocol_nonce_json_types_match_wire_contracts() -> anyhow::Result<()> {
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
        )?;
        assert!(mcaptcha.answer["nonce"].is_u64());

        let goaway = solve_goaway(
            json!({
                "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "difficulty": 1
            }),
            Implementation::Auto,
        )?;
        assert!(goaway.answer["nonce"].is_string());
        Ok(())
    }
}
