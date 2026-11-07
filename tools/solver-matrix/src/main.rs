#![cfg_attr(feature = "gpu", recursion_limit = "256")]

use std::{
    num::{NonZeroU8, NonZeroU32, NonZeroU64},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result};
use argon2::Argon2;
use challenge_solvers::{
    build_mcaptcha_prefix, compute_mask_cerberus, compute_mask_goaway, compute_mask_zero_nibbles,
    compute_target_mcaptcha,
    message::{
        AltchaMessage, BinaryMessage, CapJSEmitter, CerberusBinaryMessage, CerberusDecimalMessage,
        CerberusMessage, DecimalMessage, GoAwayMessage,
    },
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LT, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use clap::{Parser, ValueEnum};
use hashwx::HashWx;
use num_bigint::BigUint;
use serde::Serialize;
use serde_json::{Value, json};
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha512};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Correctness,
    Throughput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Protocol {
    Anubis,
    Haphash,
    Mcaptcha,
    Capjs,
    Linear,
    GoAway,
    Cerberus,
    Altcha,
}

#[derive(Parser, Debug)]
struct Args {
    /// Fresh deterministic challenge instances per protocol/backend.
    #[arg(long, default_value_t = 5)]
    samples: u32,
    /// Hard candidate ceiling per solve.
    #[arg(long, default_value_t = 50_000_000)]
    limit: u64,
    /// Exact candidate ceiling for the memory-hard Anubis Argon2id rows.
    #[arg(long, default_value_t = 64)]
    argon2_limit: u64,
    /// Include CPU implementations in the matrix.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cpu: bool,
    /// Include the generalized WGPU backend selected by `WGPU_ADAPTER_NAME/default` selection.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    gpu: bool,
    /// Print usable GPU adapter names as JSON and exit.
    #[arg(long, default_value_t = false)]
    list_gpu_adapters: bool,
    /// Matrix phase. Throughput preserves each comparison type with a ~2^-64 hit condition and must exhaust exactly `limit` candidates.
    #[arg(long, value_enum, default_value_t = Mode::Correctness)]
    mode: Mode,
    /// Git commit reported in JSONL metadata.
    #[arg(long, default_value = "unknown")]
    commit: String,
    /// Select protocol families to execute (repeat to select multiple); default is all.
    #[arg(long, value_enum)]
    protocol: Vec<Protocol>,
}

impl Args {
    fn includes(&self, candidate: Protocol) -> bool {
        self.protocol.is_empty() || self.protocol.contains(&candidate)
    }

    fn has_gpu_work(&self) -> bool {
        self.protocol.is_empty() || self.protocol.iter().any(|p| *p != Protocol::Linear)
    }
}

#[derive(Serialize)]
struct Record {
    schema: u32,
    phase: &'static str,
    timestamp_unix_ms: u128,
    host: String,
    cpu_model: String,
    commit: String,
    protocol: &'static str,
    variant: &'static str,
    implementation: String,
    accelerator: &'static str,
    sample: u32,
    parameters: Value,
    attempted_work: u64,
    solve_ns: u128,
    work_per_second: f64,
    found: bool,
    nonce: Option<u64>,
    target_condition_ok: bool,
    gpu_adapter: Option<String>,
}

trait DynPow {
    fn set_limit_dyn(&mut self, limit: u64);
    fn attempted(&self) -> u64;
    fn solve_dyn_local(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

impl<T> DynPow for T
where
    T: Solver<Output = [u32; 8]> + SolverDyn<Output = [u32; 8]>,
{
    fn set_limit_dyn(&mut self, limit: u64) {
        self.set_limit(limit);
    }
    fn attempted(&self) -> u64 {
        self.attempted_nonces()
    }
    fn solve_dyn_local(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])> {
        self.solve_dyn(target, ty, mask)
    }
}

struct Meta {
    host: String,
    cpu_model: String,
    commit: String,
}

struct Case<'a> {
    protocol: &'static str,
    variant: &'static str,
    implementation: &'a str,
    accelerator: &'static str,
    sample: u32,
    parameters: Value,
    target: u64,
    mask: u64,
    ty: u8,
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn read_trimmed(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name\t: ").map(str::to_owned))
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn condition_ok(ty: u8, target: u64, mask: u64, hash: [u32; 8]) -> bool {
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    match ty {
        challenge_solvers::solver::SOLVE_TYPE_GT => value > target,
        challenge_solvers::solver::SOLVE_TYPE_LT => value < target,
        challenge_solvers::solver::SOLVE_TYPE_MASK => value & mask == target & mask,
        _ => false,
    }
}

const fn phase_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Correctness => "correctness",
        Mode::Throughput => "throughput",
    }
}

const fn benchmark_condition(mode: Mode, target: u64, mask: u64, ty: u8) -> (u64, u64, u8) {
    match mode {
        Mode::Correctness => (target, mask, ty),
        // Keep the protocol's comparison path while making a hit about 2^-64 likely per
        // candidate. Using GT with u64::MAX is provably impossible and can let LTO erase
        // hash work; changing every case to MASK benchmarks a different hot path.
        Mode::Throughput => match ty {
            SOLVE_TYPE_GT => (u64::MAX - 1, u64::MAX, SOLVE_TYPE_GT),
            SOLVE_TYPE_LT => (1, u64::MAX, SOLVE_TYPE_LT),
            SOLVE_TYPE_MASK => (0, u64::MAX, SOLVE_TYPE_MASK),
            _ => (target, mask, ty),
        },
    }
}

fn emit_cpu(
    meta: &Meta,
    case: Case<'_>,
    mut solver: Box<dyn DynPow>,
    limit: u64,
    mode: Mode,
) -> Result<()> {
    let (target, mask, ty) = benchmark_condition(mode, case.target, case.mask, case.ty);
    solver.set_limit_dyn(limit);
    let started = Instant::now();
    let solution = solver.solve_dyn_local(target, ty, mask);
    let solve_ns = started.elapsed().as_nanos();
    let attempted = solver.attempted();
    let (nonce, ok) = solution.map_or((None, true), |(nonce, hash)| {
        (Some(nonce), condition_ok(ty, target, mask, hash))
    });
    if matches!(mode, Mode::Throughput) {
        anyhow::ensure!(
            solution.is_none(),
            "throughput target unexpectedly found a proof"
        );
        anyhow::ensure!(
            attempted == limit,
            "throughput run attempted {attempted} candidates, expected exactly {limit}"
        );
    }
    emit(&Record {
        schema: 1,
        phase: phase_name(mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: case.protocol,
        variant: case.variant,
        implementation: case.implementation.to_owned(),
        accelerator: case.accelerator,
        sample: case.sample,
        parameters: case.parameters,
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: nonce.is_some(),
        nonce,
        target_condition_ok: ok,
        gpu_adapter: None,
    })
}

fn emit(record: &Record) -> Result<()> {
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark throughput is floating-point telemetry, not an exact integer result"
)]
fn rate(work: u64, ns: u128) -> f64 {
    if ns == 0 {
        0.0
    } else {
        work as f64 * 1e9 / ns as f64
    }
}

fn decimal_backends(
    mut message: impl FnMut() -> Result<DecimalMessage>,
) -> Result<Vec<(&'static str, Box<dyn DynPow>)>> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(challenge_solvers::solver::safe::DecimalSolver::from(
                message()?,
            )),
        ),
        (
            "auto",
            Box::new(challenge_solvers::DecimalSolver::from(message()?)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(challenge_solvers::solver::x86::Avx2DecimalSolver::from(
                    message()?,
                )),
            ));
        }
        if std::arch::is_x86_feature_detected!("sha") {
            out.push((
                "sha-ni",
                Box::new(challenge_solvers::solver::sha_ni::DecimalSolver::from(
                    message()?,
                )),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(challenge_solvers::solver::x86::Avx512DecimalSolver::from(
                    message()?,
                )),
            ));
        }
    }
    Ok(out)
}

fn binary_backends(
    mut message: impl FnMut() -> Result<BinaryMessage>,
) -> Result<Vec<(&'static str, Box<dyn DynPow>)>> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(challenge_solvers::solver::safe::BinarySolver::from(
                message()?,
            )),
        ),
        (
            "auto",
            Box::new(challenge_solvers::BinarySolver::from(message()?)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        out.push((
            "avx2",
            Box::new(challenge_solvers::solver::x86::Avx2BinarySolver::from(
                message()?,
            )),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        out.push((
            "avx512",
            Box::new(challenge_solvers::solver::x86::Avx512BinarySolver::from(
                message()?,
            )),
        ));
    }
    Ok(out)
}

fn leading_zero_bits(hash: &[u8; 32], difficulty: u32) -> bool {
    if difficulty > 256 {
        return false;
    }
    let Ok(full_bytes) = usize::try_from(difficulty / 8) else {
        return false;
    };
    let remaining = difficulty % 8;
    if hash[..full_bytes].iter().any(|byte| *byte != 0) {
        return false;
    }
    remaining == 0 || (hash[full_bytes] & (0xff << (8 - remaining))) == 0
}

fn run_argon2_case(meta: &Meta, sample: u32, data: &[u8], args: &Args) -> Result<()> {
    anyhow::ensure!(
        args.argon2_limit > 0,
        "argon2-limit must be greater than zero"
    );
    anyhow::ensure!(
        args.argon2_limit <= (1_u64 << 32),
        "argon2-limit must fit the Anubis u32 nonce space"
    );
    let difficulty = 1_u32;
    let started = Instant::now();
    let mut attempted = 0_u64;
    let mut found = None;
    for nonce in 0..args.argon2_limit {
        let salt = nonce.to_le_bytes();
        let mut hash = [0_u8; 32];
        Argon2::default()
            .hash_password_into(data, &salt, &mut hash)
            .map_err(|error| anyhow::anyhow!("Argon2id matrix hash failed: {error}"))?;
        attempted += 1;
        match args.mode {
            Mode::Correctness if leading_zero_bits(&hash, difficulty) => {
                found = Some(nonce);
                break;
            }
            Mode::Throughput => {
                let prefix = u64::from_be_bytes([
                    hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
                ]);
                anyhow::ensure!(prefix != 0, "throughput target unexpectedly found a proof");
            }
            Mode::Correctness => {}
        }
    }
    if matches!(args.mode, Mode::Correctness) {
        anyhow::ensure!(
            found.is_some(),
            "Argon2id correctness search exhausted candidate limit"
        );
    } else {
        anyhow::ensure!(
            attempted == args.argon2_limit,
            "Argon2id throughput run attempted {attempted} candidates, expected exactly {}",
            args.argon2_limit
        );
    }
    let solve_ns = started.elapsed().as_nanos();
    emit(&Record {
        schema: 1,
        phase: phase_name(args.mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: "anubis",
        variant: "wasm-argon2id-1.28",
        implementation: "argon2-0.6.0".to_owned(),
        accelerator: "cpu",
        sample,
        parameters: json!({"difficulty":difficulty,"data_len":data.len(),"candidate_limit":args.argon2_limit}),
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: found.is_some(),
        nonce: found,
        target_condition_ok: true,
        gpu_adapter: None,
    })
}

fn goaway_backends(message: impl Fn() -> GoAwayMessage) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(challenge_solvers::solver::safe::GoAwaySolver::from(
                message(),
            )),
        ),
        (
            "auto",
            Box::new(challenge_solvers::GoAwaySolver::from(message())),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(challenge_solvers::solver::x86::Avx2GoAwaySolver::from(
                    message(),
                )),
            ));
        }
        if std::arch::is_x86_feature_detected!("sha") {
            out.push((
                "sha-ni",
                Box::new(challenge_solvers::solver::sha_ni::GoAwaySolver::from(
                    message(),
                )),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(challenge_solvers::solver::x86::Avx512GoAwaySolver::from(
                    message(),
                )),
            ));
        }
    }
    out
}

fn cerberus_backends(
    mut message: impl FnMut() -> Result<CerberusMessage>,
) -> Result<Vec<(&'static str, Box<dyn DynPow>)>> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(challenge_solvers::solver::safe::CerberusSolver::from(
                message()?,
            )),
        ),
        (
            "auto",
            Box::new(challenge_solvers::CerberusSolver::from(message()?)),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(challenge_solvers::solver::x86::Avx2CerberusSolver::from(
                    message()?,
                )),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(challenge_solvers::solver::x86::Avx512CerberusSolver::from(
                    message()?,
                )),
            ));
        }
    }
    Ok(out)
}

fn altcha_backends(message: impl Fn() -> AltchaMessage) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(challenge_solvers::solver::safe::AltchaSha256Solver::from(
                message(),
            )),
        ),
        (
            "auto",
            Box::new(challenge_solvers::AltchaSha256Solver::from(message())),
        ),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(challenge_solvers::solver::x86::Avx2AltchaSha256Solver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(challenge_solvers::solver::x86::Avx512AltchaSha256Solver::from(message())),
            ));
        }
    }
    out
}

fn hashwx_seed(challenge: &[u8; 32], block: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(challenge);
    digest.update(block.to_le_bytes());
    digest.finalize().into()
}

const CAP_RSW_T: u32 = 75_000;
const CAP_RSW_MODULUS_HEX: &str = concat!(
    "bdbf80c89a6616ba09da637cb6c5199dbea3ad7fddeb56847a8a89dcf097e464",
    "bf7e1ab4aee4827fdcb00e1aa3096caaeba5e68eac491ddd0b79a0d2baf2efdc",
    "353479cbb74478408a0a2a7fcccd39ddd4f8702aaba08e049411706c16f57d004",
    "4f2c4ad3a6d6db86ce2ccfdcf869e17ab8e6438a4adda41629c98a2e2b4e2b8",
    "38a66114bb17f23dc353cdaa10d1dd2d957ec6d893ddc725276e23c9c02eaab0b",
    "2ab92a30ceccb695a305877eada3b02157cdcce2412b9dc8db76de114da843dad7",
    "16960625b15ddceb69c6a30375223e05c2e08287c274b851e1bca3a940edb0086",
    "2f16146ce27d8814951b87de48f03d9df639c950fafb2ad6807396b5741d",
);

fn cap_rsw_input(sample: u32) -> Result<(BigUint, BigUint)> {
    let modulus = BigUint::parse_bytes(CAP_RSW_MODULUS_HEX.as_bytes(), 16)
        .context("parse fixed Cap RSW modulus")?;
    let mut material = Vec::with_capacity(256);
    for block in 0..4_u32 {
        let mut digest = Sha512::new();
        digest.update(format!("challenge-disasterer-cap-rsw:{sample}:{block}"));
        material.extend_from_slice(&digest.finalize());
    }
    let x =
        BigUint::from_bytes_be(&material) % (&modulus - BigUint::from(3_u8)) + BigUint::from(2_u8);
    Ok((modulus, x))
}

fn run_cap_rsw_case(meta: &Meta, sample: u32, args: &Args) -> Result<()> {
    let (modulus, mut y) = cap_rsw_input(sample)?;
    let started = Instant::now();
    for _ in 0..CAP_RSW_T {
        y = (&y * &y) % &modulus;
    }
    let solve_ns = started.elapsed().as_nanos();
    // Keep the final value observable so an optimizer can never erase the fixed-work loop.
    let final_low64 = y.to_u64_digits().first().copied().unwrap_or(0);
    emit(&Record {
        schema: 1,
        phase: phase_name(args.mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: "capjs",
        variant: "rsw-format2-2048-t75000",
        implementation: "num-bigint-sequential-square".to_owned(),
        accelerator: "cpu",
        sample,
        parameters: json!({
            "modulus_bits": modulus.bits(),
            "squarings": CAP_RSW_T,
            "final_low64": final_low64,
        }),
        attempted_work: u64::from(CAP_RSW_T),
        solve_ns,
        work_per_second: rate(u64::from(CAP_RSW_T), solve_ns),
        found: true,
        nonce: None,
        target_condition_ok: true,
        gpu_adapter: None,
    })
}

const fn decimal_u64_bytes(mut value: u64) -> ([u8; 20], usize) {
    let mut bytes = [0_u8; 20];
    let mut start = bytes.len();
    loop {
        start -= 1;
        bytes[start] = b'0' + (value % 10).to_le_bytes()[0];
        value /= 10;
        if value == 0 {
            break;
        }
    }
    (bytes, start)
}

fn run_linear_case(meta: &Meta, sample: u32, args: &Args) -> Result<()> {
    const TARGET: [u8; 2] = [0xb0, 0x0b];
    const LIVE_SEED: &str = "72BBE3B063367AB2F323D46D2ADA8A8A29BB5B58";
    let mut seed = LIVE_SEED.as_bytes().to_vec();
    // Preserve a legal first nibble (and therefore digest offset) while varying
    // the remaining seed bytes deterministically between samples.
    for (index, byte) in seed.iter_mut().enumerate().skip(1) {
        let nibble =
            (usize::from(sample.to_le_bytes()[0]).wrapping_add(index) & 0xf).to_le_bytes()[0];
        *byte = if nibble < 10 {
            b'0' + nibble
        } else {
            b'A' + nibble - 10
        };
    }
    let offset = usize::from(seed[0] - b'0');
    let started = Instant::now();
    let mut attempted = 0_u64;
    let mut found = None;
    let mut sink = 0_u8;
    for nonce in 0..args.limit {
        let mut hasher = Sha1::new();
        hasher.update(&seed);
        let (decimal, start) = decimal_u64_bytes(nonce);
        hasher.update(&decimal[start..]);
        let digest = hasher.finalize();
        sink ^= digest[0];
        attempted += 1;
        if matches!(args.mode, Mode::Correctness)
            && digest[offset] == TARGET[0]
            && digest[offset + 1] == TARGET[1]
        {
            found = Some(nonce);
            break;
        }
    }
    let solve_ns = started.elapsed().as_nanos();
    if matches!(args.mode, Mode::Correctness) {
        anyhow::ensure!(
            found.is_some(),
            "Linear.pub correctness search exhausted candidate limit"
        );
    } else {
        anyhow::ensure!(
            attempted == args.limit,
            "Linear.pub throughput did not exhaust exact work budget"
        );
    }
    emit(&Record {
        schema: 1,
        phase: phase_name(args.mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: "linear",
        variant: "sha1-fixed-work",
        implementation: "rustcrypto-auto".to_owned(),
        accelerator: "cpu",
        sample,
        parameters: json!({
            "digest_offset": offset,
            "target": "b00b",
            "sink": sink,
        }),
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: found.is_some(),
        nonce: found,
        target_condition_ok: matches!(args.mode, Mode::Throughput) || found.is_some(),
        gpu_adapter: None,
    })
}

fn run_hashwx_case(meta: &Meta, sample: u32, args: &Args) -> Result<()> {
    const NONCES_PER_HASH: u64 = 65_536;
    const DIFFICULTY: u64 = 1_000_000;

    let mut challenge = [0_u8; 32];
    challenge[..4].copy_from_slice(&sample.to_le_bytes());
    for (index, byte) in challenge[4..].iter_mut().enumerate() {
        *byte = index.to_le_bytes()[0].wrapping_mul(17).wrapping_add(0x5a);
    }
    let correctness_target = u64::MAX / DIFFICULTY;
    let benchmark_target = match args.mode {
        Mode::Correctness => correctness_target,
        Mode::Throughput => 0,
    };
    let started = Instant::now();
    let mut attempted = 0_u64;
    let mut found = None;
    let mut current_block = 0_u64;
    let mut hashwx = HashWx::compile(&hashwx_seed(&challenge, current_block))
        .context("compile HashWX matrix program")?;
    while attempted < args.limit {
        let nonce = attempted;
        let block = nonce / NONCES_PER_HASH;
        if block != current_block {
            hashwx = HashWx::compile(&hashwx_seed(&challenge, block))
                .context("compile HashWX matrix program")?;
            current_block = block;
        }
        let hash = hashwx.hash(nonce);
        attempted += 1;
        if hash <= benchmark_target {
            found = Some(nonce);
            break;
        }
    }
    let solve_ns = started.elapsed().as_nanos();
    match args.mode {
        Mode::Correctness => anyhow::ensure!(
            found.is_some(),
            "HashWX correctness search exhausted candidate limit"
        ),
        Mode::Throughput => {
            anyhow::ensure!(
                found.is_none(),
                "HashWX throughput target unexpectedly found a proof"
            );
            anyhow::ensure!(
                attempted == args.limit,
                "HashWX throughput run attempted {attempted} candidates, expected exactly {}",
                args.limit
            );
        }
    }
    emit(&Record {
        schema: 1,
        phase: phase_name(args.mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol: "capjs",
        variant: "hashwx-format2",
        implementation: "native-jit".to_owned(),
        accelerator: "cpu",
        sample,
        parameters: json!({
            "difficulty": DIFFICULTY,
            "nonces_per_hash": NONCES_PER_HASH,
            "challenge_count": 1,
        }),
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: found.is_some(),
        nonce: found,
        target_condition_ok: matches!(args.mode, Mode::Throughput) || found.is_some(),
        gpu_adapter: None,
    })
}

fn run_cpu_anubis_haphash_cases(meta: &Meta, args: &Args, sample: u32) -> Result<()> {
    let sample_bytes = sample.to_le_bytes();

    if args.includes(Protocol::Anubis) {
        // Anubis: same decimal message semantics as the live service, d5 ~= 2^20 expected candidates.
        let mut anubis_prefix = vec![b'a'; 96];
        anubis_prefix[..4].copy_from_slice(&sample_bytes);
        let anubis_difficulty = NonZeroU8::new(5).context("Anubis difficulty must be non-zero")?;
        let anubis_mask = compute_mask_zero_nibbles(anubis_difficulty);
        for (implementation, solver) in decimal_backends(|| {
            DecimalMessage::new(&anubis_prefix, 0).context("construct Anubis decimal fixture")
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "anubis",
                    variant: "decimal",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":5,"prefix_len":anubis_prefix.len()}),
                    target: 0,
                    mask: anubis_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }
    }

    if args.includes(Protocol::Haphash) {
        // haphash: same decimal SHA-256 work primitive, but distinct protocol prefixes.
        // Use d5 (~2^20 expected candidates) for stable apples-to-apples throughput.
        for (variant, prefix) in [
            (
                "haproxy",
                format!("203.0.113.{};example.test;1759954402;", sample % 255),
            ),
            ("debian-varnish", format!("102026194332.0x{sample:064x};")),
        ] {
            let difficulty = NonZeroU8::new(5).context("haphash difficulty must be non-zero")?;
            let mask = compute_mask_zero_nibbles(difficulty);
            for (implementation, solver) in decimal_backends(|| {
                DecimalMessage::new(prefix.as_bytes(), 0)
                    .context("construct haphash decimal fixture")
            })? {
                emit_cpu(
                    meta,
                    Case {
                        protocol: "haphash",
                        variant,
                        implementation,
                        accelerator: "cpu",
                        sample,
                        parameters: json!({"difficulty":difficulty.get(),"prefix_len":prefix.len()}),
                        target: 0,
                        mask,
                        ty: SOLVE_TYPE_MASK,
                    },
                    solver,
                    args.limit,
                    args.mode,
                )?;
            }
        }
    }

    if args.includes(Protocol::Anubis) {
        // Anubis 1.28 WASM SHA-256: raw challenge bytes plus a 32-bit nonce.
        // End the challenge with 0x80 so upstream selects little-endian nonce bytes directly.
        let mut anubis_wasm_data = vec![0x42_u8; 48];
        anubis_wasm_data[..4].copy_from_slice(&sample_bytes);
        anubis_wasm_data[47] = 0x80;
        let anubis_sha_mask = u64::MAX << (64 - 20);
        let nonce_bytes = NonZeroU8::new(4).context("Anubis nonce width must be non-zero")?;
        for (implementation, solver) in binary_backends(|| {
            BinaryMessage::new(&anubis_wasm_data, nonce_bytes)
                .context("construct Anubis binary fixture")
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "anubis",
                    variant: "wasm-sha256-1.28",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":20,"data_len":anubis_wasm_data.len(),"nonce_endian":"little"}),
                    target: 0,
                    mask: anubis_sha_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }

        run_argon2_case(meta, sample, &anubis_wasm_data, args)?;
    }

    Ok(())
}

fn run_cpu_mcaptcha_case(meta: &Meta, args: &Args, sample: u32) -> Result<()> {
    // mCaptcha: canonical salt + bincode-style string prefix, expected candidates ~= difficulty factor.
    let mut mcaptcha_prefix = Vec::new();
    build_mcaptcha_prefix(
        &mut mcaptcha_prefix,
        "challenge-disasterer-matrix",
        &format!("matrix-salt-{sample:08x}"),
    )
    .context("mCaptcha matrix prefix exceeds the wire format")?;
    let mcaptcha_factor = 1_000_000_u64;
    let mcaptcha_difficulty =
        NonZeroU64::new(mcaptcha_factor).context("mCaptcha difficulty factor must be non-zero")?;
    let mcaptcha_target = compute_target_mcaptcha(mcaptcha_difficulty);
    for (implementation, solver) in decimal_backends(|| {
        DecimalMessage::new(&mcaptcha_prefix, 0).context("construct mCaptcha decimal fixture")
    })? {
        emit_cpu(
            meta,
            Case {
                protocol: "mcaptcha",
                variant: "sha256",
                implementation,
                accelerator: "cpu",
                sample,
                parameters: json!({"difficulty_factor":mcaptcha_factor,"prefix_len":mcaptcha_prefix.len()}),
                target: mcaptcha_target,
                mask: u64::MAX,
                ty: SOLVE_TYPE_GT,
            },
            solver,
            args.limit,
            args.mode,
        )?;
    }
    Ok(())
}

fn run_cpu_capjs_cases(meta: &Meta, args: &Args, sample: u32) -> Result<()> {
    if args.includes(Protocol::Capjs) {
        // Cap.js: real emitter shape, one subchallenge at d5 ~= 2^20 expected candidates.
        let token = format!("challenge-disasterer-matrix-token-{sample:08x}");
        let emitter = CapJSEmitter::new(token.as_bytes());
        let mut cap_salt = vec![0_u8; 32];
        let mut cap_target = [0_u32; 2];
        emitter.emit(&mut cap_salt, &mut cap_target, 1);
        let cap_difficulty = 5_u8;
        let cap_mask = u64::MAX << (64 - u64::from(cap_difficulty) * 4);
        let cap_target = (u64::from(cap_target[0]) << 32) | u64::from(cap_target[1]);
        for (implementation, solver) in decimal_backends(|| {
            DecimalMessage::new_f64(&cap_salt, 0)
                .map(|(message, _)| message)
                .context("construct Cap.js decimal fixture")
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "capjs",
                    variant: "single-subchallenge",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":cap_difficulty,"salt_len":cap_salt.len()}),
                    target: cap_target,
                    mask: cap_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }

        // Cap.js format-2 SHA-256 uses a direct salt + hexadecimal prefix target.
        // Keep a first-class row even though it shares the decimal solver engine with
        // format 1: this prevents protocol coverage from disappearing behind a primitive alias.
        let cap_v2_salt = format!("challenge-disasterer-cap-v2-{sample:08x}");
        let cap_v2_target_hex = "abcde";
        let cap_v2_target = 0xabcde_u64 << (64 - 20);
        let cap_v2_mask = u64::MAX << (64 - 20);
        for (implementation, solver) in decimal_backends(|| {
            DecimalMessage::new_f64(cap_v2_salt.as_bytes(), 0)
                .map(|(message, _)| message)
                .context("construct Cap.js format-2 fixture")
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "capjs",
                    variant: "format2-sha256-pow",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({
                        "target": cap_v2_target_hex,
                        "salt_len": cap_v2_salt.len(),
                    }),
                    target: cap_v2_target,
                    mask: cap_v2_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }

        // Cap.js format-2 HashWX at the production default reuse window. Include
        // per-seed program generation/JIT cost rather than benchmarking only exec().
        run_hashwx_case(meta, sample, args)?;

        // Cap.js RSW default production shape: a fixed 2048-bit modulus and 75,000
        // sequential modular squarings. The external matrix runs the same input through
        // the pinned official Cap worker and GMP.
        run_cap_rsw_case(meta, sample, args)?;
    }

    // Linear.pub: current deployed SHA-1 gate, exact seed + decimal nonce shape.
    if args.includes(Protocol::Linear) {
        run_linear_case(meta, sample, args)?;
    }
    Ok(())
}

fn run_cpu_goaway_cerberus_cases(meta: &Meta, args: &Args, sample: u32) -> Result<()> {
    if args.includes(Protocol::GoAway) {
        // go-away: actual 32-byte challenge message, d20 ~= 2^20 expected candidates.
        let mut go_bytes = [0_u8; 32];
        let sample_low = sample.to_le_bytes()[0];
        for (i, byte) in go_bytes.iter_mut().enumerate() {
            *byte = i.to_le_bytes()[0].wrapping_add(sample_low);
        }
        let go_difficulty = NonZeroU8::new(20).context("GoAway difficulty must be non-zero")?;
        let go_mask = compute_mask_goaway(go_difficulty);
        for (implementation, solver) in goaway_backends(|| GoAwayMessage::new_bytes(&go_bytes, 0)) {
            emit_cpu(
                meta,
                Case {
                    protocol: "go-away",
                    variant: "js-pow-sha256",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":go_difficulty.get()}),
                    target: 0,
                    mask: go_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }
    }

    // Legacy Cerberus (<0.4.6) decimal nonce format, retained because the service supports it.
    if args.includes(Protocol::Cerberus) {
        let cerberus_legacy_salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        let cerberus_legacy_difficulty =
            NonZeroU8::new(10).context("legacy Cerberus difficulty must be non-zero")?;
        let cerberus_legacy_mask = compute_mask_cerberus(cerberus_legacy_difficulty);
        for (implementation, solver) in cerberus_backends(|| {
            CerberusDecimalMessage::new(cerberus_legacy_salt.as_bytes(), 0)
                .map(CerberusMessage::Decimal)
                .context("construct legacy Cerberus fixture")
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "cerberus",
                    variant: "decimal-pre-0.4.6",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":cerberus_legacy_difficulty.get(),"format":"decimal"}),
                    target: 0,
                    mask: cerberus_legacy_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }

        // Current Cerberus (0.4.6+) binary format, d10 ~= 2^20 expected candidates.
        let cerberus_salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        let cerberus_difficulty =
            NonZeroU8::new(10).context("Cerberus difficulty must be non-zero")?;
        let cerberus_mask = compute_mask_cerberus(cerberus_difficulty);
        for (implementation, solver) in cerberus_backends(|| {
            Ok(CerberusMessage::Binary(CerberusBinaryMessage::new(
                cerberus_salt.as_bytes(),
                0,
            )))
        })? {
            emit_cpu(
                meta,
                Case {
                    protocol: "cerberus",
                    variant: "binary-0.4.6+",
                    implementation,
                    accelerator: "cpu",
                    sample,
                    parameters: json!({"difficulty":cerberus_difficulty.get()}),
                    target: 0,
                    mask: cerberus_mask,
                    ty: SOLVE_TYPE_MASK,
                },
                solver,
                args.limit,
                args.mode,
            )?;
        }
    }

    Ok(())
}

fn run_cpu_altcha_cases(meta: &Meta, args: &Args, sample: u32) -> Result<()> {
    // ALTCHA SHA-256, 20-bit prefix ~= 2^20 candidates.
    let altcha_cost = NonZeroU32::new(2).context("ALTCHA cost must be non-zero")?;
    let altcha_key_length = NonZeroU32::new(32).context("ALTCHA key length must be non-zero")?;
    let altcha_message = || AltchaMessage {
        nonce: [sample.to_le_bytes()[0]; 16],
        salt: [sample.wrapping_add(17).to_le_bytes()[0]; 16],
        cost: altcha_cost,
        pbkdf2: false,
        key_length: altcha_key_length,
    };
    let altcha_mask = 0xffff_f000_0000_0000_u64;
    for (implementation, solver) in altcha_backends(altcha_message) {
        emit_cpu(
            meta,
            Case {
                protocol: "altcha",
                variant: "sha256",
                implementation,
                accelerator: "cpu",
                sample,
                parameters: json!({"cost":2,"prefix_bits":20,"key_length":32}),
                target: 0,
                mask: altcha_mask,
                ty: SOLVE_TYPE_MASK,
            },
            solver,
            args.limit,
            args.mode,
        )?;
    }

    // ALTCHA PBKDF2: 16-bit prefix * cost 16 ~= 2^20 SHA-scale work.
    let altcha_pbkdf2_cost = NonZeroU32::new(16).context("ALTCHA PBKDF2 cost must be non-zero")?;
    let altcha_pbkdf2 = || AltchaMessage {
        nonce: [sample.wrapping_add(1).to_le_bytes()[0]; 16],
        salt: [sample.wrapping_add(33).to_le_bytes()[0]; 16],
        cost: altcha_pbkdf2_cost,
        pbkdf2: true,
        key_length: altcha_key_length,
    };
    let altcha_pbkdf2_mask = 0xffff_0000_0000_0000_u64;
    for (implementation, solver) in altcha_backends(altcha_pbkdf2) {
        emit_cpu(
            meta,
            Case {
                protocol: "altcha",
                variant: "pbkdf2-sha256",
                implementation,
                accelerator: "cpu",
                sample,
                parameters: json!({"cost":16,"prefix_bits":16,"key_length":32}),
                target: 0,
                mask: altcha_pbkdf2_mask,
                ty: SOLVE_TYPE_MASK,
            },
            solver,
            args.limit,
            args.mode,
        )?;
    }
    Ok(())
}

fn run_cpu_matrix(meta: &Meta, args: &Args) -> Result<()> {
    for sample in 0..args.samples {
        if args.includes(Protocol::Anubis) || args.includes(Protocol::Haphash) {
            run_cpu_anubis_haphash_cases(meta, args, sample)?;
        }
        if args.includes(Protocol::Mcaptcha) {
            run_cpu_mcaptcha_case(meta, args, sample)?;
        }
        if args.includes(Protocol::Capjs) || args.includes(Protocol::Linear) {
            run_cpu_capjs_cases(meta, args, sample)?;
        }
        if args.includes(Protocol::GoAway) || args.includes(Protocol::Cerberus) {
            run_cpu_goaway_cerberus_cases(meta, args, sample)?;
        }
        if args.includes(Protocol::Altcha) {
            run_cpu_altcha_cases(meta, args, sample)?;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_matrix<'a>(
    meta: &'a Meta,
    args: &'a Args,
) -> core::pin::Pin<Box<dyn core::future::Future<Output = Result<()>> + 'a>> {
    Box::pin(async move {
        let gpu = challenge_solvers::gpu::GpuContext::create()
            .await
            .context("create generalized WGPU context")?;
        gpu.warm_up()
            .await
            .context("warm generalized WGPU context")?;
        let adapter = gpu.adapter().to_owned();

        for sample in 0..args.samples {
            if args.includes(Protocol::Anubis) || args.includes(Protocol::Haphash) {
                run_gpu_anubis_haphash_cases(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
            if args.includes(Protocol::Anubis) || args.includes(Protocol::Mcaptcha) {
                run_gpu_wasm_mcaptcha_cases(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
            if args.includes(Protocol::Capjs) {
                run_gpu_capjs_cases(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
            if args.includes(Protocol::GoAway) {
                run_gpu_goaway_case(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
            if args.includes(Protocol::Cerberus) {
                run_gpu_cerberus_cases(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
            if args.includes(Protocol::Altcha) {
                run_gpu_altcha_cases(meta, args, &gpu, adapter.as_str(), sample).await?;
            }
        }
        Ok(())
    })
}

#[cfg(feature = "gpu")]
async fn run_gpu_anubis_haphash_cases(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    if args.includes(Protocol::Anubis) {
        let sample_bytes = sample.to_le_bytes();
        let mut prefix = vec![b'a'; 96];
        prefix[..4].copy_from_slice(&sample_bytes);
        let difficulty = NonZeroU8::new(5).context("Anubis difficulty must be non-zero")?;
        let mask = compute_mask_zero_nibbles(difficulty);
        let message =
            DecimalMessage::new(&prefix, 0).context("construct GPU Anubis decimal fixture")?;
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
        let solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
            .await?;
        let elapsed = started.elapsed().as_nanos();
        emit_gpu(
            meta,
            "anubis",
            "decimal",
            sample,
            json!({"difficulty":5,"prefix_len":prefix.len()}),
            0,
            mask,
            SOLVE_TYPE_MASK,
            solver.attempted_nonces(),
            elapsed,
            solution.map(|x| (x.nonce, x.hash)),
            adapter,
            args.mode,
            args.limit,
        )?;
    }

    if args.includes(Protocol::Haphash) {
        for (variant, prefix) in [
            (
                "haproxy",
                format!("203.0.113.{};example.test;1759954402;", sample % 255),
            ),
            ("debian-varnish", format!("102026194332.0x{sample:064x};")),
        ] {
            let difficulty = NonZeroU8::new(5).context("haphash difficulty must be non-zero")?;
            let mask = compute_mask_zero_nibbles(difficulty);
            let message = DecimalMessage::new(prefix.as_bytes(), 0)
                .context("construct GPU haphash decimal fixture")?;
            let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
            solver.set_limit(args.limit);
            let started = Instant::now();
            let (bench_target, bench_mask, _) =
                benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
            let solution = solver
                .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
                .await?;
            let elapsed = started.elapsed().as_nanos();
            emit_gpu(
                meta,
                "haphash",
                variant,
                sample,
                json!({"difficulty":difficulty.get(),"prefix_len":prefix.len()}),
                0,
                mask,
                SOLVE_TYPE_MASK,
                solver.attempted_nonces(),
                elapsed,
                solution.map(|x| (x.nonce, x.hash)),
                adapter,
                args.mode,
                args.limit,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_wasm_mcaptcha_cases(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    if args.includes(Protocol::Anubis) {
        let sample_bytes = sample.to_le_bytes();
        let mut anubis_wasm_data = vec![0x42_u8; 48];
        anubis_wasm_data[..4].copy_from_slice(&sample_bytes);
        anubis_wasm_data[47] = 0x80;
        let anubis_sha_mask = u64::MAX << (64 - 20);
        let nonce_bytes = NonZeroU8::new(4).context("Anubis nonce width must be non-zero")?;
        let binary_message = BinaryMessage::new(&anubis_wasm_data, nonce_bytes)
            .context("construct GPU Anubis binary fixture")?;
        let mut solver = challenge_solvers::gpu::BinarySolver::from(binary_message);
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, anubis_sha_mask, SOLVE_TYPE_MASK);
        let solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
            .await?;
        let elapsed = started.elapsed().as_nanos();
        emit_gpu(
            meta,
            "anubis",
            "wasm-sha256-1.28",
            sample,
            json!({"difficulty":20,"data_len":anubis_wasm_data.len(),"nonce_endian":"little"}),
            0,
            anubis_sha_mask,
            SOLVE_TYPE_MASK,
            solver.attempted_nonces(),
            elapsed,
            solution.map(|x| (x.nonce, x.hash)),
            adapter,
            args.mode,
            args.limit,
        )?;
    }

    if args.includes(Protocol::Mcaptcha) {
        let mut mcaptcha_prefix = Vec::new();
        build_mcaptcha_prefix(
            &mut mcaptcha_prefix,
            "challenge-disasterer-matrix",
            &format!("matrix-salt-{sample:08x}"),
        )
        .context("mCaptcha matrix prefix exceeds the wire format")?;
        let factor = 1_000_000_u64;
        let difficulty =
            NonZeroU64::new(factor).context("mCaptcha difficulty factor must be non-zero")?;
        let target = compute_target_mcaptcha(difficulty);
        let message = DecimalMessage::new(&mcaptcha_prefix, 0)
            .context("construct GPU mCaptcha decimal fixture")?;
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, bench_ty) =
            benchmark_condition(args.mode, target, u64::MAX, SOLVE_TYPE_GT);
        let solution = if bench_ty == SOLVE_TYPE_MASK {
            solver
                .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
                .await?
        } else {
            solver
                .solve::<SOLVE_TYPE_GT>(gpu, bench_target, bench_mask)
                .await?
        };
        let elapsed = started.elapsed().as_nanos();
        emit_gpu(
            meta,
            "mcaptcha",
            "sha256",
            sample,
            json!({"difficulty_factor":factor}),
            target,
            u64::MAX,
            SOLVE_TYPE_GT,
            solver.attempted_nonces(),
            elapsed,
            solution.map(|x| (x.nonce, x.hash)),
            adapter,
            args.mode,
            args.limit,
        )?;
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_capjs_cases(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    let token = format!("challenge-disasterer-matrix-token-{sample:08x}");
    let emitter = CapJSEmitter::new(token.as_bytes());
    let mut cap_salt = vec![0_u8; 32];
    let mut cap_target = [0_u32; 2];
    emitter.emit(&mut cap_salt, &mut cap_target, 1);
    let cap_difficulty = 5_u8;
    let cap_mask = u64::MAX << (64 - u64::from(cap_difficulty) * 4);
    let cap_target = (u64::from(cap_target[0]) << 32) | u64::from(cap_target[1]);
    let cap_message = DecimalMessage::new_f64(&cap_salt, 0)
        .map(|(message, _)| message)
        .context("construct GPU Cap.js decimal fixture")?;
    let mut solver = challenge_solvers::gpu::DecimalSolver::from(cap_message);
    solver.set_limit(args.limit);
    let started = Instant::now();
    let (bench_target, bench_mask, _) =
        benchmark_condition(args.mode, cap_target, cap_mask, SOLVE_TYPE_MASK);
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
        .await?;
    let elapsed = started.elapsed().as_nanos();
    emit_gpu(
        meta,
        "capjs",
        "single-subchallenge",
        sample,
        json!({"difficulty":cap_difficulty,"salt_len":cap_salt.len()}),
        cap_target,
        cap_mask,
        SOLVE_TYPE_MASK,
        solver.attempted_nonces(),
        elapsed,
        solution.map(|x| (x.nonce, x.hash)),
        adapter,
        args.mode,
        args.limit,
    )?;

    let cap_v2_salt = format!("challenge-disasterer-cap-v2-{sample:08x}");
    let cap_v2_target_hex = "abcde";
    let cap_v2_target = 0xabcde_u64 << (64 - 20);
    let cap_v2_mask = u64::MAX << (64 - 20);
    let cap_v2_message = DecimalMessage::new_f64(cap_v2_salt.as_bytes(), 0)
        .map(|(message, _)| message)
        .context("construct GPU Cap.js format-2 fixture")?;
    let mut solver = challenge_solvers::gpu::DecimalSolver::from(cap_v2_message);
    solver.set_limit(args.limit);
    let started = Instant::now();
    let (bench_target, bench_mask, _) =
        benchmark_condition(args.mode, cap_v2_target, cap_v2_mask, SOLVE_TYPE_MASK);
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
        .await?;
    let elapsed = started.elapsed().as_nanos();
    emit_gpu(
        meta,
        "capjs",
        "format2-sha256-pow",
        sample,
        json!({"target":cap_v2_target_hex,"salt_len":cap_v2_salt.len()}),
        cap_v2_target,
        cap_v2_mask,
        SOLVE_TYPE_MASK,
        solver.attempted_nonces(),
        elapsed,
        solution.map(|x| (x.nonce, x.hash)),
        adapter,
        args.mode,
        args.limit,
    )?;
    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_goaway_case(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    let mut go_bytes = [0_u8; 32];
    let sample_low = sample.to_le_bytes()[0];
    for (i, byte) in go_bytes.iter_mut().enumerate() {
        *byte = i.to_le_bytes()[0].wrapping_add(sample_low);
    }
    let d = NonZeroU8::new(20).context("GoAway difficulty must be non-zero")?;
    let mask = compute_mask_goaway(d);
    let mut solver =
        challenge_solvers::gpu::GoAwaySolver::from(GoAwayMessage::new_bytes(&go_bytes, 0));
    solver.set_limit(args.limit);
    let started = Instant::now();
    let (bench_target, bench_mask, _) = benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
        .await?;
    let elapsed = started.elapsed().as_nanos();
    emit_gpu(
        meta,
        "go-away",
        "js-pow-sha256",
        sample,
        json!({"difficulty":20}),
        0,
        mask,
        SOLVE_TYPE_MASK,
        solver.attempted_nonces(),
        elapsed,
        solution.map(|x| (x.nonce, x.hash)),
        adapter,
        args.mode,
        args.limit,
    )?;
    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_cerberus_cases(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    // Legacy Cerberus (<0.4.6) decimal nonce format.
    let legacy_salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
    let legacy_d = NonZeroU8::new(10).context("legacy Cerberus difficulty must be non-zero")?;
    let legacy_mask = compute_mask_cerberus(legacy_d);
    let legacy_message = CerberusDecimalMessage::new(legacy_salt.as_bytes(), 0)
        .context("construct GPU legacy Cerberus fixture")?;
    let mut solver =
        challenge_solvers::gpu::CerberusSolver::from(CerberusMessage::Decimal(legacy_message));
    solver.set_limit(args.limit);
    let started = Instant::now();
    let (bench_target, bench_mask, _) =
        benchmark_condition(args.mode, 0, legacy_mask, SOLVE_TYPE_MASK);
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
        .await?;
    let elapsed = started.elapsed().as_nanos();
    emit_gpu(
        meta,
        "cerberus",
        "decimal-pre-0.4.6",
        sample,
        json!({"difficulty":10,"format":"decimal"}),
        0,
        legacy_mask,
        SOLVE_TYPE_MASK,
        solver.attempted_nonces(),
        elapsed,
        solution.map(|x| (x.nonce, x.hash)),
        adapter,
        args.mode,
        args.limit,
    )?;

    let salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
    let d = NonZeroU8::new(10).context("Cerberus difficulty must be non-zero")?;
    let mask = compute_mask_cerberus(d);
    let mut solver = challenge_solvers::gpu::CerberusSolver::from(CerberusMessage::Binary(
        CerberusBinaryMessage::new(salt.as_bytes(), 0),
    ));
    solver.set_limit(args.limit);
    let started = Instant::now();
    let (bench_target, bench_mask, _) = benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
        .await?;
    let elapsed = started.elapsed().as_nanos();
    emit_gpu(
        meta,
        "cerberus",
        "binary-0.4.6+",
        sample,
        json!({"difficulty":10}),
        0,
        mask,
        SOLVE_TYPE_MASK,
        solver.attempted_nonces(),
        elapsed,
        solution.map(|x| (x.nonce, x.hash)),
        adapter,
        args.mode,
        args.limit,
    )?;
    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_altcha_cases(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    adapter: &str,
    sample: u32,
) -> Result<()> {
    for (variant, pbkdf2, cost, mask) in [
        ("sha256", false, 2_u32, 0xffff_f000_0000_0000_u64),
        ("pbkdf2-sha256", true, 16_u32, 0xffff_0000_0000_0000_u64),
    ] {
        let cost = NonZeroU32::new(cost).context("ALTCHA cost must be non-zero")?;
        let key_length = NonZeroU32::new(32).context("ALTCHA key length must be non-zero")?;
        let message = AltchaMessage {
            nonce: [sample.to_le_bytes()[0]; 16],
            salt: [sample.wrapping_add(17).to_le_bytes()[0]; 16],
            cost,
            pbkdf2,
            key_length,
        };
        let mut solver = challenge_solvers::gpu::AltchaSha256Solver::from(message);
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
        let solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, bench_target, bench_mask)
            .await?;
        let elapsed = started.elapsed().as_nanos();
        emit_gpu(
            meta,
            "altcha",
            variant,
            sample,
            json!({"cost":cost,"key_length":32}),
            0,
            mask,
            SOLVE_TYPE_MASK,
            solver.attempted_nonces(),
            elapsed,
            solution.map(|x| (x.nonce, x.hash)),
            adapter,
            args.mode,
            args.limit,
        )?;
    }
    Ok(())
}
#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "matrix row emission keeps protocol dimensions explicit at the reporting boundary"
)]
fn emit_gpu(
    meta: &Meta,
    protocol: &'static str,
    variant: &'static str,
    sample: u32,
    parameters: Value,
    target: u64,
    mask: u64,
    ty: u8,
    attempted: u64,
    solve_ns: u128,
    solution: Option<(u64, [u32; 8])>,
    adapter: &str,
    mode: Mode,
    limit: u64,
) -> Result<()> {
    let (actual_target, actual_mask, actual_ty) = benchmark_condition(mode, target, mask, ty);
    let (nonce, ok) = solution.map_or((None, true), |(nonce, hash)| {
        (
            Some(nonce),
            condition_ok(actual_ty, actual_target, actual_mask, hash),
        )
    });
    if matches!(mode, Mode::Throughput) {
        anyhow::ensure!(
            nonce.is_none(),
            "throughput target unexpectedly found a proof"
        );
        anyhow::ensure!(
            attempted == limit,
            "throughput run attempted {attempted} candidates, expected exactly {limit}"
        );
    }
    emit(&Record {
        schema: 1,
        phase: phase_name(mode),
        timestamp_unix_ms: timestamp_ms(),
        host: meta.host.clone(),
        cpu_model: meta.cpu_model.clone(),
        commit: meta.commit.clone(),
        protocol,
        variant,
        implementation: "wgpu".to_owned(),
        accelerator: "gpu",
        sample,
        parameters,
        attempted_work: attempted,
        solve_ns,
        work_per_second: rate(attempted, solve_ns),
        found: nonce.is_some(),
        nonce,
        target_condition_ok: ok,
        gpu_adapter: Some(adapter.to_owned()),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_gpu_adapters {
        #[cfg(feature = "gpu")]
        {
            println!(
                "{}",
                serde_json::to_string(&challenge_solvers::gpu::GpuContext::adapter_names().await)?
            );
            return Ok(());
        }
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("GPU support is not compiled into this solver-matrix binary");
    }
    let meta = Meta {
        host: read_trimmed("/proc/sys/kernel/hostname"),
        cpu_model: cpu_model(),
        commit: args.commit.clone(),
    };
    if args.cpu {
        run_cpu_matrix(&meta, &args)?;
    }
    if args.gpu && args.has_gpu_work() {
        #[cfg(feature = "gpu")]
        run_gpu_matrix(&meta, &args).await?;
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu true requires the solver-matrix gpu feature");
    } else if args.gpu && !args.cpu {
        anyhow::bail!(
            "selected protocols have no GPU benchmarks; enable CPU or select another protocol"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_selection_defaults_to_all_and_supports_multiple_families() -> Result<()> {
        let all = Args::try_parse_from(["matrix"])?;
        for protocol in [
            Protocol::Anubis,
            Protocol::Haphash,
            Protocol::Mcaptcha,
            Protocol::Capjs,
            Protocol::Linear,
            Protocol::GoAway,
            Protocol::Cerberus,
            Protocol::Altcha,
        ] {
            assert!(all.includes(protocol));
        }
        let selected =
            Args::try_parse_from(["matrix", "--protocol", "go-away", "--protocol", "capjs"])?;
        assert!(selected.includes(Protocol::GoAway));
        assert!(selected.includes(Protocol::Capjs));
        assert!(!selected.includes(Protocol::Anubis));
        assert!(!selected.includes(Protocol::Cerberus));
        let linear = Args::try_parse_from(["matrix", "--protocol", "linear"])?;
        assert!(linear.includes(Protocol::Linear));
        assert!(!linear.has_gpu_work());
        Ok(())
    }

    #[test]
    fn cap_rsw_fixed_vector_matches_external_reference() -> Result<()> {
        let (modulus, mut y) = cap_rsw_input(0)?;
        for _ in 0..CAP_RSW_T {
            y = (&y * &y) % &modulus;
        }
        assert_eq!(
            y.to_u64_digits().first().copied().unwrap_or(0),
            0x92a2_5a0a_9175_dee7
        );
        Ok(())
    }

    /// Requires an actual native GPU. Exact u32 rollover and the maximum
    /// JavaScript-safe integer are both covered without traversing the
    /// intervening billions of candidate hashes.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    #[ignore = "requires a Vulkan-capable GPU"]
    async fn altcha_gpu_decimal_counters_cover_full_safe_integer_domain() -> Result<()> {
        use std::num::NonZeroU32;

        use challenge_solvers::{
            gpu::{AltchaSha256Solver, GpuContext},
            message::AltchaMessage,
        };

        let gpu = GpuContext::create().await?;
        gpu.warm_up().await?;
        for counter in [
            0_u64,
            u64::from(u32::MAX) - 1,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            (1_u64 << 53) - 1,
        ] {
            for pbkdf2 in [false, true] {
                let message = AltchaMessage {
                    nonce: [0x9a; 16],
                    salt: [0x27; 16],
                    cost: NonZeroU32::new(if pbkdf2 { 16 } else { 2 })
                        .context("positive iterations")?,
                    pbkdf2,
                    key_length: NonZeroU32::new(32).context("positive key length")?,
                };
                let mut solver = AltchaSha256Solver::from(message);
                solver.set_string_counter();
                solver.set_start_nonce(counter);
                solver.set_limit(counter + 1);
                let result = solver
                    .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                    .await?
                    .context("GPU failed to find a one-candidate full-mask solution")?;
                assert_eq!(result.nonce, counter, "pbkdf2={pbkdf2}");
                assert_eq!(solver.attempted_nonces(), 1);
            }
        }
        Ok(())
    }

    #[test]
    fn throughput_condition_preserves_comparison_type() {
        assert_eq!(
            benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_GT),
            (u64::MAX - 1, u64::MAX, SOLVE_TYPE_GT)
        );
        assert_eq!(
            benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_LT),
            (1, u64::MAX, SOLVE_TYPE_LT)
        );
        assert_eq!(
            benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_MASK),
            (0, u64::MAX, SOLVE_TYPE_MASK)
        );
    }
}
