use std::{
    num::{NonZeroU8, NonZeroU32, NonZeroU64},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "gpu")]
use anyhow::Context as _;
use anyhow::Result;
use argon2::Argon2;
use clap::{Parser, ValueEnum};
use hashwx::HashWx;
use pow_core::{
    build_mcaptcha_prefix, compute_mask_cerberus, compute_mask_goaway, compute_mask_zero_nibbles,
    compute_target_mcaptcha,
    message::{
        AltchaMessage, BinaryMessage, CapJSEmitter, CerberusBinaryMessage, CerberusDecimalMessage,
        CerberusMessage, DecimalMessage, GoAwayMessage,
    },
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LT, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Correctness,
    Throughput,
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
    /// Include the generalized WGPU backend selected by WGPU_ADAPTER_NAME/default selection.
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
        pow_core::solver::SOLVE_TYPE_GT => value > target,
        pow_core::solver::SOLVE_TYPE_LT => value < target,
        pow_core::solver::SOLVE_TYPE_MASK => value & mask == target & mask,
        _ => false,
    }
}

fn phase_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Correctness => "correctness",
        Mode::Throughput => "throughput",
    }
}

fn benchmark_condition(mode: Mode, target: u64, mask: u64, ty: u8) -> (u64, u64, u8) {
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
    let solved = solver.solve_dyn_local(target, ty, mask);
    let solve_ns = started.elapsed().as_nanos();
    let attempted = solver.attempted();
    let (nonce, ok) = solved
        .map(|(nonce, hash)| (Some(nonce), condition_ok(ty, target, mask, hash)))
        .unwrap_or((None, true));
    if matches!(mode, Mode::Throughput) {
        anyhow::ensure!(
            solved.is_none(),
            "throughput target unexpectedly found a proof"
        );
        anyhow::ensure!(
            attempted == limit,
            "throughput run attempted {attempted} candidates, expected exactly {limit}"
        );
    }
    emit(Record {
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

fn emit(record: Record) -> Result<()> {
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

fn rate(work: u64, ns: u128) -> f64 {
    if ns == 0 {
        0.0
    } else {
        work as f64 * 1e9 / ns as f64
    }
}

fn decimal_backends(message: impl Fn() -> DecimalMessage) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(pow_core::solver::safe::DecimalSolver::from(message())),
        ),
        ("auto", Box::new(pow_core::DecimalSolver::from(message()))),
    ];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(pow_core::solver::avx2::DecimalSolver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("sha") {
            out.push((
                "sha-ni",
                Box::new(pow_core::solver::sha_ni::DecimalSolver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(pow_core::solver::avx512::DecimalSolver::from(message())),
            ));
        }
    }
    out
}

fn binary_backends(message: impl Fn() -> BinaryMessage) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(pow_core::solver::safe::BinarySolver::from(message())),
        ),
        ("auto", Box::new(pow_core::BinarySolver::from(message()))),
    ];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        out.push((
            "avx2",
            Box::new(pow_core::solver::avx2::BinarySolver::from(message())),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        out.push((
            "avx512",
            Box::new(pow_core::solver::avx512::BinarySolver::from(message())),
        ));
    }
    out
}

fn leading_zero_bits(hash: &[u8; 32], difficulty: u32) -> bool {
    let full_bytes = (difficulty / 8) as usize;
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
                let prefix = u64::from_be_bytes(hash[..8].try_into().expect("8-byte prefix"));
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
    emit(Record {
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
            Box::new(pow_core::solver::safe::GoAwaySolver::from(message())),
        ),
        ("auto", Box::new(pow_core::GoAwaySolver::from(message()))),
    ];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(pow_core::solver::avx2::GoAwaySolver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("sha") {
            out.push((
                "sha-ni",
                Box::new(pow_core::solver::sha_ni::GoAwaySolver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(pow_core::solver::avx512::GoAwaySolver::from(message())),
            ));
        }
    }
    out
}

fn cerberus_backends(
    message: impl Fn() -> CerberusMessage,
) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(pow_core::solver::safe::CerberusSolver::from(message())),
        ),
        ("auto", Box::new(pow_core::CerberusSolver::from(message()))),
    ];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(pow_core::solver::avx2::CerberusSolver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(pow_core::solver::avx512::CerberusSolver::from(message())),
            ));
        }
    }
    out
}

fn altcha_backends(message: impl Fn() -> AltchaMessage) -> Vec<(&'static str, Box<dyn DynPow>)> {
    let mut out: Vec<(&'static str, Box<dyn DynPow>)> = vec![
        (
            "safe",
            Box::new(pow_core::solver::safe::AltchaSha256Solver::from(message())),
        ),
        (
            "auto",
            Box::new(pow_core::AltchaSha256Solver::from(message())),
        ),
    ];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push((
                "avx2",
                Box::new(pow_core::solver::avx2::AltchaSha256Solver::from(message())),
            ));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push((
                "avx512",
                Box::new(pow_core::solver::avx512::AltchaSha256Solver::from(message())),
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

fn run_hashwx_case(meta: &Meta, sample: u32, args: &Args) -> Result<()> {
    const NONCES_PER_HASH: u64 = 65_536;
    const DIFFICULTY: u64 = 1_000_000;

    let mut challenge = [0_u8; 32];
    challenge[..4].copy_from_slice(&sample.to_le_bytes());
    for (index, byte) in challenge[4..].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(17).wrapping_add(0x5a);
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
    emit(Record {
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

fn run_cpu_matrix(meta: &Meta, args: &Args) -> Result<()> {
    for sample in 0..args.samples {
        let sample_bytes = sample.to_le_bytes();

        // Anubis: same decimal message semantics as the live service, d5 ~= 2^20 expected candidates.
        let mut anubis_prefix = vec![b'a'; 96];
        anubis_prefix[..4].copy_from_slice(&sample_bytes);
        let anubis_mask = compute_mask_zero_nibbles(NonZeroU8::new(5).unwrap());
        for (implementation, solver) in
            decimal_backends(|| DecimalMessage::new(&anubis_prefix, 0).unwrap())
        {
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

        // haphash: same decimal SHA-256 work primitive, but distinct protocol prefixes.
        // Use d5 (~2^20 expected candidates) for stable apples-to-apples throughput.
        for (variant, prefix) in [
            (
                "haproxy",
                format!("203.0.113.{};example.test;1759954402;", sample % 255),
            ),
            ("debian-varnish", format!("102026194332.0x{sample:064x};")),
        ] {
            let difficulty = NonZeroU8::new(5).unwrap();
            let mask = compute_mask_zero_nibbles(difficulty);
            for (implementation, solver) in
                decimal_backends(|| DecimalMessage::new(prefix.as_bytes(), 0).unwrap())
            {
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

        // Anubis 1.28 WASM SHA-256: raw challenge bytes plus a 32-bit nonce.
        // End the challenge with 0x80 so upstream selects little-endian nonce bytes directly.
        let mut anubis_wasm_data = vec![0x42_u8; 48];
        anubis_wasm_data[..4].copy_from_slice(&sample_bytes);
        *anubis_wasm_data.last_mut().expect("nonempty challenge") = 0x80;
        let anubis_sha_mask = u64::MAX << (64 - 20);
        for (implementation, solver) in binary_backends(|| {
            BinaryMessage::new(
                &anubis_wasm_data,
                NonZeroU8::new(4).expect("four is nonzero"),
            )
        }) {
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

        // mCaptcha: canonical salt + bincode-style string prefix, expected candidates ~= difficulty factor.
        let mut mcaptcha_prefix = Vec::new();
        build_mcaptcha_prefix(
            &mut mcaptcha_prefix,
            "challenge-disasterer-matrix",
            &format!("matrix-salt-{sample:08x}"),
        );
        let mcaptcha_factor = 1_000_000_u64;
        let mcaptcha_target = compute_target_mcaptcha(NonZeroU64::new(mcaptcha_factor).unwrap());
        for (implementation, solver) in
            decimal_backends(|| DecimalMessage::new(&mcaptcha_prefix, 0).unwrap())
        {
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

        // Cap.js: real emitter shape, one subchallenge at d5 ~= 2^20 expected candidates.
        let token = format!("challenge-disasterer-matrix-token-{sample:08x}");
        let emitter = CapJSEmitter::new(token.as_bytes());
        let mut cap_salt = vec![0_u8; 32];
        let mut cap_target = [0_u32; 2];
        emitter.emit(&mut cap_salt, &mut cap_target, 1);
        let cap_difficulty = 5_u8;
        let cap_mask = u64::MAX << (64 - u64::from(cap_difficulty) * 4);
        let cap_target = (u64::from(cap_target[0]) << 32) | u64::from(cap_target[1]);
        for (implementation, solver) in
            decimal_backends(|| DecimalMessage::new_f64(&cap_salt, 0).unwrap().0)
        {
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

        // Cap.js format-2 HashWX at the production default reuse window. Include
        // per-seed program generation/JIT cost rather than benchmarking only exec().
        run_hashwx_case(meta, sample, args)?;

        // go-away: actual 32-byte challenge message, d20 ~= 2^20 expected candidates.
        let mut go_bytes = [0_u8; 32];
        for (i, byte) in go_bytes.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(sample as u8);
        }
        let go_difficulty = NonZeroU8::new(20).unwrap();
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

        // Legacy Cerberus (<0.4.6) decimal nonce format, retained because the service supports it.
        let cerberus_legacy_salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        let cerberus_legacy_difficulty = NonZeroU8::new(10).unwrap();
        let cerberus_legacy_mask = compute_mask_cerberus(cerberus_legacy_difficulty);
        for (implementation, solver) in cerberus_backends(|| {
            CerberusMessage::Decimal(
                CerberusDecimalMessage::new(cerberus_legacy_salt.as_bytes(), 0).unwrap(),
            )
        }) {
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
        let cerberus_difficulty = NonZeroU8::new(10).unwrap();
        let cerberus_mask = compute_mask_cerberus(cerberus_difficulty);
        for (implementation, solver) in cerberus_backends(|| {
            CerberusMessage::Binary(CerberusBinaryMessage::new(cerberus_salt.as_bytes(), 0))
        }) {
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

        // ALTCHA SHA-256, 20-bit prefix ~= 2^20 candidates.
        let altcha_message = || AltchaMessage {
            nonce: [sample as u8; 16],
            salt: [sample.wrapping_add(17) as u8; 16],
            cost: NonZeroU32::new(2).unwrap(),
            pbkdf2: false,
            key_length: NonZeroU32::new(32).unwrap(),
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
        let altcha_pbkdf2 = || AltchaMessage {
            nonce: [sample.wrapping_add(1) as u8; 16],
            salt: [sample.wrapping_add(33) as u8; 16],
            cost: NonZeroU32::new(16).unwrap(),
            pbkdf2: true,
            key_length: NonZeroU32::new(32).unwrap(),
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
    }
    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_gpu_matrix(meta: &Meta, args: &Args) -> Result<()> {
    let mut gpu = pow_core::gpu::GpuContext::create()
        .await
        .context("create generalized WGPU context")?;
    gpu.warm_up()
        .await
        .context("warm generalized WGPU context")?;
    let adapter = gpu.adapter().to_owned();

    for sample in 0..args.samples {
        let sample_bytes = sample.to_le_bytes();
        let mut prefix = vec![b'a'; 96];
        prefix[..4].copy_from_slice(&sample_bytes);
        let mask = compute_mask_zero_nibbles(NonZeroU8::new(5).unwrap());
        let mut solver =
            pow_core::gpu::DecimalSolver::from(DecimalMessage::new(&prefix, 0).unwrap());
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        for (variant, prefix) in [
            (
                "haproxy",
                format!("203.0.113.{};example.test;1759954402;", sample % 255),
            ),
            ("debian-varnish", format!("102026194332.0x{sample:064x};")),
        ] {
            let difficulty = NonZeroU8::new(5).unwrap();
            let mask = compute_mask_zero_nibbles(difficulty);
            let mut solver = pow_core::gpu::DecimalSolver::from(
                DecimalMessage::new(prefix.as_bytes(), 0).unwrap(),
            );
            solver.set_limit(args.limit);
            let started = Instant::now();
            let (bench_target, bench_mask, _) =
                benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
            let solved = solver
                .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
                solved.map(|x| (x.nonce, x.hash)),
                &adapter,
                args.mode,
                args.limit,
            )?;
        }

        let mut anubis_wasm_data = vec![0x42_u8; 48];
        anubis_wasm_data[..4].copy_from_slice(&sample_bytes);
        *anubis_wasm_data.last_mut().expect("nonempty challenge") = 0x80;
        let anubis_sha_mask = u64::MAX << (64 - 20);
        let mut solver = pow_core::gpu::BinarySolver::from(BinaryMessage::new(
            &anubis_wasm_data,
            NonZeroU8::new(4).expect("four is nonzero"),
        ));
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, anubis_sha_mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        let mut mcaptcha_prefix = Vec::new();
        build_mcaptcha_prefix(
            &mut mcaptcha_prefix,
            "challenge-disasterer-matrix",
            &format!("matrix-salt-{sample:08x}"),
        );
        let factor = 1_000_000_u64;
        let target = compute_target_mcaptcha(NonZeroU64::new(factor).unwrap());
        let mut solver =
            pow_core::gpu::DecimalSolver::from(DecimalMessage::new(&mcaptcha_prefix, 0).unwrap());
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, bench_ty) =
            benchmark_condition(args.mode, target, u64::MAX, SOLVE_TYPE_GT);
        let solved = if bench_ty == SOLVE_TYPE_MASK {
            solver
                .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
                .await?
        } else {
            solver
                .solve::<SOLVE_TYPE_GT>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        let token = format!("challenge-disasterer-matrix-token-{sample:08x}");
        let emitter = CapJSEmitter::new(token.as_bytes());
        let mut cap_salt = vec![0_u8; 32];
        let mut cap_target = [0_u32; 2];
        emitter.emit(&mut cap_salt, &mut cap_target, 1);
        let cap_difficulty = 5_u8;
        let cap_mask = u64::MAX << (64 - u64::from(cap_difficulty) * 4);
        let cap_target = (u64::from(cap_target[0]) << 32) | u64::from(cap_target[1]);
        let mut solver =
            pow_core::gpu::DecimalSolver::from(DecimalMessage::new_f64(&cap_salt, 0).unwrap().0);
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, cap_target, cap_mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        let mut go_bytes = [0_u8; 32];
        for (i, byte) in go_bytes.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(sample as u8);
        }
        let d = NonZeroU8::new(20).unwrap();
        let mask = compute_mask_goaway(d);
        let mut solver = pow_core::gpu::GoAwaySolver::from(GoAwayMessage::new_bytes(&go_bytes, 0));
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        // Legacy Cerberus (<0.4.6) decimal nonce format.
        let legacy_salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        let legacy_d = NonZeroU8::new(10).unwrap();
        let legacy_mask = compute_mask_cerberus(legacy_d);
        let mut solver = pow_core::gpu::CerberusSolver::from(CerberusMessage::Decimal(
            CerberusDecimalMessage::new(legacy_salt.as_bytes(), 0).unwrap(),
        ));
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, legacy_mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        let salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        let d = NonZeroU8::new(10).unwrap();
        let mask = compute_mask_cerberus(d);
        let mut solver = pow_core::gpu::CerberusSolver::from(CerberusMessage::Binary(
            CerberusBinaryMessage::new(salt.as_bytes(), 0),
        ));
        solver.set_limit(args.limit);
        let started = Instant::now();
        let (bench_target, bench_mask, _) =
            benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
        let solved = solver
            .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
            solved.map(|x| (x.nonce, x.hash)),
            &adapter,
            args.mode,
            args.limit,
        )?;

        for (variant, pbkdf2, cost, mask) in [
            ("sha256", false, 2_u32, 0xffff_f000_0000_0000_u64),
            ("pbkdf2-sha256", true, 16_u32, 0xffff_0000_0000_0000_u64),
        ] {
            let message = AltchaMessage {
                nonce: [sample as u8; 16],
                salt: [sample.wrapping_add(17) as u8; 16],
                cost: NonZeroU32::new(cost).unwrap(),
                pbkdf2,
                key_length: NonZeroU32::new(32).unwrap(),
            };
            let mut solver = pow_core::gpu::AltchaSha256Solver::from(message);
            solver.set_limit(args.limit);
            let started = Instant::now();
            let (bench_target, bench_mask, _) =
                benchmark_condition(args.mode, 0, mask, SOLVE_TYPE_MASK);
            let solved = solver
                .solve::<SOLVE_TYPE_MASK>(&mut gpu, bench_target, bench_mask)
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
                solved.map(|x| (x.nonce, x.hash)),
                &adapter,
                args.mode,
                args.limit,
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
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
    solved: Option<(u64, [u32; 8])>,
    adapter: &str,
    mode: Mode,
    limit: u64,
) -> Result<()> {
    let (actual_target, actual_mask, actual_ty) = benchmark_condition(mode, target, mask, ty);
    let (nonce, ok) = solved
        .map(|(nonce, hash)| {
            (
                Some(nonce),
                condition_ok(actual_ty, actual_target, actual_mask, hash),
            )
        })
        .unwrap_or((None, true));
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
    emit(Record {
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
                serde_json::to_string(&pow_core::gpu::GpuContext::adapter_names().await)?
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
    if args.gpu {
        #[cfg(feature = "gpu")]
        run_gpu_matrix(&meta, &args).await?;
        #[cfg(not(feature = "gpu"))]
        anyhow::bail!("--gpu true requires the solver-matrix gpu feature");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
