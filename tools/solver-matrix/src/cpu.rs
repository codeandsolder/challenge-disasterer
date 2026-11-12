//! CPU protocol matrix fixtures, backend selection and measured rows.

use super::*;

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

pub fn cap_rsw_input(sample: u32) -> Result<(BigUint, BigUint)> {
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

/// Execute the same Cap format-2 RSW protocol path used by the actual
/// service, rather than benchmarking a separate, out-of-date bigint loop.
pub fn cap_rsw_solve(modulus: &BigUint, input: &BigUint) -> Result<BigUint> {
    let challenge = json!({
        "token": "benchmark-rsw",
        "format": 2,
        "challenges": [{
            "protocol": "rsw",
            "payload": {
                "N": modulus.to_str_radix(16),
                "x": input.to_str_radix(16),
                "t": CAP_RSW_T,
            }
        }]
    });
    let result = challenge_protocols::classic_pow::solve("capjs", challenge, Some("safe"), 1)?;
    anyhow::ensure!(result.implementation == "rsw", "unexpected Cap RSW backend");
    let hex = result
        .answer
        .pointer("/solutions/0/y")
        .and_then(Value::as_str)
        .context("Cap RSW solver did not return format-2 solution")?;
    BigUint::parse_bytes(hex.as_bytes(), 16).context("Cap RSW solver returned invalid hex")
}

fn run_cap_rsw_case(meta: &Meta, sample: u32, args: &Args) -> Result<()> {
    let (modulus, x) = cap_rsw_input(sample)?;
    let started = Instant::now();
    let y = cap_rsw_solve(&modulus, &x)?;
    let solve_ns = started.elapsed().as_nanos();
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
        implementation: "rug-gmp-modpow".to_owned(),
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

pub fn run_cpu_matrix(meta: &Meta, args: &Args) -> Result<()> {
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
