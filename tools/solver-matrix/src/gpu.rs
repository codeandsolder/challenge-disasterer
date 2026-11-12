//! WGPU protocol matrix fixtures, measured rows and device orchestration.

use super::*;

pub(super) fn run_gpu_matrix<'a>(
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

pub(super) async fn run_gpu_anubis_haphash_cases(
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

pub(super) async fn run_gpu_wasm_mcaptcha_cases(
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

pub(super) async fn run_gpu_capjs_cases(
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

pub(super) async fn run_gpu_goaway_case(
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

pub(super) async fn run_gpu_cerberus_cases(
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

pub(super) async fn run_gpu_altcha_cases(
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
