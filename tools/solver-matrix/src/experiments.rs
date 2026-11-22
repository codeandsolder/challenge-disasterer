//! Explicit input configuration for local hardware benchmark scenarios.

use super::gpu::{
    run_gpu_altcha_cases, run_gpu_anubis_haphash_cases, run_gpu_capjs_cases,
    run_gpu_cerberus_cases, run_gpu_goaway_case, run_gpu_wasm_mcaptcha_cases,
};
use super::*;
use clap::{Args as ClapArgs, Subcommand};
use std::{num::NonZeroUsize, str::FromStr};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Shader {
    #[value(name = "merged-vec4-fc56391")]
    MergedVec4Fc56391,
    Directed,
    InterleavedScratch,
    #[value(name = "tiled-scratch-32")]
    TiledScratch32,
    #[value(name = "tiled-scratch-64")]
    TiledScratch64,
}

impl Shader {
    const fn label(self) -> &'static str {
        match self {
            Self::MergedVec4Fc56391 => "merged-vec4-fc56391",
            Self::Directed => "directed",
            Self::InterleavedScratch => "interleaved-scratch",
            Self::TiledScratch32 => "tiled-scratch-32",
            Self::TiledScratch64 => "tiled-scratch-64",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RomixShape([u32; 3]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KdfShape([u32; 4]);

const ROMIX_SHAPES: [[u32; 3]; 9] = [
    [64, 1, 1],
    [64, 1, 64],
    [256, 1, 64],
    [256, 1, 256],
    [256, 8, 1],
    [256, 8, 64],
    [256, 8, 256],
    [1024, 8, 64],
    [1024, 8, 256],
];
const KDF_SHAPES: [[u32; 4]; 9] = [
    [64, 1, 1, 1],
    [64, 1, 1, 64],
    [256, 1, 1, 64],
    [256, 1, 1, 256],
    [256, 8, 1, 1],
    [256, 8, 1, 64],
    [256, 8, 1, 256],
    [1024, 8, 1, 64],
    [256, 8, 4, 8],
];

fn parse_shape<const N: usize>(input: &str, supported: &[[u32; N]]) -> Result<[u32; N]> {
    let shape: [u32; N] = input
        .split('/')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected {N} slash-separated shape parameters"))?;
    anyhow::ensure!(
        supported.contains(&shape),
        "unknown benchmark scenario: {input}"
    );
    Ok(shape)
}

impl FromStr for RomixShape {
    type Err = anyhow::Error;
    fn from_str(input: &str) -> Result<Self> {
        Ok(Self(parse_shape(input, &ROMIX_SHAPES)?))
    }
}

impl FromStr for KdfShape {
    type Err = anyhow::Error;
    fn from_str(input: &str) -> Result<Self> {
        Ok(Self(parse_shape(input, &KDF_SHAPES)?))
    }
}

#[derive(ClapArgs, Debug)]
pub struct RomixOptions {
    /// Shader variants to compare, in initial execution order.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "merged-vec4-fc56391,directed"
    )]
    shaders: Vec<Shader>,
    #[arg(long, default_value = "5")]
    trials: NonZeroU32,
    /// Select n/r/lanes scenarios; omit for all predefined cases.
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<RomixShape>,
}

#[derive(ClapArgs, Debug)]
pub struct KdfOptions {
    /// Two distinct shader variants to compare.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "merged-vec4-fc56391,directed"
    )]
    shaders: Vec<Shader>,
    /// Select n/r/p/count scenarios; omit for all predefined cases.
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<KdfShape>,
}

#[derive(ClapArgs, Debug)]
pub struct CrossoverOptions {
    #[arg(long, default_value = "8")]
    threads: NonZeroUsize,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    request_counts: Vec<NonZeroUsize>,
    #[arg(long, default_value = "5")]
    trials: NonZeroU32,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,64,1024,8192,65536,262144,1048576"
    )]
    budgets: Vec<NonZeroU64>,
}

#[derive(ClapArgs, Debug)]
pub struct MajorityOptions {
    #[arg(long, default_value = "10")]
    trials: NonZeroU32,
    #[arg(long, default_value = "1048576")]
    limit: NonZeroU64,
}

#[derive(Subcommand, Debug)]
pub enum Experiment {
    ScryptRomix(RomixOptions),
    ScryptKdf(KdfOptions),
    Sha256Crossover(CrossoverOptions),
    Sha256Majority(MajorityOptions),
}

fn distinct_shaders(names: &[Shader], exact_pair: bool) -> Result<()> {
    anyhow::ensure!(
        names.len() >= 2 && (!exact_pair || names.len() == 2),
        "select {} distinct shaders",
        if exact_pair {
            "exactly two"
        } else {
            "at least two"
        }
    );
    anyhow::ensure!(
        names
            .iter()
            .enumerate()
            .all(|(i, name)| !names[..i].contains(name)),
        "shader selections must be distinct"
    );
    Ok(())
}

fn select_variants<'a, T>(
    names: &[Shader],
    variants: &[(Shader, &'a T)],
) -> Result<Vec<(&'static str, &'a T)>> {
    names
        .iter()
        .map(|name| {
            let (_, solver) = variants
                .iter()
                .find(|(variant, _)| variant == name)
                .context("missing shader variant")?;
            Ok((name.label(), *solver))
        })
        .collect()
}

impl Experiment {
    fn validate(&mut self) -> Result<()> {
        match self {
            Self::ScryptRomix(options) => {
                distinct_shaders(&options.shaders, false)?;
                let selected = &options.shapes;
                options.shapes = ROMIX_SHAPES
                    .into_iter()
                    .map(RomixShape)
                    .filter(|shape| selected.is_empty() || selected.contains(shape))
                    .collect();
            }
            Self::ScryptKdf(options) => {
                distinct_shaders(&options.shaders, true)?;
                let selected = &options.shapes;
                options.shapes = KDF_SHAPES
                    .into_iter()
                    .map(KdfShape)
                    .filter(|shape| selected.is_empty() || selected.contains(shape))
                    .collect();
            }
            Self::Sha256Crossover(_) => {}
            Self::Sha256Majority(options) => {
                anyhow::ensure!(
                    options.limit.get().is_multiple_of(8192),
                    "limit must be a positive multiple of 8192"
                );
            }
        }
        Ok(())
    }

    #[expect(
        clippy::future_not_send,
        reason = "CLI experiment is awaited directly on the main thread; upstream WGPU initialization returns a local future"
    )]
    pub async fn run(mut self) -> Result<()> {
        self.validate()?;
        match self {
            Self::ScryptRomix(options) => scrypt_romix_wgpu_phase_timestamps(options).await,
            Self::ScryptKdf(options) => scrypt_full_kdf_interleaved_ab(options).await,
            Self::Sha256Crossover(options) => sha256_concurrent_cpu_gpu_crossover(options).await,
            Self::Sha256Majority(options) => sha256_gpu_majority_interleaved(options).await,
        }
    }
}

/// Interleaved hardware timestamp comparison of merged vec4 and directed
/// `ROMix` revisions, with full `RustCrypto` RFC7914 key verification.
#[expect(
    clippy::future_not_send,
    reason = "WGPU context initialization returns a local future; the experiment never spawns this future onto another thread"
)]
async fn scrypt_romix_wgpu_phase_timestamps(options: RomixOptions) -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptRomixGpu};

    let gpu = GpuContext::create_with_timestamps().await?;
    // Exact merged vec4 shader from challenge-solvers@fc56391.
    // The two implementations use the same context, queue and timestamp clock.
    let baseline = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_vec4_fc56391.wgsl"),
    );
    let directed = ScryptRomixGpu::new(&gpu);
    let interleaved = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_interleaved_scratch.wgsl"),
    );
    let tiled32 = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_32.wgsl"),
    );
    let tiled64 = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_64.wgsl"),
    );
    let variants = [
        (Shader::MergedVec4Fc56391, &baseline),
        (Shader::Directed, &directed),
        (Shader::InterleavedScratch, &interleaved),
        (Shader::TiledScratch32, &tiled32),
        (Shader::TiledScratch64, &tiled64),
    ];
    let pair = select_variants(&options.shaders, &variants)?;
    let trials = options.trials.get();
    let salt: Vec<u8> = (0_u8..53).map(|i| i.wrapping_mul(31)).collect();
    for RomixShape([n, r, lanes]) in options.shapes {
        let shape = format!("{n}/{r}/{lanes}");
        let lanes = usize::try_from(lanes)?;
        let block_bytes = usize::try_from(r)?
            .checked_mul(128)
            .context("scrypt profile block size overflow")?;
        let passwords = (0..lanes)
            .map(|i| format!("scrypt-profile-candidate-{i}").into_bytes())
            .collect::<Vec<Vec<u8>>>();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, 1)
            .map_err(|_| anyhow::anyhow!("invalid scrypt profiling parameters"))?;
        let mut initial = Vec::with_capacity(lanes * block_bytes);
        let mut expected = Vec::with_capacity(lanes);
        for password in &passwords {
            let mut block = vec![0_u8; block_bytes];
            pbkdf2::pbkdf2_hmac::<Sha256>(password, &salt, 1, &mut block);
            initial.extend_from_slice(&block);
            let mut output = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut output)
                .map_err(|_| anyhow::anyhow!("independent scrypt profile reference failed"))?;
            expected.push(output);
        }
        // Pre-compile and warm both pipelines before interleaved timing.
        for (_, solver) in &pair {
            let (mixed, _) = solver.mix_blocks_with_timestamps(&initial, n, r).await?;
            anyhow::ensure!(mixed.len() == initial.len(), "ROMix output length mismatch");
        }
        for trial in 0..trials {
            // Rotate execution order to keep every implementation equally
            // likely to run first despite thermals and competing GPU work.
            for position in 0..pair.len() {
                let (variant, romix) = pair[(usize::try_from(trial)? + position) % pair.len()];
                let wall_start = Instant::now();
                let (mixed, timestamps) = romix.mix_blocks_with_timestamps(&initial, n, r).await?;
                let wall_ns = wall_start.elapsed().as_nanos();
                anyhow::ensure!(mixed.len() == initial.len(), "ROMix output length mismatch");
                let mut actual = Vec::with_capacity(lanes);
                for (password, block) in passwords.iter().zip(mixed.chunks_exact(block_bytes)) {
                    let mut output = [0_u8; 32];
                    pbkdf2::pbkdf2_hmac::<Sha256>(password, block, 1, &mut output);
                    actual.push(output);
                }
                anyhow::ensure!(actual == expected, "ROMix profile: {shape}, {variant}");
                anyhow::ensure!(
                    !timestamps.is_empty(),
                    "GPU produced no ROMix phase timestamps"
                );
                for (batch, timing) in timestamps.iter().enumerate() {
                    println!(
                        "{}",
                        json!({
                            "variant":variant, "adapter":gpu.adapter(),
                            "n":n, "r":r, "lanes":lanes, "trial":trial, "batch":batch,
                            "fill_ticks":timing.fill_ticks,
                            "mix_ticks":timing.mix_ticks,
                            "timestamp_period_ns":timing.timestamp_period_ns,
                            "wall_ns":wall_ns, "verified":true,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}

fn cpu_solve(prefix: &[u8], budget: u64) -> Result<()> {
    let message = DecimalMessage::new(prefix, 0)
        .context("construct matching CPU Anubis decimal challenge")?;
    let mut solver = challenge_solvers::DecimalSolver::from(message);
    solver.set_limit(budget);
    let solution = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
    anyhow::ensure!(solution.is_none(), "unexpected 64-bit SHA CPU proof");
    anyhow::ensure!(
        solver.attempted_nonces() == budget,
        "CPU did not exhaust exact search budget"
    );
    Ok(())
}

async fn gpu_solve(
    gpu: &challenge_solvers::gpu::GpuContext,
    prefix: &[u8],
    budget: u64,
) -> Result<()> {
    let message = DecimalMessage::new(prefix, 0)
        .context("construct matching GPU Anubis decimal challenge")?;
    let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
    solver.set_limit(budget);
    let solution = solver.solve::<SOLVE_TYPE_MASK>(gpu, 0, u64::MAX).await?;
    anyhow::ensure!(solution.is_none(), "unexpected 64-bit SHA GPU proof");
    anyhow::ensure!(
        solver.attempted_nonces() == budget,
        "GPU did not exhaust exact search budget"
    );
    Ok(())
}

/// Measure the real crossover between independent SHA-256 GPU requests
/// (serialized on one shared GPU context) and the same independent requests
/// on one CPU worker or a bounded Rayon CPU pool. All tasks do exact work.
///
/// The measured CPU alternatives include their own solver construction.
/// The GPU includes real queue submissions and readbacks for each request.
#[expect(
    clippy::future_not_send,
    reason = "WGPU context initialization returns a local future; the experiment never spawns this future onto another thread"
)]
async fn sha256_concurrent_cpu_gpu_crossover(options: CrossoverOptions) -> Result<()> {
    use rayon::prelude::*;

    let threads = options.threads.get();
    let request_counts = options.request_counts;
    let max_requests = request_counts
        .iter()
        .map(|count| count.get())
        .max()
        .context("at least one request count required")?;
    let trials = options.trials.get();
    let budgets = options.budgets;

    let gpu = challenge_solvers::gpu::GpuContext::create().await?;
    gpu.warm_up().await?;
    let cpu_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let prefixes: Vec<Vec<u8>> = (0..max_requests)
        .map(|request| -> Result<Vec<u8>> {
            let mut prefix = vec![b'a'; 96];
            prefix[..8].copy_from_slice(&u64::try_from(request)?.to_le_bytes());
            Ok(prefix)
        })
        .collect::<Result<_>>()?;

    // Warm actual protocol encoding, CPU SIMD initialization, Rayon
    // workers, GPU pipeline and readback outside timed observations.
    cpu_pool.install(|| {
        prefixes
            .par_iter()
            .try_for_each(|prefix| cpu_solve(prefix, 8192))
    })?;
    for prefix in &prefixes {
        gpu_solve(&gpu, prefix, 8192).await?;
    }
    let adapter = gpu.adapter();
    for requests in request_counts {
        let requests = requests.get();
        let prefixes = &prefixes[..requests];
        for budget in &budgets {
            let budget = budget.get();
            for trial in 0..trials {
                // Rotating order prevents systematic first-backend advantage.
                for phase in 0..3_u32 {
                    let backend = (trial + phase) % 3;
                    let started = Instant::now();
                    match backend {
                        0 => {
                            for prefix in prefixes {
                                cpu_solve(prefix, budget)?;
                            }
                        }
                        1 => {
                            cpu_pool.install(|| {
                                prefixes
                                    .par_iter()
                                    .try_for_each(|prefix| cpu_solve(prefix, budget))
                            })?;
                        }
                        2 => {
                            for prefix in prefixes {
                                gpu_solve(&gpu, prefix, budget).await?;
                            }
                        }
                        _ => anyhow::bail!("invalid benchmark backend"),
                    }
                    let elapsed_ns = started.elapsed().as_nanos();
                    println!(
                        "{}",
                        json!({
                            "benchmark": "sha256-independent-requests-crossover-v1",
                            "host": read_trimmed("/proc/sys/kernel/hostname"),
                            "cpu": cpu_model(),
                            "gpu": adapter,
                            "protocol": "anubis",
                            "variant": "decimal",
                            "backend": match backend {
                                0 => "cpu-serial-auto",
                                1 => "cpu-parallel-auto",
                                2 => "gpu-serial-wgpu",
                                _ => anyhow::bail!("invalid benchmark backend"),
                            },
                            "threads": if backend == 1 { threads } else { 1 },
                            "requests": requests,
                            "budget_per_request": budget,
                            "attempted_work": u64::try_from(requests)?.checked_mul(budget).context("total work overflow")?,
                            "trial": trial,
                            "elapsed_ns": elapsed_ns,
                            "verified": true
                        })
                    );
                }
            }
        }
    }
    Ok(())
}

/// Compare equivalent SHA-256 majority formulas across all GPU protocol
/// families with the same device, queue, work budget and live solver paths.
/// Each pair uses identical challenge bytes, rotating execution order.
#[expect(
    clippy::future_not_send,
    reason = "WGPU context initialization returns a local future; the experiment never spawns this future onto another thread"
)]
async fn sha256_gpu_majority_interleaved(options: MajorityOptions) -> Result<()> {
    use challenge_solvers::gpu::GpuContext;

    let baseline = GpuContext::create().await?;
    let or_majority = baseline.with_alternative_majority();
    let samples = options.trials.get();
    let limit = options.limit.get();
    anyhow::ensure!(baseline.adapter() == or_majority.adapter());
    baseline.warm_up().await?;
    or_majority.warm_up().await?;
    let mut args = Args {
        samples: 1,
        limit,
        argon2_limit: 64,
        cpu: false,
        gpu: true,
        list_gpu_adapters: false,
        mode: Mode::Throughput,
        commit: String::new(),
        protocol: Vec::new(),
    };
    let host = read_trimmed("/proc/sys/kernel/hostname");
    let cpu = cpu_model();
    let protocols = [
        Protocol::Anubis,
        Protocol::Haphash,
        Protocol::Mcaptcha,
        Protocol::Capjs,
        Protocol::GoAway,
        Protocol::Cerberus,
        Protocol::Altcha,
    ];
    // Warm actual challenge formats, not only the synthetic startup
    // dispatch. Shader compilation and first-use driver work then occur
    // outside the timed pairs.
    args.limit = 8192;
    for (variant, gpu) in [("xor-majority", &baseline), ("or-majority", &or_majority)] {
        let meta = Meta {
            host: host.clone(),
            cpu_model: cpu.clone(),
            commit: format!("warmup-{variant}"),
        };
        for protocol in protocols {
            args.protocol = vec![protocol];
            run_sha256_majority_case(&meta, &args, gpu, protocol, 0).await?;
        }
    }
    args.limit = limit;
    for trial in 0..samples {
        let order = if trial.is_multiple_of(2) {
            [("xor-majority", &baseline), ("or-majority", &or_majority)]
        } else {
            [("or-majority", &or_majority), ("xor-majority", &baseline)]
        };
        for protocol in protocols {
            args.protocol = vec![protocol];
            for (variant, gpu) in order {
                let meta = Meta {
                    host: host.clone(),
                    cpu_model: cpu.clone(),
                    commit: variant.to_owned(),
                };
                run_sha256_majority_case(&meta, &args, gpu, protocol, trial).await?;
            }
        }
    }
    Ok(())
}

async fn run_sha256_majority_case(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    protocol: Protocol,
    sample: u32,
) -> Result<()> {
    match protocol {
        Protocol::Anubis => {
            run_gpu_anubis_haphash_cases(meta, args, gpu, gpu.adapter(), sample).await?;
            run_gpu_wasm_mcaptcha_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Haphash => {
            run_gpu_anubis_haphash_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Mcaptcha => {
            run_gpu_wasm_mcaptcha_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Capjs => run_gpu_capjs_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::GoAway => run_gpu_goaway_case(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Cerberus => run_gpu_cerberus_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Altcha => run_gpu_altcha_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Linear => anyhow::bail!("Linear.pub has no SHA-256 GPU solver"),
    }
}

fn kdf_variants(
    gpu: &challenge_solvers::gpu::GpuContext,
) -> [(Shader, challenge_solvers::gpu::ScryptKdfGpu<'_>); 5] {
    use challenge_solvers::gpu::ScryptKdfGpu;
    let baseline = ScryptKdfGpu::new_from_romix_wgsl(
        gpu,
        include_str!("../fixtures/scrypt_romix_vec4_fc56391.wgsl"),
    );
    let directed = ScryptKdfGpu::new(gpu);
    let interleaved = ScryptKdfGpu::new_from_romix_wgsl(
        gpu,
        include_str!("../fixtures/scrypt_romix_interleaved_scratch.wgsl"),
    );
    let tiled32 = ScryptKdfGpu::new_from_romix_wgsl(
        gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_32.wgsl"),
    );
    let tiled64 = ScryptKdfGpu::new_from_romix_wgsl(
        gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_64.wgsl"),
    );
    [
        (Shader::MergedVec4Fc56391, baseline),
        (Shader::Directed, directed),
        (Shader::InterleavedScratch, interleaved),
        (Shader::TiledScratch32, tiled32),
        (Shader::TiledScratch64, tiled64),
    ]
}

/// Same-device paired benchmark of complete GPU PBKDF2/ROMix/PBKDF2
/// using the exact merged vec4 shader and the directed candidate shader.
/// Every full derived key is checked against independent `RustCrypto` scrypt.
#[expect(
    clippy::future_not_send,
    reason = "WGPU context initialization returns a local future; the experiment never spawns this future onto another thread"
)]
async fn scrypt_full_kdf_interleaved_ab(options: KdfOptions) -> Result<()> {
    use challenge_solvers::gpu::GpuContext;
    use rayon::prelude::*;

    let cpu_pool = rayon::ThreadPoolBuilder::new().build()?;
    let cpu_threads = cpu_pool.current_num_threads();
    let gpu = GpuContext::create().await?;
    let owned_variants = kdf_variants(&gpu);
    let variants = owned_variants
        .each_ref()
        .map(|(shader, solver)| (*shader, solver));
    let pair = select_variants(&options.shaders, &variants)?;
    let salt: Vec<u8> = (0_u8..59).map(|i| i.wrapping_mul(23)).collect();
    for KdfShape([n, r, p, count]) in options.shapes {
        let shape = format!("{n}/{r}/{p}/{count}");
        let count = usize::try_from(count)?;
        let passwords: Vec<Vec<u8>> = (0..count)
            .map(|index| {
                (0..(65 + index % 3))
                    .map(|byte| {
                        byte.to_le_bytes()[0]
                            .wrapping_mul(31)
                            .wrapping_add(index.to_le_bytes()[0])
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = passwords.iter().map(Vec::as_slice).collect();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid RFC7914 parameters: {shape}"))?;
        let cpu_start = Instant::now();
        let mut expected = Vec::with_capacity(count);
        for password in &passwords {
            let mut derived = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut derived)
                .map_err(|_| anyhow::anyhow!("RustCrypto scrypt failed: {shape}"))?;
            expected.push(derived.to_vec());
        }
        let cpu_serial_ns = cpu_start.elapsed().as_nanos();
        for (_, solver) in &pair {
            let warm = solver.derive_keys(&refs, &salt, n, r, p, 32).await?;
            anyhow::ensure!(warm == expected, "full KDF A/B warmup: {shape}");
        }
        for trial in 0..5_u32 {
            // Compare against a warm, concurrently dispatched CPU
            // reference rather than treating serial RustCrypto as the
            // achievable CPU throughput on a multi-core machine.
            let cpu_parallel_start = Instant::now();
            let parallel_reference: Vec<Vec<u8>> = cpu_pool.install(|| {
                passwords
                    .par_iter()
                    .map(|password| -> Result<Vec<u8>> {
                        let mut key = [0_u8; 32];
                        scrypt::scrypt(password, &salt, &params, &mut key)
                            .map_err(|_| anyhow::anyhow!("CPU reference failed: {shape}"))?;
                        Ok(key.to_vec())
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            let cpu_parallel_ns = cpu_parallel_start.elapsed().as_nanos();
            anyhow::ensure!(
                parallel_reference == expected,
                "CPU parallel reference: {shape}"
            );
            let order = if trial.is_multiple_of(2) {
                [pair[0], pair[1]]
            } else {
                [pair[1], pair[0]]
            };
            for (variant, solver) in order {
                let start = Instant::now();
                let actual = solver.derive_keys(&refs, &salt, n, r, p, 32).await?;
                let gpu_full_kdf_ns = start.elapsed().as_nanos();
                anyhow::ensure!(
                    actual == expected,
                    "full RFC7914 KDF mismatch: {shape}, {variant}, trial={trial}"
                );
                println!(
                    "{}",
                    json!({
                        "variant": variant, "adapter": gpu.adapter(),
                        "n": n, "r": r, "p": p, "candidates": count,
                        "trial": trial, "gpu_full_kdf_ns": gpu_full_kdf_ns,
                        "cpu_serial_reference_ns": cpu_serial_ns,
                        "cpu_parallel_ns": cpu_parallel_ns,
                        "cpu_threads": cpu_threads,
                        "verified": true,
                    })
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod options_tests {
    use super::*;

    #[test]
    fn default_shapes_and_explicit_selection() -> Result<()> {
        let mut command = Cli::try_parse_from(["matrix", "scrypt-romix"])?
            .experiment
            .context("missing experiment")?;
        command.validate()?;
        let Experiment::ScryptRomix(options) = command else {
            anyhow::bail!("unexpected experiment");
        };
        assert_eq!(options.shapes, ROMIX_SHAPES.map(RomixShape));
        let mut command = Cli::try_parse_from([
            "matrix",
            "scrypt-kdf",
            "--shapes",
            "256/8/4/8",
            "--shaders",
            "directed,tiled-scratch-32",
        ])?
        .experiment
        .context("missing experiment")?;
        command.validate()?;
        let Experiment::ScryptKdf(options) = command else {
            anyhow::bail!("unexpected experiment");
        };
        assert_eq!(options.shapes, [KdfShape([256, 8, 4, 8])]);
        assert_eq!(options.shaders, [Shader::Directed, Shader::TiledScratch32]);
        Ok(())
    }

    #[test]
    fn reject_invalid_scenarios_before_initializing_gpu() -> Result<()> {
        assert!(Cli::try_parse_from(["matrix", "scrypt-kdf", "--shapes", "0/1/1/1"]).is_err());
        assert!(Cli::try_parse_from(["matrix", "sha256-crossover", "--threads", "0"]).is_err());
        assert!(Cli::try_parse_from(["matrix", "sha256-crossover", "--budgets", "0"]).is_err());
        for args in [
            ["matrix", "scrypt-romix", "--shaders", "directed,directed"],
            ["matrix", "scrypt-kdf", "--shaders", "directed"],
            ["matrix", "sha256-majority", "--limit", "8193"],
        ] {
            let mut command = Cli::try_parse_from(args)?
                .experiment
                .context("missing experiment")?;
            assert!(command.validate().is_err());
        }
        Ok(())
    }
}
