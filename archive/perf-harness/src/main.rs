use std::{env, hint::black_box, time::Instant};
use pow_core::{
    message::{DecimalMessage, GoAwayMessage},
    solver::{Solver, SOLVE_TYPE_MASK},
};
#[cfg(feature = "full")]
use pow_core::solver::SOLVE_TYPE_LEADING_ZERO_BITS;

const CPU_LIMIT: u64 = 8_000_000;
const GPU_LIMIT: u64 = 32_000_000;
const REPEATS: usize = 5;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a,b| a.total_cmp(b));
    xs[xs.len()/2]
}

fn bench_cpu<S, F>(name: &str, protocol: &str, mode: &str, make: F, ty: u8, target: u64, mask: u64)
where
    S: Solver<Output=[u32;8]>,
    F: Fn() -> S,
{
    let mut rates = Vec::new();
    for i in 0..REPEATS + 1 {
        let mut solver = make();
        solver.set_limit(CPU_LIMIT);
        let start = Instant::now();
        let result = match ty {
            SOLVE_TYPE_MASK => solver.solve::<SOLVE_TYPE_MASK>(target, mask),
            #[cfg(feature = "full")]
            SOLVE_TYPE_LEADING_ZERO_BITS => solver.solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(target, mask),
            _ => unreachable!(),
        };
        black_box(&result);
        let ns = start.elapsed().as_nanos() as f64;
        let attempted = solver.get_attempted_nonces();
        assert_eq!(attempted, CPU_LIMIT);
        if i > 0 { rates.push(attempted as f64 * 1e9 / ns); }
    }
    let med = median(rates.clone());
    let min = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let max = rates.iter().copied().fold(0.0, f64::max);
    println!("CPU\t{name}\t{protocol}\t{mode}\t{:.3}\t{:.3}\t{:.3}", med/1e6, min/1e6, max/1e6);
}

fn run_cpu() {
    let prefix = b"203.0.113.77;example.test;1759954402;";
    macro_rules! decimal {
        ($ty:path, $name:expr) => {{
            bench_cpu::<$ty,_>($name, "haphash", "mask64", || <$ty>::from(DecimalMessage::new(prefix,0).unwrap()), SOLVE_TYPE_MASK, 0, u64::MAX);
            #[cfg(feature = "full")]
            bench_cpu::<$ty,_>($name, "haphash", "full68", || <$ty>::from(DecimalMessage::new(prefix,0).unwrap()), SOLVE_TYPE_LEADING_ZERO_BITS, 68, u64::MAX);
        }};
    }
    decimal!(pow_core::solver::safe::DecimalSolver, "safe");
    #[cfg(target_arch="x86_64")]
    decimal!(pow_core::solver::avx2::DecimalSolver, "avx2");
    #[cfg(target_arch="x86_64")]
    decimal!(pow_core::solver::sha_ni::DecimalSolver, "sha-ni");

    let mut bytes=[0u8;32];
    for (i,b) in bytes.iter_mut().enumerate(){*b=i as u8;}
    macro_rules! goaway {
        ($ty:path, $name:expr) => {{
            bench_cpu::<$ty,_>($name, "goaway", "mask64", || <$ty>::from(GoAwayMessage::new_bytes(&bytes,0)), SOLVE_TYPE_MASK, 0, u64::MAX);
            #[cfg(feature = "full")]
            bench_cpu::<$ty,_>($name, "goaway", "full65", || <$ty>::from(GoAwayMessage::new_bytes(&bytes,0)), SOLVE_TYPE_LEADING_ZERO_BITS, 65, u64::MAX);
        }};
    }
    goaway!(pow_core::solver::safe::GoAwaySolver, "safe");
    #[cfg(target_arch="x86_64")]
    goaway!(pow_core::solver::avx2::GoAwaySolver, "avx2");
    #[cfg(target_arch="x86_64")]
    goaway!(pow_core::solver::sha_ni::GoAwaySolver, "sha-ni");
}

#[cfg(feature="gpu")]
async fn bench_gpu_one(adapter: &str) -> anyhow::Result<()> {
    unsafe { env::set_var("WGPU_ADAPTER_NAME", adapter); }
    let mut gpu = pow_core::gpu::GpuContext::create().await?;
    gpu.warm_up().await?;
    let prefix=b"203.0.113.77;example.test;1759954402;";
    let mut rates=Vec::new();
    for i in 0..REPEATS+1 {
        let mut solver=pow_core::gpu::DecimalSolver::from(DecimalMessage::new(prefix,0).unwrap());
        solver.set_limit(GPU_LIMIT);
        let st=Instant::now();
        let result=solver.solve::<SOLVE_TYPE_MASK>(&mut gpu,0,u64::MAX).await?;
        black_box(&result);
        let ns=st.elapsed().as_nanos() as f64;
        assert_eq!(solver.get_attempted_nonces(),GPU_LIMIT);
        if i>0 {rates.push(GPU_LIMIT as f64*1e9/ns);}
    }
    println!("GPU\t{}\thaphash\tmask64\t{:.3}\t{:.3}\t{:.3}", gpu.adapter(), median(rates.clone())/1e6, rates.iter().copied().fold(f64::INFINITY,f64::min)/1e6, rates.iter().copied().fold(0.0,f64::max)/1e6);
    #[cfg(feature="full")]
    {
        let mut rates=Vec::new();
        for i in 0..REPEATS+1 {
            let mut solver=pow_core::gpu::DecimalSolver::from(DecimalMessage::new(prefix,0).unwrap());
            solver.set_limit(GPU_LIMIT);
            let st=Instant::now();
            let result=solver.solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(&mut gpu,68,u64::MAX).await?;
            black_box(&result);
            let ns=st.elapsed().as_nanos() as f64;
            assert_eq!(solver.get_attempted_nonces(),GPU_LIMIT);
            if i>0 {rates.push(GPU_LIMIT as f64*1e9/ns);}
        }
        println!("GPU\t{}\thaphash\tfull68\t{:.3}\t{:.3}\t{:.3}", gpu.adapter(), median(rates.clone())/1e6, rates.iter().copied().fold(f64::INFINITY,f64::min)/1e6, rates.iter().copied().fold(0.0,f64::max)/1e6);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args=env::args().skip(1);
    match args.next().as_deref() {
        Some("cpu")|None => run_cpu(),
        #[cfg(feature="gpu")]
        Some("gpu") => pollster::block_on(bench_gpu_one(&args.next().expect("adapter name")))?,
        Some(x) => panic!("unknown mode {x}"),
    }
    Ok(())
}
