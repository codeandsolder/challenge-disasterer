#[cfg(not(any(feature = "typed", feature = "erased")))]
compile_error!("select either typed, erased, or both dispatch features");

use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::{num::NonZeroU8, time::Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Protocol {
    Anubis,
    Haphash,
    Mcaptcha,
    Capjs,
    GoAway,
    Cerberus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Dispatch {
    #[cfg(feature = "typed")]
    Typed,
    #[cfg(feature = "erased")]
    Erased,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 5)]
    samples: u32,
    #[arg(long, default_value_t = 1_120_000)]
    limit: u64,
    /// Restrict the workload to selected protocol families; repeat as needed.
    #[arg(long, value_enum)]
    protocol: Vec<Protocol>,
    /// Static generic or service-style boxed dynamic dispatch; available modes depend on build features.
    #[arg(long, value_enum)]
    dispatch: Option<Dispatch>,
}

impl Args {
    fn includes(&self, protocol: Protocol) -> bool {
        self.protocol.is_empty() || self.protocol.contains(&protocol)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.limit > 0, "limit must be > 0");
        Ok(())
    }

    fn dispatch(&self) -> Dispatch {
        if let Some(dispatch) = self.dispatch {
            return dispatch;
        }
        #[cfg(feature = "typed")]
        {
            Dispatch::Typed
        }
        #[cfg(all(not(feature = "typed"), feature = "erased"))]
        {
            Dispatch::Erased
        }
    }
}

#[derive(Serialize)]
struct Row<'a> {
    schema: u8,
    baseline: &'a str,
    protocol: &'a str,
    variant: &'a str,
    implementation: &'a str,
    sample: u32,
    attempted_work: u64,
    solve_ns: u128,
    work_per_second: f64,
}

fn emit(
    baseline: &'static str,
    protocol: &'static str,
    variant: &'static str,
    implementation: &'static str,
    sample: u32,
    attempted: u64,
    ns: u128,
) {
    let row = Row {
        schema: 1,
        baseline,
        protocol,
        variant,
        implementation,
        sample,
        attempted_work: attempted,
        solve_ns: ns,
        work_per_second: attempted as f64 * 1e9 / ns as f64,
    };
    println!("{}", serde_json::to_string(&row).unwrap());
}

#[cfg(feature = "erased")]
trait ErasedCurrentSolver {
    fn set_limit_erased(&mut self, limit: u64);
    fn attempted_erased(&self) -> u64;
    fn solve_erased(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

#[cfg(feature = "erased")]
impl<T> ErasedCurrentSolver for T
where
    T: current::solver::Solver<Output = [u32; 8]> + current::solver::SolverDyn<Output = [u32; 8]>,
{
    fn set_limit_erased(&mut self, limit: u64) {
        current::solver::Solver::set_limit(self, limit);
    }

    fn attempted_erased(&self) -> u64 {
        current::solver::Solver::attempted_nonces(self)
    }

    fn solve_erased(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])> {
        current::solver::SolverDyn::solve_dyn(self, target, ty, mask)
    }
}

#[cfg(feature = "erased")]
#[inline(never)]
fn call_erased_solver(
    solver: &mut dyn ErasedCurrentSolver,
    target: u64,
    ty: u8,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    solver.solve_erased(target, ty, mask)
}

#[cfg(feature = "typed")]
macro_rules! timed_current_typed {
    ($protocol:expr, $variant:expr, $sample:expr, $limit:expr, $solver:expr, $ty:expr) => {{
        use current::solver::Solver as _;
        let mut solver = $solver;
        solver.set_limit($limit);
        let t = Instant::now();
        let found = solver.solve::<{ $ty }>(
            if $ty == current::solver::SOLVE_TYPE_GT {
                u64::MAX - 1
            } else if $ty == current::solver::SOLVE_TYPE_LT {
                1
            } else {
                0
            },
            u64::MAX,
        );
        let ns = t.elapsed().as_nanos();
        anyhow::ensure!(
            found.is_none(),
            "current {} {} unexpectedly found proof",
            $protocol,
            $variant
        );
        anyhow::ensure!(
            solver.attempted_nonces() == $limit,
            "current {} {} attempted {} != {}",
            $protocol,
            $variant,
            solver.attempted_nonces(),
            $limit
        );
        emit("current", $protocol, $variant, "auto", $sample, $limit, ns);
    }};
}
#[cfg(feature = "erased")]
macro_rules! timed_current_erased {
    ($protocol:expr, $variant:expr, $sample:expr, $limit:expr, $solver:expr, $ty:expr) => {{
        let mut solver: Box<dyn ErasedCurrentSolver> = Box::new($solver);
        solver.set_limit_erased($limit);
        let started = Instant::now();
        let found = call_erased_solver(
            solver.as_mut(),
            if $ty == current::solver::SOLVE_TYPE_GT {
                u64::MAX - 1
            } else if $ty == current::solver::SOLVE_TYPE_LT {
                1
            } else {
                0
            },
            $ty,
            u64::MAX,
        );
        let elapsed = started.elapsed().as_nanos();
        anyhow::ensure!(
            found.is_none(),
            "erased current {} {} unexpectedly found proof",
            $protocol,
            $variant
        );
        let attempted = solver.attempted_erased();
        anyhow::ensure!(
            attempted == $limit,
            "erased current {} {} attempted {} != {}",
            $protocol,
            $variant,
            attempted,
            $limit
        );
        emit(
            "current",
            $protocol,
            $variant,
            "auto-erased",
            $sample,
            attempted,
            elapsed,
        );
    }};
}

macro_rules! timed_current {
    ($dispatch:expr, $protocol:expr, $variant:expr, $sample:expr, $limit:expr, $solver:expr, $ty:expr) => {{
        match $dispatch {
            #[cfg(feature = "typed")]
            Dispatch::Typed => {
                timed_current_typed!($protocol, $variant, $sample, $limit, $solver, $ty)
            }
            #[cfg(feature = "erased")]
            Dispatch::Erased => {
                timed_current_erased!($protocol, $variant, $sample, $limit, $solver, $ty)
            }
        }
    }};
}

fn current_cases(args: &Args, sample: u32) -> anyhow::Result<()> {
    let limit = args.limit;
    if args.includes(Protocol::Anubis) {
        let mut prefix = vec![b'a'; 96];
        prefix[..4].copy_from_slice(&sample.to_le_bytes());
        timed_current!(
            args.dispatch(),
            "anubis",
            "decimal",
            sample,
            limit,
            current::DecimalSolver::from(
                current::message::DecimalMessage::new(&prefix, 0).unwrap()
            ),
            current::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::Haphash) {
        for (variant, haphash_prefix) in [
            (
                "haproxy",
                format!("203.0.113.{};example.test;1759954402;", sample % 255),
            ),
            ("debian-varnish", format!("102026194332.0x{sample:064x};")),
        ] {
            timed_current!(
                args.dispatch(),
                "haphash",
                variant,
                sample,
                limit,
                current::DecimalSolver::from(
                    current::message::DecimalMessage::new(haphash_prefix.as_bytes(), 0).unwrap()
                ),
                current::solver::SOLVE_TYPE_MASK
            );
        }
    }
    if args.includes(Protocol::Anubis) {
        let mut binary = vec![0x42; 48];
        binary[..4].copy_from_slice(&sample.to_le_bytes());
        *binary.last_mut().unwrap() = 0x80;
        timed_current!(
            args.dispatch(),
            "anubis",
            "wasm-sha256-1.28",
            sample,
            limit,
            current::BinarySolver::from(
                current::message::BinaryMessage::new(&binary, NonZeroU8::new(4).unwrap())
                    .expect("four-byte baseline nonce width"),
            ),
            current::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::Mcaptcha) {
        let mut mc = Vec::new();
        let _ = current::build_mcaptcha_prefix(
            &mut mc,
            "challenge-disasterer-matrix",
            &format!("matrix-salt-{sample:08x}"),
        );
        timed_current!(
            args.dispatch(),
            "mcaptcha",
            "sha256",
            sample,
            limit,
            current::DecimalSolver::from(current::message::DecimalMessage::new(&mc, 0).unwrap()),
            current::solver::SOLVE_TYPE_GT
        );
    }
    if args.includes(Protocol::Capjs) {
        let cap = [b'1'; 32];
        timed_current!(
            args.dispatch(),
            "capjs",
            "single-subchallenge",
            sample,
            limit,
            current::DecimalSolver::from(
                current::message::DecimalMessage::new_f64(&cap, 0)
                    .unwrap()
                    .0
            ),
            current::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::GoAway) {
        let mut go = [0u8; 32];
        for (i, b) in go.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(sample as u8);
        }
        timed_current!(
            args.dispatch(),
            "go-away",
            "js-pow-sha256",
            sample,
            limit,
            current::GoAwaySolver::from(current::message::GoAwayMessage::new_bytes(&go, 0)),
            current::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::Cerberus) {
        let salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        timed_current!(
            args.dispatch(),
            "cerberus",
            "binary-0.4.6+",
            sample,
            limit,
            current::CerberusSolver::from(current::message::CerberusMessage::Binary(
                current::message::CerberusBinaryMessage::new(salt.as_bytes(), 0)
            )),
            current::solver::SOLVE_TYPE_MASK
        );
        timed_current!(
            args.dispatch(),
            "cerberus",
            "decimal-pre-0.4.6",
            sample,
            limit,
            current::CerberusSolver::from(current::message::CerberusMessage::Decimal(
                current::message::CerberusDecimalMessage::new(salt.as_bytes(), 0).unwrap()
            )),
            current::solver::SOLVE_TYPE_MASK
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.validate()?;
    for sample in 0..args.samples {
        current_cases(&args, sample)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_filter_and_dispatch_variant_are_exact() -> anyhow::Result<()> {
        let args = Args::try_parse_from(["dispatch-bench", "--protocol", "go-away"])?;
        assert!(args.includes(Protocol::GoAway));
        assert!(!args.includes(Protocol::Cerberus));
        args.validate()?;
        assert!(
            Args::try_parse_from(["dispatch-bench", "--limit", "0"])?
                .validate()
                .is_err()
        );
        #[cfg(feature = "typed")]
        {
            assert!(matches!(args.dispatch(), Dispatch::Typed));
            assert!(matches!(
                Args::try_parse_from(["dispatch-bench", "--dispatch", "typed"])?.dispatch(),
                Dispatch::Typed
            ));
        }
        #[cfg(all(not(feature = "typed"), feature = "erased"))]
        {
            assert!(matches!(args.dispatch(), Dispatch::Erased));
            assert!(Args::try_parse_from(["dispatch-bench", "--dispatch", "typed"]).is_err());
        }
        #[cfg(feature = "erased")]
        assert!(matches!(
            Args::try_parse_from(["dispatch-bench", "--dispatch", "erased"])?.dispatch(),
            Dispatch::Erased
        ));
        Ok(())
    }
}
