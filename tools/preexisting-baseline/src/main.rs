use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::{num::NonZeroU8, time::Instant};

const LEGACY_CERBERUS_DECIMAL_WORK_QUANTUM: u64 = 160_000;

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
    Typed,
    Erased,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 5)]
    samples: u32,
    #[arg(long, default_value_t = 1_120_000)]
    limit: u64,
    /// Restrict the paired comparison to selected protocol families; repeat as needed.
    #[arg(long, value_enum)]
    protocol: Vec<Protocol>,
    /// Current solver invocation: static generic or service-style erased dynamic.
    #[arg(long, value_enum, default_value_t = Dispatch::Typed)]
    dispatch: Dispatch,
}

impl Args {
    fn includes(&self, protocol: Protocol) -> bool {
        self.protocol.is_empty() || self.protocol.contains(&protocol)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.limit > 0, "limit must be > 0");
        if self.includes(Protocol::Cerberus) {
            anyhow::ensure!(
                self.limit
                    .is_multiple_of(LEGACY_CERBERUS_DECIMAL_WORK_QUANTUM),
                "limit must be a multiple of {LEGACY_CERBERUS_DECIMAL_WORK_QUANTUM} so historical Cerberus decimal executes exactly the requested work"
            );
        }
        Ok(())
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

trait ErasedCurrentSolver {
    fn set_limit_erased(&mut self, limit: u64);
    fn attempted_erased(&self) -> u64;
    fn solve_erased(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

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
macro_rules! timed_current_erased {
    ($protocol:expr, $variant:expr, $sample:expr, $limit:expr, $solver:expr, $ty:expr) => {{
        let mut solver: Box<dyn ErasedCurrentSolver> = Box::new($solver);
        solver.set_limit_erased($limit);
        let started = Instant::now();
        let found = solver.solve_erased(
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
            Dispatch::Typed => {
                timed_current_typed!($protocol, $variant, $sample, $limit, $solver, $ty)
            }
            Dispatch::Erased => {
                timed_current_erased!($protocol, $variant, $sample, $limit, $solver, $ty)
            }
        }
    }};
}

macro_rules! timed_legacy {
    ($protocol:expr, $variant:expr, $sample:expr, $limit:expr, $solver:expr, $ty:expr) => {{
        use legacy::solver::Solver as _;
        let mut solver = $solver;
        solver.set_limit($limit);
        let t = Instant::now();
        let found = solver.solve::<{ $ty }>(
            if $ty == legacy::solver::SOLVE_TYPE_GT {
                u64::MAX - 1
            } else if $ty == legacy::solver::SOLVE_TYPE_LT {
                1
            } else {
                0
            },
            u64::MAX,
        );
        let ns = t.elapsed().as_nanos();
        anyhow::ensure!(
            found.is_none(),
            "legacy {} {} unexpectedly found proof",
            $protocol,
            $variant
        );
        anyhow::ensure!(
            solver.get_attempted_nonces() == $limit,
            "legacy {} {} attempted {} != {}",
            $protocol,
            $variant,
            solver.get_attempted_nonces(),
            $limit
        );
        emit(
            "eternal-flame-AD/pow-buster@d359adb",
            $protocol,
            $variant,
            "auto",
            $sample,
            $limit,
            ns,
        );
    }};
}

fn current_cases(args: &Args, sample: u32) -> anyhow::Result<()> {
    let limit = args.limit;
    if args.includes(Protocol::Anubis) {
        let mut prefix = vec![b'a'; 96];
        prefix[..4].copy_from_slice(&sample.to_le_bytes());
        timed_current!(
            args.dispatch,
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
                args.dispatch,
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
            args.dispatch,
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
            args.dispatch,
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
            args.dispatch,
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
            args.dispatch,
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
            args.dispatch,
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
            args.dispatch,
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

fn legacy_cases(args: &Args, sample: u32) -> anyhow::Result<()> {
    let limit = args.limit;
    if args.includes(Protocol::Anubis) {
        let mut prefix = vec![b'a'; 96];
        prefix[..4].copy_from_slice(&sample.to_le_bytes());
        timed_legacy!(
            "anubis",
            "decimal",
            sample,
            limit,
            legacy::DecimalSolver::from(legacy::message::DecimalMessage::new(&prefix, 0).unwrap()),
            legacy::solver::SOLVE_TYPE_MASK
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
            timed_legacy!(
                "haphash",
                variant,
                sample,
                limit,
                legacy::DecimalSolver::from(
                    legacy::message::DecimalMessage::new(haphash_prefix.as_bytes(), 0).unwrap()
                ),
                legacy::solver::SOLVE_TYPE_MASK
            );
        }
    }
    if args.includes(Protocol::Anubis) {
        let mut binary = vec![0x42; 48];
        binary[..4].copy_from_slice(&sample.to_le_bytes());
        *binary.last_mut().unwrap() = 0x80;
        timed_legacy!(
            "anubis",
            "wasm-sha256-1.28",
            sample,
            limit,
            legacy::BinarySolver::from(legacy::message::BinaryMessage::new(
                &binary,
                NonZeroU8::new(4).unwrap()
            )),
            legacy::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::Mcaptcha) {
        let mut mc = Vec::new();
        legacy::build_mcaptcha_prefix(
            &mut mc,
            "challenge-disasterer-matrix",
            &format!("matrix-salt-{sample:08x}"),
        );
        timed_legacy!(
            "mcaptcha",
            "sha256",
            sample,
            limit,
            legacy::DecimalSolver::from(legacy::message::DecimalMessage::new(&mc, 0).unwrap()),
            legacy::solver::SOLVE_TYPE_GT
        );
    }
    if args.includes(Protocol::Capjs) {
        let cap = [b'1'; 32];
        timed_legacy!(
            "capjs",
            "single-subchallenge",
            sample,
            limit,
            legacy::DecimalSolver::from(
                legacy::message::DecimalMessage::new_f64(&cap, 0).unwrap().0
            ),
            legacy::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::GoAway) {
        let mut go = [0u8; 32];
        for (i, b) in go.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(sample as u8);
        }
        timed_legacy!(
            "go-away",
            "js-pow-sha256",
            sample,
            limit,
            legacy::GoAwaySolver::from(legacy::message::GoAwayMessage::new_bytes(&go, 0)),
            legacy::solver::SOLVE_TYPE_MASK
        );
    }
    if args.includes(Protocol::Cerberus) {
        let salt = format!("matrix-{sample:08x}|3959614364|1759954402|signature|");
        timed_legacy!(
            "cerberus",
            "binary-0.4.6+",
            sample,
            limit,
            legacy::CerberusSolver::from(legacy::message::CerberusMessage::Binary(
                legacy::message::CerberusBinaryMessage::new(salt.as_bytes(), 0)
            )),
            legacy::solver::SOLVE_TYPE_MASK
        );
        timed_legacy!(
            "cerberus",
            "decimal-pre-0.4.6",
            sample,
            limit,
            legacy::CerberusSolver::from(legacy::message::CerberusMessage::Decimal(
                legacy::message::CerberusDecimalMessage::new(salt.as_bytes(), 0).unwrap()
            )),
            legacy::solver::SOLVE_TYPE_MASK
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.validate()?;
    for sample in 0..args.samples {
        // Alternate order by sample to reduce thermal/order bias.
        if sample % 2 == 0 {
            current_cases(&args, sample)?;
            legacy_cases(&args, sample)?;
        } else {
            legacy_cases(&args, sample)?;
            current_cases(&args, sample)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_default_to_all_and_accept_repetition() -> anyhow::Result<()> {
        let defaults = Args::try_parse_from(["baseline"])?;
        for protocol in [
            Protocol::Anubis,
            Protocol::Haphash,
            Protocol::Mcaptcha,
            Protocol::Capjs,
            Protocol::GoAway,
            Protocol::Cerberus,
        ] {
            assert!(defaults.includes(protocol));
        }
        let selected = Args::try_parse_from([
            "baseline",
            "--protocol",
            "go-away",
            "--protocol",
            "capjs",
            "--limit",
            "17",
        ])?;
        assert!(selected.includes(Protocol::GoAway));
        assert!(selected.includes(Protocol::Capjs));
        assert!(!selected.includes(Protocol::Anubis));
        assert!(!selected.includes(Protocol::Cerberus));
        selected.validate()?;
        let erased =
            Args::try_parse_from(["baseline", "--protocol", "go-away", "--dispatch", "erased"])?;
        assert!(matches!(erased.dispatch, Dispatch::Erased));
        assert!(matches!(defaults.dispatch, Dispatch::Typed));
        assert!(Args::try_parse_from(["baseline", "--dispatch", "invalid"]).is_err());
        assert!(
            Args::try_parse_from(["baseline", "--protocol", "cerberus", "--limit", "17",])?
                .validate()
                .is_err()
        );
        Ok(())
    }
}
