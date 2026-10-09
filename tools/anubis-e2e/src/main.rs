use std::{
    fmt::Write as _,
    fs::{self, File},
    io::{ErrorKind, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow, bail, ensure};
use clap::Parser;
use reqwest::{
    Url,
    blocking::{Client, ClientBuilder},
    cookie::Jar,
    redirect::Policy,
};
use serde::Deserialize;
use serde_json::json;
use tempfile::TempDir;

const BACKEND_MARKER: &str = "challenge-disasterer-anubis-e2e-ok";
const READY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
struct CaseSpec {
    algorithm: String,
    solver: String,
    difficulty: u32,
    implementation: Option<String>,
}

impl FromStr for CaseSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut fields = value.split(':');
        let algorithm = fields.next().context("missing Anubis algorithm")?;
        let solver = fields.next().context("missing solver name")?;
        let difficulty = fields
            .next()
            .context("missing difficulty")?
            .parse::<u32>()
            .context("difficulty must be an integer")?;
        let implementation = fields.next().map(str::to_owned);
        ensure!(
            fields.next().is_none(),
            "case must be algorithm:solver:difficulty[:implementation]"
        );
        validate_token("algorithm", algorithm)?;
        validate_token("solver", solver)?;
        if let Some(ref implementation) = implementation {
            validate_token("implementation", implementation)?;
        }
        ensure!(difficulty > 0, "difficulty must be greater than zero");
        Ok(Self {
            algorithm: algorithm.to_owned(),
            solver: solver.to_owned(),
            difficulty,
            implementation,
        })
    }
}

fn validate_token(label: &str, value: &str) -> anyhow::Result<()> {
    ensure!(!value.is_empty(), "{label} cannot be empty");
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
        "{label} contains unsupported characters: {value}"
    );
    Ok(())
}

#[derive(Parser, Debug)]
#[command(about = "Exercise challenge-disasterer against a real Anubis server")]
#[allow(clippy::struct_excessive_bools, reason = "independent CLI feature switches are clearer as booleans")]
struct Args {
    /// Current upstream Anubis binary, built from the source revision under test.
    #[arg(long)]
    anubis_bin: PathBuf,

    /// challenge-disasterer binary under test.
    #[arg(long)]
    solver_bin: PathBuf,

    /// Anubis algorithm, solver, difficulty, and optional required implementation.
    /// Repeat to test multiple solver/protocol pairs.
    #[arg(long = "case", value_name = "ALGORITHM:SOLVER:DIFFICULTY[:IMPLEMENTATION]")]
    cases: Vec<CaseSpec>,

    /// Enable challenge-disasterer's GPU backend. CI defaults to CPU-only.
    #[arg(long, default_value_t = false)]
    solver_gpu: bool,

    /// Require HashX GPU fuzz/benchmark cases to use this concrete implementation.
    #[arg(long)]
    expect_hashx_gpu_implementation: Option<String>,

    /// Extra argument forwarded verbatim to challenge-disasterer.
    #[arg(long = "solver-arg")]
    solver_forwarded: Vec<String>,

    /// Run a fresh-challenge correctness fuzz matrix before explicit cases.
    #[arg(long, default_value_t = false)]
    fuzz: bool,

    /// Fresh random challenges per fuzz matrix entry.
    #[arg(long, default_value_t = 4)]
    fuzz_repeats: usize,

    /// Benchmark representative solver/backend combinations after correctness fuzzing.
    #[arg(long, default_value_t = false)]
    benchmark: bool,

    /// Fresh random challenges per benchmark entry.
    #[arg(long, default_value_t = 12)]
    benchmark_samples: usize,

    /// Submit a deliberately corrupted proof before each fuzz proof and require rejection.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    negative_checks: bool,

    /// Benchmark the stock Anubis HEAD browser implementations in Chromium too.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    browser_reference: bool,

    /// Chromium executable used for stock-browser benchmarks.
    #[arg(long, default_value = "chromium")]
    chromium_bin: PathBuf,

    /// Node/CDP driver for stock-browser benchmarks.
    #[arg(long, default_value = "tools/anubis-e2e/browser-bench.mjs")]
    browser_script: PathBuf,

    /// Preserve logs and generated policy after success as well as failure.
    #[arg(long, default_value_t = false)]
    keep_temp: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChallengeRecord {
    id: String,
    random_data: String,
}

#[derive(Debug, Deserialize)]
struct ChallengeRules {
    algorithm: String,
    difficulty: u32,
}

#[derive(Debug, Deserialize)]
struct ChallengeEnvelope {
    challenge: ChallengeRecord,
    rules: ChallengeRules,
}

#[derive(Debug, Deserialize)]
struct SolverAnswer {
    nonce: String,
    hash: String,
}

#[derive(Debug, Deserialize)]
struct SolverResponse {
    solver: String,
    backend: String,
    implementation: Option<String>,
    answer: SolverAnswer,
    solve_ms: f64,
}

#[derive(Debug)]
struct TrialResult {
    wall_ms: f64,
    solve_ms: f64,
    backend: String,
    implementation: Option<String>,
}

struct ManagedChild {
    name: &'static str,
    child: Child,
    log_path: PathBuf,
}

impl ManagedChild {
    fn spawn(name: &'static str, command: &mut Command, log_path: PathBuf) -> anyhow::Result<Self> {
        let stdout = File::create(&log_path)
            .with_context(|| format!("create {} log {}", name, log_path.display()))?;
        let stderr = stdout.try_clone().context("clone log handle")?;
        command
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        let child = command
            .spawn()
            .with_context(|| format!("spawn {name}: {command:?}"))?;
        Ok(Self {
            name,
            child,
            log_path,
        })
    }

    fn ensure_running(&mut self) -> anyhow::Result<()> {
        if let Some(status) = self.child.try_wait().context("poll child")? {
            let log = fs::read_to_string(&self.log_path).unwrap_or_default();
            bail!(
                "{} exited early with {status}\n--- {} log ---\n{log}",
                self.name,
                self.name
            );
        }
        Ok(())
    }

    fn log_tail(&self) -> String {
        let Ok(log) = fs::read_to_string(&self.log_path) else {
            return String::new();
        };
        let lines = log.lines().collect::<Vec<_>>();
        let start = lines.len().saturating_sub(80);
        lines[start..].join("\n")
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct BackendServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl BackendServer {
    fn start() -> anyhow::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind test backend")?;
        listener
            .set_nonblocking(true)
            .context("make backend nonblocking")?;
        let addr = listener.local_addr().context("backend local address")?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let join = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => respond_backend(&mut stream),
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Self {
            addr,
            stop,
            join: Some(join),
        })
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for BackendServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.addr);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn respond_backend(stream: &mut TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut request = [0_u8; 8192];
    let _ = stream.read(&mut request);
    let body = format!("{BACKEND_MARKER}\n");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn reserve_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("reserve TCP port")?;
    Ok(listener
        .local_addr()
        .context("reserved port address")?
        .port())
}

fn write_policy(path: &Path, cases: &[CaseSpec]) -> anyhow::Result<()> {
    let mut policy = String::from("bots:\n");
    for (index, case) in cases.iter().enumerate() {
        writeln!(
            policy,
            "  - name: e2e-{index}\n    user_agent_regex: '^challenge-disasterer-e2e/{index}/{}$'\n    action: CHALLENGE\n    challenge:\n      algorithm: {}\n      difficulty: {}",
            case.algorithm, case.algorithm, case.difficulty
        )
        .expect("writing to String cannot fail");
    }
    policy.push_str("\nstatus_codes:\n  CHALLENGE: 200\n  DENY: 403\n");
    fs::write(path, policy).with_context(|| format!("write policy {}", path.display()))
}

fn build_client(user_agent: &str) -> anyhow::Result<Client> {
    let cookies = Arc::new(Jar::default());
    ClientBuilder::new()
        .user_agent(user_agent)
        .cookie_provider(cookies)
        .redirect(Policy::limited(10))
        .timeout(Duration::from_secs(30))
        .build()
        .context("build HTTP client")
}

fn wait_ready(url: &str, process: &mut ManagedChild) -> anyhow::Result<()> {
    let client = ClientBuilder::new()
        .redirect(Policy::none())
        .timeout(Duration::from_secs(1))
        .build()
        .context("build readiness client")?;
    let started = Instant::now();
    loop {
        process.ensure_running()?;
        if client.get(url).send().is_ok() {
            return Ok(());
        }
        if started.elapsed() >= READY_TIMEOUT {
            bail!(
                "{} did not become ready at {url}\n--- {} log ---\n{}",
                process.name,
                process.name,
                process.log_tail()
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn extract_json_script<T: for<'de> Deserialize<'de>>(html: &str, id: &str) -> anyhow::Result<T> {
    let id_marker = format!("id=\"{id}\"");
    let id_pos = html
        .find(&id_marker)
        .with_context(|| format!("HTML did not contain JSON script {id}"))?;
    let script_start = html[..id_pos]
        .rfind("<script")
        .with_context(|| format!("could not find opening script tag for {id}"))?;
    let content_start = html[script_start..]
        .find('>')
        .map(|offset| script_start + offset + 1)
        .context("unterminated JSON script start tag")?;
    let content_end = html[content_start..]
        .find("</script>")
        .map(|offset| content_start + offset)
        .context("unterminated JSON script")?;
    serde_json::from_str(&html[content_start..content_end])
        .with_context(|| format!("decode JSON script {id}"))
}

#[allow(clippy::too_many_lines, reason = "keeps one complete protocol round trip together")]
fn solve_case(
    index: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
    negative_check: bool,
) -> anyhow::Result<TrialResult> {
    let wall_started = Instant::now();
    let user_agent = format!("challenge-disasterer-e2e/{index}/{}", case.algorithm);
    let client = build_client(&user_agent)?;

    let challenge_response = client
        .get(format!("{anubis_base}/"))
        .send()
        .context("request challenge page")?;
    ensure!(
        challenge_response.status().is_success(),
        "Anubis challenge request failed with {}",
        challenge_response.status()
    );
    let challenge_html = challenge_response.text().context("read challenge page")?;
    let envelope: ChallengeEnvelope = extract_json_script(&challenge_html, "anubis_challenge")?;
    ensure!(
        envelope.rules.algorithm == case.algorithm,
        "Anubis issued algorithm {}, expected {}",
        envelope.rules.algorithm,
        case.algorithm
    );
    ensure!(
        envelope.rules.difficulty == case.difficulty,
        "Anubis issued difficulty {}, expected {}",
        envelope.rules.difficulty,
        case.difficulty
    );

    let solver_response = Client::new()
        .post(format!("{solver_base}/solve"))
        .json(&json!({
            "solver": case.solver,
            "challenge": {
                "data": envelope.challenge.random_data,
                "difficulty": envelope.rules.difficulty,
            }
        }))
        .send()
        .context("submit challenge to solver")?;
    let solver_status = solver_response.status();
    let solver_body = solver_response.text().context("read solver response")?;
    ensure!(
        solver_status.is_success(),
        "solver returned {solver_status}: {solver_body}"
    );
    let solved: SolverResponse =
        serde_json::from_str(&solver_body).context("decode solver response")?;
    ensure!(
        solved.solver == case.solver,
        "solver response identified itself as {}, expected {}",
        solved.solver,
        case.solver
    );
    if let Some(expected) = case.implementation.as_deref() {
        ensure!(
            solved.implementation.as_deref() == Some(expected),
            "solver implementation mismatch: expected {expected}, got {:?}",
            solved.implementation
        );
    }

    let pass_url = Url::parse(&format!(
        "{anubis_base}/.within.website/x/cmd/anubis/api/pass-challenge"
    ))
    .context("construct pass-challenge URL")?;
    if negative_check {
        let mut corrupted = solved.answer.hash.clone().into_bytes();
        if let Some(first) = corrupted.first_mut() {
            *first = if *first == b'0' { b'1' } else { b'0' };
        }
        let corrupted = String::from_utf8(corrupted).context("corrupt proof hash")?;
        let rejected = client
            .get(pass_url.clone())
            .query(&[
                ("id", envelope.challenge.id.as_str()),
                ("response", corrupted.as_str()),
                ("nonce", solved.answer.nonce.as_str()),
                ("redir", "/"),
                ("elapsedTime", "1"),
            ])
            .send()
            .context("submit deliberately corrupted proof")?;
        let rejected_body = rejected.text().context("read corrupted-proof response")?;
        ensure!(
            !rejected_body.contains(BACKEND_MARKER),
            "Anubis accepted a deliberately corrupted proof"
        );
    }

    let accepted = client
        .get(pass_url)
        .query(&[
            ("id", envelope.challenge.id.as_str()),
            ("response", solved.answer.hash.as_str()),
            ("nonce", solved.answer.nonce.as_str()),
            ("redir", "/"),
            ("elapsedTime", "1"),
        ])
        .send()
        .context("submit proof to Anubis")?;
    let status = accepted.status();
    let final_url = accepted.url().clone();
    let body = accepted.text().context("read Anubis acceptance response")?;
    ensure!(
        status.is_success() && body.contains(BACKEND_MARKER),
        "Anubis did not accept proof (status={status}, url={final_url}); body starts with {:?}",
        body.chars().take(300).collect::<String>()
    );

    let wall_ms = wall_started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "PASS algorithm={} solver={} backend={} implementation={} difficulty={} nonce={} solver_ms={:.3} wall_ms={:.3} final_url={}",
        case.algorithm,
        case.solver,
        solved.backend,
        solved.implementation.as_deref().unwrap_or("-"),
        case.difficulty,
        solved.answer.nonce,
        solved.solve_ms,
        wall_ms,
        final_url
    );
    Ok(TrialResult {
        wall_ms,
        solve_ms: solved.solve_ms,
        backend: solved.backend,
        implementation: solved.implementation,
    })
}

fn expected_hashes(case: &CaseSpec) -> f64 {
    match case.algorithm.as_str() {
        "fast" | "slow" => 16_f64.powi(i32::try_from(case.difficulty).unwrap_or(i32::MAX)),
        "hashx" | "argon2id" | "sha256" => 2_f64.powi(i32::try_from(case.difficulty).unwrap_or(i32::MAX)),
        _ => f64::NAN,
    }
}

fn fuzz_matrix(gpu: bool, implementation: Option<&str>) -> anyhow::Result<Vec<CaseSpec>> {
    let mut specs = Vec::new();
    for difficulty in [1_u32, 2, 3, 4, 5] {
        specs.push(format!("fast:anubis-cpu:{difficulty}").parse()?);
    }
    for difficulty in [2_u32, 4] {
        specs.push(format!("fast:anubis:{difficulty}").parse()?);
    }
    for difficulty in [1_u32, 3] {
        specs.push(format!("slow:anubis-cpu:{difficulty}").parse()?);
    }
    for difficulty in [1_u32, 2, 4, 8, 12, 16] {
        specs.push(format!("hashx:anubis-hashx-jit:{difficulty}").parse()?);
    }
    if gpu {
        for difficulty in [1_u32, 2, 3, 4, 5] {
            specs.push(format!("fast:anubis-gpu:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 3] {
            specs.push(format!("slow:anubis-gpu:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 2, 4, 8, 12, 16, 20] {
            let suffix = implementation.map_or_else(String::new, |value| format!(":{value}"));
            specs.push(format!("hashx:anubis-hashx:{difficulty}{suffix}").parse()?);
        }
    }
    Ok(specs)
}

fn benchmark_matrix(gpu: bool, implementation: Option<&str>) -> anyhow::Result<Vec<CaseSpec>> {
    let mut specs = vec![
        "fast:anubis-cpu:5".parse()?,
        "hashx:anubis-hashx-jit:18".parse()?,
    ];
    if gpu {
        specs.push("fast:anubis-gpu:5".parse()?);
        let suffix = implementation.map_or_else(String::new, |value| format!(":{value}"));
        specs.push(format!("hashx:anubis-hashx:18{suffix}").parse()?);
    }
    Ok(specs)
}

fn browser_benchmark_matrix() -> anyhow::Result<Vec<CaseSpec>> {
    Ok(vec![
        "fast:browser:5".parse()?,
        "hashx:browser:18".parse()?,
        "sha256:browser:18".parse()?,
        "argon2id:browser:4".parse()?,
    ])
}

fn print_benchmark(case: &CaseSpec, trials: &[TrialResult]) -> anyhow::Result<()> {
    let wall_ms: f64 = trials.iter().map(|trial| trial.wall_ms).sum();
    let solve_ms: f64 = trials.iter().map(|trial| trial.solve_ms).sum();
    let sample_count = u32::try_from(trials.len()).expect("benchmark sample count fits u32");
    let sample_count = f64::from(sample_count);
    let hashes_per_s = expected_hashes(case) * sample_count * 1000.0
        / solve_ms.max(f64::MIN_POSITIVE);
    let backend = trials.first().map_or("unknown", |trial| trial.backend.as_str());
    let implementation = trials
        .first()
        .and_then(|trial| trial.implementation.as_deref());
    ensure!(
        trials.iter().all(|trial| {
            trial.backend == backend && trial.implementation.as_deref() == implementation
        }),
        "benchmark mixed backend/implementation results"
    );
    println!(
        "BENCH algorithm={} solver={} backend={} implementation={} difficulty={} n={} avg_answer_ms={:.3} effective_h_s={:.3} effective_mh_s={:.3} mean_solver_ms={:.3}",
        case.algorithm,
        case.solver,
        backend,
        implementation.unwrap_or("-"),
        case.difficulty,
        trials.len(),
        wall_ms / sample_count,
        hashes_per_s,
        hashes_per_s / 1e6,
        solve_ms / sample_count,
    );
    Ok(())
}

#[allow(clippy::too_many_lines, reason = "top-level test orchestration is intentionally linear")]
fn run(args: Args) -> anyhow::Result<Option<TempDir>> {
    ensure!(
        args.anubis_bin.is_file(),
        "Anubis binary not found: {}",
        args.anubis_bin.display()
    );
    ensure!(
        args.solver_bin.is_file(),
        "solver binary not found: {}",
        args.solver_bin.display()
    );

    if let Some(ref implementation) = args.expect_hashx_gpu_implementation {
        validate_token("implementation", implementation)?;
        ensure!(
            args.solver_gpu,
            "--expect-hashx-gpu-implementation requires --solver-gpu"
        );
    }
    let mut cases = args.cases;
    if args.fuzz {
        cases.extend(fuzz_matrix(
            args.solver_gpu,
            args.expect_hashx_gpu_implementation.as_deref(),
        )?);
    }
    let benchmark_cases = if args.benchmark {
        benchmark_matrix(
            args.solver_gpu,
            args.expect_hashx_gpu_implementation.as_deref(),
        )?
    } else {
        Vec::new()
    };
    let browser_cases = if args.benchmark && args.browser_reference {
        browser_benchmark_matrix()?
    } else {
        Vec::new()
    };
    cases.extend(benchmark_cases.iter().cloned());
    cases.extend(browser_cases.iter().cloned());
    if cases.is_empty() {
        cases.push("fast:anubis:2".parse()?);
    }

    let temp = tempfile::tempdir().context("create e2e temp directory")?;
    let policy_path = temp.path().join("anubis.yaml");
    write_policy(&policy_path, &cases)?;

    let backend = BackendServer::start()?;
    let solver_port = reserve_port()?;
    let anubis_port = reserve_port()?;
    let metrics_port = reserve_port()?;
    let solver_base = format!("http://127.0.0.1:{solver_port}");
    let anubis_base = format!("http://127.0.0.1:{anubis_port}");

    let mut solver_command = Command::new(&args.solver_bin);
    solver_command
        .arg("--listen")
        .arg(format!("127.0.0.1:{solver_port}"))
        .arg("--gpu")
        .arg(args.solver_gpu.to_string())
        .arg("--cpu-workers")
        .arg(if args.benchmark { "1" } else { "2" })
        .args(&args.solver_forwarded);
    let mut solver = ManagedChild::spawn(
        "challenge-disasterer",
        &mut solver_command,
        temp.path().join("solver.log"),
    )?;
    wait_ready(&format!("{solver_base}/stats"), &mut solver)?;

    let mut anubis_command = Command::new(&args.anubis_bin);
    anubis_command
        .arg(format!("--bind=127.0.0.1:{anubis_port}"))
        .arg(format!("--metrics-bind=127.0.0.1:{metrics_port}"))
        .arg(format!("--policy-fname={}", policy_path.display()))
        .arg(format!("--target={}", backend.url()))
        .arg("--use-remote-address")
        .arg("--cookie-secure=false")
        .arg("--cookie-partitioned=false")
        .arg("--cookie-same-site=Lax")
        .arg("--slog-level=DEBUG");
    let mut anubis = ManagedChild::spawn(
        "anubis",
        &mut anubis_command,
        temp.path().join("anubis.log"),
    )?;
    wait_ready(
        &format!("http://127.0.0.1:{metrics_port}/healthz"),
        &mut anubis,
    )?;
    wait_ready(&format!("{anubis_base}/"), &mut anubis)?;

    let benchmark_start = cases
        .len()
        .saturating_sub(benchmark_cases.len() + browser_cases.len());
    let mut fuzz_trials = 0_usize;
    for (index, case) in cases[..benchmark_start].iter().enumerate() {
        let repeats = if args.fuzz { args.fuzz_repeats } else { 1 };
        for _ in 0..repeats {
            if let Err(error) = solve_case(
                index,
                case,
                &anubis_base,
                &solver_base,
                args.fuzz && args.negative_checks,
            ) {
                eprintln!("\n--- challenge-disasterer log ---\n{}", solver.log_tail());
                eprintln!("\n--- Anubis log ---\n{}", anubis.log_tail());
                return Err(error).with_context(|| {
                    format!(
                        "end-to-end case failed: {}:{}:{}{}",
                        case.algorithm,
                        case.solver,
                        case.difficulty,
                        case.implementation
                            .as_deref()
                            .map_or_else(String::new, |value| format!(":{value}"))
                    )
                });
            }
            fuzz_trials += 1;
        }
    }

    for (offset, case) in benchmark_cases.iter().enumerate() {
        let index = benchmark_start + offset;
        let mut trials = Vec::with_capacity(args.benchmark_samples);
        for _ in 0..args.benchmark_samples {
            let trial = solve_case(index, case, &anubis_base, &solver_base, false)
                .with_context(|| format!("benchmark case failed: {}:{}:{}", case.algorithm, case.solver, case.difficulty))?;
            trials.push(trial);
        }
        print_benchmark(case, &trials)?;
    }

    if !browser_cases.is_empty() {
        let browser_base = benchmark_start + benchmark_cases.len();
        let specs = browser_cases
            .iter()
            .enumerate()
            .map(|(offset, case)| {
                let samples = if case.algorithm == "argon2id" {
                    args.benchmark_samples.min(4)
                } else {
                    args.benchmark_samples
                };
                json!({
                    "index": browser_base + offset,
                    "algorithm": case.algorithm,
                    "difficulty": case.difficulty,
                    "samples": samples,
                })
            })
            .collect::<Vec<_>>();
        let output = Command::new("node")
            .arg(&args.browser_script)
            .arg("--chromium")
            .arg(&args.chromium_bin)
            .arg("--url")
            .arg(&anubis_base)
            .arg("--cases")
            .arg(serde_json::to_string(&specs).context("encode browser benchmark matrix")?)
            .arg("--timeoutMs")
            .arg("60000")
            .output()
            .context("run stock Anubis browser benchmark")?;
        print!("{}", String::from_utf8_lossy(&output.stdout));
        if !output.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            bail!("stock Anubis browser benchmark failed with {}", output.status);
        }
    }

    println!("PASS {fuzz_trials} live-Anubis correctness trial(s)");
    if args.keep_temp {
        let kept = temp.keep();
        println!("kept e2e artifacts at {}", kept.display());
        Ok(None)
    } else {
        Ok(Some(temp))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _temp = run(args).map_err(|error| anyhow!("{error:#}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_case_spec() {
        let case: CaseSpec = "hashx:anubis-hashx:16".parse().expect("valid case");
        assert_eq!(case.algorithm, "hashx");
        assert_eq!(case.solver, "anubis-hashx");
        assert_eq!(case.difficulty, 16);
        assert_eq!(case.implementation, None);

        let concrete: CaseSpec = "hashx:anubis-hashx:18:cuda-sm75-sass"
            .parse()
            .expect("valid implementation-qualified case");
        assert_eq!(concrete.difficulty, 18);
        assert_eq!(concrete.implementation.as_deref(), Some("cuda-sm75-sass"));
    }

    #[test]
    fn gpu_matrices_propagate_required_implementation() -> anyhow::Result<()> {
        let expected = "vulkan-extmul-persistent";
        let fuzz = fuzz_matrix(true, Some(expected))?;
        let bench = benchmark_matrix(true, Some(expected))?;
        assert!(fuzz.iter().filter(|case| case.solver == "anubis-hashx").all(|case| case.implementation.as_deref() == Some(expected)));
        assert!(bench.iter().filter(|case| case.solver == "anubis-hashx").all(|case| case.implementation.as_deref() == Some(expected)));
        Ok(())
    }

    #[test]
    fn extracts_templ_json_script() {
        let html = r#"<html><script type="application/json" id="anubis_challenge">{"challenge":{"id":"abc","randomData":"deadbeef"},"rules":{"algorithm":"fast","difficulty":2}}</script></html>"#;
        let challenge: ChallengeEnvelope =
            extract_json_script(html, "anubis_challenge").expect("extract challenge");
        assert_eq!(challenge.challenge.id, "abc");
        assert_eq!(challenge.challenge.random_data, "deadbeef");
        assert_eq!(challenge.rules.algorithm, "fast");
        assert_eq!(challenge.rules.difficulty, 2);
    }
}
