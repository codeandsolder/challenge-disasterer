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
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, anyhow, bail, ensure};
use clap::Parser;
use reqwest::{
    Url,
    blocking::{Client, ClientBuilder},
    cookie::Jar,
    redirect::Policy,
};
use serde::{Deserialize, Serialize};
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
    force_implementation: Option<String>,
    batch_size_hint: usize,
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
            force_implementation: None,
            batch_size_hint: 1,
        })
    }
}

#[derive(Debug, Clone)]
struct ExternalSolverSpec {
    name: String,
    url: String,
}

impl FromStr for ExternalSolverSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (name, url) = value
            .split_once('=')
            .context("external solver must be NAME=URL")?;
        validate_token("external solver name", name)?;
        let parsed = Url::parse(url).context("external solver URL is invalid")?;
        ensure!(
            matches!(parsed.scheme(), "http" | "https"),
            "external solver URL must use http or https"
        );
        Ok(Self {
            name: name.to_owned(),
            url: url.trim_end_matches('/').to_owned(),
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

#[derive(Parser, Debug, Clone)]
#[command(about = "Exercise challenge-disasterer against a real Anubis server")]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent CLI feature switches are clearer as booleans"
)]
struct Args {
    /// Current upstream Anubis binary, built from the source revision under test.
    #[arg(long)]
    anubis_bin: PathBuf,

    /// challenge-disasterer binary under test.
    #[arg(long)]
    solver_bin: PathBuf,

    /// Anubis algorithm, solver, difficulty, and optional required implementation.
    /// Repeat to test multiple solver/protocol pairs.
    #[arg(
        long = "case",
        value_name = "ALGORITHM:SOLVER:DIFFICULTY[:IMPLEMENTATION]"
    )]
    cases: Vec<CaseSpec>,

    /// Enable challenge-disasterer's GPU backend. CI defaults to CPU-only.
    #[arg(long, default_value_t = false)]
    solver_gpu: bool,

    /// Restrict fuzz/benchmark matrices to GPU-backed routes. Requires `--solver-gpu`.
    #[arg(long, default_value_t = false)]
    gpu_only: bool,

    /// Require `HashX` GPU fuzz/benchmark cases to use this concrete implementation.
    #[arg(long)]
    expect_hashx_gpu_implementation: Option<String>,

    /// Exercise every concrete `HashX` GPU implementation advertised by the solver.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    exercise_all_hashx_implementations: bool,

    /// Run the GPU implementation matrix on every usable Vulkan GPU adapter.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    exercise_all_gpu_adapters: bool,

    /// `HashX` difficulties used for paired routing/implementation benchmarks.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,8,16,18,20,21,22,23,24"
    )]
    hashx_routing_difficulties: Vec<u32>,

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

    /// Run the exhaustive routing calibration matrix and emit JSON.
    #[arg(long, default_value_t = false)]
    calibrate_routing: bool,

    /// JSON file for routing calibration results. Stdout is used when omitted.
    #[arg(long)]
    calibration_output: Option<PathBuf>,

    /// Concurrent solution counts measured for every calibration point.
    #[arg(long, value_delimiter = ',', default_value = "1,10,100")]
    calibration_batch_sizes: Vec<usize>,

    /// Fresh repetitions of each calibration workload size.
    #[arg(long, default_value_t = 3)]
    calibration_rounds: usize,

    /// Do not live-benchmark parameter sets whose expected work exceeds this many hashes.
    #[arg(long, default_value_t = 16_777_216)]
    calibration_max_expected_work: u64,

    /// CPU workers used by the local solver during routing calibration.
    #[arg(long, default_value_t = 4)]
    calibration_cpu_workers: usize,

    /// Additional solver service speaking challenge-disasterer's /stats + /solve protocol.
    /// Repeat as NAME=URL.
    #[arg(long = "external-solver", value_name = "NAME=URL")]
    external_solvers: Vec<ExternalSolverSpec>,

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

#[derive(Debug, Default, Deserialize)]
struct SolverCalibration {
    gpu_adapter: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SolverCapability {
    solver: String,
    algorithms: Vec<String>,
    difficulty_min: u32,
    difficulty_max: u32,
    #[serde(default)]
    router: bool,
    #[serde(default)]
    implementations: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SolverStats {
    #[serde(default)]
    calibration: SolverCalibration,
    #[serde(default)]
    available_solvers: Vec<String>,
    #[serde(default)]
    solver_capabilities: Vec<SolverCapability>,
    #[serde(default)]
    hashx_gpu_implementations: Vec<String>,
}

#[derive(Debug, Clone)]
struct CalibrationCandidate {
    provider: String,
    base_url: String,
    adapter: Option<String>,
    solver: String,
    implementation: Option<String>,
    algorithms: Vec<String>,
    difficulty_min: u32,
    difficulty_max: u32,
    router: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CalibrationEntry {
    provider: String,
    provider_url: Option<String>,
    adapter: Option<String>,
    solver: String,
    implementation: Option<String>,
    algorithm: String,
    difficulty: u32,
    expected_work: u64,
    batch_size: usize,
    rounds: usize,
    makespan_ms: f64,
    mean_request_ms: f64,
    p50_request_ms: f64,
    p90_request_ms: f64,
    mean_solver_ms: f64,
    validation_ms: f64,
    solutions_per_second: f64,
    router: bool,
}

#[derive(Debug, Serialize)]
struct CalibrationReport {
    version: u32,
    generated_unix_ms: u128,
    max_expected_work: u64,
    batch_sizes: Vec<usize>,
    rounds: usize,
    entries: Vec<CalibrationEntry>,
    winners: Vec<CalibrationWinner>,
}

#[derive(Debug, Serialize)]
struct CalibrationWinner {
    algorithm: String,
    difficulty: u32,
    batch_size: usize,
    provider: String,
    provider_url: Option<String>,
    adapter: Option<String>,
    solver: String,
    implementation: Option<String>,
    makespan_ms: f64,
}

struct RunResult {
    temp: Option<TempDir>,
    calibration_entries: Vec<CalibrationEntry>,
}

#[derive(Debug)]
struct PairedSolve {
    label: String,
    solve_ms: f64,
    nonce: String,
    hash: String,
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

fn fetch_solver_stats(solver_base: &str) -> anyhow::Result<SolverStats> {
    Client::new()
        .get(format!("{solver_base}/stats"))
        .send()
        .context("request solver stats")?
        .error_for_status()
        .context("solver stats returned error")?
        .json()
        .context("decode solver stats")
}

fn discover_gpu_adapters(solver_bin: &Path) -> anyhow::Result<Vec<String>> {
    let output = Command::new(solver_bin)
        .arg("--list-hashx-gpu-adapters")
        .output()
        .context("enumerate solver GPU adapters")?;
    ensure!(
        output.status.success(),
        "GPU adapter discovery failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).context("decode GPU adapter list")
}

fn fallback_capabilities(stats: &SolverStats) -> Vec<SolverCapability> {
    let mut out = Vec::new();
    for solver in &stats.available_solvers {
        let capability = match solver.as_str() {
            "anubis" | "anubis-cpu" | "anubis-gpu" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["fast".to_owned(), "slow".to_owned()],
                difficulty_min: 1,
                difficulty_max: 16,
                router: solver == "anubis",
                implementations: Vec::new(),
            },
            "anubis-sha256" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["sha256".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: false,
                implementations: Vec::new(),
            },
            "anubis-argon2id" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["argon2id".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: false,
                implementations: Vec::new(),
            },
            "anubis-hashx-jit" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["hashx".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: false,
                implementations: Vec::new(),
            },
            "anubis-hashx" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["hashx".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: true,
                implementations: stats.hashx_gpu_implementations.clone(),
            },
            "anubis-hashx-auto" => SolverCapability {
                solver: solver.clone(),
                algorithms: vec!["hashx".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: true,
                implementations: Vec::new(),
            },
            _ => continue,
        };
        out.push(capability);
    }
    out
}

fn calibration_candidates(
    provider: &str,
    base_url: &str,
    stats: &SolverStats,
    include_cpu_only: bool,
) -> Vec<CalibrationCandidate> {
    let capabilities = if stats.solver_capabilities.is_empty() {
        fallback_capabilities(stats)
    } else {
        stats.solver_capabilities.clone()
    };
    let selected_adapter = stats.calibration.gpu_adapter.clone();
    let mut out = Vec::new();
    for capability in capabilities {
        let cpu_only = matches!(
            capability.solver.as_str(),
            "anubis-cpu" | "anubis-sha256" | "anubis-argon2id" | "anubis-hashx-jit"
        );
        if cpu_only && !include_cpu_only {
            continue;
        }
        let uses_adapter = matches!(
            capability.solver.as_str(),
            "anubis" | "anubis-gpu" | "anubis-hashx" | "anubis-hashx-auto"
        );
        if !capability.router && capability.implementations.len() == 1 {
            out.push(CalibrationCandidate {
                provider: provider.to_owned(),
                base_url: base_url.to_owned(),
                adapter: uses_adapter.then(|| selected_adapter.clone()).flatten(),
                solver: capability.solver,
                implementation: capability.implementations.into_iter().next(),
                algorithms: capability.algorithms,
                difficulty_min: capability.difficulty_min,
                difficulty_max: capability.difficulty_max,
                router: false,
            });
            continue;
        }
        let router = capability.router || !capability.implementations.is_empty();
        out.push(CalibrationCandidate {
            provider: provider.to_owned(),
            base_url: base_url.to_owned(),
            adapter: uses_adapter.then(|| selected_adapter.clone()).flatten(),
            solver: capability.solver.clone(),
            implementation: None,
            algorithms: capability.algorithms.clone(),
            difficulty_min: capability.difficulty_min,
            difficulty_max: capability.difficulty_max,
            router,
        });
        for implementation in capability.implementations {
            out.push(CalibrationCandidate {
                provider: provider.to_owned(),
                base_url: base_url.to_owned(),
                adapter: selected_adapter.clone(),
                solver: capability.solver.clone(),
                implementation: Some(implementation),
                algorithms: capability.algorithms.clone(),
                difficulty_min: capability.difficulty_min,
                difficulty_max: capability.difficulty_max,
                router: false,
            });
        }
    }
    out
}

fn expected_work_u64(algorithm: &str, difficulty: u32) -> Option<u64> {
    let base = match algorithm {
        "fast" | "slow" => 16_u128,
        "hashx" | "sha256" | "argon2id" => 2_u128,
        _ => return None,
    };
    let work = base.checked_pow(difficulty)?;
    u64::try_from(work).ok()
}

fn calibration_parameter_cases(
    candidates: &[CalibrationCandidate],
    max_expected_work: u64,
) -> anyhow::Result<Vec<CaseSpec>> {
    let mut keys = std::collections::BTreeSet::new();
    for candidate in candidates {
        for algorithm in &candidate.algorithms {
            for difficulty in candidate.difficulty_min..=candidate.difficulty_max {
                let Some(work) = expected_work_u64(algorithm, difficulty) else {
                    break;
                };
                if work > max_expected_work {
                    break;
                }
                keys.insert((algorithm.clone(), difficulty));
            }
        }
    }
    keys.into_iter()
        .map(|(algorithm, difficulty)| format!("{algorithm}:calibration:{difficulty}").parse())
        .collect()
}

fn candidate_supports(candidate: &CalibrationCandidate, case: &CaseSpec) -> bool {
    candidate
        .algorithms
        .iter()
        .any(|algorithm| algorithm == &case.algorithm)
        && (candidate.difficulty_min..=candidate.difficulty_max).contains(&case.difficulty)
}

fn percentile(values: &mut [f64], numerator: usize, denominator: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = values
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_sub(1)
        .min(values.len().saturating_sub(1));
    values[index]
}

struct CalibrationWorkItem {
    client: Client,
    envelope: ChallengeEnvelope,
}

struct CalibrationSolvedItem {
    work: CalibrationWorkItem,
    solved: SolverResponse,
    request_ms: f64,
}

fn issue_calibration_challenge(
    index: usize,
    parameter: &CaseSpec,
    anubis_base: &str,
) -> anyhow::Result<CalibrationWorkItem> {
    let user_agent = format!("challenge-disasterer-e2e/{index}/{}", parameter.algorithm);
    let client = build_client(&user_agent)?;
    let response = client
        .get(format!("{anubis_base}/"))
        .send()
        .context("request calibration challenge page")?;
    ensure!(
        response.status().is_success(),
        "Anubis calibration challenge request failed with {}",
        response.status()
    );
    let html = response.text().context("read calibration challenge page")?;
    let envelope: ChallengeEnvelope = extract_json_script(&html, "anubis_challenge")?;
    ensure!(
        envelope.rules.algorithm == parameter.algorithm,
        "Anubis issued algorithm {}, expected {}",
        envelope.rules.algorithm,
        parameter.algorithm
    );
    ensure!(
        envelope.rules.difficulty == parameter.difficulty,
        "Anubis issued difficulty {}, expected {}",
        envelope.rules.difficulty,
        parameter.difficulty
    );
    Ok(CalibrationWorkItem { client, envelope })
}

fn solve_calibration_item(
    case: &CaseSpec,
    solver_base: &str,
    solver_client: &Client,
    work: CalibrationWorkItem,
) -> anyhow::Result<CalibrationSolvedItem> {
    let mut request = json!({
        "solver": case.solver,
        "algorithm": work.envelope.rules.algorithm,
        "batch_size_hint": case.batch_size_hint,
        "challenge": {
            "data": work.envelope.challenge.random_data,
            "difficulty": work.envelope.rules.difficulty,
        }
    });
    if let Some(implementation) = case.force_implementation.as_deref() {
        request["implementation"] = json!(implementation);
    }
    let started = Instant::now();
    let response = solver_client
        .post(format!("{solver_base}/solve"))
        .json(&request)
        .send()
        .context("submit calibration challenge to solver")?;
    let request_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = response.status();
    let body = response
        .text()
        .context("read calibration solver response")?;
    ensure!(status.is_success(), "solver returned {status}: {body}");
    let solved: SolverResponse =
        serde_json::from_str(&body).context("decode calibration solver response")?;
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
    Ok(CalibrationSolvedItem {
        work,
        solved,
        request_ms,
    })
}

fn validate_calibration_proof(
    anubis_base: &str,
    item: &CalibrationSolvedItem,
) -> anyhow::Result<()> {
    let pass_url = Url::parse(&format!(
        "{anubis_base}/.within.website/x/cmd/anubis/api/pass-challenge"
    ))
    .context("construct calibration pass-challenge URL")?;
    let accepted = item
        .work
        .client
        .get(pass_url)
        .query(&[
            ("id", item.work.envelope.challenge.id.as_str()),
            ("response", item.solved.answer.hash.as_str()),
            ("nonce", item.solved.answer.nonce.as_str()),
            ("redir", "/"),
            ("elapsedTime", "1"),
        ])
        .send()
        .context("submit calibration proof to Anubis")?;
    let status = accepted.status();
    let final_url = accepted.url().clone();
    let body = accepted
        .text()
        .context("read calibration acceptance response")?;
    ensure!(
        status.is_success() && body.contains(BACKEND_MARKER),
        "Anubis did not accept calibration proof (status={status}, url={final_url}); body starts with {:?}",
        body.chars().take(2_000).collect::<String>()
    );
    Ok(())
}

fn calibrate_batch(
    index: usize,
    parameter: &CaseSpec,
    candidate: &CalibrationCandidate,
    anubis_base: &str,
    batch_size: usize,
    rounds: usize,
) -> anyhow::Result<CalibrationEntry> {
    ensure!(batch_size > 0, "calibration batch size must be nonzero");
    ensure!(rounds > 0, "calibration rounds must be nonzero");
    let mut case = parameter.clone();
    case.solver.clone_from(&candidate.solver);
    case.implementation.clone_from(&candidate.implementation);
    case.force_implementation
        .clone_from(&candidate.implementation);
    case.batch_size_hint = batch_size;
    let solver_client = ClientBuilder::new()
        .timeout(Duration::from_secs(600))
        .build()
        .context("build calibration solver client")?;

    let total_samples = batch_size.saturating_mul(rounds);
    let mut request_ms = Vec::with_capacity(total_samples);
    let mut solver_ms = Vec::with_capacity(total_samples);
    let mut makespans = Vec::with_capacity(rounds);
    let mut validation_times = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        // Challenge issuance is intentionally outside the timed region. The quantity the
        // router needs is how quickly this solver can complete an already-issued workload.
        let mut work = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            work.push(issue_calibration_challenge(index, parameter, anubis_base)?);
        }

        let started = Instant::now();
        let solved =
            thread::scope(|scope| -> anyhow::Result<Vec<CalibrationSolvedItem>> {
                let mut handles = Vec::with_capacity(batch_size);
                for item in work {
                    let case = &case;
                    let solver_base = candidate.base_url.as_str();
                    let solver_client = &solver_client;
                    handles.push(scope.spawn(move || {
                        solve_calibration_item(case, solver_base, solver_client, item)
                    }));
                }
                let mut solved = Vec::with_capacity(batch_size);
                for handle in handles {
                    solved.push(
                        handle
                            .join()
                            .map_err(|_| anyhow!("calibration solver thread panicked"))??,
                    );
                }
                Ok(solved)
            })?;
        makespans.push(started.elapsed().as_secs_f64() * 1000.0);

        // Current Anubis uses a non-reentrant WASM validation runner. Validate serially
        // after the solver timer so correctness still comes from real Anubis without
        // contaminating solver throughput with verifier serialization.
        let validation_started = Instant::now();
        for item in &solved {
            validate_calibration_proof(anubis_base, item)?;
        }
        validation_times.push(validation_started.elapsed().as_secs_f64() * 1000.0);

        request_ms.extend(solved.iter().map(|item| item.request_ms));
        solver_ms.extend(solved.iter().map(|item| item.solved.solve_ms));
    }

    let round_count = f64::from(u32::try_from(rounds).context("too many calibration rounds")?);
    let sample_count =
        f64::from(u32::try_from(request_ms.len()).context("calibration batch too large")?);
    let makespan_ms = makespans.iter().sum::<f64>() / round_count;
    let mean_request_ms = request_ms.iter().sum::<f64>() / sample_count;
    let mean_solver_ms = solver_ms.iter().sum::<f64>() / sample_count;
    let validation_ms = validation_times.iter().sum::<f64>() / round_count;
    let mut request_values = request_ms;
    let mut request_values_p90 = request_values.clone();
    let p50_request_ms = percentile(&mut request_values, 1, 2);
    let p90_request_ms = percentile(&mut request_values_p90, 9, 10);
    let expected_work = expected_work_u64(&parameter.algorithm, parameter.difficulty)
        .context("unsupported calibration algorithm")?;
    let batch_count = f64::from(u32::try_from(batch_size).context("calibration batch too large")?);
    Ok(CalibrationEntry {
        provider: candidate.provider.clone(),
        provider_url: (candidate.provider != "local").then(|| candidate.base_url.clone()),
        adapter: candidate.adapter.clone(),
        solver: candidate.solver.clone(),
        implementation: candidate.implementation.clone(),
        algorithm: parameter.algorithm.clone(),
        difficulty: parameter.difficulty,
        expected_work,
        batch_size,
        rounds,
        makespan_ms,
        mean_request_ms,
        p50_request_ms,
        p90_request_ms,
        mean_solver_ms,
        validation_ms,
        solutions_per_second: batch_count * 1000.0 / makespan_ms.max(f64::MIN_POSITIVE),
        router: candidate.router,
    })
}

fn calibration_winners(entries: &[CalibrationEntry]) -> Vec<CalibrationWinner> {
    let mut best = std::collections::BTreeMap::<(String, u32, usize), &CalibrationEntry>::new();
    for entry in entries.iter().filter(|entry| !entry.router) {
        let key = (entry.algorithm.clone(), entry.difficulty, entry.batch_size);
        if best
            .get(&key)
            .is_none_or(|current| entry.makespan_ms < current.makespan_ms)
        {
            best.insert(key, entry);
        }
    }
    best.into_iter()
        .map(
            |((algorithm, difficulty, batch_size), entry)| CalibrationWinner {
                algorithm,
                difficulty,
                batch_size,
                provider: entry.provider.clone(),
                provider_url: entry.provider_url.clone(),
                adapter: entry.adapter.clone(),
                solver: entry.solver.clone(),
                implementation: entry.implementation.clone(),
                makespan_ms: entry.makespan_ms,
            },
        )
        .collect()
}

fn write_calibration_report(args: &Args, entries: Vec<CalibrationEntry>) -> anyhow::Result<()> {
    let generated_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_millis();
    let report = CalibrationReport {
        version: 1,
        generated_unix_ms,
        max_expected_work: args.calibration_max_expected_work,
        batch_sizes: args.calibration_batch_sizes.clone(),
        rounds: args.calibration_rounds,
        winners: calibration_winners(&entries),
        entries,
    };
    let encoded = serde_json::to_string_pretty(&report).context("encode calibration report")?;
    if let Some(path) = args.calibration_output.as_ref() {
        fs::write(path, &encoded)
            .with_context(|| format!("write calibration report {}", path.display()))?;
        println!("CALIBRATION_REPORT path={}", path.display());
    } else {
        println!("CALIBRATION_REPORT_JSON\n{encoded}");
    }
    Ok(())
}

fn post_hashx_solver(
    solver_base: &str,
    solver: &str,
    implementation: Option<&str>,
    data: &str,
    difficulty: u32,
) -> anyhow::Result<SolverResponse> {
    let mut request = json!({
        "solver": solver,
        "algorithm": "hashx",
        "challenge": {"data": data, "difficulty": difficulty}
    });
    if let Some(implementation) = implementation {
        request["implementation"] = json!(implementation);
    }
    let response = Client::new()
        .post(format!("{solver_base}/solve"))
        .json(&request)
        .send()
        .with_context(|| format!("submit HashX solve via {solver}/{implementation:?}"))?;
    let status = response.status();
    let body = response.text().context("read paired solver response")?;
    ensure!(
        status.is_success(),
        "paired solver returned {status}: {body}"
    );
    serde_json::from_str(&body).context("decode paired solver response")
}

fn implementation_fuzz_matrix(implementations: &[String]) -> anyhow::Result<Vec<CaseSpec>> {
    let mut specs = Vec::new();
    for implementation in implementations {
        for difficulty in [1_u32, 8, 16, 20] {
            let mut spec: CaseSpec =
                format!("hashx:anubis-hashx:{difficulty}:{implementation}").parse()?;
            spec.force_implementation = Some(implementation.clone());
            specs.push(spec);
        }
    }
    Ok(specs)
}

fn wasm_implementation_fuzz_matrix(stats: &SolverStats) -> anyhow::Result<Vec<CaseSpec>> {
    let mut specs = Vec::new();
    for capability in &stats.solver_capabilities {
        let difficulties: &[u32] = match capability.solver.as_str() {
            "anubis-sha256" => &[1, 8, 16, 20],
            "anubis-argon2id" => &[1, 2],
            _ => continue,
        };
        for implementation in &capability.implementations {
            for &difficulty in difficulties {
                if !(capability.difficulty_min..=capability.difficulty_max).contains(&difficulty) {
                    continue;
                }
                let algorithm = capability
                    .algorithms
                    .first()
                    .context("Anubis WASM capability has no algorithm")?;
                let mut spec: CaseSpec = format!(
                    "{algorithm}:{}:{difficulty}:{implementation}",
                    capability.solver
                )
                .parse()?;
                spec.force_implementation = Some(implementation.clone());
                specs.push(spec);
            }
        }
    }
    Ok(specs)
}

fn routing_matrix(difficulties: &[u32]) -> anyhow::Result<Vec<CaseSpec>> {
    difficulties
        .iter()
        .map(|difficulty| format!("hashx:anubis-hashx:{difficulty}").parse())
        .collect()
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

fn solve_case(
    index: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
    negative_check: bool,
) -> anyhow::Result<TrialResult> {
    solve_case_inner(index, case, anubis_base, solver_base, negative_check, true)
}

#[allow(
    clippy::too_many_lines,
    reason = "keeps one complete protocol round trip together"
)]
fn solve_case_inner(
    index: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
    negative_check: bool,
    emit_pass: bool,
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

    let mut solver_request = json!({
        "solver": case.solver,
        "algorithm": envelope.rules.algorithm,
        "batch_size_hint": case.batch_size_hint,
        "challenge": {
            "data": envelope.challenge.random_data,
            "difficulty": envelope.rules.difficulty,
        }
    });
    if let Some(implementation) = case.force_implementation.as_deref() {
        solver_request["implementation"] = json!(implementation);
    }
    let solver_timeout = if emit_pass {
        Duration::from_secs(30)
    } else {
        Duration::from_secs(600)
    };
    let solver_response = ClientBuilder::new()
        .timeout(solver_timeout)
        .build()
        .context("build solver client")?
        .post(format!("{solver_base}/solve"))
        .json(&solver_request)
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
        body.chars().take(2_000).collect::<String>()
    );

    let wall_ms = wall_started.elapsed().as_secs_f64() * 1000.0;
    if emit_pass {
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
    }
    Ok(TrialResult {
        wall_ms,
        solve_ms: solved.solve_ms,
        backend: solved.backend,
        implementation: solved.implementation,
    })
}

fn paired_hashx_sample(
    index: usize,
    sample: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
    implementations: &[String],
) -> anyhow::Result<(Vec<PairedSolve>, String)> {
    let user_agent = format!("challenge-disasterer-e2e/{index}/{}", case.algorithm);
    let client = build_client(&user_agent)?;
    let response = client
        .get(format!("{anubis_base}/"))
        .send()
        .context("request paired HashX challenge")?;
    ensure!(
        response.status().is_success(),
        "paired challenge request failed"
    );
    let html = response.text().context("read paired challenge page")?;
    let envelope: ChallengeEnvelope = extract_json_script(&html, "anubis_challenge")?;
    ensure!(
        envelope.rules.algorithm == "hashx",
        "paired benchmark got non-HashX challenge"
    );
    ensure!(
        envelope.rules.difficulty == case.difficulty,
        "paired benchmark difficulty mismatch"
    );

    let mut variants = vec![("cpu-jit".to_owned(), "anubis-hashx-jit", None)];
    variants.push(("auto".to_owned(), "anubis-hashx", None));
    for implementation in implementations {
        variants.push((
            implementation.clone(),
            "anubis-hashx",
            Some(implementation.as_str()),
        ));
    }
    let variant_count = variants.len();
    variants.rotate_left(sample % variant_count);

    let mut solves = Vec::with_capacity(variants.len());
    let mut auto_implementation = None;
    for (label, solver, requested) in variants {
        let response = post_hashx_solver(
            solver_base,
            solver,
            requested,
            &envelope.challenge.random_data,
            case.difficulty,
        )?;
        if label == "auto" {
            auto_implementation = Some(
                response
                    .implementation
                    .clone()
                    .context("auto HashX GPU response omitted implementation")?,
            );
        } else if let Some(expected) = requested {
            ensure!(
                response.implementation.as_deref() == Some(expected),
                "forced implementation {expected} returned {:?}",
                response.implementation
            );
        }
        solves.push(PairedSolve {
            label,
            solve_ms: response.solve_ms,
            nonce: response.answer.nonce,
            hash: response.answer.hash,
        });
    }

    let reference = solves
        .iter()
        .find(|solve| solve.label == "cpu-jit")
        .context("paired benchmark missing CPU JIT reference")?;
    for solved in &solves {
        ensure!(
            solved.nonce == reference.nonce && solved.hash == reference.hash,
            "HashX implementation disagreement at d{} sample {}: {} returned nonce={} hash={}, cpu-jit nonce={} hash={}",
            case.difficulty,
            sample,
            solved.label,
            solved.nonce,
            solved.hash,
            reference.nonce,
            reference.hash
        );
    }
    Ok((
        solves,
        auto_implementation.context("paired benchmark missing auto implementation")?,
    ))
}

fn paired_mean(values: &[f64]) -> f64 {
    let count = u32::try_from(values.len()).expect("paired sample count fits u32");
    values.iter().sum::<f64>() / f64::from(count)
}

fn print_paired_hashx_report(
    difficulty: u32,
    samples: usize,
    timings: &std::collections::BTreeMap<String, Vec<f64>>,
    auto_implementations: &std::collections::BTreeMap<String, usize>,
    implementations: &[String],
) -> anyhow::Result<()> {
    let auto_ms = paired_mean(timings.get("auto").context("missing auto timings")?);
    let mut best_gpu = None::<(&str, f64)>;
    for implementation in implementations {
        let value = paired_mean(
            timings
                .get(implementation)
                .with_context(|| format!("missing timings for {implementation}"))?,
        );
        if best_gpu.is_none_or(|(_, best)| value < best) {
            best_gpu = Some((implementation.as_str(), value));
        }
    }
    let (best_gpu_name, best_gpu_ms) = best_gpu.context("no GPU implementations to benchmark")?;
    let jit_ms = paired_mean(timings.get("cpu-jit").context("missing cpu-jit timings")?);
    let slowdown = auto_ms / best_gpu_ms.max(f64::MIN_POSITIVE);
    let selection_matches_best =
        auto_implementations.len() == 1 && auto_implementations.contains_key(best_gpu_name);
    let routing = if selection_matches_best || slowdown <= 1.25 {
        "OK"
    } else {
        "SUBOPTIMAL"
    };
    let (best_overall, best_overall_ms) = if jit_ms < best_gpu_ms {
        ("cpu-jit", jit_ms)
    } else {
        (best_gpu_name, best_gpu_ms)
    };
    println!(
        "ROUTING status={routing} difficulty={difficulty} n={samples} auto_ms={auto_ms:.3} auto_impls={auto_implementations:?} best_gpu={best_gpu_name} best_gpu_ms={best_gpu_ms:.3} selection_matches_best={selection_matches_best} slowdown={slowdown:.3} cpu_jit_ms={jit_ms:.3} best_overall={best_overall} best_overall_ms={best_overall_ms:.3}"
    );
    for (label, values) in timings {
        println!(
            "PAIRED_HASHX difficulty={} implementation={} n={} mean_solver_ms={:.3}",
            difficulty,
            label,
            values.len(),
            paired_mean(values)
        );
    }
    Ok(())
}

fn paired_hashx_benchmark(
    index: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
    implementations: &[String],
    samples: usize,
) -> anyhow::Result<()> {
    let mut timings = std::collections::BTreeMap::<String, Vec<f64>>::new();
    let mut auto_implementations = std::collections::BTreeMap::<String, usize>::new();
    for sample in 0..samples {
        let (solves, auto_implementation) = paired_hashx_sample(
            index,
            sample,
            case,
            anubis_base,
            solver_base,
            implementations,
        )?;
        *auto_implementations.entry(auto_implementation).or_default() += 1;
        for solved in solves {
            timings
                .entry(solved.label)
                .or_default()
                .push(solved.solve_ms);
        }
    }
    print_paired_hashx_report(
        case.difficulty,
        samples,
        &timings,
        &auto_implementations,
        implementations,
    )
}

fn expected_hashes(case: &CaseSpec) -> f64 {
    match case.algorithm.as_str() {
        "fast" | "slow" => 16_f64.powi(i32::try_from(case.difficulty).unwrap_or(i32::MAX)),
        "hashx" | "argon2id" | "sha256" => {
            2_f64.powi(i32::try_from(case.difficulty).unwrap_or(i32::MAX))
        }
        _ => f64::NAN,
    }
}

fn fuzz_matrix(
    gpu: bool,
    implementation: Option<&str>,
    gpu_only: bool,
) -> anyhow::Result<Vec<CaseSpec>> {
    ensure!(
        !gpu_only || gpu,
        "GPU-only fuzzing requires the GPU backend"
    );
    let mut specs = Vec::new();
    if !gpu_only {
        for difficulty in [1_u32, 2, 3, 4, 5] {
            specs.push(format!("fast:anubis-cpu:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 3] {
            specs.push(format!("slow:anubis-cpu:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 2, 4, 8, 12, 16] {
            specs.push(format!("hashx:anubis-hashx-jit:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 4, 8] {
            specs.push(format!("sha256:anubis-sha256:{difficulty}").parse()?);
        }
        for difficulty in [1_u32, 2] {
            specs.push(format!("argon2id:anubis-argon2id:{difficulty}").parse()?);
        }
    }
    for difficulty in [2_u32, 4] {
        specs.push(format!("fast:anubis:{difficulty}").parse()?);
    }
    for difficulty in [1_u32, 3] {
        specs.push(format!("slow:anubis:{difficulty}").parse()?);
    }
    for difficulty in [1_u32, 8, 16] {
        specs.push(format!("hashx:anubis-hashx-auto:{difficulty}").parse()?);
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

fn benchmark_matrix(
    gpu: bool,
    implementation: Option<&str>,
    gpu_only: bool,
) -> anyhow::Result<Vec<CaseSpec>> {
    ensure!(
        !gpu_only || gpu,
        "GPU-only benchmarking requires the GPU backend"
    );
    let mut specs = if gpu_only {
        Vec::new()
    } else {
        vec![
            "fast:anubis-cpu:5".parse()?,
            "hashx:anubis-hashx-jit:18".parse()?,
            "sha256:anubis-sha256:18".parse()?,
            "argon2id:anubis-argon2id:4".parse()?,
        ]
    };
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
    let hashes_per_s =
        expected_hashes(case) * sample_count * 1000.0 / solve_ms.max(f64::MIN_POSITIVE);
    let backend = trials
        .first()
        .map_or("unknown", |trial| trial.backend.as_str());
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

#[allow(
    clippy::too_many_lines,
    reason = "top-level test orchestration is intentionally linear"
)]
fn run(
    args: Args,
    adapter_override: Option<&str>,
    include_shared_calibration_candidates: bool,
) -> anyhow::Result<RunResult> {
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
    ensure!(
        args.hashx_routing_difficulties
            .iter()
            .all(|difficulty| *difficulty > 0),
        "HashX routing difficulties must be greater than zero"
    );

    ensure!(
        !args.calibrate_routing || !args.calibration_batch_sizes.is_empty(),
        "routing calibration needs at least one batch size"
    );
    ensure!(
        args.calibration_batch_sizes.iter().all(|size| *size > 0),
        "calibration batch sizes must be greater than zero"
    );
    ensure!(
        args.calibration_rounds > 0,
        "calibration rounds must be greater than zero"
    );
    ensure!(
        args.calibration_cpu_workers > 0,
        "calibration CPU workers must be greater than zero"
    );

    if let Some(ref implementation) = args.expect_hashx_gpu_implementation {
        validate_token("implementation", implementation)?;
        ensure!(
            args.solver_gpu,
            "--expect-hashx-gpu-implementation requires --solver-gpu"
        );
    }

    let temp = tempfile::tempdir().context("create e2e temp directory")?;
    let solver_port = reserve_port()?;
    let solver_base = format!("http://127.0.0.1:{solver_port}");
    let mut solver_command = Command::new(&args.solver_bin);
    solver_command
        .arg("--listen")
        .arg(format!("127.0.0.1:{solver_port}"))
        .arg("--gpu")
        .arg(args.solver_gpu.to_string())
        .arg("--cpu-workers")
        .arg(if args.calibrate_routing {
            args.calibration_cpu_workers.to_string()
        } else if args.benchmark {
            "1".to_owned()
        } else {
            "2".to_owned()
        })
        .args(&args.solver_forwarded);
    if let Some(adapter) = adapter_override {
        solver_command.env("WGPU_ADAPTER_NAME", adapter);
    }
    let mut solver = ManagedChild::spawn(
        "challenge-disasterer",
        &mut solver_command,
        temp.path().join("solver.log"),
    )?;
    wait_ready(&format!("{solver_base}/stats"), &mut solver)?;
    let solver_stats = fetch_solver_stats(&solver_base)?;
    if let Some(adapter) = adapter_override {
        let selected = solver_stats
            .calibration
            .gpu_adapter
            .as_deref()
            .context("solver did not report a selected GPU adapter")?;
        ensure!(
            selected.contains(adapter),
            "requested GPU adapter {adapter:?}, solver selected {selected:?}"
        );
        println!("GPU_ADAPTER selected={selected:?}");
    }
    let implementations = if args.solver_gpu && args.exercise_all_hashx_implementations {
        solver_stats.hashx_gpu_implementations.clone()
    } else {
        Vec::new()
    };
    if args.solver_gpu && args.exercise_all_hashx_implementations {
        ensure!(
            !implementations.is_empty(),
            "solver advertised no concrete HashX GPU implementations"
        );
        println!("HASHX_IMPLEMENTATIONS available={implementations:?}");
    }
    if let Some(expected) = args.expect_hashx_gpu_implementation.as_deref() {
        ensure!(
            implementations.is_empty() || implementations.iter().any(|value| value == expected),
            "required HashX implementation {expected} is not advertised; available: {implementations:?}"
        );
    }

    let mut calibration_candidates_for_run = Vec::new();
    if args.calibrate_routing {
        calibration_candidates_for_run.extend(calibration_candidates(
            "local",
            &solver_base,
            &solver_stats,
            include_shared_calibration_candidates,
        ));
        if include_shared_calibration_candidates {
            for external in &args.external_solvers {
                let stats = fetch_solver_stats(&external.url).with_context(|| {
                    format!("fetch external solver capabilities from {}", external.name)
                })?;
                calibration_candidates_for_run.extend(calibration_candidates(
                    &external.name,
                    &external.url,
                    &stats,
                    true,
                ));
            }
        }
        ensure!(
            !calibration_candidates_for_run.is_empty(),
            "routing calibration discovered no solver candidates"
        );
    }

    let mut cases = args.cases;
    if args.fuzz {
        cases.extend(fuzz_matrix(
            args.solver_gpu,
            args.expect_hashx_gpu_implementation.as_deref(),
            args.gpu_only,
        )?);
        let mut wasm_cases = wasm_implementation_fuzz_matrix(&solver_stats)?;
        if args.gpu_only {
            wasm_cases.retain(|case| case.implementation.as_deref() == Some("wgpu"));
        }
        cases.extend(wasm_cases);
    }
    if !implementations.is_empty() {
        cases.extend(implementation_fuzz_matrix(&implementations)?);
        if !args.fuzz {
            cases.push("hashx:anubis-hashx:16".parse()?);
        }
    }
    let benchmark_cases = if args.benchmark {
        benchmark_matrix(
            args.solver_gpu,
            args.expect_hashx_gpu_implementation.as_deref(),
            args.gpu_only,
        )?
    } else {
        Vec::new()
    };
    let routing_cases = if args.benchmark && !implementations.is_empty() {
        routing_matrix(&args.hashx_routing_difficulties)?
    } else {
        Vec::new()
    };
    let browser_cases = if args.benchmark && args.browser_reference && !args.gpu_only {
        browser_benchmark_matrix()?
    } else {
        Vec::new()
    };
    let calibration_cases = if args.calibrate_routing {
        calibration_parameter_cases(
            &calibration_candidates_for_run,
            args.calibration_max_expected_work,
        )?
    } else {
        Vec::new()
    };
    cases.extend(benchmark_cases.iter().cloned());
    cases.extend(routing_cases.iter().cloned());
    cases.extend(browser_cases.iter().cloned());
    cases.extend(calibration_cases.iter().cloned());
    if cases.is_empty() {
        cases.push("fast:anubis:2".parse()?);
    }

    let policy_path = temp.path().join("anubis.yaml");
    write_policy(&policy_path, &cases)?;
    let backend = BackendServer::start()?;
    let anubis_port = reserve_port()?;
    let metrics_port = reserve_port()?;
    let anubis_base = format!("http://127.0.0.1:{anubis_port}");
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

    let correctness_count = cases.len().saturating_sub(
        benchmark_cases.len() + routing_cases.len() + browser_cases.len() + calibration_cases.len(),
    );
    let benchmark_start = correctness_count;
    let routing_start = benchmark_start + benchmark_cases.len();
    let browser_start = routing_start + routing_cases.len();
    let calibration_start = browser_start + browser_cases.len();

    let mut fuzz_trials = 0_usize;
    for (index, case) in cases[..correctness_count].iter().enumerate() {
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
            let trial =
                solve_case(index, case, &anubis_base, &solver_base, false).with_context(|| {
                    format!(
                        "benchmark case failed: {}:{}:{}",
                        case.algorithm, case.solver, case.difficulty
                    )
                })?;
            trials.push(trial);
        }
        print_benchmark(case, &trials)?;
    }

    for (offset, case) in routing_cases.iter().enumerate() {
        paired_hashx_benchmark(
            routing_start + offset,
            case,
            &anubis_base,
            &solver_base,
            &implementations,
            args.benchmark_samples,
        )?;
    }

    if !browser_cases.is_empty() {
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
                    "index": browser_start + offset,
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
            bail!(
                "stock Anubis browser benchmark failed with {}",
                output.status
            );
        }
    }

    let mut calibration_entries = Vec::new();
    if args.calibrate_routing {
        for (offset, parameter) in calibration_cases.iter().enumerate() {
            let index = calibration_start + offset;
            for candidate in &calibration_candidates_for_run {
                if !candidate_supports(candidate, parameter) {
                    continue;
                }
                for &batch_size in &args.calibration_batch_sizes {
                    let entry = match calibrate_batch(
                        index,
                        parameter,
                        candidate,
                        &anubis_base,
                        batch_size,
                        args.calibration_rounds,
                    ) {
                        Ok(entry) => entry,
                        Err(error) => {
                            eprintln!("\n--- challenge-disasterer log ---\n{}", solver.log_tail());
                            eprintln!("\n--- Anubis log ---\n{}", anubis.log_tail());
                            return Err(error).with_context(|| {
                                format!(
                                    "routing calibration failed: provider={} solver={} implementation={:?} algorithm={} difficulty={} batch={}",
                                    candidate.provider,
                                    candidate.solver,
                                    candidate.implementation,
                                    parameter.algorithm,
                                    parameter.difficulty,
                                    batch_size,
                                )
                            });
                        }
                    };
                    println!(
                        "CALIBRATE provider={} adapter={} solver={} implementation={} algorithm={} difficulty={} batch={} rounds={} makespan_ms={:.3} mean_request_ms={:.3} validation_ms={:.3} solutions_s={:.3} router={}",
                        entry.provider,
                        entry.adapter.as_deref().unwrap_or("-"),
                        entry.solver,
                        entry.implementation.as_deref().unwrap_or("-"),
                        entry.algorithm,
                        entry.difficulty,
                        entry.batch_size,
                        entry.rounds,
                        entry.makespan_ms,
                        entry.mean_request_ms,
                        entry.validation_ms,
                        entry.solutions_per_second,
                        entry.router,
                    );
                    calibration_entries.push(entry);
                }
            }
        }
    }

    println!("PASS {fuzz_trials} live-Anubis correctness trial(s)");
    let temp = if args.keep_temp {
        let kept = temp.keep();
        println!("kept e2e artifacts at {}", kept.display());
        None
    } else {
        Some(temp)
    };
    Ok(RunResult {
        temp,
        calibration_entries,
    })
}

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    ensure!(
        !args.gpu_only || args.solver_gpu,
        "--gpu-only requires --solver-gpu"
    );
    if args.calibration_output.is_some() {
        args.calibrate_routing = true;
    }
    args.calibration_batch_sizes.sort_unstable();
    args.calibration_batch_sizes.dedup();

    let mut all_calibration_entries = Vec::new();
    if args.solver_gpu && args.exercise_all_gpu_adapters {
        let adapters = discover_gpu_adapters(&args.solver_bin)?;
        ensure!(
            !adapters.is_empty(),
            "no usable Vulkan GPU adapters discovered"
        );
        println!("GPU_ADAPTERS discovered={adapters:?}");
        let mut temps = Vec::with_capacity(adapters.len());
        for (index, adapter) in adapters.iter().enumerate() {
            println!("GPU_ADAPTER_RUN index={index} adapter={adapter:?}");
            let mut per_adapter = args.clone();
            if index != 0 {
                per_adapter.browser_reference = false;
            }
            let result = run(per_adapter, Some(adapter), index == 0)?;
            all_calibration_entries.extend(result.calibration_entries);
            temps.push(result.temp);
        }
        drop(temps);
    } else {
        let result = run(args.clone(), None, true).map_err(|error| anyhow!("{error:#}"))?;
        all_calibration_entries.extend(result.calibration_entries);
        let _temp = result.temp;
    }
    if args.calibrate_routing {
        write_calibration_report(&args, all_calibration_entries)?;
    }
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
        assert_eq!(case.force_implementation, None);

        let concrete: CaseSpec = "hashx:anubis-hashx:18:cuda-sm75-sass"
            .parse()
            .expect("valid implementation-qualified case");
        assert_eq!(concrete.difficulty, 18);
        assert_eq!(concrete.implementation.as_deref(), Some("cuda-sm75-sass"));
    }

    #[test]
    fn fuzz_matrix_covers_anubis_router_algorithms() -> anyhow::Result<()> {
        let matrix = fuzz_matrix(false, None, false)?;
        assert!(
            matrix
                .iter()
                .any(|case| case.solver == "anubis" && case.algorithm == "fast")
        );
        assert!(
            matrix
                .iter()
                .any(|case| case.solver == "anubis" && case.algorithm == "slow")
        );
        Ok(())
    }

    #[test]
    fn fuzz_matrix_covers_hashx_auto_router() -> anyhow::Result<()> {
        for gpu in [false, true] {
            let matrix = fuzz_matrix(gpu, None, false)?;
            let difficulties = matrix
                .iter()
                .filter(|case| case.solver == "anubis-hashx-auto")
                .map(|case| case.difficulty)
                .collect::<Vec<_>>();
            assert_eq!(difficulties, vec![1, 8, 16]);
        }
        Ok(())
    }

    #[test]
    fn gpu_only_matrices_exclude_cpu_only_work() -> anyhow::Result<()> {
        let fuzz = fuzz_matrix(true, None, true)?;
        assert!(fuzz.iter().any(|case| case.solver == "anubis"));
        assert!(fuzz.iter().any(|case| case.solver == "anubis-gpu"));
        assert!(fuzz.iter().any(|case| case.solver == "anubis-hashx-auto"));
        assert!(fuzz.iter().any(|case| case.solver == "anubis-hashx"));
        assert!(!fuzz.iter().any(|case| case.solver == "anubis-cpu"));
        assert!(!fuzz.iter().any(|case| case.solver == "anubis-hashx-jit"));
        assert!(!fuzz.iter().any(|case| case.solver == "anubis-argon2id"));

        let benchmark = benchmark_matrix(true, None, true)?;
        assert!(benchmark.iter().any(|case| case.solver == "anubis-gpu"));
        assert!(benchmark.iter().any(|case| case.solver == "anubis-hashx"));
        assert!(!benchmark.iter().any(|case| case.solver == "anubis-cpu"));
        assert!(
            !benchmark
                .iter()
                .any(|case| case.solver == "anubis-hashx-jit")
        );
        assert!(
            !benchmark
                .iter()
                .any(|case| case.solver == "anubis-argon2id")
        );
        Ok(())
    }

    #[test]
    fn gpu_matrices_propagate_required_implementation() -> anyhow::Result<()> {
        let expected = "vulkan-extmul-persistent";
        let fuzz = fuzz_matrix(true, Some(expected), false)?;
        let bench = benchmark_matrix(true, Some(expected), false)?;
        assert!(
            fuzz.iter()
                .filter(|case| case.solver == "anubis-hashx")
                .all(|case| case.implementation.as_deref() == Some(expected))
        );
        assert!(
            bench
                .iter()
                .filter(|case| case.solver == "anubis-hashx")
                .all(|case| case.implementation.as_deref() == Some(expected))
        );
        Ok(())
    }

    #[test]
    fn implementation_matrix_forces_and_asserts_each_backend() -> anyhow::Result<()> {
        let implementations = vec![
            "vulkan-extmul-persistent".to_owned(),
            "wgsl-specialized".to_owned(),
        ];
        let matrix = implementation_fuzz_matrix(&implementations)?;
        assert_eq!(matrix.len(), 8);
        assert!(
            matrix
                .iter()
                .all(|case| case.force_implementation == case.implementation)
        );
        Ok(())
    }

    #[test]
    fn wasm_implementation_matrix_forces_every_advertised_backend() -> anyhow::Result<()> {
        let stats: SolverStats = serde_json::from_value(json!({
            "solver_capabilities": [
                {
                    "solver": "anubis-sha256",
                    "algorithms": ["sha256"],
                    "difficulty_min": 1,
                    "difficulty_max": 64,
                    "implementations": ["auto", "safe", "avx2", "wgpu"]
                },
                {
                    "solver": "anubis-argon2id",
                    "algorithms": ["argon2id"],
                    "difficulty_min": 1,
                    "difficulty_max": 64,
                    "implementations": ["portable-simd"]
                }
            ]
        }))?;
        let matrix = wasm_implementation_fuzz_matrix(&stats)?;
        assert_eq!(matrix.len(), 18);
        assert!(
            matrix
                .iter()
                .all(|case| case.force_implementation == case.implementation)
        );
        assert_eq!(
            matrix
                .iter()
                .filter(|case| case.implementation.as_deref() == Some("wgpu"))
                .count(),
            4
        );
        Ok(())
    }

    #[test]
    fn external_solver_spec_parses_name_and_url() -> anyhow::Result<()> {
        let spec: ExternalSolverSpec = "remote=http://127.0.0.1:17418".parse()?;
        assert_eq!(spec.name, "remote");
        assert_eq!(spec.url, "http://127.0.0.1:17418");
        Ok(())
    }

    #[test]
    fn minimal_external_stats_need_only_capabilities() -> anyhow::Result<()> {
        let stats: SolverStats = serde_json::from_value(json!({
            "solver_capabilities": [{
                "solver": "remote-hashx",
                "algorithms": ["hashx"],
                "difficulty_min": 1,
                "difficulty_max": 32,
                "router": false,
                "implementations": []
            }]
        }))?;
        assert_eq!(stats.calibration.gpu_adapter, None);
        assert_eq!(stats.available_solvers, Vec::<String>::new());
        assert_eq!(stats.solver_capabilities.len(), 1);
        assert_eq!(stats.solver_capabilities[0].solver, "remote-hashx");
        Ok(())
    }

    #[test]
    fn singleton_implementation_calibrates_once_as_concrete_backend() -> anyhow::Result<()> {
        let stats: SolverStats = serde_json::from_value(json!({
            "solver_capabilities": [{
                "solver": "anubis-argon2id",
                "algorithms": ["argon2id"],
                "difficulty_min": 1,
                "difficulty_max": 64,
                "router": false,
                "implementations": ["portable-simd"]
            }]
        }))?;
        let candidates = calibration_candidates("local", "http://127.0.0.1:1", &stats, true);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].solver, "anubis-argon2id");
        assert_eq!(
            candidates[0].implementation.as_deref(),
            Some("portable-simd")
        );
        assert!(!candidates[0].router);
        Ok(())
    }

    #[test]
    fn calibration_parameter_matrix_respects_expected_work_cap() -> anyhow::Result<()> {
        let candidates = vec![
            CalibrationCandidate {
                provider: "local".to_owned(),
                base_url: "http://127.0.0.1:1".to_owned(),
                adapter: None,
                solver: "anubis-cpu".to_owned(),
                implementation: None,
                algorithms: vec!["fast".to_owned()],
                difficulty_min: 1,
                difficulty_max: 16,
                router: false,
            },
            CalibrationCandidate {
                provider: "local".to_owned(),
                base_url: "http://127.0.0.1:1".to_owned(),
                adapter: None,
                solver: "anubis-hashx-jit".to_owned(),
                implementation: None,
                algorithms: vec!["hashx".to_owned()],
                difficulty_min: 1,
                difficulty_max: 64,
                router: false,
            },
        ];
        let cases = calibration_parameter_cases(&candidates, 256)?;
        let keys = cases
            .iter()
            .map(|case| (case.algorithm.as_str(), case.difficulty))
            .collect::<Vec<_>>();
        assert!(keys.contains(&("fast", 1)));
        assert!(keys.contains(&("fast", 2)));
        assert!(!keys.contains(&("fast", 3)));
        assert!(keys.contains(&("hashx", 8)));
        assert!(!keys.contains(&("hashx", 9)));
        Ok(())
    }

    #[test]
    fn calibration_winners_exclude_router_rows() {
        let make = |solver: &str, router: bool, makespan_ms: f64| CalibrationEntry {
            provider: "local".to_owned(),
            provider_url: None,
            adapter: None,
            solver: solver.to_owned(),
            implementation: None,
            algorithm: "hashx".to_owned(),
            difficulty: 8,
            expected_work: 256,
            batch_size: 10,
            rounds: 1,
            makespan_ms,
            mean_request_ms: makespan_ms,
            p50_request_ms: makespan_ms,
            p90_request_ms: makespan_ms,
            mean_solver_ms: makespan_ms,
            validation_ms: makespan_ms,
            solutions_per_second: 1.0,
            router,
        };
        let winners = calibration_winners(&[
            make("anubis-hashx-auto", true, 1.0),
            make("anubis-hashx-jit", false, 10.0),
            make("anubis-hashx", false, 5.0),
        ]);
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].solver, "anubis-hashx");
        assert_eq!(winners[0].makespan_ms, 5.0);
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
