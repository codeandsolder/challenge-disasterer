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
    force_implementation: Option<String>,
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
    #[arg(long, value_delimiter = ',', default_value = "1,8,16,18,20,21,22,23,24")]
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

#[derive(Debug, Deserialize)]
struct SolverCalibration {
    gpu_adapter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SolverStats {
    calibration: SolverCalibration,
    #[serde(default)]
    hashx_gpu_implementations: Vec<String>,
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

fn post_hashx_solver(
    solver_base: &str,
    solver: &str,
    implementation: Option<&str>,
    data: &str,
    difficulty: u32,
) -> anyhow::Result<SolverResponse> {
    let mut request = json!({
        "solver": solver,
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
    ensure!(status.is_success(), "paired solver returned {status}: {body}");
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

    let mut solver_request = json!({
        "solver": case.solver,
        "challenge": {
            "data": envelope.challenge.random_data,
            "difficulty": envelope.rules.difficulty,
        }
    });
    if let Some(implementation) = case.force_implementation.as_deref() {
        solver_request["implementation"] = json!(implementation);
    }
    let solver_response = Client::new()
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
    ensure!(response.status().is_success(), "paired challenge request failed");
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
    let selection_matches_best = auto_implementations.len() == 1
        && auto_implementations.contains_key(best_gpu_name);
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
            timings.entry(solved.label).or_default().push(solved.solve_ms);
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
fn run(args: Args, adapter_override: Option<&str>) -> anyhow::Result<Option<TempDir>> {
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
        args.hashx_routing_difficulties.iter().all(|difficulty| *difficulty > 0),
        "HashX routing difficulties must be greater than zero"
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
        .arg(if args.benchmark { "1" } else { "2" })
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
        solver_stats.hashx_gpu_implementations
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

    let mut cases = args.cases;
    if args.fuzz {
        cases.extend(fuzz_matrix(
            args.solver_gpu,
            args.expect_hashx_gpu_implementation.as_deref(),
        )?);
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
        )?
    } else {
        Vec::new()
    };
    let routing_cases = if args.benchmark && !implementations.is_empty() {
        routing_matrix(&args.hashx_routing_difficulties)?
    } else {
        Vec::new()
    };
    let browser_cases = if args.benchmark && args.browser_reference {
        browser_benchmark_matrix()?
    } else {
        Vec::new()
    };
    cases.extend(benchmark_cases.iter().cloned());
    cases.extend(routing_cases.iter().cloned());
    cases.extend(browser_cases.iter().cloned());
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
        benchmark_cases.len() + routing_cases.len() + browser_cases.len(),
    );
    let benchmark_start = correctness_count;
    let routing_start = benchmark_start + benchmark_cases.len();
    let browser_start = routing_start + routing_cases.len();

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
            let trial = solve_case(index, case, &anubis_base, &solver_base, false)
                .with_context(|| format!("benchmark case failed: {}:{}:{}", case.algorithm, case.solver, case.difficulty))?;
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
    if args.solver_gpu && args.exercise_all_gpu_adapters {
        let adapters = discover_gpu_adapters(&args.solver_bin)?;
        ensure!(!adapters.is_empty(), "no usable Vulkan GPU adapters discovered");
        println!("GPU_ADAPTERS discovered={adapters:?}");
        let mut temps = Vec::with_capacity(adapters.len());
        for (index, adapter) in adapters.iter().enumerate() {
            println!("GPU_ADAPTER_RUN index={index} adapter={adapter:?}");
            let mut per_adapter = args.clone();
            if index != 0 {
                per_adapter.browser_reference = false;
            }
            temps.push(run(per_adapter, Some(adapter))?);
        }
        drop(temps);
    } else {
        let _temp = run(args, None).map_err(|error| anyhow!("{error:#}"))?;
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
    fn gpu_matrices_propagate_required_implementation() -> anyhow::Result<()> {
        let expected = "vulkan-extmul-persistent";
        let fuzz = fuzz_matrix(true, Some(expected))?;
        let bench = benchmark_matrix(true, Some(expected))?;
        assert!(fuzz.iter().filter(|case| case.solver == "anubis-hashx").all(|case| case.implementation.as_deref() == Some(expected)));
        assert!(bench.iter().filter(|case| case.solver == "anubis-hashx").all(|case| case.implementation.as_deref() == Some(expected)));
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
        assert!(matrix.iter().all(|case| case.force_implementation == case.implementation));
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
