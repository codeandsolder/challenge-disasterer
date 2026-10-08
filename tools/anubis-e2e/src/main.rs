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
        ensure!(
            fields.next().is_none(),
            "case must be algorithm:solver:difficulty"
        );
        validate_token("algorithm", algorithm)?;
        validate_token("solver", solver)?;
        ensure!(difficulty > 0, "difficulty must be greater than zero");
        Ok(Self {
            algorithm: algorithm.to_owned(),
            solver: solver.to_owned(),
            difficulty,
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
struct Args {
    /// Current upstream Anubis binary, built from the source revision under test.
    #[arg(long)]
    anubis_bin: PathBuf,

    /// challenge-disasterer binary under test.
    #[arg(long)]
    solver_bin: PathBuf,

    /// Anubis algorithm, challenge-disasterer solver name, and difficulty.
    /// Repeat to test multiple solver/protocol pairs.
    #[arg(long = "case", value_name = "ALGORITHM:SOLVER:DIFFICULTY")]
    cases: Vec<CaseSpec>,

    /// Enable challenge-disasterer's GPU backend. CI defaults to CPU-only.
    #[arg(long, default_value_t = false)]
    solver_gpu: bool,

    /// Extra argument forwarded verbatim to challenge-disasterer.
    #[arg(long = "solver-arg")]
    solver_forwarded: Vec<String>,

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
    answer: SolverAnswer,
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

fn solve_case(
    index: usize,
    case: &CaseSpec,
    anubis_base: &str,
    solver_base: &str,
) -> anyhow::Result<()> {
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

    let pass_url = Url::parse(&format!(
        "{anubis_base}/.within.website/x/cmd/anubis/api/pass-challenge"
    ))
    .context("construct pass-challenge URL")?;
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

    println!(
        "PASS algorithm={} solver={} backend={} difficulty={} nonce={} final_url={}",
        case.algorithm,
        case.solver,
        solved.backend,
        case.difficulty,
        solved.answer.nonce,
        final_url
    );
    Ok(())
}

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

    let cases = if args.cases.is_empty() {
        vec!["fast:anubis:2".parse()?]
    } else {
        args.cases
    };

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
        .arg("2")
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

    for (index, case) in cases.iter().enumerate() {
        if let Err(error) = solve_case(index, case, &anubis_base, &solver_base) {
            eprintln!("\n--- challenge-disasterer log ---\n{}", solver.log_tail());
            eprintln!("\n--- Anubis log ---\n{}", anubis.log_tail());
            return Err(error).with_context(|| {
                format!(
                    "end-to-end case failed: {}:{}:{}",
                    case.algorithm, case.solver, case.difficulty
                )
            });
        }
    }

    println!("PASS all {} live-Anubis end-to-end case(s)", cases.len());
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
