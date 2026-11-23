//! Local authoritative-verifier fixture orchestration. No production-site tests.
mod process;

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use serde_json::json;

use process::{GroupChild, Supervisor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Implementation {
    Auto,
    Safe,
    Avx2,
    #[value(name = "sha-ni")]
    ShaNi,
    Avx512,
    Wgpu,
    Reference,
}

impl Implementation {
    const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Avx2 => "avx2",
            Self::ShaNi => "sha-ni",
            Self::Avx512 => "avx512",
            Self::Wgpu => "wgpu",
            Self::Reference => "reference",
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchmarkProfile {
    Standard,
    ProductionArgon2,
    Argon2Only,
    ProductionArgon2Only,
}

impl BenchmarkProfile {
    const fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::ProductionArgon2 => "production-argon2",
            Self::Argon2Only => "argon2-only",
            Self::ProductionArgon2Only => "production-argon2-only",
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Source checkout containing the verifier scripts.
    #[arg(long)]
    root: PathBuf,
    /// Already-built solver for this revision.
    #[arg(long)]
    solver_bin: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    implementation: Implementation,
    #[arg(long)]
    cpu_calibration_hashes: Option<NonZeroU64>,
    #[arg(long)]
    gpu_calibration_hashes: Option<NonZeroU64>,
    #[arg(long, default_value = "90")]
    startup_seconds: NonZeroU64,
    #[arg(long, default_value = "900")]
    command_seconds: NonZeroU64,
    #[command(subcommand)]
    fixture: Fixture,
}

#[derive(Subcommand, Debug)]
enum Fixture {
    Altcha {
        #[arg(long, default_value = "main")]
        upstream_ref: String,
        #[arg(long)]
        benchmark_output: Option<PathBuf>,
        #[arg(long, default_value = "1", requires = "benchmark_output")]
        benchmark_samples: NonZeroU32,
        #[arg(
            long,
            value_enum,
            default_value = "standard",
            requires = "benchmark_output"
        )]
        benchmark_profile: BenchmarkProfile,
    },
    Capjs {
        #[arg(long, default_value = "main")]
        upstream_ref: String,
        #[arg(long)]
        production_shape: bool,
    },
    Cerberus {
        #[arg(long, default_value = "dist")]
        upstream_ref: String,
    },
    Goaway {
        #[arg(long, default_value = "master")]
        upstream_ref: String,
    },
    Mcaptcha {
        #[arg(long, default_value = "HEAD")]
        upstream_ref: String,
        /// Already-built official verifier adapter.
        #[arg(long)]
        verifier_bin: PathBuf,
    },
}

struct Run<'a> {
    args: &'a Args,
    root: PathBuf,
    solver: PathBuf,
    scratch: &'a Path,
    supervisor: Supervisor,
    client: Client,
}

fn executable(path: &Path) -> Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = path
        .canonicalize()
        .with_context(|| format!("resolve {}", path.display()))?;
    let metadata = path.metadata()?;
    ensure!(
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
        "not an executable file: {}",
        path.display()
    );
    Ok(path)
}

fn log_tail(path: &Path) -> String {
    let read = || -> std::io::Result<String> {
        let mut file = File::open(path)?;
        let start = file.metadata()?.len().saturating_sub(16384);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(16384).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    };
    read().unwrap_or_default()
}

fn running(child: &mut GroupChild, log: &Path) -> Result<()> {
    if let Some(status) = child.poll()? {
        bail!("fixture process exited with {status}: {}", log_tail(log));
    }
    Ok(())
}

fn parse_address(log: &str) -> Result<Option<SocketAddr>> {
    let Some(value) = log
        .lines()
        .find_map(|line| line.strip_prefix("LISTEN_ADDRESS "))
    else {
        return Ok(None);
    };
    // A partially written final line is not a complete readiness record.
    if !log.ends_with('\n') {
        return Ok(None);
    }
    let address: SocketAddr = value.parse().context("invalid solver readiness address")?;
    ensure!(
        address.ip().is_loopback() && address.port() != 0,
        "solver did not bind a loopback endpoint"
    );
    Ok(Some(address))
}

impl Run<'_> {
    fn clone(&self, protocol: &str, url: &str, upstream_ref: &str) -> Result<PathBuf> {
        let checkout = self.scratch.join(protocol);
        // Arguments stay separate; no shell evaluation of selected refs.
        let mut command = Command::new("git");
        command.args(["clone", "--quiet", "--depth", "1"]);
        if upstream_ref != "HEAD" {
            command.args(["--branch", upstream_ref]);
        }
        self.supervisor
            .run(command.arg("--").arg(url).arg(&checkout))?;
        let commit = self.supervisor.capture(
            Command::new("git")
                .arg("-C")
                .arg(&checkout)
                .args(["rev-parse", "HEAD"]),
        )?;
        println!(
            "UPSTREAM_COMMIT protocol={protocol} ref={upstream_ref} commit={}",
            commit.trim()
        );
        Ok(checkout)
    }

    fn wait_http(&self, child: &mut GroupChild, log: &Path, url: &str) -> Result<()> {
        let start = Instant::now();
        loop {
            running(child, log)?;
            if self
                .client
                .get(url)
                .send()
                .is_ok_and(|response| response.status().is_success())
            {
                running(child, log)?;
                return Ok(());
            }
            self.supervisor
                .check(start, Duration::from_secs(self.args.startup_seconds.get()))
                .with_context(|| format!("waiting for {url}: {}", log_tail(log)))?;
            thread::sleep(Duration::from_millis(50));
        }
    }

    // A challenge verifier deliberately returns 401/403/redirects until a
    // valid proof is supplied. Its readiness is a listening TCP socket,
    // not an HTTP 2xx response; the fixture validates the real protocol.
    fn wait_verifier(&self, child: &mut GroupChild, log: &Path, address: SocketAddr) -> Result<()> {
        let start = Instant::now();
        loop {
            running(child, log)?;
            if TcpStream::connect_timeout(&address, Duration::from_millis(250)).is_ok() {
                running(child, log)?;
                return Ok(());
            }
            self.supervisor
                .check(start, Duration::from_secs(self.args.startup_seconds.get()))
                .with_context(|| format!("waiting for verifier at {address}: {}", log_tail(log)))?;
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn start_solver(&self) -> Result<(GroupChild, String)> {
        let log = self.scratch.join("solver.log");
        let address_file = self.scratch.join("solver-address.txt");
        let gpu = self.args.implementation == Implementation::Wgpu;
        let altcha = matches!(self.args.fixture, Fixture::Altcha { .. });
        let cpu_hashes = self
            .args
            .cpu_calibration_hashes
            .map_or(if altcha { 8192 } else { 2_000_000 }, NonZeroU64::get);
        let gpu_hashes = self
            .args
            .gpu_calibration_hashes
            .map_or(if altcha { 8192 } else { 64_000_000 }, NonZeroU64::get);
        let mut command = Command::new(&self.solver);
        command
            .current_dir(&self.root)
            .args(["--listen", "127.0.0.1:0", "--print-listen-address"])
            .arg(format!("--gpu={gpu}"))
            .arg("--cpu-calibration-hashes")
            .arg(cpu_hashes.to_string());
        if gpu {
            command
                .arg("--gpu-calibration-hashes")
                .arg(gpu_hashes.to_string());
        }
        let mut child = GroupChild::spawn(
            command
                .stdout(File::create(&address_file)?)
                .stderr(File::create(&log)?)
                .stdin(Stdio::null()),
        )?;
        let start = Instant::now();
        let address = loop {
            running(&mut child, &log)?;
            if let Some(address) = parse_address(&fs::read_to_string(&address_file)?)? {
                break address;
            }
            self.supervisor
                .check(start, Duration::from_secs(self.args.startup_seconds.get()))
                .with_context(|| format!("waiting for solver readiness: {}", log_tail(&log)))?;
            thread::sleep(Duration::from_millis(50));
        };
        let url = format!("http://{address}");
        self.wait_http(&mut child, &log, &format!("{url}/stats"))?;
        Ok((child, url))
    }

    fn node(&self, script: &str) -> Command {
        let mut command = Command::new("node");
        command
            .current_dir(&self.root)
            .arg(self.root.join("tools").join(script));
        command
    }

    fn altcha(
        &self,
        upstream_ref: &str,
        benchmark_output: Option<&Path>,
        samples: NonZeroU32,
        profile: BenchmarkProfile,
    ) -> Result<()> {
        let upstream = self.clone(
            "altcha",
            "https://github.com/altcha-org/altcha-lib",
            upstream_ref,
        )?;
        self.supervisor.run(
            Command::new("npm")
                .current_dir(&upstream)
                .args(["ci", "--silent"]),
        )?;
        self.supervisor.run(
            Command::new("npm")
                .current_dir(&upstream)
                .args(["run", "build", "--silent"]),
        )?;
        let (_solver, url) = self.start_solver()?;
        self.supervisor.run(
            self.node("altcha-e2e.mjs")
                .arg(&upstream)
                .arg(&url)
                .arg(self.args.implementation.label()),
        )?;
        if let Some(output) = benchmark_output {
            self.supervisor.run(
                self.node("altcha-bench.mjs")
                    .arg(&upstream)
                    .arg(&url)
                    .arg(samples.to_string())
                    .arg(profile.label())
                    .stdout(File::create(output)?),
            )?;
        }
        Ok(())
    }

    fn capjs(&self, upstream_ref: &str, production_shape: bool) -> Result<()> {
        let upstream = self.clone("capjs", "https://github.com/tiagozip/cap", upstream_ref)?;
        self.supervisor.run(
            Command::new("npm")
                .current_dir(upstream.join("core"))
                .args(["install", "--silent", "--ignore-scripts"]),
        )?;
        let (_solver, url) = self.start_solver()?;
        self.supervisor.run(
            self.node("capjs-e2e.mjs")
                .arg(&upstream)
                .arg(&url)
                .arg(self.args.implementation.label())
                .arg(if production_shape { "true" } else { "false" }),
        )
    }

    fn goaway(&self, upstream_ref: &str) -> Result<()> {
        let upstream = self.clone(
            "goaway",
            "https://github.com/WeebDataHoarder/go-away",
            upstream_ref,
        )?;
        let harness = self.scratch.join("goaway-e2e");
        fs::create_dir(&harness)?;
        for file in ["go.mod", "main.go"] {
            fs::copy(
                self.root.join("tools/goaway-e2e").join(file),
                harness.join(file),
            )?;
        }
        self.supervisor.run(
            Command::new("go")
                .current_dir(&harness)
                .args(["mod", "edit", "-replace"])
                .arg(format!(
                    "git.gammaspectra.live/git/go-away={}",
                    upstream.display()
                )),
        )?;
        self.supervisor.run(
            Command::new("go")
                .current_dir(&harness)
                .args(["mod", "tidy"]),
        )?;
        let (_solver, url) = self.start_solver()?;
        self.supervisor.run(
            Command::new("go")
                .current_dir(&harness)
                .args(["run", "."])
                .arg(upstream.join("embed/challenge/js-pow-sha256/runtime/runtime.wasm"))
                .arg(&url)
                .arg(self.args.implementation.label()),
        )
    }

    fn cerberus(&self, upstream_ref: &str) -> Result<()> {
        let upstream = self.clone(
            "cerberus",
            "https://github.com/sjtug/cerberus",
            upstream_ref,
        )?;
        let source = fs::read_to_string(upstream.join("core/const.go"))?;
        let version = source
            .lines()
            .find_map(|line| {
                let (key, value) = line.trim().split_once('=')?;
                (key.trim() == "Version")
                    .then(|| {
                        value
                            .trim()
                            .strip_prefix("\"v")?
                            .split_once('"')
                            .map(|(version, _)| version)
                    })
                    .flatten()
            })
            .context("could not derive Cerberus protocol version from checked-out core/const.go")?;
        ensure!(
            !version.is_empty()
                && version
                    .split('.')
                    .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())),
            "invalid Cerberus source version"
        );
        println!("UPSTREAM_PROTOCOL_VERSION protocol=cerberus version={version}");
        let caddy = self.scratch.join("caddy");
        self.supervisor.run(
            Command::new("go")
                .current_dir(self.scratch)
                .args([
                    "run",
                    "github.com/caddyserver/xcaddy/cmd/xcaddy@latest",
                    "build",
                    "--output",
                ])
                .arg(&caddy)
                .arg("--with")
                .arg(format!("github.com/sjtug/cerberus={}", upstream.display())),
        )?;
        // Caddy is an external verifier without a readiness-address protocol.
        // Use a fresh kernel-assigned port and require the owned child to stay alive.
        let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?
            .local_addr()?
            .port();
        let config = self.scratch.join("Caddyfile");
        fs::write(
            &config,
            include_str!("../fixtures/Caddyfile").replace("@PORT@", &port.to_string()),
        )?;
        let log = self.scratch.join("cerberus.log");
        let mut verifier = GroupChild::logged(
            Command::new(caddy)
                .current_dir(self.scratch)
                .args(["run", "--config"])
                .arg(config)
                .args(["--adapter", "caddyfile"]),
            &log,
        )?;
        let url = format!("http://127.0.0.1:{port}/protected.iso");
        self.wait_verifier(&mut verifier, &log, (Ipv4Addr::LOCALHOST, port).into())?;
        let (_solver, solver_url) = self.start_solver()?;
        self.supervisor.run(
            self.node("cerberus-e2e.mjs")
                .arg(solver_url)
                .arg(url)
                .arg(self.args.implementation.label())
                .arg(version),
        )
    }

    fn mcaptcha(&self, upstream_ref: &str, verifier: &Path) -> Result<()> {
        let verifier = executable(verifier)?;
        let upstream = self.clone(
            "mcaptcha",
            "https://github.com/mCaptcha/mCaptcha",
            upstream_ref,
        )?;
        let manifest = fs::read_to_string(upstream.join("Cargo.toml"))?;
        ensure!(
            manifest.contains("mcaptcha_pow_sha256 = \"0.5"),
            "current mCaptcha no longer declares mcaptcha_pow_sha256 0.5; update this authoritative harness"
        );
        let (_solver, url) = self.start_solver()?;
        let request = self.scratch.join("request.json");
        let path = self.scratch.join("response.json");
        fs::write(
            &request,
            serde_json::to_vec(&json!({
                "solver": "mcaptcha",
                "implementation": self.args.implementation.label(),
                "challenge": {
                    "config": {
                        "string": "challenge-disasterer-mcaptcha-e2e",
                        "difficulty_factor": 10000,
                        "salt": "mcaptcha-e2e-salt-long-enough-and-stable"
                    },
                    "key": "challenge-disasterer-e2e"
                }
            }))?,
        )?;
        let mut data = std::ffi::OsString::from("@");
        data.push(&request);
        self.supervisor.run(
            Command::new("curl")
                .args([
                    "--noproxy",
                    "*",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--header",
                    "content-type: application/json",
                    "--data-binary",
                ])
                .arg(data)
                .arg(format!("{url}/solve"))
                .stdout(File::create(&path)?),
        )?;
        self.supervisor.run(
            Command::new(verifier)
                .arg("--implementation")
                .arg(self.args.implementation.label())
                .stdin(Stdio::from(File::open(path)?)),
        )
    }

    fn execute(&self) -> Result<()> {
        match &self.args.fixture {
            Fixture::Altcha {
                upstream_ref,
                benchmark_output,
                benchmark_samples,
                benchmark_profile,
            } => self.altcha(
                upstream_ref,
                benchmark_output.as_deref(),
                *benchmark_samples,
                *benchmark_profile,
            ),
            Fixture::Capjs {
                upstream_ref,
                production_shape,
            } => self.capjs(upstream_ref, *production_shape),
            Fixture::Cerberus { upstream_ref } => self.cerberus(upstream_ref),
            Fixture::Goaway { upstream_ref } => self.goaway(upstream_ref),
            Fixture::Mcaptcha {
                upstream_ref,
                verifier_bin,
            } => self.mcaptcha(upstream_ref, verifier_bin),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args.root.canonicalize()?;
    ensure!(
        root.join("tools/altcha-e2e.mjs").is_file(),
        "invalid fixture source root"
    );
    let solver = executable(&args.solver_bin)?;
    let scratch = tempfile::tempdir()?;
    let run = Run {
        args: &args,
        root,
        solver,
        scratch: scratch.path(),
        supervisor: Supervisor::new(Duration::from_secs(args.command_seconds.get()))?,
        client: Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("Mozilla/5.0 fixture readiness")
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(1))
            .build()?,
    };
    run.execute()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_protocol_requires_complete_loopback_record() -> Result<()> {
        assert_eq!(
            parse_address("startup\nLISTEN_ADDRESS 127.0.0.1:12345")?,
            None
        );
        assert_eq!(
            parse_address("startup\nLISTEN_ADDRESS 127.0.0.1:12345\n")?,
            Some("127.0.0.1:12345".parse()?)
        );
        assert!(parse_address("LISTEN_ADDRESS 0.0.0.0:12345\n").is_err());
        assert!(parse_address("LISTEN_ADDRESS 127.0.0.1:0\n").is_err());
        Ok(())
    }

    #[test]
    fn options_reject_invalid_inputs() {
        let base = ["fixture", "--root", ".", "--solver-bin", "solver"];
        assert!(Args::try_parse_from(base.into_iter().chain(["altcha"])).is_ok());
        assert!(
            Args::try_parse_from(
                base.into_iter()
                    .chain(["--implementation", "invalid", "altcha"])
            )
            .is_err()
        );
        assert!(
            Args::try_parse_from(base.into_iter().chain(["--startup-seconds", "0", "altcha"]))
                .is_err()
        );
        assert!(Args::try_parse_from(base.into_iter().chain(["mcaptcha"])).is_err());
        assert!(
            Args::try_parse_from(
                base.into_iter()
                    .chain(["altcha", "--benchmark-samples", "2"])
            )
            .is_err()
        );
    }
}
