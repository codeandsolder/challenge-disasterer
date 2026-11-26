use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, ensure};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::records::{AltchaRow, AnubisBenchmark, PairedHashx, Run};

fn json_lines<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(
                serde_json::from_str(&line)
                    .with_context(|| format!("{}:{}", path.display(), index + 1)),
            ),
            Err(error) => {
                Some(Err(error).with_context(|| format!("{}:{}", path.display(), index + 1)))
            }
        })
        .collect()
}

fn optional_json_lines<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    match fs::metadata(path) {
        Ok(_) => json_lines(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

fn paths(dir: &Path, prefix: &str, suffix: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))? {
        let entry = entry?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(prefix) && name.ends_with(suffix))
        {
            paths.push(entry.path());
        }
    }
    paths.sort_unstable();
    Ok(paths)
}

struct Fields<'a>(BTreeMap<&'a str, &'a str>);

impl<'a> Fields<'a> {
    fn parse(line: &'a str) -> Result<Self> {
        let mut fields = BTreeMap::new();
        for token in line.split_whitespace().skip(1) {
            let (key, value) = token
                .split_once('=')
                .with_context(|| format!("invalid benchmark token {token}"))?;
            ensure!(
                !key.is_empty() && !value.is_empty(),
                "empty benchmark token {token}"
            );
            ensure!(
                fields.insert(key, value).is_none(),
                "duplicate benchmark field {key}"
            );
        }
        Ok(Self(fields))
    }

    fn take(&mut self, key: &str) -> Result<&'a str> {
        self.0
            .remove(key)
            .with_context(|| format!("missing benchmark field {key}"))
    }

    fn number<T: FromStr>(&mut self, key: &str) -> Result<T>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        self.take(key)?
            .parse()
            .with_context(|| format!("invalid benchmark field {key}"))
    }

    fn measurement(&mut self, key: &str) -> Result<f64> {
        let value: f64 = self.number(key)?;
        ensure!(
            value.is_finite() && value >= 0.0,
            "invalid benchmark measurement {key}"
        );
        Ok(value)
    }

    fn finish(self) -> Result<()> {
        ensure!(
            self.0.is_empty(),
            "unexpected benchmark fields: {:?}",
            self.0
        );
        Ok(())
    }
}

fn paired(line: &str, host: &str) -> Result<PairedHashx> {
    let mut fields = Fields::parse(line)?;
    let result = PairedHashx {
        host: host.to_owned(),
        difficulty: fields.number("difficulty")?,
        implementation: fields.take("implementation")?.to_owned(),
        samples: fields.number("n")?,
        mean_solver_ms: fields.measurement("mean_solver_ms")?,
    };
    fields.finish()?;
    Ok(result)
}

fn benchmark(line: &str, host: &str) -> Result<AnubisBenchmark> {
    let mut fields = Fields::parse(line)?;
    let result = AnubisBenchmark {
        host: host.to_owned(),
        algorithm: fields.take("algorithm")?.to_owned(),
        solver: fields.take("solver")?.to_owned(),
        backend: fields.take("backend")?.to_owned(),
        implementation: match fields.take("implementation")? {
            "-" => None,
            implementation => Some(implementation.to_owned()),
        },
        difficulty: fields.number("difficulty")?,
        samples: fields.number("n")?,
        avg_answer_ms: fields.measurement("avg_answer_ms")?,
        effective_h_s: fields.measurement("effective_h_s")?,
        mean_solver_ms: fields.measurement("mean_solver_ms")?,
    };
    let _: f64 = fields.measurement("effective_mh_s")?;
    fields.finish()?;
    Ok(result)
}

fn logs(run: &Path, host: &str) -> Result<(Vec<PairedHashx>, Vec<AnubisBenchmark>)> {
    let mut pairs = Vec::new();
    let mut benchmarks = Vec::new();
    for path in paths(run, "live-anubis", ".log")? {
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        for (index, line) in String::from_utf8_lossy(&bytes).lines().enumerate() {
            if line.starts_with("PAIRED_HASHX ") {
                pairs.push(
                    paired(line, host)
                        .with_context(|| format!("{}:{}", path.display(), index + 1))?,
                );
            } else if line.starts_with("BENCH ") {
                benchmarks.push(
                    benchmark(line, host)
                        .with_context(|| format!("{}:{}", path.display(), index + 1))?,
                );
            }
        }
    }
    Ok((pairs, benchmarks))
}

pub fn run(dir: &Path) -> Result<Run> {
    let metadata: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)
        .with_context(|| format!("invalid metadata in {}", dir.display()))?;
    let host = metadata
        .get("hostname")
        .and_then(Value::as_str)
        .context("metadata hostname must be a string")?
        .to_owned();
    let mut solver = Vec::new();
    for path in paths(dir, "solver-", ".jsonl")? {
        solver.extend(json_lines(&path)?);
    }
    let baseline = optional_json_lines(&dir.join("preexisting-baseline.jsonl"))?;
    let mut external = optional_json_lines(&dir.join("external-baselines.jsonl"))?;
    for row in optional_json_lines(&dir.join("altcha-current.jsonl"))? {
        match row {
            AltchaRow::Solver(row) => solver.push(row),
            AltchaRow::External(row) => external.push(row),
        }
    }
    let (paired, benchmarks) = logs(dir, &host)?;
    Ok(Run {
        metadata,
        host,
        solver,
        baseline,
        external,
        paired,
        benchmarks,
    })
}
