//! Client-side HTTP benchmarking using a Rust/Tokio/Reqwest workload driver.
//! At most the configured concurrency is ever spawned; the client timer starts
//! inside each task and includes request encoding and reading the entire response.
//! `Server-Timing` isolates application time; the difference is NOT solely
//! attributable to the server, since client scheduling and transport remain.

use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use num_traits::ToPrimitive;
use reqwest::{Client, Method, StatusCode, header};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;

#[derive(Debug, Parser)]
#[command(about = "Measure HTTP load without Python/httpx client latency artifacts")]
struct Args {
    #[arg(long)]
    base_url: String,
    /// NDJSON: `body`/`raw_body`, label, `expected_status`, JSON checks and required pointers.
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 480)]
    requests: usize,
    #[arg(long, default_value_t = 24)]
    concurrency: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 0)]
    shuffle_seed: u64,
    #[arg(long, value_enum, default_value_t = ConnectionPolicy::Reuse)]
    connections: ConnectionPolicy,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ConnectionPolicy {
    Reuse,
    Close,
}

impl ConnectionPolicy {
    const fn forces_close(self) -> bool {
        matches!(self, Self::Close)
    }
}

fn default_method() -> String {
    "POST".to_owned()
}
fn default_path() -> String {
    "/solve".to_owned()
}

#[derive(Debug, Deserialize)]
struct CorpusLine {
    label: String,
    #[serde(default = "default_method")]
    method: String,
    #[serde(default = "default_path")]
    path: String,
    #[serde(default)]
    body: Option<Value>,
    #[serde(default)]
    raw_body: Option<String>,
    expected_status: u16,
    #[serde(default)]
    check: BTreeMap<String, Value>,
    #[serde(default)]
    present: Vec<String>,
}

#[derive(Debug)]
struct Case {
    label: String,
    method: Method,
    path: String,
    request_body: Option<String>,
    expected_status: StatusCode,
    check: BTreeMap<String, Value>,
    present: Vec<String>,
}

impl TryFrom<CorpusLine> for Case {
    type Error = anyhow::Error;
    fn try_from(item: CorpusLine) -> Result<Self> {
        ensure!(!item.label.trim().is_empty(), "empty corpus label");
        ensure!(item.path.starts_with('/'), "request path must start with /");
        ensure!(
            !(item.body.is_some() && item.raw_body.is_some()),
            "body and raw_body are mutually exclusive"
        );
        let request_body = if let Some(body) = item.body {
            Some(serde_json::to_string(&body)?)
        } else {
            item.raw_body
        };
        for pointer in item.check.keys().chain(item.present.iter()) {
            ensure!(
                pointer.starts_with('/'),
                "JSON check must be a pointer: {pointer}"
            );
        }
        Ok(Self {
            label: item.label,
            method: Method::from_bytes(item.method.as_bytes())?,
            path: item.path,
            request_body,
            expected_status: StatusCode::from_u16(item.expected_status)?,
            check: item.check,
            present: item.present,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct Sample {
    index: usize,
    label: String,
    client_ms: f64,
    server_ms: f64,
    outside_ms: f64,
    pre_handler_ms: Option<f64>,
    handler_ms: Option<f64>,
    queue_ms: Option<f64>,
    solve_ms: Option<f64>,
}

fn parse_timing(input: &str, name: &str) -> Result<Option<f64>> {
    let parsed = input
        .split(',')
        .find_map(|part| {
            let (label, duration) = part.trim().split_once(";dur=")?;
            (label == name).then_some(duration)
        })
        .map(str::parse::<f64>)
        .transpose()?;
    if let Some(duration) = parsed {
        ensure!(
            duration.is_finite() && duration >= 0.0,
            "invalid {name} duration"
        );
    }
    Ok(parsed)
}

async fn execute(
    index: usize,
    client: Client,
    case: Arc<Case>,
    base_url: &str,
    connections: ConnectionPolicy,
) -> Result<Sample> {
    let started = Instant::now();
    let mut request = client.request(case.method.clone(), format!("{base_url}{}", case.path));
    if let Some(body) = &case.request_body {
        request = request
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.clone());
    }
    if connections.forces_close() {
        request = request.header(header::CONNECTION, "close");
    }
    let response = request.send().await?;
    ensure!(
        response.status() == case.expected_status,
        "{}: expected HTTP {}, got {}",
        case.label,
        case.expected_status,
        response.status()
    );
    let timing = response
        .headers()
        .get("server-timing")
        .context("response missing Server-Timing")?
        .to_str()?
        .to_owned();
    let body = response.bytes().await?;
    let client_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let response_json = if case.check.is_empty() && case.present.is_empty() {
        None
    } else {
        Some(
            serde_json::from_slice::<Value>(&body)
                .with_context(|| format!("{}: response is not JSON", case.label))?,
        )
    };
    if let Some(json) = response_json {
        for (path, expected) in &case.check {
            ensure!(
                json.pointer(path) == Some(expected),
                "{}: {path}: expected {expected}, got {:?}",
                case.label,
                json.pointer(path)
            );
        }
        for path in &case.present {
            ensure!(
                json.pointer(path).is_some(),
                "{}: missing {path}",
                case.label
            );
        }
    }
    let server_ms = parse_timing(&timing, "server")?.context("missing server duration")?;
    Ok(Sample {
        index,
        label: case.label.clone(),
        client_ms,
        server_ms,
        outside_ms: client_ms - server_ms,
        pre_handler_ms: parse_timing(&timing, "pre_handler")?,
        handler_ms: parse_timing(&timing, "handler")?,
        queue_ms: parse_timing(&timing, "queue")?,
        solve_ms: parse_timing(&timing, "solve")?,
    })
}

fn load_cases(path: &PathBuf) -> Result<Vec<Arc<Case>>> {
    let mut cases = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let content = line?;
        if content.trim().is_empty() {
            continue;
        }
        let raw: CorpusLine =
            serde_json::from_str(&content).with_context(|| format!("input line {}", index + 1))?;
        cases.push(Arc::new(
            Case::try_from(raw).with_context(|| format!("input line {}", index + 1))?,
        ));
    }
    ensure!(!cases.is_empty(), "empty input corpus");
    Ok(cases)
}

fn metric(values: &[f64]) -> Result<Value> {
    if values.is_empty() {
        return Ok(Value::Null);
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let len = ordered.len();
    Ok(
        json!({"median": ordered[(len-1)/2], "p95":ordered[(len*95/100).min(len-1)],
        "p99":ordered[(len*99/100).min(len-1)],
        "mean":ordered.iter().sum::<f64>()/len.to_f64().context("sample count conversion")?}),
    )
}

fn summarize_group(rows: &[&Sample]) -> Result<Value> {
    let client = rows.iter().map(|r| r.client_ms).collect::<Vec<_>>();
    let server = rows.iter().map(|r| r.server_ms).collect::<Vec<_>>();
    let outside = rows.iter().map(|r| r.outside_ms).collect::<Vec<_>>();
    let pre_handler = rows
        .iter()
        .filter_map(|r| r.pre_handler_ms)
        .collect::<Vec<_>>();
    let handler = rows.iter().filter_map(|r| r.handler_ms).collect::<Vec<_>>();
    let queue = rows.iter().filter_map(|r| r.queue_ms).collect::<Vec<_>>();
    let solve = rows.iter().filter_map(|r| r.solve_ms).collect::<Vec<_>>();
    Ok(json!({"n":rows.len(),"client_ms":metric(&client)?,
        "server_ms":metric(&server)?,"outside_ms":metric(&outside)?,
        "pre_handler_ms":metric(&pre_handler)?,"handler_ms":metric(&handler)?,
        "queue_ms":metric(&queue)?,"solve_ms":metric(&solve)?}))
}

fn summarize(samples: &[Sample]) -> Result<Value> {
    let mut by_label: BTreeMap<&str, Vec<&Sample>> = BTreeMap::new();
    for sample in samples {
        by_label.entry(&sample.label).or_default().push(sample);
    }
    let mut groups = BTreeMap::new();
    for (label, rows) in by_label {
        groups.insert(label.to_owned(), summarize_group(&rows)?);
    }
    // Label names are user input and can even be `total`. Keep aggregates
    // outside the label map so no label can silently corrupt total counts.
    Ok(
        json!({"total": summarize_group(&samples.iter().collect::<Vec<_>>())?,
             "by_label":groups}),
    )
}

fn shuffle(indices: &mut [usize], mut seed: u64) -> Result<()> {
    if seed == 0 {
        return Ok(());
    }
    for index in (1..indices.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let size = u64::try_from(index)?
            .checked_add(1)
            .context("shuffle index overflow")?;
        let selected = usize::try_from(seed % size)?;
        indices.swap(index, selected);
    }
    Ok(())
}

/// Keep at most `concurrency` Tokio tasks alive: a pending backlog of tens of
/// thousands of tasks would manufacture the very client-side latency this
/// benchmark is intended to avoid.
fn spawn_request(
    workers: &mut JoinSet<Result<Sample>>,
    index: usize,
    client: &Client,
    cases: &[Arc<Case>],
    url: &str,
    connections: ConnectionPolicy,
) {
    let client = client.clone();
    let case = Arc::clone(&cases[index % cases.len()]);
    let url = url.to_owned();
    workers.spawn(async move { execute(index, client, case, &url, connections).await });
}

async fn run(args: Args) -> Result<()> {
    ensure!(args.requests > 0 && args.concurrency > 0 && args.threads > 0);
    let base_url = reqwest::Url::parse(&args.base_url)?.to_string();
    ensure!(matches!(base_url.split(':').next(), Some("http" | "https")));
    let cases = load_cases(&args.input)?;
    let client = Client::builder()
        .http1_only()
        .pool_max_idle_per_host(if args.connections.forces_close() {
            0
        } else {
            args.concurrency
        })
        .build()?;
    fs::create_dir_all(&args.output_dir)?;
    let raw = args.output_dir.join("samples.jsonl");
    let summary = args.output_dir.join("summary.json");
    ensure!(
        !raw.exists() && !summary.exists(),
        "output directory already contains benchmark results"
    );
    // Reserve the raw output atomically, before sending a request, so two
    // concurrently launched runs can never overwrite each other's results.
    let mut writer = BufWriter::new(File::create_new(&raw)?);
    let mut ordering: Vec<_> = (0..args.requests).collect();
    shuffle(&mut ordering, args.shuffle_seed)?;
    let mut indices = ordering.into_iter();
    let mut workers = JoinSet::new();
    let start = Instant::now();
    let endpoint = base_url.trim_end_matches('/').to_owned();
    for index in (&mut indices).take(args.concurrency) {
        spawn_request(
            &mut workers,
            index,
            &client,
            &cases,
            &endpoint,
            args.connections,
        );
    }
    let mut samples = Vec::with_capacity(args.requests);
    while let Some(item) = workers.join_next().await {
        samples.push(item??);
        if let Some(index) = indices.next() {
            spawn_request(
                &mut workers,
                index,
                &client,
                &cases,
                &endpoint,
                args.connections,
            );
        }
    }
    samples.sort_by_key(|item| item.index);
    let elapsed = start.elapsed().as_secs_f64();
    ensure!(samples.len() == args.requests);
    for sample in &samples {
        serde_json::to_writer(&mut writer, sample)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    let aggregate = json!({"requests":args.requests,"concurrency":args.concurrency,
        "threads":args.threads,"connections":args.connections,"seed":args.shuffle_seed,
        "seconds":elapsed,"throughput_rps":args.requests.to_f64().context("request count conversion")?/elapsed,
        "groups":summarize(&samples)?});
    let mut summary_writer = BufWriter::new(File::create_new(&summary)?);
    serde_json::to_writer_pretty(&mut summary_writer, &aggregate)?;
    summary_writer.flush()?;
    println!("{}", serde_json::to_string_pretty(&aggregate)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=256).contains(&args.threads),
        "invalid Tokio runtime worker count"
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()?;
    runtime.block_on(run(args))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn valid_corpus_and_json_pointer_contract() -> Result<()> {
        let case: CorpusLine = serde_json::from_value(
            json!({"label":"x","body":{"solver":"missing","challenge":{}},
            "expected_status":400,"check":{"/error":"unknown"},"present":["/error"]}),
        )?;
        let compiled = Case::try_from(case)?;
        assert_eq!(compiled.expected_status, StatusCode::BAD_REQUEST);
        assert!(compiled.request_body.is_some());
        assert!(compiled.check.contains_key("/error"));
        Ok(())
    }
    #[test]
    fn arbitrary_labels_cannot_collide_with_totals() -> Result<()> {
        let cases = [
            Sample {
                index: 0,
                label: "total".to_owned(),
                client_ms: 1.5,
                server_ms: 0.5,
                outside_ms: 1.0,
                pre_handler_ms: None,
                handler_ms: None,
                queue_ms: None,
                solve_ms: None,
            },
            Sample {
                index: 1,
                label: "by_label".to_owned(),
                client_ms: 2.5,
                server_ms: 0.5,
                outside_ms: 2.0,
                pre_handler_ms: None,
                handler_ms: None,
                queue_ms: None,
                solve_ms: None,
            },
        ];
        let summary = summarize(&cases)?;
        assert_eq!(summary["total"]["n"], 2);
        assert_eq!(summary["by_label"]["total"]["n"], 1);
        assert_eq!(summary["by_label"]["by_label"]["n"], 1);
        assert_eq!(summary["total"]["outside_ms"]["mean"], 1.5);
        Ok(())
    }

    #[test]
    fn parse_server_timing_and_missing_metrics() -> Result<()> {
        let input = "pre_handler;dur=1.23, handler;dur=2.34, server;dur=3.58";
        assert_eq!(parse_timing(input, "server")?, Some(3.58));
        assert_eq!(parse_timing(input, "solve")?, None);
        assert!(parse_timing("server;dur=NaN", "server").is_err());
        Ok(())
    }
}
