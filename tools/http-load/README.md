# HTTP load diagnostics (Rust)

A native Rust/Tokio/Reqwest load client that measures the **entire** HTTP
request/response body time, instead of folding Python/httpx event-loop overhead
into the server's apparent latency. Designed for the server's `/solve`
`Server-Timing` metrics. No subprocesses, servers, or GPU programs are started
by the client: launch the exact service build independently and keep its source
SHA alongside benchmark output.

```sh
cargo +1.99 run --release --locked -p challenge-disasterer-http-load -- \
  --base-url http://127.0.0.1:17417 \
  --input tools/http-load/fixtures/no-solver.ndjson \
  --output-dir /home/jan/benchmarks/challenge-disasterer/http-load-example \
  --requests 480 --concurrency 24 --threads 4 \
  --connections reuse --shuffle-seed 123
```

The input is newline-delimited JSON. A record supplies a nonempty `label`,
optional `method` (defaults to POST), optional `path` (defaults to `/solve`),
**exactly one** `body` (any JSON object) or `raw_body` (for intentionally
malformed JSON), mandatory numeric `expected_status`, and optional `check`
(JSON Pointer → expected exact value) and `present` (list of required JSON
Pointers). Every response must pass status and JSON checks; the benchmark
fails rather than measuring incorrectly routed or incomplete proof results.
For successful GPU challenges, include checks such as `"/backend":"gpu"`
and the expected solution count, plus `present` pointers into the answer.

The tool repeats the supplied corpus round-robin, then optionally shuffles
its request order deterministically. At most the configured number of requests are in flight at any instant;
when one completes, the client starts the next. This avoids the artificial
Tokio-task backlog that would contaminate a latency benchmark. The client
timer starts inside each active task (after scheduling) and includes request
serialization, any connection establishment, and the full response body. Setting
`--connections close` forces fresh HTTP/1.1 connections; `reuse` uses an idle
pool sized for the requested concurrency. Full response bodies are consumed
and checked before stopping the client stopwatch. All replies must expose a
valid `Server-Timing: server;dur=...` value; other server timing phases such as
`pre_handler`, `handler`, `queue`, `solve` may be absent and are represented
as `null`, not zero.

Outputs are `samples.jsonl` (one request/response timing record each,
including original request index and protocol label) and `summary.json`
(aggregate statistics under `groups.total`, arbitrary user labels under
`groups.by_label.<label>`, median/p95/p99, throughput, connection policy,
concurrency and thread count). Labels such as `total` are valid and never
collide with the aggregate. Existing results are **never overwritten**.
All `*_ms` values are durations in milliseconds. `outside_ms = client_ms -
server_ms` includes the client's event-loop scheduling, connection creation,
transport, and processing before Axum's application boundary as well as
response delivery. **It does not mean time blocked in the server's Tokio
executor.** Cross-client benchmark comparisons are diagnostic, not reliable
substitutes for source-matched server throughput A/B trials.

The historical 2026-11-15 physical experiment and all client libraries'
raw outputs are recorded in the Notion [HTTP ingress latency worklog](https://app.notion.com/p/419f74be90208101bdb3de5367e4d68f).
