#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
HAPHASH_PORT="${HAPHASH_E2E_PORT:-17431}"
SOLVER_PORT="${HAPHASH_SOLVER_PORT:-17432}"
CHALLENGE_IMPLEMENTATION="${CHALLENGE_IMPLEMENTATION:-auto}"
HAPHASH_REF="${HAPHASH_REF:-main}"

cleanup() {
  [[ -n "${SOLVER_PID:-}" ]] && kill "$SOLVER_PID" 2>/dev/null || true
  [[ -n "${HAPROXY_PID:-}" ]] && kill "$HAPROXY_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

for command in git haproxy node curl; do
  command -v "$command" >/dev/null || { echo "missing haphash E2E dependency: $command" >&2; exit 2; }
done

git clone --quiet --depth 1 --branch "$HAPHASH_REF" https://github.com/dgl/haphash.git "$TMP/haphash"
echo "UPSTREAM_COMMIT protocol=haphash ref=${HAPHASH_REF} commit=$(git -C "$TMP/haphash" rev-parse HEAD)"

cat >"$TMP/haproxy.cfg" <<EOF

global
  expose-experimental-directives

defaults
  mode http
  timeout connect 2s
  timeout client 10s
  timeout server 10s

frontend www
  bind 127.0.0.1:${HAPHASH_PORT}
  stick-table type ipv6 size 1m expire 2d store gpt(2)
  http-request track-sc0 src
  http-request normalize-uri path-merge-slashes
  http-request normalize-uri path-strip-dot
  http-request normalize-uri path-strip-dotdot
  acl challenge_path path /_challenge
  acl protected_path path /protected
  acl protected_ua hdr(User-Agent) -m beg Mozilla/
  acl protected acl(protected_path,protected_ua)
  acl accepted sc_get_gpt(1,0) gt 0
  http-request set-var(txn.diff) int(4)
  http-request return status 200 content-type "text/html; charset=UTF-8" hdr "Cache-control" "max-age=0, no-cache" lf-file $TMP/haphash/challenge.html if protected !challenge_path !accepted
  http-request return status 200 content-type text/plain string "challenge-disasterer-haphash-e2e-ok" if protected accepted
  http-request return status 404 content-type text/plain string "not found" if !challenge_path
  use_backend challenge if challenge_path

backend challenge
  mode http
  option http-buffer-request
  http-request track-sc0 src table www
  acl challenge_req method POST
  http-request set-var(txn.tries) req.body_param(tries)
  http-request set-var(txn.ts) req.body_param(ts)
  http-request set-var(txn.host) hdr(Host),host_only
  http-request set-var(txn.hash) src,concat(;,txn.host,),concat(;,txn.ts,),concat(;,txn.tries),sha2,hex
  acl ts_recent date,neg,add(txn.ts) ge -60
  acl hash_good var(txn.hash),ltrim(0),length,add(txn.diff) -m int le 64
  http-request sc-set-gpt(1,0) 1 if challenge_req ts_recent hash_good
  http-request return status 200 if challenge_req hash_good
  http-request return status 400 content-type "text/html; charset=UTF-8" hdr "Cache-control" "max-age=0" string "Bad request" if !challenge_req OR !hash_good
EOF
haproxy -c -f "$TMP/haproxy.cfg" >/dev/null
haproxy -db -f "$TMP/haproxy.cfg" >"$TMP/haproxy.log" 2>&1 &
HAPROXY_PID=$!

cd "$ROOT"
BUILD_ARGS=(--locked --release)
if [[ "$CHALLENGE_IMPLEMENTATION" != "wgpu" ]]; then BUILD_ARGS+=(--no-default-features); fi
cargo build "${BUILD_ARGS[@]}"
"$ROOT/target/release/challenge-disasterer" \
  --listen "127.0.0.1:$SOLVER_PORT" \
  --gpu="$([[ "$CHALLENGE_IMPLEMENTATION" == "wgpu" ]] && echo true || echo false)" \
  >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 200); do
  if curl -fsS "http://127.0.0.1:$SOLVER_PORT/stats" >/dev/null 2>&1 \
    && curl -fsS -A Mozilla/5.0 "http://127.0.0.1:$HAPHASH_PORT/protected" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$SOLVER_PORT/stats" >/dev/null

node "$ROOT/tools/haphash-e2e.mjs" \
  "http://127.0.0.1:$SOLVER_PORT" \
  "http://127.0.0.1:$HAPHASH_PORT/protected"
