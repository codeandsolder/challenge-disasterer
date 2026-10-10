#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
SOLVER_PORT="${CERBERUS_SOLVER_PORT:-17427}"
CHALLENGE_IMPLEMENTATION="${CHALLENGE_IMPLEMENTATION:-auto}"
CERBERUS_PORT="${CERBERUS_E2E_PORT:-17429}"
CERBERUS_REF="${CERBERUS_REF:-dist}"

cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  if [[ -n "${CADDY_PID:-}" ]]; then kill "$CADDY_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$CERBERUS_REF" https://github.com/sjtug/cerberus "$TMP/cerberus"
echo "UPSTREAM_COMMIT protocol=cerberus ref=${CERBERUS_REF:-HEAD} commit=$(git -C "$TMP/cerberus" rev-parse HEAD)"

(
  cd "$TMP"
  go run github.com/caddyserver/xcaddy/cmd/xcaddy@latest build \
    --output "$TMP/caddy" \
    --with "github.com/sjtug/cerberus=$TMP/cerberus"
)

cat >"$TMP/Caddyfile" <<EOF
{
  auto_https off
  admin off
  cerberus {
    difficulty 4
    max_pending 128
    access_per_approval 8
    block_ttl "24h"
    pending_ttl "1h"
    approval_ttl "1h"
    max_mem_usage "64MiB"
    cookie_name "cerberus-auth"
    header_name "X-Cerberus-Status"
    title "Cerberus E2E"
    mail "nobody@example.invalid"
    prefix_cfg 20 64
  }
}

http://127.0.0.1:${CERBERUS_PORT} {
  handle_path /.cerberus/* {
    cerberus_endpoint
  }

  @protected {
    path /protected.iso
  }

  cerberus @protected {
    base_url "/.cerberus"
  }

  handle /protected.iso {
    respond "challenge-disasterer-cerberus-e2e-ok"
  }
}
EOF

"$TMP/caddy" run --config "$TMP/Caddyfile" --adapter caddyfile >"$TMP/cerberus.log" 2>&1 &
CADDY_PID=$!

cd "$ROOT"
BUILD_ARGS=(--locked --release)
if [[ "$CHALLENGE_IMPLEMENTATION" != "wgpu" ]]; then BUILD_ARGS+=(--no-default-features); fi
cargo build "${BUILD_ARGS[@]}"
"$ROOT/target/release/challenge-disasterer" \
  --listen "127.0.0.1:$SOLVER_PORT" --gpu="$([[ "$CHALLENGE_IMPLEMENTATION" == "wgpu" ]] && echo true || echo false)" >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 200); do
  if curl -fsS "http://127.0.0.1:$SOLVER_PORT/stats" >/dev/null 2>&1 \
    && curl -fsS -A Mozilla "http://127.0.0.1:$CERBERUS_PORT/protected.iso" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$SOLVER_PORT/stats" >/dev/null

node "$ROOT/tools/cerberus-e2e.mjs" \
  "http://127.0.0.1:$SOLVER_PORT" \
  "http://127.0.0.1:$CERBERUS_PORT/protected.iso"
