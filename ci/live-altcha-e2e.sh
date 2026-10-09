#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${ALTCHA_E2E_PORT:-17427}"
ALTCHA_REF="${ALTCHA_REF:-main}"
cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$ALTCHA_REF" https://github.com/altcha-org/altcha-lib "$TMP/altcha-lib"
(
  cd "$TMP/altcha-lib"
  npm ci --silent
  npm run build --silent
)

cd "$ROOT"
cargo build --locked --release --no-default-features
"$ROOT/target/release/challenge-disasterer" --listen "127.0.0.1:$PORT" --gpu=false >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null

node "$ROOT/tools/altcha-e2e.mjs" "$TMP/altcha-lib" "http://127.0.0.1:$PORT"
