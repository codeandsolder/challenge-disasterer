#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${CAPJS_E2E_PORT:-17428}"
CHALLENGE_IMPLEMENTATION="${CHALLENGE_IMPLEMENTATION:-auto}"
CAP_REF="${CAP_REF:-main}"

cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$CAP_REF" https://github.com/tiagozip/cap "$TMP/cap"
(
  cd "$TMP/cap/core"
  npm install --silent --ignore-scripts
)

cd "$ROOT"
BUILD_ARGS=(--locked --release)
if [[ "$CHALLENGE_IMPLEMENTATION" != "wgpu" ]]; then BUILD_ARGS+=(--no-default-features); fi
cargo build "${BUILD_ARGS[@]}"
"$ROOT/target/release/challenge-disasterer" --listen "127.0.0.1:$PORT" --gpu="$([[ "$CHALLENGE_IMPLEMENTATION" == "wgpu" ]] && echo true || echo false)" >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null

node "$ROOT/tools/capjs-e2e.mjs" "$TMP/cap" "http://127.0.0.1:$PORT"
