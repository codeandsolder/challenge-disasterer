#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${GOAWAY_E2E_PORT:-17430}"
GOAWAY_REF="${GOAWAY_REF:-master}"

cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$GOAWAY_REF" https://github.com/WeebDataHoarder/go-away "$TMP/go-away"

cd "$ROOT"
cargo build --locked --release --no-default-features
"$ROOT/target/release/challenge-disasterer" --listen "127.0.0.1:$PORT" --gpu=false >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null

cp -a "$ROOT/tools/goaway-e2e" "$TMP/goaway-e2e"
(
  cd "$TMP/goaway-e2e"
  go mod edit -replace "git.gammaspectra.live/git/go-away=$TMP/go-away"
  go mod tidy
  go run . \
    "$TMP/go-away/embed/challenge/js-pow-sha256/runtime/runtime.wasm" \
    "http://127.0.0.1:$PORT"
)
