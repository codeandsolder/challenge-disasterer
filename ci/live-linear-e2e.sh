#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${LINEAR_SOLVER_PORT:-17437}"

cleanup() {
  [[ -n "${SOLVER_PID:-}" ]] && kill "$SOLVER_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

for command in cargo curl python3; do
  command -v "$command" >/dev/null || { echo "missing Linear.pub E2E dependency: $command" >&2; exit 2; }
done

cd "$ROOT"
cargo build --locked --release --no-default-features
"$ROOT/target/release/challenge-disasterer" \
  --listen "127.0.0.1:$PORT" \
  --gpu=false \
  --cpu-calibration-hashes 10000 \
  >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null

python3 "$ROOT/tools/linear-e2e.py" "http://127.0.0.1:$PORT"
