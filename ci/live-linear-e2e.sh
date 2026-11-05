#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_SOLVER_BINARY="${CHALLENGE_SOLVER_BINARY:?set CHALLENGE_SOLVER_BINARY to a built challenge-disasterer executable}"
[[ -x "$CHALLENGE_SOLVER_BINARY" ]] || { echo "solver executable is not executable: $CHALLENGE_SOLVER_BINARY" >&2; exit 2; }
TMP="$(mktemp -d)"
PORT="${LINEAR_SOLVER_PORT:-17437}"

cleanup() {
  [[ -n "${SOLVER_PID:-}" ]] && kill "$SOLVER_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

for command in curl uv; do
  command -v "$command" >/dev/null || { echo "missing Linear.pub E2E dependency: $command" >&2; exit 2; }
done

cd "$ROOT"
"$CHALLENGE_SOLVER_BINARY" \
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

uv run "$ROOT/ci/linear-e2e.py" "http://127.0.0.1:$PORT"
