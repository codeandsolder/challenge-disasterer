#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_SOLVER_BINARY="${CHALLENGE_SOLVER_BINARY:?set CHALLENGE_SOLVER_BINARY to a built challenge-disasterer executable}"
[[ -x "$CHALLENGE_SOLVER_BINARY" ]] || { echo "solver executable is not executable: $CHALLENGE_SOLVER_BINARY" >&2; exit 2; }
TMP="$(mktemp -d)"
PORT="${ALTCHA_E2E_PORT:-17427}"
CHALLENGE_IMPLEMENTATION="${CHALLENGE_IMPLEMENTATION:-auto}"
ALTCHA_REF="${ALTCHA_REF:-main}"
cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$ALTCHA_REF" https://github.com/altcha-org/altcha-lib "$TMP/altcha-lib"
echo "UPSTREAM_COMMIT protocol=altcha ref=${ALTCHA_REF:-HEAD} commit=$(git -C "$TMP/altcha-lib" rev-parse HEAD)"
(
  cd "$TMP/altcha-lib"
  npm ci --silent
  npm run build --silent
)

cd "$ROOT"
"$CHALLENGE_SOLVER_BINARY" --listen "127.0.0.1:$PORT" --gpu="$([[ "$CHALLENGE_IMPLEMENTATION" == "wgpu" ]] && echo true || echo false)" >"$TMP/solver.log" 2>&1 &
SOLVER_PID=$!

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/stats" >/dev/null

node "$ROOT/tools/altcha-e2e.mjs" "$TMP/altcha-lib" "http://127.0.0.1:$PORT"

if [[ -n "${ALTCHA_BENCH_OUT:-}" ]]; then
  node "$ROOT/tools/altcha-bench.mjs" "$TMP/altcha-lib" "http://127.0.0.1:$PORT" "${ALTCHA_BENCH_SAMPLES:-1}" >"$ALTCHA_BENCH_OUT"
fi
