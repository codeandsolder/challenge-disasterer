#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${MCAPTCHA_E2E_PORT:-17431}"
CHALLENGE_IMPLEMENTATION="${CHALLENGE_IMPLEMENTATION:-auto}"
cleanup() {
  if [[ -n "${SOLVER_PID:-}" ]]; then kill "$SOLVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

git clone --quiet --depth 1 https://github.com/mCaptcha/mCaptcha "$TMP/mcaptcha"
echo "UPSTREAM_COMMIT protocol=mcaptcha ref=${MCAPTCHA_REF:-HEAD} commit=$(git -C "$TMP/mcaptcha" rev-parse HEAD)"
if ! rg -q 'mcaptcha_pow_sha256 = "0\.5"' "$TMP/mcaptcha/Cargo.toml"; then
  echo "current mCaptcha no longer declares mcaptcha_pow_sha256 0.5; update this authoritative harness" >&2
  exit 1
fi

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

cat >"$TMP/request.json" <<JSON
{
  "solver": "mcaptcha",
  "implementation": "$CHALLENGE_IMPLEMENTATION",
  "challenge": {
    "config": {
      "string": "challenge-disasterer-mcaptcha-e2e",
      "difficulty_factor": 10000,
      "salt": "mcaptcha-e2e-salt-long-enough-and-stable"
    },
    "key": "challenge-disasterer-e2e"
  }
}
JSON

curl -fsS \
  -H "content-type: application/json" \
  --data-binary @"$TMP/request.json" \
  "http://127.0.0.1:$PORT/solve" >"$TMP/response.json"

cargo run --quiet --locked --manifest-path "$ROOT/tools/mcaptcha-e2e/Cargo.toml" <"$TMP/response.json"
