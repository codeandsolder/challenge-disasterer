#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
anubis_src="${ANUBIS_SRC:-${root}/target/anubis-live}"
anubis_bin="${ANUBIS_BIN:-${root}/target/anubis-live-bin}"
anubis_ref="${ANUBIS_REF:-main}"

if [[ ! -d "${anubis_src}/.git" ]]; then
  mkdir -p "$(dirname "${anubis_src}")"
  git clone --filter=blob:none --no-checkout https://github.com/TecharoHQ/anubis.git "${anubis_src}"
  git -C "${anubis_src}" fetch --depth=1 origin "${anubis_ref}"
  git -C "${anubis_src}" checkout --detach FETCH_HEAD
elif [[ "${ANUBIS_UPDATE:-1}" == 1 ]]; then
  git -C "${anubis_src}" fetch --depth=1 origin "${anubis_ref}"
  git -C "${anubis_src}" checkout --detach FETCH_HEAD
fi

anubis_commit="$(git -C "${anubis_src}" rev-parse HEAD)"
echo "Testing against TecharoHQ/anubis ${anubis_commit}"

if [[ "${ANUBIS_SKIP_BUILD:-0}" != 1 ]]; then
  for command in go node npm brotli zstd; do
    command -v "${command}" >/dev/null || { echo "missing Anubis build dependency: ${command}" >&2; exit 2; }
  done
  node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' || { echo "Anubis main currently requires Node >= 22 for its asset toolchain" >&2; exit 2; }
  (
    cd "${anubis_src}"
    npm ci --ignore-scripts
    RUSTUP_TOOLCHAIN="${ANUBIS_RUST_TOOLCHAIN:-1.99.0}" npm run assets
    go build -o "${anubis_bin}" ./cmd/anubis
  )
fi

solver_gpu=0
for arg in "$@"; do
  if [[ "$arg" == "--solver-gpu" ]]; then
    solver_gpu=1
    break
  fi
done
solver_build=(cargo build --locked --manifest-path "${root}/Cargo.toml")
if [[ "$solver_gpu" != 1 ]]; then
  solver_build+=(--no-default-features)
fi
"${solver_build[@]}"
cargo build --locked --manifest-path "${root}/tools/anubis-e2e/Cargo.toml"

"${root}/target/debug/challenge-disasterer-anubis-e2e" \
  --anubis-bin "${anubis_bin}" \
  --solver-bin "${root}/target/debug/challenge-disasterer" \
  --case fast:anubis:2 \
  "$@"
