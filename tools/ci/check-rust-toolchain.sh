#!/usr/bin/env bash
set -euo pipefail

# The repository pin is canonical. Shared CI currently passes a toolchain
# input which sets RUSTUP_TOOLCHAIN, so reject silently stale workflow inputs.
required=$(awk -F '"' '/^[[:space:]]*channel[[:space:]]*=/ { print $2; exit }' rust-toolchain.toml)
if [[ -z "$required" ]]; then
  echo 'rust-toolchain.toml is missing its toolchain.channel' >&2
  exit 1
fi
actual=${RUSTUP_TOOLCHAIN:-}
if [[ "$actual" != "$required" ]]; then
  echo "Rust toolchain drift: repo=$required; CI RUSTUP_TOOLCHAIN=${actual:-unset}" >&2
  exit 1
fi
