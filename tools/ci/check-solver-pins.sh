#!/usr/bin/env bash
# Cargo allows multiple revisions of one git repo in a dependency graph.
# All CPU/GPU proof implementations must share the same solver revision.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
command -v jq >/dev/null || { echo "CI solver revision invariant requires jq" >&2; exit 1; }

declare -A revisions=()
seen=0
while IFS= read -r -d '' lock; do
  manifest="${lock%Cargo.lock}Cargo.toml"
  if [[ ! -f "$manifest" ]]; then
    echo "Tracked Cargo.lock missing Cargo.toml: $lock" >&2
    exit 1
  fi
  json="$(cargo metadata --locked --no-deps --format-version 1 --manifest-path "$manifest")" || exit 1
  sources="$(jq -r '
    .packages[].dependencies[]
    | (.source // empty)
    | select(startswith("git+https://github.com/codeandsolder/challenge-solvers"))
  ' <<<"$json")" || exit 1
  while IFS= read -r source; do
    [[ -z "$source" ]] && continue
    if [[ "$source" =~ ^git\+https://github\.com/codeandsolder/challenge-solvers\?rev=([0-9a-f]{40})$ ]]; then
      revisions["${BASH_REMATCH[1]}"]=1
      seen=$((seen + 1))
    else
      echo "Invalid solver git dependency in $manifest: $source" >&2
      exit 1
    fi
  done <<<"$sources"
done < <(git ls-files -z -- '*Cargo.lock')

if ((seen == 0 || ${#revisions[@]} != 1)); then
  echo "Solver dependency revisions diverge across Cargo workspaces: ${!revisions[*]}" >&2
  exit 1
fi
echo "Solver git dependency invariant OK: ${!revisions[*]} ($seen occurrences)"
