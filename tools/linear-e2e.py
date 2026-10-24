#!/usr/bin/env python3
"""Exercise challenge-disasterer against Linear.pub's live SHA-1 gate."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request

LINEAR_URL = "https://linear.pub/"
USER_AGENT = "Mozilla/5.0 challenge-disasterer-live-compat"
SEED_RE = re.compile(r"[0-9A-Fa-f]{40}")


def fetch(cookie: str | None = None) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT}
    if cookie is not None:
        headers["Cookie"] = cookie
    request = urllib.request.Request(LINEAR_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def challenge_seed(page: bytes) -> str:
    match = SEED_RE.search(page.decode(errors="replace"))
    if match is None:
        raise RuntimeError("could not find the 40-hex Linear.pub seed in live challenge JS")
    return match.group(0)


def fetch_challenge() -> str | None:
    status, page = fetch()
    if status == 200:
        return None
    if status != 503:
        raise RuntimeError(f"unexpected Linear.pub challenge status: {status}")
    return challenge_seed(page)


def solve(solver_url: str, seed: str) -> dict[str, object]:
    body = json.dumps(
        {
            "solver": "linear",
            "implementation": "auto",
            "challenge": {"seed": seed},
        }
    ).encode()
    request = urllib.request.Request(
        f"{solver_url}/solve",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def target_matches(seed: str, nonce: int) -> bool:
    digest = hashlib.sha1(f"{seed}{nonce}".encode(), usedforsecurity=False).digest()
    offset = int(seed[0], 16)
    return digest[offset : offset + 2] == bytes((0xB0, 0x0B))


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} SOLVER_URL", file=sys.stderr)
        return 2
    solver_url = sys.argv[1]

    reject_seed = fetch_challenge()
    if reject_seed is None:
        print("LINEAR_LIVE_SKIP challenge is not currently active")
        return 0

    # Check rejection on one issuance. Linear may rotate challenge state after a
    # bad proof, so acceptance deliberately uses a fresh issuance below.
    bad_nonce = 0
    while target_matches(reject_seed, bad_nonce):
        bad_nonce += 1
    bad_cookie = f"yX3={reject_seed}{bad_nonce}"
    bad_status, _ = fetch(bad_cookie)
    if bad_status != 503:
        raise RuntimeError(f"Linear.pub accepted deliberately corrupted proof: HTTP {bad_status}")

    seed = fetch_challenge()
    if seed is None:
        raise RuntimeError("Linear.pub stopped challenging after rejected proof")
    response = solve(solver_url, seed)
    answer = response["answer"]
    if not isinstance(answer, dict):
        raise RuntimeError("solver answer is not an object")
    nonce = int(answer["nonce"])
    cookie = str(answer["cookie"])
    if not target_matches(seed, nonce):
        raise RuntimeError("solver returned a nonce that does not satisfy the live JS condition")

    good_status, page = fetch(cookie)
    if good_status != 200:
        raise RuntimeError(f"Linear.pub rejected solver proof: HTTP {good_status}")
    if b"Linear.pub" not in page:
        raise RuntimeError("accepted response does not look like the Linear.pub storefront")

    print("LINEAR_LIVE_PASS", f"seed={seed}", f"nonce={nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
