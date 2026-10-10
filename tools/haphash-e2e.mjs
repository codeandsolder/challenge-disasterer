import { createHash } from "node:crypto";

const [serviceUrl = "http://127.0.0.1:17432", haphashUrl = "http://127.0.0.1:17431/protected"] = process.argv.slice(2);
const implementation = process.env.CHALLENGE_IMPLEMENTATION ?? "auto";
const headers = { "user-agent": "Mozilla/5.0 challenge-disasterer-haphash-e2e", accept: "text/html" };

function input(html, name) {
  const pattern = new RegExp(`<input[^>]*name=["']${name}["'][^>]*value=["']([^"']*)["']`, "i");
  const match = html.match(pattern);
  if (!match) throw new Error(`missing haphash input ${name}`);
  return match[1];
}

async function solve(prefix, difficulty) {
  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ solver: "haphash", challenge: { prefix, difficulty }, implementation }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`haphash solver failed: ${JSON.stringify(body)}`);
  if (body.implementation !== implementation) {
    throw new Error(`requested ${implementation}, got ${body.implementation}`);
  }
  if (typeof body.answer?.nonce !== "string") {
    throw new Error(`haphash nonce must be a decimal string, got ${typeof body.answer?.nonce}`);
  }
  return body;
}

function valid(prefix, nonce, difficulty) {
  const hash = createHash("sha256").update(`${prefix}${nonce}`).digest();
  let nibbles = difficulty;
  for (const byte of hash) {
    if (nibbles === 0) return true;
    if ((byte >> 4) !== 0) return false;
    nibbles--;
    if (nibbles === 0) return true;
    if ((byte & 0x0f) !== 0) return false;
    nibbles--;
  }
  return nibbles === 0;
}

function invalidNonce(prefix, start, difficulty) {
  let nonce = BigInt(start) + 1n;
  while (valid(prefix, nonce.toString(), difficulty)) nonce++;
  return nonce.toString();
}

async function localHaproxy() {
  const page = await fetch(haphashUrl, { headers, redirect: "manual" });
  if (!page.ok) throw new Error(`haphash challenge page status ${page.status}`);
  const html = await page.text();
  const ip = input(html, "ip");
  const ts = input(html, "ts");
  const difficulty = Number(input(html, "diff"));
  const host = new URL(haphashUrl).hostname;
  const prefix = `${ip};${host};${ts};`;
  const solved = await solve(prefix, difficulty);
  const nonce = String(solved.answer.nonce);
  if (!valid(prefix, nonce, difficulty)) throw new Error("solver returned invalid local haphash nonce");

  const answerUrl = new URL("/_challenge", haphashUrl);
  const bad = await fetch(answerUrl, {
    method: "POST",
    headers: { ...headers, "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ tries: invalidNonce(prefix, nonce, difficulty), ts }),
    redirect: "manual",
  });
  if (bad.status < 400) throw new Error(`HAProxy accepted corrupted haphash proof: ${bad.status}`);

  const accepted = await fetch(answerUrl, {
    method: "POST",
    headers: { ...headers, "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ tries: nonce, ts }),
    redirect: "manual",
  });
  if (!accepted.ok) throw new Error(`HAProxy rejected haphash proof: ${accepted.status}`);
  const protectedResponse = await fetch(haphashUrl, { headers, redirect: "manual" });
  const protectedBody = await protectedResponse.text();
  if (!protectedResponse.ok || !protectedBody.includes("challenge-disasterer-haphash-e2e-ok")) {
    throw new Error(`HAProxy approval did not unlock protected route: ${protectedResponse.status}`);
  }
  console.log(`haphash-haproxy: PASS implementation=${implementation} difficulty=${difficulty} solve_ms=${solved.solve_ms.toFixed(3)}`);
}

async function debianVarnish() {
  if (process.env.HAPHASH_DEBIAN_LIVE !== "1") return;
  const origin = process.env.HAPHASH_DEBIAN_URL ?? "https://bugs.debian.org/";
  const initial = await fetch(origin, { headers, redirect: "manual" });
  if (initial.status !== 302) throw new Error(`Debian challenge expected 302, got ${initial.status}`);
  const location = initial.headers.get("location");
  if (!location?.includes("/.internal/challenge.html")) throw new Error(`unexpected Debian challenge redirect: ${location}`);
  const challengeUrl = new URL(location, origin);
  const challengeResponse = await fetch(challengeUrl, { headers, redirect: "manual" });
  if (!challengeResponse.ok) throw new Error(`Debian challenge page status ${challengeResponse.status}`);
  const setCookie = challengeResponse.headers.get("set-cookie") ?? "";
  const match = setCookie.match(/(?:^|\s)pow_challenge=([^;]+)/);
  if (!match) throw new Error(`missing Debian pow_challenge cookie: ${setCookie}`);
  const challenge = match[1];
  const html = await challengeResponse.text();
  if (!html.includes("Diff=2") || !html.includes('challenge + ";" + tries')) {
    throw new Error("Debian haphash protocol shape changed; update the adapter before benchmarking");
  }
  const prefix = `${challenge};`;
  const difficulty = 2;
  const solved = await solve(prefix, difficulty);
  const nonce = String(solved.answer.nonce);
  if (!valid(prefix, nonce, difficulty)) throw new Error("solver returned invalid Debian haphash nonce");
  const badNonce = invalidNonce(prefix, nonce, difficulty);
  const bad = await fetch(origin, {
    headers: { ...headers, cookie: `pow_challenge=${challenge}; pow_nonce=${badNonce}` },
    redirect: "manual",
  });
  if (bad.status !== 302 || !bad.headers.get("location")?.includes("/.internal/challenge.html")) {
    throw new Error(`Debian accepted corrupted proof or protocol changed: ${bad.status}`);
  }
  const accepted = await fetch(origin, {
    headers: { ...headers, cookie: `pow_challenge=${challenge}; pow_nonce=${nonce}` },
    redirect: "manual",
  });
  const acceptedLocation = accepted.headers.get("location") ?? "";
  const bypassCookie = accepted.headers.get("set-cookie") ?? "";
  if (
    accepted.status >= 400 ||
    acceptedLocation.includes("/.internal/challenge.html") ||
    !bypassCookie.includes("pow_bypass=")
  ) {
    throw new Error(`Debian rejected valid haphash proof: ${accepted.status} ${acceptedLocation}`);
  }
  console.log(`haphash-debian-varnish: PASS implementation=${implementation} difficulty=${difficulty} solve_ms=${solved.solve_ms.toFixed(3)}`);
}

await localHaproxy();
await debianVarnish();
