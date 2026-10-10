const [serviceUrl = "http://127.0.0.1:17427", cerberusUrl = "http://127.0.0.1:17429/protected.iso"] = process.argv.slice(2);
const implementation = process.env.CHALLENGE_IMPLEMENTATION ?? "auto";

function attr(html, name) {
  const match = html.match(new RegExp(`${name}="([^"]*)"`));
  if (!match) throw new Error(`missing ${name}`);
  return match[1]
    .replaceAll("&quot;", '"')
    .replaceAll("&#34;", '"')
    .replaceAll("&#x22;", '"')
    .replaceAll("&amp;", "&");
}

const headers = {
  "user-agent": "Mozilla/5.0 challenge-disasterer-cerberus-e2e",
  "accept": "text/html",
};

async function issueAndSolve() {
  const page = await fetch(cerberusUrl, { headers, redirect: "manual" });
  if (!page.ok) throw new Error(`challenge page status ${page.status}`);
  const html = await page.text();
  const challenge = JSON.parse(attr(html, "x-challenge"));
  const meta = JSON.parse(attr(html, "x-meta"));
  const scriptUrl = new URL(attr(html, "src"), cerberusUrl);
  const expectedVersion = process.env.CERBERUS_EXPECTED_VERSION;
  if (!expectedVersion) throw new Error("missing CERBERUS_EXPECTED_VERSION from pinned upstream source");
  const rawVersion = scriptUrl.searchParams.get("v");
  if (rawVersion && rawVersion.replace(/^v/, "") !== expectedVersion) {
    throw new Error(`Cerberus page/source version mismatch: page=${rawVersion} source=${expectedVersion}`);
  }
  challenge.version = expectedVersion;

  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ solver: "cerberus", challenge, implementation }),
  });
  const solved = await response.json();
  if (!response.ok) throw new Error(`solver failed: ${JSON.stringify(solved)}`);
  if (solved.implementation !== implementation) throw new Error(`requested ${implementation}, got ${solved.implementation}`);
  return { challenge, meta, solved };
}

function answerParams(challenge, solved, response = solved.answer.response) {
  return new URLSearchParams({
    response,
    solution: solved.answer.solution,
    nonce: String(challenge.nonce),
    ts: String(challenge.ts),
    signature: challenge.signature,
    redir: "/protected.iso",
  });
}

async function submit(meta, params) {
  const answerUrl = new URL(`${meta.baseURL}/answer`, cerberusUrl);
  return fetch(answerUrl, {
    method: "POST",
    headers: {
      ...headers,
      "content-type": "application/x-www-form-urlencoded",
      "referer": cerberusUrl,
    },
    body: params,
    redirect: "manual",
  });
}

// Cerberus burns a challenge nonce before checking the proof. Exercise the
// negative case on its own fresh challenge so rejection cannot be a replay hit.
const negative = await issueAndSolve();
const badResponse = `${negative.solved.answer.response.slice(0, -1)}${
  negative.solved.answer.response.endsWith("0") ? "1" : "0"
}`;
const bad = await submit(
  negative.meta,
  answerParams(negative.challenge, negative.solved, badResponse),
);
if (bad.status < 400) {
  throw new Error(`corrupted Cerberus proof unexpectedly accepted: ${bad.status}`);
}

const valid = await issueAndSolve();
const accepted = await submit(
  valid.meta,
  answerParams(valid.challenge, valid.solved),
);
if (accepted.status !== 303) {
  throw new Error(
    `official Cerberus rejected proof: status=${accepted.status} body=${await accepted.text()}`,
   );
}
const cookie = accepted.headers.get("set-cookie");
if (!cookie || !cookie.includes("cerberus-auth=")) {
  throw new Error(`missing Cerberus approval cookie: ${cookie}`);
}
const cookiePair = cookie.split(";", 1)[0];
const protectedResponse = await fetch(cerberusUrl, {
  headers: { ...headers, cookie: cookiePair },
  redirect: "manual",
});
const protectedBody = await protectedResponse.text();
if (
  !protectedResponse.ok ||
  !protectedBody.includes("challenge-disasterer-cerberus-e2e-ok")
) {
  throw new Error(
    `approval cookie did not reach protected upstream: ${protectedResponse.status} ${protectedBody}`,
   );
}

console.log(
  `Cerberus: PASS version=${valid.challenge.version} solve_ms=${valid.solved.solve_ms.toFixed(3)}`,
);
