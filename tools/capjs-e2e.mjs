import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const implementation = process.env.CHALLENGE_IMPLEMENTATION ?? "auto";
const [capRoot, serviceUrl = "http://127.0.0.1:17428"] = process.argv.slice(2);
if (!capRoot) throw new Error("usage: node capjs-e2e.mjs CAP_ROOT [SERVICE_URL]");

const core = await import(pathToFileURL(resolve(capRoot, "core/src/index.js")).href);
const { generateChallenge, generateRswKeypair, validateChallenge } = core;
const secret = "challenge-disasterer-capjs-e2e-secret-32bytes";
const scope = "challenge-disasterer-e2e";

async function assertCapabilities() {
  const response = await fetch(`${serviceUrl}/stats`);
  const stats = await response.json();
  if (!response.ok) throw new Error(`stats failed: ${JSON.stringify(stats)}`);
  const rows = stats.solver_capabilities.filter((row) => row.solver === "capjs");
  for (const algorithm of ["rsw", "hashwx"]) {
    const row = rows.find((candidate) => candidate.algorithms.includes(algorithm));
    if (!row) throw new Error(`missing Cap.js capability row for ${algorithm}`);
    if (JSON.stringify(row.implementations) !== JSON.stringify(["auto"])) {
      throw new Error(`Cap.js ${algorithm} must advertise only the selectable auto backend: ${JSON.stringify(row)}`);
    }
  }
}

await assertCapabilities();

async function solve(challenge) {
  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ solver: "capjs", implementation, challenge }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`solver failed: ${JSON.stringify(body)}`);
  return body;
}

async function verify(issued, solutions) {
  return validateChallenge(secret, { token: issued.token, solutions }, { scope });
}

function corrupt(protocol, solution) {
  if (protocol === "sha256-pow") return { nonce: "not-a-number" };
  if (protocol === "rsw") return { y: "0" };
  if (protocol === "hashwx") return { nonce: "18446744073709551616" };
  throw new Error(`cannot corrupt unsupported protocol ${protocol}`);
}

async function runLegacy(challengeCount, challengeDifficulty, challengeSize) {
  const issued = await generateChallenge(secret, {
    challengeCount,
    challengeDifficulty,
    challengeSize,
    instrumentation: false,
    scope,
  });
  const body = await solve({ challenge: issued.challenge, token: issued.token });
  if (body.implementation !== implementation) {
    throw new Error(`requested ${implementation}, got ${body.implementation}`);
  }
  const solutions = body.answer.solutions;
  if (!Array.isArray(solutions) || solutions.length !== challengeCount) {
    throw new Error(`bad legacy solution count: ${JSON.stringify(body.answer)}`);
  }
  const verified = await verify(issued, solutions);
  if (!verified.success) throw new Error(`legacy verifier rejected: ${JSON.stringify(verified)}`);
  const corrupted = solutions.slice();
  corrupted[0] += 1;
  const rejected = await verify(issued, corrupted);
  if (rejected.success) throw new Error("legacy verifier accepted corrupted solutions");
  console.log(`Cap.js format1: PASS count=${challengeCount} difficulty=${challengeDifficulty} solve_ms=${body.solve_ms.toFixed(3)}`);
}

async function runV2(name, options) {
  const issued = await generateChallenge(secret, { format: 2, scope, ...options });
  const body = await solve(issued);
  const solutions = body.answer.solutions;
  if (!Array.isArray(solutions) || solutions.length !== issued.challenges.length) {
    throw new Error(`bad ${name} solution count: ${JSON.stringify(body.answer)}`);
  }
  const verified = await verify(issued, solutions);
  if (!verified.success) throw new Error(`${name} verifier rejected: ${JSON.stringify(verified)}`);

  const corrupted = structuredClone(solutions);
  corrupted[0] = corrupt(issued.challenges[0].protocol, corrupted[0]);
  const rejected = await verify(issued, corrupted);
  if (rejected.success) throw new Error(`${name} verifier accepted corrupted solutions`);
  console.log(`Cap.js format2 ${name}: PASS implementation=${body.implementation} solve_ms=${body.solve_ms.toFixed(3)}`);
}

await runLegacy(4, 2, 24);
await runLegacy(3, 3, 32);

const keypair = generateRswKeypair(256);
await runV2("sha256-pow", {
  protocols: ["sha256-pow"],
  challengeCount: 3,
  challengeDifficulty: 3,
  challengeSize: 24,
});
await runV2("rsw", { protocols: ["rsw"], keypair, t: 512 });
await runV2("hashwx", {
  protocols: ["hashwx"],
  hashwxDifficulty: 64,
  hashwxChallengeCount: 2,
  hashwxNoncesPerHash: 16,
});
await runV2("mixed", {
  protocols: ["sha256-pow", "rsw", "hashwx"],
  keypair,
  t: 256,
  challengeCount: 1,
  challengeDifficulty: 2,
  challengeSize: 16,
  hashwxDifficulty: 32,
  hashwxChallengeCount: 1,
  hashwxNoncesPerHash: 16,
});

if (process.env.CAPJS_PRODUCTION_SHAPE === "1") {
  // Keep this out of the hosted backend matrix: the point is to validate the
  // production workload shape once, not to multiply expensive HashWX work by
  // every implementation selector. Omitting algorithm knobs intentionally tracks
  // current Cap defaults instead of copying them into this repository.
  const productionKeypair = generateRswKeypair(2048);
  await runV2("rsw-production-defaults", {
    protocols: ["rsw"],
    keypair: productionKeypair,
  });
  await runV2("hashwx-production-defaults", {
    protocols: ["hashwx"],
  });
}
