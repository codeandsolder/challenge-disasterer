import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const [capRoot, serviceUrl = "http://127.0.0.1:17428"] = process.argv.slice(2);
if (!capRoot) throw new Error("usage: node capjs-e2e.mjs CAP_ROOT [SERVICE_URL]");

const core = await import(pathToFileURL(resolve(capRoot, "core/src/index.js")).href);
const { generateChallenge, validateChallenge } = core;
const secret = "challenge-disasterer-capjs-e2e-secret-32bytes";

async function runCase(challengeCount, challengeDifficulty, challengeSize) {
  const issued = await generateChallenge(secret, {
    challengeCount,
    challengeDifficulty,
    challengeSize,
    instrumentation: false,
    scope: "challenge-disasterer-e2e",
  });

  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      solver: "capjs",
      challenge: {
        challenge: issued.challenge,
        token: issued.token,
      },
    }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`solver failed: ${JSON.stringify(body)}`);

  const solutions = body.answer.solutions;
  if (!Array.isArray(solutions) || solutions.length !== challengeCount) {
    throw new Error(`bad solution count: ${JSON.stringify(body.answer)}`);
  }

  const verified = await validateChallenge(
    secret,
    { token: issued.token, solutions },
    { scope: "challenge-disasterer-e2e" },
  );
  if (!verified.success) {
    throw new Error(`official verifier rejected: ${JSON.stringify(verified)}`);
  }

  const corrupted = solutions.slice();
  corrupted[0] += 1;
  const rejected = await validateChallenge(
    secret,
    { token: issued.token, solutions: corrupted },
    { scope: "challenge-disasterer-e2e" },
  );
  if (rejected.success) throw new Error("official verifier accepted corrupted solutions");

  console.log(
    `Cap.js: PASS count=${challengeCount} difficulty=${challengeDifficulty} ` +
      `solve_ms=${body.solve_ms.toFixed(3)}`,
  );
}

await runCase(4, 2, 24);
await runCase(3, 3, 32);
