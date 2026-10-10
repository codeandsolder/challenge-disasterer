import { pathToFileURL } from "node:url";
const implementation = process.env.CHALLENGE_IMPLEMENTATION ?? "auto";
import { resolve } from "node:path";

const [altchaRoot, serviceUrl = "http://127.0.0.1:17427"] = process.argv.slice(2);
if (!altchaRoot) throw new Error("usage: node altcha-e2e.mjs ALTCHA_ROOT [SERVICE_URL]");

const load = (path) => import(pathToFileURL(resolve(altchaRoot, path)).href);
const { createChallenge, verifySolution } = await load("dist/esm/v2/index.js");
const { deriveKey: deriveSha } = await load("dist/esm/v2/algorithms/sha.js");
const { deriveKey: derivePbkdf2 } = await load("dist/esm/v2/algorithms/pbkdf2.js");

const secret = "challenge-disasterer-altcha-e2e-secret";

async function runCase(algorithm, cost, deriveKey) {
  const challenge = await createChallenge({
    algorithm,
    cost,
    deriveKey,
    hmacSignatureSecret: secret,
    keyLength: 32,
    keyPrefix: "00",
  });

  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ solver: "altcha", challenge, implementation }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`${algorithm}: solver failed: ${JSON.stringify(body)}`);
  if (body.implementation !== implementation) throw new Error(`${algorithm}: requested ${implementation}, got ${body.implementation}`);

  const solution = {
    counter: body.answer.counter,
    derivedKey: body.answer.derivedKey,
    time: body.answer.time,
  };
  const verified = await verifySolution({
    challenge,
    solution,
    deriveKey,
    hmacSignatureSecret: secret,
  });
  if (!verified.verified) {
    throw new Error(`${algorithm}: official verifier rejected ${JSON.stringify(verified)}`);
  }

  const negative = { ...solution, derivedKey: `ff${solution.derivedKey.slice(2)}` };
  const rejected = await verifySolution({
    challenge,
    solution: negative,
    deriveKey,
    hmacSignatureSecret: secret,
  });
  if (rejected.verified) throw new Error(`${algorithm}: corrupted solution was accepted`);

  console.log(`${algorithm}: PASS counter=${solution.counter} solve_ms=${body.solve_ms.toFixed(3)}`);
}

await runCase("SHA-256", 2, deriveSha);
await runCase("PBKDF2/SHA-256", 16, derivePbkdf2);
