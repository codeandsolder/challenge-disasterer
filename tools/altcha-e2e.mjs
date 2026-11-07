import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const implementation = process.env.CHALLENGE_IMPLEMENTATION ?? "auto";
const [altchaRoot, serviceUrl = "http://127.0.0.1:17427"] = process.argv.slice(2);
if (!altchaRoot) throw new Error("usage: node altcha-e2e.mjs ALTCHA_ROOT [SERVICE_URL]");

const load = (path) => import(pathToFileURL(resolve(altchaRoot, path)).href);
const v2 = await load("dist/esm/v2/index.js");
const v1 = await load("dist/esm/v1/index.js");
const { deriveKey: deriveSha } = await load("dist/esm/v2/algorithms/sha.js");
const { deriveKey: derivePbkdf2 } = await load("dist/esm/v2/algorithms/pbkdf2.js");
const { deriveKey: deriveArgon2id } = await load("dist/esm/v2/algorithms/argon2id.js");
const { deriveKey: deriveScrypt } = await load("dist/esm/v2/algorithms/scrypt.js");

const secret = "challenge-disasterer-altcha-e2e-secret";

async function assertCapabilities() {
  const response = await fetch(`${serviceUrl}/stats`);
  const stats = await response.json();
  if (!response.ok) throw new Error(`stats failed: ${JSON.stringify(stats)}`);
  const rows = stats.solver_capabilities.filter((row) => row.solver === "altcha");
  const uint32 = rows.filter((row) => row.counter_mode === "uint32");
  const string = rows.filter((row) => row.counter_mode === "string");
  const stringSha = string.find((row) => row.algorithms.includes("sha256"));
  const stringKdf = string.find((row) => row.algorithms.includes("argon2id"));
  if (uint32.length < 2 || !stringSha || !stringKdf) {
    throw new Error("ALTCHA string-mode capabilities incomplete: " + JSON.stringify(rows));
  }
  if (string.some((row) => row.counter_max !== Number.MAX_SAFE_INTEGER)) {
    throw new Error("ALTCHA string counter ceiling mismatch: " + JSON.stringify(string));
  }
  if (implementation === "wgpu" && stringSha.implementations.join(",") !== "auto,safe,wgpu") {
    throw new Error("ALTCHA string SHA GPU capability missing: " + JSON.stringify(stringSha));
  }
  if (stringKdf.implementations.join(",") !== "auto,safe") {
    throw new Error("ALTCHA generic string KDF incorrectly advertises GPU: " + JSON.stringify(stringKdf));
  }
  if (uint32.some((row) => row.implementations.includes("reference"))) {
    throw new Error(`ALTCHA production capabilities must not expose the external reference solver: ${JSON.stringify(uint32)}`);
  }
}

await assertCapabilities();

async function callSolver(challenge, requestedImplementation = implementation, counterMode) {
  const wireChallenge = counterMode ? { ...challenge, counterMode } : challenge;
  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      solver: "altcha",
      challenge: wireChallenge,
      implementation: requestedImplementation,
    }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`solver failed: ${JSON.stringify(body)}`);
  return body;
}

async function runV2({ algorithm, cost, deriveKey, counterMode = "uint32", keyPrefix = "00", keyLength = 32, memoryCost, parallelism, requestedImplementation = implementation, expectedImplementation }) {
  const challenge = await v2.createChallenge({
    algorithm,
    cost,
    deriveKey,
    hmacSignatureSecret: secret,
    keyLength,
    keyPrefix,
    memoryCost,
    parallelism,
    counterMode,
  });

  const body = await callSolver(challenge, requestedImplementation, counterMode === "uint32" ? undefined : counterMode);
  if (expectedImplementation && body.implementation !== expectedImplementation) {
    throw new Error(`${algorithm}/${counterMode}: expected implementation ${expectedImplementation}, got ${body.implementation}`);
  }

  const solution = {
    counter: body.answer.counter,
    derivedKey: body.answer.derivedKey,
    time: body.answer.time,
  };
  const verified = await v2.verifySolution({
    challenge,
    solution,
    counterMode,
    deriveKey,
    hmacSignatureSecret: secret,
  });
  if (!verified.verified) {
    throw new Error(`${algorithm}/${counterMode}: official verifier rejected ${JSON.stringify(verified)}`);
  }

  const negative = { ...solution, derivedKey: `ff${solution.derivedKey.slice(2)}` };
  const rejected = await v2.verifySolution({
    challenge,
    solution: negative,
    counterMode,
    deriveKey,
    hmacSignatureSecret: secret,
  });
  if (rejected.verified) throw new Error(`${algorithm}/${counterMode}: corrupted solution was accepted`);

  console.log(`${algorithm}/${counterMode}: PASS impl=${body.implementation} counter=${solution.counter} solve_ms=${body.solve_ms.toFixed(3)}`);
}

async function runV1(algorithm) {
  const challenge = await v1.createChallenge({
    algorithm,
    hmacKey: secret,
    number: 17,
    maxnumber: 32,
    salt: "0123456789abcdef",
  });
  const body = await callSolver(challenge);
  const payload = {
    algorithm: challenge.algorithm,
    challenge: challenge.challenge,
    number: body.answer.number,
    salt: challenge.salt,
    signature: challenge.signature,
  };
  if (!(await v1.verifySolution(payload, secret))) {
    throw new Error(`v1 ${algorithm}: official verifier rejected ${JSON.stringify(payload)}`);
  }
  const rejected = { ...payload, number: payload.number + 1 };
  if (await v1.verifySolution(rejected, secret)) {
    throw new Error(`v1 ${algorithm}: corrupted solution was accepted`);
  }
  console.log(`ALTCHA v1 ${algorithm}: PASS impl=${body.implementation} number=${payload.number}`);
}

// ISA/GPU implementations only apply to the optimized SHA-256 family. The
// generic KDF path is deliberately auto/safe rather than pretending those
// implementation selectors accelerate algorithms they do not implement.
const optimizedOnly = ["avx2", "avx512", "sha-ni", "wgpu"].includes(implementation);

await runV2({ algorithm: "SHA-256", cost: 2, deriveKey: deriveSha, expectedImplementation: implementation });
await runV2({ algorithm: "PBKDF2/SHA-256", cost: 16, deriveKey: derivePbkdf2, expectedImplementation: implementation });

if (!optimizedOnly) {
  const genericImplementation = "altcha-generic";
  for (const algorithm of ["SHA-384", "SHA-512"]) {
    await runV2({ algorithm, cost: 2, deriveKey: deriveSha, expectedImplementation: genericImplementation });
  }
  for (const algorithm of ["PBKDF2/SHA-384", "PBKDF2/SHA-512"]) {
    await runV2({ algorithm, cost: 16, deriveKey: derivePbkdf2, expectedImplementation: genericImplementation });
  }
  await runV2({ algorithm: "ARGON2ID", cost: 1, deriveKey: deriveArgon2id, keyPrefix: "0", memoryCost: 64, parallelism: 1, expectedImplementation: genericImplementation });
  await runV2({ algorithm: "ARGON2ID", cost: 2, deriveKey: deriveArgon2id, keyLength: 96, keyPrefix: "0", memoryCost: 129, parallelism: 2, expectedImplementation: genericImplementation });
  await runV2({ algorithm: "SCRYPT", cost: 16, deriveKey: deriveScrypt, keyPrefix: "0", memoryCost: 1, parallelism: 1, expectedImplementation: genericImplementation });
}

if (implementation === "wgpu") {
  for (const [algorithm, cost, deriveKey] of [
    ["SHA-256", 2, deriveSha],
    ["PBKDF2/SHA-256", 16, derivePbkdf2],
  ]) {
    await runV2({ algorithm, cost, deriveKey, counterMode: "string", expectedImplementation: "wgpu" });
    // Odd-nibble hex prefixes are the common ALTCHA v2 shape. Exercise
    // truncated outputs and >64-bit prefix matching on real GPU hardware;
    // a long exact-prefix negative is tested directly in the solver matrix.
    for (const counterMode of ["uint32", "string"]) {
      for (const [keyPrefix, keyLength] of [["0", 32], ["abc", 12]]) {
        await runV2({
          algorithm, cost, deriveKey, counterMode, keyPrefix, keyLength,
          expectedImplementation: "wgpu",
        });
      }
    }
  }
}

if (implementation === "auto" || implementation === "safe") {
  // counterMode is orthogonal to the KDF: exercise the full current v2 KDF × string matrix.
  for (const algorithm of ["SHA-256", "SHA-384", "SHA-512"]) {
    await runV2({ algorithm, cost: 2, deriveKey: deriveSha, counterMode: "string", expectedImplementation: "altcha-string" });
  }
  for (const algorithm of ["PBKDF2/SHA-256", "PBKDF2/SHA-384", "PBKDF2/SHA-512"]) {
    await runV2({ algorithm, cost: 16, deriveKey: derivePbkdf2, counterMode: "string", expectedImplementation: "altcha-string" });
  }
  await runV2({ algorithm: "ARGON2ID", cost: 1, deriveKey: deriveArgon2id, counterMode: "string", keyPrefix: "0", memoryCost: 64, parallelism: 1, expectedImplementation: "altcha-string" });
  await runV2({ algorithm: "ARGON2ID", cost: 2, deriveKey: deriveArgon2id, counterMode: "string", keyLength: 96, keyPrefix: "0", memoryCost: 129, parallelism: 2, expectedImplementation: "altcha-string" });
  await runV2({ algorithm: "SCRYPT", cost: 16, deriveKey: deriveScrypt, counterMode: "string", keyPrefix: "0", memoryCost: 1, parallelism: 1, expectedImplementation: "altcha-string" });
}

// Still-supported v1 protocol. Its generic digest loop intentionally exposes
// only auto/safe rather than pretending ISA-specific backends exist.
if (implementation === "auto" || implementation === "safe") {
  for (const algorithm of ["SHA-1", "SHA-256", "SHA-512"]) {
    await runV1(algorithm);
  }
}
