import { createHash } from "node:crypto";
import os from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const [altchaRoot, serviceUrl = "http://127.0.0.1:17427", samplesArg = "1"] = process.argv.slice(2);
if (!altchaRoot) throw new Error("usage: node altcha-bench.mjs ALTCHA_ROOT [SERVICE_URL] [SAMPLES]");
const samples = Number(samplesArg);
if (!Number.isInteger(samples) || samples < 1) throw new Error("SAMPLES must be a positive integer");

const load = (path) => import(pathToFileURL(resolve(altchaRoot, path)).href);
const v2 = await load("dist/esm/v2/index.js");
const v1 = await load("dist/esm/v1/index.js");
const { PasswordBuffer, signChallenge } = await load("dist/esm/v2/pow.js");
const { deriveKey: deriveSha } = await load("dist/esm/v2/algorithms/sha.js");
const { deriveKey: derivePbkdf2 } = await load("dist/esm/v2/algorithms/pbkdf2.js");
const { deriveKey: deriveArgon2id } = await load("dist/esm/v2/algorithms/argon2id.js");
const { deriveKey: deriveScrypt } = await load("dist/esm/v2/algorithms/scrypt.js");

const commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const host = os.hostname();
const cpuModel = os.cpus()[0]?.model ?? "unknown";
const upstreamCommit = execFileSync("git", ["-C", altchaRoot, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const benchmarkSecret = "challenge-disasterer-altcha-benchmark-secret";

const configs = [
  { algorithm: "SHA-256", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 4 },
  { algorithm: "SHA-384", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 16 },
  { algorithm: "SHA-512", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 16 },
  { algorithm: "PBKDF2/SHA-256", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 4 },
  { algorithm: "PBKDF2/SHA-384", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 16 },
  { algorithm: "PBKDF2/SHA-512", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 16 },
  { algorithm: "ARGON2ID", cost: 1, deriveKey: deriveArgon2id, counter: 63, prefixBytes: 16, memoryCost: 64, parallelism: 1 },
  { algorithm: "SCRYPT", cost: 16, deriveKey: deriveScrypt, counter: 255, prefixBytes: 16, memoryCost: 1, parallelism: 1 },
  { algorithm: "SHA-256", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 16, counterMode: "string" },
  { algorithm: "SHA-384", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 16, counterMode: "string" },
  { algorithm: "SHA-512", cost: 2, deriveKey: deriveSha, counter: 262_143, prefixBytes: 16, counterMode: "string" },
  { algorithm: "PBKDF2/SHA-256", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 16, counterMode: "string" },
  { algorithm: "PBKDF2/SHA-384", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 16, counterMode: "string" },
  { algorithm: "PBKDF2/SHA-512", cost: 16, deriveKey: derivePbkdf2, counter: 16_383, prefixBytes: 16, counterMode: "string" },
  { algorithm: "ARGON2ID", cost: 1, deriveKey: deriveArgon2id, counter: 63, prefixBytes: 16, memoryCost: 64, parallelism: 1, counterMode: "string" },
  { algorithm: "SCRYPT", cost: 16, deriveKey: deriveScrypt, counter: 255, prefixBytes: 16, memoryCost: 1, parallelism: 1, counterMode: "string" },
];

function variant(config) {
  return `v2-${config.algorithm.toLowerCase().replaceAll("/", "-")}-${config.counterMode ?? "uint32"}`;
}
function emit(row) {
  process.stdout.write(`${JSON.stringify(row)}\n`);
}

function hex(bytes) {
  return Buffer.from(bytes).toString("hex");
}
function fixed16(label) {
  return createHash("sha256").update(label).digest("hex").slice(0, 32);
}

async function makeChallenge(config, sample) {
  const counterMode = config.counterMode ?? "uint32";
  const parameters = {
    algorithm: config.algorithm,
    nonce: fixed16(`challenge-disasterer-altcha-bench:nonce:${config.algorithm}:${counterMode}:${sample}`),
    salt: fixed16(`challenge-disasterer-altcha-bench:salt:${config.algorithm}:${counterMode}:${sample}`),
    cost: config.cost,
    keyLength: 32,
    keyPrefix: "00",
    ...(config.memoryCost === undefined ? {} : { memoryCost: config.memoryCost }),
    ...(config.parallelism === undefined ? {} : { parallelism: config.parallelism }),
    data: { sample },
  };
  const password = new PasswordBuffer(Buffer.from(parameters.nonce, "hex"), counterMode).setCounter(config.counter);
  const result = await config.deriveKey(parameters, Buffer.from(parameters.salt, "hex"), password);
  if (result.parameters) Object.assign(parameters, result.parameters);
  parameters.keyPrefix = hex(result.derivedKey.slice(0, config.prefixBytes));
  return signChallenge("SHA-256", parameters, result.derivedKey, benchmarkSecret);
}

async function callService(challenge, counterMode) {
  const wireChallenge = counterMode === "string" ? { ...challenge, counterMode } : challenge;
  const response = await fetch(`${serviceUrl}/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ solver: "altcha", challenge: wireChallenge, implementation: "auto" }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`service failed: ${JSON.stringify(body)}`);
  return body;
}

for (let sample = 0; sample < samples; sample++) {
  for (const config of configs) {
    const counterMode = config.counterMode ?? "uint32";
    const challenge = await makeChallenge(config, sample);
    const expectedWork = config.counter + 1;

    const refStart = performance.now();
    const reference = await v2.solveChallenge({
      challenge,
      counterMode,
      deriveKey: config.deriveKey,
      timeout: 0,
    });
    const refNs = Math.round((performance.now() - refStart) * 1e6);
    if (!reference || reference.counter !== config.counter) {
      throw new Error(`${variant(config)} reference found ${reference?.counter}, expected ${config.counter}`);
    }

    const body = await callService(challenge, counterMode);
    if (Number(body.answer.counter) !== config.counter) {
      throw new Error(`${variant(config)} service found ${body.answer.counter}, expected ${config.counter}`);
    }
    if (Number(body.answer.attempted_nonces) !== expectedWork) {
      throw new Error(`${variant(config)} service attempted ${body.answer.attempted_nonces}, expected ${expectedWork}`);
    }
    const verified = await v2.verifySolution({
      challenge,
      solution: {
        counter: Number(body.answer.counter),
        derivedKey: body.answer.derivedKey,
        time: body.answer.time,
      },
      counterMode,
      deriveKey: config.deriveKey,
      hmacSignatureSecret: benchmarkSecret,
    });
    if (!verified.verified) throw new Error(`${variant(config)} official verifier rejected service result`);

    const common = {
      schema: 1,
      protocol: "altcha",
      variant: variant(config),
      sample,
      attempted_work: expectedWork,
      parameters: {
        algorithm: config.algorithm,
        cost: config.cost,
        key_length: 32,
        key_prefix_bytes: config.prefixBytes,
        counter_mode: counterMode,
        target_counter: config.counter,
        memory_cost: config.memoryCost ?? null,
        parallelism: config.parallelism ?? null,
      },
    };
    const serviceNs = Math.max(1, Math.round(body.solve_ms * 1e6));
    emit({
      ...common,
      row_kind: "solver",
      phase: "throughput",
      timestamp_unix_ms: Date.now(),
      host,
      cpu_model: cpuModel,
      commit,
      implementation: body.implementation,
      accelerator: "cpu",
      solve_ns: serviceNs,
      work_per_second: expectedWork * 1e9 / serviceNs,
      found: true,
      nonce: config.counter,
      target_condition_ok: true,
      gpu_adapter: null,
    });
    emit({
      ...common,
      row_kind: "external",
      baseline: `altcha-org/altcha-lib@${upstreamCommit.slice(0, 12)}`,
      implementation: "official-js",
      accelerator: "cpu",
      solve_ns: refNs,
      work_per_second: expectedWork * 1e9 / refNs,
      comparison_scope: "protocol-reference",
    });
  }
}

const v1Secret = "challenge-disasterer-altcha-bench-v1";
for (let sample = 0; sample < samples; sample++) {
  for (const algorithm of ["SHA-1", "SHA-256", "SHA-512"]) {
    const targetNumber = 16_383;
    const challenge = await v1.createChallenge({
      algorithm,
      hmacKey: v1Secret,
      number: targetNumber,
      maxnumber: targetNumber,
      salt: `challenge-disasterer-v1-bench-${sample}`,
    });
    const refStart = performance.now();
    const reference = await v1.solveChallenge(
      challenge.challenge,
      challenge.salt,
      algorithm,
      targetNumber,
      0,
    ).promise;
    const refNs = Math.round((performance.now() - refStart) * 1e6);
    if (!reference || reference.number !== targetNumber) {
      throw new Error(`v1-${algorithm} reference found ${reference?.number}, expected ${targetNumber}`);
    }
    const body = await callService(challenge, "uint32");
    if (Number(body.answer.number) !== targetNumber) {
      throw new Error(`v1-${algorithm} service found ${body.answer.number}, expected ${targetNumber}`);
    }
    const payload = {
      algorithm: challenge.algorithm,
      challenge: challenge.challenge,
      number: Number(body.answer.number),
      salt: challenge.salt,
      signature: challenge.signature,
    };
    if (!(await v1.verifySolution(payload, v1Secret))) {
      throw new Error(`v1-${algorithm} official verifier rejected service result`);
    }
    const expectedWork = targetNumber + 1;
    const name = `v1-${algorithm.toLowerCase()}`;
    const common = {
      schema: 1,
      protocol: "altcha",
      variant: name,
      sample,
      attempted_work: expectedWork,
      parameters: { algorithm, target_number: targetNumber },
    };
    const serviceNs = Math.max(1, Math.round(body.solve_ms * 1e6));
    emit({
      ...common,
      row_kind: "solver",
      phase: "throughput",
      timestamp_unix_ms: Date.now(),
      host,
      cpu_model: cpuModel,
      commit,
      implementation: body.implementation,
      accelerator: "cpu",
      solve_ns: serviceNs,
      work_per_second: expectedWork * 1e9 / serviceNs,
      found: true,
      nonce: targetNumber,
      target_condition_ok: true,
      gpu_adapter: null,
    });
    emit({
      ...common,
      row_kind: "external",
      baseline: `altcha-org/altcha-lib@${upstreamCommit.slice(0, 12)}`,
      implementation: "official-js-v1",
      accelerator: "cpu",
      solve_ns: refNs,
      work_per_second: expectedWork * 1e9 / refNs,
      comparison_scope: "protocol-reference",
    });
  }
}
