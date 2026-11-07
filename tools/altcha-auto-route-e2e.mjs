import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [root, service = "http://127.0.0.1:17427", gpuMode = "enabled"] = process.argv.slice(2);
if (!root || !["enabled", "disabled"].includes(gpuMode)) {
  throw Error("usage: node altcha-auto-route-e2e.mjs ALTCHA_LIB_ROOT SERVICE_URL enabled|disabled");
}
const load = (file) => import(pathToFileURL(resolve(root, file)).href);
const v2 = await load("dist/esm/v2/index.js");
const { PasswordBuffer, signChallenge } = await load("dist/esm/v2/pow.js");
const { deriveKey: sha } = await load("dist/esm/v2/algorithms/sha.js");
const { deriveKey: pbkdf2 } = await load("dist/esm/v2/algorithms/pbkdf2.js");

const secret = "challenge-disasterer-auto-gpu-e2e";
const fixed16 = (label) => createHash("sha256").update(label).digest("hex").slice(0, 32);
const cases = [
  { name: "tiny-sha", algorithm: "SHA-256", deriveKey: sha, cost: 2, prefix: 1, counter: 0, mode: "uint32", gpu: false },
  { name: "borderline-sha", algorithm: "SHA-256", deriveKey: sha, cost: 2, prefix: 2, counter: 0, mode: "uint32", gpu: false },
  { name: "substantial-sha", algorithm: "SHA-256", deriveKey: sha, cost: 2, prefix: 17, counter: 4095, mode: "uint32", keyLength: 12, gpu: true },
  { name: "substantial-string", algorithm: "SHA-256", deriveKey: sha, cost: 2, prefix: 17, counter: 4095, mode: "string", keyLength: 12, gpu: true },
  { name: "pbkdf2-middle", algorithm: "PBKDF2/SHA-256", deriveKey: pbkdf2, cost: 16, prefix: 17, counter: 255, mode: "uint32", keyLength: 12, gpu: true },
  { name: "unsupported-sha512", algorithm: "SHA-512", deriveKey: sha, cost: 2, prefix: 3, counter: 0, mode: "uint32", gpu: false },
];
async function makeChallenge(row) {
  const params = {
    algorithm: row.algorithm, cost: row.cost, keyLength: row.keyLength ?? 32,
    nonce: fixed16("auto-gpu-e2e-nonce-"+row.name),
    salt: fixed16("auto-gpu-e2e-salt-"+row.name),
    keyPrefix: "0",
  };
  const password = new PasswordBuffer(Buffer.from(params.nonce, "hex"), row.mode).setCounter(row.counter);
  const derived = await row.deriveKey(params, Buffer.from(params.salt, "hex"), password);
  Object.assign(params, derived.parameters ?? {});
  params.keyPrefix = Buffer.from(derived.derivedKey).toString("hex").slice(0, row.prefix);
  return await signChallenge("SHA-256", params, derived.derivedKey, secret);
}
async function verify(row, challenge, implementation, expectedBackend) {
  const wireChallenge = row.mode === "string" ? {...challenge, counterMode:"string"} : challenge;
  const response = await fetch(service + "/solve", {
    method: "POST", headers: {"content-type":"application/json"},
    body: JSON.stringify({solver:"altcha",challenge:wireChallenge,implementation}),
  });
  const body = await response.json();
  if(!response.ok || body.backend !== expectedBackend) {
    throw Error(row.name + " impl=" + implementation + " expected backend " + expectedBackend + ", received " + JSON.stringify(body));
  }
  const result = await v2.verifySolution({
    challenge,
    solution: { counter: body.answer.counter, derivedKey: body.answer.derivedKey, time: body.answer.time },
    counterMode: row.mode,
    deriveKey: row.deriveKey,
    hmacSignatureSecret: secret,
  });
  if(!result.verified) throw Error(row.name + " failed official verification "+JSON.stringify(result));
  console.log(JSON.stringify({ name:row.name, requested:implementation, backend:body.backend, reported:body.implementation,
    counter:body.answer.counter, attempted_nonces:body.answer.attempted_nonces,
    solve_ms:body.solve_ms, queue_ms:body.queue_ms, official_verified:true }));
}

for (const row of cases) {
  const challenge = await makeChallenge(row);
  await verify(row, challenge, "auto", gpuMode === "enabled" && row.gpu ? "gpu" : "cpu");
  if(row.name==="substantial-sha" && gpuMode==="enabled") {
    await verify(row, challenge, "safe", "cpu");
  }
  if(row.name==="tiny-sha" && gpuMode==="enabled") {
    await verify(row, challenge, "wgpu", "gpu");
  }
}
