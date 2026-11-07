import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [altchaRoot, serviceUrl = "http://127.0.0.1:18232"] = process.argv.slice(2);
if (!altchaRoot) throw new Error("usage: node altcha-general-sha-gpu-e2e.mjs ALTCHA_LIB_ROOT SERVICE_URL");
const load = (path) => import(pathToFileURL(resolve(altchaRoot, path)).href);
const v2 = await load("dist/esm/v2/index.js");
const { PasswordBuffer, signChallenge } = await load("dist/esm/v2/pow.js");
const { deriveKey: deriveSha } = await load("dist/esm/v2/algorithms/sha.js");
const secret = "challenge-disasterer-general-sha-e2e";

const cases = [
  { salt:0, nonce:0, mode:"uint32", target:0 },
  { salt:1, nonce:7, mode:"uint32", target:127 },
  { salt:31, nonce:16, mode:"uint32", target:63 },
  { salt:55, nonce:0, mode:"uint32", target:31 },
  { salt:56, nonce:1, mode:"string", target:63 },
  { salt:63, nonce:16, mode:"string", target:127 },
  { salt:64, nonce:0, mode:"uint32", target:63 },
  { salt:65, nonce:31, mode:"string", target:63 },
  { salt:119, nonce:0, mode:"uint32", target:63 },
  { salt:120, nonce:7, mode:"string", target:63 },
  { salt:127, nonce:16, mode:"uint32", target:63 },
  { salt:128, nonce:0, mode:"string", target:127 },
  { salt:129, nonce:19, mode:"uint32", target:63 },
  { salt:1024, nonce:7, mode:"string", target:63 },
];
const hex = (length, offset) => Buffer.from(Array.from({length}, (_, i) => (offset + i * 47) & 255)).toString("hex");
for (const [index, sample] of cases.entries()) {
  const params = {
    algorithm:"SHA-256",
    cost:3,
    keyLength:12,
    salt:hex(sample.salt, 0x29),
    nonce:hex(sample.nonce, 0xe1),
    keyPrefix:"0",
  };
  const password = new PasswordBuffer(Buffer.from(params.nonce,"hex"), sample.mode).setCounter(sample.target);
  const derived = await deriveSha(params,Buffer.from(params.salt,"hex"),password);
  Object.assign(params, derived.parameters ?? {});
  params.keyPrefix = Buffer.from(derived.derivedKey).toString("hex").slice(0,17);
  const challenge = await signChallenge("SHA-256",params,derived.derivedKey,secret);
  const wire = sample.mode === "string" ? {...challenge,counterMode:"string"} : challenge;
  const req = { solver:"altcha",implementation:"wgpu",challenge:wire };
  const res = await fetch(serviceUrl + "/solve",{
    method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(req),
  });
  const body = await res.json();
  if(!res.ok || body.backend !== "gpu" || body.implementation !== "wgpu") {
    throw Error("case=" + index + " GPU failed " + res.status + ": " + JSON.stringify(body));
  }
  const solution = {
    counter:body.answer.counter,
    derivedKey:body.answer.derivedKey,
    time:body.answer.time,
  };
  const verified = await v2.verifySolution({
    challenge,solution,counterMode:sample.mode,deriveKey:deriveSha,hmacSignatureSecret:secret,
  });
  if(!verified.verified || solution.counter !== sample.target) {
    throw Error("case=" + index + " verification failed " + JSON.stringify(verified) + " " + JSON.stringify(solution));
  }
  console.log(JSON.stringify({
    case:index,...sample,solved_counter:solution.counter,
    implementation:body.implementation,
    solve_ms:body.solve_ms,
    attempted_nonces:body.answer.attempted_nonces,
    official_verified:true,
  }));
}
