import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [root, service = "http://127.0.0.1:18242"] = process.argv.slice(2);
if (!root) throw Error("usage: node altcha-pbkdf2-general-gpu-e2e.mjs ALTCHA_LIB_ROOT SERVICE_URL");
const load = file => import(pathToFileURL(resolve(root, file)).href);
const v2 = await load("dist/esm/v2/index.js");
const { PasswordBuffer, signChallenge } = await load("dist/esm/v2/pow.js");
const { deriveKey: derivePbkdf2 } = await load("dist/esm/v2/algorithms/pbkdf2.js");
const secret = "challenge-disasterer-variable-hmac-pbkdf2-e2e";
const cases = [
  { salt:0, nonce:0, mode:"uint32", target:0, cost:8 },
  { salt:1, nonce:1, mode:"string", target:3, cost:8 },
  { salt:16, nonce:16, mode:"uint32", target:31, cost:8 },
  { salt:51, nonce:60, mode:"uint32", target:31, cost:8 },
  { salt:52, nonce:61, mode:"uint32", target:63, cost:8 },
  { salt:59, nonce:63, mode:"string", target:63, cost:8 },
  { salt:60, nonce:64, mode:"string", target:31, cost:8 },
  { salt:63, nonce:65, mode:"uint32", target:31, cost:8 },
  { salt:64, nonce:127, mode:"string", target:63, cost:8 },
  { salt:115, nonce:31, mode:"uint32", target:31, cost:8 },
  { salt:116, nonce:48, mode:"string", target:63, cost:8 },
  { salt:119, nonce:49, mode:"string", target:31, cost:8 },
  { salt:120, nonce:128, mode:"uint32", target:31, cost:8 },
  { salt:127, nonce:7, mode:"uint32", target:31, cost:8 },
  { salt:128, nonce:0, mode:"string", target:63, cost:8 },
  { salt:1024, nonce:65, mode:"string", target:31, cost:3 },
  { salt:4096, nonce:64, mode:"uint32", target:0, cost:1 },
];
const hex = (len, seed) => Buffer.from(Array.from({length:len},(_,i)=>(seed+i*47)&255)).toString("hex");
for (const [index, sample] of cases.entries()) {
  const params = {
    algorithm:"PBKDF2/SHA-256",
    cost:sample.cost,
    keyLength:12,
    salt:hex(sample.salt,13),
    nonce:hex(sample.nonce,197),
    keyPrefix:"0",
  };
  const password = new PasswordBuffer(Buffer.from(params.nonce,"hex"),sample.mode).setCounter(sample.target);
  const derived = await derivePbkdf2(params,Buffer.from(params.salt,"hex"),password);
  Object.assign(params,derived.parameters ?? {});
  params.keyPrefix = Buffer.from(derived.derivedKey).toString("hex").slice(0,17);
  const challenge = await signChallenge("SHA-256",params,derived.derivedKey,secret);
  const wire = sample.mode === "string" ? {...challenge,counterMode:"string"} : challenge;
  const implementations = [0,2,4,8,15,16].includes(index) ? ["wgpu","auto"] : ["wgpu"];
  for (const implementation of implementations) {
    const response = await fetch(service+"/solve",{
      method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({solver:"altcha",challenge:wire,implementation}),
    });
    const body = await response.json();
    if(!response.ok || body.backend!=="gpu" || body.implementation!=="wgpu") {
      throw Error("case="+index+" "+implementation+" unexpected GPU response "+
        response.status+" "+JSON.stringify(body));
    }
    const solution = {counter:body.answer.counter,derivedKey:body.answer.derivedKey,time:body.answer.time};
    const verdict = await v2.verifySolution({
      challenge,solution,counterMode:sample.mode,deriveKey:derivePbkdf2,hmacSignatureSecret:secret,
    });
    if(!verdict.verified || solution.counter!==sample.target) {
      throw Error("case="+index+" "+implementation+" invalid official PBKDF2 solution "+
        JSON.stringify(verdict)+" "+JSON.stringify(solution));
    }
    console.log(JSON.stringify({
      case:index,...sample,requested:implementation,backend:body.backend,
      actual:body.implementation,counter:solution.counter,
      solve_ms:body.solve_ms,official_verified:true,
    }));
  }
}
