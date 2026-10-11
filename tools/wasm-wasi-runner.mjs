import fs from "node:fs";
import { WASI } from "node:wasi";

const [, , wasm, ...rest] = process.argv;
if (!wasm) {
  throw new Error("missing WASI module path");
}

const wasi = new WASI({
  version: "preview1",
  args: [wasm, ...rest],
  env: process.env,
  preopens: { ".": process.cwd() },
});
const module = await WebAssembly.compile(fs.readFileSync(wasm));
const instance = await WebAssembly.instantiate(module, {
  wasi_snapshot_preview1: wasi.wasiImport,
});
wasi.start(instance);
