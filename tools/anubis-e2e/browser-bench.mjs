import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function parseArgs(argv) {
  const out = { chromium: "chromium", timeoutMs: 60000 };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i]?.replace(/^--/, "");
    out[key] = argv[i + 1];
  }
  if (!out.url || !out.cases) throw new Error("--url and --cases are required");
  out.cases = JSON.parse(out.cases);
  out.timeoutMs = Number(out.timeoutMs ?? 60000);
  return out;
}

function expectedHashes(algorithm, difficulty) {
  if (algorithm === "fast" || algorithm === "slow") return 16 ** difficulty;
  return 2 ** difficulty;
}

async function chromiumEndpoint(child, timeoutMs) {
  return await new Promise((resolve, reject) => {
    let stderr = "";
    const timer = setTimeout(() => reject(new Error(`Chromium DevTools timeout: ${stderr}`)), timeoutMs);
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timer);
        resolve(match[1]);
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`Chromium exited before DevTools was ready: ${code}\n${stderr}`));
    });
  });
}

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
  }
  async open() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    });
  }
  call(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  close() { this.ws.close(); }
}

async function waitForMarker(cdp, marker, timeoutMs) {
  const started = performance.now();
  let last = "";
  while (performance.now() - started < timeoutMs) {
    try {
      const result = await cdp.call("Runtime.evaluate", {
        expression: "JSON.stringify({text: document.body?.innerText || '', href: location.href})",
        returnByValue: true,
      });
      last = result?.result?.value ?? "";
      if (last.includes(marker)) return performance.now() - started;
    } catch (_) {
      // Navigation can invalidate an evaluation; retry against the same target.
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(`browser challenge timed out; last page state=${last.slice(0, 500)}`);
}

const args = parseArgs(process.argv);
const profile = mkdtempSync(join(tmpdir(), "anubis-browser-bench-"));
const chrome = spawn(args.chromium, [
  "--headless=new",
  "--no-sandbox",
  "--disable-dev-shm-usage",
  "--disable-background-networking",
  "--disable-default-apps",
  "--disable-extensions",
  "--no-first-run",
  "--remote-debugging-port=0",
  `--user-data-dir=${profile}`,
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });

try {
  const browserWs = await chromiumEndpoint(chrome, args.timeoutMs);
  const devtoolsHttp = browserWs.replace(/^ws:/, "http:").replace(/\/devtools\/browser\/.*$/, "");
  const target = await (await fetch(`${devtoolsHttp}/json/new?about:blank`, { method: "PUT" })).json();
  const cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.open();
  await cdp.call("Page.enable");
  await cdp.call("Runtime.enable");
  await cdp.call("Network.enable");

  for (const spec of args.cases) {
    const times = [];
    for (let sample = 0; sample < spec.samples; sample++) {
      await cdp.call("Network.clearBrowserCookies");
      await cdp.call("Network.setUserAgentOverride", {
        userAgent: `challenge-disasterer-e2e/${spec.index}/${spec.algorithm}`,
      });
      const started = performance.now();
      await cdp.call("Page.navigate", { url: `${args.url}/?browser_sample=${sample}&nonce=${Math.random()}` });
      await waitForMarker(cdp, "challenge-disasterer-anubis-e2e-ok", args.timeoutMs);
      times.push(performance.now() - started);
    }
    const totalMs = times.reduce((a, b) => a + b, 0);
    const meanMs = totalMs / times.length;
    const solutionsS = times.length * 1000 / totalMs;
    const effectiveMhS = expectedHashes(spec.algorithm, spec.difficulty) * solutionsS / 1e6;
    times.sort((a, b) => a - b);
    const p50 = times[Math.floor(times.length / 2)];
    const p90 = times[Math.min(times.length - 1, Math.floor(times.length * 0.9))];
    console.log(`BENCH reference=browser algorithm=${spec.algorithm} difficulty=${spec.difficulty} n=${times.length} solutions_s=${solutionsS.toFixed(3)} effective_h_s=${(effectiveMhS * 1e6).toFixed(3)} effective_mh_s=${effectiveMhS.toFixed(3)} mean_wall_ms=${meanMs.toFixed(3)} p50_ms=${p50.toFixed(3)} p90_ms=${p90.toFixed(3)}`);
  }
  cdp.close();
} finally {
  chrome.kill("SIGTERM");
  rmSync(profile, { recursive: true, force: true });
}
