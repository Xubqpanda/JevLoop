#!/usr/bin/env node
/**
 * nanojev · demo
 *
 *   node --experimental-strip-types examples/demo.ts
 *   node --experimental-strip-types examples/demo.ts --laya     # 用本地 Laya
 *   node --experimental-strip-types examples/demo.ts --jev      # 用官方 Jev（需 TYPESAFE_API_KEY）
 *
 * 零依赖、零 key、离线可跑 —— 默认用 Mock 后端把 loop 走通。
 *
 * 重点看最后那行汇总：**判定 : 模型** 的比值。
 */

import { mkdir, writeFile, rm, mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { Decider, Meter, runAgent, ScriptedGenerator, HttpProvider, FallbackProvider } from "../src/index.ts";
import { RuleJudge } from "./rule-judge.ts";

const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(`--${f}`);
const prefer: "jev" | "laya" | "mock" | "rule" | undefined = has("jev") ? "jev" : has("laya") ? "laya" : has("mock") ? "mock" : has("rule") ? "rule" : undefined;

const C = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  magenta: (s: string) => `\x1b[35m${s}\x1b[0m`,
};

// ── 造一个工作目录 ───────────────────────────────────────────

const cwd = await mkdtemp(join(tmpdir(), "nanojev-"));
await mkdir(cwd, { recursive: true });
await writeFile(
  join(cwd, "invoice.ts"),
  `export interface Invoice {
  id: string;
  amount: number;
  paid: boolean;
}

/** 计算未付款总额 */
export function outstanding(invoices: Invoice[]): number {
  return invoices
    .filter((i) => !i.paid)
    .reduce((sum, i) => sum + i.amount, 0);
}
`,
  "utf8",
);
await writeFile(join(cwd, "notes.md"), "# 说明\n\n这是一个演示目录。\n", "utf8");

// ── 组装 ─────────────────────────────────────────────────────

// 判定后端链：官方 Jev（有 key 时）→ 本地 Laya → 规则判定器（保证 demo 一定能跑完）
function buildProvider() {
  if (prefer === "rule") return new RuleJudge();
  const rule = new RuleJudge();
  const laya = new HttpProvider({
    baseUrl: process.env.JEVOS_SIDECAR ?? "http://127.0.0.1:7789",
    name: "laya",
    defaultModel: "typed-decisions",
    timeoutMs: 20_000,
  });
  const key = process.env.TYPESAFE_API_KEY;
  const jev = new HttpProvider({
    baseUrl: "https://api.typesafe.ai",
    ...(key ? { apiKey: key } : {}),
    name: "jev",
    defaultModel: "jev-latest",
  });
  if (prefer === "jev") return new FallbackProvider([jev, rule], warn);
  if (prefer === "laya") return new FallbackProvider([laya, rule], warn);

  // ★ 默认用规则判定器，**不自动去连本地 sidecar**。
  //   原因：开源的 Laya checkpoint 零样本做"选哪个工具"这类新任务能力不足
  //   （实测：正确和错误的选项概率都挤在 0.55–0.66，没有区分度）。
  //   自动连上去只会让 `npm run demo` 输出一堆看不懂的升级。
  //   想看真实判定：显式加 --laya 或 --jev。
  return rule;
}
// 降级只提示一次 —— 每次判定都打一遍会把 trace 淹掉
let warned = false;
const warn = (err: unknown, from: string, to: string) => {
  if (warned) return;
  warned = true;
  console.error(`  ▲ ${from} 不可用（${(err as Error).message.slice(0, 60)}），改用 ${to}`);
};

const provider = buildProvider();
const meter = new Meter();
const decider = new Decider({ provider, meter });
const generator = new ScriptedGenerator();

const TASK = "列出工作目录里的文件，读取其中的 TypeScript 文件，说明它定义了哪些函数。";

console.log(C.bold("\nnanojev · demo"));
console.log(C.dim(`  task      : ${TASK}`));
console.log(C.dim(`  cwd       : ${cwd}`));
console.log(C.dim(`  判定后端  : ${provider.name}`));
console.log(C.dim(`  生成后端  : ${generator.name}（脚本化，让 demo 离线可跑）`));
console.log("");
console.log(C.bold("  ── loop trace ──────────────────────────────────────────"));

const result = await runAgent({
  task: TASK,
  cwd,
  decider,
  generator,
  maxSteps: 8,
  onTrace: (line) => console.log(C.dim(line)),
});

// ── 输出 ─────────────────────────────────────────────────────

console.log("");
console.log(C.bold("  ── 逐条明细 ────────────────────────────────────────────"));
console.log(C.dim(meter.trace()));

console.log("");
console.log(C.bold("  ── 结果 ────────────────────────────────────────────────"));
console.log(`  halt      : ${C.cyan(result.halt)}`);
console.log(`  steps     : ${result.steps}`);
console.log("");
console.log(C.dim("  " + result.answer.split("\n").join("\n  ").slice(0, 600)));

const s = meter.stats;
console.log("");
console.log(C.bold("  ── 记账 ────────────────────────────────────────────────"));
console.log(
  `  判定  ${C.green(String(s.decisions).padStart(3))} 次   ${C.dim(`${s.decisionMs}ms（均 ${s.avgDecisionMs}ms）`)}`,
);
console.log(
  `  模型  ${C.magenta(String(s.modelCalls).padStart(3))} 次   ${C.dim(`${s.modelMs}ms`)}`,
);
console.log("");
console.log(
  `  ${C.bold("判定 : 模型 =")} ${C.bold(C.green(s.modelCalls ? s.ratio.toFixed(1) : String(s.decisions)) + " : 1")}` +
    C.dim(`   判定耗时只占 ${(s.decisionShare * 100).toFixed(1)}%`),
);
console.log("");

await rm(cwd, { recursive: true, force: true });
