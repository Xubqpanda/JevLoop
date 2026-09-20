/**
 * nanojev · 公开 API
 *
 *   import { Decider, runAgent, defineDecision, noul, choice, score } from "nanojev";
 *
 * 概念只有三个：Question / Decision / Provider。
 */

// ── 概念 ─────────────────────────────────────────────────────
export type {
  QuestionType,
  NoulQuestion,
  ChoiceQuestion,
  ScoreQuestion,
  Question,
  QuestionSet,
  NoulAnswer,
  ChoiceAnswer,
  ScoreAnswer,
  Answer,
  AnswerSet,
  AnswerMap,
  PolicyRule,
  DecisionSpec,
  DecisionResult,
  Provider,
  DecideRequest,
  DecideResponse,
} from "./types.ts";
export { noul, choice, score, defineDecision, isDecision, confidenceOf } from "./types.ts";

// ── 判定 ─────────────────────────────────────────────────────
export { Decider } from "./decide.ts";
export type { DeciderOptions, DecideOptions } from "./decide.ts";
export { resolvePolicy, gte, topGte, probGte, probLt, scoreGte, picked, probabilityOf } from "./policy.ts";
export type { PolicyOutcome, PolicyWarning } from "./policy.ts";
export { clip, pick, estimateTokens, validate, LIMITS } from "./budget.ts";
export type { BudgetWarning, Checkpoint } from "./budget.ts";

// ── 后端 ─────────────────────────────────────────────────────
export { HttpProvider, MockProvider, FallbackProvider, normalizeAnswers } from "./provider.ts";
export type { HttpProviderOptions } from "./provider.ts";

// ── 记账 ─────────────────────────────────────────────────────
export { Meter } from "./meter.ts";
export type { MeterStats, DecisionRecord, ModelCallRecord } from "./meter.ts";

// ── agent ────────────────────────────────────────────────────
export { runAgent } from "./agent.ts";
export type { AgentOptions, AgentResult } from "./agent.ts";
export { TOOLS, callTool, toolNames } from "./tools.ts";
export type { Tool } from "./tools.ts";
export { ScriptedGenerator, HttpGenerator } from "./llm.ts";
export type { Generator, GenerateRequest, GenerateResult } from "./llm.ts";

// ── 预设的判定节点 ───────────────────────────────────────────
export { needsTool, pickTool, gradeRisk, stepOk, isDone, canDeliver } from "./decisions.ts";
export type { AgentCtx, StepRecord } from "./decisions.ts";

// ── 一行创建默认后端 ─────────────────────────────────────────

import { HttpProvider, MockProvider, FallbackProvider } from "./provider.ts";
import type { Provider } from "./types.ts";

export interface BackendOptions {
  /** 官方 Jev，默认 https://api.typesafe.ai */
  jevUrl?: string;
  /** 本地 Laya sidecar，默认 http://127.0.0.1:7789 */
  layaUrl?: string;
  apiKey?: string;
  model?: string;
  /** 强制只用本地 */
  prefer?: "jev" | "laya" | "mock";
}

/**
 * 挑一个判定后端。
 *
 * 默认顺序：官方 Jev（有 key 时）→ 本地 Laya → Mock。
 * **永远有兜底** —— 没有 key、没有网也必须能跑起来。
 */
export function defaultProvider(opts: BackendOptions = {}): Provider {
  const mock = new MockProvider();
  if (opts.prefer === "mock") return mock;

  const laya = new HttpProvider({
    baseUrl: opts.layaUrl ?? "http://127.0.0.1:7789",
    name: "laya",
    defaultModel: opts.model ?? "typed-decisions",
    timeoutMs: 20_000,
  });

  if (opts.prefer === "laya") return new FallbackProvider([laya, mock], warnFallback);

  const key = opts.apiKey ?? process.env.TYPESAFE_API_KEY;
  const jev = new HttpProvider({
    baseUrl: opts.jevUrl ?? "https://api.typesafe.ai",
    ...(key ? { apiKey: key } : {}),
    name: "jev",
    defaultModel: opts.model ?? "jev-latest",
  });

  if (opts.prefer === "jev") return new FallbackProvider([jev, mock], warnFallback);
  // 没有 key 时不要浪费一次网络往返
  return key ? new FallbackProvider([jev, laya, mock], warnFallback) : new FallbackProvider([laya, mock], warnFallback);
}

const warnFallback = (err: unknown, from: string, to: string) =>
  console.error(`  ▲ ${from} 不可用（${(err as Error).message.slice(0, 80)}），降级到 ${to}`);
