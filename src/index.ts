/**
 * JevLoop · 公开 API
 *
 *   import { Decider, runAgent, defineDecision, noul, choice, score } from 'jevloop';
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
} from './types.ts'
export { noul, choice, score, defineDecision, isDecision, confidenceOf } from './types.ts'

// ── 判定 ─────────────────────────────────────────────────────
export { Decider } from './decide.ts'
export type { DeciderOptions, DecideOptions } from './decide.ts'
export { resolvePolicy, gte, topGte, probGte, probLt, scoreGte, picked, probabilityOf } from './policy.ts'
export type { PolicyOutcome, PolicyWarning } from './policy.ts'
export { clip, pick, estimateTokens, validate, LIMITS } from './budget.ts'
export type { BudgetWarning, Checkpoint } from './budget.ts'

// ── 后端 ─────────────────────────────────────────────────────
export { HttpProvider, MockProvider, FallbackProvider, normalizeAnswers } from './provider.ts'
export type { HttpProviderOptions } from './provider.ts'

// ── 记账 ─────────────────────────────────────────────────────
export { Meter } from './meter.ts'
export type { MeterStats, DecisionRecord, ModelCallRecord } from './meter.ts'

// ── agent ────────────────────────────────────────────────────
export { runAgent } from './agent.ts'
export type { AgentOptions, AgentResult } from './agent.ts'
export { TOOLS, callTool, toolNames } from './tools.ts'
export type { Tool } from './tools.ts'
export { ScriptedGenerator, HttpGenerator } from './llm.ts'
export type { Generator, GenerateRequest, GenerateResult } from './llm.ts'

// ── 预设的判定节点 ───────────────────────────────────────────
export { needsTool, pickTool, gradeRisk, stepOk, isDone, canDeliver } from './decisions.ts'
export type { AgentCtx, StepRecord } from './decisions.ts'

// ── 环境 ─────────────────────────────────────────────────────
export { loadEnv } from './env.ts'
export type { LoadEnvOptions } from './env.ts'

// ── 后端解析 ─────────────────────────────────────────────────
export { resolveProvider, resolveGenerator } from './backends.ts'
export type { ProviderChoice, GeneratorChoice, FallbackNotice } from './backends.ts'
