/**
 * JevLoop · 公开 API
 *
 *   import { Decider, runAgent, defineDecision, noul, choice, score } from 'jevloop';
 *
 * 概念只有三个：Question / Decision / Provider。
 *
 * @module JevLoop/index
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
} from './vocab.ts'
export type { PolicyRule, DecisionSpec, DecisionResult } from './vocab-decision.ts'
export type { Provider, DecideRequest, DecideResponse } from './seam-provider.ts'
export { noul, choice, score, confidenceOf } from './vocab.ts'
export { defineDecision, isDecision } from './vocab-decision.ts'

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
export { Meter, formatRatio } from './meter.ts'
export { assertNever } from './util.ts'
export type { MeterStats, DecisionRecord, ModelCallRecord, AuditRecord } from './meter.ts'

// ── agent ────────────────────────────────────────────────────
export { runAgent } from './agent.ts'
export type { AgentOptions, AgentResult } from './agent.ts'

// ── 事件缝 ───────────────────────────────────────────────────
export { decisionEvent, fanOut } from './events.ts'
export type { AgentEvent, AgentObserver } from './events.ts'
export { TOOLS, callTool, toolNames, isToolName } from './tools.ts'
export type { Tool, ToolName } from './tools.ts'
export { ScriptedGenerator, HttpGenerator } from './llm.ts'
export type { Generator, GenerateRequest, GenerateResult } from './llm.ts'

// ── 预设的判定节点 ───────────────────────────────────────────
export { needsTool, pickTool, pickInput, gradeRisk, stepOk, isDone, canDeliver } from './decisions.ts'
export { hasFileOptions, unreadFiles } from './frame.ts'
export type { AgentCtx, StepRecord } from './frame.ts'

// ── 环境 ─────────────────────────────────────────────────────
export { loadEnv } from './env.ts'
export type { LoadEnvOptions, LoadEnvResult } from './env.ts'

// ── 后端解析 ─────────────────────────────────────────────────
export { resolveProvider, resolveGenerator } from './backends.ts'
export type { ProviderChoice, GeneratorChoice, FallbackNotice } from './backends.ts'
