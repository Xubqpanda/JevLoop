/**
 * JevLoop · L0 词汇 —— Decision 的形状
 *
 *     Decision = State 投影 + 类型化问题 + 策略（答案 → 动作）
 *
 * 这里只有**形状**和两个构造/判定助手，没有任何一个具体的判定节点 ——
 * 具体节点是产品主张，住在 L4（`decisions.ts`）。
 *
 * 分开的理由：这一层被 L1-L5 全都引用，而具体节点只该被循环层引用。
 * 合在一起时，`budget.ts`（L1）和 `decisions.ts`（L4）的 import 长得一模一样，
 * 分层就没法用依赖表达（见 DESIGN-layers-2026-09-21.md 的 D1）。
 *
 * @module JevLoop/vocab-decision
 */

import type { AnswerMap, AnswerSet, QuestionSet } from './vocab.ts'

// ═══════════════════════════════════════════════════════════
// Decision —— 唯一的原语
//   Decision = State 投影 + 类型化问题 + 策略（答案 → 动作）
// ═══════════════════════════════════════════════════════════

/**
 * 策略规则：概率 → 动作。
 *
 * 纯代码，不碰模型。所以调阈值不需要重跑任何东西 ——
 * 这是「决策便宜」的第二层含义：不只是调用便宜，改起来也便宜。
 */
export interface PolicyRule<A> {
  /** 省略 = 兜底规则（catch-all），必须放最后 */
  when?: (a: A) => boolean
  action: string
  /** 写清楚为什么。会进 trace，出问题时能一眼看出命中了哪条 */
  reason?: string
}

export interface DecisionSpec<Ctx, Q extends QuestionSet = QuestionSet> {
  /** 唯一标识。用「域.动作」的写法 */
  id: string
  describe?: string
  /**
   * State 投影：把 agent 状态压成一段**有界**的决策帧。
   *
   * 这一步决定了判定的上限 —— **帧里没有的东西，模型判不出来**。
   * 而且上下文很短（512/1024 token），所以不能把原始对话塞进去。
   */
  state: (ctx: Ctx) => unknown
  /** 问题。可以是 ctx 的函数（选项随状态变化时必须这样写） */
  questions: Q | ((ctx: Ctx) => Q)
  policy: PolicyRule<AnswerMap<Q>>[]
  model?: string
}

const DECISION = Symbol.for('JevLoop.decision')

/**
 * 定义一个判定节点。**它只做一件事：盖一个不可枚举的标记。**
 *
 * 标记的作用是让 `isDecision()` 在运行时认得出「这是一个判定」——
 * 因为 `questions` 允许写成 `ctx => Q` 的函数，光看类型分不出来。
 * 标记用 `Symbol.for` 而不是字符串键：别让它在 `JSON.stringify` 里出现，
 * 也别让外部能伪造一个同名的普通对象混进来。
 */
export function defineDecision<Ctx, Q extends QuestionSet>(
  spec: DecisionSpec<Ctx, Q>,
): DecisionSpec<Ctx, Q> {
  Object.defineProperty(spec, DECISION, { value: true, enumerable: false })
  return spec
}

/**
 * 用 Symbol 标记而不是鸭子类型判断（"看起来像决策节点就是"）——
 * 后者太脆：一个恰好有 id/questions 的对象会被误收。
 */
export function isDecision(v: unknown): boolean {
  return !!v && typeof v === 'object' && (v as Record<symbol, unknown>)[DECISION] === true
}

/** 一次判定的完整结果 */
export interface DecisionResult<A = AnswerSet> {
  id: string
  step: number
  /** 实际发给模型的 state 帧（投影后） */
  state: unknown
  questions: QuestionSet
  answers: A
  action: string
  reason: string
  latencyMs: number
  provider: string
  model?: string
  degraded: boolean
  /** true = 无人接住，该走兜底路径了 */
  escalate: boolean
}
