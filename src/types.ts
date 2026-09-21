/**
 * nanojev · 类型契约
 *
 * 整个框架的概念只有三个：
 *
 *     Question  一个问题（noul / choice / score）
 *     Decision  State 投影 + 问题 + 策略（答案 → 动作）
 *     Provider  谁来判定（Jev / Laya / Mock）
 *
 * 形状与 Jev wire 协议 (POST /v1/systemone) 1:1 对应，所以换后端
 * 只是换一个 baseUrl。
 */

// ═══════════════════════════════════════════════════════════
// Question —— Jev 的三个原语
// ═══════════════════════════════════════════════════════════

export type QuestionType = 'noul' | 'choice' | 'score'

/** 真/假概率。用来做闸门：放行 / 拦截。 */
export interface NoulQuestion {
  type: 'noul'
  instructions: string
  /** 可选：把 true/false 两端讲清楚，显著提升判定质量 */
  criteria?: { true: string; false: string }
}

/** 从选项里选一个。用来做路由：走哪条路。 */
export interface ChoiceQuestion {
  type: 'choice'
  instructions: string
  /** 选项 id → 说明。注意 >20 个选项会明显掉点 */
  criteria: Record<string, string>
}

/** 落在有序量表上，可为小数（期望档位）。用来打分：多好 / 多严重。 */
export interface ScoreQuestion {
  type: 'score'
  instructions: string
  /** 有序量表，从低到高。档位别超过 5 个 */
  criteria: string[]
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion
export type QuestionSet = Record<string, Question>

export const noul = (
  instructions: string,
  criteria?: { true: string; false: string },
): NoulQuestion => (criteria ? { type: 'noul', instructions, criteria } : { type: 'noul', instructions })

export const choice = (instructions: string, criteria: Record<string, string>): ChoiceQuestion => ({
  type: 'choice',
  instructions,
  criteria,
})

export const score = (instructions: string, criteria: string[]): ScoreQuestion => ({
  type: 'score',
  instructions,
  criteria,
})

// ═══════════════════════════════════════════════════════════
// Answer
// ═══════════════════════════════════════════════════════════

export interface NoulAnswer {
  type: 'noul'
  /** P(真)，0..1 */
  noul: number
}

export interface ChoiceAnswer {
  type: 'choice'
  choice: string
  probabilities: Record<string, number>
  confidence: number
}

export interface ScoreAnswer {
  type: 'score'
  /** 期望档位，可为小数 */
  score: number
  legend: Record<string, string>
  probabilities: Record<string, number>
  confidence: number
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer
export type AnswerSet = Record<string, Answer>

/** 统一取置信度。noul 的置信度 = max(p, 1-p)。 */
export function confidenceOf(a: Answer | undefined): number {
  if (!a) return 0
  if (a.type === 'noul') return Math.max(a.noul, 1 - a.noul)
  return typeof a.confidence === 'number' ? a.confidence : 0
}

/** 把问题 id 映射到答案类型，让策略函数拿到精确类型而不是 any */
export type AnswerFor<Q> = Q extends { type: 'noul' }
  ? NoulAnswer
  : Q extends { type: 'choice' }
    ? ChoiceAnswer
    : Q extends { type: 'score' }
      ? ScoreAnswer
      : never

export type AnswerMap<Q extends QuestionSet> = { [K in keyof Q]: AnswerFor<Q[K]> }

// ═══════════════════════════════════════════════════════════
// Provider —— 内核与模型之间的唯一界面
// ═══════════════════════════════════════════════════════════

export interface DecideRequest {
  state: unknown
  questions: QuestionSet
  model?: string
  timeoutMs?: number
}

export interface Usage {
  input_tokens?: number
  output_tokens?: number
  /** true = 后端自己估的，不是真实计量 */
  estimated?: boolean
}

export interface DecideResponse {
  answers: AnswerSet
  /** 实际服务方 */
  provider: string
  model?: string
  latencyMs: number
  usage?: Usage
  /** true = 走了降级路径（主 Provider 失败 / 答案不完整） */
  degraded?: boolean
  /** Provider 中途发现的非致命问题，会进 trace */
  warnings?: string[]
}

export interface Provider {
  readonly name: string
  decide(req: DecideRequest): Promise<DecideResponse>
}

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

const DECISION = Symbol.for('nanojev.decision')

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
