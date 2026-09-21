/**
 * JevLoop · L0 词汇 —— Question 与 Answer
 *
 * 决策模型的线协议里只有三个原语，这个文件是它们的**唯一**定义处：
 *
 *     noul    P(真)                 → 闸门：放行 / 拦截
 *     choice  从固定选项里选一个      → 路由：走哪条路
 *     score   落在有序量表上          → 打分：多好 / 多严重
 *
 * 形状与 Jev wire 协议 (POST /v1/systemone) 1:1 对应，所以换后端
 * 只是换一个 baseUrl。
 *
 * **这一层不依赖任何东西**，也不该依赖任何东西。任何需要「一个问题长什么样」
 * 或「一个答案长什么样」的代码都从这里取 —— 而不是各自定义一份。
 *
 * @module JevLoop/vocab
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
