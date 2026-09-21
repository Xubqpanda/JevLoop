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

/**
 * 是否型问题。答案是一个概率：`noul` = 「true 有多可能」。
 *
 * `criteria` 可省，但**给了就要写成判据**（"什么条件下算 true"），
 * 不是名词标签 —— 问题 ID 不会到达模型，判据必须写进指令和选项里。
 *
 * ⚠️ 门限是**闭区间**（见 `probGte`）：`0.5` 是最不确定的取值，却能通过
 * 任何阈值为 0.5 的门。所以凡是「放行 / 继续」那一条，阈值必须严格大于 0.5。
 */
export const noul = (
  instructions: string,
  criteria?: { true: string; false: string },
): NoulQuestion => (criteria ? { type: 'noul', instructions, criteria } : { type: 'noul', instructions })

/**
 * 封闭选择。`criteria` 的**键是选项本身**（会原样回到 `answers[id].choice`），
 * 值是「什么条件下该选它」。
 *
 * ⚠️ 两个物理限制，超了要自己截断：选项**总数**别过 20，且全部选项文本共享
 * 一个固定的 head 预算（192/256 token）—— 实测 77 个选项时选中项概率掉到 0.425。
 * `validate()` 会在发请求之前把这两种情况报出来。
 */
export const choice = (instructions: string, criteria: Record<string, string>): ChoiceQuestion => ({
  type: 'choice',
  instructions,
  criteria,
})

/**
 * 有序量表。`criteria` 是档位说明，**从低到高**，别超过 5 档。
 *
 * 三个原语里它最弱（档位越多越不准），所以阈值类判断优先用 `noul` 或 `choice`。
 */
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
