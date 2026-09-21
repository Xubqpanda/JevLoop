/**
 * nanojev · 决策帧预算
 *
 * 决策模型的上下文很短（512 / 1024 token），而且 choice 的选项共享一个
 * 固定 head 预算（192 / 256）—— 选项一多，每个只分到几个 token，文本就
 * 不可区分了。实测 77 个选项会掉到 0.425。
 *
 * 所以 State 投影必须**有界**。这里提供截断工具和发请求之前的预算校验。
 */

import type { QuestionSet, ChoiceQuestion } from './types.ts'

/** ` …[+N]` 这个提示本身占的字符预算 */
const HINT_BUDGET = 12

/**
 * 按**字符**截断。
 *
 * 注意单位：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token。
 * 所以给中文正文设预算时，字符数和 token 数基本是 1:1 —— 不要按英文估。
 *
 * 只接受 string。要截对象请先自己 `JSON.stringify`。
 */
export function clip(s: string, maxChars: number): string {
  const str = String(s ?? '')
  if (str.length <= maxChars) return str
  const keep = Math.max(0, maxChars - HINT_BUDGET)
  return str.slice(0, keep) + ` …[+${str.length - keep}]`
}

/** 只挑需要的字段，防止整个 ctx 泄漏进决策帧 */
export function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const out = {} as Pick<T, K>
  for (const k of keys) if (obj?.[k] !== undefined) out[k] = obj[k];
  return out
}

/**
 * **字符启发式**，不是真 tokenizer。偏差约 ±30%，方向不可控。
 * 用途是「提前拦住明显超预算的帧」，不是精确计量 —— 名字里的 Rough 是刻意的。
 */
export function estimateTokens(value: unknown): number {
  const s = typeof value === 'string' ? value : JSON.stringify(value ?? '')
  if (!s) return 0
  let cjk = 0
  for (const ch of s) {
    const c = ch.codePointAt(0)!
    if ((c >= 0x2e80 && c <= 0x9fff) || (c >= 0xac00 && c <= 0xd7af) || (c >= 0x3040 && c <= 0x30ff)) cjk++
  }
  return Math.ceil(cjk + (s.length - cjk) / 4)
}

/** 各 checkpoint 的硬限制 */
export const LIMITS = {
  english: { context: 512, headBudget: 192, maxOptions: 20 },
  multilingual: { context: 1024, headBudget: 256, maxOptions: 20 },
  'typed-decisions': { context: 1024, headBudget: 256, maxOptions: 20 },
} as const

export type Checkpoint = keyof typeof LIMITS

export interface BudgetWarning {
  level: 'warn' | 'error'
  message: string
  hint?: string
}

/**
 * 发请求**之前**就发现问题。把模型的物理限制编码进代码，
 * 而不是等它在运行时静默掉点。
 */
export function validate(
  state: unknown,
  questions: QuestionSet,
  checkpoint: Checkpoint = 'typed-decisions',
): BudgetWarning[] {
  const out: BudgetWarning[] = []
  const lim = LIMITS[checkpoint] ?? LIMITS['typed-decisions']

  const tokens = estimateTokens(state)
  if (tokens > lim.context) {
    out.push({
      level: 'error',
      message: `决策帧约 ${tokens} token（估算 ±30%），超过 ${checkpoint} 的 ${lim.context} 上限`,
      hint: '用 clip() 截断，只留判定真正需要的字段。中文 1 字 ≈ 1 token，别按英文估。',
    })
  } else if (tokens > lim.context * 0.8) {
    out.push({
      level: 'warn',
      message: `决策帧约 ${tokens} token，已用掉 ${checkpoint} 上下文的 ${Math.round((tokens / lim.context) * 100)}%`,
      hint: '留余量：问题指令和选项还要占上下文。',
    })
  }

  for (const [id, q] of Object.entries(questions)) {
    if (q.type === 'choice') {
      const n = Object.keys((q as ChoiceQuestion).criteria).length
      if (n > lim.maxOptions) {
        out.push({
          level: n > 50 ? 'error' : 'warn',
          message: `问题 '${id}' 有 ${n} 个选项（安全线 ${lim.maxOptions}）`,
          hint: '改成两段式：先选类别，再在类别内选具体项',
        })
      }
    }
    if (q.type === 'score' && q.criteria.length > 7) {
      out.push({
        level: 'warn',
        message: `问题 '${id}' 的 score 有 ${q.criteria.length} 档，建议 ≤5 档`,
        hint: 'score 是三个原语里最弱的，档位越多越不准',
      })
    }
  }

  return out
}
