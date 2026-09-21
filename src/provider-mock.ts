/**
 * JevLoop · Mock —— 保守占位，故意「不知道」
 *
 * **从 `provider.ts` 切出来** —— 那个文件住着**四件事**
 * （HTTP 判定后端 / 响应归一化 / Mock / 降级链），按 docs/CODE-STYLE.md §12
 * 「一句话说不完就是住着两件以上」该拆。接缝是**各自的契约形状不同**：
 *  * Mock 只保证「离线可跑」，返回的 degraded 永远是 true —— 它不是判定质量的证据
 *
 * @module JevLoop/provider-mock
 */

import type { Provider, DecideRequest, DecideResponse } from './seam-provider.ts'
import type { Answer, AnswerSet, Question } from './vocab.ts'

// ═══════════════════════════════════════════════════════════
// Mock —— 保守占位，故意"不知道"
//
// 这里**故意不做启发式猜测**。
// 猜得越像，越容易让人误以为判定是对的。
// 保守答案（概率 0.5、第一个选项、低置信度）会落在各条 policy 的兜底分支上，
// 这正是框架要演示的事：
//
//     **判定质量不够时，正确的行为是把决定交回上层，而不是硬猜。**
//
// ⚠️ 别把这句话读成「一定会走到 escalate」——`0.5` 恰好**越过**任何阈值为 0.5 的门
// （`>=` 是闭区间）。实测 Mock 的真实结果是：
// gradeRisk→ask_human、isDone→keep_going、canDeliver→revise、needsTool→use_tool。
// 所以**凡是「放行 / 继续」那一条，阈值必须严格大于 0.5** ——
// 否则「毫无信息」会被读成「同意」。`T.stepOk` 因此从 0.5 提到了 0.6
// （见 `docs/REVIEWS-2026-09-21-round4.md` 方案 2）。
// ═══════════════════════════════════════════════════════════

/**
 * 保守判定后端：不接任何模型，对每个问题返回「不表态」的答案
 * （`noul` 给 0.5、`choice` 给第一个选项、`score` 给中间档）。
 *
 * 它让整条 loop **离线可跑**，并演示「判定质量不够时，正确的行为是把决定
 * 交回上层」。返回的 `degraded` 永远是 `true` 并带一条 warning ——
 * 用 Mock 跑出来的数字**不能**当成判定质量的证据。
 */
export class MockProvider implements Provider {
  readonly name = 'mock'
  #latencyMs: number

  constructor(opts: { latencyMs?: number } = {}) {
    this.#latencyMs = opts.latencyMs ?? 3
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    const t0 = performance.now()
    if (this.#latencyMs > 0) await new Promise((r) => setTimeout(r, this.#latencyMs))

    const answers: AnswerSet = {}
    for (const [id, q] of Object.entries(req.questions)) answers[id] = conservative(q)

    return {
      answers,
      provider: this.name,
      model: 'mock',
      latencyMs: performance.now() - t0,
      degraded: true,
      // ★ 这句以前说「策略会因置信度不足而交回上层」—— **那是过度承诺**。
      //   实测跑全部七个节点，只有 3 个真的交回上层（pickTool / gradeRisk / stepOk），
      //   另外 4 个（needsTool / pickInput / isDone / canDeliver）会继续往下走，
      //   因为 0.5 恰好**越过**任何阈值为 0.5 的门（`>=` 是闭区间）。
      //   警告的用处就是别让人误以为 Mock 的行为是判定结果，所以它必须说实测到的那个。
      warnings: [
        '未接真实判定后端：返回的是保守占位答案（noul 恒 0.5、choice 取第一个、score 取中间档）。' +
          '各节点落在自己的兜底分支上，但**不是**全部交回上层 —— 0.5 会越过阈值为 0.5 的门，' +
          '所以 7 个节点里只有 pickTool / gradeRisk / stepOk 会停下。用 Mock 跑出来的结果不是判定质量的证据。',
      ],
    }
  }
}

/** 三个原语的保守答案：不表态，让 policy 去处理 */
function conservative(q: Question): Answer {
  if (q.type === 'noul') return { type: 'noul', noul: 0.5 }
  if (q.type === 'score') {
    const mid = (q.criteria.length - 1) / 2
    const legend: Record<string, string> = {}
    const probabilities: Record<string, number> = {}
    q.criteria.forEach((lvl, i) => {
      legend[String(i)] = lvl
      probabilities[String(i)] = 1 / q.criteria.length
    })
    return { type: 'score', score: mid, legend, probabilities, confidence: 1 / q.criteria.length }
  }
  const ids = Object.keys(q.criteria)
  const probabilities: Record<string, number> = {}
  for (const id of ids) probabilities[id] = 1 / Math.max(1, ids.length)
  return { type: 'choice', choice: ids[0] ?? '', probabilities, confidence: 1 / Math.max(1, ids.length) }
}
