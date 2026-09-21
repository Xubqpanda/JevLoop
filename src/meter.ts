/**
 * JevLoop · Meter
 *
 * 这个文件是 JevLoop 存在的理由。
 *
 * 常规 agent 的 loop 里，每一次分叉都是一次大模型调用 —— 贵、慢。
 * JevLoop 把那些分叉交给判定模型，只有「生成」才叫大模型。
 *
 * 所以最重要的指标不是"跑了多少 token"，而是：
 *
 *     **判定 : 模型 = ? : 1**
 *
 * 这个比值就是 JevLoop 的全部主张，它应该被打印在每一次运行的结尾。
 *
 * @module JevLoop/meter
 */

import type { DecisionResult } from './vocab-decision.ts'

import type { AuditRecord, DecisionRecord, MeterStats, ModelCallRecord } from './vocab-records.ts'

export type { AuditRecord, DecisionRecord, MeterStats, ModelCallRecord } from './vocab-records.ts'

/**
 * 记账本 —— 判定次数和模型调用次数**分开记**。
 *
 * 这两件事分开记是这个项目的主张本身：分母是模型调用，分子是判定，
 * 两者的比值说明「一个 loop 里有多少决定是判定做的」。合成一个计数器
 * 就再也说不出这句话了。`decisions` 与 `modelCalls` 都带各自的延迟，
 * 所以「判定耗时占墙钟多少」也能直接算出来（见 `stats.decisionShare`）。
 */
export class Meter {
  readonly decisions: DecisionRecord[] = []
  readonly modelCalls: ModelCallRecord[] = []
  /** 审计留痕。`auto_audit` 动作的落点 —— 见 AuditRecord 的说明 */
  readonly audit: AuditRecord[] = []

  recordDecision(step: number, d: DecisionResult<unknown>): DecisionRecord {
    const rec: DecisionRecord = {
      step,
      id: d.id,
      action: d.action,
      reason: d.reason,
      latencyMs: d.latencyMs,
      provider: d.provider,
      degraded: d.degraded,
      escalate: d.escalate,
      answers: summarizeAnswers(d.answers as Record<string, any>),
    }
    this.decisions.push(rec)
    return rec
  }

  recordModelCall(step: number, c: Omit<ModelCallRecord, 'step'>): ModelCallRecord {
    const rec: ModelCallRecord = { step, ...c }
    this.modelCalls.push(rec)
    return rec
  }

  /** 记一条审计。`auto_audit` 判定走这里，让那个 action 名不再是空头支票 */
  recordAudit(step: number, a: Omit<AuditRecord, 'step' | 'at'>): AuditRecord {
    const rec: AuditRecord = { step, at: Date.now(), ...a }
    this.audit.push(rec)
    return rec
  }

  get stats(): MeterStats {
    const decisionMs = this.decisions.reduce((a, d) => a + d.latencyMs, 0)
    const modelMs = this.modelCalls.reduce((a, m) => a + m.latencyMs, 0)
    const n = this.decisions.length || 1
    const total = decisionMs + modelMs

    return {
      decisions: this.decisions.length,
      decisionMs: round(decisionMs),
      avgDecisionMs: round(decisionMs / n),
      modelCalls: this.modelCalls.length,
      modelMs: round(modelMs),
      // ★ `null`，不是 `Infinity`。这个值要跨 JSON 出去，而
      //   `JSON.stringify(Infinity)` 是 `null` —— 让 JSON 来替我们决定
      //   等于把语义交给一个静默的转换。见 `MeterStats.ratio` 的说明。
      ratio: this.modelCalls.length ? this.decisions.length / this.modelCalls.length : null,
      decisionShare: total > 0 ? decisionMs / total : 1,
      escalated: this.decisions.filter((d) => d.escalate).length,
      degraded: this.decisions.filter((d) => d.degraded).length,
      audits: this.audit.length,
      inputTokens: this.modelCalls.reduce((a, m) => a + (m.inputTokens ?? 0), 0),
      outputTokens: this.modelCalls.reduce((a, m) => a + (m.outputTokens ?? 0), 0),
    }
  }

  /** 逐条明细，用于 --trace */
  trace(): string {
    const lines: string[] = []
    const byStep = new Map<number, { d: DecisionRecord[]; m: ModelCallRecord[] }>()
    for (const d of this.decisions) {
      const e = byStep.get(d.step) ?? { d: [], m: [] }
      e.d.push(d)
      byStep.set(d.step, e)
    }
    for (const m of this.modelCalls) {
      const e = byStep.get(m.step) ?? { d: [], m: [] }
      e.m.push(m)
      byStep.set(m.step, e)
    }

    for (const step of [...byStep.keys()].sort((a, b) => a - b)) {
      const { d, m } = byStep.get(step)!
      lines.push(`  step ${step}`)
      for (const dec of d) {
        const mark = dec.escalate ? '!' : dec.degraded ? '~' : ' '
        lines.push(
          `   ${mark} decide  ${dec.id.padEnd(22)} ${dec.action.padEnd(18)} ${dec.latencyMs.toFixed(1)}ms  ${dec.answers}`,
        )
      }
      for (const call of m) {
        lines.push(`     model   ${call.kind.padEnd(22)} ${' '.repeat(18)} ${call.latencyMs.toFixed(0)}ms`)
      }
      for (const a of this.audit.filter((x) => x.step === step)) {
        lines.push(`     audit   ${a.tool.padEnd(22)} risk=${a.risk ?? '—'}  ${a.target}`)
      }
    }
    return lines.join('\n')
  }

  /** 一句能直接发给别人看的话 */
  summary(): string {
    const s = this.stats
    return (
      `${s.decisions} decisions / ${s.decisionMs}ms  ·  ` +
      `${s.modelCalls} model calls / ${s.modelMs}ms  ·  ` +
      `decisions:models = ${formatRatio(s)}  ·  ` +
      `decision time = ${(s.decisionShare * 100).toFixed(1)}%`
    )
  }

  reset(): void {
    this.decisions.length = 0
    this.modelCalls.length = 0
    this.audit.length = 0
  }
}

function summarizeAnswers(answers: Record<string, any>): string {
  return Object.entries(answers)
    .map(([k, a]) => {
      if (!a) return `${k}=—`
      if (a.type === 'noul') return `${k}=${a.noul.toFixed(2)}`
      if (a.type === 'score') return `${k}=${a.score.toFixed(1)}`
      return `${k}=${a.choice}`
    })
    .join(' ')
}

const round = (v: number) => Math.round(v * 10) / 10

/**
 * 「判定 : 模型」的统一格式化。**所有出口都必须调它**，不要在调用点各自拼字符串。
 *
 * 为什么值得单独一个函数：`examples/demo.ts` 以前自己拼了一份，写成
 * `(s.modelCalls ? ratio.toFixed(1) : String(s.decisions)) + ' : 1'` ——
 * 0 次模型调用时输出 `3 : 1`，而真相是「3 次判定 / 0 次模型调用」。
 * **这个比值是项目的卖点本身**，在离线和规则模式下把它报反，等于把卖点报反。
 *
 * 分隔符也统一成 `:`（以前 `meter` 用 `:`、`demo` 用 ` : `）。
 */
export function formatRatio(s: { decisions: number; modelCalls: number; ratio: number | null }): string {
  // `ratio === null` = 没有模型调用 —— 也可能是老的、被 JSON 变成 `null` 的 `Infinity`，
  // 两种都落到 `N:0`。那才是真相：跑了 N 次判定，一次模型调用都没有。
  if (s.ratio === null || !Number.isFinite(s.ratio)) return `${s.decisions}:0`
  return `${s.ratio.toFixed(1)}:1`
}
