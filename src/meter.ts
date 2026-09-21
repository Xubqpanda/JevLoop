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
 */

import type { DecisionResult } from './types.ts'

export interface DecisionRecord {
  step: number
  id: string
  action: string
  reason: string
  latencyMs: number
  provider: string
  degraded: boolean
  escalate: boolean
  /** 每个问题的答案摘要，用于 trace */
  answers: string
}

export interface ModelCallRecord {
  step: number
  kind: string
  latencyMs: number
  inputTokens?: number
  outputTokens?: number
}

/**
 * 一条审计记录 —— 「不可逆操作要留痕」这件事的落点。
 *
 * `loop.gradeRisk` 判出 `auto_audit` 时写一条。以前那个 action 名承诺了审计，
 * 实际和 `auto` 完全一样（只多打一行 trace）—— 名字和行为脱钩，
 * 而读 policy 的人会以为写了 `auto_audit` 就有留痕保障。
 */
export interface AuditRecord {
  step: number
  tool: string
  /** 目标（工具输入，截断过） */
  target: string
  /** 判定给出的理由，原样保留 */
  reason: string
  /** 风险分（`gradeRisk` 的 score），拿不到就是 undefined */
  risk: number | undefined
  at: number
}

export interface MeterStats {
  decisions: number
  decisionMs: number
  avgDecisionMs: number
  modelCalls: number
  modelMs: number
  /** 判定次数 : 模型调用次数。没有模型调用时是 Infinity */
  ratio: number
  /** 判定耗时占「判定 + 模型」总耗时的比例 */
  decisionShare: number
  escalated: number
  degraded: number
  /** 审计留痕条数（`auto_audit` 动作触发） */
  audits: number
  inputTokens: number
  outputTokens: number
}

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
      ratio: this.modelCalls.length ? this.decisions.length / this.modelCalls.length : Infinity,
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
          `   ${mark} 判定  ${dec.id.padEnd(22)} ${dec.action.padEnd(18)} ${dec.latencyMs.toFixed(1)}ms  ${dec.answers}`,
        )
      }
      for (const call of m) {
        lines.push(`     模型  ${call.kind.padEnd(22)} ${' '.repeat(18)} ${call.latencyMs.toFixed(0)}ms`)
      }
      for (const a of this.audit.filter((x) => x.step === step)) {
        lines.push(`     审计  ${a.tool.padEnd(22)} risk=${a.risk ?? '—'}  ${a.target}`)
      }
    }
    return lines.join('\n')
  }

  /** 一句能直接发给别人看的话 */
  summary(): string {
    const s = this.stats
    const ratio = s.modelCalls ? `${s.ratio.toFixed(1)}:1` : `${s.decisions}:0`
    return (
      `${s.decisions} decisions / ${s.decisionMs}ms  ·  ` +
      `${s.modelCalls} model calls / ${s.modelMs}ms  ·  ` +
      `decisions:models = ${ratio}  ·  ` +
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
