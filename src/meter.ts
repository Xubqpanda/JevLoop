/**
 * nanojev · Meter
 *
 * 这个文件是 nanojev 存在的理由。
 *
 * 常规 agent 的 loop 里，每一次分叉都是一次大模型调用 —— 贵、慢。
 * nanojev 把那些分叉交给判定模型，只有「生成」才叫大模型。
 *
 * 所以最重要的指标不是"跑了多少 token"，而是：
 *
 *     **判定 : 模型 = ? : 1**
 *
 * 这个比值就是 nanojev 的全部主张，它应该被打印在每一次运行的结尾。
 */

import type { DecisionResult } from "./types.ts";

export interface DecisionRecord {
  step: number;
  id: string;
  action: string;
  reason: string;
  latencyMs: number;
  provider: string;
  degraded: boolean;
  escalate: boolean;
  /** 每个问题的答案摘要，用于 trace */
  answers: string;
}

export interface ModelCallRecord {
  step: number;
  kind: string;
  latencyMs: number;
  inputTokens?: number;
  outputTokens?: number;
}

export interface MeterStats {
  decisions: number;
  decisionMs: number;
  avgDecisionMs: number;
  modelCalls: number;
  modelMs: number;
  /** 判定次数 : 模型调用次数。没有模型调用时是 Infinity */
  ratio: number;
  /** 判定耗时占「判定 + 模型」总耗时的比例 */
  decisionShare: number;
  escalated: number;
  degraded: number;
  inputTokens: number;
  outputTokens: number;
}

export class Meter {
  readonly decisions: DecisionRecord[] = [];
  readonly modelCalls: ModelCallRecord[] = [];

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
    };
    this.decisions.push(rec);
    return rec;
  }

  recordModelCall(step: number, c: Omit<ModelCallRecord, "step">): ModelCallRecord {
    const rec: ModelCallRecord = { step, ...c };
    this.modelCalls.push(rec);
    return rec;
  }

  get stats(): MeterStats {
    const decisionMs = this.decisions.reduce((a, d) => a + d.latencyMs, 0);
    const modelMs = this.modelCalls.reduce((a, m) => a + m.latencyMs, 0);
    const n = this.decisions.length || 1;
    const total = decisionMs + modelMs;

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
      inputTokens: this.modelCalls.reduce((a, m) => a + (m.inputTokens ?? 0), 0),
      outputTokens: this.modelCalls.reduce((a, m) => a + (m.outputTokens ?? 0), 0),
    };
  }

  /** 逐条明细，用于 --trace */
  trace(): string {
    const lines: string[] = [];
    const byStep = new Map<number, { d: DecisionRecord[]; m: ModelCallRecord[] }>();
    for (const d of this.decisions) {
      const e = byStep.get(d.step) ?? { d: [], m: [] };
      e.d.push(d);
      byStep.set(d.step, e);
    }
    for (const m of this.modelCalls) {
      const e = byStep.get(m.step) ?? { d: [], m: [] };
      e.m.push(m);
      byStep.set(m.step, e);
    }

    for (const step of [...byStep.keys()].sort((a, b) => a - b)) {
      const { d, m } = byStep.get(step)!;
      lines.push(`  step ${step}`);
      for (const dec of d) {
        const mark = dec.escalate ? "!" : dec.degraded ? "~" : " ";
        lines.push(
          `   ${mark} 判定  ${dec.id.padEnd(22)} ${dec.action.padEnd(18)} ${dec.latencyMs.toFixed(1)}ms  ${dec.answers}`,
        );
      }
      for (const call of m) {
        lines.push(`     模型  ${call.kind.padEnd(22)} ${" ".repeat(18)} ${call.latencyMs.toFixed(0)}ms`);
      }
    }
    return lines.join("\n");
  }

  /** 一句能直接发给别人看的话 */
  summary(): string {
    const s = this.stats;
    const ratio = s.modelCalls ? `${s.ratio.toFixed(1)}:1` : `${s.decisions}:0`;
    return (
      `${s.decisions} decisions / ${s.decisionMs}ms  ·  ` +
      `${s.modelCalls} model calls / ${s.modelMs}ms  ·  ` +
      `decisions:models = ${ratio}  ·  ` +
      `decision time = ${(s.decisionShare * 100).toFixed(1)}%`
    );
  }

  reset(): void {
    this.decisions.length = 0;
    this.modelCalls.length = 0;
  }
}

function summarizeAnswers(answers: Record<string, any>): string {
  return Object.entries(answers)
    .map(([k, a]) => {
      if (!a) return `${k}=—`;
      if (a.type === "noul") return `${k}=${a.noul.toFixed(2)}`;
      if (a.type === "score") return `${k}=${a.score.toFixed(1)}`;
      return `${k}=${a.choice}`;
    })
    .join(" ");
}

const round = (v: number) => Math.round(v * 10) / 10;
