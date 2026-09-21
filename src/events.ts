/**
 * JevLoop · 事件缝
 *
 * agent loop 把「发生了什么」发成事件，而不是直接打印或直接渲染。
 * 好处是**同一个 loop 可以喂给多个消费者**：
 *
 *     runAgent({ onEvent })  ──▶  终端 · SSE 到浏览器 · 测试断言 · 未来的标定台
 *
 * 这一层是只读的观察面：观察者拿不到取消、改写的句柄，
 * 所以一个观察者出错不会影响 loop 本身（见 {@link fanOut}）。
 *
 * @module JevLoop/events
 */

import type { DecisionResult } from './vocab-decision.ts'
import type { QuestionSet, AnswerSet } from './vocab.ts'
import type { MeterStats, AuditRecord } from './meter.ts'

/** 每一步里发生的一件事。判别式联合，新增成员时编译器会逼消费者处理。 */
export type AgentEvent =
  | { type: 'run:start'; task: string; cwd: string; at: number }
  /** 一次判定完成 —— 这是界面最主要的信息来源 */
  | {
      type: 'decision'
      step: number
      id: string
      /** 实际发给模型的决策帧 */
      state: unknown
      questions: QuestionSet
      answers: AnswerSet
      action: string
      reason: string
      latencyMs: number
      provider: string
      degraded: boolean
      escalate: boolean
    }
  /** 判定节点要求人工授权，以及结果 */
  | { type: 'authorize'; step: number; tool: string; reason: string; approved: boolean }
  /** `auto_audit` 承诺的留痕真的发生了 */
  | { type: 'audit'; step: number; record: AuditRecord }
  | { type: 'tool:call'; step: number; tool: string; input: string }
  | { type: 'tool:result'; step: number; tool: string; output: string; ms: number }
  /** 一次生成调用。整个运行里通常只有一次，最多两次（修订） */
  | { type: 'generate'; step: number; kind: string; latencyMs: number; tokens: number }
  | { type: 'run:end'; halt: string; steps: number; answer: string; stats: MeterStats }

/** 观察者。返回值被忽略，抛出的异常被隔离。 */
export type AgentObserver = (event: AgentEvent) => void

/** 把一次判定的结果转成事件。字段逐个搬，避免把 DecisionResult 的内部形状泄漏成事件契约。 */
export function decisionEvent(step: number, d: DecisionResult<unknown>): AgentEvent {
  return {
    type: 'decision',
    step,
    id: d.id,
    state: d.state,
    questions: d.questions,
    answers: d.answers as AnswerSet,
    action: d.action,
    reason: d.reason,
    latencyMs: d.latencyMs,
    provider: d.provider,
    degraded: d.degraded,
    escalate: d.escalate,
  }
}

/**
 * 把若干观察者合成一个。
 *
 * **一个观察者抛异常不影响其它观察者，也不影响 loop。** UI 崩了不该让 agent 停下，
 * 但也不能静默 —— 异常通过 `onError` 报出来（默认打到 stderr）。
 */
export function fanOut(
  observers: readonly AgentObserver[],
  onError: (err: unknown, index: number) => void = (err, i) =>
    console.error(`  ▲ 事件观察者 #${i} 抛异常（已隔离）：${(err as Error)?.message ?? String(err)}`),
): AgentObserver {
  return (event) => {
    observers.forEach((o, i) => {
      try {
        o(event)
      } catch (err) {
        onError(err, i)
      }
    })
  }
}
