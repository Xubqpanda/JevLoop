/**
 * JevLoop · decide()
 *
 * 一个判定走六步：
 *
 *   ① State 投影      把 ctx 压成有界决策帧
 *   ② 取问题          可以是 ctx 的函数（选项随状态变化）
 *   ③ 预算校验        发请求【之前】检查模型限制
 *   ④ 一次前向        拿到所有问题的答案
 *   ⑤ 策略            答案 → 动作（纯代码）
 *   ⑥ 记账            进 meter
 *
 * 后台挂了也不会崩：记一笔 degraded，然后交回上层，而不是瞎猜一个动作。
 *
 * @module JevLoop/decide
 */

import type { Provider } from './seam-provider.ts'
import type { DecisionSpec, DecisionResult } from './vocab-decision.ts'
import type { QuestionSet, AnswerMap, AnswerSet } from './vocab.ts'
import { resolvePolicy, type PolicyWarning } from './policy.ts'
import { validate, type BudgetWarning, type Checkpoint } from './budget.ts'
import { Meter } from './meter.ts'

export interface DeciderOptions {
  provider: Provider
  /**
   * 不传就自建一个 —— **永远不会是 undefined**。
   *
   * 早先这里是可选的，于是 `runAgent` 里写成 `decider.meter ?? new Meter()`：
   * 判定记进了 `undefined`，而返回给调用方的是那个新建的、没人写过的 Meter。
   * 结果是最自然的用法（`new Decider({ provider })`）拿到的比值是 `0 : 1` ——
   * 这个项目赖以成立的数字什么都报不出来。改成必填后，那个状态在类型上不存在。
   */
  meter?: Meter
  checkpoint?: Checkpoint
  /** 预算超限时直接抛，而不是只告警 */
  strict?: boolean
  timeoutMs?: number
  onWarn?: (id: string, warnings: BudgetWarning[]) => void
  onPolicyWarn?: (id: string, warnings: PolicyWarning[]) => void
}

export interface DecideOptions {
  step?: number
  model?: string
}

export class Decider {
  readonly provider: Provider
  readonly meter: Meter
  readonly checkpoint: Checkpoint
  readonly strict: boolean
  #timeoutMs: number
  #onWarn: DeciderOptions['onWarn']
  #onPolicyWarn: DeciderOptions['onPolicyWarn']
  #step = 0

  constructor(opts: DeciderOptions) {
    this.provider = opts.provider
    this.meter = opts.meter ?? new Meter()
    this.checkpoint = opts.checkpoint ?? 'typed-decisions'
    this.strict = opts.strict ?? false
    this.#timeoutMs = opts.timeoutMs ?? 30_000
    this.#onWarn = opts.onWarn
    this.#onPolicyWarn = opts.onPolicyWarn
  }

  setStep(n: number): void {
    this.#step = n
  }

  /** 跑一次判定 —— 这是唯一的公开入口 */
  async decide<Ctx, Q extends QuestionSet>(
    spec: DecisionSpec<Ctx, Q>,
    ctx: Ctx,
    opts: DecideOptions = {},
  ): Promise<DecisionResult<AnswerMap<Q>>> {
    const step = opts.step ?? this.#step

    // ① State 投影
    const state = spec.state(ctx)

    // ② 问题
    const questions: QuestionSet =
      typeof spec.questions === 'function' ? spec.questions(ctx) : spec.questions

    // ③ 发请求之前就检查预算
    const warnings = validate(state, questions, this.checkpoint)
    if (warnings.length) this.#onWarn?.(spec.id, warnings)
    if (this.strict && warnings.some((w) => w.level === 'error')) {
      throw new Error(`决策 '${spec.id}' 超出 ${this.checkpoint} 限制：${warnings.map((w) => w.message).join('; ')}`)
    }

    // ④ 一次前向
    let answers: AnswerSet = {}
    let latencyMs = 0
    let provider = this.provider.name
    let model: string | undefined
    let degraded = false
    const notes: string[] = []
    const t0 = performance.now()

    try {
      const res = await this.provider.decide({
        state,
        questions,
        ...(opts.model ?? spec.model ? { model: opts.model ?? spec.model } : {}),
        timeoutMs: this.#timeoutMs,
      })
      answers = res.answers
      latencyMs = res.latencyMs
      provider = res.provider
      model = res.model
      degraded = !!res.degraded
      if (res.warnings) notes.push(...res.warnings)
    } catch (err) {
      // 防线：后端整个不可用也不能让 loop 崩。
      // 记 degraded，然后交回上层 —— 不猜。
      const failedMs = performance.now() - t0
      const result: DecisionResult<AnswerMap<Q>> = {
        id: spec.id,
        step,
        state,
        questions,
        answers: {} as AnswerMap<Q>,
        action: 'escalate',
        reason: `${provider} 不可用：${(err as Error).message}`,
        latencyMs: failedMs,
        provider, // 记**原计划用的**，不覆盖成 "none"
        degraded: true,
        escalate: true,
      }
      this.meter.recordDecision(step, result as DecisionResult<unknown>)
      return result
    }

    // ⑤ 策略
    const policyWarnings: PolicyWarning[] = []
    const outcome = resolvePolicy(spec.policy, answers as never, (w) => policyWarnings.push(w))
    if (policyWarnings.length) this.#onPolicyWarn?.(spec.id, policyWarnings)

    const result: DecisionResult<AnswerMap<Q>> = {
      id: spec.id,
      step,
      state,
      questions,
      answers: answers as AnswerMap<Q>,
      action: outcome.action,
      reason: outcome.reason,
      latencyMs,
      provider,
      ...(model !== undefined ? { model } : {}),
      degraded: degraded || policyWarnings.length > 0,
      escalate: outcome.action === 'escalate',
    }

    // ⑥ 记账
    this.meter.recordDecision(step, result as DecisionResult<unknown>)
    return result
  }
}
