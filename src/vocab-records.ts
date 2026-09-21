/**
 * JevLoop · L0 词汇 —— 记账与审计的记录形状
 *
 * 这四个接口是**契约**：它们跨过事件边界（`events.ts` 的 `run:end` 带
 * `MeterStats`、`audit` 事件带 `AuditRecord`），所以属于词汇层，不属于记账实现。
 *
 * 为什么从 `meter.ts` 搬出来：`events.ts`（L1）曾经 `import type` 它们，
 * 于是**观察契约反向依赖了记账实现** —— 换一个记账方式就得动事件定义。
 * 搬过来之后 L1 内部不再互相依赖，依赖方向可以被机器检查
 * （见 `scripts/check.ts` 的 `layers` 规则）。
 *
 * 它们是纯数据，没有任何方法 —— `Meter` 类（有行为）仍住在 L1 的 `meter.ts`。
 *
 * @module JevLoop/vocab-records
 */

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
  /**
   * 判定次数 : 模型调用次数。**没有模型调用时是 `null`**，不是 `Infinity`。
   *
   * 为什么不是 `Infinity`：这个值要跨 JSON 出去（SSE 的 `run:end`），
   * 而 `JSON.stringify(Infinity)` 是 `null` —— 也就是说 **JSON 会静默改写它**，
   * 前端那条专门为 `Infinity` 写的 `!Number.isFinite` 分支因此**永远不可达**，
   * 界面显示成 `?` 而不是它本来想显示的 `N : 0`。
   *
   * 让不可表示的值**在跨越 JSON 之前**就变成可表示的：
   * `null` 是 JSON 能忠实携带的，而且语义明确（"没有模型调用"）。
   */
  ratio: number | null
  /** 判定耗时占「判定 + 模型」总耗时的比例 */
  decisionShare: number
  escalated: number
  degraded: number
  /** 审计留痕条数（`auto_audit` 动作触发） */
  audits: number
  inputTokens: number
  outputTokens: number
}
