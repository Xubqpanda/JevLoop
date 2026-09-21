/**
 * JevLoop · L2 接缝 —— 判定后端的**定义角**
 *
 * 能力缝是三角（docs/CODE-STYLE.md §10）：定义 / 提供者 / 消费。
 * 这个文件只有**定义**：
 *
 *     定义     Provider 接口                    ← 这里
 *     提供者   HttpProvider / MockProvider      （provider.ts）
 *     消费     Decider                          （decide.ts）
 *
 * 内核只认一件事：`{state, questions} → {answers}`。
 * 谁能提供这个，谁就能当判定后端 —— 包括这个仓库之外的实现。
 *
 * @module JevLoop/seam-provider
 */

import type { AnswerSet, QuestionSet } from './vocab.ts'

// ═══════════════════════════════════════════════════════════
// Provider —— 内核与模型之间的唯一界面
// ═══════════════════════════════════════════════════════════

export interface DecideRequest {
  state: unknown
  questions: QuestionSet
  model?: string
  timeoutMs?: number
}

export interface Usage {
  input_tokens?: number
  output_tokens?: number
  /** true = 后端自己估的，不是真实计量 */
  estimated?: boolean
}

export interface DecideResponse {
  answers: AnswerSet
  /** 实际服务方 */
  provider: string
  model?: string
  latencyMs: number
  usage?: Usage
  /** true = 走了降级路径（主 Provider 失败 / 答案不完整） */
  degraded?: boolean
  /** Provider 中途发现的非致命问题，会进 trace */
  warnings?: string[]
}

export interface Provider {
  readonly name: string
  decide(req: DecideRequest): Promise<DecideResponse>
}

// ═══════════════════════════════════════════════════════════
// 失败分类
//
// ══════════════════════════════════════════════════════════════
//  **按 `code` 分支，永远不要解析 `message`。**
// ══════════════════════════════════════════════════════════════
//
// message 是给人看的，改措辞是常事；而按措辞分支的调用方会在**没人改
// 调用方**的情况下静默失效。所以分类要有一个稳定的、机器可路由的名字。
//
// 形状抄 DSH 的 `HarnessError`（`packages/llm/llm/src/error.ts`），
// 包括它那句注释：「Stable machine-routable failure class; route on
// this, never by parsing `message`」。
//
// ── 哪些可重试，判断依据是什么 ──────────────────────────────────
//
// 判据是**「再试一次会不会有不一样的结果」**：
//
//   · 过载 / 限流 / 5xx / 超时 / 传输失败 → 会变，值得重试
//   · 凭据坏了 → **每一次都一模一样地失败**，重试只是白等
//   · 请求本身不合法（含上下文超限）→ 重试发的是同一个请求，
//     结果不会变；该做的是改请求
//   · 余额耗尽 → 同上
//
// DSH 对 `INVALID_CREDENTIAL` 的注释就是这个意思：「a malformed
// credential fails identically on every attempt」。
// ═══════════════════════════════════════════════════════════

/**
 * 机器可路由的失败分类。
 *
 * `HTTP_<n>` 是**没归类**的那些（4xx 里除 401/403/413/429 之外的）——
 * 它们默认**不可重试**：一个我们不认识的 4xx 多半是请求本身的问题。
 */
export type ProviderErrorCode =
  /** 401 / 403。凭据缺失或不合法 —— 重试每一次结果都一样 */
  | 'AUTH'
  /** 429 —— 服务端明确说了「慢一点」 */
  | 'RATE_LIMIT'
  /** ≥ 500。**含 529 `system_overloaded`** —— 实测就是这个把整轮打到 mock 上 */
  | 'SERVER'
  /** 我们自己的超时（本地掐断），不是服务端报的 */
  | 'TIMEOUT'
  /** 连不上：DNS、TLS、连接被拒。`fetch failed` 那一类 */
  | 'TRANSPORT'
  /** 余额 / 配额耗尽 */
  | 'QUOTA'
  /** 400 / 413 等：请求本身不合法 */
  | 'INVALID_REQUEST'
  /** 请求超出了模型的上下文窗口 */
  | 'CONTEXT_WINDOW_EXCEEDED'
  /** 后端正常返回但**一个答案都没有** —— 退化的完成，重试是安全的 */
  | 'EMPTY_RESPONSE'
  /** 没归类的状态码 */
  | `HTTP_${number}`

/** 带分类的判定失败。提供者抛它，重试层按 `code` 决定要不要再来一次 */
export class ProviderError extends Error {
  readonly code: ProviderErrorCode
  /** 原始 HTTP 状态，有的话 */
  readonly status?: number
  /**
   * 服务端要求的等待时间（`Retry-After` 头）。
   *
   * 有它就**优先用它**而不是自己算退避 —— 服务端知道它什么时候能好，
   * 我们不知道。（DSH 同：`providerRetryAfterMs`，秒数和 HTTP-date 两种
   * 格式都认。）
   */
  readonly retryAfterMs?: number

  constructor(
    message: string,
    code: ProviderErrorCode,
    extra: { status?: number; retryAfterMs?: number; cause?: unknown } = {},
  ) {
    super(message, extra.cause !== undefined ? { cause: extra.cause } : undefined)
    this.name = 'ProviderError'
    this.code = code
    if (extra.status !== undefined) this.status = extra.status
    if (extra.retryAfterMs !== undefined) this.retryAfterMs = extra.retryAfterMs
  }
}

/**
 * 默认可重试的分类。**封闭词表** —— 加一个进来是一次决定。
 *
 * 和 DSH 的默认集一致（`EMPTY_RESPONSE` / `RATE_LIMIT` / `SERVER` /
 * `TIMEOUT` / `TRANSPORT`），理由见上面那段判据。
 */
export const RETRYABLE_CODES: ReadonlySet<string> = new Set<string>([
  'RATE_LIMIT',
  'SERVER',
  'TIMEOUT',
  'TRANSPORT',
  'EMPTY_RESPONSE',
])

/**
 * 这个失败值得再试一次吗。
 *
 * 认不出的错误一律**不可重试**：一个我们没分类的失败，重试只是在
 * 不确定上再赌一次。
 */
export function isRetryable(err: unknown): err is ProviderError {
  return err instanceof ProviderError && RETRYABLE_CODES.has(err.code)
}
