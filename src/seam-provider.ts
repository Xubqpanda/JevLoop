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
// 失败分类 —— 实现在 `http-error.ts`（L0）
//
// ★ **为什么搬走**：这个仓库有**两条**通向后端的缝（判定、生成），两条都要
//   这套分类，而它们是同层（都 L2），§11 不许互相依赖 —— 所以共用的东西
//   必须下沉。`http-error.ts` 零 IO、零 import。
//
// 这里再导出，是为了让「判定后端」这条缝的消费方仍然只认一个入口。
// ═══════════════════════════════════════════════════════════

export { ProviderError, isRetryable, RETRYABLE_CODES } from './http-error.ts'
export type { ProviderErrorCode } from './http-error.ts'
