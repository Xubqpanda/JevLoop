/**
 * JevLoop · 判定后端的有界重试
 *
 * ══════════════════════════════════════════════════════════════
 *  **补的是一个实测出来的窟窿：瞬时过载会一路掉到 mock。**
 * ══════════════════════════════════════════════════════════════
 *
 * 实测（2026-09-21）：托管 Jev 返回 `HTTP 529 system_overloaded`，
 * 而 `FallbackProvider` 把它当成**永久失败** —— 不重试，直接下一跳。
 * 本地 Laya sidecar 没在跑，于是整轮判定**静默跑在 mock 的恒定 0.5 上**，
 * 轨迹里只有一句 `degraded: true`，没有任何地方说得出原因。
 *
 * ── 重试和降级是两件事 ──────────────────────────────────────────
 *
 *     RetryingProvider   这一跳**暂时**不行 → 等一下再问**同一个**
 *     FallbackProvider   这一跳**就是**不行 → 换下一个
 *
 * 所以重试在链**里面**，包住每一个会瞬时失败的后端：
 *
 *     FallbackProvider([ Retrying(jev), Retrying(laya), mock ])
 *
 * ⚠️ 包在链**外面**是不行的，这一点试过就知道：链的最后一级是 mock，
 * 它**永远不抛**，所以外面的重试包装器永远看不到错误、一次也不会触发。
 *
 * 机制（分类、退避、`Retry-After`）在 `retry.ts`（L1）—— 生成那条缝
 * 也要同一套，而两条缝同层，§11 不许互相依赖。
 *
 * @module JevLoop/provider-retry
 */

import type { DecideRequest, DecideResponse, Provider } from './seam-provider.ts'
import { resolveRetry, retryCall, type RetryOptions } from './retry.ts'

export type { RetryInfo, RetryOptions } from './retry.ts'

/**
 * 给会瞬时失败的判定后端加一层有界重试。
 *
 * **不可重试的错误原样抛出去**，不在这里吞掉 —— 降级链靠它决定要不要
 * 换下一个后端。这一层只回答一个问题：**同一个后端，要不要再问一次。**
 */
export class RetryingProvider implements Provider {
  /**
   * 和里层同名 —— 界面上那条 `jev→laya→mock` 是描述**用哪些后端**的，
   * 加一层重试不该改变那句话。重试本身由 `onRetry` 报出来。
   */
  readonly name: string

  readonly #inner: Provider
  readonly #cfg

  constructor(inner: Provider, opts: RetryOptions = {}) {
    this.#inner = inner
    this.name = inner.name
    this.#cfg = resolveRetry(opts)
  }

  decide(req: DecideRequest): Promise<DecideResponse> {
    return retryCall(this.name, this.#cfg, () => this.#inner.decide(req))
  }
}
