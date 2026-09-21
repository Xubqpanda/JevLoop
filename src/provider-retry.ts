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
 * 那一次 bench 报出来的「产物合格 3/8」因此什么都不能证明 —— 它是
 * mock 的数字。
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
 * 形状照 DSH 的 `llm-retry`：它的策略是 **provider-owned**——
 * 「Adapters expose one resolved policy per registered provider route」，
 * 每个后端路由一份，降级在更外层。分类和可重试集见 `seam-provider.ts`。
 *
 * ── 参数为什么比 DSH 小一号 ─────────────────────────────────────
 *
 * DSH 的默认是 5 次 / 500ms 起 / 上限 10s —— 那是按**大模型生成**的耗时
 * 定的（一次几秒到几十秒，等 500ms 不算什么）。判定是另一个量级：托管
 * Jev 一次约 390ms，所以这里 3 次 / 300ms 起 / 上限 5s，总等待约 2 秒，
 * 仍然远小于一次生成。
 *
 * 有 IO（`fetch` 在里层，等待是真的等），所以放 L2。
 *
 * @module JevLoop/provider-retry
 */

import { ProviderError, isRetryable, type DecideRequest, type DecideResponse, type Provider } from './seam-provider.ts'

/** 一次重试的安排。`onRetry` 拿它去记账 */
export interface RetryInfo {
  /** 哪个后端要重试 */
  provider: string
  /** 失败分类（`ProviderError.code`） */
  code: string
  /** 这是第几次重试（从 1 开始） */
  attempt: number
  /** 这次等了多久 */
  delayMs: number
  /** 等待是不是服务端指定的（`Retry-After`）而不是我们算的 */
  fromServer: boolean
}

export interface RetryOptions {
  /** 首次请求之后最多再试几次（默认 3） */
  maxRetries?: number
  /** 退避起点，毫秒（默认 300） */
  initialDelayMs?: number
  /** 单次等待上限，毫秒（默认 5000） */
  maxDelayMs?: number
  /** 对称抖动比例，0–1（默认 0.2）。见下面 `delayFor` 的说明 */
  jitterRatio?: number
  /** 每次重试**之前**报一下 —— 让调用方看得见「它在重试」而不是卡住了 */
  onRetry?: (info: RetryInfo) => void
  /** 测试用：注入随机数。默认 `Math.random` */
  random?: () => number
  /** 测试用：注入等待。默认 `setTimeout` */
  sleep?: (ms: number) => Promise<void>
}

const DEFAULTS = { maxRetries: 3, initialDelayMs: 300, maxDelayMs: 5_000, jitterRatio: 0.2 } as const

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
  readonly #maxRetries: number
  readonly #initialDelayMs: number
  readonly #maxDelayMs: number
  readonly #jitterRatio: number
  readonly #onRetry?: (info: RetryInfo) => void
  readonly #random: () => number
  readonly #sleep: (ms: number) => Promise<void>

  constructor(inner: Provider, opts: RetryOptions = {}) {
    this.#inner = inner
    this.name = inner.name
    this.#maxRetries = opts.maxRetries ?? DEFAULTS.maxRetries
    this.#initialDelayMs = opts.initialDelayMs ?? DEFAULTS.initialDelayMs
    this.#maxDelayMs = opts.maxDelayMs ?? DEFAULTS.maxDelayMs
    this.#jitterRatio = opts.jitterRatio ?? DEFAULTS.jitterRatio
    this.#onRetry = opts.onRetry
    this.#random = opts.random ?? Math.random
    this.#sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)))
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    let lastErr: unknown

    for (let attempt = 0; attempt <= this.#maxRetries; attempt++) {
      try {
        return await this.#inner.decide(req)
      } catch (err) {
        lastErr = err

        // 不是「再试一次会不一样」的失败 —— 立刻交给降级链，别白等。
        // 凭据坏了、请求不合法、余额耗尽都属于这一类（判据见 seam-provider.ts）
        if (!isRetryable(err)) throw err
        if (attempt === this.#maxRetries) throw err

        const planned = this.#delayFor(err, attempt + 1)
        // 服务端要求的等待超过了我们愿意等的上限 → **放弃，交给降级链**。
        // 不是把它的要求截短 —— 那等于在它说「还没好」的时候硬问
        if (planned === undefined) throw err

        this.#onRetry?.({
          provider: this.#inner.name,
          code: err.code,
          attempt: attempt + 1,
          delayMs: planned.delayMs,
          fromServer: planned.fromServer,
        })
        await this.#sleep(planned.delayMs)
      }
    }

    // 循环要么返回、要么抛出，走不到这里；留着是为了让类型收敛
    throw lastErr
  }

  /**
   * 下一次要等多久。`undefined` = **不等了**（服务端要的时间超过上限）。
   *
   * 顺序和 DSH 一致：**服务端说了算**。
   *
   *     有 Retry-After 且在上限内  → 原样用它，不算退避
   *     有 Retry-After 但超上限    → 放弃这一跳
   *     没有                       → 本地指数退避 × 对称抖动
   *
   * 为什么要抖动：多个调用方同时被限流时，固定退避会让它们在**同一刻**
   * 一起回来，然后一起再被限流。抖动把它们岔开。
   */
  #delayFor(err: ProviderError, retry: number): { delayMs: number; fromServer: boolean } | undefined {
    const after = err.retryAfterMs
    if (after !== undefined) {
      if (after > this.#maxDelayMs) return undefined
      return { delayMs: after, fromServer: true }
    }

    const exponent = Math.min(retry - 1, 16)
    const exponential = Math.min(this.#initialDelayMs * 2 ** exponent, this.#maxDelayMs)
    // 对称抖动：落在 [1-r, 1+r] 上，再夹一次上限——
    // 乘法可能把已经到顶的值推出上限（r > 0 时），不夹就超了
    const jitter = 1 - this.#jitterRatio + 2 * this.#jitterRatio * this.#random()
    return { delayMs: Math.min(exponential * jitter, this.#maxDelayMs), fromServer: false }
  }
}
