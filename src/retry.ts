/**
 * JevLoop · 有界重试的机制
 *
 * ══════════════════════════════════════════════════════════════
 *  「同一个后端，要不要再问一次」—— **策略在这一层，接线在各自的缝上。**
 * ══════════════════════════════════════════════════════════════
 *
 *     判定后端   `RetryingProvider`   （provider-retry.ts）
 *     生成后端   `RetryingGenerator`  （llm.ts）
 *
 * 两条缝都要这一套：分类好的失败（`http-error.ts`）+ 有界退避 + 认
 * `Retry-After`。而它们是**同层**（都 L2），§11 不许互相依赖 —— 所以
 * 机制必须下沉到这里。
 *
 * ── 重试和降级是两件事 ──────────────────────────────────────────
 *
 *     重试   这一跳**暂时**不行 → 等一下再问**同一个**
 *     降级   这一跳**就是**不行 → 换下一个
 *
 * 判定那条缝上，降级链是 `FallbackProvider`，重试包住链里**每一跳**
 * （包在链外面不行：链尾永远不抛，外面的包装器一次也不会触发 —— 见
 * `provider-retry.ts` 的文件头）。
 *
 * ── 为什么放 L1 ─────────────────────────────────────────────────
 *
 * 它是**机制**，不认识任何领域概念：给一个会抛的函数 + 一套参数，它决定
 * 等多久、要不要再试。
 *
 * ⚠️ **它唯一的杂质是等待**（`setTimeout`），而那是个定时器不是 IO ——
 * 没有文件、没有网络、没有进程外的东西。而且它是**可注入的**（`sleep`），
 * 所以测试里这一层是纯的：测退避不会变成测机器有多快。
 *
 * @module JevLoop/retry
 */

import { isRetryable, type ProviderError } from './http-error.ts'

/** 一次重试的安排。`onRetry` 拿它去记账 */
export interface RetryInfo {
  /** 哪个后端要重试 */
  who: string
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
  /** 对称抖动比例，0–1（默认 0.2）。见下面 `planDelay` 的说明 */
  jitterRatio?: number
  /** 每次重试**之前**报一下 —— 让调用方看得见「它在重试」而不是卡住了 */
  onRetry?: (info: RetryInfo) => void
  /** 测试用：注入随机数。默认 `Math.random` */
  random?: () => number
  /** 测试用：注入等待。默认 `setTimeout` */
  sleep?: (ms: number) => Promise<void>
}

/** 参数补齐之后的样子。**每个调用方解析一次，不要每跳解析一次** */
export interface ResolvedRetry {
  readonly maxRetries: number
  readonly initialDelayMs: number
  readonly maxDelayMs: number
  readonly jitterRatio: number
  readonly onRetry?: (info: RetryInfo) => void
  readonly random: () => number
  readonly sleep: (ms: number) => Promise<void>
}

/**
 * 缺省参数：3 次、300ms 起、单次上限 5s、抖动 ±20%。
 *
 * 比 DSH 的默认（5 次 / 500ms / 10s）小一号，因为它那套是按大模型生成的
 * 耗时定的，而判定一次只有约 390ms。理由见下面 `resolveRetry` 的说明。
 */
export const RETRY_DEFAULTS = { maxRetries: 3, initialDelayMs: 300, maxDelayMs: 5_000, jitterRatio: 0.2 } as const

/**
 * 参数为什么比 DSH 小一号：它的默认是 5 次 / 500ms 起 / 上限 10s ——
 * 那是按**大模型生成**的耗时定的（一次几秒到几十秒，等 500ms 不算什么）。
 * 判定是另一个量级：托管 Jev 一次约 390ms，所以 3 次 / 300ms 起 /
 * 上限 5s，总等待约 2 秒，仍然远小于一次生成。
 */
export function resolveRetry(opts: RetryOptions = {}): ResolvedRetry {
  return {
    maxRetries: opts.maxRetries ?? RETRY_DEFAULTS.maxRetries,
    initialDelayMs: opts.initialDelayMs ?? RETRY_DEFAULTS.initialDelayMs,
    maxDelayMs: opts.maxDelayMs ?? RETRY_DEFAULTS.maxDelayMs,
    jitterRatio: opts.jitterRatio ?? RETRY_DEFAULTS.jitterRatio,
    ...(opts.onRetry ? { onRetry: opts.onRetry } : {}),
    random: opts.random ?? Math.random,
    sleep: opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms))),
  }
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
export function planDelay(
  err: ProviderError,
  retry: number,
  cfg: ResolvedRetry,
): { delayMs: number; fromServer: boolean } | undefined {
  const after = err.retryAfterMs
  if (after !== undefined) {
    if (after > cfg.maxDelayMs) return undefined
    return { delayMs: after, fromServer: true }
  }

  const exponent = Math.min(retry - 1, 16)
  const exponential = Math.min(cfg.initialDelayMs * 2 ** exponent, cfg.maxDelayMs)
  // 对称抖动：落在 [1-r, 1+r] 上，再夹一次上限 ——
  // 乘法可能把已经到顶的值推出上限（r > 0 时），不夹就超了
  const jitter = 1 - cfg.jitterRatio + 2 * cfg.jitterRatio * cfg.random()
  return { delayMs: Math.min(exponential * jitter, cfg.maxDelayMs), fromServer: false }
}

/**
 * 跑一次，失败就按策略重试。
 *
 * **不可重试的错误原样抛出去**，不在这里吞掉 —— 降级链靠它决定要不要
 * 换下一个后端。这一层只回答一个问题：**同一个后端，要不要再问一次。**
 */
export async function retryCall<T>(who: string, cfg: ResolvedRetry, run: () => Promise<T>): Promise<T> {
  let lastErr: unknown

  for (let attempt = 0; attempt <= cfg.maxRetries; attempt++) {
    try {
      return await run()
    } catch (err) {
      lastErr = err

      // 不是「再试一次会不一样」的失败 —— 立刻交给降级链，别白等。
      // 凭据坏了、请求不合法、余额耗尽都属于这一类（判据见 http-error.ts）
      if (!isRetryable(err)) throw err
      if (attempt === cfg.maxRetries) throw err

      const planned = planDelay(err, attempt + 1, cfg)
      // 服务端要求的等待超过了我们愿意等的上限 → **放弃，交给降级链**。
      // 不是把它的要求截短 —— 那等于在它说「还没好」的时候硬问
      if (planned === undefined) throw err

      cfg.onRetry?.({
        who,
        code: err.code,
        attempt: attempt + 1,
        delayMs: planned.delayMs,
        fromServer: planned.fromServer,
      })
      await cfg.sleep(planned.delayMs)
    }
  }

  // 循环要么返回、要么抛出，走不到这里；留着是为了让类型收敛
  throw lastErr
}
