/**
 * JevLoop · 远端后端的失败分类
 *
 * ══════════════════════════════════════════════════════════════
 *  **按 `code` 分支，永远不要解析 `message`。**
 * ══════════════════════════════════════════════════════════════
 *
 * message 是给人看的，改措辞是常事；而按措辞分支的调用方会在**没人改
 * 调用方**的情况下静默失效。所以分类要有一个稳定的、机器可路由的名字。
 *
 * 形状抄 DSH 的 `HarnessError`（`packages/llm/llm/src/error.ts`），
 * 包括它那句注释：「Stable machine-routable failure class; route on
 * this, never by parsing `message`」。
 *
 * ── 为什么在 L0，而不是某个接缝旁边 ──────────────────────────────
 *
 * 这个仓库有**两条**通向后端的缝，两条都要这套分类：
 *
 *     判定后端   `Provider.decide()`    →  `provider-http.ts` / `provider-retry.ts`
 *     生成后端   `Generator.generate()` →  `llm.ts`
 *
 * 而它们是**同层**（都 L2），§11 不许互相依赖 —— 所以共用的东西必须下沉。
 * 这个文件零 IO、零 import，放 L0 名副其实。
 *
 * ⚠️ 名字里的 `Provider` 现在**同时指这两条缝**。改名要 grep 全部消费方
 *   （§8.12），所以留着旧名字，在这里说明。
 *
 * ── 哪些可重试，判断依据是什么 ──────────────────────────────────
 *
 * 判据是**「再试一次会不会有不一样的结果」**：
 *
 *   · 过载 / 限流 / 5xx / 超时 / 传输失败 → 会变，值得重试
 *   · 凭据坏了 → **每一次都一模一样地失败**，重试只是白等
 *   · 请求本身不合法（含上下文超限）→ 重试发的是同一个请求，结果不会变
 *   · 余额耗尽 → 同上
 *
 * DSH 对 `INVALID_CREDENTIAL` 的注释就是这个意思：「a malformed
 * credential fails identically on every attempt」。
 *
 * @module JevLoop/http-error
 */

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

/** 带分类的后端失败。两条缝都抛它，重试层按 `code` 决定要不要再来一次 */
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

/**
 * 一个非 2xx 响应属于哪一类。
 *
 * 形状照 DSH 的 `httpErrorCode`（`llm-deepseek/src/adapter.ts`），
 * 保留它对**顺序**的处理：先认少数几个确定的，再按区间兜底。
 *
 * @param status HTTP 状态码
 * @param detail 响应体前若干字符。**只在 400 时读**，用来分辨
 *   「上下文超限」和「请求不合法」—— 那两种该怎么处理完全相反。
 */
export function httpErrorCode(status: number, detail = ''): ProviderErrorCode {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 413) return 'INVALID_REQUEST'
  // 余额耗尽的措辞各家不同，但状态码多半是 402，也有用 429 的 ——
  // 所以先看措辞再看码，否则一个 429 的「余额不足」会被当成限流去重试
  if (looksLikeQuota(detail)) return 'QUOTA'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) {
    return looksLikeContextOverflow(detail) ? 'CONTEXT_WINDOW_EXCEEDED' : 'INVALID_REQUEST'
  }
  // ★ **≥ 500 一律 SERVER，含 529。** 实测 2026-09-21：托管 Jev 的
  //   529 `system_overloaded` 因为当时没有分类、没有重试，直接把整轮
  //   判定打到 mock 的恒定 0.5 上，而轨迹里只有一句 `degraded: true`。
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/** 余额 / 配额耗尽的措辞。**归到 QUOTA 是为了不重试** —— 它不会自己好 */
function looksLikeQuota(detail: string): boolean {
  return /insufficient|quota|balance|billing|credit/i.test(detail)
}

/** 上下文超限的措辞。归到这一类是为了**不重试**（同一个请求发多少次都一样） */
function looksLikeContextOverflow(detail: string): boolean {
  return /context[\s_-]*(length|window)|too\s+(large|long)\s+for|maximum\s+context/i.test(detail)
}

/**
 * `Retry-After` 头 → 毫秒。
 *
 * **两种格式都认**（RFC 9110）：秒数，或者一个 HTTP-date。认不出来返回
 * `undefined` —— 那时按本地退避算，而不是当成 0。
 */
export function parseRetryAfter(value: string | null, now = Date.now()): number | undefined {
  if (value === null) return undefined
  const v = value.trim()
  if (/^\d+$/.test(v)) {
    const ms = Number(v) * 1000
    return Number.isFinite(ms) && ms > 0 ? ms : undefined
  }
  const at = Date.parse(v)
  if (!Number.isFinite(at)) return undefined
  const ms = at - now
  return ms > 0 ? ms : undefined
}

/**
 * 把一个非 2xx 响应变成分类好的失败。
 *
 * 两条缝共用 —— 它们的 `fetch` 写法不同，但「失败长什么样」必须是同一个，
 * 否则同一个 529 在一条缝上可重试、在另一条上不可重试。
 */
export function httpFailure(
  who: string,
  status: number,
  body: string,
  retryAfterHeader: string | null,
): ProviderError {
  const code = httpErrorCode(status, body.slice(0, 200))
  const retryAfterMs = parseRetryAfter(retryAfterHeader)
  return new ProviderError(`${who} HTTP ${status} (${code}): ${body.slice(0, 200)}`, code, {
    status,
    ...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
  })
}

/**
 * 把 `fetch` 自己抛的错变成分类好的失败。
 *
 * 传输层的两类要**分得开**：超时是我们自己掐的（本地定时器），传输失败是
 * 根本连不上。两者都可重试，但排查方向完全不同 —— 前者要调 `timeoutMs`，
 * 后者要看网络和地址。
 */
export function transportFailure(who: string, err: unknown, timedOut: boolean, timeoutMs: number): ProviderError {
  if (err instanceof ProviderError) return err
  const code: ProviderErrorCode = timedOut ? 'TIMEOUT' : 'TRANSPORT'
  const what = timedOut ? `超时（${timeoutMs}ms）` : '连不上'
  return new ProviderError(`${who} ${what}：${(err as Error).message}`, code, { cause: err })
}
