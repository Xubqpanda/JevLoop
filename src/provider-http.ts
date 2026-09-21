/**
 * JevLoop · HTTP 判定后端与响应归一化
 *
 * **从 `provider.ts` 切出来** —— 那个文件住着**四件事**
 * （HTTP 判定后端 / 响应归一化 / Mock / 降级链），按 docs/CODE-STYLE.md §12
 * 「一句话说不完就是住着两件以上」该拆。接缝是**各自的契约形状不同**：
 *  * HTTP 管线协议与解析（含只服务它的 4 个工具函数），归一化把畸形响应如实报出来
 *
 * @module JevLoop/provider-http
 */

import { ProviderError, type ProviderErrorCode, type Provider, type DecideRequest, type DecideResponse } from './seam-provider.ts'
import type { AnswerSet } from './vocab.ts'

// ═══════════════════════════════════════════════════════════
// 失败分类：HTTP 状态 → 稳定的 code
// ═══════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════
// HTTP —— 说 Jev 线格式（官方 API 和本地 Laya 共用这一个）
// ═══════════════════════════════════════════════════════════

export interface HttpProviderOptions {
  baseUrl: string
  apiKey?: string
  name?: string
  defaultModel?: string
  timeoutMs?: number
}

/**
 * 判定后端：`POST <baseUrl>/v1/systemone`，一次请求拿到全部问题的答案。
 *
 * 线协议是 `{model, state, questions}` → `{answers, model, usage}`。
 * **响应永远要过 `normalizeAnswers()`** —— 后端返回畸形 JSON 比整个挂掉
 * 更危险，因为挂掉会被 `FallbackProvider` 抓到，畸形响应不会（见 §8.10）。
 */
export class HttpProvider implements Provider {
  readonly name: string
  #baseUrl: string
  #apiKey?: string
  #model?: string
  #timeout: number

  constructor(opts: HttpProviderOptions) {
    this.#baseUrl = opts.baseUrl.replace(/\/+$/, '')
    this.#apiKey = opts.apiKey
    this.#model = opts.defaultModel
    this.name = opts.name ?? 'http'
    this.#timeout = opts.timeoutMs ?? 30_000
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    const t0 = performance.now()
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), req.timeoutMs ?? this.#timeout)

    try {
      const res = await fetch(`${this.#baseUrl}/v1/systemone`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(this.#apiKey ? { authorization: `Bearer ${this.#apiKey}` } : {}),
        },
        body: JSON.stringify({ model: req.model ?? this.#model, state: req.state, questions: req.questions }),
        signal: ctrl.signal,
      })

      const text = await res.text()
      if (!res.ok) {
        // ★ 抛**带分类**的错误，而不是一句拼出来的 message。
        //
        //   以前这里是 `throw new Error(\`${this.name} HTTP ${res.status}: …\`)`
        //   —— 状态码只活在**给人看的那句话里**，于是调用方要分支就只能
        //   去解析 message，而那正是「改一次措辞就静默失效」的写法。
        const code = httpErrorCode(res.status, text.slice(0, 200))
        const retryAfterMs = parseRetryAfter(res.headers.get('retry-after'))
        throw new ProviderError(`${this.name} HTTP ${res.status} (${code}): ${text.slice(0, 200)}`, code, {
          status: res.status,
          ...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
        })
      }

      const body: any = JSON.parse(text)
      const { answers, dropped } = normalizeAnswers(body.answers ?? {})

      // 覆盖度检查：后端少给了答案，**不能假装成功**。
      // 静默丢答案会让一次彻底的解析失败在日志上和一次正常判定长得一模一样。
      const missing = Object.keys(req.questions).filter((id) => !(id in answers))
      const notes: string[] = []
      if (Array.isArray(body.warnings)) notes.push(...body.warnings.map(String))
      if (missing.length) {
        notes.push(`后端未返回 ${missing.length}/${Object.keys(req.questions).length} 个答案：${missing.join(', ')}`)
      }
      if (dropped.length) notes.push(`无法识别的答案已丢弃：${dropped.join(', ')}`)

      return {
        answers,
        provider: this.name,
        model: typeof body.model === 'string' ? body.model : undefined,
        latencyMs: performance.now() - t0,
        usage: normalizeUsage(body.usage),
        degraded: missing.length > 0 || dropped.length > 0,
        ...(notes.length ? { warnings: notes } : {}),
      }
    } catch (err) {
      if (err instanceof ProviderError) throw err
      // 传输层的两类失败要**分得开**：超时是我们自己掐的（本地定时器），
      // 传输失败是根本连不上。两者都可重试，但排查方向完全不同 ——
      // 前者要调 `timeoutMs`，后者要看网络和地址。
      const aborted = ctrl.signal.aborted
      const code: ProviderErrorCode = aborted ? 'TIMEOUT' : 'TRANSPORT'
      throw new ProviderError(
        `${this.name} ${aborted ? `超时（${req.timeoutMs ?? this.#timeout}ms）` : '连不上'}：${(err as Error).message}`,
        code,
        { cause: err },
      )
    } finally {
      clearTimeout(timer)
    }
  }
}

/** 把后端返回补齐成规范答案。丢掉认不出来的，但**把丢掉的报出来**。 */
export function normalizeAnswers(raw: Record<string, any>): { answers: AnswerSet; dropped: string[] } {
  const answers: AnswerSet = {}
  const dropped: string[] = []

  for (const [id, a] of Object.entries(raw ?? {})) {
    if (!a || typeof a !== 'object') {
      dropped.push(id)
      continue
    }
    // ★ 判定要认**值**在不在，不能只认类型标签。
    //
    //   以前三个分支的写法是 `a.type === 'noul' || typeof a.noul === 'number'`：
    //   一个只有标签、没有值的畸形答案（`{"type":"noul"}`）会走到
    //   `Number(undefined) || 0`，被**补成 `noul: 0`**。它既不进 `dropped`
    //   也不进 `missing`，于是 `degraded` 保持 false —— 在日志上和一次正常
    //   判定完全一样。而 `0` 在策略里是一个**明确的否定**（"不需要工具"、
    //   "没成功"），比整个后端挂掉危险得多：挂掉会被 FallbackProvider 抓到，
    //   伪造的 0 不会。这是 §8.10「不假装成功」在解析层的漏洞。
    //
    //   标签仍然可以不写（`{ noul: 0.7 }` 照收）—— 容忍的是缺标签，不是缺值。
    if (a.type === 'noul' || a.noul !== undefined) {
      if (!isNum(a.noul)) {
        dropped.push(id)
        continue
      }
      answers[id] = { type: 'noul', noul: clamp(a.noul, 0, 1) }
      continue
    }
    if (a.type === 'score' || a.score !== undefined) {
      if (!isNum(a.score)) {
        dropped.push(id)
        continue
      }
      const p = numMap(a.probabilities)
      answers[id] = {
        type: 'score',
        score: a.score,
        legend: a.legend ?? {},
        probabilities: p,
        confidence: isNum(a.confidence) ? a.confidence : maxOf(p),
      }
      continue
    }
    if (a.type === 'choice' || a.choice !== undefined) {
      // 空字符串不是一个选项 —— 放过去会让 `pickInput` 返回 `''`，
      // 上游拿着它去拼路径，错误要隔好几层才暴露出来。
      if (typeof a.choice !== 'string' || a.choice === '') {
        dropped.push(id)
        continue
      }
      const p = numMap(a.probabilities)
      answers[id] = {
        type: 'choice',
        choice: a.choice,
        probabilities: p,
        confidence: isNum(a.confidence) ? a.confidence : maxOf(p),
      }
      continue
    }
    dropped.push(id)
  }
  return { answers, dropped }
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function normalizeUsage(u: any): DecideResponse['usage'] {
  if (!u || typeof u !== 'object') return undefined
  return {
    ...(typeof u.input_tokens === 'number' ? { input_tokens: u.input_tokens } : {}),
    ...(typeof u.output_tokens === 'number' ? { output_tokens: u.output_tokens } : {}),
    ...(u.estimated === true ? { estimated: true } : {}),
  }
}

// ── 工具 ─────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

function numMap(v: any): Record<string, number> {
  const out: Record<string, number> = {}
  for (const [k, n] of Object.entries(v ?? {})) out[k] = Number(n) || 0
  return out
}

const maxOf = (m: Record<string, number>): number => {
  const vals = Object.values(m)
  return vals.length ? Math.max(...vals) : 0
}
