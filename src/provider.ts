/**
 * nanojev · Provider
 *
 * 内核只认一件事：`{state, questions} → {answers}`。
 * 谁能提供这个，谁就能当判定后端。
 *
 *   JevProvider   → https://api.typesafe.ai   （官方，需要 key）
 *   LayaProvider  → 本地 sidecar               （开源权重，免费）
 *   MockProvider  → 保守占位答案                （零依赖，离线可跑）
 *
 * 后两个的存在是刻意的：**没有 API key 的人也必须能在 30 秒内跑起来。**
 */

import type { Provider, DecideRequest, DecideResponse, Answer, AnswerSet, Question } from './types.ts'

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
      if (!res.ok) throw new Error(`${this.name} HTTP ${res.status}: ${text.slice(0, 200)}`)

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
    if (a.type === 'noul' || typeof a.noul === 'number') {
      answers[id] = { type: 'noul', noul: clamp(Number(a.noul) || 0, 0, 1) }
      continue
    }
    if (a.type === 'score' || typeof a.score === 'number') {
      const p = numMap(a.probabilities)
      answers[id] = {
        type: 'score',
        score: Number(a.score) || 0,
        legend: a.legend ?? {},
        probabilities: p,
        confidence: typeof a.confidence === 'number' ? a.confidence : maxOf(p),
      }
      continue
    }
    if (a.type === 'choice' || typeof a.choice === 'string') {
      const p = numMap(a.probabilities)
      answers[id] = {
        type: 'choice',
        choice: String(a.choice ?? ''),
        probabilities: p,
        confidence: typeof a.confidence === 'number' ? a.confidence : maxOf(p),
      }
      continue
    }
    dropped.push(id)
  }
  return { answers, dropped }
}

function normalizeUsage(u: any): DecideResponse['usage'] {
  if (!u || typeof u !== 'object') return undefined
  return {
    ...(typeof u.input_tokens === 'number' ? { input_tokens: u.input_tokens } : {}),
    ...(typeof u.output_tokens === 'number' ? { output_tokens: u.output_tokens } : {}),
    ...(u.estimated === true ? { estimated: true } : {}),
  }
}

// ═══════════════════════════════════════════════════════════
// Mock —— 保守占位，故意"不知道"
//
// 这里**故意不做启发式猜测**。
// 猜得越像，越容易让人误以为判定是对的。
// 保守答案（概率 0.5、第一个选项、低置信度）会让 policy 的
// 置信度门限自动走到 escalate —— 这正是框架要演示的事：
//
//     **判定质量不够时，正确的行为是把决定交回上层，而不是硬猜。**
// ═══════════════════════════════════════════════════════════

export class MockProvider implements Provider {
  readonly name = 'mock'
  #latencyMs: number

  constructor(opts: { latencyMs?: number } = {}) {
    this.#latencyMs = opts.latencyMs ?? 3
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    const t0 = performance.now()
    if (this.#latencyMs > 0) await new Promise((r) => setTimeout(r, this.#latencyMs))

    const answers: AnswerSet = {}
    for (const [id, q] of Object.entries(req.questions)) answers[id] = conservative(q);

    return {
      answers,
      provider: this.name,
      model: 'mock',
      latencyMs: performance.now() - t0,
      degraded: true,
      warnings: ['未接真实判定后端：返回的是保守占位答案，策略会因置信度不足而交回上层'],
    }
  }
}

/** 三个原语的保守答案：不表态，让 policy 去处理 */
function conservative(q: Question): Answer {
  if (q.type === 'noul') return { type: 'noul', noul: 0.5 }
  if (q.type === 'score') {
    const mid = (q.criteria.length - 1) / 2
    const legend: Record<string, string> = {}
    const probabilities: Record<string, number> = {}
    q.criteria.forEach((lvl, i) => {
      legend[String(i)] = lvl
      probabilities[String(i)] = 1 / q.criteria.length
    })
    return { type: 'score', score: mid, legend, probabilities, confidence: 1 / q.criteria.length }
  }
  const ids = Object.keys(q.criteria)
  const probabilities: Record<string, number> = {}
  for (const id of ids) probabilities[id] = 1 / Math.max(1, ids.length);
  return { type: 'choice', choice: ids[0] ?? '', probabilities, confidence: 1 / Math.max(1, ids.length) }
}

// ═══════════════════════════════════════════════════════════
// Fallback —— 主 Provider 挂了怎么办
// ═══════════════════════════════════════════════════════════

export class FallbackProvider implements Provider {
  readonly name: string
  #chain: Provider[]
  #onFallback?: (err: unknown, from: string, to: string) => void

  constructor(chain: Provider[], onFallback?: (err: unknown, from: string, to: string) => void) {
    if (!chain.length) throw new Error('FallbackProvider 至少需要一个 Provider')
    this.#chain = chain
    this.name = chain.map((p) => p.name).join('→')
    this.#onFallback = onFallback
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    let lastErr: unknown
    for (let i = 0; i < this.#chain.length; i++) {
      const p = this.#chain[i]!
      try {
        const res = await p.decide(req)
        if (i > 0) {
          res.degraded = true
          res.warnings = [...(res.warnings ?? []), `主 Provider 失败，降级到 ${p.name}`]
        }
        return res
      } catch (err) {
        lastErr = err
        if (this.#chain[i + 1]) this.#onFallback?.(err, p.name, this.#chain[i + 1]!.name)
      }
    }
    throw lastErr
  }
}

// ── 工具 ─────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

function numMap(v: any): Record<string, number> {
  const out: Record<string, number> = {}
  for (const [k, n] of Object.entries(v ?? {})) out[k] = Number(n) || 0;
  return out
}

const maxOf = (m: Record<string, number>): number => {
  const vals = Object.values(m)
  return vals.length ? Math.max(...vals) : 0
}
