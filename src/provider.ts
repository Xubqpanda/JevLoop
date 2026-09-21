/**
 * JevLoop · Provider
 *
 * 内核只认一件事：`{state, questions} → {answers}`。
 * 谁能提供这个，谁就能当判定后端。
 *
 *   JevProvider   → https://api.typesafe.ai   （官方，需要 key）
 *   LayaProvider  → 本地 sidecar               （开源权重，免费）
 *   MockProvider  → 保守占位答案                （零依赖，离线可跑）
 *
 * 后两个的存在是刻意的：**没有 API key 的人也必须能在 30 秒内跑起来。**
 *
 * @module JevLoop/provider
 */

import type { Provider, DecideRequest, DecideResponse } from './seam-provider.ts'
import type { Answer, AnswerSet, Question } from './vocab.ts'

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

// ═══════════════════════════════════════════════════════════
// Mock —— 保守占位，故意"不知道"
//
// 这里**故意不做启发式猜测**。
// 猜得越像，越容易让人误以为判定是对的。
// 保守答案（概率 0.5、第一个选项、低置信度）会落在各条 policy 的兜底分支上，
// 这正是框架要演示的事：
//
//     **判定质量不够时，正确的行为是把决定交回上层，而不是硬猜。**
//
// ⚠️ 别把这句话读成「一定会走到 escalate」——`0.5` 恰好**越过**任何阈值为 0.5 的门
// （`>=` 是闭区间）。实测 Mock 的真实结果是：
// gradeRisk→ask_human、isDone→keep_going、canDeliver→revise、needsTool→use_tool。
// 所以**凡是「放行 / 继续」那一条，阈值必须严格大于 0.5** ——
// 否则「毫无信息」会被读成「同意」。`T.stepOk` 因此从 0.5 提到了 0.6
// （见 `docs/REVIEWS-2026-09-21-round4.md` 方案 2）。
// ═══════════════════════════════════════════════════════════

/**
 * 保守判定后端：不接任何模型，对每个问题返回「不表态」的答案
 * （`noul` 给 0.5、`choice` 给第一个选项、`score` 给中间档）。
 *
 * 它让整条 loop **离线可跑**，并演示「判定质量不够时，正确的行为是把决定
 * 交回上层」。返回的 `degraded` 永远是 `true` 并带一条 warning ——
 * 用 Mock 跑出来的数字**不能**当成判定质量的证据。
 */
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
    for (const [id, q] of Object.entries(req.questions)) answers[id] = conservative(q)

    return {
      answers,
      provider: this.name,
      model: 'mock',
      latencyMs: performance.now() - t0,
      degraded: true,
      // ★ 这句以前说「策略会因置信度不足而交回上层」—— **那是过度承诺**。
      //   实测跑全部七个节点，只有 3 个真的交回上层（pickTool / gradeRisk / stepOk），
      //   另外 4 个（needsTool / pickInput / isDone / canDeliver）会继续往下走，
      //   因为 0.5 恰好**越过**任何阈值为 0.5 的门（`>=` 是闭区间）。
      //   警告的用处就是别让人误以为 Mock 的行为是判定结果，所以它必须说实测到的那个。
      warnings: [
        '未接真实判定后端：返回的是保守占位答案（noul 恒 0.5、choice 取第一个、score 取中间档）。' +
          '各节点落在自己的兜底分支上，但**不是**全部交回上层 —— 0.5 会越过阈值为 0.5 的门，' +
          '所以 7 个节点里只有 pickTool / gradeRisk / stepOk 会停下。用 Mock 跑出来的结果不是判定质量的证据。',
      ],
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
  for (const id of ids) probabilities[id] = 1 / Math.max(1, ids.length)
  return { type: 'choice', choice: ids[0] ?? '', probabilities, confidence: 1 / Math.max(1, ids.length) }
}

// ═══════════════════════════════════════════════════════════
// Fallback —— 主 Provider 挂了怎么办
// ═══════════════════════════════════════════════════════════

/**
 * 链式兜底：按顺序试每一个后端，第一个成功的胜出。
 *
 * 全部失败时**记原计划用的那个后端名**（不覆盖成 `none`）再抛出，
 * 这样日志里看得出「本来想用谁」—— 覆盖掉就等于把排查线索删了（§8.10）。
 */
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
  for (const [k, n] of Object.entries(v ?? {})) out[k] = Number(n) || 0
  return out
}

const maxOf = (m: Record<string, number>): number => {
  const vals = Object.values(m)
  return vals.length ? Math.max(...vals) : 0
}
