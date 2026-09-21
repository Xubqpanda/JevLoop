/**
 * JevLoop · 一次调用里「握手」占多少
 *
 * ══════════════════════════════════════════════════════════════
 *  **为什么要有这个文件。**
 *
 *  「Jev 一次判定 330ms」这句话本身说明不了任何事 —— 它没有说那 330ms
 *  里有多少是**在算**，有多少是**在等一个来回**。而这两件事的优化方向
 *  完全相反：前者要换模型，后者要换部署方式（搬到本地就没有来回）。
 *
 *  实测（2026-09-21）把它拆开之后，结论是反直觉的：
 *
 *      一次判定 329ms = 216ms 往返 + 113ms 计算
 *      一次生成 2036ms =  76ms 往返 + 1960ms 计算
 *
 *  也就是说**判定那边的三分之二是在等包**，而它的往返比生成后端还贵
 *  近 3 倍（216ms vs 76ms）。「问一个小问题很便宜」在托管 API 上不成立 ——
 *  便宜的是**算**，不是**问**。
 * ══════════════════════════════════════════════════════════════
 *
 * ── 基线是怎么取的 ──────────────────────────────────────────────
 *
 * 打**同一个 URL、同样的鉴权头**，但 body 缺字段 —— 服务端反序列化时就
 * 拒掉，**永远走不到模型**。所以它量的是：DNS（复用连接时为 0）+
 * 传输 + 服务端鉴权与校验 + 返回。
 *
 * 这比拿一个 404 页面当基线准：那是另一条路径、另一层（边缘节点），
 * 可能根本不经过鉴权和校验。**基线必须和被测的东西走同一条路。**
 *
 * ⚠️ 它仍然是**减法**得到的估算，不是服务端自报的耗时。真正的分解只有
 *    服务端能给（`server-timing` 之类），这两个后端都不给。所以下面的
 *    `computeMs` 读作「总耗时减去同路径的无推理基线」，别读成「GPU 跑了多久」。
 *
 * 零依赖、零副作用（不打印、不退出），两台台子都 import 它。
 *
 * @module JevLoop/transport
 */

import { loadEnv } from '../src/index.ts'

export interface Floor {
  who: string
  url: string
  /** 中位数（毫秒）。**第一次被丢掉** —— 它含 TCP+TLS 握手，不是稳态 */
  medianMs: number
  samples: number
  /** 最后一次的 HTTP 状态码，用来确认「确实被拒了」而不是走了推理 */
  status: number
  detail: string
  /**
   * **这次基线成立吗。**
   *
   * ★ `false` 时调用方**不许**拿它去做减法 —— 而第一版没这个字段，
   *   于是在 `api.deepseek.com` 连不稳的那段时间里，一次 5 秒的
   *   「空 body 请求」被当成了「握手耗时」，算出来的「纯计算时间」是
   *   **负数**（-3268ms），而台子照印不误。
   *
   *   判据：状态码必须是 4xx（服务端**校验层**真的拒了），而且中位数要
   *   落在「一次往返」的量级里。拿不到这两条就说明这次量到的不是基线，
   *   是网络抖动或超时。
   */
  ok: boolean
  /** 不成立的原因，直接印给人看 */
  problem?: string
}

/** 判定后端的端点与基线请求体 */
export function decisionEndpoint(baseUrl: string): { url: string; headers: Record<string, string>; body: string } {
  loadEnv() // 写进 process.env。返回值只带**键名**（不包含值，免得密钥进日志）
  const key = process.env.TYPESAFE_API_KEY
  return {
    url: `${baseUrl.replace(/\/+$/, '')}/v1/systemone`,
    headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) },
    // 缺 `model` 和 `questions` → 服务端在校验层就拒掉
    body: JSON.stringify({}),
  }
}

/** 生成后端的端点与基线请求体 */
export function generationEndpoint(baseUrl: string): { url: string; headers: Record<string, string>; body: string } {
  loadEnv()
  const key = process.env.DEEPSEEK_API_KEY
  return {
    url: `${baseUrl.replace(/\/+$/, '')}/chat/completions`,
    headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) },
    body: JSON.stringify({}),
  }
}

/**
 * 量一次「同路径、不做推理」的往返。
 *
 * `warmup` 次先打，把 TCP + TLS 握手摊掉 —— 真实运行时连接是复用的
 * （undici 的连接池），把冷启动算进每次调用会把基线抬高一大截：
 * 实测冷启动 684ms 而稳态 216ms，差的是 DNS 38ms + TCP 249ms + TLS 468ms。
 */
export async function measureFloor(opts: {
  who: string
  url: string
  headers: Record<string, string>
  body: string
  samples?: number
}): Promise<Floor> {
  const samples = opts.samples ?? 6
  const xs: number[] = []
  let status = 0
  let detail = ''

  for (let i = 0; i < samples + 1; i++) {
    const t = performance.now()
    try {
      const res = await fetch(opts.url, { method: 'POST', headers: opts.headers, body: opts.body })
      detail = (await res.text()).replace(/\s+/g, ' ').slice(0, 70)
      status = res.status
      xs.push(performance.now() - t)
    } catch (err) {
      // 空 catch 说明：吞的是「这次探测本身失败」（网络断了）。它不该把
      // 测量带走 —— 少一个样本比整场挂掉好，而 `samples` 会把真相说出去。
      detail = (err as Error).message.slice(0, 70)
      xs.push(performance.now() - t)
    }
  }

  // 第 0 个是冷启动（含 DNS/TCP/TLS），丢掉
  const warm = xs.slice(1).sort((a, b) => a - b)
  const median = warm.length ? warm[warm.length >> 1]! : NaN

  // ★ 基线成不成立，在这里判 —— 判完才好意思让调用方做减法
  let problem: string | undefined
  if (!(status >= 400 && status < 500)) {
    problem = `基线请求没被校验层拒掉（HTTP ${status || '连不上'}）—— 这次量到的不是一次往返`
  } else if (!Number.isFinite(median) || median > 2000) {
    problem = `基线中位数 ${Math.round(median)}ms 超过一次往返的量级 —— 多半是超时或限流，不是基线`
  }

  return { who: opts.who, url: opts.url, medianMs: median, samples: warm.length, status, detail, ok: !problem, ...(problem ? { problem } : {}) }
}
