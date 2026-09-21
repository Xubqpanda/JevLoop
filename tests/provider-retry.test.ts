/**
 * 判定后端重试的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这一层补的是一个实测出来的窟窿：托管 Jev 的 `529 system_overloaded`
 * 因为没有重试，直接把整轮判定打到 mock 的恒定 0.5 上（见
 * `provider-retry.ts` 的文件头）。所以这里要钉住的是**两类错误的区别**：
 * 会变的值得重试，不会变的一次都不该等。
 *
 * 时间全部注入（`sleep` 和 `random`）—— 靠真实 `setTimeout` 的话，
 * 测退避就变成测机器有多快。
 *
 * @module JevLoop/provider-retry.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { ProviderError, isRetryable, RETRYABLE_CODES } from '../src/seam-provider.ts'
import { RetryingProvider, type RetryInfo } from '../src/provider-retry.ts'
import { httpErrorCode, parseRetryAfter } from '../src/provider-http.ts'
import type { DecideRequest, DecideResponse, Provider } from '../src/seam-provider.ts'

const REQ: DecideRequest = { state: {}, questions: {} }
const OK: DecideResponse = { answers: {}, provider: 'inner', latencyMs: 1 }

/** 一个每次调用都按剧本走的假后端 */
function scripted(steps: (Error | 'ok')[]): { provider: Provider; calls: () => number } {
  let i = 0
  return {
    calls: () => i,
    provider: {
      name: 'inner',
      async decide(): Promise<DecideResponse> {
        const step = steps[i++] ?? steps[steps.length - 1]!
        if (step === 'ok') return OK
        throw step
      },
    },
  }
}

/** 记录等待的毫秒数，不真的等 */
function fakeSleep(): { slept: number[]; sleep: (ms: number) => Promise<void> } {
  const slept: number[] = []
  return { slept, sleep: async (ms: number) => void slept.push(ms) }
}

/** 固定随机数 → 抖动可预测 */
const fixedRandom = (v: number) => () => v

// ═══════════════════════════════════════════════════════════
// 分类：HTTP 状态 → code
// ═══════════════════════════════════════════════════════════

test('★ 529（system_overloaded）归到 SERVER，而且**可重试**', () => {
  // 这就是实测把整轮打到 mock 上的那个状态码
  assert.equal(httpErrorCode(529), 'SERVER')
  assert.equal(isRetryable(new ProviderError('x', httpErrorCode(529))), true)
})

test('状态码 → 分类的完整对照', () => {
  assert.equal(httpErrorCode(401), 'AUTH')
  assert.equal(httpErrorCode(403), 'AUTH')
  assert.equal(httpErrorCode(413), 'INVALID_REQUEST')
  assert.equal(httpErrorCode(429), 'RATE_LIMIT')
  assert.equal(httpErrorCode(500), 'SERVER')
  assert.equal(httpErrorCode(503), 'SERVER')
  assert.equal(httpErrorCode(404), 'HTTP_404')
})

test('400 要分得开「上下文超限」和「请求不合法」—— 两者该怎么办完全相反', () => {
  assert.equal(httpErrorCode(400, 'context length exceeded'), 'CONTEXT_WINDOW_EXCEEDED')
  assert.equal(httpErrorCode(400, 'maximum context window is 8192'), 'CONTEXT_WINDOW_EXCEEDED')
  assert.equal(httpErrorCode(400, 'missing field: model'), 'INVALID_REQUEST')
  assert.equal(httpErrorCode(400), 'INVALID_REQUEST')
})

test('★ 余额耗尽可能报 429，但那不是限流 —— 认措辞优先于认码', () => {
  // 认成 RATE_LIMIT 的话会一直重试一个不会自己好的东西
  assert.equal(httpErrorCode(429, 'insufficient balance'), 'QUOTA')
  assert.equal(httpErrorCode(402, 'quota exceeded'), 'QUOTA')
  assert.equal(httpErrorCode(429, 'rate limit exceeded'), 'RATE_LIMIT')
})

test('可重试集是封闭的，而且**不含**那几类再试也没用的', () => {
  const yes = ['RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT', 'EMPTY_RESPONSE']
  const no = ['AUTH', 'QUOTA', 'INVALID_REQUEST', 'CONTEXT_WINDOW_EXCEEDED', 'HTTP_404']
  for (const c of yes) assert.ok(RETRYABLE_CODES.has(c), `${c} 应当可重试`)
  for (const c of no) assert.equal(RETRYABLE_CODES.has(c), false, `${c} 不该可重试 —— 再试一次结果一样`)
  // 认不出的错误一律不重试：在不确定上再赌一次没有依据
  assert.equal(isRetryable(new Error('随便什么错')), false)
})

test('Retry-After 两种格式都认，认不出来返回 undefined（不是 0）', () => {
  assert.equal(parseRetryAfter('3'), 3000)
  assert.equal(parseRetryAfter(' 2 '), 2000)
  assert.equal(parseRetryAfter('0'), undefined, '0 秒不是「立刻重试」，是没有说')
  assert.equal(parseRetryAfter(null), undefined)
  assert.equal(parseRetryAfter('不是时间'), undefined)
  const now = Date.parse('2026-09-21T10:00:00Z')
  assert.equal(parseRetryAfter('Sun, 21 Sep 2026 10:00:05 GMT', now), 5000)
  assert.equal(parseRetryAfter('Sun, 21 Sep 2026 09:59:00 GMT', now), undefined, '已经过去了')
})

// ═══════════════════════════════════════════════════════════
// 重试行为
// ═══════════════════════════════════════════════════════════

test('可重试的失败会重试到上限，然后抛出', async () => {
  const { provider, calls } = scripted([new ProviderError('overloaded', 'SERVER')])
  const { slept, sleep } = fakeSleep()
  const p = new RetryingProvider(provider, { maxRetries: 2, sleep, random: fixedRandom(0.5) })

  await assert.rejects(() => p.decide(REQ), /overloaded/)
  assert.equal(calls(), 3, '首次 + 2 次重试')
  assert.equal(slept.length, 2)
})

test('★ 不可重试的失败**一次都不等**，直接交给降级链', async () => {
  const { provider, calls } = scripted([new ProviderError('bad key', 'AUTH')])
  const { slept, sleep } = fakeSleep()
  const p = new RetryingProvider(provider, { maxRetries: 3, sleep })

  await assert.rejects(() => p.decide(REQ), /bad key/)
  assert.equal(calls(), 1, '凭据坏了，重试每一次都一样')
  assert.deepEqual(slept, [], '一秒都不该等')
})

test('中间成功就返回，不再重试', async () => {
  const { provider, calls } = scripted([new ProviderError('overloaded', 'SERVER'), 'ok'])
  const { slept, sleep } = fakeSleep()
  const p = new RetryingProvider(provider, { maxRetries: 3, sleep, random: fixedRandom(0.5) })

  assert.deepEqual(await p.decide(REQ), OK)
  assert.equal(calls(), 2)
  assert.equal(slept.length, 1)
})

test('maxRetries: 0 = 不重试', async () => {
  const { provider, calls } = scripted([new ProviderError('overloaded', 'SERVER')])
  const { sleep } = fakeSleep()
  await assert.rejects(() => new RetryingProvider(provider, { maxRetries: 0, sleep }).decide(REQ))
  assert.equal(calls(), 1)
})

// ═══════════════════════════════════════════════════════════
// 退避
// ═══════════════════════════════════════════════════════════

test('★ 指数退避，被上限夹住，抖动是对称的', async () => {
  const mk = new ProviderError('overloaded', 'SERVER')
  const { provider } = scripted([mk])
  const { slept, sleep } = fakeSleep()
  // random 恒 0.5 → 抖动系数 = 1 - r + 2r·0.5 = 1，即无偏移
  await assert.rejects(() =>
    new RetryingProvider(provider, {
      maxRetries: 4, initialDelayMs: 100, maxDelayMs: 500, jitterRatio: 0.2, sleep, random: fixedRandom(0.5),
    }).decide(REQ),
  )
  assert.deepEqual(slept, [100, 200, 400, 500], '翻倍，到上限就不再涨')
})

test('抖动在 [1-r, 1+r] 上，而且**夹上限在抖动之后**再做一次', async () => {
  const mk = new ProviderError('overloaded', 'SERVER')

  // 向下抖：random 0 → 系数 1-r = 0.75（r = 0.25）
  const a = fakeSleep()
  await assert.rejects(() =>
    new RetryingProvider(scripted([mk]).provider, {
      maxRetries: 1, initialDelayMs: 100, maxDelayMs: 5000, jitterRatio: 0.25, sleep: a.sleep, random: fixedRandom(0),
    }).decide(REQ),
  )
  assert.deepEqual(a.slept, [75])

  // 向上抖：random 1 → 系数 1+r = 1.25。
  // initialDelay 已经等于上限时，乘法会把它推到 625 —— 必须被夹回 500
  const b = fakeSleep()
  await assert.rejects(() =>
    new RetryingProvider(scripted([mk]).provider, {
      maxRetries: 1, initialDelayMs: 500, maxDelayMs: 500, jitterRatio: 0.25, sleep: b.sleep, random: fixedRandom(1),
    }).decide(REQ),
  )
  assert.deepEqual(b.slept, [500], '抖动之后再夹一次，否则会超过声明的上限')
})

test('★ 服务端说了等多久就等多久 —— 不算退避', async () => {
  const after = new ProviderError('slow down', 'RATE_LIMIT', { retryAfterMs: 1234 })
  const { slept, sleep } = fakeSleep()
  const got: RetryInfo[] = []
  await assert.rejects(() =>
    new RetryingProvider(scripted([after]).provider, {
      maxRetries: 1, initialDelayMs: 100, maxDelayMs: 5000, sleep, random: fixedRandom(0.5),
      onRetry: (i) => got.push(i),
    }).decide(REQ),
  )
  assert.deepEqual(slept, [1234], '服务端知道它什么时候能好，我们不知道')
  assert.equal(got[0]!.fromServer, true)
})

test('★ 服务端要等的超过我们愿意等的 → **放弃这一跳**，不截短', async () => {
  const after = new ProviderError('come back later', 'SERVER', { retryAfterMs: 60_000 })
  const { provider, calls } = scripted([after])
  const { slept, sleep } = new (class { slept: number[] = []; sleep = async (ms: number) => void this.slept.push(ms) })()
  const p = new RetryingProvider(provider, { maxRetries: 3, maxDelayMs: 5000, sleep })

  await assert.rejects(() => p.decide(REQ), /come back later/)
  assert.equal(calls(), 1, '不在它说「还没好」的时候硬问')
  assert.deepEqual(slept, [])
})

test('onRetry 报出每一次安排 —— 「特别慢」和「主后端在过载」要分得开', async () => {
  const mk = new ProviderError('overloaded', 'SERVER')
  const { sleep } = fakeSleep()
  const got: RetryInfo[] = []
  await assert.rejects(() =>
    new RetryingProvider(scripted([mk]).provider, {
      maxRetries: 2, initialDelayMs: 100, sleep, random: fixedRandom(0.5), onRetry: (i) => got.push(i),
    }).decide(REQ),
  )
  assert.deepEqual(got.map((g) => g.attempt), [1, 2])
  assert.deepEqual(got.map((g) => g.delayMs), [100, 200])
  assert.equal(got[0]!.provider, 'inner')
  assert.equal(got[0]!.code, 'SERVER')
})

test('包装器的名字和里层一样 —— 界面上那句「用哪些后端」不该被重试改变', () => {
  assert.equal(new RetryingProvider(scripted(['ok']).provider).name, 'inner')
})

test('超时（TIMEOUT）和连不上（TRANSPORT）都可重试 —— 但排查方向不同', () => {
  for (const code of ['TIMEOUT', 'TRANSPORT'] as const) {
    assert.equal(isRetryable(new ProviderError(code, code)), true, code)
  }
})
