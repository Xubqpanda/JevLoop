/**
 * 核心不变量的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这些逻辑**没有类型保护** —— 它们全是运行时行为，tsc 通过不代表行为正确。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { resolvePolicy, gte, probGte, scoreGte, picked } from '../src/policy.ts'
import { Meter } from '../src/meter.ts'
import { normalizeAnswers, MockProvider } from '../src/provider.ts'
import { validate, clip, estimateTokens } from '../src/budget.ts'
import { defineDecision, isDecision, noul, choice } from '../src/types.ts'
import type { AnswerSet } from '../src/types.ts'

const ans = (o: Record<string, unknown>): AnswerSet => o as AnswerSet

// ═══════════════════════════════════════════════════════════
// policy
// ═══════════════════════════════════════════════════════════

test('策略按顺序求值，第一个命中的胜出', () => {
  const out = resolvePolicy(
    [
      { when: probGte('a', 0.9), action: 'first' },
      { when: probGte('a', 0.5), action: 'second' },
      { action: 'fallback' },
    ],
    ans({ a: { type: 'noul', noul: 0.95 } }),
  )
  assert.equal(out.action, 'first')
  assert.equal(out.ruleIndex, 0)
})

test('没有规则命中且没有兜底 → escalate（安全默认）', () => {
  const out = resolvePolicy([{ when: probGte('a', 0.99), action: 'x' }], ans({ a: { type: 'noul', noul: 0.1 } }))
  assert.equal(out.action, 'escalate')
  assert.equal(out.ruleIndex, -1)
})

test('兜底规则不在末尾时告警 —— 它后面的规则永远不会执行', () => {
  const out = resolvePolicy(
    [{ action: 'catch' }, { when: probGte('a', 0), action: 'never' }],
    ans({ a: { type: 'noul', noul: 1 } }),
  )
  assert.equal(out.action, 'catch')
  assert.ok(out.warnings.some((w) => w.code === 'catch_all_not_last'))
})

test('策略函数抛异常被记录，不和「条件不满足」混为一谈', () => {
  const out = resolvePolicy(
    [{ when: (a) => (a as any).missing.noul > 0.5, action: 'boom' }, { action: 'ok' }],
    ans({}),
  )
  assert.equal(out.action, 'ok')
  assert.ok(out.warnings.some((w) => w.code === 'when_threw'))
})

test('谓词帮手都对', () => {
  const a = ans({
    n: { type: 'noul', noul: 0.8 },
    s: { type: 'score', score: 2.5, legend: {}, probabilities: {}, confidence: 0.9 },
    c: { type: 'choice', choice: 'b', probabilities: { a: 0.2, b: 0.75 }, confidence: 0.75 },
  })
  assert.equal(probGte('n', 0.75)(a), true)
  assert.equal(scoreGte('s', 2)(a), true)
  assert.equal(picked('c', 'b')(a), true)
  assert.equal(gte('c', 0.7)(a), true)
  assert.equal(gte('c', 0.8)(a), false)
})

// ═══════════════════════════════════════════════════════════
// meter —— 这是 JevLoop 的全部主张，算错了整个卖点就没了
// ═══════════════════════════════════════════════════════════

const fakeDecision = (latencyMs: number, escalate = false) =>
  ({ id: 'x', step: 1, state: {}, questions: {}, answers: {}, action: 'a', reason: '', latencyMs, provider: 'p', degraded: false, escalate }) as never

test('meter 分开统计判定与模型调用', () => {
  const m = new Meter()
  m.recordDecision(1, fakeDecision(10))
  m.recordDecision(1, fakeDecision(20))
  m.recordModelCall(1, { kind: 'gen', latencyMs: 600 })

  const s = m.stats
  assert.equal(s.decisions, 2)
  assert.equal(s.decisionMs, 30)
  assert.equal(s.modelCalls, 1)
  assert.equal(s.modelMs, 600)
  assert.equal(s.ratio, 2)
  assert.equal(s.decisionShare, 30 / 630)
})

test('没有模型调用时比值是 Infinity，不是 0 或 NaN', () => {
  const m = new Meter()
  m.recordDecision(1, fakeDecision(5))
  assert.equal(m.stats.ratio, Infinity)
  assert.equal(m.stats.decisionShare, 1)
})

test('meter 统计 escalate 次数', () => {
  const m = new Meter()
  m.recordDecision(1, fakeDecision(1, true))
  m.recordDecision(1, fakeDecision(1, false))
  assert.equal(m.stats.escalated, 1)
})

// ═══════════════════════════════════════════════════════════
// provider —— 不假装成功
// ═══════════════════════════════════════════════════════════

test('normalizeAnswers 认三种类型，并报出丢弃的', () => {
  const { answers, dropped } = normalizeAnswers({
    a: { type: 'noul', noul: 0.7 },
    b: { type: 'choice', choice: 'x', probabilities: { x: 0.9, y: 0.1 } },
    c: { type: 'score', score: 1.5, legend: {}, probabilities: { '0': 0.5, '1': 0.5 } },
    d: { type: 'weird' },
    e: null,
  })
  assert.equal(Object.keys(answers).length, 3)
  assert.deepEqual(dropped.sort(), ['d', 'e'])
  // choice 缺 confidence 时用最大概率补
  assert.equal((answers.b as any).confidence, 0.9)
})

test('Mock 返回保守答案 —— 故意不猜', async () => {
  const r = await new MockProvider({ latencyMs: 0 }).decide({
    state: {},
    questions: { n: noul('q?'), c: choice('q?', { x: '1', y: '2' }) },
  })
  assert.equal((r.answers.n as any).noul, 0.5)
  assert.equal((r.answers.c as any).confidence, 0.5)
  // 必须标成 degraded，不能让调用方以为这是真判定
  assert.equal(r.degraded, true)
})

// ═══════════════════════════════════════════════════════════
// budget
// ═══════════════════════════════════════════════════════════

test('clip 不超过预算，且提示说的是丢掉的字符数', () => {
  const out = clip('x'.repeat(100), 50)
  assert.ok(out.length <= 50)
  assert.match(out, /…\[\+\d+\]/)
})

test('clip 对短文本原样返回', () => {
  assert.equal(clip('hi', 50), 'hi')
})

test('中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token', () => {
  assert.equal(estimateTokens('一'.repeat(100)), 100)
  assert.equal(estimateTokens('a'.repeat(400)), 100)
})

test('选项超过安全线时告警', () => {
  const many: Record<string, string> = {}
  for (let i = 0; i < 30; i++) many[`opt${i}`] = `选项 ${i}`;
  const ws = validate({ short: 'state' }, { pick: choice('which?', many) }, 'typed-decisions')
  assert.ok(ws.some((w) => /30 个选项/.test(w.message)))
})

// ═══════════════════════════════════════════════════════════
// defineDecision
// ═══════════════════════════════════════════════════════════

test('defineDecision 打标记；普通对象不会被误认', () => {
  const d = defineDecision({ id: 'a', state: () => ({}), questions: {}, policy: [] })
  assert.equal(isDecision(d), true)
  assert.equal(isDecision({ id: 'a', state: () => ({}), questions: {}, policy: [] }), false)
})
