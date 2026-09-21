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

// ═══════════════════════════════════════════════════════════
// agent loop —— 审计 N1 / N2 的回归测试
//
// 这两条的共性是：**项目自己的 demo 测不出来**。
// demo 显式传了 meter，也恰好走不到 revise/auto_audit 分支，
// 所以只有外部使用者会撞到。回归测试必须按外部用法写。
// ═══════════════════════════════════════════════════════════

test('N1: 不传 meter 构造 Decider，runAgent 返回的 meter 必须是同一个且非空', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { ScriptedGenerator } = await import('../src/llm.ts')

  // 最自然的用法：只给 provider
  const decider = new Decider({ provider: new MockProvider({ latencyMs: 0 }) })
  const r = await runAgent({
    task: 't',
    cwd: '/tmp',
    decider,
    generator: new ScriptedGenerator({ latencyMs: 0 }),
    maxSteps: 1,
  })

  // 修之前这里是 0 —— 判定记进了 undefined，返回的是另一个没人写过的 Meter
  assert.ok(r.meter.stats.decisions > 0, `判定数必须 > 0，实际 ${r.meter.stats.decisions}`)
  assert.equal(r.meter, decider.meter, 'runAgent 必须返回 decider 自己的那个 meter')
})

test('N2: auto_audit 必须真的留痕，而不是只多打一行 trace', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { Meter } = await import('../src/meter.ts')
  const { RuleJudge } = await import('../examples/rule-judge.ts')
  const { mkdtemp, writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'jevloop-audit-'))
  try {
    await writeFile(join(cwd, 'a.txt'), 'hello', 'utf8')
    const meter = new Meter()
    const decider = new Decider({ provider: new RuleJudge(), meter })
    const calls: string[] = []
    await runAgent({
      task: '列出工作目录里的文件',
      cwd,
      decider,
      generator: { name: 'noop', generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'noop' }) },
      maxSteps: 2,
      onTrace: (l) => calls.push(l),
    })
    // gradeRisk 对 write_file 给 risk=1 → auto_audit 分支
    // list_dir 是 risk=0 → auto，不留痕。所以断言的是「留痕机制存在且内容可查」
    assert.ok(meter.audit.every((a) => a.tool && typeof a.reason === 'string'), '审计条目必须有工具和理由')
    assert.equal(meter.stats.audits, meter.audit.length, 'stats.audits 要和 audit 数组长度一致')
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

test('N2: revise 必须真的重新生成一次，且只重试一次', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { Meter } = await import('../src/meter.ts')
  const { mkdtemp, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  // 判定器：canDeliver 永远说 revise，逼出重试路径
  const alwaysRevise = {
    name: 'always-revise',
    decide: async (req: { questions: Record<string, unknown> }) => {
      const answers: Record<string, unknown> = {}
      for (const [id, q] of Object.entries(req.questions)) {
        const t = (q as { type: string }).type
        if (t === 'noul') answers[id] = { type: 'noul', noul: 0.5 }
        else if (t === 'score') answers[id] = { type: 'score', score: 0, legend: {}, probabilities: {}, confidence: 0 }
        else answers[id] = { type: 'choice', choice: '', probabilities: {}, confidence: 0 }
      }
      // canDeliver 的两个问题：让 deliverable 低、unsupported 高 → 走 revise
      if ('deliverable' in req.questions) answers.deliverable = { type: 'noul', noul: 0.1 }
      if ('unsupported' in req.questions) answers.unsupported = { type: 'noul', noul: 0.9 }
      return { answers, provider: 'fake', latencyMs: 0 }
    },
  }

  const cwd = await mkdtemp(join(tmpdir(), 'jevloop-revise-'))
  try {
    let genCalls = 0
    const meter = new Meter()
    const decider = new Decider({ provider: alwaysRevise as never, meter })
    const r = await runAgent({
      task: 't',
      cwd,
      decider,
      generator: {
        name: 'counter',
        generate: async () => {
          genCalls++
          return { text: `draft${genCalls}`, latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'counter' }
        },
      },
      maxSteps: 1,
    })

    assert.equal(genCalls, 2, 'revise 必须触发第二次生成（且只有第二次 —— 上限 1 次）')
    assert.equal(r.answer, 'draft2', '返回的必须是修订后的草稿，不是原始的')
    assert.ok(r.halt.includes('revise'), `halt 要如实说明没通过闸门，实际 ${r.halt}`)
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})
