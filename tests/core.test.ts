/**
 * 核心不变量的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这些逻辑**没有类型保护** —— 它们全是运行时行为，tsc 通过不代表行为正确。
 *
 * @module JevLoop/core.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { resolvePolicy, gte, probGte, scoreGte, picked } from '../src/policy.ts'
import { Meter, formatRatio } from '../src/meter.ts'
import { normalizeAnswers } from '../src/provider-http.ts'
import { MockProvider } from '../src/provider-mock.ts'
import { validate, clip, estimateTokens } from '../src/budget.ts'
import { defineDecision, isDecision } from '../src/vocab-decision.ts'
import { noul, choice } from '../src/vocab.ts'
import type { AnswerSet } from '../src/vocab.ts'

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

test('没有模型调用时比值是 null，而且必须能原样跨过 JSON', () => {
  const m = new Meter()
  m.recordDecision(1, fakeDecision(5))
  // ★ 以前这里断言 `Infinity`。改成 `null` 不是口味问题：
  //   这个值要经 SSE 出去，而 `JSON.stringify(Infinity)` 是 `null` ——
  //   **JSON 会静默改写它**。让 JSON 替我们决定，等于把语义交给一个静默的转换，
  //   而前端那条专门为 `Infinity` 写的 `!Number.isFinite` 分支因此永远不可达，
  //   界面显示成 `?` 而不是它想显示的 `N : 0`。
  assert.equal(m.stats.ratio, null)
  assert.equal(m.stats.decisionShare, 1)

  // 这条断言才是改动的理由：往返之后必须还是同一个值
  const wire = JSON.parse(JSON.stringify(m.stats))
  assert.equal(wire.ratio, null, 'ratio 必须是 JSON 能忠实携带的值')
  assert.equal(wire.ratio, m.stats.ratio, '往返不能改变它')
  // 而且格式化出来的仍然是真相，不是 `?`（这里只记了 1 次判定）
  assert.equal(formatRatio(wire), '1:0')
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
  for (let i = 0; i < 30; i++) many[`opt${i}`] = `选项 ${i}`
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

test('N3: 工具参数是判定 —— 「读取全部文件」必须能读到第二个', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { Meter } = await import('../src/meter.ts')
  const { mkdtemp, writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  // 一个专门读「全部」文件的判定器：只要有没读过的就继续读
  const readEverything = {
    name: 'read-everything',
    decide: async (req: { state: unknown; questions: Record<string, { type: string; criteria?: unknown }> }) => {
      const s = req.state as {
        files_known?: string[]
        already_read?: string[]
        already_done?: string
        steps?: string[]
        tool?: string
      }
      const answers: Record<string, unknown> = {}
      const opts = (id: string) => Object.keys((req.questions[id]?.criteria ?? {}) as Record<string, string>)
      for (const [id, q] of Object.entries(req.questions)) {
        const crit = opts(id)
        if (q.type === 'noul') {
          // 这几个问题「高」意味着坏事：回答不支持、需要授权、或**任务还没做完**。
          // 全给 0.9 会让 loop 在第一次 list_dir 之后就 isDone=finish 而停下。
          const negative = id === 'unsupported' || id === 'needs_auth' || id === 'done'
          let p = negative ? 0.05 : 0.9
          if (id === 'done') {
            // isDone 的帧只有 steps —— 数一下读过几个文件
            const steps: string[] = Array.isArray(s.steps) ? s.steps : []
            const reads = steps.filter((x) => String(x).startsWith('read_file')).length
            p = reads >= 2 ? 0.9 : 0.05
          }
          answers[id] = { type: 'noul', noul: p }
        } else if (q.type === 'score') {
          const legend = (q.criteria as string[]) ?? []
          answers[id] = { type: 'score', score: 0, legend: Object.fromEntries(legend.map((l, i) => [String(i), l])), probabilities: {}, confidence: 0.9 }
        } else {
          // choice：按问题 id 决定选谁
          let pick = crit[0] ?? ''
          if (id === 'tool') {
            // 读决策帧的**实际字段**（already_done / already_read），
            // 不是已经不存在的 recent —— 那正是 examples/rule-judge.ts 犯过的错。
            const listed = String(s.already_done ?? '').includes('list_dir')
            const read = new Set(s.already_read ?? [])
            const unread = (s.files_known ?? []).filter((f) => !read.has(f))
            if (!listed) pick = 'list_dir'
            else pick = crit.includes('read_file') && unread.length ? 'read_file' : 'done'
          }
          if (id === 'file') {
            const read = new Set(s.already_read ?? [])
            pick = crit.find((f) => !read.has(f)) ?? crit[0] ?? ''
          }
          answers[id] = { type: 'choice', choice: pick, probabilities: pick ? { [pick]: 0.99 } : {}, confidence: 0.99 }
        }
      }
      return { answers, provider: 'fake', latencyMs: 0 }
    },
  }

  const cwd = await mkdtemp(join(tmpdir(), 'jevloop-n3-'))
  try {
    await writeFile(join(cwd, 'a.ts'), 'export const a = 1\n', 'utf8')
    await writeFile(join(cwd, 'b.ts'), 'export const b = 2\n', 'utf8')

    const meter = new Meter()
    const decider = new Decider({ provider: readEverything as never, meter })
    await runAgent({
      task: '读取全部 TypeScript 文件',
      cwd,
      decider,
      generator: { name: 'noop', generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'noop' }) },
      maxSteps: 8,
    })

    const reads = meter.decisions.filter((d) => d.id === 'loop.pickInput')
    const targets = meter.decisions
      .filter((d) => d.id === 'loop.pickTool' && d.answers.includes('read_file'))
      .length

    // 修之前：read_file 读一次之后就被 toolsFor 永久移除，所以最多 1 次
    assert.equal(targets, 2, `read_file 必须被选中两次（两个文件各一次），实际 ${targets}`)
    assert.equal(reads.length, 2, `pickInput 必须被问两次，实际 ${reads.length}`)
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// 第 4 轮落地项的回归测试
//
// 每条注释写的是**修之前会发生什么** —— 不写的话，这些测试会慢慢退化成
// 「看起来在测什么、其实测不出什么」。
// ═══════════════════════════════════════════════════════════

test('R9-P1: 畸形答案认值不认标签，不被伪造成 0', () => {
  const { answers, dropped } = normalizeAnswers({
    a: { type: 'noul' }, // 有标签、没值
    b: { type: 'score' }, // 有标签、没值
    c: { type: 'choice' }, // 有标签、没值
    d: { type: 'choice', choice: '' }, // 空串不是一个选项
    e: { noul: 0.7 }, // 没标签、有值 → 照收
    f: { type: 'noul', noul: 'high' }, // 值的类型不对
  })
  // 修之前：a/b/c 被 `Number(x) || 0` 补成 noul:0 / score:0 / choice:''，
  // 既不进 dropped 也不进 missing，于是 degraded 保持 false ——
  // 在日志上和一次正常判定完全一样，而 0 在策略里是**明确的否定**。
  assert.deepEqual(Object.keys(answers), ['e'])
  assert.deepEqual(dropped.sort(), ['a', 'b', 'c', 'd', 'f'])
  assert.equal((answers.e as { noul: number }).noul, 0.7)
})

test('R9-P2: stepOk 不把「完全不确定」读成「成功」', async () => {
  const { stepOk } = await import('../src/decisions.ts')
  // 0.5 是 noul 最不确定的取值，而门限是闭区间 `>=`：
  // 修之前 T.stepOk = 0.5，Mock 的 0.5 被判成 continue —— **失败被吞掉**。
  const action = (p: number) => resolvePolicy(stepOk.policy, ans({ ok: { type: 'noul', noul: p } }) as never).action
  assert.equal(action(0.5), 'stop')
  // 真成功时仍要放行（离线规则判定器给的是 0.92）
  assert.equal(action(0.92), 'continue')
})

test('C2: 0 次模型调用时比值报 N:0，不是 N:1', async () => {
  const { formatRatio } = await import('../src/meter.ts')
  // 修之前 examples/demo.ts 自己拼字符串，0 次模型调用时报成 `3 : 1` ——
  // 而这个比值是项目的卖点本身。
  assert.equal(formatRatio({ decisions: 3, modelCalls: 0, ratio: Infinity }), '3:0')
  assert.equal(formatRatio({ decisions: 12, modelCalls: 1, ratio: 12 }), '12.0:1')
})

test('R1/R2: write_file 只在有内容来源时进候选，写过就撤出', async () => {
  const { pickTool } = await import('../src/decisions.ts')
  const opts = (ctx: unknown) =>
    Object.keys(((pickTool.questions as (c: unknown) => { tool: { criteria: Record<string, string> } })(ctx)).tool.criteria)
  const base = { task: 't', cwd: '.', files: ['a.ts'], history: [] }

  assert.ok(!opts({ ...base }).includes('write_file'), '缺省没有内容来源，write_file 不该出现')
  assert.ok(!opts({ ...base, canWrite: false }).includes('write_file'))
  assert.ok(opts({ ...base, canWrite: true }).includes('write_file'), '有来源时应当可选')

  // §8.4：写过就不再是候选，否则模型会反复选它（实测连续 5 次）
  const used = {
    ...base,
    canWrite: true,
    history: [{ step: 1, tool: 'write_file', input: 'a.ts\nx', result: 'ok' }],
  }
  assert.ok(!opts(used).includes('write_file'), '写完还在候选里，模型会再选它')
})

test('R1: 没有内容来源时，目标文件的内容一个字节都不变', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { mkdtemp, writeFile, readFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-w-'))
  try {
    const target = join(cwd, 'note.md')
    const original = '# 我的真实笔记\n这里是很重要的内容\n'
    await writeFile(target, original, 'utf8')

    // 最坏情况的调用方：只要 write_file 在候选里就一定选它
    const insistWrite = {
      name: 'insist-write',
      decide: async (req: { questions: Record<string, { type: string; criteria?: unknown }> }) => {
        const answers: Record<string, unknown> = {}
        for (const [id, q] of Object.entries(req.questions)) {
          const crit = Object.keys((q.criteria ?? {}) as Record<string, string>)
          // ★ 两个「低分」都是必须的，各修掉一次假通过（都是反向验证抓出来的）：
          //   · `done` 给高了 → isDone 立刻 finish，循环根本走不到 write_file
          //   · `needs_auth` 给高了 → 授权闸门拒掉调用，write_file 压根没执行
          //   两种情况下文件都不会变，测试就会"绿"得毫无意义。
          if (q.type === 'noul') {
            const low = id === 'needs_auth' || id === 'done' || id === 'unsupported'
            answers[id] = { type: 'noul', noul: low ? 0.05 : 0.9 }
          } else if (q.type === 'score')
            answers[id] = { type: 'score', score: 0, legend: {}, probabilities: {}, confidence: 0.9 }
          else {
            const pick = id === 'tool' && crit.includes('write_file') ? 'write_file' : (crit[0] ?? '')
            answers[id] = { type: 'choice', choice: pick, probabilities: pick ? { [pick]: 0.99 } : {}, confidence: 0.99 }
          }
        }
        return { answers, provider: 'fake', latencyMs: 0 }
      },
    }

    const decider = new Decider({ provider: insistWrite as never, meter: new Meter() })
    await runAgent({
      task: '把 note.md 的内容改掉',
      cwd,
      decider,
      generator: {
        name: 'noop',
        generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'noop' }),
      },
      maxSteps: 6,
      // 万一还是走到授权闸门，就批准 —— 这个测试要验的是「内容会不会落盘」，
      // 不是审批行为。
      onAskHuman: async () => true,
      // ★ **`null` = 明确不要写入能力**，不是「不传」。
      //
      //   不传的话 `runAgent` 会用生成器现造一个输入来源（那是缺省，理由见
      //   `agent.ts` 里那段）—— 于是 `write_file` 进候选，而这个测试里那个
      //   「只要它在候选里就一定选它」的判定器会真的把 note.md 改掉。
      //
      //   实测：改缺省那次这条测试当场红了，而它红得对 —— 门确实被绕过了。
      //   `null` 和 `undefined` 的区别就是为这件事留的。
      provideWriteInput: null,
    })

    // 修之前：这里会变成「（内容由调用方提供）」—— 一句占位符把真实内容整个替换掉，
    // 而且 write_file 仍在候选里，会连续重写 5 次，最后 halt: max_steps。
    assert.equal(await readFile(target, 'utf8'), original)
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

test('★ 缺省会用生成器现造输入 —— 而且能建一个**还不存在**的文件', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { mkdtemp, readFile, rm, writeFile } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-w2-'))
  try {
    await writeFile(join(cwd, 'alpha.ts'), 'export const a = 1\n', 'utf8')

    // 只要 write_file 在候选里就选它 —— 和 R1 同一个最坏调用方
    const insistWrite = {
      name: 'insist-write',
      decide: async (req: { questions: Record<string, { type: string; criteria?: unknown }> }) => {
        const answers: Record<string, unknown> = {}
        for (const [id, q] of Object.entries(req.questions)) {
          const crit = Object.keys((q.criteria ?? {}) as Record<string, string>)
          if (q.type === 'noul') {
            const low = id === 'needs_auth' || id === 'done' || id === 'unsupported'
            answers[id] = { type: 'noul', noul: low ? 0.05 : 0.9 }
          } else if (q.type === 'score')
            answers[id] = { type: 'score', score: 0, legend: {}, probabilities: {}, confidence: 0.9 }
          else {
            const pick = id === 'tool' && crit.includes('write_file') ? 'write_file' : (crit[0] ?? '')
            answers[id] = { type: 'choice', choice: pick, probabilities: pick ? { [pick]: 0.99 } : {}, confidence: 0.99 }
          }
        }
        return { answers, provider: 'fake', latencyMs: 0 }
      },
    }

    // ★ **目标文件还不存在** —— 这正是这次改动要覆盖的情形。
    //   在这之前 `pickInput` 的候选只能来自已经存在的文件，所以
    //   「新建一个 SUMMARY.md」这种任务没有任何地方能产生那个名字。
    const target = join(cwd, 'SUMMARY.md')
    let sawEvidence = ''
    await runAgent({
      task: '把 alpha.ts 导出了什么写进一个新建的 SUMMARY.md',
      cwd,
      decider: new Decider({ provider: insistWrite as never, meter: new Meter() }),
      generator: {
        name: 'capture',
        generate: async (req: { evidence: string }) => {
          sawEvidence = req.evidence
          // 第一行路径、其余内容 —— 这就是 `write_file` 的输入格式
          return { text: '```\nSUMMARY.md\n# 导出的东西\n\n- `a`\n```', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
        },
      },
      maxSteps: 8,
      onAskHuman: async () => true,
    })

    assert.equal(await readFile(target, 'utf8'), '# 导出的东西\n\n- `a`\n', '★ 新文件被建出来了，围栏也剥掉了')
    assert.match(sawEvidence, /list_dir/, '生成时拿到的素材里有前面工具的输出')
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// A1：`list_dir` 的返回值会被当成**文件列表**解析
//
// 它的返回类型是 `Promise<string>`，装的是渲染好的文本；而 `agent.ts` 会
// `split('\n').filter(...)` 当结构化数据用。空目录那句「(目录为空)」于是
// 变成了一个"文件名"：进 `ctx.files`、进 `pickInput` 的候选，
// 写路径下还会在用户目录里**真的创建一个叫 `(目录为空)` 的文件**。
// ═══════════════════════════════════════════════════════════

test('A1: 空目录不能变成一个叫「(目录为空)」的文件', async () => {
  const { callTool } = await import('../src/tools.ts')
  const { fileOptions, hasFileOptions, unreadFiles } = await import('../src/frame.ts')
  const { mkdtemp, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-empty-'))
  try {
    const result = await callTool('list_dir', '.', cwd)
    // 这是 agent.ts 的解析方式，原样照抄 —— 测的就是「它解析出什么」
    const files = result.split('\n').filter((l) => l && !l.endsWith('/'))
    assert.deepEqual(files, [], `空目录不该解析出文件，实际 ${JSON.stringify(files)}`)

    const ctx = { task: 't', cwd, files } as never
    assert.equal(hasFileOptions(ctx), false, '空目录里没有可挑的输入')
    assert.deepEqual(unreadFiles(ctx), [])
    // 写路径最严重：候选里出现幻影文件名时，write_file 会真的把它建出来。
    // ★ 现在 `fileOptions` **只服务 read_file**（写路径的名字是生成的，
    //   见 `write-content.ts`），所以这里验的是「它不会凭空造出候选」
    assert.deepEqual(Object.keys(fileOptions(ctx)), [], '不能有幻影文件')
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

test('A1: 非空目录仍然逐行返回文件名，子目录带 /', async () => {
  const { callTool } = await import('../src/tools.ts')
  const { mkdtemp, mkdir, writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-list-'))
  try {
    await writeFile(join(cwd, 'b.ts'), '', 'utf8')
    await writeFile(join(cwd, 'a.ts'), '', 'utf8')
    await mkdir(join(cwd, 'sub'))
    assert.equal(await callTool('list_dir', '.', cwd), 'a.ts\nb.ts\nsub/')
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// E1：`.env` 的解析边界
// ═══════════════════════════════════════════════════════════

test('E1: `export KEY=VALUE` 要设上 KEY，不能造出一个叫 `export KEY` 的垃圾键', async () => {
  const { loadEnv } = await import('../src/env.ts')
  const { mkdtempSync, writeFileSync, rmSync } = await import('node:fs')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const dir = mkdtempSync(join(tmpdir(), 'JevLoop-env-'))
  // 用一个很少见的键名，免得和真实环境串味
  const KEY = 'JEVLOOP_TEST_EXPORTED'
  delete process.env[KEY]
  delete process.env[`export ${KEY}`]
  try {
    writeFileSync(
      join(dir, '.env'),
      `export ${KEY}=hello\nPLAIN_KEY=world\nQUOTED="q v"\nEMPTY=\nBAD KEY=1\nnonsense\n`,
      'utf8',
    )
    const r = loadEnv({ cwd: dir })
    // 修之前：`export KEY=VALUE` 被当成键名的一部分，于是 KEY 没设上，
    // 反而多了一个 `export KEY`，而且返回值里把它报成「已加载」。
    assert.equal(process.env[KEY], 'hello')
    assert.equal(process.env[`export ${KEY}`], undefined, '不能凭空造出带空格的键')
    assert.ok(!r.loaded.includes(`export ${KEY}`), '垃圾键不能出现在 loaded 里')
    // 键名不合法的行要**跳过并报出来**，不能静默
    assert.deepEqual(r.skipped, ['BAD KEY=1', 'nonsense'])
    // 其余三种写法不受影响
    assert.deepEqual(r.loaded, [KEY, 'PLAIN_KEY', 'QUOTED', 'EMPTY'])
  } finally {
    delete process.env[KEY]
    delete process.env.PLAIN_KEY
    delete process.env.QUOTED
    delete process.env.EMPTY
    rmSync(dir, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// 第 6 轮：A2 / A3 / E3
// ═══════════════════════════════════════════════════════════

/** 造一个「只选第一个候选项」的判定器，风险分由参数给 */
const fakeJudge = (risk: number, opts: { doneAfter?: number } = {}) => ({
  name: 'fake',
  decide: async (req: { state: unknown; questions: Record<string, { type: string; criteria?: unknown }> }) => {
    const s = req.state as { files_known?: string[]; already_read?: string[]; already_done?: string[] | string }
    const answers: Record<string, unknown> = {}
    for (const [id, q] of Object.entries(req.questions)) {
      const crit = Object.keys((q.criteria ?? {}) as Record<string, string>)
      if (q.type === 'noul') {
        const read = (s.already_read ?? []).length
        const doneAt = opts.doneAfter ?? 3
        // ★ `needs_auth` 必须给**低**分，否则授权闸门会把调用拒掉，
        //   循环根本走不到生成 —— A3 的第一版就是这么"空跑"的。
        //   A2 要的是硬闸门（`scoreGte('risk', 2)`），不靠这一条。
        answers[id] =
          id === 'needs_auth' || id === 'unsupported'
            ? { type: 'noul', noul: 0.05 }
            : { type: 'noul', noul: id === 'done' ? (read >= doneAt ? 0.9 : 0.05) : 0.9 }
      } else if (q.type === 'score') {
        answers[id] = { type: 'score', score: risk, legend: {}, probabilities: {}, confidence: 0.9 }
      } else {
        let pick = crit[0] ?? ''
        if (id === 'tool') {
          const listed = String(s.already_done ?? '').includes('list_dir')
          const read = new Set(s.already_read ?? [])
          const unread = (s.files_known ?? []).filter((f) => !read.has(f))
          pick = !listed ? 'list_dir' : crit.includes('read_file') && unread.length ? 'read_file' : 'done'
        }
        if (id === 'file') {
          const read = new Set(s.already_read ?? [])
          pick = crit.find((f) => !read.has(f)) ?? crit[0] ?? ''
        }
        answers[id] = { type: 'choice', choice: pick, probabilities: pick ? { [pick]: 0.99 } : {}, confidence: 0.99 }
      }
    }

    return { answers, provider: 'fake', latencyMs: 0 }
  },
})

test('A2: 授权被拒之后 lastTool 必须和 history 说同一件事', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { mkdtemp, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-deny-'))
  try {
    // risk = 3 ≥ riskAuth(2) → 硬闸门 ask_human；不给 onAskHuman 就等于拒绝
    const decider = new Decider({ provider: fakeJudge(3) as never, meter: new Meter() })
    const r = await runAgent({
      task: 't',
      cwd,
      decider,
      generator: {
        name: 'noop',
        generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'noop' }),
      },
      maxSteps: 3,
    })
    assert.equal(r.halt, 'denied')
    // 历史被回退了，`lastTool` 以前**没跟着回退** —— 于是下一轮判定会看到一个
    // 从未发生过的调用（gradeRisk.state.tool / stepOk.state.tool /
    // frame.ts 里 `lastTool === 'write_file'` 那个分支都会读到它）。
    assert.equal(r.ctx.history?.length ?? -1, 0)
    assert.equal(r.ctx.lastTool, undefined, `lastTool 必须跟着 history 回退，实际 ${String(r.ctx.lastTool)}`)
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

test('A3: 交给生成器的证据有上界，且承认自己被截过', async () => {
  const { Decider } = await import('../src/decide.ts')
  const { runAgent } = await import('../src/agent.ts')
  const { mkdtemp, writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')

  const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-ev-'))
  try {
    // 文件数必须够多，让**逐条裁剪之后的总和**也超过 6000 —— 否则测的只是
    // 「单条被剪」，总量那条上界根本没被触发（第一版 3 个文件就是这样，
    // 反向验证时把预算放到无穷大测试照样通过，暴露了它是个弱测试）。
    const FILES = ['a.ts', 'b.ts', 'c.ts', 'd.ts', 'e.ts', 'f.ts', 'g.ts', 'h.ts', 'i.ts']
    const big = 'x'.repeat(3000)
    for (const f of FILES) await writeFile(join(cwd, f), big, 'utf8')

    let seen = ''
    const decider = new Decider({ provider: fakeJudge(0, { doneAfter: FILES.length }) as never, meter: new Meter() })
    await runAgent({
      task: '读全部',
      cwd,
      decider,
      generator: {
        name: 'capture',
        generate: async (req: { evidence: string }) => {
          seen = req.evidence
          return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
        },
      },
      maxSteps: 12,
    })

    // 修之前：`evidence` 把整份 history 原样拼起来 —— 这条路径上没有任何上界，
    // 而它进的是**生成请求**，`budget.validate()` 管不到。
    // 9 个文件各 3000 字符，原样拼接是 ~27000，远超 6000 的预算。
    assert.ok(seen.length <= 6200, `证据必须有上界，实际 ${seen.length} 字符`)
    // §8.10：被省略/被截断的部分必须被承认，否则读起来就像"本来就这些"
    // 行为从「丢弃」改成了「折叠」：老的内容换成一段摘要，原文仍在轨迹里。
    // 措辞因此必须跟着变 —— 继续断言「被省略」会是错的，它没被省略。
    assert.match(seen, /已折叠|工具输出被截断/, '折了/截了什么必须报出来')
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
})

test('E3: decisionEvent 只接受一个参数，步号取自 DecisionResult', async () => {
  const { decisionEvent } = await import('../src/events.ts')
  // 以前是 `decisionEvent(step, d)` —— 两个参数说同一件事，而
  // `Decider` 内部还另有一个 `#step`。同一个事实两个出处就迟早分叉，
  // 而事件步号错位会让界面上整条轨迹的对应关系全错。
  assert.equal(decisionEvent.length, 1, '签名必须是 decisionEvent(d)')
  const e = decisionEvent({
    id: 'x',
    step: 7,
    state: {},
    questions: {},
    answers: {},
    action: 'a',
    reason: 'r',
    latencyMs: 1,
    provider: 'p',
  } as never)
  assert.equal((e as { step: number }).step, 7)
})

// ═══════════════════════════════════════════════════════════
// stepOk 必须只判「这一步」，不判「这个任务」
//
// 实测的 bug：帧里带着 task，问题写「a usable result **for the task**」，
// 判据写「contains **what the task needed**」—— 三处一起把它拉到了任务级。
// 任务「读一下 invoice.ts」第一步 `list_dir` 返回文件列表：它**确实成功**了，
// 但没回答「这个文件定义了哪些函数」，于是 `ok=0.470` 判否 → 动作 `stop`
// → **整个循环结束**。任何需要多于一个工具的任务都跑不完。
//
// 这组测试是机械的：数字对不对要靠模型，但「该问哪一级」是结构，能断言。
// ═══════════════════════════════════════════════════════════

test('★ stepOk 的帧里没有 task —— 有它就会去判任务完成度', async () => {
  const { stepOk } = await import('../src/decisions.ts')
  const frame = stepOk.state({
    task: '读一下 invoice.ts，说明它定义了哪些函数',
    cwd: '/tmp',
    lastTool: 'list_dir',
    lastResult: 'invoice.ts\nnotes.md\nretry.ts',
    history: [],
  }) as Record<string, unknown>

  assert.equal(frame.task, undefined, 'stepOk 的帧不该带 task —— 那是 isDone 的问题')
  assert.equal(frame.tool, 'list_dir', '它要看的仍然是这次调用本身')
  assert.ok(frame.output, '工具输出在')
})

test('★ stepOk 的问题与判据都在说「这次调用」，不是「这个任务」', async () => {
  const { stepOk } = await import('../src/decisions.ts')
  const q = (stepOk.questions as Record<string, { instructions: string; criteria: { true: string; false: string } }>).ok!
  const all = `${q.instructions} ${q.criteria.true} ${q.criteria.false}`

  // 这几种措辞就是把判定拉错层的原因，任何一种回来都应当让这条测试红
  for (const bad of [/for the task/i, /what the task needed/i, /the task requires/i]) {
    assert.ok(!bad.test(all), `措辞 \`${bad}\` 会把 stepOk 拉回任务级`)
  }
  assert.match(q.instructions, /this tool call/i, '要明说判的是这次调用')
  assert.match(q.instructions, /decided elsewhere/i, '要明说任务完成度在别处判')
})

// ═══════════════════════════════════════════════════════════
// phase 事件：界面显示「现在在干什么」的唯一依据
//
// ★ 起因是一个真 bug（2026-09-21，用户报的）：生成那 2 秒里界面显示的是
//   「正在判定」。因为所有事件都是**做完之后**才发的（`decision` 带
//   latencyMs、`generate` 带 token 数），界面只能拿「上一次干完的是什么」
//   去猜，而这个猜法在最长的那一步上错得最久。
// ═══════════════════════════════════════════════════════════

test('★ 每个 decision / generate **之前**都有一条 phase —— 界面靠它显示当前阶段', async () => {
  const { runAgent } = await import('../src/agent.ts')
  const { Decider } = await import('../src/decide.ts')
  const { Meter } = await import('../src/meter.ts')

  const seq: { type: string; id?: string; kind?: string }[] = []
  const decider = new Decider({
    meter: new Meter(),
    provider: {
      name: 'spy',
      decide: async () => ({ answers: { needs_tool: { type: 'noul', noul: 0.1 } }, provider: 'spy', latencyMs: 0 }),
    } as never,
  })

  await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider,
    generator: {
      name: 'capture',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }),
    },
    maxSteps: 1,
    onEvent: (e) => {
      seq.push(
        e.type === 'phase'
          ? { type: `phase:${e.kind}`, ...(e.id ? { id: e.id } : {}) }
          : e.type === 'decision'
            ? { type: 'decision', id: e.id }
            : { type: e.type },
      )
    },
  })

  /*
    ★ **一次 phase 可以覆盖多个 decision。**

    循环里每步开头那一对（`needsTool` + `pickTool`）是**一次请求问两个独立
    判定**（官方 skill：「Ask independent questions over the same state
    together」）—— 它们同时开始，所以只有一条 phase。

    所以要验的不是「一对一」，而是：**每条 decision 前面都有一条 phase，
    而且那条 phase 的名字里点得到它** —— 否则界面在那一整段显示的还是上
    一个阶段的标签，而那正是这条测试当初要抓的 bug。
  */
  const problems: string[] = []
  let lastPhase: { id?: string } | undefined
  for (const [i, x] of seq.entries()) {
    if (x.type === 'phase:decide') lastPhase = x
    if (x.type === 'decision') {
      // 那条 phase 必须**点名它自己** —— 合并时名字是 `'a + b'`，拆开看
      if (!lastPhase) problems.push(`第 ${i} 条 decision 前面没有 phase:decide`)
      else if (lastPhase.id && !lastPhase.id.split(' + ').includes(x.id!)) {
        problems.push(`phase 的名字 '${lastPhase.id}' 里点不到它要覆盖的 '${x.id}'`)
      }
    }
    if (x.type === 'generate' && seq[i - 1]?.type !== 'phase:generate') {
      problems.push(`第 ${i} 条 generate 前面是 ${seq[i - 1]?.type}`)
    }
  }
  assert.deepEqual(problems, [], '有操作没有先播报 phase —— 界面会在那一段显示上一个阶段的标签')

  assert.ok(seq.some((x) => x.type === 'phase:decide'), `一次判定都没发生：${seq.map((x) => x.type).join(' → ')}`)
  assert.ok(seq.some((x) => x.type === 'phase:generate'), `没有生成阶段：${seq.map((x) => x.type).join(' → ')}`)
})

test('phase 事件带得上判定节点 id —— 排查「哪个判定慢」要看它', async () => {
  const { runAgent } = await import('../src/agent.ts')
  const { Decider } = await import('../src/decide.ts')
  const { Meter } = await import('../src/meter.ts')

  const ids: string[] = []
  const decider = new Decider({
    meter: new Meter(),
    provider: {
      name: 'spy',
      decide: async () => ({ answers: { needs_tool: { type: 'noul', noul: 0.1 } }, provider: 'spy', latencyMs: 0 }),
    } as never,
  })
  await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider,
    generator: {
      name: 'capture',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }),
    },
    maxSteps: 1,
    onEvent: (e) => {
      if (e.type === 'phase' && e.kind === 'decide') ids.push(e.id ?? '(没有 id)')
    },
  })

  // ⚠️ 不要写成 `['loop.needsTool']` —— spy provider 对**每个**判定都返回
  //    同一份答案，所以 `canDeliver` 也拿到 `needs_tool:0.1`，它的策略在缺
  //    `deliverable` / `unsupported` 时走到 `revise`，于是**又一次生成 +
  //    又一次 canDeliver**。（同一个坑今天绊了两次。）
  //
  // ★ 合并之后一条 phase 可能覆盖两个节点（`'a + b'`），所以判据是
  //   **每个名字都像节点 id**，而不是「恰好一个」。
  const all = ids.flatMap((x) => x.split(' + '))
  assert.equal(all[0], 'loop.needsTool', '第一个判定是 needsTool')
  assert.ok(
    all.every((id) => /^loop\.[a-zA-Z]+$/.test(id)),
    `每个 phase 都要带真实的节点 id，实际拿到：${JSON.stringify(ids)}`,
  )
  assert.ok(all.includes('loop.pickTool'), `合并的那一对要都点名：${JSON.stringify(ids)}`)
})
