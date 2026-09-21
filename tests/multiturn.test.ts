/**
 * 多轮会话的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 多轮的全部意义是**指代关系**：「再读一遍**那个**文件」里的"那个"，
 * 只有在上文到达模型时才有处可指。所以这里测的不是「history 被传了」，
 * 而是**它到没到该到的地方** —— 生成请求的 messages 里，和判定帧里。
 *
 * @module JevLoop/multiturn.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { HttpGenerator, ScriptedGenerator, type ConversationTurn } from '../src/llm.ts'
import { needsTool, pickTool } from '../src/decisions.ts'
import { runAgent } from '../src/agent.ts'
import { Decider } from '../src/decide.ts'
import { Meter } from '../src/meter.ts'
import type { AgentCtx } from '../src/frame.ts'
import type { QuestionSet } from '../src/vocab.ts'

const TURNS: ConversationTurn[] = [
  { task: '列出目录里的文件', answer: '有三个：invoice.ts、notes.md、retry.ts' },
  { task: '读一下 invoice.ts', answer: '它定义了 outstanding() 和 Invoice' },
]

// ═══════════════════════════════════════════════════════════
// 生成侧：上文必须真的进 messages
// ═══════════════════════════════════════════════════════════

test('HttpGenerator 把每一轮铺成一对 user/assistant', async () => {
  let sent: { messages: { role: string; content: string }[] } | undefined
  const realFetch = globalThis.fetch
  globalThis.fetch = (async (_url: string, init: { body: string }) => {
    sent = JSON.parse(init.body)
    return {
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }], usage: {} }),
    }
  }) as never

  try {
    const g = new HttpGenerator({ baseUrl: 'http://x', model: 'm' })
    await g.generate({ task: '再读一遍那个文件', evidence: '', history: TURNS })
  } finally {
    globalThis.fetch = realFetch
  }

  const msgs = sent!.messages
  // system + 2 轮 × 2 条 + 当前这一条
  assert.equal(msgs.length, 1 + TURNS.length * 2 + 1)
  assert.equal(msgs[0]!.role, 'system')
  for (const [i, t] of TURNS.entries()) {
    assert.equal(msgs[1 + i * 2]!.role, 'user')
    assert.match(msgs[1 + i * 2]!.content, new RegExp(t.task))
    assert.equal(msgs[2 + i * 2]!.role, 'assistant')
    assert.equal(msgs[2 + i * 2]!.content, t.answer)
  }
  // 当前这一轮永远在最后，且带着这一轮的工具证据
  assert.match(msgs.at(-1)!.content, /再读一遍那个文件/)
  assert.match(msgs.at(-1)!.content, /What was done/)
})

test('没有 history 时 messages 和以前一模一样（不多发空轮）', async () => {
  let sent: { messages: { role: string; content: string }[] } | undefined
  const realFetch = globalThis.fetch
  globalThis.fetch = (async (_u: string, init: { body: string }) => {
    sent = JSON.parse(init.body)
    return { ok: true, json: async () => ({ choices: [{ message: { content: 'ok' } }], usage: {} }) }
  }) as never
  try {
    await new HttpGenerator({ baseUrl: 'http://x', model: 'm' }).generate({ task: 'T', evidence: 'E' })
  } finally {
    globalThis.fetch = realFetch
  }
  assert.equal(sent!.messages.length, 2, '只有 system + 当前这一条')
})

test('ScriptedGenerator 也提一句上文 —— 离线路径要能看出多轮', async () => {
  const out = await new ScriptedGenerator({ latencyMs: 0 }).generate({
    task: '当前这句',
    evidence: '',
    history: TURNS,
  })
  assert.match(out.text, /earlier turns: 2/)
  assert.match(out.text, /列出目录里的文件/, '要能看出前几轮问过什么')
})

// ═══════════════════════════════════════════════════════════
// 判定侧：上文必须进决策帧
// ═══════════════════════════════════════════════════════════

const ctxWith = (earlier?: string): AgentCtx => ({
  task: '再读一遍那个文件',
  cwd: '/tmp',
  files: ['a.ts', 'b.ts'],
  readFiles: [],
  history: [],
  earlier,
})

test('★ 上文进得了 needsTool 的帧 —— 不进的话「那个」无处可指', () => {
  const frame = needsTool.state(ctxWith('读过 invoice.ts')) as Record<string, unknown>
  assert.equal(frame.earlier, '读过 invoice.ts')
})

test('★ 上文进得了 pickTool 的帧', () => {
  const frame = pickTool.state(ctxWith('读过 invoice.ts')) as Record<string, unknown>
  assert.equal(frame.earlier, '读过 invoice.ts')
})

test('没有上文时帧里是空串，不是缺字段', () => {
  // 缺字段和空字段在模型那边是两回事：前者是"没有这个信号"，
  // 后者是"有这个信号，内容是空的"。发出去的东西要稳定。
  for (const spec of [needsTool, pickTool]) {
    const frame = spec.state(ctxWith(undefined)) as Record<string, unknown>
    assert.equal(frame.earlier, '')
  }
})

test('★ 上文进帧时是**有界**的 —— 它是背景，不能挤掉主体', () => {
  const huge = '很长的一段上文'.repeat(100)
  const frame = needsTool.state(ctxWith(huge)) as Record<string, unknown>
  const earlier = frame.earlier as string
  assert.ok(earlier.length <= 200, `上文必须是 200 字符以内，实际 ${earlier.length}`)
  // 同一帧里这一轮真正要看的东西不能被挤掉
  assert.equal(frame.task, '再读一遍那个文件')
})

// ═══════════════════════════════════════════════════════════
// 端到端：一遍跑下来，生成器真的收到了上文
// ═══════════════════════════════════════════════════════════

test('runAgent 把 history 交给了生成器，并压进了 ctx.earlier', async () => {
  let seenHistory: readonly ConversationTurn[] | undefined
  let seenFrame: Record<string, unknown> | undefined

  // 判定器：直接说"不需要工具"，于是 loop 跳过工具循环直奔生成
  const decider = new Decider({
    meter: new Meter(),
    provider: {
      name: 'spy',
      decide: async (req: { state: unknown; questions: QuestionSet }) => {
        seenFrame ??= req.state as Record<string, unknown>
        return {
          answers: { needs_tool: { type: 'noul', noul: 0.1 } },
          provider: 'spy',
          latencyMs: 0,
        }
      },
    } as never,
  })

  await runAgent({
    task: '再读一遍那个文件',
    cwd: '/tmp',
    decider,
    history: TURNS,
    generator: {
      name: 'capture',
      generate: async (req) => {
        seenHistory = req.history
        return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
      },
    },
    maxSteps: 2,
  })

  assert.deepEqual(seenHistory, TURNS, '生成器必须收到完整的 history')
  // ★ **顺序是倒的，这是刻意的**（见 `agent.ts` 那段注释）：`clip` 保留头部，
  //   所以只有倒序拼接才能把有限的 `EARLIER_MAX_CHARS` 留给**最近**那几轮。
  //   这条断言以前写的是正序 —— 它编码的正是 M1 要修的那个行为。
  assert.equal(seenFrame?.earlier, '读一下 invoice.ts / 列出目录里的文件', '判定帧也必须看到上文')
})

test('runAgent 不带 history 时，生成器收到的是 undefined 而不是空数组', async () => {
  let seen: unknown = 'unset'
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
      generate: async (req) => {
        seen = req.history
        return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
      },
    },
    maxSteps: 1,
  })
  assert.equal(seen, undefined)
})

test('★ M1：上文截断保留的是**最近**几轮，不是最早的', async () => {
  // `clip` 保留的是**头部**（`budget.ts` 的 `slice(0, …)`），所以拼接方向
  // 决定有限的 `EARLIER_MAX_CHARS` 留给哪一端。**正序拼接（修之前）会把最近的
  // 那几轮截掉**，留下的全是对话开头 —— 而多轮 agent 里与当前这一步最相关的
  // 恰恰是紧邻的上一轮（用户刚改了什么要求）。`context.ts` 的 `fitEvidence`
  // 特意从最近往回取，这里要和它对齐。
  let seenFrame: Record<string, unknown> | undefined
  const decider = new Decider({
    meter: new Meter(),
    provider: {
      name: 'spy',
      decide: async (req: { state: Record<string, unknown> }) => {
        if (seenFrame === undefined) seenFrame = req.state
        // needs_tool 给低分 → 直接去生成，不进工具循环
        return { answers: { needs_tool: { type: 'noul', noul: 0.05 } }, provider: 'spy', latencyMs: 0 }
      },
    } as never,
  })
  const turns = Array.from({ length: 8 }, (_, i) => ({
    task: `第${i + 1}轮：把 invoice-${i + 1}.ts 的函数名改掉`,
    answer: 'ok',
  }))
  await runAgent({
    task: '第九轮',
    cwd: '/tmp',
    decider,
    history: turns,
    generator: {
      name: 'noop',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'noop' }),
    },
    maxSteps: 1,
  })

  const earlier = String(seenFrame?.earlier ?? '')
  assert.ok(earlier.length <= 200, `必须被 clip 住，实际 ${earlier.length} 字符`)
  assert.ok(earlier.includes('第8轮'), `最近的必须留下，实际：${earlier}`)
  assert.ok(!earlier.includes('第1轮'), `最早的必须让位给最近的，实际：${earlier}`)
})

// ═══════════════════════════════════════════════════════════
// 会话折叠（`conversation.ts`）—— 上文有预算了
//
// 上面测的是「上文到没到」。这里测的是**上文太长了会怎样**：
// 旧的折成摘要，留尾那几轮逐字，而且两件事都要在账面上看得见。
// ═══════════════════════════════════════════════════════════

/** 造一段超线的上文：12 轮 × 约 1500 字符 ≈ 18000，触发线是 12800 */
const HUGE: ConversationTurn[] = Array.from({ length: 12 }, (_, i) => ({
  task: `第${i}轮的问题`,
  answer: 'x'.repeat(1500),
}))

/** 一个只会说「不需要工具」的判定器 —— 让 loop 直接走到生成那一步 */
function quietDecider(): Decider {
  return new Decider({
    meter: new Meter(),
    provider: {
      name: 'spy',
      decide: async () => ({ answers: { needs_tool: { type: 'noul', noul: 0.1 } }, provider: 'spy', latencyMs: 0 }),
    } as never,
  })
}

test('HttpGenerator 把折叠摘要放进 system，而不是伪装成一问一答', async () => {
  let sent: { messages: { role: string; content: string }[] } | undefined
  const realFetch = globalThis.fetch
  globalThis.fetch = (async (_url: string, init: { body: string }) => {
    sent = JSON.parse(init.body)
    return { ok: true, json: async () => ({ choices: [{ message: { content: 'ok' } }], usage: {} }) }
  }) as never

  try {
    const g = new HttpGenerator({ baseUrl: 'http://x', model: 'm' })
    await g.generate({ task: 'T', evidence: '', history: TURNS, historyDigest: '[更早的 3 轮已折叠成要点]' })
  } finally {
    globalThis.fetch = realFetch
  }

  const msgs = sent!.messages
  // ★ 摘要**不能**多出两条 user/assistant —— 那会让模型以为有人真的这么说过
  assert.equal(msgs.length, 1 + TURNS.length * 2 + 1, '摘要不该增加消息条数')
  assert.equal(msgs[0]!.role, 'system')
  assert.match(msgs[0]!.content, /更早的 3 轮已折叠成要点/)
  // 逐字那几轮还是老老实实的一问一答
  assert.equal(msgs[1]!.role, 'user')
  assert.equal(msgs[2]!.role, 'assistant')
})

test('没有摘要时 system 就是纯指令 —— 不留一段空壳', async () => {
  let sent: { messages: { role: string; content: string }[] } | undefined
  const realFetch = globalThis.fetch
  globalThis.fetch = (async (_url: string, init: { body: string }) => {
    sent = JSON.parse(init.body)
    return { ok: true, json: async () => ({ choices: [{ message: { content: 'ok' } }], usage: {} }) }
  }) as never

  try {
    const g = new HttpGenerator({ baseUrl: 'http://x', model: 'm' })
    await g.generate({ task: 'T', evidence: '' })
  } finally {
    globalThis.fetch = realFetch
  }
  assert.equal(sent!.messages.length, 2, 'system + 当前这一条')
  assert.doesNotMatch(sent!.messages[0]!.content, /folded/)
})

test('runAgent 会折叠超线的上文，生成器收到的是折过的那一份', async () => {
  let seenHistory: unknown = 'unset'
  let seenDigest: unknown = 'unset'
  await runAgent({
    task: '那第一个函数接受什么参数？',
    cwd: '/tmp',
    decider: quietDecider(),
    history: HUGE,
    generator: {
      name: 'capture',
      generate: async (req) => {
        seenHistory = req.history
        seenDigest = req.historyDigest
        return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
      },
    },
    maxSteps: 1,
  })

  const recent = seenHistory as ConversationTurn[]
  // ★ 留尾那几轮是**逐字**的 —— 指代的落点
  assert.deepEqual(recent, HUGE.slice(HUGE.length - recent.length))
  assert.equal(recent.length < HUGE.length, true, '旧的轮次必须被折掉')
  assert.equal(typeof seenDigest, 'string')
  assert.match(seenDigest as string, /已折叠成要点/)
  // 摘要里要留有问句，否则「那个函数」无处可指
  assert.match(seenDigest as string, /第\d轮的问题/)
})

test('折叠会在账目里报出来 —— AgentResult 和事件都要有', async () => {
  const events: { type: string }[] = []
  const result = await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider: quietDecider(),
    history: HUGE,
    generator: {
      name: 'capture',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }),
    },
    onEvent: (e) => events.push(e as { type: string }),
    maxSteps: 1,
  })

  const ev = events.find((e) => e.type === 'conversation') as
    | { type: 'conversation'; rawTurns: number; keptTurns: number; foldedTurns: number; rawChars: number; keptChars: number }
    | undefined
  assert.ok(ev, '必须发 conversation 事件 —— 折了什么要让人看得见（§8.10）')
  assert.equal(ev.foldedTurns > 0, true)
  assert.equal(ev.keptTurns + ev.foldedTurns, ev.rawTurns, '一轮都不能凭空消失')
  assert.equal(ev.keptChars < ev.rawChars, true, '折了就必须变小')

  assert.ok(result.conversation, 'AgentResult 里也要有这份账')
  assert.equal(result.conversation!.foldedTurns, ev.foldedTurns)
})

test('上文没超线时一个字节都不动，也不发事件', async () => {
  const events: { type: string }[] = []
  let seenHistory: unknown = 'unset'
  const result = await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider: quietDecider(),
    history: TURNS,
    generator: {
      name: 'capture',
      generate: async (req) => {
        seenHistory = req.history
        return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
      },
    },
    onEvent: (e) => events.push(e as { type: string }),
    maxSteps: 1,
  })

  assert.deepEqual(seenHistory, TURNS, '没超线就该原样传下去')
  // 轨迹里的 `conversation` 事件仍然只在动手时发 —— 它是记录，
  //「什么都没做」不该占一行
  assert.equal(events.some((e) => e.type === 'conversation'), false, '没动手就别发轨迹事件')
  // ★ 但 `AgentResult` 里的账**总是有**，`acted: false` 表达「没动手」。
  //   以前靠**缺席**表达，于是界面分不清「没超线」和「拿不到账」——
  //   而「离触发线还有多远」正是要在这时候才看得见。
  assert.ok(result.conversation, '没动手也要有账')
  assert.equal(result.conversation.acted, false)
  assert.equal(result.conversation.rawChars, result.conversation.keptChars, '没动手就该一字节没变')
  assert.equal(result.conversation.triggerChars > 0, true, '账里必须带上线在哪，否则「离触发线多远」算不出来')
})

test('run:end 上带着预算账 —— 没动手时也要发，这是「离触发线多远」的唯一来源', async () => {
  const events: { type: string }[] = []
  await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider: quietDecider(),
    history: TURNS,
    generator: {
      name: 'capture',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }),
    },
    onEvent: (e) => events.push(e as { type: string }),
    maxSteps: 1,
  })

  const end = events.find((e) => e.type === 'run:end') as never as {
    budget?: {
      evidence: { rawChars: number; triggerChars: number; retainChars: number; acted: boolean; note: string }
      conversation: { rawChars: number; triggerChars: number; acted: boolean; note: string }
      request: { evidence: number; history: number; task: number; total: number }
    }
  }
  assert.ok(end?.budget, 'run:end 必须带 budget')

  // 两块预算各自带线 —— 界面靠它算「还剩多少」
  assert.equal(end.budget.evidence.triggerChars > 0, true)
  assert.equal(end.budget.conversation.triggerChars > 0, true)
  // 没动手时 note 是空的（界面靠 acted 判断说什么，不是靠一串同义文字）
  assert.equal(end.budget.conversation.acted, false)
  assert.equal(end.budget.conversation.note, '')
})

test('预算账里的 token 构成必须自洽，而且合计就是 generate 事件上那个估算值', async () => {
  const events: { type: string; estimatedInputTokens?: number; budget?: unknown }[] = []
  await runAgent({
    task: '说明这个项目',
    cwd: '/tmp',
    decider: quietDecider(),
    history: TURNS,
    generator: {
      name: 'capture',
      generate: async () => ({ text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }),
    },
    onEvent: (e) => events.push(e as never),
    maxSteps: 1,
  })

  const end = events.find((e) => e.type === 'run:end') as never as {
    budget: { request: { evidence: number; history: number; task: number; total: number } }
  }
  const gen = events.find((e) => e.type === 'generate')
  const r = end.budget.request

  assert.equal(r.evidence + r.history + r.task, r.total, '三个分量必须加起来等于合计')
  // ★ 这两处以前是**各算各的**：generate 事件拿的是合计，而三个分量被丢掉。
  //   现在它们是同一个数的两种看法，所以必须对得上。
  assert.equal(gen?.estimatedInputTokens, r.total, 'generate 上的估算值就是这份构成的合计')
  assert.equal(r.task > 0, true, '当前这一句总得占点')
})

test('判定帧和生成请求看到的是**同一份**上文', async () => {
  let seenFrame: { earlier?: string } | undefined
  let seenHistory: ConversationTurn[] = []
  await runAgent({
    task: 'T',
    cwd: '/tmp',
    decider: quietDecider(),
    history: HUGE,
    onEvent: (e) => {
      const ev = e as { type: string; id?: string; state?: unknown }
      if (ev.type === 'decision' && ev.id === needsTool.id) seenFrame = ev.state as { earlier?: string }
    },
    generator: {
      name: 'capture',
      generate: async (req) => {
        seenHistory = (req.history ?? []) as ConversationTurn[]
        return { text: 'ok', latencyMs: 0, inputTokens: 0, outputTokens: 0, model: 'capture' }
      },
    },
    maxSteps: 1,
  })

  // 折过之后两边都只看得到留尾那几轮；判定帧只留最近 200 字符，
  // 所以它必须是**留尾里最新的那一句**，不能是被折掉的最老那一句
  assert.equal(seenFrame?.earlier?.includes('第0轮的问题') ?? false, false, '被折掉的最老轮次不该出现在判定帧里')
  assert.equal(seenHistory.length < HUGE.length, true)
})
