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
  assert.match(out.text, /上文 2 轮/)
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
  assert.equal(seenFrame?.earlier, '列出目录里的文件 / 读一下 invoice.ts', '判定帧也必须看到上文')
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
