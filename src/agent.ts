/**
 * JevLoop · agent loop
 *
 * ══════════════════════════════════════════════════════════════
 *  这个 loop 里，只有最后那个 generate() 是一次大模型调用。
 *  其余每一个 ↗ 都是判定模型，10ms 量级，近乎免费。
 * ══════════════════════════════════════════════════════════════
 *
 *   step ─┬─ loop.needsTool   ↗ 需要动手吗？ ──否──▶ 直接生成
 *         │
 *         ├─ loop.pickTool    ↗ 用哪个工具？（选项每步重建）
 *         │
 *         ├─ loop.gradeRisk   ↗ 这个操作多危险？ ──▶ 需要授权就问人
 *         │
 *         ├─ [ 工具执行 ]      ← 唯一有真实副作用的地方
 *         │
 *         ├─ loop.stepOk      ↗ 成功了吗？
 *         │
 *         └─ loop.isDone      ↗ 做完了吗？ ──否──▶ 下一个 step
 *                             │
 *                             ▼
 *                        [ 模型生成 ]   ← 整个 loop 里唯一贵的一步
 *                             │
 *                        loop.canDeliver  ↗ 能交付吗？
 */

import { Decider } from './decide.ts'
import { Meter } from './meter.ts'
import { needsTool, pickTool, gradeRisk, stepOk, isDone, canDeliver, type AgentCtx, type StepRecord } from './decisions.ts'
import { callTool } from './tools.ts'
import type { Generator } from './llm.ts'

export interface AgentOptions {
  task: string
  cwd: string
  decider: Decider
  generator: Generator
  maxSteps?: number
  /** 判定节点要求显式授权时调用。**默认拒绝** —— 宁可不动，也不擅自做不可逆操作 */
  onAskHuman?: (reason: string, tool: string) => Promise<boolean>
  onTrace?: (line: string) => void
}

export interface AgentResult {
  answer: string
  /** 停机原因 */
  halt: string
  steps: number
  ctx: AgentCtx
  meter: Meter
}

export async function runAgent(opts: AgentOptions): Promise<AgentResult> {
  const { decider, generator } = opts
  // Decider 的 meter 是必填的（见 decide.ts 的说明）—— 这里不再自建。
  // 自建会导致：判定记进 decider 的那个，返回给调用方的是另一个空的。
  // Decider 的 meter 是必填的（见 decide.ts 的说明）—— 这里不再自建。
  // 自建会导致：判定记进 decider 的那个，返回给调用方的是另一个空的。
  const meter = decider.meter
  const maxSteps = opts.maxSteps ?? 12
  const trace = opts.onTrace ?? (() => {})

  const ctx: AgentCtx = { task: opts.task, cwd: opts.cwd, files: [], history: [] }
  let step = 0
  let halt = 'max_steps'

  // ── 工具循环 ──────────────────────────────────────────────
  while (step < maxSteps) {
    step++
    decider.setStep(step)

    // ↗ 需要动手吗
    const need = await decider.decide(needsTool, ctx)
    if (need.action === 'answer') {
      halt = 'answered_directly'
      trace(`  直接回答（不需要工具）`)
      break
    }

    // ↗ 用哪个工具（候选每步重建）
    const pick = await decider.decide(pickTool, ctx)
    if (pick.escalate || pick.action !== 'call') {
      halt = 'tool_unclear'
      trace(`  工具选择不确定 → 停下（${pick.reason}）`)
      break
    }
    const tool = pick.answers.tool.choice

    if (tool === 'done') {
      halt = 'agent_done'
      trace(`  agent 主动结束工具循环`)
      break
    }

    // 记录这次要做什么，好让 gradeRisk 看到
    const input = defaultInput(tool, ctx)
    const pending: StepRecord = { step, tool, input, result: '' }
    ctx.history = [...(ctx.history ?? []), pending]
    ctx.lastTool = tool

    // ↗ 这个操作多危险
    const risk = await decider.decide(gradeRisk, ctx)
    if (risk.escalate || risk.action === 'ask_human') {
      const approved = opts.onAskHuman ? await opts.onAskHuman(risk.reason, tool) : false
      trace(`  ⚠ 需要授权：${tool}（${risk.reason}）→ ${approved ? '已批准' : '已拒绝'}`)
      if (!approved) {
        halt = 'denied'
        ctx.history = ctx.history.slice(0, -1)
        break
      }
    } else {
      trace(`  判定放行：${tool}（${risk.action}）`)
      // `auto_audit` 承诺了留痕，那留痕就必须真的发生 ——
      // 以前这条分支和 `auto` 完全一样，只多打一行 trace。
      if (risk.action === 'auto_audit') {
        meter.recordAudit(step, {
          tool,
          target: input.length > 200 ? `${input.slice(0, 200)}…` : input,
          reason: risk.reason,
          risk: risk.answers.risk.score,
        })
        trace(`  审计留痕 #${meter.audit.length}：${tool} risk=${risk.answers.risk.score}`)
      }
    }

    // ── 唯一有真实副作用的地方 ──
    const result = await callTool(tool, input, ctx.cwd)
    pending.result = result
    ctx.lastResult = result

    // 工具产生了文件列表 → 灌进 ctx，下一轮的候选动作会跟着变
    if (tool === 'list_dir') {
      ctx.files = result.split('\n').filter((l) => l && !l.endsWith('/'))
    }

    // ↗ 成功了吗
    const ok = await decider.decide(stepOk, ctx)
    if (ok.action !== 'continue') {
      halt = 'step_failed'
      trace(`  这一步没有成功 → 停下（${ok.reason}）`)
      break
    }

    // ↗ 做完了吗
    const done = await decider.decide(isDone, ctx)
    if (done.action === 'finish') {
      halt = 'task_done'
      break
    }
  }

  // ── 生成（整个 loop 里唯一贵的一步）────────────────────────
  const evidence = () => (ctx.history ?? []).map((h) => `${h.tool}(${h.input}) → ${h.result}`).join('\n')

  let genStep = step + 1
  decider.setStep(genStep)
  const gen = await generator.generate({ task: ctx.task, evidence: evidence() })
  meter.recordModelCall(genStep, {
    kind: `generate (${generator.name})`,
    latencyMs: gen.latencyMs,
    inputTokens: gen.inputTokens,
    outputTokens: gen.outputTokens,
  })
  ctx.draft = gen.text

  // ↗ 能交付吗
  let deliver = await decider.decide(canDeliver, ctx)

  // `revise` 承诺了「修订」，那修订就必须真的发生 ——
  // 以前它只是被拼进 halt 字符串，草稿原样返回。
  // **上限 1 次**：第二次还不合格就如实返回并说明，不无限重试（那会变成一个收费循环）。
  if (deliver.action === 'revise') {
    trace(`  交付闸门要求修订（${deliver.reason}）→ 带着反馈重新生成一次`)
    genStep += 1
    decider.setStep(genStep)
    const retry = await generator.generate({
      task: ctx.task,
      evidence: evidence(),
      instruction: `上一次的回答没有通过交付闸门：${deliver.reason}。请据此修正，不要重复同样的写法。`,
    })
    meter.recordModelCall(genStep, {
      kind: `generate/revise (${generator.name})`,
      latencyMs: retry.latencyMs,
      inputTokens: retry.inputTokens,
      outputTokens: retry.outputTokens,
    })
    ctx.draft = retry.text
    deliver = await decider.decide(canDeliver, ctx)
  }

  if (deliver.action !== 'deliver') {
    halt = `${halt}+${deliver.action}`
  }

  return { answer: ctx.draft, halt, steps: step, ctx, meter }
}

/** 每个工具的默认输入。真实 agent 里这一步也由模型生成 —— 这里为了 demo 保持确定性 */
function defaultInput(tool: string, ctx: AgentCtx): string {
  const firstFile = (ctx.files ?? [])[0] ?? ''
  switch (tool) {
    case 'list_dir':
      return '.'
    case 'read_file':
      return firstFile
    case 'write_file':
      return `${firstFile}\n（内容由调用方提供）`
    default:
      return ''
  }
}
