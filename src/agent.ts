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
 *
 * @module JevLoop/agent
 */

import { Decider } from './decide.ts'
import type { DecisionResult } from './vocab-decision.ts'
import { Meter } from './meter.ts'
import {
  needsTool,
  pickTool,
  pickInput,
  gradeRisk,
  stepOk,
  isDone,
  canDeliver,
} from './decisions.ts'
import { hasFileOptions, type AgentCtx, type StepRecord } from './frame.ts'
import { callTool, isToolName, type ToolName } from './tools.ts'
import { assertNever } from './util.ts'
import { decisionEvent, type AgentObserver } from './events.ts'
export type { AgentEvent, AgentObserver } from './events.ts'
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
  /**
   * `write_file` 的内容来源。**不提供时 `write_file` 根本不会进候选**，
   * 于是判定模型没有机会去选一个 loop 兑现不了的动作。
   *
   * 「写什么内容」是**生成**，按三分法不属于判定模型（AGENTS.md §8.1）——
   * 判定只负责挑「写哪个文件」。以前这里没有这个钩子，`resolveInput`
   * 只能返回一句占位符 `（内容由调用方提供）`，而它是**真的会被写到盘上的**：
   * 实测一次运行把目标文件的全部内容替换成了这句话，然后因为
   * `write_file` 仍在候选里而连续重写了 5 次，最后 `halt: max_steps`。
   *
   * 返回 `undefined` 表示这次写不了 → loop 停机，而不是写个占位符交差。
   */
  provideWriteContent?: (
    file: string,
    ctx: AgentCtx,
  ) => string | undefined | Promise<string | undefined>
  /**
   * 观察者。每次判定、每次工具调用、每次生成都会发一个事件。
   *
   * 和 `onTrace` 的分工：`onTrace` 是给人读的一行字，`onEvent` 是**结构化的**，
   * 给界面、测试、日志消费。两者可以同时用。
   */
  onEvent?: AgentObserver
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
  const meter = decider.meter
  const maxSteps = opts.maxSteps ?? 12
  const trace = opts.onTrace ?? (() => {})
  const emit: AgentObserver = opts.onEvent ?? (() => {})

  const ctx: AgentCtx = {
    task: opts.task,
    cwd: opts.cwd,
    files: [],
    history: [],
    // 没有内容来源时 `write_file` 不进候选（见 AgentOptions.provideWriteContent）
    canWrite: typeof opts.provideWriteContent === 'function',
  }
  let step = 0
  let halt = 'max_steps'

  emit({ type: 'run:start', task: opts.task, cwd: opts.cwd, at: Date.now() })

  /**
   * 记一笔判定并发事件。
   *
   * loop 里每个 `decider.decide` 的返回值都过这个函数 —— 漏一处，
   * 界面上就少一个决策点，而那种缺失不会报错，只会静默地少一块。
   */
  const record = <A>(d: DecisionResult<A>): DecisionResult<A> => {
    emit(decisionEvent(step, d as DecisionResult<unknown>))
    return d
  }

  // ── 工具循环 ──────────────────────────────────────────────
  while (step < maxSteps) {
    step++
    decider.setStep(step)

    // ↗ 需要动手吗
    const need = record(await decider.decide(needsTool, ctx))
    if (need.action === 'answer') {
      halt = 'answered_directly'
      trace(`  直接回答（不需要工具）`)
      break
    }

    // ↗ 用哪个工具（候选每步重建）
    const pick = record(await decider.decide(pickTool, ctx))
    if (pick.escalate || pick.action !== 'call') {
      halt = 'tool_unclear'
      trace(`  工具选择不确定 → 停下（${pick.reason}）`)
      break
    }
    const picked = pick.answers.tool.choice

    // 模型返回的工具名是**不可信输入**，调用前必须过这一道（AGENTS.md §6 允许的真实边界）。
    // 不过会怎样：`callTool` 返回「错误：没有这个工具」，而这个字符串会被当成
    // 普通工具输出喂给 `stepOk` —— 判定模型分不清「工具跑出来的结果」和「工具不存在」。
    if (!isToolName(picked)) {
      halt = 'unknown_tool'
      trace(`  模型返回了不存在的工具 '${picked}' → 停下（不当成结果喂给下一步判定）`)
      break
    }
    const tool: ToolName = picked

    if (tool === 'done') {
      halt = 'agent_done'
      trace(`  agent 主动结束工具循环`)
      break
    }

    // 先记下要调用什么，`loop.pickInput` 的候选依赖 lastTool
    ctx.lastTool = tool

    // 工具参数是一次**判定**，不是写死的代码（审计 N3）
    const input = await resolveInput(tool, ctx, decider, record, opts.provideWriteContent)
    if (input === undefined) {
      halt = 'input_unclear'
      trace(`  选不出 ${tool} 的输入 → 停下`)
      break
    }

    const pending: StepRecord = { step, tool, input, result: '' }
    ctx.history = [...(ctx.history ?? []), pending]

    // ↗ 这个操作多危险
    const risk = record(await decider.decide(gradeRisk, ctx))
    if (risk.escalate || risk.action === 'ask_human') {
      const approved = opts.onAskHuman ? await opts.onAskHuman(risk.reason, tool) : false
      emit({ type: 'authorize', step, tool, reason: risk.reason, approved })
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
        emit({ type: 'audit', step, record: meter.audit[meter.audit.length - 1]! })
        trace(`  审计留痕 #${meter.audit.length}：${tool} risk=${risk.answers.risk.score}`)
      }
    }

    // ── 唯一有真实副作用的地方 ──
    emit({ type: 'tool:call', step, tool, input })
    const toolT0 = Date.now()
    const result = await callTool(tool, input, ctx.cwd)
    emit({ type: 'tool:result', step, tool, output: result, ms: Date.now() - toolT0 })
    pending.result = result
    ctx.lastResult = result

    // 工具产生了文件列表 → 灌进 ctx，下一轮的候选动作会跟着变
    if (tool === 'list_dir') {
      ctx.files = result.split('\n').filter((l) => l && !l.endsWith('/'))
    }
    // 读过的文件要记下来 —— 否则 `pickInput` 会一直提议读同一个文件
    if (tool === 'read_file') {
      ctx.readFiles = [...(ctx.readFiles ?? []), input.trim()]
    }

    // ↗ 成功了吗
    const ok = record(await decider.decide(stepOk, ctx))
    if (ok.action !== 'continue') {
      halt = 'step_failed'
      trace(`  这一步没有成功 → 停下（${ok.reason}）`)
      break
    }

    // ↗ 做完了吗
    const done = record(await decider.decide(isDone, ctx))
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
  emit({
    type: 'generate',
    step: genStep,
    kind: `generate (${generator.name})`,
    latencyMs: gen.latencyMs,
    tokens: gen.inputTokens + gen.outputTokens,
  })
  ctx.draft = gen.text

  // ↗ 能交付吗
  let deliver = record(await decider.decide(canDeliver, ctx))

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
    emit({
      type: 'generate',
      step: genStep,
      kind: `generate/revise (${generator.name})`,
      latencyMs: retry.latencyMs,
      tokens: retry.inputTokens + retry.outputTokens,
    })
    ctx.draft = retry.text
    deliver = record(await decider.decide(canDeliver, ctx))
  }

  if (deliver.action !== 'deliver') {
    halt = `${halt}+${deliver.action}`
  }

  emit({ type: 'run:end', halt, steps: step, answer: ctx.draft, stats: meter.stats })
  return { answer: ctx.draft, halt, steps: step, ctx, meter }
}

/**
 * 给一个工具决定它的输入。
 *
 * 审计 N3 之前这里是写死的 `defaultInput`：永远返回 `files[0]`，配合
 * `toolsFor` 移除用过的动作，导致一个 agent 生命周期内 `read_file`
 * 只能触发一次、且只能读第一个文件 —— 「读取全部 TypeScript 文件」
 * 这类任务不可能完成。
 *
 * 按三分法，「读哪个文件」是**挑选**，交给判定；「写什么内容」是**生成**，
 * 仍由调用方提供。`list_dir` 没有有意义的输入选择，不占用一次判定。
 *
 * @param tool 已经过 `isToolName` 校验的工具名
 * @param ctx 当前上下文，`pickInput` 的候选从这里构造
 * @param decider 判定器
 * @param writeContent `write_file` 的内容来源，见 `AgentOptions.provideWriteContent`
 * @returns 工具的输入字符串；无法确定时返回 `undefined`（调用方应停机，不要猜）
 */
async function resolveInput(
  tool: ToolName,
  ctx: AgentCtx,
  decider: Decider,
  record: <A>(d: DecisionResult<A>) => DecisionResult<A>,
  writeContent?: AgentOptions['provideWriteContent'],
): Promise<string | undefined> {
  switch (tool) {
    case 'list_dir':
      // 输入恒为工作目录 —— 这里没有可挑的东西，不值得问一次判定
      return '.'
    case 'done':
      return ''
    case 'read_file':
    case 'write_file': {
      // 没有候选就**不要问** —— criteria 为空的 choice 是无效问题
      if (!hasFileOptions(ctx)) return undefined
      const d = record(await decider.decide(pickInput, ctx))
      if (d.escalate || d.action !== 'use') return undefined
      const file = d.answers.file.choice
      if (tool === 'read_file') return file

      // ★ 内容必须由调用方提供。拿不到就**停机**，绝不退化成占位符：
      //   这里以前返回的是 `${file}\n（内容由调用方提供）`，而 `write_file`
      //   会把第二行起的内容原样写进目标文件 —— 一次调用就把文件替换成了
      //   那句说明，且循环会继续选中 `write_file` 反复重写（实测 5 次）。
      //   「写不了」是一个诚实的结果；写一句假内容不是。
      const content = await writeContent?.(file, ctx)
      if (content === undefined) return undefined
      return `${file}\n${content}`
    }
    default:
      return assertNever(tool)
  }
}
