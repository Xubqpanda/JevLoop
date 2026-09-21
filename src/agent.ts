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
 * ## 待拆
 *
 * 两件事：**loop 本身**（`runAgent`）与**给工具定输入**（`resolveInput`）。
 * 接缝清楚 —— `resolveInput` 只依赖 `frame.ts` 的候选与 `pickInput` 判定，
 * 不认识 loop 的状态机。排在 `AGENTS.md` §12 的待拆队列里。
 * **行数不在这里写** —— 它会漂，而且这行注释本身就在改变它（见 §12）。
 *
 * @module JevLoop/agent
 */

import { Decider } from './decide.ts'
import type { DecisionResult } from './vocab-decision.ts'
import { Meter } from './meter.ts'
import { clip } from './budget.ts'
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
import {
  foldEvidence,
  priceGenerateRequest,
  EVIDENCE_POLICY,
  type ContextReport,
  type RequestEstimate,
} from './context.ts'
import { decisionEvent, type AgentObserver, type BudgetLine, type RunBudget } from './events.ts'
export type { AgentEvent, AgentObserver } from './events.ts'
import type { Generator, ConversationTurn } from './llm.ts'
import { foldConversation, type ConversationReport } from './conversation.ts'

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
   * 之前的轮次。**多轮会话的入口** —— 没有它，每一句都是孤立的任务，
   * 「再读一遍那个文件」里的"那个"无处可指。
   *
   * 只有问答、没有中间过程（见 `llm.ts` 的 `ConversationTurn`）。
   * 调用方负责给出**有界**的份数。
   */
  history?: readonly ConversationTurn[]
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
  /**
   * 工具证据那块预算的账目。**总是有** —— 没超触发线时是一份
   * 「什么都没做」的账（`acted: false`），那也是一条信息：
   * 它告诉你**离触发线还有多远**。
   *
   * ★ 以前这里写的是「可能没有」，而代码从来都给它赋值（`evidence()`
   *   无条件被调用）—— 文档和代码对不上。现在按代码的事实写，并且让
   *   两份账目**统一**：`context` 和 `conversation` 都总是有。
   *   界面要能回答「离触发线还剩多少」，而那要求没动手时也有账。
   */
  context: ContextReport
  /**
   * 上文那块预算的账目（单位是**轮**，不是步）。同样**总是有**。
   *
   * 和 `context` 各报各的：两块是独立预算，合成一个数就说不清是哪个超了。
   */
  conversation: ConversationReport
  /**
   * 这次生成请求的 token 构成 —— 证据 / 上文 / 本句各占多少。
   *
   * ★ **它一直算着，只是以前算完就扔了。** `priceGenerateRequest`
   *   每次都返回四个数，而这里只用了合计（`controlTokens`）去填
   *   `generate` 事件的 `estimatedInputTokens`，另外三个从未离开这个文件。
   *   后果是界面上看得见「这次花了 5440 token（估）」，看不见
   *   「这 5440 是什么构成的」—— 而后者才是能拿来调预算的那一半。
   */
  request: RequestEstimate
}

/**
 * 跑一次 agent。**整个 loop 里只有最后一次 `generate()` 是大模型调用**，
 * 其余每一步都是判定（见模块头部的图）。
 *
 * 停机原因在返回值的 `halt` 里，而且它是**诚实的**：`max_steps` 就是撞上了
 * 迭代上限、`input_unclear` 就是选不出输入、`denied` 就是授权被拒 ——
 * 不要把这些折叠成一句「失败了」，因为排查方向完全不同。
 *
 * ── 会不会抛：分两半，别只记前半句 ──────────────────────────
 *
 * **判定后端挂掉不会抛**：会记 `degraded` 并走到 `escalate`（见 `Decider.decide`）。
 *
 * ⚠️ **但生成后端挂掉会抛。** 两处 `generator.generate()` 没有 try，
 *    异常原样穿出去。这不是漏了 —— 判定那条线每一步都有兜底（不确定就别猜），
 *    而生成这条线**没有可用的兜底**：它只有一次调用，兜底等于回答
 *    「用一个不存在的回答」这个没有答案的问题。
 *
 *    代价落在调用方身上：**必须自己接住**。实测（2026-09-21，写标定台时踩的）：
 *    一次 `UND_ERR_CONNECT_TIMEOUT` 直接穿出 `runAgent`，把整轮 21 次测量
 *    带走了。服务端那条路径本来就接住了（`server.ts` 的 catch 合成一条
 *    `halt: 'error'` 的 `run:end`，并把真实原因写进 answer）。
 *
 *    改动前这里只写了「不会抛：判定后端…」—— 理由只覆盖一半，结论却写成了
 *    全称。照着这句话写调用方的人不会去接，然后就会撞上。
 */
export async function runAgent(opts: AgentOptions): Promise<AgentResult> {
  const { decider, generator } = opts
  // Decider 的 meter 是必填的（见 decide.ts 的说明）—— 这里不再自建。
  // 自建会导致：判定记进 decider 的那个，返回给调用方的是另一个空的。
  const meter = decider.meter
  const maxSteps = opts.maxSteps ?? 12
  const trace = opts.onTrace ?? (() => {})
  const emit: AgentObserver = opts.onEvent ?? (() => {})

  // 上文压成**一句话**进 ctx —— 判定帧是**有界**的（§8.2），把整段对话
  // 塞进去会把真正要看的东西挤掉。只留每一轮「问过什么」，因为判定需要的是
  // **指代关系**（"再读一遍那个文件"里的"那个"），不是上一轮的完整过程。
  //
  // ★ **倒序拼接**。`clip` 保留的是**头部**（见 `budget.ts` 的 `slice(0, …)`），
  //   所以拼接方向决定了有限预算留给哪一端。这里要和 `context.ts` 的
  //   `fitEvidence` 对齐 —— 它也**从最近往回取**，理由同样是「最新的最相关」：
  //   与当前这一步最相关的是**紧邻的上一轮**（用户刚改了什么要求、刚澄清了什么），
  //   不是第 1 轮。
  //
  //   实测（8 轮、拼接 221 字符、预算 `EARLIER_MAX_CHARS = 200`）：
  //     正序 → 保留第 1–7 轮，**第 8 轮（最近的）被截掉**
  //     倒序 → 保留第 8–2 轮，第 1 轮被截掉          ← 这才是想要的
  //
  //   **别看它"顺序不对"就顺手正过来** —— 这个方向是故意的；
  //   正过来就退回成「模型看得见第 1 轮、看不见用户在最后一轮改的口径」。
  //
  // ★ 上文的折叠**在这里算一次**，和下面的证据同一条规矩：`opts.history`
  //   在一次运行里不会变，算第二遍只会重复发同一个事件。
  //   它必须算在 `earlier` **之前** —— 判定帧和生成请求要看**同一份**上文，
  //   否则「模型看得见第 7 轮、判定看不见」这种错位会静默地影响选工具。
  const folded = foldConversation(opts.history ?? [])

  // ★ **缺省就是缺省**：没有上文时传 `undefined`，不传空数组。
  //
  //   两者对现在的两个生成器没差别（都写 `req.history ?? []`），但 `[]`
  //   在 JS 里是**真值** —— 一个写 `if (req.history)` 的生成器会走进
  //   「有上文」的分支去读一个空列表。而且空数组只可能出现在「本来就没有
  //   上文」这一种情况（留尾至少一轮），所以它并不比 `undefined` 多带信息。
  //   摘要同理：没折过就没有摘要，而不是「有一段空摘要」。
  const genHistory = folded.recent.length > 0 ? folded.recent : undefined
  const genDigest = folded.digest.length > 0 ? folded.digest : undefined

  const earlier = folded.recent.slice().reverse().map((t) => t.task).join(' / ')
  const ctx: AgentCtx = {
    task: opts.task,
    cwd: opts.cwd,
    earlier,
    files: [],
    history: [],
    // 没有内容来源时 `write_file` 不进候选（见 AgentOptions.provideWriteContent）
    canWrite: typeof opts.provideWriteContent === 'function',
  }
  let step = 0
  let halt = 'max_steps'

  emit({ type: 'run:start', task: opts.task, cwd: opts.cwd, at: Date.now() })

  // 上文被折过就说出来。它发生在 loop 之前，所以步号是 0。
  // **只有真的动了才发** —— 没超触发线时什么都不做，那没什么可报的。
  if (folded.report.acted) {
    emit({
      type: 'conversation',
      step: 0,
      rawChars: folded.report.rawChars,
      keptChars: folded.report.keptChars,
      rawTurns: folded.report.rawTurns,
      keptTurns: folded.report.keptTurns,
      foldedTurns: folded.report.foldedTurns,
      overRetain: folded.report.overRetain,
    })
  }

  /**
   * 记一笔判定并发事件。
   *
   * loop 里每个 `decider.decide` 的返回值都过这个函数 —— 漏一处，
   * 界面上就少一个决策点，而那种缺失不会报错，只会静默地少一块。
   */
  const record = <A>(d: DecisionResult<A>): DecisionResult<A> => {
    emit(decisionEvent(d as DecisionResult<unknown>))
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
        // ★ `lastTool` 必须跟着一起回退。它和 `history` 是**两个字段说同一件事**
        //   （「最后发生了什么」），而两者都会被读进后续的帧：
        //   `gradeRisk.state.tool`、`stepOk.state.tool`、以及 `frame.ts` 里
        //   `ctx.lastTool === 'write_file'` 那个分支。
        //   只回退一个，下一轮判定就会看到一个**从未发生过的调用** —— 帧在说谎，
        //   而下游每个判定都会"正确地"基于它做判断（同 A1 的失效形状）。
        ctx.lastTool = ctx.history.at(-1)?.tool
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
  /**
   * 交给生成器的证据。**必须有界，而且必须承认自己被截过。**
   *
   * 以前这里没有任何上界：`ctx.history` 有多长证据就有多长 ——
   * `maxSteps` 是 12、单次工具结果最多 4000 字符（`read_file` 的截断），
   * 最坏能到约 200KB。
   *
   * 为什么没人发现：它进的是**生成请求**（LLM 侧），**不是决策帧**，
   * 所以 `budget.ts` 的 `validate()` 管不到它 —— 帧有预算、证据没有。
   * 这是个遗漏，不是有意的设计。
   *
   * ── 规则在 `context.ts`，不在这里 ──────────────────────────
   *
   * 这里原来有一份内联实现（单阈值 6000 / 每条 800 / 只留头）。它和
   * `context.ts` 是同一件事的两份实现（AGENTS.md §3.1），已按所有者指示合并：
   * **位置取这里**（每条工具结果进 history 的那一层，粒度对，而且判定帧
   * 以后能共用同一份账），**规则取 `context.ts`**。
   *
   * 取它那套的三条理由，都是「不报错但一直在错」的情形：
   *
   * · **留头也留尾** —— 工具输出最有用的两端是开头（这是什么）和结尾
   *   （错误、汇总）。只留头会把错误信息砍掉。
   * · `head + 标记 + tail ≤ 阈值`，**配置期校验** —— 挡「越裁越大」。
   * · `retain < trigger` 且 `retain ≥ 单条裁剪后的最大体积` ——
   *   分别挡「每轮都压、永远压不下去」和「目标线永远达不到」。
   *
   * 账目（剪了几条、丢了几条、有没有压到目标）由 `fitEvidence` 返回，
   * 这里存进 `lastEvidence`，随 `AgentResult` 交给调用方显示 ——
   * §8.10：被丢掉的东西要报出来，否则读起来就像"本来就这些"。
   */
  const EVIDENCE_INPUT_CHARS = 120

  /**
   * 备好证据，并把账目**一起返回**。
   *
   * ★ 以前它只返回文本，账目写进一个外层的 `let lastEvidence`。那是
   *   「调用多次、只留最后一次」的写法 —— 而它**只被调一次**（下面那段
   *   注释解释了为什么必须只调一次）。副作用改成返回值之后，类型是确定的
   *   （不再是 `ContextReport | undefined`），也不用再断言「它一定有值」。
   */
  const buildEvidence = (): { text: string; report: ContextReport } => {
    // `label` 是给折叠摘要用的短名字 —— `context.ts` 不认识工具，所以由这里给
    const parts = (ctx.history ?? []).map((x) => ({
      text: `${x.tool}(${clip(x.input, EVIDENCE_INPUT_CHARS)}) → ${x.result}`,
      label: x.input ? `${x.tool}(${clip(x.input, 60)})` : x.tool,
    }))
    const { text, folds, report } = foldEvidence(parts, EVIDENCE_POLICY)
    // `context` 事件**仍然只在真的动手时发** —— 它是轨迹里的一条记录，
    // 「什么都没做」不该占一行。而「没动手时也要看得见预算」由 `run:end`
    // 上那份**每轮都发**的 `budget` 负责，两者分工不同。
    if (report.acted) {
      emit({
        type: 'context',
        step,
        rawChars: report.rawChars,
        keptChars: report.keptChars,
        prunedCount: report.prunedCount,
        foldedCount: report.foldedCount,
        overRetain: report.overRetain,
        folds: folds.map((f) => ({
          toSeq: f.toSeq,
          foldedNodes: f.foldedNodes,
          removedChars: f.removedChars,
        })),
      })
    }
    return { text, report }
  }

  // ★ 证据**只算一次**。
  //
  //   `evidence()` 里有一次折叠（重活），而它原本被调了两遍 ——
  //   一遍给生成、一遍给定价 —— 加上 revise 那一轮一共 4 遍。
  //   后果不只是白算：每算一遍就发一个 `context` 事件，于是界面和轨迹里
  //   出现 4 条一模一样的账目。
  //
  //   循环已经结束，`ctx.history` 在两次生成之间不会变，所以一次就够。
  const { text: evidenceText, report: evidenceReport } = buildEvidence()
  const evidenceEstimate = priceGenerateRequest({
    task: ctx.task,
    evidence: evidenceText,
    history: genHistory,
    historyDigest: genDigest,
  })

  let genStep = step + 1
  decider.setStep(genStep)
  const gen = await generator.generate({
    task: ctx.task,
    evidence: evidenceText,
    history: folded.recent,
    historyDigest: folded.digest,
  })
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
    inputTokens: gen.inputTokens,
    outputTokens: gen.outputTokens,
    estimatedInputTokens: evidenceEstimate.controlTokens,
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
      // 同一份证据和同一份上文 —— 循环早就结束了，两者都没变过
      evidence: evidenceText,
      history: genHistory,
      historyDigest: genDigest,
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
      inputTokens: retry.inputTokens,
      outputTokens: retry.outputTokens,
      estimatedInputTokens: evidenceEstimate.controlTokens,
    })
    ctx.draft = retry.text
    deliver = record(await decider.decide(canDeliver, ctx))
  }

  if (deliver.action !== 'deliver') {
    halt = `${halt}+${deliver.action}`
  }

  emit({
    type: 'run:end',
    halt,
    steps: step,
    answer: ctx.draft,
    stats: meter.stats,
    budget: runBudget(evidenceReport, folded.report, evidenceEstimate),
  })
  return {
    answer: ctx.draft,
    halt,
    steps: step,
    ctx,
    meter,
    // 两份账目**都给**，不管有没有动手 —— 「没动手」本身是一条信息
    // （离触发线还有多远），而「只报出事的那种」让正常运行里看不见预算。
    context: evidenceReport,
    conversation: folded.report,
    request: evidenceEstimate,
  }
}

/**
 * 把两块预算的报告 + 这次请求的 token 构成，压成事件和界面要的那一个形状。
 *
 * ── 为什么在这里映射，而不是让界面认两种报告 ────────────────────
 *
 * 两块报告的形状不同：一块数**步**（`ContextReport`，有 `prunedCount` /
 * `foldedCount`），一块数**轮**（`ConversationReport`，有 `foldedTurns`）。
 * 但界面要回答的是**同一组问题**：用了多少、线在哪、动手没有、动的是什么。
 *
 * 让界面去认两种形状，等于以后加第三块预算（比如决策帧）时还要改界面。
 * 所以在这一层一次性映射：**新增一块预算，界面不用动**。
 *
 * @param evidence 工具证据那块（单位：步）
 * @param conversation 上文那块（单位：轮）
 * @param request 这次请求的 token 构成
 */
function runBudget(
  evidence: ContextReport | undefined,
  conversation: ConversationReport,
  request: RequestEstimate,
): RunBudget {
  return {
    // `evidence` 理论上有值（`evidence()` 无条件被调用），但这个函数是
    // 纯映射、不该假设调用顺序，所以缺了就给一份全 0 的账而不是抛。
    evidence: budgetLine(
      evidence ?? {
        rawChars: 0,
        keptChars: 0,
        prunedCount: 0,
        foldedCount: 0,
        triggerChars: 0,
        retainChars: 0,
        acted: false,
        overRetain: false,
      },
      evidence ? noteForEvidence(evidence) : '',
    ),
    conversation: budgetLine(conversation, noteForConversation(conversation)),
    request: {
      evidence: request.evidenceTokens,
      history: request.historyTokens,
      task: request.taskTokens,
      total: request.controlTokens,
    },
  }
}

/** 报告 → 事件形状。两块报告都有的那七个字段逐个搬 */
function budgetLine(
  r: {
    rawChars: number
    keptChars: number
    triggerChars: number
    retainChars: number
    acted: boolean
    overRetain: boolean
  },
  note: string,
): BudgetLine {
  return {
    rawChars: r.rawChars,
    keptChars: r.keptChars,
    triggerChars: r.triggerChars,
    retainChars: r.retainChars,
    acted: r.acted,
    overRetain: r.overRetain,
    note,
  }
}

/**
 * 证据那一块动了什么。
 *
 * **没动手时返回 `''`**，而不是「未压缩」之类的字眼 —— 界面靠 `acted`
 * 判断该说什么，再给一句同义的话只会多一处会漂移的地方。
 */
function noteForEvidence(r: ContextReport): string {
  const bits: string[] = []
  if (r.prunedCount > 0) bits.push(`剪了中间 ${r.prunedCount} 条`)
  if (r.foldedCount > 0) bits.push(`折成摘要 ${r.foldedCount} 步`)
  return bits.join(' · ')
}

/** 上文那一块动了什么。同样没动手就是 `''` */
function noteForConversation(r: ConversationReport): string {
  return r.foldedTurns > 0 ? `折成摘要 ${r.foldedTurns} 轮` : ''
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
