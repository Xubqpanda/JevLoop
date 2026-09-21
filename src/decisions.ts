/**
 * JevLoop · agent 的全部判定
 *
 * ══════════════════════════════════════════════════════════════
 *  这个文件就是 JevLoop 的全部主张。
 * ══════════════════════════════════════════════════════════════
 *
 * 一个常规 agent 的 loop 里，下面这六件事都会写成一次大模型调用：
 *
 *     要不要动手？  用哪个工具？  这个操作危险吗？
 *     成功了吗？    做完了吗？    这个回答能发出去吗？
 *
 * 它们全都不是「生成」，而是「选择 / 打分 / 是否」。
 * 也就是说：**你一直在用生成的价格，买判定的答案。**
 *
 * 这六个判定一共 6 次前向传播，加起来通常不到 100ms，
 * 换来的是整个 loop 里只剩一次真正的大模型调用。
 *
 * @module JevLoop/decisions
 */

import { defineDecision } from './vocab-decision.ts'
import { noul, choice, score } from './vocab.ts'
import { topGte, probGte, scoreGte } from './policy.ts'
import { clip } from './budget.ts'
import { TOOLS, isToolName } from './tools.ts'

// ═══════════════════════════════════════════════════════════
// 判定需要的上下文
//
// 只声明用得到的字段，且全部可选 —— 判定节点应该容忍一个
// 缺字段的 ctx，而不是抛异常。State 投影负责兜底。
// ═══════════════════════════════════════════════════════════

export interface StepRecord {
  step: number
  tool: string
  input: string
  result: string
}

export interface AgentCtx {
  task: string
  cwd: string
  /** 已知的文件列表，由 ls 工具填充 */
  files?: string[]
  /** 已经读过的文件。`loop.pickInput` 用它把读过的从候选里去掉 */
  readFiles?: string[]
  /** 已经做过的动作 */
  history?: StepRecord[]
  /**
   * 调用方有没有提供 `write_file` 的内容来源。
   *
   * **为 false/undefined 时 `write_file` 不进候选。** 「写什么内容」是生成，
   * 按三分法不属于判定模型（见 AGENTS.md §8.1），判定只能挑「写哪个文件」。
   * 没有内容来源却把 `write_file` 放进候选，模型只能选它、而 loop 又拿不出内容 ——
   * 以前那里填的是一个占位符字符串，它会**真的写进目标文件**（见 agent.ts）。
   */
  canWrite?: boolean
  lastTool?: string
  lastResult?: string
  draft?: string
}

// ═══════════════════════════════════════════════════════════
// 阈值集中在这里
//
// 单独抽出来是为了「改一个数就能调整行为」，而且这些数**不该拍脑袋定**：
// 判定模型出厂往往是未校准的，阈值该用你自己的标注数据算出来。
// ═══════════════════════════════════════════════════════════

const T = {
  needsTool: 0.5,
  toolAuto: 0.6,
  inputPick: 0.5,
  riskAuth: 2,
  riskAudit: 1,
  stepOk: 0.5,
  done: 0.6,
  deliver: 0.6,
}

// ═══════════════════════════════════════════════════════════
// 1 · 这一步需要动手吗
//
// 不需要动手就直接生成回答，**省掉整个工具循环**。
// 常规 agent 也"判断"这件事，但方式是让大模型输出一段话来表达它。
// ═══════════════════════════════════════════════════════════

export const needsTool = defineDecision({
  id: 'loop.needsTool',
  describe: '这一步需要调用工具，还是可以直接回答？',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    steps_done: (ctx.history ?? []).length,
    last: clip(ctx.lastResult ?? '（还没有做过任何动作）', 300),
  }),

  questions: {
    needs_tool: noul(
      'The agent still needs to call a tool before it can answer the task; no tool call now would mean answering with information it does not have yet',
      {
        true: 'the task requires reading, listing, writing or running something first',
        false: 'there is already enough information to answer directly',
      },
    ),
  },

  policy: [
    { when: probGte('needs_tool', T.needsTool), action: 'use_tool', reason: `needs_tool ≥ ${T.needsTool}` },
    { action: 'answer', reason: '信息已足够，直接生成回答' },
  ],
})

// ═══════════════════════════════════════════════════════════
// 2 · 用哪个工具
//
// ★ 选项**每一步重建**，不是一开始定死的。
//   定死的选项列表会让模型去选一个早就不存在的动作 ——
//   比如刚写完文件，"write_file" 就不该再出现在候选里。
// ═══════════════════════════════════════════════════════════

export const pickTool = defineDecision({
  id: 'loop.pickTool',
  describe: '下一步调用哪个工具（选项随已做的动作动态重建）',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    // ★ 用一句话讲清"已经做过什么"，而不是丢一个数组让模型自己解析。
    //   决策帧的表达方式直接决定判定质量 —— 实测：只放数组时，
    //   模型会重复选已经做过的动作。
    already_done: describeDone(ctx),
    files_known: (ctx.files ?? []).slice(0, 20),
    // 模型必须知道读过哪些 —— 帧里没有的信号它判不出来（见 §8.2），
    // 少了这个它会重复读同一个文件。
    already_read: (ctx.readFiles ?? []).slice(0, 10),
    last_result: clip(ctx.lastResult ?? '', 300),
  }),

  questions: (ctx: AgentCtx) => ({
    tool: choice('Which tool should the agent call next?', toolsFor(ctx)),
  }),

  policy: [
    { when: topGte('tool', T.toolAuto), action: 'call', reason: `选中项概率 ≥ ${T.toolAuto}` },
    { action: 'escalate', reason: '工具选择置信度不足 → 交回上层，不猜' },
  ],
})

// ═══════════════════════════════════════════════════════════
// 2b · 给选定的工具挑一个输入
//
// 审计 N3：工具参数以前是**写死的代码**（`defaultInput` 永远返回 `files[0]`），
// 配合 `toolsFor` 会把用过的动作从候选里删掉，结果是
// **一个 agent 生命周期内 read_file 只能触发一次，且只能读第一个文件** ——
// 任务「读取目录里的**全部** TypeScript 文件」在这个实现下不可能完成。
//
// 按三分法，这是「挑选」不是「生成」，所以它该是一次判定：
// 候选 = 还没读过的文件，由 `fileOptions` 每步重建。
//
// 「写什么内容」是生成，仍由调用方提供（不在本次范围内）。
// ═══════════════════════════════════════════════════════════

export const pickInput = defineDecision({
  id: 'loop.pickInput',
  describe: '给已选定的工具挑一个输入（读/写哪个文件）',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    tool: ctx.lastTool ?? '',
    already_read: (ctx.readFiles ?? []).slice(0, 10),
    candidates: (ctx.files ?? []).slice(0, 20),
  }),

  questions: (ctx: AgentCtx) => ({
    file: choice(pickInputInstructions(ctx), fileOptions(ctx)),
  }),

  policy: [
    { when: topGte('file', T.inputPick), action: 'use', reason: `选中项概率 ≥ ${T.inputPick}` },
    { action: 'escalate', reason: '文件选择置信度不足 → 交回上层，不猜' },
  ],
})

/**
 * 每一步重建候选动作。
 *
 * 这是被反复验证过的一条经验：**固定的选项列表会让判定模型
 * 去选一个已经不适用的动作。** 所以候选要跟着状态走。
 */
function toolsFor(ctx: AgentCtx): Record<string, string> {
  const done = new Set((ctx.history ?? []).map((h) => h.tool))
  const out: Record<string, string> = {}

  // ★ 两条经验都写在这里：
  //
  //   1. **做过的动作不再是候选** —— 固定候选列表会让模型去选一个
  //      已经不适用的动作。
  //
  //   2. **criteria 要写成"什么条件下该选它"，不是名词标签。**
  //      实测对比：写成 "列出工作目录里的文件" 时，模型列完文件就选了 done；
  //      写成条件句之后它才知道"任务还没做完"。
  //      （Jev Engineering 规则 2：问题 ID 不会到达模型，判据必须写进指令和选项里）

  if (!done.has('list_dir'))
    out.list_dir = 'The agent does not yet know which files exist in the working directory.'

  if ((ctx.files ?? []).length > 0) {
    // ★ 审计 N3：以前这里只要有 done.has('read_file') 就永久移除它，
    //   于是「读取全部 TypeScript 文件」这类任务不可能完成。
    //   现在只要**还有没读过的文件**，read_file 就保持候选。
    const unread = unreadFiles(ctx)
    if (unread.length)
      out.read_file =
        unread.length === 1
          ? 'The content of one file is still needed to make progress and has not been read yet.'
          : `The contents of ${unread.length} files are still needed: ${unread.slice(0, 5).join(', ')}.`

    // ★ `write_file` 有两道门，缺一不可：
    //
    //   1. **调用方必须提供内容来源**（`ctx.canWrite`）。没有来源时它根本
    //      不该出现在候选里 —— 出现了模型就会选，而 loop 拿不出内容，
    //      旧代码只能填占位符，那个占位符会被真的写到盘上。
    //
    //   2. **写过就不再是候选**（§8.4 实测：写完文件后 `write_file` 还在候选里，
    //      模型会接着选它）。这和 read_file 不对称是有意的：read_file 有
    //      「还没读过」这个可判定的目标（`unreadFiles`），而 write_file
    //      没有「还没写过」的对应概念 —— 与其猜，不如撤掉。
    if (ctx.canWrite && !done.has('write_file'))
      out.write_file = 'A file must be created or its content changed.'
  }

  out.done =
    'Everything the task asks for has already been done; calling any other tool would not add information.'

  return out
}

/** 还没读过的文件 */
export function unreadFiles(ctx: AgentCtx): string[] {
  const read = new Set(ctx.readFiles ?? [])
  return (ctx.files ?? []).filter((f) => !read.has(f))
}

/**
 * 候选文件的硬上界。
 *
 * 和 `budget.ts` 的 `LIMITS[*].maxOptions` 是同一个数：choice 的选项共享一个
 * 固定 head 预算（192/256 token），选项越多每个分到的越少，文本就互相不可区分
 * ——实测 77 个选项时选中项概率掉到 0.425，也就是**基本在瞎猜**。
 *
 * 截断不会永久丢能力：`unreadFiles` 每步重建，读掉前 20 个之后，
 * 下一批 20 个自动进入窗口。但**窗口本身必须说出来**——
 * 只列 20 个而不提总数，读起来就像"目录里只有这 20 个文件"。
 */
export const MAX_FILE_OPTIONS = 20

/**
 * 给 `pickInput` 构造候选。**每步重建** —— 读过的文件不再出现。
 *
 * criteria 写成条件句而不是名词标签：`read_file` 的每个候选都要说清
 * 「为什么还需要读它」。这是 Jev Engineering 规则 2 的落地
 * （问题 ID 不会到达模型，判据必须写进指令和选项里）。
 */
function fileOptions(ctx: AgentCtx): Record<string, string> {
  const out: Record<string, string> = {}
  if (ctx.lastTool === 'write_file') {
    for (const f of (ctx.files ?? []).slice(0, MAX_FILE_OPTIONS))
      out[f] = `The task requires creating or changing ${f}.`
    return out
  }
  for (const f of unreadFiles(ctx).slice(0, MAX_FILE_OPTIONS)) {
    out[f] = `The task still needs the contents of ${f}, and it has not been read yet.`
  }
  return out
}

/** 候选总数（截断前），用来判断窗口有没有藏掉东西 */
function fileOptionTotal(ctx: AgentCtx): number {
  return ctx.lastTool === 'write_file' ? (ctx.files ?? []).length : unreadFiles(ctx).length
}

/**
 * `pickInput` 的指令。窗口截断时必须**明说** —— 见 `MAX_FILE_OPTIONS`。
 */
function pickInputInstructions(ctx: AgentCtx): string {
  const base = 'Which file should this tool call target?'
  const total = fileOptionTotal(ctx)
  if (total <= MAX_FILE_OPTIONS) return base
  return `${base} Only the first ${MAX_FILE_OPTIONS} of ${total} candidates are listed.`
}

/**
 * 这个 ctx 下有没有可选的输入。
 *
 * **没有候选就不要问** —— 一个 `criteria` 为空的 choice 是无效问题，
 * 会得到无意义的答案。调用方据此决定「不做这次判定」。
 */
export function hasFileOptions(ctx: AgentCtx): boolean {
  return Object.keys(fileOptions(ctx)).length > 0
}

/** 把"已经做过什么"写成一句人能读的话，喂给判定模型 */
function describeDone(ctx: AgentCtx): string {
  const h = ctx.history ?? []
  if (!h.length) return 'nothing yet'
  const tools = [...new Set(h.map((x) => x.tool))]
  return `already called: ${tools.join(', ')} (${h.length} step${h.length > 1 ? 's' : ''})`
}

// ═══════════════════════════════════════════════════════════
// 3 · 这次调用多危险
//
// 原始问题：以前的判断只有两种极端 ——
//   要么所有工具全放行（危险），要么每次都弹窗问人（没法用）。
// 按风险分级之后，只有真正不可逆的操作才需要授权。
//
// ★ 硬闸门：不可逆操作必须显式授权，**不给概率任何绕过机会**。
//   判定模型可以决定「要不要问人」，绝不能决定「要不要跳过授权」。
// ═══════════════════════════════════════════════════════════

export const gradeRisk = defineDecision({
  id: 'loop.gradeRisk',
  describe: '给这次工具调用打风险分，驱动分级审批',

  state: (ctx: AgentCtx) => {
    const t = ctx.lastTool
    // ★ 工具的静态风险基线。`tools.ts` 里四个工具都声明了 `baseRisk`，
    //   但**一直没有任何读取方** —— 声明了却不喂进帧，等于没声明：
    //   帧里没有的信号判定模型看不见（见 AGENTS.md §8.2）。
    //
    //   工具名是不可信输入（模型给的），认不出来时**不编一个数**，
    //   直接不放这个字段 —— 0 分的意思是"只读"，不能用它冒充"未知"。
    const base = t && isToolName(t) ? { base_risk: TOOLS[t].baseRisk } : {}
    return {
      tool: t ?? 'unknown',
      ...base,
      target: clip(lastInput(ctx), 200),
      task: clip(ctx.task, 300),
    }
  },

  questions: {
    risk: score('How risky is this tool call?', [
      'read-only',
      'reversible write',
      'irreversible',
      'destructive',
    ]),
    needs_auth: noul(
      'This call must be explicitly authorised by a human before it runs',
      {
        true: 'it can destroy data, spend money, or leave the machine',
        false: 'it only reads or writes inside the working directory',
      },
    ),
  },

  policy: [
    // 不接受概率绕过 —— 风险分够高就是必须授权
    { when: scoreGte('risk', T.riskAuth), action: 'ask_human', reason: `risk ≥ ${T.riskAuth} → 必须显式授权（不接受概率绕过）` },
    { when: probGte('needs_auth', 0.5), action: 'ask_human', reason: '模型判定需要授权' },
    { when: scoreGte('risk', T.riskAudit), action: 'auto_audit', reason: `risk ≥ ${T.riskAudit} → 执行但记审计` },
    { action: 'auto', reason: '只读，直接放行' },
  ],
})

// ═══════════════════════════════════════════════════════════
// 4 · 这一步成功了吗
//
// 常规做法是每一步都叫一次大模型来判断"工具输出看起来对吗"。
// ═══════════════════════════════════════════════════════════

export const stepOk = defineDecision({
  id: 'loop.stepOk',
  describe: '刚才那次工具调用是否达到了预期效果',

  state: (ctx: AgentCtx) => ({
    tool: ctx.lastTool ?? 'unknown',
    input: clip(lastInput(ctx), 200),
    output: clip(ctx.lastResult ?? '', 500),
    task: clip(ctx.task, 300),
  }),

  questions: {
    ok: noul(
      'The tool call succeeded and produced a usable result for the task; there is no error or empty output that blocks progress',
      {
        true: 'the output contains what the task needed',
        false: 'the output is an error, empty, or clearly not what was asked for',
      },
    ),
  },

  policy: [
    { when: probGte('ok', T.stepOk), action: 'continue', reason: `ok ≥ ${T.stepOk}` },
    // 名字只承诺实际发生的事：loop 收到这个动作就停机，没有重试分支。
    // 以前叫 `retry_or_stop`，但重试需要一个错误分类策略 —— 那个策略不存在，
    // 所以「retry」不能写进 action 名里。约束见 JevLoop/AGENTS.md。
    { action: 'stop', reason: '工具结果不可用 → 停下（没有错误分类策略，盲目重试不是改进）' },
  ],
})

// ═══════════════════════════════════════════════════════════
// 5 · 任务完成了吗
//
// 常规做法是 max_iter 硬切。语义早停能让简单的任务立刻结束，
// 而不是傻等到迭代上限。
// ═══════════════════════════════════════════════════════════

export const isDone = defineDecision({
  id: 'loop.isDone',
  describe: '任务是否已经完成，可以开始生成回答了',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    steps: (ctx.history ?? []).slice(-6).map((h) => `${h.tool}(${clip(h.input, 60)}) → ${clip(h.result, 80)}`),
  }),

  questions: {
    done: noul(
      'The agent has done everything the task requires; any further tool call would not add information or change the outcome',
      {
        true: 'the goal stated in the task has been reached',
        false: 'something the task asks for is still missing',
      },
    ),
  },

  policy: [
    { when: probGte('done', T.done), action: 'finish', reason: `done ≥ ${T.done}` },
    { action: 'keep_going', reason: '任务还没完成' },
  ],
})

// ═══════════════════════════════════════════════════════════
// 6 · 这个回答能交付吗
//
// 原始问题：以前根本没有这一步，生成完直接返回，
// 靠事后人工抽查。现在每条输出都能过一遍闸门。
// ═══════════════════════════════════════════════════════════

export const canDeliver = defineDecision({
  id: 'loop.canDeliver',
  describe: '生成的回答是否完整、准确、可以直接交付',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    answer: clip(ctx.draft ?? '', 900),
    // 最近一次结果给足空间：交付闸门要拿它逐句核对回答，
    // 截太短会让闸门正确地判出「回答里有证据不支持的内容」——
    // 那是帧的问题，不是回答的问题（实测在 100 字符预算下误报）
    evidence: (ctx.history ?? []).slice(-3).map((h, i, all) => {
      const budget = i === all.length - 1 ? 600 : 200
      return `${h.tool}(${clip(h.input, 60)}) → ${clip(h.result, budget)}`
    }),
  }),

  questions: {
    deliverable: noul(
      'The answer is complete and correct for the task, and can be returned to the user as-is',
      {
        true: 'it addresses the task and is consistent with what the tools returned',
        false: 'it is incomplete, off-topic, or contradicts the tool output',
      },
    ),
    unsupported: noul(
      'The answer states something that the tool output does not support',
      {
        true: 'it claims a fact, file or result that was never observed',
        false: 'everything it says traces back to a tool result',
      },
    ),
  },

  policy: [
    { when: probGte('unsupported', 0.5), action: 'revise', reason: '回答里有工具输出不支持的内容' },
    { when: probGte('deliverable', T.deliver), action: 'deliver', reason: `deliverable ≥ ${T.deliver}` },
    { action: 'revise', reason: '回答不达标 → 重来一次' },
  ],
})

// ── 工具 ─────────────────────────────────────────────────────

function lastInput(ctx: AgentCtx): string {
  const h = (ctx.history ?? []).at(-1)
  return h ? h.input : ''
}
