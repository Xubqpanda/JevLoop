/**
 * JevLoop · L3 State Compiler —— ctx → 有界决策帧
 *
 * AGENTS.md §8.2 说这个组件「是必需组件，不是优化」，因为
 * **帧里没有的东西，判定模型判不出来** —— 不是判错，是压根看不见。
 * 这个文件就是那个组件：它把 agent 状态压成有界、可判定的帧，
 * 并决定每一步有哪些候选。
 *
 * 实测依据（§8.2）：`canDeliver` 曾把工具结果 clip 到 100 字符，
 * 交付闸门拿被截断的证据去核对回答，**正确地**判出 `unsupported=0.67`。
 * 判定没错，是帧喂少了。改成 600 后立刻通过。
 *
 * 这一层**不许知道有哪些工具存在、也不许知道判定节点长什么样** ——
 * 它只做投影和有界化。候选的最终取舍是 L4 的策略（见 `decisions.ts`）。
 * 分开的理由：帧 bug 和策略 bug 是两类 bug，住在一起时无法分别测试
 * （第十轮 R2/R5 就是两条只能靠整个 agent 才能复现的帧 bug）。
 *
 * @module JevLoop/frame
 */

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

/**
 * 每一步重建候选动作。
 *
 * 这是被反复验证过的一条经验：**固定的选项列表会让判定模型
 * 去选一个已经不适用的动作。** 所以候选要跟着状态走。
 */
export function toolsFor(ctx: AgentCtx): Record<string, string> {
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
export function fileOptions(ctx: AgentCtx): Record<string, string> {
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
export function pickInputInstructions(ctx: AgentCtx): string {
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
export function describeDone(ctx: AgentCtx): string {
  const h = ctx.history ?? []
  if (!h.length) return 'nothing yet'
  const tools = [...new Set(h.map((x) => x.tool))]
  return `already called: ${tools.join(', ')} (${h.length} step${h.length > 1 ? 's' : ''})`
}

// ── 工具 ─────────────────────────────────────────────────────

export function lastInput(ctx: AgentCtx): string {
  const h = (ctx.history ?? []).at(-1)
  return h ? h.input : ''
}
