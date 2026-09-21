/**
 * JevLoop · agent 的全部判定
 *
 * ══════════════════════════════════════════════════════════════
 *  这个文件就是 JevLoop 的全部主张。
 * ══════════════════════════════════════════════════════════════
 *
 * 一个常规 agent 的 loop 里，下面这些事都会写成一次大模型调用：
 *
 *     要不要动手？  用哪个工具？  读哪个文件？  这个操作危险吗？
 *     成功了吗？    做完了吗？    这个回答能发出去吗？
 *
 * 它们全都不是「生成」，而是「选择 / 打分 / 是否」。
 * 也就是说：**你一直在用生成的价格，买判定的答案。**
 *
 * 这些判定每个只要一次前向传播，加起来通常几十到几百毫秒，
 * 换来的是整个 loop 里只剩一次真正的大模型调用。
 *
 * ── 这里只剩「帧」，问题和策略来自 DECISION.md ────────────────
 *
 * 每个节点现在长这样：
 *
 *     defineDecision({
 *       id: 'loop.needsTool',
 *       state: ctx => ({ ... }),        ← 留在这里：帧构造器是个**函数**
 *       ...compiled('needs_tool'),      ← questions + policy 来自文件
 *     })
 *
 * **为什么帧必须留在这里**：`DECISION.md` 是一门声明式的格式，能表达
 * 「问什么」和「答成什么就走哪一步」，但它表达不了**函数** ——
 * 而帧恰恰是「从完整的 ctx 里挑哪几个字段、各截多长」的一段代码。
 * 硬把它塞进 markdown 只有两条路：发明一门真正的 DSL，或者让帧退化成
 * 「把 ctx 全塞进去」（那会撑爆 512/1024 的上下文，见 §8.2）。
 * 两条都不划算，所以这条边界是**刻意的**，不是没做完。
 *
 * 好处是那句主张现在成立：改 `DECISION.md` 里的问题和策略，
 * 运行时跟着变，不需要改这个文件。
 *
 * ## 为什么不能再拆
 *
 * 七个判定节点是**同一件事** —— 这个 loop 会问哪些问题 —— 每个 35-45 行，
 * 每个本身就是最小闭包。拆成七个文件只会让「一共有哪些判定」从一眼可见
 * 变成要翻目录。**内聚不是越小越好**；AGENTS.md §12 特意把它留作反例。
 * **行数不在这里写** —— 它会漂，跑 `npm run check` 看当前值。
 *
 * @module JevLoop/decisions
 */

import { readFileSync } from 'node:fs'

import { defineDecision } from './vocab-decision.ts'
import { choice } from './vocab.ts'
import type {
  AnswerSet,
  ChoiceQuestion,
  NoulQuestion,
  QuestionSet,
  ScoreQuestion,
} from './vocab.ts'
import type { PolicyRule } from './vocab-decision.ts'
import { clip } from './budget.ts'
import { TOOLS, isToolName } from './tools.ts'
import { parseDecisionDoc, type DecisionDoc, type DocBlock } from './decisiondoc.ts'
import { compilePolicy, compileQuestions } from './decision-compile.ts'

import { toolsFor, fileOptions, pickInputInstructions, describeDone, lastInput } from './frame.ts'
import type { AgentCtx } from './frame.ts'
export type { AgentCtx, StepRecord } from './frame.ts'

// ═══════════════════════════════════════════════════════════
// DECISION.md 的编译
//
// ★ **这是运行时真的在用的那一份**，不是文档。
//
// 在这之前 `DECISION.md` 是一份被机器对账的文档：代码是正本，测试
// 双向检查两边一致。现在反过来了 —— 问到判定模型的**问题和策略**
// 都从文件里编译出来，代码只留帧构造器（理由见模块头）。
//
// ⚠️ **代价要说清楚：这个模块在 import 时会读盘。** 一个纯计算的内核
// 多了一处 IO 副作用。换来的是「改文件 → 行为变」这条主张成立，
// 而那正是 DECISION.md 存在的全部理由。
// ═══════════════════════════════════════════════════════════

/**
 * 读并解析 `DECISION.md`。
 *
 * 路径用 `import.meta.url` 相对定位，所以 `src/` 下跑和发布后从 `dist/`
 * 跑都找得到同一个文件（`package.json` 的 `files` 里有它）。
 *
 * @throws 文件读不到、或者解析出任何 problem。
 *   **解析出问题必须当场炸，不能降级。** 一个编译不了的谓词会变成
 *   「这条规则永不命中」= **一道空闸门** —— 作者以为自己加了一道人
 *   工审核，实际没有，而且是 fail open 的。§8.5 那两条闸门的存在理由
 *   就是这个，静默丢掉它们的后果比启动失败严重得多。
 */
function loadDecisionDoc(): DecisionDoc {
  const url = new URL('../DECISION.md', import.meta.url)
  let md: string
  try {
    md = readFileSync(url, 'utf8')
  } catch (err) {
    // 吞的是「文件不在」—— 把它换成一条**能直接照做**的报错。
    // 内核没有 DECISION.md 就没有判定规格，没有可用的降级。
    throw new Error(
      `读不到判定规格 ${url.pathname}（${(err as Error).message}）—— ` +
        `JevLoop 的问题与策略都从 DECISION.md 编译，没有它就没有判定节点`,
    )
  }
  const doc = parseDecisionDoc(md)
  if (doc.problems.length > 0) {
    const lines = doc.problems.map((p) => `  DECISION.md:${p.line}  ${p.message}`).join('\n')
    throw new Error(`DECISION.md 有 ${doc.problems.length} 处解析不了：\n${lines}`)
  }
  return doc
}

const DOC = loadDecisionDoc()

/**
 * `DECISION.md` 里 `## generator` 那一段 —— **生成器的 system prompt**。
 *
 * 这就是「一份文件，两个消费者」的另一半：结构块编译成判定模型的问题，
 * 散文进 system prompt。它和判定节点住同一个文件，所以「这条 loop 怎么
 * 作决定」和「它怎么写答案」是放在一起改的，不会一个改了另一个忘掉。
 *
 * 空字符串是允许的（解析器不强制这一段），这时 `HttpGenerator` 退回它
 * 内置的那句 —— 见 `llm.ts` 的 `DEFAULT_INSTRUCTION`。
 */
export const GENERATOR_INSTRUCTION: string = DOC.generatorSection

/** 按块 id 取块。写错名字要当场知道，不是静默给一个空问题集 */
function block(id: string): DocBlock {
  const b = DOC.blocks.find((x) => x.id === id)
  if (!b) {
    throw new Error(
      `DECISION.md 里没有 '${id}' 这个块（现有：${DOC.blocks.map((x) => x.id).join('、')}）`,
    )
  }
  return b
}

/**
 * 把一条策略编译出来。
 *
 * 两条策略（`pick_tool` / `pick_input`）的**问题**是运行时算的（候选每步
 * 重建），所以它们不能整块编译 —— 但策略仍然是文件说了算，分开取。
 */
function policyOf(id: string): { policy: PolicyRule<AnswerSet>[] } {
  const pol = compilePolicy(block(id))
  if (!pol) throw new Error(`DECISION.md 的 '${id}' 没有 policy —— 一个没有策略的判定节点不会产生任何动作`)
  if (!pol.ok) {
    throw new Error(
      `DECISION.md 的 '${id}' 有编译不了的谓词：${pol.problems.join('；')} —— ` +
        `「看不懂这个谓词」不能编码成「这条永远命中」，所以只能停下`,
    )
  }
  return { policy: pol.rules }
}

/**
 * 整块编译：问题 + 策略都来自文件。
 *
 * 用于问题集**不随状态变**的那五个节点。候选每步重建的那两个用
 * `askOf` + `policyOf` —— 因为 markdown 表达不了 `toolsFor(ctx)`。
 *
 * ── 为什么要显式传 `expect` ────────────────────────────────────
 *
 * ★ 问题的名字现在住在 **markdown** 里，而代码在按名字读答案
 *   （`agent.ts` 读 `answers.risk.score`）。这层耦合是真的存在，
 *   藏起来只会让它变成运行时的 `undefined`。
 *
 *   传 `expect` 一次做两件事：
 *
 *   1. **类型**：TS 拿得到 `{ risk: ScoreQuestion }`，于是 `answers.risk.score`
 *      仍然是有类型的，不是 `any`。
 *   2. **启动时校验**：文件里的问题 id 和声明的不一致就当场炸 ——
 *      改错了名字会立刻知道，而不是等到某一步 `answers.risk` 是 undefined。
 *
 *   这比改之前**更严**：以前 id 只存在于代码里，没人检查文件对不对得上。
 */
function compiled<Q extends QuestionSet>(
  id: string,
  expect: readonly (keyof Q & string)[],
): { questions: Q; policy: PolicyRule<AnswerSet>[] } {
  const b = block(id)
  const q = compileQuestions(b)
  if (!q) throw new Error(`DECISION.md 的 '${id}' 编译不出问题集`)

  const actual = Object.keys(q).sort()
  const wanted = [...expect].sort()
  if (actual.join(',') !== wanted.join(',')) {
    throw new Error(
      `DECISION.md 的 '${id}' 问的是 [${actual.join(', ')}]，而代码声明的是 [${wanted.join(', ')}] —— ` +
        `代码在按名字读答案，对不上就是运行时的 undefined`,
    )
  }
  return { questions: q as Q, ...policyOf(id) }
}

/** 块里唯一那个问题的 `ask` 原文 —— 给「候选运行时算」的那两个块用 */
function askOf(id: string, expect: readonly string[]): string {
  const b = block(id)
  const actual = b.questions.map((q) => q.id)
  // 同样校验：这两个块的**问题**由代码算（候选每步重建），但问题 id
  // 仍然要和文件对上 —— 代码在按 `answers.tool` / `answers.file` 读
  if (actual.join(',') !== [...expect].join(',')) {
    throw new Error(
      `DECISION.md 的 '${id}' 问的是 [${actual.join(', ')}]，而代码声明的是 [${expect.join(', ')}]`,
    )
  }
  const q = b.questions[0]
  if (!q) throw new Error(`DECISION.md 的 '${id}' 里没有问题，取不到 ask`)
  return q.ask
}


// ═══════════════════════════════════════════════════════════
// 阈值集中在这里
//
// 单独抽出来是为了「改一个数就能调整行为」，而且这些数**不该拍脑袋定**：
// 判定模型出厂往往是未校准的，阈值该用你自己的标注数据算出来。
// ═══════════════════════════════════════════════════════════

/**
 * 上文进决策帧的字符上限。
 *
 * 200 是**背景**的量级：要能说清「前面问过什么」，但不能挤掉这一轮真正
 * 要看的东西（任务 400、上次结果 300）。帧上下文只有 512/1024，
 * 多轮的代价必须显式地小（§8.2）。
 */
const EARLIER_MAX_CHARS = 200

/*
 * 门限**不在这里** —— 它们现在住在 `DECISION.md` 的 `policy:` 里
 * （`prob:ok >= 0.6` 这样写）。这里曾经有一个 `T` 对象，理由是
 * 「改一个数就能调整行为」；搬进文件之后同一个目的达标得更彻底：
 * 连**哪条规则用它、命中之后做什么**都在一起，不用两头看。
 */
// ═══════════════════════════════════════════════════════════
// 1 · 这一步需要动手吗
//
// 不需要动手就直接生成回答，**省掉整个工具循环**。
// 常规 agent 也"判断"这件事，但方式是让大模型输出一段话来表达它。
// ═══════════════════════════════════════════════════════════

/**
 * 这一步需要动手吗。`use_tool` / `answer` —— 后者**直接跳到生成、省掉整个工具循环**，
 * 所以这是最省的一步，也是最该早判的一步。
 */
export const needsTool = defineDecision({
  id: 'loop.needsTool',
  describe: '这一步需要调用工具，还是可以直接回答？',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    // ★ 多轮：这一句是「再读一遍那个文件」里的"那个"唯一能落地的地方。
    //   §8.2：帧里没有的，模型判不出来 —— 不是判错，是压根看不见。
    //   有界（200 字符）是因为帧本身有预算，而它是**背景**不是主体。
    earlier: clip(ctx.earlier ?? '', EARLIER_MAX_CHARS),
    steps_done: (ctx.history ?? []).length,
    last: clip(ctx.lastResult ?? '（还没有做过任何动作）', 300),
  }),

  // 问题与策略都来自 DECISION.md 的 needs_tool 块
  ...compiled<{ needs_tool: NoulQuestion }>('needs_tool', ['needs_tool']),
})

// ═══════════════════════════════════════════════════════════
// 2 · 用哪个工具
//
// ★ 选项**每一步重建**，不是一开始定死的。
//   定死的选项列表会让模型去选一个早就不存在的动作 ——
//   比如刚写完文件，"write_file" 就不该再出现在候选里。
// ═══════════════════════════════════════════════════════════

/**
 * 下一步调哪个工具。候选由 `toolsFor(ctx)` **每步重建**：做过的动作会消失，
 * 还没读过的文件仍然在。选中项概率不过门限就 `escalate`，**不猜**。
 */
export const pickTool = defineDecision({
  id: 'loop.pickTool',
  describe: '下一步调用哪个工具（选项随已做的动作动态重建）',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    // ★ 多轮：同上。挑工具时「上文」尤其重要 ——
    //   「那个文件」要靠它才能落到一个具体路径上。
    earlier: clip(ctx.earlier ?? '', EARLIER_MAX_CHARS),
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

  // ★ 候选**每步重建**（`toolsFor`），而 markdown 表达不了函数 ——
  //   所以问题由代码算，`ask` 原文和策略仍来自文件。
  //   文件里那个块写了 `dynamic: toolsFor(ctx)` 就是在声明这件事。
  questions: (ctx: AgentCtx) => ({
    tool: choice(askOf('pick_tool', ['tool']), toolsFor(ctx)),
  }) satisfies { tool: ChoiceQuestion },
  ...policyOf('pick_tool'),
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

/**
 * 给已选定的工具挑一个输入（读 / 写哪个文件）。
 *
 * 调用方必须先问 `hasFileOptions(ctx)`：**没有候选就不要问** ——
 * 一个 `criteria` 为空的 choice 是无效问题，只会拿到无意义的答案。
 */
export const pickInput = defineDecision({
  id: 'loop.pickInput',
  describe: '给已选定的工具挑一个输入（读/写哪个文件）',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    tool: ctx.lastTool ?? '',
    already_read: (ctx.readFiles ?? []).slice(0, 10),
    candidates: (ctx.files ?? []).slice(0, 20),
  }),

  // 同 `pickTool`：候选由 `fileOptions` 每步重建，问题留代码，策略来自文件
  // `satisfies` 在这里而不是在 `choice(...)` 上：要断言的是**这个对象**
  // 的形状 —— 代码在按 `answers.file` 读答案
  questions: (ctx: AgentCtx) =>
    ({
      file: choice(pickInputInstructions(ctx), fileOptions(ctx)),
    }) satisfies { file: ChoiceQuestion },
  ...policyOf('pick_input'),
})


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

/**
 * 给这次工具调用打风险分，驱动分级审批。`ask_human` / `auto_audit` / `auto`。
 *
 * ★ **授权闸门不接受概率绕过**：`risk ≥ 2` 是一条硬规则。
 * 判定模型可以决定「要不要问人」，**绝不能**决定「要不要跳过授权」。
 */
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

  // 问题与策略都来自 DECISION.md 的 grade_risk 块。
  // 那条硬闸门（risk 够高就必须授权）现在写在文件里 —— 见那个块的 policy。
  ...compiled<{ risk: ScoreQuestion; needs_auth: NoulQuestion }>('grade_risk', ['risk', 'needs_auth']),
})

// ═══════════════════════════════════════════════════════════
// 4 · 这一步成功了吗
//
// 常规做法是每一步都叫一次大模型来判断"工具输出看起来对吗"。
// ═══════════════════════════════════════════════════════════

/**
 * 刚才那次工具调用成功了吗。`continue` / `stop`。
 *
 * **没有重试分支**：重试需要一个错误分类策略，而那个策略不存在 ——
 * 所以动作名只承诺实际发生的事（以前叫 `retry_or_stop`，名字承诺了做不到的事）。
 */
export const stepOk = defineDecision({
  id: 'loop.stepOk',
  describe: '刚才那次工具调用是否达到了预期效果',

  state: (ctx: AgentCtx) => ({
    tool: ctx.lastTool ?? 'unknown',
    input: clip(lastInput(ctx), 200),
    output: clip(ctx.lastResult ?? '', 500),
    // ★ **没有 `task`**，而且这是修出来的，不是一开始就这么写的。
    //
    //   帧里带着 task，问题又写着「produced a usable result **for the task**」、
    //   判据写着「the output contains **what the task needed**」—— 三处一起
    //   把这一步的判定拉到了**任务级**。而「任务完成了吗」是 `isDone` 的职责。
    //
    //   实测（这个 bug 就是这么发现的）：任务「读一下 invoice.ts」，
    //   第一步 `list_dir` 返回文件列表 —— 它**确实成功**了，但它没有回答
    //   「这个文件定义了哪些函数」，于是 `ok=0.470 < 0.6` 判否，
    //   动作是 `stop`，**整个循环结束**。任何需要多于一个工具的任务都跑不完。
    //
    //   `stepOk` 的 `describe` 一直写的是「刚才那次**工具调用**是否达到预期效果」。
    //   越界的是帧和措辞，不是这个节点的意图。
    already_read: (ctx.readFiles ?? []).length,
  }),

  // 问题与策略都来自 DECISION.md 的 step_ok 块
  ...compiled<{ ok: NoulQuestion }>('step_ok', ['ok']),
})

// ═══════════════════════════════════════════════════════════
// 5 · 任务完成了吗
//
// 常规做法是 max_iter 硬切。语义早停能让简单的任务立刻结束，
// 而不是傻等到迭代上限。
// ═══════════════════════════════════════════════════════════

/**
 * 任务完成了吗。`finish` / `keep_going`。
 *
 * 语义早停：简单的任务立刻结束，而不是傻等到 `maxSteps` 上限。
 */
export const isDone = defineDecision({
  id: 'loop.isDone',
  describe: '任务是否已经完成，可以开始生成回答了',

  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    steps: (ctx.history ?? []).slice(-6).map((h) => `${h.tool}(${clip(h.input, 60)}) → ${clip(h.result, 80)}`),
  }),

  // 问题与策略都来自 DECISION.md 的 is_done 块
  ...compiled<{ done: NoulQuestion }>('is_done', ['done']),
})

// ═══════════════════════════════════════════════════════════
// 6 · 这个回答能交付吗
//
// 原始问题：以前根本没有这一步，生成完直接返回，
// 靠事后人工抽查。现在每条输出都能过一遍闸门。
// ═══════════════════════════════════════════════════════════

/**
 * 生成的回答能交付吗。`deliver` / `revise`。
 *
 * 它拿工具结果逐句核对回答，所以 `evidence` 的预算给得比别处宽（最近一次 600 字符）：
 * **帧喂少了会让它"正确地"判出「回答里有证据不支持的内容」—— 那是帧的问题，
 * 不是回答的问题**（§8.2）。
 */
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

  // 问题与策略都来自 DECISION.md 的 can_deliver 块
  ...compiled<{ deliverable: NoulQuestion; unsupported: NoulQuestion }>('can_deliver', ['deliverable', 'unsupported']),
})
