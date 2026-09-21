/**
 * JevLoop · 造 `write_file` 的输入
 *
 * ══════════════════════════════════════════════════════════════
 *  **目标路径和内容都是「生成」，不是「挑选」。**
 * ══════════════════════════════════════════════════════════════
 *
 * `write_file` 的输入格式是 `路径\n内容`（第一行路径，其余是内容 —— 见
 * `tools.ts`）。这两样都由这里调生成器产出。
 *
 * ── 为什么目标路径不能交给判定 ──────────────────────────────────
 *
 * 这一步原来是「判定挑一个文件」，候选来自 `ctx.files`，也就是**已经存在
 * 的文件**。而 `write_file` 最常见的用法恰恰是**建一个新文件** ——
 * 那个名字不在候选里，也**不可能**在候选里：它还不存在。
 *
 * 实测（2026-09-21）：任务「把 invoice.ts 和 retry.ts 里的函数写进一个
 * **新建的** SUMMARY.md」——
 *
 *     pickTool   → write_file（0.72，正确）✓
 *     pickInput  → 候选只有 invoice.ts / notes.md / retry.ts（三个都已存在）
 *                  它选了 notes.md（1.00）—— 和任务毫无关系
 *     内容生成   → 证据里没有 notes.md 的任何东西 → 正确地说「写不了」→ 停机
 *
 * 每一环的行为都是对的。**缺的是「SUMMARY.md」这个名字没有任何地方能产生。**
 *
 * `read_file` 不受影响：它挑的就是已经存在的文件，那是一个真的闭集
 * （「还没读过的那几个」），继续交给判定（`pickInput`）。
 *
 * ── 和「写什么内容」的关系 ──────────────────────────────────────
 *
 * 同一次调用产两样：模型先写路径，再写内容。这不是省一次调用的技巧 ——
 * 而是**它们本来就是一件事**（「把这个写进那个文件」），拆成两次调用会让
 * 第二次看不见第一次选了什么。
 *
 * ── 三层各自负责什么 ────────────────────────────────────────────
 *
 *     写哪个文件 / 内容是什么   生成器（这个文件）
 *     要不要写                  判定模型（`pickTool`）
 *     能不能写                  代码（`tools.ts` 的 `safePath`、`canWrite` 那道门）
 *
 * 放 L3：它把 agent 的上下文**编译**成一个生成请求，和 `frame.ts` 编判定帧
 * 是同一件事；它需要 import `llm.ts`（L2）和 `context.ts`（L1），而 L3 可以。
 *
 * @module JevLoop/write-content
 */

import { clip } from './budget.ts'
import { EVIDENCE_POLICY, foldEvidence, priceGenerateRequest } from './context.ts'
import type { GenerateRequest, GenerateResult, Generator } from './llm.ts'

/** 一次工具调用。`StepRecord` 结构上满足它 —— 这里只声明用得上的三个字段 */
export interface WriteStep {
  tool: string
  input: string
  result: string
}

/**
 * 造输入时拿得到的上下文。**是 `AgentCtx` 的一个结构子集** ——
 * 这样 `provideWriteInput` 能直接收 `AgentCtx`，不需要任何转换或断言。
 */
export interface WriteContext {
  task: string
  /** 已经做过的工具调用。`result` 里就是**读到的文件内容**，也就是证据 */
  history?: readonly WriteStep[]
}

/**
 * `write_file` 的输入来源。签名**结构上兼容** `AgentOptions.provideWriteInput`。
 *
 * 返回 `undefined` 表示「这次写不了」→ loop 停机，而不是写一句假内容交差。
 */
export type WriteInput = (ctx: WriteContext) => Promise<string | undefined>

/** 一次工具调用的输入截多长。和 `agent.ts` 里那份**同一个数**，理由也一样：定长前缀 */
const STEP_INPUT_CHARS = 120

/** 路径那一行最多多长。超过它多半是模型把一整句话当成了文件名 */
const MAX_PATH_CHARS = 200

/**
 * 模型说「素材不够」时回的那个词。
 *
 * ★ 用一个**显式的哨兵**而不是「输出为空」：空输出分不清「它判断写不了」
 *   和「它挂了 / 被截断了」，而这两件事的处理完全不同 —— 前者该停机，
 *   后者该报错。哨兵至少是它**主动说**的。
 */
const NOTHING = 'NOTHING'

const INSTRUCTION = `Write a file.

Output exactly two parts, in this order, and nothing else:

1. The first line is the path of the file, relative to the working directory.
2. Every line from the second one on is that file's contents.

- Use only the evidence above. Do not invent identifiers, filenames or behaviour that the evidence does not show.
- No code fence around the output and no explanation of what you are about to do.
- If the evidence does not contain what this file needs, reply with exactly ${NOTHING} and nothing else.`

/**
 * 一次写文件的生成，发生后交给调用方。
 *
 * ★ **为什么要有这个回调**：它和主回答一样是**一次大模型调用**，而它原本
 *   **不发任何事件** —— 轨迹里看不见，界面的「判定 : 模型」计数也少算一次。
 *   实测（2026-09-21）：一次运行的轨迹里只有 2 个 `generate` 事件，而两次
 *   都是主回答（首次 + 修订）；写文件那一次是隐形的。
 *
 *   这个项目的全部主张就是那个比例，**漏计一次模型调用会让它是错的**。
 */
export interface WriteGeneration {
  result: GenerateResult
  /** 我们估的「能控制的那部分」输入 token。和主生成路径同一个算法 */
  estimatedInputTokens: number
}

/**
 * 用生成器造 `write_file` 的输入。
 *
 * @param generator 生成器
 * @param system system prompt。传 `DECISION.md` 的 generator 段 ——
 *   里面那两条承重规则（只用证据、用任务的语言回答）对写文件同样成立。
 * @param onGenerate 生成发生后调一次，让调用方把它记进轨迹（见 `WriteGeneration`）
 */
export function writeInputVia(
  generator: Generator,
  system?: string,
  onGenerate?: (g: WriteGeneration) => void,
): WriteInput {
  return async (ctx) => {
    const history = ctx.history ?? []

    // ★ **一次工具都没调过就不写。**
    //
    //   没有证据却让模型「写个文件」，它只能编 —— 而编出来的东西会被
    //   **真的写到盘上**。实测过的同类事故：占位符 `（内容由调用方提供）`
    //   把目标文件的全部内容替换掉了，还连写了 5 次（见 `agent.ts`）。
    //   「写不了」是诚实的结果；写一份编造的不是。
    if (history.length === 0) return undefined

    // 证据的形状和主生成路径**一模一样**（`agent.ts` 的 `buildEvidence`）——
    // 两处形状不同的话，「只用证据」这条规则在两处的含义就不一样了
    const parts = history.map((x) => ({
      text: `${x.tool}(${clip(x.input, STEP_INPUT_CHARS)}) → ${x.result}`,
      label: x.input ? `${x.tool}(${clip(x.input, 60)})` : x.tool,
    }))
    const { text: evidence } = foldEvidence(parts, EVIDENCE_POLICY)

    const req: GenerateRequest = {
      task: ctx.task,
      evidence,
      ...(system ? { system } : {}),
      instruction: INSTRUCTION,
    }

    const out = await generator.generate(req)
    // 先报出去再判内容 —— **判成「写不了」也是一次真的调用**，也要计数
    onGenerate?.({
      result: out,
      estimatedInputTokens: priceGenerateRequest({ task: req.task, evidence }).controlTokens,
    })

    return parseWriteInput(unwrapFence(out.text))
  }
}

/**
 * 模型输出的全文 → `write_file` 的输入。认不出来就 `undefined`（调用方停机）。
 *
 * 拆出来是为了能脱离生成器测 —— 这里每一条边界都对应一种**会写到盘上的错**。
 */
export function parseWriteInput(raw: string): string | undefined {
  const text = raw.trim()

  // 它主动说素材不够 —— 停机，不要把那个词当成文件名写下去
  if (text === NOTHING) return undefined

  const nl = text.indexOf('\n')
  // 只有一行：要么有路径没内容，要么模型没按格式来。两种都写不了
  if (nl < 0) return undefined

  const path = text.slice(0, nl).trim()
  if (!looksLikePath(path)) return undefined

  // 去掉首尾空白之后**补回恰好一个换行**。
  //
  //   模型常常在前后各留空行，直接写进去就是文件开头一个空行、结尾两个
  //   换行。而盘上的文本文件该以恰好一个换行结尾，这是这个仓库自己的规矩
  //   （`docs/CODE-STYLE.md` §1），也是绝大多数工具期望的。
  const content = text.slice(nl + 1).trim()
  // 空内容和「没写」在盘上分不出来，而调用方以为发生了前者
  if (content.length === 0) return undefined

  return `${path}\n${content}\n`
}

/**
 * 这一行像不像一个路径。
 *
 * ⚠️ **这不是安全边界** —— 逃逸由 `tools.ts` 的 `safePath` 挡（它用
 *    `resolve` + `relative`，绝对路径和 `..` 都出不去）。这里挡的是
 *    **格式错**：模型把一整句话或一段正文当成了文件名。那种情况下写出来
 *    的文件名字是错的，而**不会有任何东西报错** —— 正是最难发现的一类。
 */
function looksLikePath(p: string): boolean {
  if (p.length === 0 || p.length > MAX_PATH_CHARS) return false
  // 控制字符（含 NUL）不可能是文件名的一部分
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(p)) return false
  // 反引号是代码围栏的痕迹；`#` / `-` / `*` / `>` 开头是 markdown 的痕迹。
  // 两者都不是路径，而它们出现了就说明模型没照格式写
  if (p.includes('`') || /^[#\-*>]/.test(p)) return false
  // **以句末标点结尾就是一句话，不是文件名。**
  //
  // ⚠️ 这里**不能要求「含空格」** —— 中文句子没有空格，实测
  //    `好的，我来写这个文件。` 就是这么漏过去的。正常文件名也不会以
  //    `.` `!` `?` `。` `！` `？` 结尾，所以直接按结尾判就够，
  //    而且对中英文都成立。末尾的冒号同理（`Path: foo.md` 这种写法）。
  if (/[.!?。！？:]$/.test(p)) return false
  return true
}

/**
 * 整个输出**就是一个** ``` 围栏时，剥掉它。
 *
 * 模型被要求「输出文件内容」时经常裹一层围栏，而把 ``` 写进文件是错的。
 *
 * ⚠️ **只在整段输出恰好是一个围栏时剥。** 不能去找「第一对 ```」，因为
 * 内容本身就可能含围栏（比如一份讲 markdown 的文档）—— 那样剥会把正文
 * 切掉一半，而且看起来像是模型写错了。
 */
function unwrapFence(s: string): string {
  const m = /^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$/.exec(s)
  return m?.[1] ?? s
}
