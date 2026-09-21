/**
 * JevLoop · 给 `write_file` 造内容
 *
 * ══════════════════════════════════════════════════════════════
 *  **「写什么」是生成，按三分法不属于判定模型。**
 * ══════════════════════════════════════════════════════════════
 *
 * 判定只负责挑「写哪个文件」（`loop.pickInput`，一个 choice），内容由这里
 * 调生成器产出（`docs/CODE-STYLE.md` §8.1）。
 *
 * ── 这个文件是补一个实测出来的窟窿 ──────────────────────────────
 *
 * `write_file` 在 `frame.ts` 里有**两道门**，第一道是「调用方必须提供内容
 * 来源」（`ctx.canWrite`）。设计是对的：**没有来源时它不该出现在候选里** ——
 * 出现了模型就会选，而 loop 拿不出内容。
 *
 * 问题是 `src/server.ts` / `src/cli.ts` / `examples/` **谁都没传那个来源**，
 * 于是通过网页和 CLI，`write_file` **永远不进候选**。实测（2026-09-21）：
 *
 *     任务：把 invoice.ts 和 retry.ts 里各有哪些函数写进新建的 SUMMARY.md
 *     第 18 步 pickTool 的候选只有：read_file、done
 *     它选了 read_file（0.58，低于 0.6 门限）→ escalate → halt: tool_unclear
 *
 * **判定模型的行为是对的** —— 它在给定的两个选项里做了最合理的选择。
 * 缺的是那个选项本身。而唯一的信号是一个看起来像「判定不确定」的停机，
 * 所以这个窟窿一直没人发现（bench 的 artifact 检查报了 0/3，那才发现）。
 *
 * ── 三层各自负责什么 ────────────────────────────────────────────
 *
 *     挑哪个文件      判定模型（choice）
 *     内容是什么      生成器（这个文件）
 *     能不能写        代码（`tools.ts` 的 `safePath`、`canWrite` 那道门）
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
 * 造内容时拿得到的上下文。**是 `AgentCtx` 的一个结构子集** ——
 * 这样 `provideWriteContent` 能直接收 `AgentCtx`，不需要任何转换或断言。
 */
export interface WriteContext {
  task: string
  /** 已经做过的工具调用。`result` 里就是**读到的文件内容**，也就是证据 */
  history?: readonly WriteStep[]
}

/**
 * `write_file` 的内容来源。签名**结构上兼容** `AgentOptions.provideWriteContent`。
 *
 * 返回 `undefined` 表示「这次写不了」→ loop 停机，而不是写一句假内容交差。
 */
export type WriteContent = (file: string, ctx: WriteContext) => Promise<string | undefined>

/** 一次工具调用的输入截多长。和 `agent.ts` 里那份**同一个数**，理由也一样：定长前缀 */
const STEP_INPUT_CHARS = 120

/**
 * 模型说「素材不够」时回的那个词。
 *
 * ★ 用一个**显式的哨兵**而不是「输出为空」：空输出分不清「它判断写不了」
 *   和「它挂了 / 被截断了」，而这两件事的处理完全不同 —— 前者该停机，
 *   后者该报错。哨兵至少是它**主动说**的。
 */
const NOTHING = 'NOTHING'

const INSTRUCTION = `Write the complete contents of the file named below.

- Output **only** what goes in the file: no preamble, no explanation, and no code fence around the whole thing.
- Use only the evidence above. Do not invent identifiers, filenames or behaviour that the evidence does not show.
- If the evidence does not contain what this file needs, reply with exactly ${NOTHING} and nothing else.

The file is: `

/**
 * 一次写内容的生成，发生后交给调用方。
 *
 * ★ **为什么要有这个回调**：写内容和主回答一样是**一次大模型调用**，而
 *   它原本**不发任何事件** —— 轨迹里看不见，界面的「判定 : 模型」计数也
 *   少算一次。实测（2026-09-21）：一次运行的轨迹里只有 2 个 `generate`
 *   事件，而两次都是主回答（首次 + 修订）；写内容那一次是隐形的。
 *
 *   这个项目的全部主张就是那个比例，**漏计一次模型调用会让它是错的**。
 */
export interface WriteGeneration {
  result: GenerateResult
  /** 我们估的「能控制的那部分」输入 token。和主生成路径同一个算法 */
  estimatedInputTokens: number
}

/**
 * 用生成器造 `write_file` 的内容。
 *
 * @param generator 生成器
 * @param system system prompt。传 `DECISION.md` 的 generator 段 ——
 *   里面那两条承重规则（只用证据、用任务的语言回答）对写文件同样成立。
 * @param onGenerate 生成发生后调一次，让调用方把它记进轨迹（见 `WriteGeneration`）
 */
export function writeContentVia(
  generator: Generator,
  system?: string,
  onGenerate?: (g: WriteGeneration) => void,
): WriteContent {
  return async (file, ctx) => {
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
      instruction: INSTRUCTION + file,
    }

    const out = await generator.generate(req)
    // 先报出去再判内容 —— **判成「写不了」也是一次真的调用**，也要计数
    onGenerate?.({
      result: out,
      estimatedInputTokens: priceGenerateRequest({ task: req.task, evidence }).controlTokens,
    })
    const content = unwrapFence(out.text).trim()

    // 它主动说素材不够 —— 停机，不要把那句话本身当成文件内容写下去
    if (content === NOTHING) return undefined
    // 空输出同样写不了。空文件和「没写」在盘上分不出来，而调用方
    // 以为发生了前者
    if (content.length === 0) return undefined

    // ★ 去掉首尾空白之后**补回恰好一个换行**。
    //
    //   模型常常在前后各留一个空行，直接写进去就是文件开头一个空行、
    //   结尾两个换行 —— 实测第一次跑就撞上了：`# 生成的内容` 后面没有
    //   换行，因为 `trim()` 把它吃掉了，而 `write_file` 是**原样**写的。
    //   盘上的文本文件该以恰好一个换行结尾，这是这个仓库自己的规矩
    //   （`docs/CODE-STYLE.md` §1），也是绝大多数工具期望的。
    return `${content}\n`
  }
}

/**
 * 整个输出**就是一个** ``` 围栏时，剥掉它。
 *
 * 模型被要求「输出文件内容」时经常裹一层围栏，而把 ``` 写进 `.md` 是错的。
 *
 * ⚠️ **只在整段输出恰好是一个围栏时剥。** 不能去找「第一对 ```」，因为
 * 文件内容本身就可能含围栏（比如一份讲 markdown 的文档）—— 那样剥会把
 * 正文切掉一半，而且看起来像是模型写错了。
 */
function unwrapFence(s: string): string {
  const m = /^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$/.exec(s)
  return m?.[1] ?? s
}
