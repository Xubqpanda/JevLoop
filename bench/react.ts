/**
 * JevLoop · 对照用的 ReAct 循环
 *
 * ══════════════════════════════════════════════════════════════
 *  **这个文件不是框架的一部分，是一把尺子。**
 *
 *  它实现同一个 agent 任务的**另一种循环形状**：每个岔路口都问一次大模型
 *  （经典 ReAct —— 想一步、调一个工具、看结果、再想）。
 *  JevLoop 问的是判定模型，只在大模型那一步付生成的钱。
 *
 *  两个循环放在一起跑**同一个任务、同一批工具、同一个模型**，
 *  才能把「14:1」这句话说准 —— 那个比值是**判定次数 : 生成次数**，
 *  不是「省了 14 次大模型调用」。同样的任务，ReAct 大概只要 3–4 次。
 *  两者不是一个量纲的东西，混着说就成了口号。
 * ══════════════════════════════════════════════════════════════
 *
 * ── 公平性上做了和没做的事（这些比数字本身重要）──────────────
 *
 * **做了**：
 *   · 同一个 `generator`（同模型、同 key、同 baseUrl）
 *   · 同一批工具，而且**是同一份实现**（`src/tools.ts` 的 `callTool`）
 *   · 同一个工作目录夹具、同一个 `maxSteps`
 *   · 同一个验收判据（`bench/oracle.ts` 的 `answerOk` / `checkArtifacts`）
 *
 * **没做，而这是有意的**：JevLoop 那条路多两道东西 ReAct 没有 ——
 * 风险分级（`gradeRisk`，判定，不花钱）和交付闸门（`canDeliver`，
 * 过不了还要**再生成一次**）。闸门是实打实的成本，所以对比里 JevLoop
 * 那一边通常更贵。**把便宜的那边装成贵的没有意义。**
 *
 * ── 为什么它不是一个稻草人 ──────────────────────────────────────
 *
 * 稻草人是「让对手用笨办法」。这里防了三种做法：
 *   · **不给它格式修理**：模型吐出 ```json 围栏或前后带话，照抽不误；
 *     抽不出来就把错误当观察喂回去让它重说（**那次也计入调用次数**）。
 *   · **不给它更少的上下文**：观察结果原样喂回去，和 JevLoop 的证据一样。
 *   · **不给它更少的步数**：同一个 `maxSteps`。
 *
 * @module JevLoop/react
 */

import type { Generator } from '../src/llm.ts'
import { callTool, isToolName, TOOLS } from '../src/tools.ts'

/** 一次工具调用 */
export interface ReactCall {
  tool: string
  input: string
}

export interface ReactRun {
  /** 最后给出的回答（`answer` 动作的内容） */
  answer: string
  calls: ReactCall[]
  /** **大模型调用次数** —— 这条对比的主数字 */
  modelCalls: number
  inputTokens: number
  outputTokens: number
  /** 这些调用一共花了多少墙钟 */
  modelMs: number
  /** 整轮墙钟（含工具执行） */
  latencyMs: number
  /** 怎么停的：`answered` / `max_steps` / `unparsable` */
  stop: string
  /** 出错时的原话（跑不起来才有） */
  failed?: string
}

const SYSTEM = [
  'You are an agent working in a directory. Do the task and then report what you did.',
  '',
  'Reply with EXACTLY ONE JSON object and nothing else. Four forms:',
  '  {"action":"list_dir","input":""}',
  '  {"action":"read_file","input":"notes.md"}',
  '  {"action":"write_file","input":"out.md\\n<the whole file content>"}',
  '  {"action":"answer","input":"<what you did, as prose>"}',
  '',
  `Available actions: ${Object.keys(TOOLS).join(', ')}.`,
  `  list_dir   ${TOOLS.list_dir.description}`,
  `  read_file  ${TOOLS.read_file.description}`,
  `  write_file ${TOOLS.write_file.description}`,
  '',
  'Do not use the "done" action: when the task is finished, reply with the "answer" action.',
].join('\n')

/**
 * 从一段模型输出里抽出那个 JSON 动作。
 *
 * ★ **宽容是刻意的。** 真实模型经常把 JSON 包在 ``` 围栏里、或者前后带一句
 *   「好的，我来读这个文件」。一个只会 `JSON.parse(trim())` 的实现会把那些
 *   全部算成失败，于是对比出来的数字说的不是「循环形状的差别」，而是
 *   「谁的解析器更脆」。抽不出来时**不猜**，返回 null 由调用方喂回去让它重说。
 */
export function parseAction(text: string): { action: string; input: string } | null {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(text)
  const body = fenced?.[1]?.trim() ?? text.trim()

  // 从第一个 `{` 到最后一个 `}` —— 模型在 JSON 前后带话时这是最稳的一刀
  const start = body.indexOf('{')
  const end = body.lastIndexOf('}')
  if (start < 0 || end <= start) return null

  try {
    const o = JSON.parse(body.slice(start, end + 1))
    if (typeof o?.action !== 'string') return null
    return { action: o.action, input: typeof o.input === 'string' ? o.input : '' }
  } catch {
    // 吞的是：**这段文本里没有一个完整的 JSON 对象**。这是正常的模型输出，
    // 调用方会把它当一次「格式错误」喂回去重说，而不是当成崩溃。
    return null
  }
}

/**
 * 跑一轮 ReAct。
 *
 * @param opts.maxSteps 和 JevLoop 那边用**同一个数**，否则数字不可比
 */
export async function runReact(opts: {
  task: string
  cwd: string
  generator: Generator
  maxSteps: number
}): Promise<ReactRun> {
  const t0 = performance.now()
  const calls: ReactCall[] = []
  let modelCalls = 0
  let inputTokens = 0
  let outputTokens = 0
  let modelMs = 0
  let answer = ''
  let stop = 'max_steps'

  /** 喂给模型的东西：任务 + 每一步的观察。和 JevLoop 的证据同源 */
  const transcript: string[] = [`Task:\n${opts.task}`]

  /**
   * **一次生成**：计数、计时、把文本拿回来。
   *
   * 单独抽出来是因为它每次都必须在三个计数器上留痕 —— 漏一处，
   * 对比出来的调用次数就是错的，而错的那个数字恰恰是结论本身。
   */
  const ask = async (): Promise<string> => {
    const r = await opts.generator.generate({
      task: opts.task,
      evidence: transcript.join('\n\n'),
      // ★ 协议必须**真的发给它** —— 第一版定义了 `SYSTEM` 却忘了传，
      //   于是模型完全不知道要回 JSON，整个对照会变成一个稻草人。
      system: SYSTEM,
    })
    modelCalls += 1
    inputTokens += r.inputTokens
    outputTokens += r.outputTokens
    modelMs += r.latencyMs
    return r.text
  }

  try {
    for (let step = 0; step < opts.maxSteps; step++) {
      let text = await ask()
      let act = parseAction(text)

      // 格式不对就**把错误当观察喂回去让它重说** —— 这一次也算一次调用。
      // 悄悄替它修好会让 ReAct 那一边的调用次数虚低，而那个次数是结论本身。
      if (!act) {
        transcript.push(`Your last reply was not a JSON action. Reply with exactly one JSON object.`)
        text = await ask()
        act = parseAction(text)
        if (!act) {
          stop = 'unparsable'
          break
        }
      }

      if (act.action === 'answer' || act.action === 'done') {
        answer = act.input.trim()
        stop = 'answered'
        break
      }

      if (!isToolName(act.action)) {
        transcript.push(`Observation: there is no action called '${act.action}'. Available: ${Object.keys(TOOLS).join(', ')}`)
        continue
      }

      calls.push({ tool: act.action, input: act.input })
      const observation = await callTool(act.action, act.input, opts.cwd)
      transcript.push(`Action: ${act.action}(${act.input.split('\n')[0]})\nObservation: ${observation}`)
    }
  } catch (err) {
    // 空 catch 说明：吞的是**这一轮跑不起来**（网络、后端），把它变成一条
    // 失败的记录而不是让整轮对比挂掉 —— 网络失败不是循环形状的差别。
    return {
      answer,
      calls,
      modelCalls,
      inputTokens,
      outputTokens,
      modelMs,
      latencyMs: performance.now() - t0,
      stop: 'failed',
      failed: (err as Error).message,
    }
  }

  return {
    answer,
    calls,
    modelCalls,
    inputTokens,
    outputTokens,
    modelMs,
    latencyMs: performance.now() - t0,
    stop,
  }
}

/**
 * 这个循环用的 system prompt。
 *
 * 导出只有一个用途：**让台子把对手拿到的东西原样印出来**。
 * 「对比」如果说不清对手拿到的指令，读的人就只能信我 —— 那不算证据。
 */
export const REACT_SYSTEM = SYSTEM
