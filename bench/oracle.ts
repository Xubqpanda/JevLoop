/**
 * JevLoop · 判定标定台：判据机
 *
 * ══════════════════════════════════════════════════════════════
 *  **把一条轨迹变成「每个判定点对没对」。**
 * ══════════════════════════════════════════════════════════════
 *
 * 它消费 `runAgent({ onEvent })` 发出来的事件流 —— 这正是 `events.ts`
 * 模块注释里写的第四个消费者（「未来的标定台」）。所以这个台子
 * **不改内核一个字节**，只是把已经发出来的东西读一遍。
 *
 * ── 判据从哪来 ──────────────────────────────────────────────────
 *
 * 期望轨迹写在 `tasks.ts` 里（`required`）。判据机维护一份**剩余待办**：
 * 每次 `tool:call` 兑现掉一条，于是每个判定点在**当时那一刻**该做什么
 * 是确定的：
 *
 *     needsTool   还有待办 → use_tool；没有了 → answer
 *     pickTool    还有待办 → 选剩下那些工具里的一个；没有了 → done
 *     pickInput   选中文件必须在「这个工具还没兑现的待办」里
 *     gradeRisk   只读 → auto；可逆写 → auto_audit 或更严
 *     stepOk      夹具里的调用全都成功 → continue
 *     isDone      待办清空 → finish
 *     canDeliver  拿它当时看到的那份草稿去核对任务的 answerMust
 *
 * ── 三件事必须说清楚，否则数字会被读歪 ──────────────────────────
 *
 * 1. **`unjudged` 不是 `right`。** 判不了的（比如 `pickInput` 给一个
 *    没有输入可挑的工具挑输入）单独计，**不并进命中率** —— 并进去会让
 *    分母虚高、命中率虚好。
 *
 * 2. **报出来的概率是「模型给正确答案多少」，不是「它对自己走的这一支多确信」。**
 *
 *    这个区别是量出来的教训。第一版取的是 `max(p, 1-p)`（对自己那一支的
 *    确信度），于是 `isDone` 的对与错**两组都报 0.8–0.9** —— 看起来像
 *    「分不开」，其实是这个数根本不可比：一个自信的「是」和一个自信的
 *    「否」算出来一样大，而它们对着的是相反的两个答案。
 *
 *    README 里记 Laya 的那句话是「**the wrong answer scored higher than the
 *    right one**」—— 那比的是「正确答案拿到多少概率」。指标必须和结论
 *    对得上，否则量出来的东西答不了想问的问题。
 *
 * 3. **`choice` 的置信度不用 `confidence`。** Laya 的 `confidence` 是
 *    **归一化香农熵**，不是选中项的概率（§8.3）—— 同一个值在 2 个选项和
 *    20 个选项下含义完全不同。这里取 `probabilities[choice]`，也就是
 *    `topGte()` 看的那个数。
 *
 * @module JevLoop/oracle
 */

import type { AgentEvent } from '../src/events.ts'
import type { Answer } from '../src/vocab.ts'
import { TOOLS, isToolName } from '../src/tools.ts'
import type { BenchTask, ExpectedCall } from './tasks.ts'

export type Verdict = 'right' | 'wrong' | 'unjudged'

export interface Judgement {
  /** 判定点 id，如 `loop.pickTool` */
  node: string
  verdict: Verdict
  /** 它实际走的动作 */
  action: string
  /** 走这个分支的置信度（见模块头第 2、3 条） */
  prob: number
  /** 一句话说清判据 —— 判错时要能直接看出错在哪 */
  why: string
}

/**
 * 模型给一个 `noul` 取指定值的概率。
 *
 * `wantTrue` 是**判据期望的答案**，不是模型给的那个 —— 这个区别就是
 * 模块头第 2 条说的那件事。
 */
function pNoul(answers: Record<string, Answer>, id: string, wantTrue: boolean): number {
  const a = answers[id]
  if (!a || a.type !== 'noul') return 0
  return wantTrue ? a.noul : 1 - a.noul
}

/**
 * 模型给一个 `choice` 选指定选项的概率。
 *
 * ★ 取 `probabilities[want]`，**不是** `confidence` —— 后者是归一化香农熵，
 *   同一个值在 2 个选项和 20 个选项下含义完全不同（§8.3）。
 */
function pChoice(answers: Record<string, Answer>, id: string, want: string): number {
  const a = answers[id]
  if (!a || a.type !== 'choice') return 0
  return a.probabilities?.[want] ?? 0
}

/** 从 `state: unknown` 里安全地取几个字段。帧是**不可信输入**，要按名字查 */
function frame<T extends Record<string, unknown>>(state: unknown, keys: (keyof T)[]): Partial<T> {
  const src = (state ?? {}) as Record<string, unknown>
  const out: Record<string, unknown> = {}
  for (const k of keys) if (src[k as string] !== undefined) out[k as string] = src[k as string]
  return out as Partial<T>
}

/**
 * 一条任务的判据机。**有状态** —— 它要跟着轨迹推进「还剩什么没做」。
 *
 * 用完即弃：一条任务一个实例。跨任务复用会让剩余待办串味。
 */
export class Oracle {
  readonly #task: BenchTask
  #remaining: ExpectedCall[]
  /** 最近一次工具调用是否拿到内容。`stepOk` 的判据要用它 */
  #lastOutputOk = true

  constructor(task: BenchTask) {
    this.#task = task
    // 复制一份：下面会 splice，不能改到 TASKS 里的那份
    this.#remaining = task.required.map((c) => ({ ...c }))
  }

  /** 还剩几条没兑现。`isDone` 和 `needsTool` 的判据都看它 */
  get remaining(): readonly ExpectedCall[] {
    return this.#remaining
  }

  /**
   * 喂一个事件。
   *
   * @returns 是 `decision` 就给一份判定；其它事件只推进状态，返回 `null`
   */
  feed(e: AgentEvent): Judgement | null {
    if (e.type === 'tool:call') {
      this.#advance(e.tool, e.input)
      return null
    }
    if (e.type === 'tool:result') {
      // 夹具里的调用都该成功。这里不猜「成功」的语义，只看有没有内容 ——
      // 空结果按失败算，因为 `stepOk` 的判据里「空结果」就是失败的一种。
      this.#lastOutputOk = String(e.output ?? '').trim().length > 0
      return null
    }
    if (e.type !== 'decision') return null
    const j = this.#judge(e)
    // 「判不了」的那几支把 want 设成了实际动作，会自己判成 right ——
    // 在这里翻回 unjudged，**不能让它冒充判对**（模块头第 1 条）
    if (j.why.includes('判不了') || j.why.includes('认不出来') || j.why.includes('没有需要挑输入') || j.why.includes('还没有')) {
      return { ...j, verdict: 'unjudged' }
    }
    return j
  }

  /** 兑现掉一条待办。找不到就什么都不做 —— 多做的那次由对应判定点自己判错 */
  #advance(tool: string, input: string): void {
    // ★ `write_file` 的输入是 `路径\n内容`，而夹具里写的是目标路径 ——
    //   所以按**第一行**比。写路径的名字现在是**生成**的（`write-content.ts`），
    //   夹具说不出完整输入，但它说得出「该写哪个文件」。
    const i = this.#remaining.findIndex(
      (c) =>
        c.tool === tool &&
        (c.input === undefined || c.input === input || input.startsWith(`${c.input}\n`)),
    )
    if (i >= 0) this.#remaining.splice(i, 1)
  }

  /**
   * 判一次决策。
   *
   * 每个分支算出两样：**期望的动作**（判对错）和**模型给那个期望的概率**
   * （判它是不是「自信地错」）。两样都由判据决定，不取模型自己走的哪一支。
   */
  #judge(e: Extract<AgentEvent, { type: 'decision' }>): Judgement {
    const v = this.#expected(e)
    return {
      node: e.id,
      action: e.action,
      prob: v.prob,
      verdict: e.action === v.want ? 'right' : 'wrong',
      why: v.why,
    }
  }

  /** 期望的动作 + 模型给它的概率 + 一句话判据 */
  #expected(e: Extract<AgentEvent, { type: 'decision' }>): { want: string; prob: number; why: string } {
    const A = e.answers
    const n = this.#remaining.length

    switch (e.id) {
      case 'loop.needsTool': {
        const want = n > 0 ? 'use_tool' : 'answer'
        return { want, prob: pNoul(A, 'needs_tool', n > 0), why: `还剩 ${n} 条待办 → 该 ${want}` }
      }

      case 'loop.pickTool': {
        // 待办清空之后，唯一正确的下一步是收工
        if (n === 0) {
          const tool = pickedTool(e)
          if (e.action === 'call' && tool === 'done') {
            return { want: 'call', prob: pChoice(A, 'tool', 'done'), why: '待办已清空 → 该 done' }
          }
          return {
            want: 'call',
            prob: pChoice(A, 'tool', 'done'),
            why: `待办已清空 → 该 done，走了 ${tool ?? e.action}`,
          }
        }
        const tool = pickedTool(e)
        const pool = this.#remaining.map((c) => c.tool)
        const allowed = this.#task.allowedTools ?? []
        const ok = tool !== undefined && (pool.includes(tool) || allowed.includes(tool))
        // 概率取**池子里所有可接受选项的概率之和** —— 「选对一个」这件事
        // 有多个正确答案时，模型的把握是把它们加起来的把握
        const prob = [...new Set([...pool, ...allowed])].reduce((acc, t) => acc + pChoice(A, 'tool', t), 0)
        return {
          want: 'call',
          prob,
          why: ok ? `${tool} 在待办或允许集里` : `该从 ${pool.join(' / ')} 里选，选了 ${tool ?? e.action}`,
        }
      }

      case 'loop.pickInput': {
        const t = frame<{ tool: string }>(e.state, ['tool']).tool ?? ''
        const todo = this.#remaining.filter((c) => c.tool === t && c.input !== undefined)
        if (todo.length === 0) {
          // 判不了，**不并进命中率**（模块头第 1 条）
          return { want: e.action, prob: 0, why: `${t} 没有需要挑输入的待办` }
        }
        const file = pickedFile(e)
        const prob = todo.reduce((acc, c) => acc + pChoice(A, 'file', c.input!), 0)
        const ok = file !== undefined && todo.some((c) => c.input === file)
        return {
          want: 'use',
          prob,
          why: ok ? `挑中了待办里的 ${file}` : `该挑 ${todo.map((c) => c.input).join(' / ')}，挑了 ${file ?? e.action}`,
        }
      }

      case 'loop.gradeRisk': {
        const t = frame<{ tool: string }>(e.state, ['tool']).tool ?? ''
        if (!isToolName(t)) {
          return { want: e.action, prob: 0, why: `工具名 ${t} 认不出来，没有风险基线可比` }
        }
        // 只读 → auto；可逆写及以上 → auto_audit 或更严（ask_human 也算对：
        // 多问一句不是错，错的是**该问却没问**）
        const readOnly = TOOLS[t].baseRisk === 0
        const want = readOnly ? 'auto' : 'auto_audit'
        // 概率用 `needs_auth` 当代理：要授权 ⇔ 不该默默放行。
        // `score` 的 `probabilities` 键由后端原样透传，没法可靠索引到档位。
        return {
          want,
          prob: pNoul(A, 'needs_auth', !readOnly),
          why: `${t} baseRisk=${TOOLS[t].baseRisk} → 该 ${readOnly ? 'auto' : 'auto_audit / ask_human'}`,
        }
      }

      case 'loop.stepOk': {
        // 夹具里的调用全都成功。**否定分支在这个夹具下到不了** ——
        // 报告的「没能测到什么」一节会写明这条。
        const ok = this.#lastOutputOk
        return { want: ok ? 'continue' : 'stop', prob: pNoul(A, 'ok', ok), why: `上一次调用${ok ? '有内容' : '是空的'}` }
      }

      case 'loop.isDone': {
        const done = n === 0
        return { want: done ? 'finish' : 'keep_going', prob: pNoul(A, 'done', done), why: `还剩 ${n} 条待办` }
      }

      case 'loop.canDeliver': {
        // ★ 拿它**当时看到的那份草稿**去核对 —— 不是最终回答。
        //   修订会再来一次，两次看到的草稿不同，两次都得各判各的。
        const draft = frame<{ answer: string }>(e.state, ['answer']).answer ?? ''
        const good = answerOk(draft, this.#task)
        // 「能交付」要求两件事同时成立：够完整，且没有证据不支持的话。
        // 所以把握是两者的较小值 —— 最弱的一环决定这次能不能放行。
        const prob = Math.min(pNoul(A, 'deliverable', true), pNoul(A, 'unsupported', false))
        return good
          ? { want: 'deliver', prob, why: '草稿合格' }
          : {
              want: 'revise',
              // 该修订时，把握是「它看到了问题」—— 也就是不该交付的那一面
              prob: Math.min(pNoul(A, 'deliverable', false), pNoul(A, 'unsupported', true)),
              why: `草稿不合格（${missing(draft, this.#task).join('、') || '含不该有的内容'}）`,
            }
      }

      default:
        // 新加的判定点没登记判据时**不猜**，报出来
        return { want: e.action, prob: 0, why: `台子还没有 ${e.id} 的判据` }
    }
  }
}

/** 判定模型给的工具名。**不可信输入** —— 只用来判分，不执行 */
function pickedTool(e: Extract<AgentEvent, { type: 'decision' }>): string | undefined {
  const a = e.answers['tool']
  return a && a.type === 'choice' ? a.choice : undefined
}

function pickedFile(e: Extract<AgentEvent, { type: 'decision' }>): string | undefined {
  const a = e.answers['file']
  return a && a.type === 'choice' ? a.choice : undefined
}

/** 这份草稿过不过任务的正误检查。`canDeliver` 和任务级验收共用同一套判据 */
export function answerOk(text: string, task: BenchTask): boolean {
  return (
    task.answerMust.every((re) => re.test(text)) &&
    (task.answerMustNot ?? []).every((re) => !re.test(text))
  )
}

/** 缺了什么、多了什么 —— 判错时要能直接看出原因 */
export function missing(text: string, task: BenchTask): string[] {
  const miss = task.answerMust.filter((re) => !re.test(text)).map((re) => `缺 ${re}`)
  const extra = (task.answerMustNot ?? []).filter((re) => re.test(text)).map((re) => `多 ${re}`)
  return [...miss, ...extra]
}
