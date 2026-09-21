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
 * 2. **置信度取的是「它实际走的那个分支」的概率，而且取所有答案里最小的那个。**
 *    一个决策可能同时看几个答案（`canDeliver` 就看两个），动作由它们
 *    共同决定，所以最弱的那一环才是诚实的概括。
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

/** 一个答案「实际走的那个分支」的概率 */
function branchProb(a: Answer): number {
  if (a.type === 'noul') return Math.max(a.noul, 1 - a.noul)
  // choice：选中项的概率。**不是** `confidence` —— 那个是熵（§8.3）
  if (a.type === 'choice') return a.probabilities?.[a.choice] ?? 0
  // score：没有「选中的那一项」，取分布里最高的一档
  return Math.max(0, ...Object.values(a.probabilities ?? { 0: 0 }))
}

/** 这次判定所有答案里**最弱**的那一环 */
function weakestProb(answers: Record<string, Answer>): number {
  const vals = Object.values(answers).map(branchProb)
  return vals.length ? Math.min(...vals) : 0
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
    return this.#judge(e)
  }

  /** 兑现掉一条待办。找不到就什么都不做 —— 多做的那次由对应判定点自己判错 */
  #advance(tool: string, input: string): void {
    const i = this.#remaining.findIndex(
      (c) => c.tool === tool && (c.input === undefined || c.input === input),
    )
    if (i >= 0) this.#remaining.splice(i, 1)
  }

  #judge(e: Extract<AgentEvent, { type: 'decision' }>): Judgement {
    const base = { node: e.id, action: e.action, prob: weakestProb(e.answers) }
    const verdict = (v: Verdict, why: string): Judgement => ({ ...base, verdict: v, why })

    switch (e.id) {
      case 'loop.needsTool': {
        const want = this.#remaining.length > 0 ? 'use_tool' : 'answer'
        return e.action === want
          ? verdict('right', `还有 ${this.#remaining.length} 条待办 → 该 ${want}`)
          : verdict('wrong', `还有 ${this.#remaining.length} 条待办 → 该 ${want}，走了 ${e.action}`)
      }

      case 'loop.pickTool': {
        // 待办清空之后，唯一正确的下一步是收工
        if (this.#remaining.length === 0) {
          return e.action === 'call' && pickedTool(e) === 'done'
            ? verdict('right', '待办已清空 → 该 done')
            : verdict('wrong', `待办已清空 → 该 done，走了 ${pickedTool(e) ?? e.action}`)
        }
        const tool = pickedTool(e)
        const want = this.#remaining.map((c) => c.tool)
        const ok = tool !== undefined && (want.includes(tool) || (this.#task.allowedTools ?? []).includes(tool))
        return ok
          ? verdict('right', `${tool} 在待办或允许集里`)
          : verdict('wrong', `该从 ${want.join(' / ')} 里选，选了 ${tool ?? e.action}`)
      }

      case 'loop.pickInput': {
        const t = frame<{ tool: string }>(e.state, ['tool']).tool ?? ''
        // 这个工具还有没有「要挑哪个」的待办
        const todo = this.#remaining.filter((c) => c.tool === t && c.input !== undefined)
        if (todo.length === 0) {
          // 判不了，**不并进命中率**（模块头第 1 条）
          return verdict('unjudged', `${t} 没有需要挑输入的待办`)
        }
        const file = pickedFile(e)
        const ok = file !== undefined && todo.some((c) => c.input === file)
        return ok
          ? verdict('right', `挑中了待办里的 ${file}`)
          : verdict('wrong', `该挑 ${todo.map((c) => c.input).join(' / ')}，挑了 ${file ?? e.action}`)
      }

      case 'loop.gradeRisk': {
        const t = frame<{ tool: string }>(e.state, ['tool']).tool ?? ''
        if (!isToolName(t)) return verdict('unjudged', `工具名 ${t} 认不出来，没有风险基线可比`)
        const base = TOOLS[t].baseRisk
        // 只读 → auto；可逆写及以上 → auto_audit 或更严（ask_human 也算对：
        // 多问一句不是错，错的是**该问却没问**）
        const ok = base === 0 ? e.action === 'auto' : e.action === 'auto_audit' || e.action === 'ask_human'
        const want = base === 0 ? 'auto' : 'auto_audit / ask_human'
        return ok
          ? verdict('right', `${t} baseRisk=${base} → ${want}`)
          : verdict('wrong', `${t} baseRisk=${base} → 该 ${want}，走了 ${e.action}`)
      }

      case 'loop.stepOk': {
        // 夹具里的调用全都成功。**否定分支在这个夹具下到不了** ——
        // 报告的「没能测到什么」一节会写明这条。
        const want = this.#lastOutputOk ? 'continue' : 'stop'
        return e.action === want
          ? verdict('right', `上一次调用${this.#lastOutputOk ? '有内容' : '是空的'} → ${want}`)
          : verdict('wrong', `上一次调用${this.#lastOutputOk ? '有内容' : '是空的'} → 该 ${want}`)
      }

      case 'loop.isDone': {
        const want = this.#remaining.length === 0 ? 'finish' : 'keep_going'
        return e.action === want
          ? verdict('right', `还剩 ${this.#remaining.length} 条待办 → ${want}`)
          : verdict('wrong', `还剩 ${this.#remaining.length} 条待办 → 该 ${want}，走了 ${e.action}`)
      }

      case 'loop.canDeliver': {
        // ★ 拿它**当时看到的那份草稿**去核对 —— 不是最终回答。
        //   修订会再来一次，两次看到的草稿不同，两次都得各判各的。
        const draft = frame<{ answer: string }>(e.state, ['answer']).answer ?? ''
        const good = answerOk(draft, this.#task)
        const want = good ? 'deliver' : 'revise'
        return e.action === want
          ? verdict('right', `草稿${good ? '合格' : '不合格'} → ${want}`)
          : verdict(
              'wrong',
              `草稿${good ? '合格' : '不合格'} → 该 ${want}，走了 ${e.action}` +
                (good ? '' : `（${missing(draft, this.#task).join('、') || '含不该有的内容'}）`),
            )
      }

      default:
        // 新加的判定点没登记判据时**不猜**，报出来
        return verdict('unjudged', `台子还没有 ${e.id} 的判据`)
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
