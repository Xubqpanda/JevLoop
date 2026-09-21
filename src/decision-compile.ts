/**
 * JevLoop · DECISION.md 的编译
 *
 * `DocBlock` → 问题与策略。**从 `decisiondoc.ts` 切出来** —— 那个文件曾同时做
 * 「解析」与「编译」两件事（docs/CODE-STYLE.md §12 的判据：一句话说不完）。
 *
 * 两半的依赖面**可测量地不同**：解析段只依赖 `vocab.ts`（L0），
 * 编译段还要 `policy.ts`（L1）—— 于是「只依赖词汇的解析器」被一起钉在 L3，
 * 更低层没法复用它。层号按**拆出来那半的依赖面有多小**定：本文件 L2。
 *
 * ⚠️ **`decisiondoc.ts` 不 re-export 本文件**：它拆完是 L0、本文件是 L2，
 * L0 import L2 违反 §11 的分层。消费方直接从本文件 import。
 *
 * @module JevLoop/decision-compile
 */

import { choice, noul, score, type AnswerSet, type Question, type QuestionSet } from './vocab.ts'
import type { PolicyRule } from './vocab-decision.ts'
import { picked, probGte, probLt, scoreGte, topGte, topLt } from './policy.ts'
import { splitGates, type GateOverrides } from './gates.ts'
import { assertNever } from './util.ts'
import type { DocBlock, DocQuestion } from './decisiondoc.ts'

// ═══════════════════════════════════════════════════════════
// 编译：块 → 问题
// ═══════════════════════════════════════════════════════════

/**
 * 块 → `QuestionSet`。
 *
 * `rule` 返回 `undefined`：它由代码答，没有要问模型的东西。硬编一个
 * 空问题集会让「这一步不过模型」这件事在类型上消失。
 */
export function compileQuestions(block: DocBlock): QuestionSet | undefined {
  if (block.kind === 'rule' || block.questions.length === 0) return undefined

  const out: QuestionSet = {}
  for (const q of block.questions) {
    out[q.id] = toQuestion(q)
  }
  return out
}

function toQuestion(q: DocQuestion): Question {
  const criteria: Record<string, string> = {}
  for (const o of q.options) criteria[o.name] = o.criteria

  switch (q.type) {
    case 'noul': {
      const t = criteria['true']
      const f = criteria['false']
      // 解析器已经保证 noul 有两个名为 true/false 的选项，这里只是让类型收窄
      return t !== undefined && f !== undefined
        ? noul(q.ask, { true: t, false: f })
        : noul(q.ask)
    }
    case 'choice':
      return choice(q.ask, criteria)
    case 'score':
      return score(
        q.ask,
        q.options.map((o) => o.criteria),
      )
    default:
      return assertNever(q.type)
  }
}

// ═══════════════════════════════════════════════════════════
// 编译：谓词 → PolicyRule
//
// 封闭词汇表。**刻意不支持 `>` 和 `<=`** —— policy.ts 只提供了
// topGte / probGte / probLt / scoreGte / picked，多一个操作符就意味着
// 内核要跟着长。写不出来的条件，就是该写进代码的信号。
// ═══════════════════════════════════════════════════════════

const RE_TOP = /^top\s*(>=|<)\s*([0-9]*\.?[0-9]+)$/
const RE_PROB = /^prob:([A-Za-z_][\w.-]*)\s*(>=|<)\s*([0-9]*\.?[0-9]+)$/
const RE_SCORE = /^score:([A-Za-z_][\w.-]*)\s*>=\s*([0-9]*\.?[0-9]+)$/
const RE_PICKED = /^picked:([A-Za-z_][\w.-]*)\s*=\s*(\S+)$/

/** 兜底规则的写法。`_` 是最省事的那个，收在这里免得散落各处 */
const ELSE_FORMS = new Set(['else', '_', '否则', '其余'])

/** 编译出来的谓词 */
export interface CompiledPredicate {
  fn: (a: AnswerSet) => boolean
  /**
   * **生效的**谓词原文 —— 有覆盖时门限已经被换成覆盖值。
   *
   * ★ 它进规则的 `reason`（见 `compilePolicy`），所以日志里写的永远是
   *   **真正用过的那个数**。留着 `when` 原文的话，一条跑在 0.7 上的运行
   *   会在日志里写着 `>= 0.5` —— 那是假装（§8.10），而且是最难查的那种：
   *   数字看起来很正常，只是不是这次的。
   */
  text: string
}

/**
 * 谓词原文 → 判定函数。
 *
 * `top` 指的是**本块唯一那个问题**，所以单问题块不必重复写问题 id。
 * 多问题块用 `top` 是歧义的，会被拒绝 —— 必须点名 `prob:` / `score:`。
 *
 * @param thresholds 这个块的**门限覆盖**：`问题 id → 新门限`。只认本块的
 *   问题，所以调用方要先用 `splitGates` 按块切开。没给 = 全用文件里的默认值。
 * @returns 判定函数 + 生效原文；原文不认识时返回 `null`（调用方应记为 problem，不要当兜底）
 */
export function compilePredicate(
  when: string,
  block: DocBlock,
  /**
   * **这一条规则的**门限覆盖值（已经解析好了是哪一个键）。
   *
   * 收一个数而不是一张表：哪条规则该用哪个键是**按块、按顺序**算出来的，
   * 那个知识在 `compilePolicy` 里 —— 在这里重算一遍就是第二份实现。
   */
  override?: number,
): CompiledPredicate | null {
  const src = when.trim()
  if (ELSE_FORMS.has(src)) return { fn: () => true, text: src }

  /**
   * 用覆盖值替掉原文里的门限。
   *
   * `which === null` 表示这条谓词不带门限（`picked:`），这时**不做替换**，
   * 但调用方仍会知道这个名字被认领了 —— 否则「覆盖了一个不带门限的谓词」
   * 会被报成「这个名字没人认识」，那是两句完全不同的话。
   */
  const effective = (fallback: number, mark: RegExp): { value: number; text: string } => {
    if (override === undefined) return { value: fallback, text: src }
    return { value: override, text: src.replace(mark, String(override)) }
  }

  const top = RE_TOP.exec(src)
  if (top) {
    if (block.questions.length !== 1) return null
    const qid = block.questions[0]!.id
    // ★ `top < x` 必须走 `topLt`，**不能**走 `probLt`。
    //   `topGte` 对三种答案类型各有各的判据（choice 看选中项概率、noul 看 max(p,1-p)、
    //   score 看 confidence），而 `probLt` 只认 noul、判的还是 p —— 两者不是补集：
    //   实测 p=0.05 时 `top >= 0.5` 和 `top < 0.5` **同时为真**，
    //   在 choice / score 上 `top < x` **恒假**（写了一道永不触发的闸门）。
    const e = effective(Number(top[2]), /([0-9]*\.?[0-9]+)$/)
    const fn = top[1] === '>=' ? topGte(qid, e.value) : topLt(qid, e.value)
    return { fn, text: e.text }
  }

  // ★ `prob:` / `score:` / `picked:` 都要**核对目标问题的类型**。
  //
  //   以前这里只把 id 交给 `probGte` / `scoreGte` / `picked`，不检查那个 id 指向的
  //   问题是什么类型。而这三个函数对**类型不符**的答案一律返回 `false` ——
  //   于是 `prob:tool >= 0.9 → ask_human` 写在 choice 块上时会**编译成功、
  //   但一次都不会触发**（实测恒为 false）。作者以为写了一道闸门，实际没有。
  //
  //   这和第十二轮 S2 是同一个根因（编译期不校验谓词与问题类型的匹配），
  //   S2 只修了 `top` 那一对。这里是它的另一半。
  //
  //   方向是 **fail open**：闸门消失而没有任何东西报错 —— 所以必须在这里拒绝，
  //   让它进 `problems`（见 `compilePolicy` 的说明）。
  // 找不到那个 id 时 `?.` 给 `undefined`，与「类型不符」一起落到下面每个 `!==` 判断上 ——
  // 两种都该拒绝。（第一版写成 `.find(...)` 再取 `.type`，id 不存在时会抛，
  //  tsc 和复现脚本都抓到了。）
  const typeOf = (id: string) => block.questions.find((q) => q.id === id)?.type

  const prob = RE_PROB.exec(src)
  if (prob) {
    if (typeOf(prob[1]!) !== 'noul') return null
    const e = effective(Number(prob[3]), /([0-9]*\.?[0-9]+)$/)
    const fn = prob[2] === '>=' ? probGte(prob[1]!, e.value) : probLt(prob[1]!, e.value)
    return { fn, text: e.text }
  }

  const sc = RE_SCORE.exec(src)
  if (sc) {
    if (typeOf(sc[1]!) !== 'score') return null
    const e = effective(Number(sc[2]), /([0-9]*\.?[0-9]+)$/)
    return { fn: scoreGte(sc[1]!, e.value), text: e.text }
  }

  const pk = RE_PICKED.exec(src)
  if (pk) {
    // 同理：`picked` 只对 choice 答案成立，用在别的类型上会恒为 false
    if (typeOf(pk[1]!) !== 'choice') return null
    // ★ 这条**不认领**任何覆盖：`picked:` 没有门限可换。有人覆盖了它的话，
    //   那个键会留在「没被用掉」那一堆里，由调用方当场报出来 ——
    //   他以为这里有个数可以调，而其实没有，这必须响（见 `gates.ts` 文件头）。
    return { fn: picked(pk[1]!, pk[2]!), text: src }
  }

  return null
}

/**
 * 块 → `PolicyRule[]`。
 *
 * 编译不了的谓词**留在结果里**并标出来，不静默丢 ——
 * 丢一条策略 = agent 少一道闸门，而它不会报错。
 *
 * @returns 没有策略时返回 `undefined`；有编译不了的谓词时 `ok` 为 false，
 *   且 `problems` 里逐条写明是哪条谓词
 */
/**
 * 这条谓词点名（或者 `top` 隐含）的是哪个问题。认不出来返回 `null`。
 *
 * 导出是因为**门限覆盖的键要用它** —— 「这个键指向哪条规则」和
 * 「这个谓词问的是哪个问题」必须是同一个判断，写两遍就会分叉。
 */
export function predicateQuestion(when: string, block: DocBlock): string | null {
  const src = when.trim()
  if (RE_TOP.test(src)) return block.questions.length === 1 ? block.questions[0]!.id : null
  const m = /^(?:prob|score|picked):([\w.-]+)/.exec(src)
  return m ? m[1]! : null
}

/** 这条谓词带数值门限吗（`picked:` 和 `else` 不带，所以覆盖不了） */
export function hasThreshold(when: string): boolean {
  return /^(?:top|prob:[\w.-]+|score:[\w.-]+)\s*(?:>=|<)\s*[0-9]/.test(when.trim())
}

export function compilePolicy(
  block: DocBlock,
  gates: GateOverrides = {},
): { rules: PolicyRule<AnswerSet>[]; ok: boolean; problems: string[]; applied: string[] } | undefined {
  if (block.policy.length === 0) return undefined
  const problems: string[] = []
  // 只取这个块的那几项 —— 键是 `<块 id>.<问题 id>`，按块切开
  const mine = splitGates(gates).byBlock[block.id] ?? {}
  const applied = new Set<string>()

  /*
    ★ **同一个问题上的多条规则，键怎么落到具体哪一条。**

    `grade_risk` 有两条都在问 `risk`：

        score:risk >= 2 → ask_human     （先判，严）
        score:risk >= 1 → auto_audit    （后判，宽）

    按问题名覆盖时，如果两条都改，它们就相等了 —— 后一条**永远不可达**，
    这个 agent 少一道闸门而一个错都不报（`tierProblems` 拦的就是它）。
    但「两条都改」也是唯一说得通的整体语义，所以不能简单地拒绝：**那样这个
    键就彻底没法用了**（第一版正是如此，写测试时当场发现）。

    所以：

      `grade_risk.risk=3`      → 只改**第 0 档**（最严的那条）。人写这个名字
                                 时想的就是「问人的那道坎」，那正是第 0 档。
      `grade_risk.risk[1]=2`   → 显式指名第 1 档

    第 0 档接受裸名字，其余档位必须带下标 —— 这样最常见的写法最短，
    而想要第二档的人也能写出来。
  */
  const tiers = new Map<string, number[]>()
  block.policy.forEach((r, i) => {
    const q = hasThreshold(r.when) ? predicateQuestion(r.when, block) : null
    if (!q) return
    const list = tiers.get(q) ?? []
    list.push(i)
    tiers.set(q, list)
  })

  const rules = block.policy.map((r, index) => {
    const q = hasThreshold(r.when) ? predicateQuestion(r.when, block) : null
    const tier = q ? (tiers.get(q) ?? []).indexOf(index) : -1
    const indexed = q && tier >= 0 ? mine[`${q}[${tier}]`] : undefined
    // 裸名字只认第 0 档；第 1 档往上必须写 `[i]`
    const bare = q && tier === 0 ? mine[q] : undefined
    const override = indexed ?? bare
    const key = q && override !== undefined ? (indexed !== undefined ? `${q}[${tier}]` : q) : null

    const fn = compilePredicate(r.when, block, override)
    if (!fn) {
      problems.push(`未编译的谓词 '${r.when}'（动作 ${r.action}）`)
      // ★ 这里**绝不能**返回一个省略 `when` 的规则。
      //
      //   `PolicyRule.when` 是可选的，而省略的含义是**无条件兜底**
      //   （`resolvePolicy` 见到它就立即返回）。以前这里返回 `{ action, reason }`，
      //   于是「这个谓词我没看懂」被编码成了「这条规则永远命中」——
      //   两种完全不同的意图共用了同一个表示。
      //
      //   后果是 `policy.ts` 的两道静态检查**同时失效**：假兜底恰好在最后一条时
      //   `catch_all_not_last` 不响（那正是合法兜底该在的位置），
      //   `policy_no_catch_all` 也被它骗过。实测：一份两条闸门都不命中的策略
      //   返回了 `ask_human` 而不是 `escalate`，且**一个警告都没有**。
      //
      //   现在产出 `when: () => false`：规则仍在列表里（界面看得见「这条没编译」），
      //   但永不命中，兜底语义保持原样。
      return { when: () => false, action: r.action, reason: `未编译的谓词 '${r.when}'` }
    }
    if (key) applied.add(`${block.id}.${key.split('[')[0]!}`)
    if (key?.includes('[')) applied.add(`${block.id}.${key}`)
    // ★ `reason` 用的是 **`fn.text`（生效原文）**，不是 `r.when`（文件原文）。
    //   覆盖过的门限在这里被如实写出来 —— 否则日志会写着一个用都没用的数。
    return { when: fn.fn, action: r.action, reason: `${fn.text} → ${r.action}` }
  })
  return { rules, ok: problems.length === 0, problems, applied: [...applied] }
}
