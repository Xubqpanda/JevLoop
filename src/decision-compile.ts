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

/**
 * 谓词原文 → 判定函数。
 *
 * `top` 指的是**本块唯一那个问题**，所以单问题块不必重复写问题 id。
 * 多问题块用 `top` 是歧义的，会被拒绝 —— 必须点名 `prob:` / `score:`。
 *
 * @returns 判定函数；原文不认识时返回 `null`（调用方应记为 problem，不要当兜底）
 */
export function compilePredicate(
  when: string,
  block: DocBlock,
): ((a: AnswerSet) => boolean) | null {
  const src = when.trim()
  if (ELSE_FORMS.has(src)) return () => true

  const top = RE_TOP.exec(src)
  if (top) {
    if (block.questions.length !== 1) return null
    const qid = block.questions[0]!.id
    // ★ `top < x` 必须走 `topLt`，**不能**走 `probLt`。
    //   `topGte` 对三种答案类型各有各的判据（choice 看选中项概率、noul 看 max(p,1-p)、
    //   score 看 confidence），而 `probLt` 只认 noul、判的还是 p —— 两者不是补集：
    //   实测 p=0.05 时 `top >= 0.5` 和 `top < 0.5` **同时为真**，
    //   在 choice / score 上 `top < x` **恒假**（写了一道永不触发的闸门）。
    return top[1] === '>=' ? topGte(qid, Number(top[2])) : topLt(qid, Number(top[2]))
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
    return prob[2] === '>=' ? probGte(prob[1]!, Number(prob[3])) : probLt(prob[1]!, Number(prob[3]))
  }

  const sc = RE_SCORE.exec(src)
  if (sc) {
    if (typeOf(sc[1]!) !== 'score') return null
    return scoreGte(sc[1]!, Number(sc[2]))
  }

  const pk = RE_PICKED.exec(src)
  if (pk) {
    // 同理：`picked` 只对 choice 答案成立，用在别的类型上会恒为 false
    if (typeOf(pk[1]!) !== 'choice') return null
    return picked(pk[1]!, pk[2]!)
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
export function compilePolicy(
  block: DocBlock,
): { rules: PolicyRule<AnswerSet>[]; ok: boolean; problems: string[] } | undefined {
  if (block.policy.length === 0) return undefined
  const problems: string[] = []
  const rules = block.policy.map((r) => {
    const fn = compilePredicate(r.when, block)
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
    return { when: fn, action: r.action, reason: `${r.when} → ${r.action}` }
  })
  return { rules, ok: problems.length === 0, problems }
}
