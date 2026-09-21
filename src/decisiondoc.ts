/**
 * JevLoop · DECISION.md
 *
 * ══════════════════════════════════════════════════════════════
 *  每一代 agent 框架都留下一个 .md：AGENTS.md 讲约定，SKILL.md
 *  讲能力。它们都是**给大模型读的散文** —— 靠模型去理解，于是每轮
 *  都付 token，而模型可以不听，也没人知道它到底听没听。
 *
 *  DECISION.md 是第一个**被编译**的。
 * ══════════════════════════════════════════════════════════════
 *
 * 一份文件，两个消费者：
 *
 *     结构块  →  questions + policy  →  判定模型（几十毫秒，不花钱）
 *     散文    →  system prompt       →  大模型（整个 loop 唯一贵的一步）
 *
 * 所以它是**减法**：每个搬进来的块，都是一个不用再问大模型的问题。
 * 省下的量可以直接数出来 —— 见 `summarize()` 和 `headline()`。
 *
 * ── 语法 ──────────────────────────────────────────────────────
 *
 *     # 标题
 *     引言散文（只给人读）
 *
 *     ## <判定 id>
 *     kind: choice | noul | score | mixed | rule
 *     when: 什么时候问（散文）
 *     ask: 问题本身
 *     - 选项名 — 什么情况下选它
 *
 *     ### <问题 id>          ← 一个判定问多件事时用
 *     ask: 第二个问题
 *     - 档位说明            ← score 的档位没有名字，直接写
 *
 *     policy:
 *       - top >= 0.6 → call
 *       - else → escalate
 *
 *     为什么这么设计（散文，只给人读）
 *
 *     ## generator
 *     这一段原样进 system prompt
 *
 * 只有三条正则：`## id`、`key: value`、`- 名字 — 说明`。装饰性的
 * markdown（粗体、引用块、表格）不参与解析，写的时候不用小心翼翼。
 *
 * ── 五条刻意的边界 ────────────────────────────────────────────
 *
 * · **不引入 YAML。** markdown 本来就够表达，而零依赖是这个仓库的
 *   对外承诺。
 *
 * · **只允许一层判定、一层问题，禁止更深。** 一个块如果需要条件判断，
 *   它就不属于 DECISION.md，属于代码。这不是能力不足，是防它长成 DSL。
 *
 * · **`kind` 描述「谁答」，不是「问什么」。** 所以没有 `gate` 这个
 *   kind —— 一个判定是不是授权闸门，从它的 policy 里有没有
 *   `ask_human` 就能推出来，不该再要人声明一遍。
 *
 * · **`kind: rule` 不编译成 DecisionSpec。** 它声明「这一步由代码答，
 *   没有模型参与」。硬把它编成一个没有问题的决策节点，是假装它们统一。
 *
 * · **谓词是封闭词汇表，不是表达式语言。** 只编译 `policy.ts` 已经
 *   提供的那几个（`topGte` / `probGte` / `probLt` / `scoreGte` /
 *   `picked`）。想写更复杂的条件，那是代码的事。
 *
 * ── 解析失败怎么办 ────────────────────────────────────────────
 *
 * 这是**真实的边界**（用户手写的文件），所以按 docs/CODE-STYLE.md §6 在这里
 * 校验，且**绝不静默丢块**：认不出来的东西进 `problems`，带行号。
 * 一个被悄悄忽略的判定块，比一个报错的块危险得多 —— 前者会让 agent
 * 安静地少问一个问题。
 *
 * ## 待拆
 *
 * **形状已独立**到 `decision-shape.ts`（L0，本文件 re-export 它以免消费方改动）。
 * 剩下两件事：**词法 + 解析**（Markdown → `DocBlock`），与一小段**汇总**
 * （`summarize` / `headline`，只是派生视图，可以随解析留着也可以再切）。
 * 解析本身是自洽的一件事（两趟：切段、逐段解释），**再切需要先重新理解它**，
 * **接缝已经定了，还没切。**
 * **行数不在这里写** —— 它会漂（这里曾写 533 与 604），见 docs/CODE-STYLE.md §12。
 *
 * @module JevLoop/decisiondoc
 */

import { ACTIONS } from './vocab-decision.ts'

/**
 * `ACTIONS` 的查表版本。
 *
 * 单独放一个 `Set` 是因为它要判**任意字符串**（手写文件里的原文），
 * 而 `ACTIONS.includes()` 的类型签名只接受联合成员 —— 那正是「边界上要放宽、
 * 内部要收紧」的分界：类型系统管内部，这个 Set 管边界。
 */
const KNOWN_ACTIONS: ReadonlySet<string> = new Set(ACTIONS)

// 形状定义在 `decision-shape.ts`（同层 L0）。这里 re-export 是为了消费方不用改 import ——
// **这一招在 `decision-compile.ts` 那次不成立**（那边是 L2，L0 不能 import L2）。
// ★ re-export **不会**把名字带进本文件作用域 —— 自己还要用的名字必须单独 import。
import {
  KINDS,
  type BlockKind,
  type DocBlock,
  type DecisionDoc,
  type DocOption,
  type DocPolicyRule,
  isGate,
  type DocProblem,
  type DocQuestion,
  type DocSummary,
  type Primitive,
} from './decision-shape.ts'

export { isGate } from './decision-shape.ts'
export type { BlockKind, DecisionDoc, DocBlock, DocOption, DocPolicyRule, DocProblem, DocQuestion, DocSummary, Primitive } from './decision-shape.ts'

// ═══════════════════════════════════════════════════════════
// 词法
// ═══════════════════════════════════════════════════════════

const RE_TITLE = /^#\s+(.+?)\s*$/
const RE_HEADING = /^##\s+(.+?)\s*$/
const RE_SUBHEADING = /^###\s+(.+?)\s*$/
const RE_KEY = /^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$/
const RE_ITEM = /^[-*]\s+(.*)$/
/** 选项的「名字 — 说明」分隔符。中文破折号和双连字符都收 */
const SEPARATORS = [/\s+—\s+/, /\s+--\s+/]

/** `## generator` 这类标题是散文段，不是判定块 */
const GENERATOR_HEADINGS = new Set(['generator', '给生成模型', '生成模型', 'system prompt'])

function splitOption(body: string): { name: string; criteria: string } | null {
  for (const sep of SEPARATORS) {
    const m = sep.exec(body)
    if (m && m.index > 0) {
      const name = body.slice(0, m.index).trim()
      const criteria = body.slice(m.index + m[0].length).trim()
      if (name && criteria) return { name, criteria }
    }
  }
  return null
}

function isKind(v: string): v is BlockKind {
  return (KINDS as readonly string[]).includes(v)
}

// ═══════════════════════════════════════════════════════════
// 解析
//
// 两趟：先把 `##` 段原样切开收集，再逐段解释。这样字段顺序、
// 空行位置怎么写都不影响结果，而报错仍能精确到行。
// ═══════════════════════════════════════════════════════════

interface RawLine {
  text: string
  line: number
}

interface RawSection {
  heading: string
  line: number
  lines: RawLine[]
}

function splitSections(lines: string[]): { title: string; intro: string; sections: RawSection[] } {
  let title = ''
  const intro: string[] = []
  const sections: RawSection[] = []
  let current: RawSection | null = null

  lines.forEach((text, i) => {
    const line = i + 1
    const h = RE_HEADING.exec(text)
    if (h) {
      current = { heading: h[1]!, line, lines: [] }
      sections.push(current)
      return
    }
    if (!current) {
      const t = RE_TITLE.exec(text)
      if (t) {
        title = t[1]!
        return
      }
      intro.push(text)
      return
    }
    current.lines.push({ text, line })
  })

  return { title, intro: intro.join('\n').trim(), sections }
}

/** 解析过程中的一个问题：选项还没定型，因为类型要靠写法判断 */
interface DraftQuestion {
  id: string
  line: number
  ask: string
  options: DocOption[]
  /** 有几个选项是「名字 — 说明」写法的。全有 ⇒ choice，全无 ⇒ score */
  named: number
}

function newDraft(id: string, line: number): DraftQuestion {
  return { id, line, ask: '', options: [], named: 0 }
}

function interpretBlock(section: RawSection, problems: DocProblem[]): DocBlock {
  let kind: BlockKind | null = null
  let sawKind = false
  let when = ''
  let dynamic = ''
  const questions: DraftQuestion[] = []
  const policy: DocPolicyRule[] = []
  const rationale: string[] = []

  // 块级的问题：`###` 出现之前，`ask:` 和 `- ` 都属于它。
  // 如果整段都没写 `###`，它就是唯一的问题。
  let current = newDraft(section.heading, section.line)
  let usedSubheading = false
  let inPolicy = false

  const pushCurrentIfUsed = () => {
    if (current.ask || current.options.length > 0) questions.push(current)
  }

  for (const { text, line } of section.lines) {
    const trimmed = text.trim()
    if (trimmed === '') {
      rationale.push('')
      inPolicy = false
      continue
    }

    const sub = RE_SUBHEADING.exec(trimmed)
    if (sub) {
      pushCurrentIfUsed()
      current = newDraft(sub[1]!, line)
      usedSubheading = true
      inPolicy = false
      continue
    }

    const item = RE_ITEM.exec(trimmed)
    if (item) {
      const body = item[1]!.trim()
      if (inPolicy) {
        const arrow = body.indexOf('→')
        if (arrow < 0) {
          problems.push({ line, message: `策略 '${body}' 缺 '→ 动作'` })
        } else {
          const w = body.slice(0, arrow).trim()
          const a = body.slice(arrow + 1).trim()
          if (!w || !a) problems.push({ line, message: `策略 '${body}' 的谓词或动作为空` })
          else {
            policy.push({ when: w, action: a })
            // ★ action 是封闭词汇表，在这里校验（§6 说的真实边界：用户手写的文件）。
            //   不校验会怎样：`→ ask_humam` 编译成功、`isGate` 返回 false、界面把这个
            //   判定块显示成正常，而 `resolvePolicy` 会返回一个**没有任何消费方能处理**
            //   的动作 —— 而且它是 fail open 的（本该拦住的那一步直接放行）。
            if (!KNOWN_ACTIONS.has(a)) {
              problems.push({
                line,
                message: `不认识的 action '${a}' —— loop 只认 ${ACTIONS.join(' / ')}`,
              })
            }
          }
        }
        continue
      }
      const parts = splitOption(body)
      if (parts) {
        if (current.options.length === 0) current.line = line
        current.options.push(parts)
        current.named++
      } else {
        // score 的档位是「从低到高的一句话」，本来就没有名字
        if (current.options.length === 0) current.line = line
        current.options.push({ name: String(current.options.length), criteria: body })
      }
      continue
    }

    const kv = RE_KEY.exec(trimmed)
    // `policy:` 是唯一会改变后续列表含义的键，要先于通用 key 判断
    if (kv && kv[1]!.toLowerCase() === 'policy') {
      inPolicy = true
      continue
    }
    inPolicy = false

    if (kv) {
      const key = kv[1]!.toLowerCase()
      const value = kv[2]!.trim()
      if (key === 'kind') {
        sawKind = true
        if (isKind(value)) kind = value
        else problems.push({ line, message: `kind 只能是 ${KINDS.join(' / ')}，收到 '${value}'` })
      } else if (key === 'when') when = value
      else if (key === 'dynamic') dynamic = value
      else if (key === 'ask') current.ask = value
      else problems.push({ line, message: `不认识的键 '${key}'（只有 kind / when / dynamic / ask / policy 是键）` })
      continue
    }

    rationale.push(trimmed)
  }

  pushCurrentIfUsed()

  if (!sawKind) {
    problems.push({
      line: section.line,
      message: `'## ${section.heading}' 缺少 kind。每个 ## 段都必须声明 kind —— 这正是三分法要你表态的地方`,
    })
  }

  const resolved = interpretQuestions(questions, usedSubheading, problems)
  validateBlock(section, kind, resolved, dynamic, problems)

  return {
    id: section.heading,
    kind: kind ?? 'rule',
    when,
    dynamic,
    questions: resolved,
    policy,
    rationale: rationale.join('\n').trim(),
    line: section.line,
  }
}

/** 定问题的原语类型。**靠写法推**，然后和 kind 对账 */
function interpretQuestions(
  drafts: DraftQuestion[],
  usedSubheading: boolean,
  problems: DocProblem[],
): DocQuestion[] {
  const out: DocQuestion[] = []
  for (const d of drafts) {
    if (d.options.length === 0) {
      problems.push({ line: d.line, message: `问题 '${d.id}' 一个选项都没有` })
      continue
    }
    let type: Primitive
    const names = d.options.map((o) => o.name)
    if (names.length === 2 && names.includes('true') && names.includes('false')) {
      // true/false 是 noul 的写法，`vocab.ts` 的 criteria 就是 { true, false }
      type = 'noul'
    } else if (d.named === d.options.length) {
      type = 'choice'
    } else if (d.named === 0) {
      type = 'score'
    } else {
      problems.push({
        line: d.line,
        message:
          `问题 '${d.id}' 的选项写法不一致：${d.named} 个有「名字 — 说明」，` +
          `${d.options.length - d.named} 个没有。要么全写名字（choice），要么全不写（score）`,
      })
      type = 'choice'
    }
    if (!d.ask) problems.push({ line: d.line, message: `问题 '${d.id}' 缺少 ask` })
    if (usedSubheading && d.id === '') problems.push({ line: d.line, message: '`###` 后面要写问题 id' })
    out.push({ id: d.id, type, ask: d.ask, options: d.options, line: d.line })
  }

  const seen = new Set<string>()
  for (const q of out) {
    if (seen.has(q.id)) problems.push({ line: drafts.find((d) => d.id === q.id)?.line ?? 0, message: `问题 id '${q.id}' 重复` })
    seen.add(q.id)
  }

  return out
}

/**
 * 逐 kind 校验。
 *
 * 每条都是「不报错就会安静地判错」的情况，所以宁可啰嗦：
 * 问题个数和 kind 不符意味着文件说的和实际编译出来的不是一个东西。
 */
function validateBlock(
  section: RawSection,
  kind: BlockKind | null,
  questions: DocQuestion[],
  dynamic: string,
  problems: DocProblem[],
): void {
  const at = section.line
  if (kind === null) return

  if (kind === 'rule') {
    if (questions.length > 0) {
      problems.push({ line: at, message: 'kind: rule 不该有问题 —— 它不经过模型。要问模型就改成 choice' })
    }
    return
  }

  if (kind === 'mixed') {
    if (questions.length < 2) {
      problems.push({
        line: at,
        message: `kind: mixed 至少要 2 个问题（收到 ${questions.length} 个）。只有一个就用 choice / noul / score`,
      })
    }
    minOptions(questions, dynamic, problems)
    return
  }

  if (questions.length !== 1) {
    problems.push({
      line: at,
      message: `kind: ${kind} 要有且只有 1 个问题（收到 ${questions.length} 个）。问了多件事就用 kind: mixed`,
    })
    return
  }
  const q = questions[0]!
  if (q.type !== kind) {
    problems.push({
      line: q.line,
      message:
        `kind: ${kind} 和选项写法对不上：按写法这是 ${q.type}` +
        (q.type === 'score'
          ? '（档位没有名字）—— 如果本来想写 choice，是选项少写了「名字 — 说明」的分隔符'
          : q.type === 'choice' && kind === 'noul'
            ? '（选项都写了名字）—— noul 的两个选项必须叫 true 和 false'
            : q.type === 'noul'
              ? '（true/false）'
              : '（选项都写了名字）'),
    })
  }
  minOptions(questions, dynamic, problems)
}

/**
 * `choice` 至少要两个选项 —— 一个选项的选择不是选择，是个常量。
 *
 * 但候选每步重建的判定（写了 `dynamic:`）在文件里可能只列一个占位，
 * 这时不要求数量：真正的候选由代码在运行时给（docs/CODE-STYLE.md §8.4）。
 */
function minOptions(questions: DocQuestion[], dynamic: string, problems: DocProblem[]): void {
  if (dynamic) return
  for (const q of questions) {
    if (q.type === 'choice' && q.options.length < 2) {
      problems.push({
        line: q.line,
        message:
          `问题 '${q.id}' 只有 ${q.options.length} 个选项，choice 至少要 2 个。` +
          `如果候选是运行时算出来的，加一行 \`dynamic: <怎么算的>\``,
      })
    }
  }
}

export function parseDecisionDoc(md: string): DecisionDoc {
  const problems: DocProblem[] = []
  const { title, intro, sections } = splitSections(md.split('\n'))

  const blocks: DocBlock[] = []
  const generator: string[] = []

  for (const s of sections) {
    if (GENERATOR_HEADINGS.has(s.heading.toLowerCase())) {
      generator.push(...s.lines.map((l) => l.text))
      continue
    }
    blocks.push(interpretBlock(s, problems))
  }

  const seen = new Set<string>()
  for (const b of blocks) {
    if (seen.has(b.id)) problems.push({ line: b.line, message: `判定 id '${b.id}' 重复` })
    seen.add(b.id)
  }

  return {
    title,
    intro,
    blocks,
    generatorSection: generator.join('\n').trim(),
    problems,
    source: md,
  }
}

// ═══════════════════════════════════════════════════════════
// 汇总
// ═══════════════════════════════════════════════════════════

export function summarize(doc: DecisionDoc): DocSummary {
  const byKind: Record<BlockKind, number> = { choice: 0, noul: 0, score: 0, mixed: 0, rule: 0 }
  const byPrimitive: Record<Primitive, number> = { choice: 0, noul: 0, score: 0 }
  let options = 0
  let questions = 0
  let gates = 0

  for (const b of doc.blocks) {
    byKind[b.kind]++
    if (isGate(b)) gates++
    for (const q of b.questions) {
      questions++
      byPrimitive[q.type]++
      options += q.options.length
    }
  }

  return {
    blocks: doc.blocks.length,
    byKind,
    questions,
    byPrimitive,
    modelDecisions: doc.blocks.length - byKind.rule,
    codeDecisions: byKind.rule,
    gates,
    options,
  }
}

/**
 * 这份文件要说的一句话。
 *
 * 数字全部**从文件本身推出来**，不是许愿 —— 改一个块的 kind，
 * 这句话就跟着变。
 *
 * 英文：它同时是 `jevloop spec` 的输出和界面「规格」页的标题，
 * 两者都是给外面的人看的门面（同 `examples/demo.ts` 的输出）。
 */
export function headline(doc: DecisionDoc): string {
  const s = summarize(doc)
  const parts = [
    `${s.blocks} decisions`,
    `${s.questions} questions`,
    `${s.modelDecisions} answered by the decision model`,
    `${s.codeDecisions} decided by code`,
  ]
  if (s.gates > 0) parts.push(`${s.gates} authorisation ${s.gates === 1 ? 'gate' : 'gates'}`)
  return `${parts.join(', ')}; ${s.options} options in total`
}
