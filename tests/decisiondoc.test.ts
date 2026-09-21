/**
 * DECISION.md 的解析、编译，以及**它和代码有没有漂移**。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这份文件是用户手写的，所以它同时是**真实的输入边界**：认不出来的东西
 * 必须报出来，不能静默丢。一个被悄悄忽略的判定块，会让 agent 安静地少问
 * 一个问题，而没有任何东西会报错。
 *
 * 最后一组是这份文件存在的理由：`DECISION.md` 声称的和
 * `src/decisions.ts` 实际做的是不是同一件事。
 *
 * @module JevLoop/decisiondoc.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  parseDecisionDoc,
  compileQuestions,
  compilePolicy,
  compilePredicate,
  summarize,
  headline,
  isGate,
  type DocBlock,
} from '../src/decisiondoc.ts'
import type { DecisionSpec } from '../src/vocab-decision.ts'
import type { QuestionSet, AnswerSet } from '../src/vocab.ts'
import {
  needsTool,
  pickTool,
  pickInput,
  gradeRisk,
  stepOk,
  isDone,
  canDeliver,
  type AgentCtx,
} from '../src/decisions.ts'

const DECISION_MD = readFileSync(new URL('../DECISION.md', import.meta.url), 'utf8')

const doc = parseDecisionDoc(DECISION_MD)
const block = (id: string): DocBlock => {
  const b = doc.blocks.find((x) => x.id === id)
  assert.ok(b, `DECISION.md 里没有 '${id}' 这个块`)
  return b
}

// ═══════════════════════════════════════════════════════════
// 真实文件
// ═══════════════════════════════════════════════════════════

test('仓库自带的 DECISION.md 解析零问题', () => {
  assert.deepEqual(
    doc.problems,
    [],
    `解析报错：\n${doc.problems.map((p) => `  L${p.line}: ${p.message}`).join('\n')}`,
  )
  assert.equal(doc.blocks.length, 7)
  assert.ok(doc.generatorSection.length > 0, 'generator 段不能是空的 —— 它要进 system prompt')
})

test('每个块都编译出问题集（rule 除外）', () => {
  for (const b of doc.blocks) {
    const qs = compileQuestions(b)
    if (b.kind === 'rule') {
      assert.equal(qs, undefined, `${b.id} 是 rule，不该编译出问题`)
      continue
    }
    assert.ok(qs, `${b.id} 没编译出问题`)
    assert.equal(Object.keys(qs).length, b.questions.length)
  }
})

test('授权闸门是从 policy 推出来的，不是声明的', () => {
  assert.ok(isGate(block('grade_risk')), 'grade_risk 有 ask_human，应当被认成闸门')
  assert.ok(!isGate(block('is_done')), 'is_done 不该是闸门')
})

test('headline 的数字是从文件推出来的', () => {
  const s = summarize(doc)
  assert.equal(s.blocks, doc.blocks.length)
  assert.equal(s.modelDecisions + s.codeDecisions, s.blocks)
  assert.equal(
    s.questions,
    doc.blocks.reduce((n, b) => n + b.questions.length, 0),
  )
  assert.match(headline(doc), /7 个判定点/)
})

// ═══════════════════════════════════════════════════════════
// 防漂移：文件说的和代码做的是不是一回事
//
// 这是 DECISION.md 存在的理由。它一旦和 decisions.ts 分叉，
// 那份文件就从文档退化成了谎言。
// ═══════════════════════════════════════════════════════════

// 七个 DecisionSpec 的问题集各不相同，没有公共类型可指 —— 这里要的只是
// 「能取出 questions」，所以显式跨过类型系统，而不是给每个都写一遍适配。
const SPECS = [
  ['needs_tool', needsTool],
  ['pick_tool', pickTool],
  ['pick_input', pickInput],
  ['grade_risk', gradeRisk],
  ['step_ok', stepOk],
  ['is_done', isDone],
  ['can_deliver', canDeliver],
] as unknown as [string, DecisionSpec<AgentCtx>][]

/** 一个字段够用的 ctx —— 问题集只需要能算出来，不需要算得对 */
const CTX: AgentCtx = {
  task: '读一下目录里的文件',
  cwd: '/tmp',
  files: ['a.ts', 'b.ts'],
  readFiles: [],
  history: [],
  lastTool: 'read_file',
  lastResult: 'ok',
  draft: 'draft',
}

function realQuestions(spec: unknown): QuestionSet {
  const s = spec as DecisionSpec<AgentCtx>
  return typeof s.questions === 'function' ? s.questions(CTX) : s.questions
}

test('DECISION.md 覆盖了 decisions.ts 里的全部问题，一个不少', () => {
  const documented = new Set(doc.blocks.flatMap((b) => b.questions.map((q) => q.id)))
  const missing: string[] = []
  for (const [blockId, spec] of SPECS) {
    for (const qid of Object.keys(realQuestions(spec))) {
      if (!documented.has(qid)) missing.push(`${blockId} 的 '${qid}'`)
    }
  }
  assert.deepEqual(missing, [], `DECISION.md 漏了这些判定问题：${missing.join('、')}`)
})

test('DECISION.md 没有编造 decisions.ts 里不存在的问题', () => {
  const real = new Set(SPECS.flatMap(([, spec]) => Object.keys(realQuestions(spec))))
  const invented = doc.blocks
    .flatMap((b) => b.questions.map((q) => q.id))
    .filter((qid) => !real.has(qid))
  assert.deepEqual(invented, [], `DECISION.md 里这些问题是凭空写的：${invented.join('、')}`)
})

test('每个问题的原语类型和代码一致', () => {
  const wrong: string[] = []
  for (const [, spec] of SPECS) {
    for (const [qid, q] of Object.entries(realQuestions(spec))) {
      const documented = doc.blocks.flatMap((b) => b.questions).find((q) => q.id === qid)
      if (documented && documented.type !== q.type) {
        wrong.push(`${qid}: 文件说 ${documented.type}，代码是 ${q.type}`)
      }
    }
  }
  assert.deepEqual(wrong, [])
})

test('score 的档位标签和代码逐字一致', () => {
  // 档位顺序直接决定 `scoreGte('risk', 2)` 的含义，错一位就是另一个语义
  const realRisk = realQuestions(gradeRisk)['risk']
  assert.equal(realRisk?.type, 'score')
  const documented = block('grade_risk').questions.find((q) => q.id === 'risk')
  assert.ok(documented)
  assert.deepEqual(
    documented.options.map((o) => o.criteria),
    realRisk && realRisk.type === 'score' ? realRisk.criteria : [],
  )
})

test('noul 问题的 true/false 说明和代码一致', () => {
  const wrong: string[] = []
  for (const [, spec] of SPECS) {
    for (const [qid, q] of Object.entries(realQuestions(spec))) {
      if (q.type !== 'noul' || !q.criteria) continue
      const d = block(
        doc.blocks.find((b) => b.questions.some((x) => x.id === qid))!.id,
      ).questions.find((x) => x.id === qid)!
      const t = d.options.find((o) => o.name === 'true')?.criteria
      const f = d.options.find((o) => o.name === 'false')?.criteria
      if (t !== q.criteria.true) wrong.push(`${qid}.true`)
      if (f !== q.criteria.false) wrong.push(`${qid}.false`)
    }
  }
  assert.deepEqual(wrong, [], `这些说明对不上：${wrong.join('、')}`)
})

test('阈值数字和代码里的一致', () => {
  // 阈值写错不会报错，只会让 agent 的行为悄悄变一档 —— 所以要钉住
  const expected: Record<string, string> = {
    needs_tool: 'prob:needs_tool >= 0.5',
    pick_tool: 'top >= 0.6',
    pick_input: 'top >= 0.5',
    step_ok: 'prob:ok >= 0.5',
    is_done: 'prob:done >= 0.6',
  }
  for (const [id, when] of Object.entries(expected)) {
    const got = block(id).policy[0]?.when
    assert.equal(got, when, `${id} 的第一条策略谓词变了`)
  }
  const risk = block('grade_risk').policy.map((r) => r.when)
  assert.deepEqual(risk, ['score:risk >= 2', 'prob:needs_auth >= 0.5', 'score:risk >= 1', 'else'])
})

// ═══════════════════════════════════════════════════════════
// 畸形输入 —— 这是真实的边界，必须报出来
// ═══════════════════════════════════════════════════════════

function problemsOf(md: string): string[] {
  return parseDecisionDoc(md).problems.map((p) => p.message)
}

test('缺 kind 的块要报错，不能当成默认值悄悄放过', () => {
  const p = problemsOf('## foo\nask: 问点什么\n- a — 甲\n- b — 乙\n')
  assert.equal(p.length, 1)
  assert.match(p[0]!, /缺少 kind/)
})

test('选项缺分隔符要报错，并说清正确写法', () => {
  // 声明 choice 却写成无名档位 —— 最可能的原因就是漏了分隔符，报错要点出来
  const p = problemsOf('## foo\nkind: choice\nask: 问\n- 甲\n- 乙\n')
  assert.equal(p.length, 1)
  assert.match(p[0]!, /分隔符/)
})

test('不认识的键要报错，不静默忽略', () => {
  const p = problemsOf('## foo\nkind: rule\ntypo_key: 1\n')
  assert.equal(p.length, 1)
  assert.match(p[0]!, /不认识的键 'typo_key'/)
})

test('kind 拼错要报错，并列出合法值', () => {
  const p = problemsOf('## foo\nkind: choise\nask: 问\n- a — 甲\n- b — 乙\n')
  assert.equal(p.length, 1)
  assert.match(p[0]!, /choice \/ noul \/ score \/ mixed \/ rule/)
})

test('kind 和选项写法对不上要报错', () => {
  // 声明 noul 却写成 choice 的有名选项
  const p = problemsOf('## foo\nkind: noul\nask: 问\n- a — 甲\n- b — 乙\n')
  assert.ok(
    p.some((m) => /和选项写法对不上/.test(m)),
    `应当报出错配，实际：${JSON.stringify(p)}`,
  )
})

test('同一块里选项写法混用要报错', () => {
  const p = problemsOf('## foo\nkind: mixed\n### a\nask: 问\n- x — 甲\n- 乙\n### b\nask: 问2\n- true — 是\n- false — 否\n')
  assert.ok(
    p.some((m) => /写法不一致/.test(m)),
    `应当报出混用，实际：${JSON.stringify(p)}`,
  )
})

test('noul 的选项必须叫 true / false', () => {
  const p = problemsOf('## foo\nkind: noul\nask: 问\n- yes — 是\n- no — 否\n')
  assert.ok(p.some((m) => /true \/ false|true/.test(m)), JSON.stringify(p))
})

test('kind: mixed 只给一个问题要报错', () => {
  const p = problemsOf('## foo\nkind: mixed\n### a\nask: 问\n- true — 是\n- false — 否\n')
  assert.ok(p.some((m) => /mixed 至少要 2 个问题/.test(m)), JSON.stringify(p))
})

test('单问题 kind 给了两个问题要报错', () => {
  const md = '## foo\nkind: noul\n### a\nask: 问\n- true — 是\n- false — 否\n### b\nask: 问2\n- true — 是\n- false — 否\n'
  const p = problemsOf(md)
  assert.ok(p.some((m) => /要有且只有 1 个问题/.test(m)), JSON.stringify(p))
})

test('choice 少于两个选项要报错，但写了 dynamic 就放行', () => {
  const bad = problemsOf('## foo\nkind: choice\nask: 问\n- x — 甲\n')
  assert.ok(bad.some((m) => /至少要 2 个/.test(m)), JSON.stringify(bad))

  const ok = parseDecisionDoc('## foo\nkind: choice\nask: 问\ndynamic: 运行时算\n- x — 甲\n')
  assert.deepEqual(ok.problems, [], '写了 dynamic 就不该再要求选项数量')
})

test('kind: rule 不该有问题', () => {
  const p = problemsOf('## foo\nkind: rule\nask: 问\n- a — 甲\n- b — 乙\n')
  assert.ok(p.some((m) => /rule 不该有问题/.test(m)), JSON.stringify(p))
})

test('策略缺箭头要报错', () => {
  const p = problemsOf('## foo\nkind: rule\npolicy:\n  - 这行没有箭头\n')
  assert.ok(p.some((m) => /缺 '→ 动作'/.test(m)), JSON.stringify(p))
})

test('问题 id 和判定 id 重复都要报错', () => {
  const dupQ = problemsOf('## foo\nkind: mixed\n### a\nask: 问\n- true — 是\n- false — 否\n### a\nask: 问2\n- true — 是\n- false — 否\n')
  assert.ok(dupQ.some((m) => /问题 id 'a' 重复/.test(m)), JSON.stringify(dupQ))

  const dupB = problemsOf('## foo\nkind: rule\n## foo\nkind: rule\n')
  assert.ok(dupB.some((m) => /判定 id 'foo' 重复/.test(m)), JSON.stringify(dupB))
})

test('报错带行号', () => {
  const doc2 = parseDecisionDoc('# t\n\n## foo\nkind: choice\nask: 问\n- 甲\n- 乙\n')
  assert.equal(doc2.problems.length, 1)
  assert.equal(doc2.problems[0]!.line, 6, '报的应当是第 6 行那个选项')
})

// ═══════════════════════════════════════════════════════════
// 谓词编译
// ═══════════════════════════════════════════════════════════

const ans = (o: Record<string, unknown>): AnswerSet => o as AnswerSet

test('谓词：prob / score / picked / else', () => {
  const b = block('grade_risk')
  assert.ok(compilePredicate('prob:needs_auth >= 0.5', b)!(ans({ needs_auth: { type: 'noul', noul: 0.9 } })))
  assert.ok(!compilePredicate('prob:needs_auth >= 0.5', b)!(ans({ needs_auth: { type: 'noul', noul: 0.1 } })))
  assert.ok(compilePredicate('score:risk >= 2', b)!(ans({ risk: { type: 'score', score: 2.4 } })))
  assert.ok(!compilePredicate('score:risk >= 2', b)!(ans({ risk: { type: 'score', score: 1.9 } })))
  assert.ok(compilePredicate('else', b)!(ans({})))
})

test('谓词：多问题块里 top 是歧义的，必须拒绝', () => {
  // grade_risk 有两个问题，`top` 指哪个说不清 —— 拒绝比猜一个安全
  assert.equal(compilePredicate('top >= 0.6', block('grade_risk')), null)
  // 单问题块可以省掉 id
  assert.ok(compilePredicate('top >= 0.6', block('is_done')))
})

test('谓词：不认识的写法返回 null，不静默当兜底', () => {
  const b = block('is_done')
  for (const src of ['top > 0.6', 'prob:done <= 0.5', 'confidence >= 0.9', '随便写点什么']) {
    assert.equal(compilePredicate(src, b), null, `'${src}' 应当被拒绝`)
  }
})

test('未编译的谓词留在策略里并标出来，不静默丢', () => {
  const doc2 = parseDecisionDoc('## foo\nkind: noul\nask: 问\n- true — 是\n- false — 否\npolicy:\n  - nonsense → act\n')
  const b = doc2.blocks[0]!
  const pol = compilePolicy(b)
  assert.ok(pol)
  assert.equal(pol.ok, false, '有编译不了的谓词时要标出来')
  assert.equal(pol.rules.length, 1, '规则不能因为编译失败就消失 —— 消失等于少一道闸门')
  assert.match(pol.rules[0]!.reason ?? '', /未编译的谓词/)
})

test('没有策略时 compilePolicy 返回 undefined，而不是空数组', () => {
  assert.equal(compilePolicy(block('needs_tool')) === undefined, false)
  const doc2 = parseDecisionDoc('## foo\nkind: rule\n')
  assert.equal(compilePolicy(doc2.blocks[0]!), undefined)
})
