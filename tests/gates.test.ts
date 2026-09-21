/**
 * 门限覆盖。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * ══════════════════════════════════════════════════════════════
 *  这一节测两件事，而它们各自对应一条**静默失败**：
 *
 *  **① 覆盖没生效。** 名字写错一个字母，门限就是默认值 —— 你以为加了一道
 *    闸门，其实什么都没发生，而且没有任何东西报错。所以「写错了要响」
 *    本身是功能，不是防御性代码。
 *
 *  **② 覆盖生效了，但日志里写的还是默认值。** 那更难查：数字看起来很正常，
 *    只是不是这一次用的那个。所以 `reason`（进日志、进交付闸门给模型看的
 *    那句话）必须是**生效**的那个数。
 * ══════════════════════════════════════════════════════════════
 *
 * @module JevLoop/gates.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { parseGates, describeGates, splitGates } from '../src/gates.ts'
import { buildDecisions, gateKeys, resolveGates } from '../src/decisions.ts'
import { resolvePolicy } from '../src/policy.ts'

// ═══════════════════════════════════════════════════════════
// 解析
// ═══════════════════════════════════════════════════════════

test('解析：一项、多项、各种分隔符', () => {
  assert.deepEqual(parseGates('can_deliver.unsupported=0.7').overrides, { 'can_deliver.unsupported': 0.7 })
  assert.deepEqual(parseGates('a.b=0.1, c.d=2').overrides, { 'a.b': 0.1, 'c.d': 2 })
  assert.deepEqual(parseGates('a.b=0.1;c.d=2').overrides, { 'a.b': 0.1, 'c.d': 2 })
  assert.deepEqual(parseGates('a.b=0.1\nc.d=2').overrides, { 'a.b': 0.1, 'c.d': 2 })
  assert.deepEqual(parseGates('  a.b = 0.1  ').overrides, { 'a.b': 0.1 })
  assert.deepEqual(parseGates('').overrides, {}, '空串 = 一项都没有，不是错误')
})

test('解析：认不出来的**不丢**，落在 problems 里', () => {
  // 丢掉的话调用方会以为一切都好 —— 那正是文件头说的第 ① 种静默失败
  const r = parseGates('a.b=0.1, 乱写, c.d=不是数, =0.5, e=0.5')
  assert.deepEqual(r.overrides, { 'a.b': 0.1 }, '只有认得的那项进了覆盖表')
  assert.equal(r.problems.length, 4, `四条都该报出来：${r.problems.join('；')}`)
  assert.ok(r.problems.some((p) => p.includes('乱写')))
  assert.ok(r.problems.some((p) => p.includes('不是数')))
})

test('describeGates：排序过，空表返回 null', () => {
  assert.equal(describeGates({}), null)
  // 排序是为了让同一份覆盖在日志里每次都长一样 —— 不然没法比对两次运行
  assert.equal(describeGates({ 'z.b': 1, 'a.b': 2 }), 'a.b=2, z.b=1')
})

test('splitGates：按块切开，点号坏掉的进 unused', () => {
  const r = splitGates({ 'x.a': 1, 'y.b': 2, 没有点: 3 })
  assert.deepEqual(r.byBlock, { x: { a: 1 }, y: { b: 2 } })
  assert.deepEqual(r.unused, ['没有点'])
})

// ═══════════════════════════════════════════════════════════
// 可覆盖的键 —— 从 DECISION.md 现算
// ═══════════════════════════════════════════════════════════

test('gateKeys 是现算的，和 DECISION.md 里带门限的谓词一一对应', () => {
  // 抄一份清单的话，文件里改了问题名而清单没改，报错就会指着一个不存在的键
  // `grade_risk.risk` 有**两档**，所以它列出的是带下标的两个键 ——
  // 裸名字只改第 0 档（见 `compilePolicy` 里那段说明），列一个裸名字
  // 会让人以为改了全部
  assert.deepEqual(gateKeys(), [
    'can_deliver.deliverable',
    'can_deliver.unsupported',
    'grade_risk.needs_auth',
    'grade_risk.risk[0]',
    'grade_risk.risk[1]',
    'is_done.done',
    'needs_tool.needs_tool',
    'pick_input.file',
    'pick_tool.tool',
    'step_ok.ok',
  ])
})

// ═══════════════════════════════════════════════════════════
// 覆盖真的改变了判定
// ═══════════════════════════════════════════════════════════

const deliverAnswers = (unsupported: number, deliverable: number) => ({
  unsupported: { type: 'noul' as const, noul: unsupported },
  deliverable: { type: 'noul' as const, noul: deliverable },
})

test('默认门限：unsupported=0.6 会被拦下（>= 0.5 → revise）', () => {
  const out = resolvePolicy(buildDecisions().canDeliver.policy!, deliverAnswers(0.6, 0.9))
  assert.equal(out.action, 'revise')
})

test('覆盖成 0.7 之后同一个答案放行了 —— 覆盖**真的**在起作用', () => {
  const d = buildDecisions({ 'can_deliver.unsupported': 0.7 })
  // 0.6 < 0.7 → 第一条不命中；deliverable 0.9 >= 0.6 → deliver
  const out = resolvePolicy(d.canDeliver.policy!, deliverAnswers(0.6, 0.9))
  assert.equal(out.action, 'deliver')
})

test('★ reason 里写的是**生效**的门限，不是文件里那个', () => {
  // 这条是「留痕」的核心：日志、以及交付闸门给模型看的那句话，都从 reason 来。
  // 写着 0.5 而实际用 0.7，是查起来最费劲的一类错 —— 数字看着很正常。
  const d = buildDecisions({ 'can_deliver.unsupported': 0.7 })
  const reasons = d.canDeliver.policy!.map((r) => r.reason ?? '')
  assert.ok(
    reasons.some((r) => r.includes('prob:unsupported >= 0.7')),
    `第一条 reason 应当是生效值：${JSON.stringify(reasons)}`,
  )
  assert.ok(!reasons.some((r) => r.includes('>= 0.5')), '不该还留着默认值')
  // 没被覆盖的那条**一个字都不变**
  assert.ok(reasons.some((r) => r === 'prob:deliverable >= 0.6 → deliver'))
})

test('不传覆盖 = 和模块级那份常量完全一致', () => {
  const d = buildDecisions()
  const reasons = d.stepOk.policy!.map((r) => r.reason ?? '')
  assert.deepEqual(reasons, ['prob:ok >= 0.6 → continue', 'else → stop'])
})

// ═══════════════════════════════════════════════════════════
// 写错了必须响
// ═══════════════════════════════════════════════════════════

test('块名写错：抛出，并列出有哪些块', () => {
  assert.throws(
    () => buildDecisions({ 'can_delivr.unsupported': 0.7 }),
    (err: Error) => {
      assert.match(err.message, /can_delivr/)
      assert.match(err.message, /can_deliver/, '要告诉人正确的名字怎么写')
      return true
    },
  )
})

test('问题名写错：抛出，并列出那个块有哪些问题', () => {
  assert.throws(
    () => buildDecisions({ 'can_deliver.unsupport': 0.7 }),
    (err: Error) => {
      assert.match(err.message, /unsupport/)
      assert.match(err.message, /unsupported/, '要列出正确的那个')
      return true
    },
  )
})

test('覆盖一个不带数值的东西（`else`）：抛出，并说清它不是问题', () => {
  // 「名字指不到任何一条带门限的规则」也算写错 —— 写的人以为这里有道闸门。
  // 报的是「没有 else 这个问题」，因为 `else` 确实不在问题表里，那句话比
  // 泛泛的「这个键没用」更指得准。
  assert.throws(
    () => buildDecisions({ 'can_deliver.deliverable': 0.9, 'can_deliver.else': 0.5 }),
    (err: Error) => {
      assert.match(err.message, /没有 'else' 这个问题/)
      assert.match(err.message, /deliverable、unsupported/, '要列出那个块真正有的问题')
      return true
    },
  )
})

test('★ 覆盖把多档门限压成不可达：抛出，并说清少的是哪一道闸门', () => {
  // `grade_risk` 是 `score:risk >= 2 → ask_human` 然后 `>= 1 → auto_audit`。
  // 把第 0 档降到 0.5 就**低于**第 1 档了，第二条于是永远不会被求值 ——
  // `resolvePolicy` 顺序求值、第一条命中就返回。这个 agent 就少了一道
  // 「中等风险也要审计」的闸门，而一个错都不报。
  assert.throws(
    () => buildDecisions({ 'grade_risk.risk': 0.5 }),
    (err: Error) => {
      assert.match(err.message, /auto_audit/, '要说清是哪条规则不可达')
      assert.match(err.message, /永远不会被求值/)
      return true
    },
  )
  // 反过来把**第 1 档**抬到第 0 档之上，同样是不可达
  assert.throws(() => buildDecisions({ 'grade_risk.risk[1]': 5 }), /永远不会被求值/)
})

test('多档问题：裸名字只改第 0 档，其余档位用下标显式指定', () => {
  // ★ 这条是「这个键到底能不能用」的判据。第一版让裸名字作用于**所有**档，
  //   而那必然把它们压成相等 → 永远抛错 → `grade_risk.risk` **根本没法用**，
  //   人调不了风险门限。写测试时当场发现。
  const bare = buildDecisions({ 'grade_risk.risk': 4 })
  assert.deepEqual(
    bare.gradeRisk.policy!.map((r) => r.reason),
    [
      'score:risk >= 4 → ask_human',
      'prob:needs_auth >= 0.5 → ask_human',
      'score:risk >= 1 → auto_audit',
      'else → auto',
    ],
    '第 0 档改了，第 1 档原样不动',
  )

  const indexed = buildDecisions({ 'grade_risk.risk[1]': 1.5 })
  assert.deepEqual(
    indexed.gradeRisk.policy!.map((r) => r.reason),
    [
      'score:risk >= 2 → ask_human',
      'prob:needs_auth >= 0.5 → ask_human',
      'score:risk >= 1.5 → auto_audit',
      'else → auto',
    ],
    '下标指名第 1 档，第 0 档不动',
  )
})

test('档位下标越界：抛出，并说清有几档', () => {
  assert.throws(() => buildDecisions({ 'grade_risk.risk[7]': 1 }), /只有 2 档/)
})

test('单档问题写 `[0]` 也认 —— 无歧义，没必要拒绝', () => {
  // `gateKeys()` 给单档问题列的是裸名字，但多写一个 `[0]` 意思完全一样。
  const d = buildDecisions({ 'can_deliver.unsupported[0]': 0.7 })
  assert.ok(d.canDeliver.policy!.some((r) => r.reason === 'prob:unsupported >= 0.7 → revise'))
})

test('resolveGates：一次做完解析 + 校验，能用的直接给回来', () => {
  assert.deepEqual(resolveGates('can_deliver.unsupported=0.7'), { 'can_deliver.unsupported': 0.7 })
  assert.deepEqual(resolveGates(''), {})
  assert.throws(() => resolveGates('乱写'), /解析不了/)
  assert.throws(() => resolveGates('can_delivr.unsupported=0.7'), /没有 'can_delivr' 这个块/)
})
