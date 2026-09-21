/**
 * 会话折叠的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里全是运行时行为，`tsc` 通过不代表行为正确。重点测三类：
 *
 *   · **留尾那几轮必须逐字不变** —— 这是整个折叠的落点。指代（"那个文件"）
 *     指向紧邻的上一轮，折掉它等于把这句话的落点抽走，而摘要里的截断文本
 *     救不回来（用户问的往往正是细节）。
 *   · **折叠必须真的变小** —— 见 `surface.ts` 模块头第 3 条不变量。
 *     摘要比原文还大时不折，而不是把问题推给下一次。
 *   · **压不动要说出来** —— 留尾是**故意不折**的，所以超线可能是正确行为。
 *     不报出来的话，「压不动」和「不该再压」就分不开了（§8.10）。
 *
 * @module JevLoop/conversation.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  foldConversation,
  CONVERSATION_POLICY,
  CONVERSATION_KEEP_TAIL,
  CONVERSATION_RETAIN_CHARS,
  CONVERSATION_TRIGGER_CHARS,
  type ConversationPolicy,
  type ConversationTurn,
} from '../src/conversation.ts'

/** 造一批轮次：问句很短、答句指定长度，便于算账 */
function turnsOf(n: number, answerChars: number): ConversationTurn[] {
  return Array.from({ length: n }, (_, i) => ({
    task: `问题${i}`,
    answer: 'x'.repeat(answerChars),
  }))
}

/** 小策略，让算术在测试里看得见，不用去猜默认值的量级 */
const SMALL: ConversationPolicy = {
  triggerChars: 100,
  retainChars: 30,
  digestTaskChars: 10,
  digestAnswerChars: 10,
}

// ═══════════════════════════════════════════════════════════
// 第一档：没到触发线就什么都不做
// ═══════════════════════════════════════════════════════════

test('没到触发线时原样返回，一个字节都不动', () => {
  const turns = turnsOf(2, 5)
  const r = foldConversation(turns, CONVERSATION_POLICY)
  assert.deepEqual(r.recent, turns)
  assert.equal(r.digest, '')
  assert.equal(r.report.acted, false)
  assert.equal(r.report.foldedTurns, 0)
  assert.equal(r.report.rawChars, r.report.keptChars)
})

test('压缩本身有成本 —— 没必要时不许动，哪怕策略线画得很低', () => {
  // 5 轮 × 6 字符 = 30 字符，低于 trigger 100
  const r = foldConversation(turnsOf(5, 5), SMALL)
  assert.equal(r.report.acted, false)
  assert.equal(r.digest, '')
  assert.equal(r.recent.length, 5)
})

// ═══════════════════════════════════════════════════════════
// 第二档：从最老的开始折，但**永远留尾**
// ═══════════════════════════════════════════════════════════

test('超线时折掉最老的，留尾那几轮逐字不变', () => {
  const turns = turnsOf(6, 60)
  const r = foldConversation(turns, SMALL)

  assert.equal(r.report.acted, true)
  assert.equal(r.report.foldedTurns > 0, true)

  // ★ 最重要的一条：留尾那几轮**逐字**等于输入的最后几轮
  const tail = turns.slice(turns.length - r.recent.length)
  assert.deepEqual(r.recent, tail)
  assert.equal(r.recent.length >= CONVERSATION_KEEP_TAIL, true)
})

test('留尾轮数是下限：折到只剩它就该停手', () => {
  // 内容足够长，压到 retain 需要折掉远多于「全部减留尾」的轮数
  const turns = turnsOf(8, 500)
  const r = foldConversation(turns, { ...SMALL, retainChars: 1 })
  assert.equal(r.recent.length, CONVERSATION_KEEP_TAIL)
  // 到这一步就该停 —— 再折就把指代的落点抽走了
  assert.equal(r.report.foldedTurns, turns.length - CONVERSATION_KEEP_TAIL)
})

test('轮数不比留尾多时没有可折的，且不算失败', () => {
  for (const n of [0, 1, 2]) {
    const r = foldConversation(turnsOf(n, 200), { ...SMALL, triggerChars: 10, retainChars: 5 })
    assert.equal(r.recent.length, n)
    assert.equal(r.digest, '')
    assert.equal(r.report.acted, false)
    assert.equal(r.report.rawTurns, n)
  }
})

test('recent 保持时间顺序（最老的在前）—— 生成器按这个顺序铺开', () => {
  const turns: ConversationTurn[] = [
    { task: 'A', answer: 'a'.repeat(60) },
    { task: 'B', answer: 'b'.repeat(60) },
    { task: 'C', answer: 'c'.repeat(60) },
    { task: 'D', answer: 'd'.repeat(60) },
  ]
  const r = foldConversation(turns, SMALL)
  assert.deepEqual(
    r.recent.map((t) => t.task),
    ['C', 'D'],
  )
})

// ═══════════════════════════════════════════════════════════
// 摘要：说清几轮、问过什么、答出来什么
// ═══════════════════════════════════════════════════════════

test('摘要说清了几轮、并且明说原文不在这次请求里', () => {
  const r = foldConversation(turnsOf(6, 60), SMALL)
  assert.equal(r.report.foldedTurns, 4)
  assert.match(r.digest, new RegExp(`更早的 ${r.report.foldedTurns} 轮已折叠`))
  // 这句话必须存在：不说的话，模型会以为自己看到了完整对话（§8.10）
  assert.match(r.digest, /完整原文不在这次请求里/)
})

test('摘要大小有上界 —— 折得越多摘要越长的话，折叠会自己把自己废掉', () => {
  // 逐轮罗列的话这里是 500 × 约 200 字符 ≈ 十万字符的摘要，
  // 而「折了就必须变小」那条不变量会拦住它 → 永远折不动，预算形同虚设
  const sizes = [5, 20, 100, 500].map(
    (n) => foldConversation(turnsOf(n, 2000), { ...SMALL, triggerChars: 100, retainChars: 50 }).digest.length,
  )
  assert.equal(Math.max(...sizes) < 1_200, true, `摘要必须封顶，实测 ${sizes.join(' / ')}`)
  // 而且封顶之后不能再随轮数涨
  assert.equal(sizes[3]! - sizes[2]! < 50, true, '500 轮和 100 轮的摘要不该差出量级')
})

test('摘要封顶时明说「只列最近几轮」—— 截了不说是 §8.10 禁止的', () => {
  const r = foldConversation(turnsOf(40, 800), { ...SMALL, triggerChars: 100, retainChars: 50 })
  assert.equal(r.report.foldedTurns > 4, true)
  assert.match(r.digest, /这里只列最近 4 轮/)
})

/** 摘要里的字段留宽一点，好让问句里的标识符（文件名、函数名）完整可见 */
const ROOMY: ConversationPolicy = {
  triggerChars: 100,
  retainChars: 30,
  digestTaskChars: 30,
  digestAnswerChars: 30,
}

test('摘要里留下每一轮的问与答 —— 指代就靠这个', () => {
  const turns: ConversationTurn[] = [
    { task: '列出工作目录里的文件', answer: '目录里有 invoice.ts 和 notes.md' },
    { task: '读取 invoice.ts', answer: '它定义了 outstanding()' },
    { task: '第三个问题', answer: 'y'.repeat(200) },
    { task: '第四个问题', answer: 'y'.repeat(200) },
    { task: '第五个问题', answer: 'y'.repeat(200) },
  ]
  const r = foldConversation(turns, ROOMY)
  assert.equal(r.report.acted, true)
  assert.match(r.digest, /列出工作目录里的文件/)
  // 文件名必须完整活下来：它正是「再读一遍那个文件」要落地的东西
  assert.match(r.digest, /invoice\.ts/)
})

test('摘要里的换行必须压平 —— 否则编号列表会被冲散', () => {
  const turns: ConversationTurn[] = [
    { task: '第一问', answer: '第一行\n第二行\n第三行' },
    { task: '第二问', answer: 'y'.repeat(200) },
    { task: '第三问', answer: 'y'.repeat(200) },
    { task: '第四问', answer: 'y'.repeat(200) },
    { task: '第五问', answer: 'y'.repeat(200) },
  ]
  const r = foldConversation(turns, ROOMY)
  assert.equal(r.report.foldedTurns, 3)
  // 抬头 + 3 条 = 4 行。答案里的换行若没压平，这里会多出 2 行，
  // 而「第 3 轮」看起来就成了「第 2 轮」的续行
  assert.equal(r.digest.split('\n').length, 1 + r.report.foldedTurns)
  assert.match(r.digest, /第一行 第二行 第三行/)
})

test('摘要截断时标出被截了多少（不假装那是全部）', () => {
  const turns: ConversationTurn[] = [
    { task: '问', answer: 'A'.repeat(200) },
    { task: '问2', answer: 'B'.repeat(80) },
    { task: '问3', answer: 'C'.repeat(80) },
  ]
  const r = foldConversation(turns, { ...SMALL, triggerChars: 40, retainChars: 20 })
  assert.match(r.digest, /…\[\+\d+\]/)
})

// ═══════════════════════════════════════════════════════════
// 账目：压不动要说出来
// ═══════════════════════════════════════════════════════════

test('折完仍超目标线时如实报 overRetain', () => {
  // retain 比留尾那两轮本身还小 → 无论怎么折都到不了
  const r = foldConversation(turnsOf(8, 500), { ...SMALL, retainChars: 1 })
  assert.equal(r.report.overRetain, true)
  assert.equal(r.report.keptChars > r.report.retainChars, true)
})

test('目标线在地板之上时才能真正压下去', () => {
  // 目标线必须高过「留尾那几轮 + 摘要自身」这块**故意不折**的地板，
  // 否则它永远达不成，而「达不成」和「压不动」在账面上长得一样
  const r = foldConversation(turnsOf(8, 60), { ...SMALL, triggerChars: 500, retainChars: 400 })
  assert.equal(r.report.acted, true)
  assert.equal(r.report.overRetain, false)
  assert.equal(r.report.keptChars <= r.report.retainChars, true)
})

test('账目自洽：keptChars 就是逐字那几轮加摘要的实际长度', () => {
  const r = foldConversation(turnsOf(8, 60), SMALL)
  const verbatim = r.recent.reduce((n, t) => n + t.task.length + 1 + t.answer.length, 0)
  assert.equal(r.report.keptChars, verbatim + r.digest.length)
})

test('keptTurns + foldedTurns 必须等于 rawTurns —— 一轮都不能凭空消失', () => {
  for (const n of [0, 1, 3, 6, 12]) {
    const r = foldConversation(turnsOf(n, 60), SMALL)
    assert.equal(r.report.keptTurns + r.report.foldedTurns, r.report.rawTurns)
  }
})

test('折叠必须真的变小 —— acted 等价于「真的小了」', () => {
  // 摘要每轮有固定开销（抬头、编号、箭头），所以原文很碎时折了反而更大。
  // 这时宁可不折（不变量第 3 条），而账面上必须**看得出来没折**。
  const tiny: ConversationTurn[] = Array.from({ length: 30 }, (_, i) => ({ task: `${i}`, answer: '' }))
  for (const p of [
    { triggerChars: 2, retainChars: 1, digestTaskChars: 40, digestAnswerChars: 40 },
    { ...SMALL },
    { ...SMALL, triggerChars: 500, retainChars: 400 },
  ]) {
    const r = foldConversation(tiny, p)
    assert.equal(r.report.acted, r.report.keptChars < r.report.rawChars, 'acted 必须等价于「真的变小了」')
    if (!r.report.acted) assert.equal(r.digest, '', '没折就不该有摘要')
  }
})

// ═══════════════════════════════════════════════════════════
// 策略自检：自相矛盾的配置必须当场炸
// ═══════════════════════════════════════════════════════════

test('retainChars >= triggerChars 必须抛 —— 否则每轮都在折且永远折不到位', () => {
  assert.throws(
    () => foldConversation(turnsOf(3, 90), { ...SMALL, triggerChars: 100, retainChars: 100 }),
    /retainChars .*必须小于 triggerChars/,
  )
})

test('预算不是正整数必须抛', () => {
  assert.throws(() => foldConversation(turnsOf(3, 90), { ...SMALL, triggerChars: 0 }), /必须是 ≥ 1 的整数/)
  assert.throws(() => foldConversation(turnsOf(3, 90), { ...SMALL, retainChars: 1.5 }), /必须是 ≥ 1 的整数/)
  assert.throws(() => foldConversation(turnsOf(3, 90), { ...SMALL, digestTaskChars: -1 }), /必须是 ≥ 1 的整数/)
})

// ═══════════════════════════════════════════════════════════
// 默认策略
// ═══════════════════════════════════════════════════════════

test('默认策略：正常长度的会话根本碰不到触发线', () => {
  // 实测一轮问答约 200–400 字符；12 轮（服务端 MAX_TURNS）约 2400–4800
  const r = foldConversation(turnsOf(12, 300), CONVERSATION_POLICY)
  assert.equal(r.report.acted, false, '12 轮正常长度不该触发压缩')
  assert.equal(CONVERSATION_TRIGGER_CHARS > 12 * 300, true)
})

test('默认策略：长回答的会话会被折，且折到目标线以下', () => {
  // 每轮 1200 字符 × 12 轮 ≈ 14400，超过触发线 9600
  const r = foldConversation(turnsOf(12, 1200), CONVERSATION_POLICY)
  assert.equal(r.report.acted, true)
  assert.equal(r.report.keptChars <= CONVERSATION_RETAIN_CHARS, true)
  assert.equal(r.recent.length, CONVERSATION_KEEP_TAIL)
})

test('默认策略的触发线是目标线的 4 倍 —— 和证据那一对同比例', () => {
  assert.equal(CONVERSATION_TRIGGER_CHARS / CONVERSATION_RETAIN_CHARS, 4)
})

test('不传策略时用默认值（和显式传 CONVERSATION_POLICY 等价）', () => {
  const turns = turnsOf(12, 1200)
  assert.deepEqual(foldConversation(turns).report, foldConversation(turns, CONVERSATION_POLICY).report)
})
