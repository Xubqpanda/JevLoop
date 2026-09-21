/**
 * 上下文预算的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里全是运行时行为，`tsc` 通过不代表行为正确。重点测两类：
 *
 *   · **裁剪必须真的变小** —— DSH 的配置期校验就是为了这个。
 *     不校验的话「阈值 100 / 头 80 / 尾 40」会让文本越裁越长，而且不报错。
 *   · **丢了什么要说出来** —— 一个只写「已省略」的标记会让模型
 *     和自己都以为看到的是全部（§8.10）。
 *
 * @module JevLoop/context.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  PRUNE_DEFAULTS,
  PRUNE_MARKER_PREFIX,
  resolvePruneBudget,
  pruneToolResult,
  fitEvidence,
  resolveEvidencePolicy,
  EVIDENCE_POLICY,
} from '../src/context.ts'

const len = (s: string) => Array.from(s).length

// ═══════════════════════════════════════════════════════════
// 预算校验（从 DSH 抄来的那条）
// ═══════════════════════════════════════════════════════════

test('预算自洽时通过，返回解析后的值', () => {
  assert.deepEqual(resolvePruneBudget(), PRUNE_DEFAULTS)
  assert.deepEqual(resolvePruneBudget({ thresholdChars: 1000, headChars: 500, tailChars: 100 }), {
    thresholdChars: 1000,
    headChars: 500,
    tailChars: 100,
  })
})

test('head + 标记 + tail 超过阈值时必须抛 —— 否则会越裁越长', () => {
  assert.throws(
    () => resolvePruneBudget({ thresholdChars: 100, headChars: 80, tailChars: 40 }),
    /必须 ≤ thresholdChars/,
  )
})

test('预算必须是整数，阈值必须为正', () => {
  assert.throws(() => resolvePruneBudget({ thresholdChars: 0 }), /必须是 ≥ 1 的整数/)
  assert.throws(() => resolvePruneBudget({ thresholdChars: 10.5 }), /整数/)
  assert.throws(() => resolvePruneBudget({ headChars: -1 }), /整数/)
})

// ═══════════════════════════════════════════════════════════
// 裁剪本身
// ═══════════════════════════════════════════════════════════

test('没超阈值就原样返回，一个字都不动', () => {
  const short = 'x'.repeat(100)
  const r = pruneToolResult(short)
  assert.equal(r.text, short)
  assert.equal(r.pruned, false)
  assert.equal(r.removedChars, 0)
})

test('刚好等于阈值不剪（边界是「超过」而不是「达到」）', () => {
  const exact = 'x'.repeat(PRUNE_DEFAULTS.thresholdChars)
  assert.equal(pruneToolResult(exact).pruned, false)
})

test('超阈值时留头留尾，标记里写明剪掉多少', () => {
  const head = 'H'.repeat(PRUNE_DEFAULTS.headChars)
  const middle = 'M'.repeat(5000)
  const tail = 'T'.repeat(PRUNE_DEFAULTS.tailChars)
  const r = pruneToolResult(head + middle + tail)

  assert.equal(r.pruned, true)
  assert.ok(r.text.startsWith(head), '开头必须是原文的头')
  assert.ok(r.text.endsWith(tail), '结尾必须是原文的尾')
  assert.ok(r.text.includes(PRUNE_MARKER_PREFIX), '必须有标记')
  assert.ok(r.text.includes(String(r.removedChars)), '标记里要写明剪了多少字符')
  assert.equal(r.removedChars, 5000)
  // ★ 这条是核心：裁剪必须真的变小
  assert.ok(len(r.text) < len(head + middle + tail), '裁剪后必须比原来短')
  assert.ok(len(r.text) <= PRUNE_DEFAULTS.thresholdChars, '裁剪后必须落回阈值以内')
})

test('tailChars 为 0 时不保留尾部，也不出错', () => {
  const budget = resolvePruneBudget({ thresholdChars: 500, headChars: 400, tailChars: 0 })
  const r = pruneToolResult('A'.repeat(400) + 'B'.repeat(400), budget)
  assert.equal(r.pruned, true)
  assert.ok(!r.text.endsWith('B'), 'tail 为 0 就不该有尾')
})

test('按码点切，不会把中文或 emoji 劈成半个', () => {
  // 每个字符都是代理对，按 UTF-16 单元切会切出孤立代理
  const text = '𠀀'.repeat(100) + 'x'.repeat(9000)
  const r = pruneToolResult(text)
  assert.equal(r.pruned, true)
  // 孤立代理的检测：编码再解码，出现 U+FFFD 就说明劈开了
  assert.ok(!r.text.includes('\uFFFD'), '切出了孤立代理')
  for (const ch of r.text) {
    assert.ok(ch.codePointAt(0) !== 0xfffd)
  }
})

test('★ 预算自洽 ⇒ 裁剪一定变小（这条不变量就是那道校验的意义）', () => {
  // 遍历一批**合法**预算 × 一批刚过阈值的长度，裁剪后必须严格更短。
  // 如果不成立，说明校验没兜住 —— 那正是「上下文越裁越大」的入口。
  const budgets = [
    { thresholdChars: 600, headChars: 500, tailChars: 20 },
    { thresholdChars: 200, headChars: 100, tailChars: 20 },
    // 阈值不可能小于标记本身（58 字符），所以 0/0 那组的最小合法阈值是 58
    { thresholdChars: 58, headChars: 0, tailChars: 0 },
    { thresholdChars: 1000, headChars: 1, tailChars: 1 },
  ]
  for (const raw of budgets) {
    const budget = resolvePruneBudget(raw)
    for (const n of [raw.thresholdChars + 1, raw.thresholdChars * 3, raw.thresholdChars * 50]) {
      const text = 'Q'.repeat(n)
      const r = pruneToolResult(text, budget)
      assert.equal(r.pruned, true, `len=${n} budget=${raw.thresholdChars} 应当被剪`)
      assert.ok(len(r.text) < n, `len=${n} budget=${raw.thresholdChars}：裁剪后没变短`)
      assert.ok(len(r.text) <= budget.thresholdChars, `len=${n}：裁剪后还在阈值外`)
    }
  }
})

// ═══════════════════════════════════════════════════════════
// 双阈值策略（从 DSH 抄的那条校验）
// ═══════════════════════════════════════════════════════════

test('默认策略可用，且 retain 严格小于 trigger', () => {
  assert.deepEqual(resolveEvidencePolicy(), EVIDENCE_POLICY)
  assert.ok(EVIDENCE_POLICY.retainChars < EVIDENCE_POLICY.triggerChars)
})

test('retain ≥ trigger 时必须抛 —— 否则会陷入「每轮都压、永远压不下去」', () => {
  assert.throws(() => resolveEvidencePolicy({ triggerChars: 100, retainChars: 100 }), /必须 < triggerChars/)
  assert.throws(() => resolveEvidencePolicy({ triggerChars: 100, retainChars: 200 }), /必须 < triggerChars/)
})

test('策略的两个数都必须是整数', () => {
  assert.throws(() => resolveEvidencePolicy({ triggerChars: 0 }), /必须是 ≥ 1 的整数/)
  assert.throws(() => resolveEvidencePolicy({ retainChars: -1 }), /整数/)
})

// ═══════════════════════════════════════════════════════════
// 整段 evidence
// ═══════════════════════════════════════════════════════════

const policy = (triggerChars: number, retainChars: number) => resolveEvidencePolicy({ triggerChars, retainChars })

test('没到触发线时**什么都不做**，一个字都不动', () => {
  const parts = ['a'.repeat(100), 'b'.repeat(100)]
  const { text, report } = fitEvidence(parts, policy(1000, 100))
  assert.equal(report.acted, false)
  assert.equal(report.prunedCount, 0)
  assert.equal(report.droppedCount, 0)
  assert.equal(text, parts.join('\n'))
})

test('刚过触发线只做逐条剪中间，不丢整条', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const parts = ['A'.repeat(500), 'B'.repeat(500)]
  // 触发 900（1000 > 900，会动手）；目标 500（剪完约 404，落得进去，所以不丢整条）
  const { text, report } = fitEvidence(parts, policy(900, 500), pruneBud)
  assert.equal(report.prunedCount, 2)
  assert.equal(report.droppedCount, 0)
  assert.ok(text.includes('A'), '整条不该被丢')
  assert.ok(text.includes('B'))
})

test('剪完还超目标线，就从**最老的整条**丢', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const parts = ['OLD'.repeat(200), 'MID'.repeat(200), 'NEW'.repeat(200)]
  const { text, report } = fitEvidence(parts, policy(1000, 400), pruneBud)

  assert.ok(report.droppedCount > 0, '应当丢了整条')
  assert.ok(!text.includes('OLD'), '最老的应当先被丢')
  assert.ok(text.includes('NEW'), '最新的必须留下')
})

test('★ 最新的一条永远不会被丢空', () => {
  const { text } = fitEvidence(['X'.repeat(50_000)], policy(100, 10))
  assert.ok(text.includes('X'), '唯一一条被丢空了')
  assert.ok(len(text) < 50_000, '但它应当被剪过')
})

test('★ 压不到目标线时如实报 overRetain，不假装压过了', () => {
  // 只剩一条且它自己就超过目标线 —— 压不下去，必须说出来
  const { report, text } = fitEvidence(['Y'.repeat(50_000)], policy(100, 10))
  assert.equal(report.overRetain, true, '压不下去就必须说')
  assert.match(text, /超过目标/, '账目里要写明')
})

test('丢了东西就必须写进账目（§8.10）', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const { text, report } = fitEvidence(['A'.repeat(500), 'B'.repeat(500)], policy(400, 100), pruneBud)
  assert.equal(report.acted, true)
  assert.match(text, /context budget/, '丢了东西却不写账目')
})

test('账目里的数对得上', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const parts = ['A'.repeat(500), 'B'.repeat(500)]
  const { text, report } = fitEvidence(parts, policy(900, 500), pruneBud)
  assert.equal(report.rawChars, 1000)
  assert.equal(report.prunedCount, 2)
  assert.equal(report.keptChars, len(text))
  assert.equal(report.triggerChars, 900)
  assert.equal(report.retainChars, 500)
})
