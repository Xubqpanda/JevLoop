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
  foldEvidence,
  resolveEvidencePolicy,
  EVIDENCE_POLICY,
  priceGenerateRequest,
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

test('★ 目标线必须容得下单条结果裁剪后的最大体积', () => {
  // 实测发现的：默认值原本是 retain 4800，而单条裁剪后最大
  // 4096 + 标记 58 + 1024 = 5178 —— 于是「压到只剩一条」时永远报 overRetain，
  // 而那是**不可能满足**的：最后一条不会被丢，它自己就比目标大。
  const mismatch = EVIDENCE_POLICY.retainChars < PRUNE_DEFAULTS.headChars + 58 + PRUNE_DEFAULTS.tailChars
  assert.equal(mismatch, false, '默认的 retainChars 容不下单条，目标线是个摆设')

  // 显式构造一组不容得下的，必须抛 —— 校验在 `foldEvidence` 里，
  // 因为只有那里**策略和裁剪预算两样都知道**
  assert.throws(
    () => foldEvidence(ev('x'.repeat(100)), policy(100_000, 1_000)),
    /小于单条结果裁剪后的最大体积/,
  )
})

test('策略的两个数都必须是整数', () => {
  assert.throws(() => resolveEvidencePolicy({ triggerChars: 0 }), /必须是 ≥ 1 的整数/)
  assert.throws(() => resolveEvidencePolicy({ retainChars: -1 }), /整数/)
})

// ═══════════════════════════════════════════════════════════
// 整段证据：折叠，不是丢弃
// ═══════════════════════════════════════════════════════════

const policy = (triggerChars: number, retainChars: number) => resolveEvidencePolicy({ triggerChars, retainChars })

/** 造几条证据。`label` 模拟调用方给的工具名 —— `context.ts` 不认识工具 */
const ev = (...texts: string[]) => texts.map((text, i) => ({ text, label: `tool${i}(target${i})` }))

test('没到触发线时**什么都不做**，一个字都不动', () => {
  const parts = ev('a'.repeat(100), 'b'.repeat(100))
  const { text, report, folds } = foldEvidence(parts, policy(20_000, 6_000))
  assert.equal(report.acted, false)
  assert.equal(folds.length, 0)
  assert.equal(text, parts.map((p) => p.text).join('\n'))
})

test('刚过触发线只做逐条剪中间，不折整段', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const { text, report, folds } = foldEvidence(
    ev('A'.repeat(500), 'B'.repeat(500)),
    policy(900, 500),
    pruneBud,
  )
  assert.equal(report.prunedCount, 2)
  assert.equal(folds.length, 0, '剪中间就够了，不该折')
  assert.ok(text.includes('A') && text.includes('B'), '两条都还在')
})

test('★ 剪完还超就**折成一个摘要**，而不是丢掉', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const { text, report, folds } = foldEvidence(
    ev('OLD'.repeat(200), 'MID'.repeat(200), 'NEW'.repeat(200)),
    policy(1000, 400),
    pruneBud,
  )
  assert.equal(folds.length, 1, '应当恰好折一段')
  assert.ok(folds[0]!.foldedNodes >= 1)
  assert.equal(report.foldedCount, folds[0]!.foldedNodes)
  assert.ok(text.includes('已折叠'), '摘要里要写明折了几步')
  assert.ok(text.includes('NEW'), '最新的那条必须留下')
  assert.match(text, /原文在轨迹里/, '要说清原文没丢，只是没进请求')
})

test('★ 摘要里带上了「碰过哪些目标」—— 指代要靠它落地', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const parts = [
    { text: 'x'.repeat(500), label: 'list_dir(.)' },
    { text: 'y'.repeat(500), label: 'read_file(invoice.ts)' },
    { text: 'z'.repeat(500), label: 'read_file(retry.ts)' },
    { text: 'w'.repeat(100), label: 'read_file(notes.md)' },
  ]
  const { text } = foldEvidence(parts, policy(800, 300), pruneBud)
  assert.match(text, /已折叠/)
  assert.match(text, /read_file×2|read_file/, '要按工具计数')
  assert.match(text, /invoice\.ts/, '碰过的目标要列出来，否则「那个文件」无处可指')
})

test('★ 最新的一条永远不会被折掉', () => {
  const { text } = foldEvidence(ev('X'.repeat(50_000)), policy(20_000, 6_000))
  assert.ok(text.includes('X'), '唯一一条被折空了')
})

test('★ 压不到目标线时如实报 overRetain，不假装压过了', () => {
  const { report, text } = foldEvidence(ev('Y'.repeat(7000)), policy(6_000, 5_200))
  assert.equal(report.prunedCount, 0, '7000 < 8192，不该被剪')
  assert.equal(report.foldedCount, 0, '只有一条，不该被折')
  assert.equal(report.overRetain, true, '压不下去就必须说')
  assert.match(text, /超过目标/, '账目里要写明')
})

test('动了就要写账（§8.10）', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const { text, report } = foldEvidence(ev('A'.repeat(500), 'B'.repeat(500)), policy(400, 250), pruneBud)
  assert.equal(report.acted, true)
  assert.match(text, /context budget/, '动了却不写账')
})

test('账目里的数对得上', () => {
  const pruneBud = resolvePruneBudget({ thresholdChars: 300, headChars: 100, tailChars: 50 })
  const { text, report } = foldEvidence(ev('A'.repeat(500), 'B'.repeat(500)), policy(900, 500), pruneBud)
  assert.equal(report.rawChars, 1000)
  assert.equal(report.prunedCount, 2)
  assert.equal(report.keptChars, text.length)
  assert.equal(report.triggerChars, 900)
  assert.equal(report.retainChars, 500)
})

// ═══════════════════════════════════════════════════════════
// 请求定价：我们估的 vs provider 数的
//
// 这一组测的是**可核对性**。以前「中文 1 字 ≈ 1 token」只是注释里的一句
// 断言；现在它产出一个数，并且和 provider 报的真值并排出现。
// ═══════════════════════════════════════════════════════════

test('定价把三块分开算，合计等于它们之和', () => {
  const e = priceGenerateRequest({
    task: '读一下那个文件',
    evidence: 'x'.repeat(400),
    history: [
      { task: '列目录', answer: 'a.ts b.ts' },
      { task: '读 a.ts', answer: 'export const x = 1' },
    ],
  })
  assert.equal(e.controlTokens, e.historyTokens + e.taskTokens + e.evidenceTokens)
  assert.ok(e.historyTokens > 0, '两轮历史要有代价')
  assert.ok(e.taskTokens > 0)
  assert.equal(e.evidenceTokens, 100, '400 个英文 x ≈ 100 token')
})

test('没有上文时历史那项是 0，不是 NaN', () => {
  const e = priceGenerateRequest({ task: 'T', evidence: '' })
  assert.equal(e.historyTokens, 0)
  assert.equal(e.evidenceTokens, 0)
  assert.ok(Number.isFinite(e.controlTokens))
})

test('★ 中文比英文贵四倍 —— 这正是不能按英文字符估的原因', () => {
  const zh = priceGenerateRequest({ task: '', evidence: '中'.repeat(400) })
  const en = priceGenerateRequest({ task: '', evidence: 'x'.repeat(400) })
  assert.equal(zh.evidenceTokens, 400, '中文 1 字 ≈ 1 token')
  assert.equal(en.evidenceTokens, 100, '英文 4 字符 ≈ 1 token')
  assert.equal(zh.evidenceTokens, en.evidenceTokens * 4)
})

test('★ 估算随证据单调增长 —— 这是它唯一能被核对的用法', () => {
  // 差值（provider 报的 − 我们估的）里混着 system prompt 和启发式偏差，
  // 两者分不开。所以这个数**不能**用来断言绝对准确度，只能看**趋势**：
  // 证据变大时它必须跟着变大，否则说明预算根本没接上。
  const hist = [{ task: '列目录', answer: 'a.ts' }]
  const sizes = [0, 1000, 5000, 20000]
  const est = sizes.map(
    (n) => priceGenerateRequest({ task: 'T', evidence: 'x'.repeat(n), history: hist }).controlTokens,
  )
  for (let i = 1; i < est.length; i++) {
    assert.ok(est[i]! > est[i - 1]!, `证据从 ${sizes[i - 1]} 涨到 ${sizes[i]} 时估算没跟着涨`)
  }
})
