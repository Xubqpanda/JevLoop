/**
 * 表面折叠的机制本身。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里测的不是「折叠压掉了多少」，而是**表面和日志不会悄悄分叉**。
 * 那是这套机制唯一真正的失败模式：折叠本身压不掉东西只是没效果，
 * 而范围对不上却会让人以为某几步还在、其实已经被换掉了。
 *
 * @module JevLoop/surface.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  planAppend,
  planReplace,
  commitSurface,
  surfaceChars,
  foldSeq,
  type SurfaceNode,
} from '../src/surface.ts'

const nodes = (...texts: string[]): SurfaceNode[] => texts.map((text, i) => ({ seq: i, text }))

// ═══════════════════════════════════════════════════════════
// 追加
// ═══════════════════════════════════════════════════════════

test('追加：接在末尾，delta 为正', () => {
  const n = nodes('aa')
  const plan = planAppend(1, 'bbb')
  assert.equal(plan.target, 'append')
  assert.equal(plan.deltaChars, 3)
  commitSurface(n, plan)
  assert.equal(n.length, 2)
  assert.equal(n[1]!.text, 'bbb')
})

// ═══════════════════════════════════════════════════════════
// 替换：范围必须真实存在
// ═══════════════════════════════════════════════════════════

test('替换：把一段换成一个节点，delta 是净变化', () => {
  const n = nodes('a'.repeat(10), 'b'.repeat(20), 'c'.repeat(30))
  const plan = planReplace(n, 0, 1, { seq: 0, text: 'x'.repeat(5) })
  assert.deepEqual(plan.target, { startIdx: 0, endIdx: 1 })
  assert.equal(plan.deltaChars, 5 - 30)
  commitSurface(n, plan)
  assert.equal(n.length, 2)
  assert.equal(n[0]!.text, 'x'.repeat(5))
  assert.equal(n[1]!.text, 'c'.repeat(30), '后面的节点不受影响')
})

test('★ 范围不存在时必须抛 —— 跳过会让表面和日志悄悄分叉', () => {
  const n = nodes('a', 'b')
  // 越界
  assert.throws(() => planReplace(n, 0, 5, { seq: 0, text: 'x' }), /不在表面上/)
  // 起点越界
  assert.throws(() => planReplace(n, 9, 9, { seq: 0, text: 'x' }), /不在表面上/)
  // 反序
  assert.throws(() => planReplace(n, 1, 0, { seq: 0, text: 'x' }), /不在表面上/)
  // 负数
  assert.throws(() => planReplace(n, -1, 0, { seq: 0, text: 'x' }), /不在表面上/)
  // 非整数
  assert.throws(() => planReplace(n, 0.5, 1, { seq: 0, text: 'x' }), /不在表面上/)
})

test('★ 规划失败时表面一个字节都没动（plan/commit 分开的全部意义）', () => {
  const n = nodes('a', 'b', 'c')
  const before = JSON.stringify(n)
  try {
    planReplace(n, 0, 99, { seq: 0, text: 'x' })
  } catch {
    /* 这里就是要它抛 */
  }
  assert.equal(JSON.stringify(n), before, '失败的规划不该留下改了一半的表面')
})

test('替换整段只剩一个节点也是合法的', () => {
  const n = nodes('a', 'b', 'c')
  commitSurface(n, planReplace(n, 0, 2, { seq: 0, text: 'S' }))
  assert.equal(n.length, 1)
  assert.equal(n[0]!.text, 'S')
})

// ═══════════════════════════════════════════════════════════
// 记账用的两个纯函数
// ═══════════════════════════════════════════════════════════

test('surfaceChars 是所有节点的字符和', () => {
  assert.equal(surfaceChars(nodes('abc', 'de')), 5)
  assert.equal(surfaceChars([]), 0)
})

test('foldSeq 用被折那段的第一个 seq 当摘要的锚', () => {
  const n: SurfaceNode[] = [
    { seq: 10, text: 'a' },
    { seq: 11, text: 'b' },
    { seq: 12, text: 'c' },
  ]
  assert.equal(foldSeq(n, 0), 10)
  assert.equal(foldSeq(n, 2), 12)
})

test('一连串操作之后表面仍然自洽（下标始终有效）', () => {
  // 模拟真实用法：追加几条，折掉最老的一段，再追加
  const n: SurfaceNode[] = []
  for (let i = 0; i < 6; i++) commitSurface(n, planAppend(i, `step${i}`.repeat(10)))
  assert.equal(n.length, 6)

  commitSurface(n, planReplace(n, 0, 3, { seq: 0, text: '[折了 4 步]' }))
  assert.equal(n.length, 3)

  commitSurface(n, planAppend(6, 'step6'))
  assert.equal(n.length, 4)
  assert.equal(n[0]!.text, '[折了 4 步]')
  assert.equal(n[3]!.text, 'step6')
  // 锚仍然是日志序号，不是下标 —— 折过之后下标变了，seq 没变
  assert.deepEqual(
    n.map((x) => x.seq),
    [0, 4, 5, 6],
  )
})
