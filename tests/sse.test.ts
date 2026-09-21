/**
 * SSE 的帧。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 把这些抽成纯函数就是为了下面这些用例 —— 它们在真服务端上**全是偶发的**：
 * 一次 `read()` 正好切在帧中间、`\r\n` 结尾、多行 `data:`、保活注释行。
 * 留在 `HttpGenerator` 的私有方法里，验它们就得起一个假 HTTP 服务，
 * 于是没人会去验（那正是它被抽出来之前的状态）。
 *
 * @module JevLoop/sse.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { encodeSse, decodeSse } from '../src/sse.ts'

test('编一帧：一行 data，空行结尾', () => {
  assert.equal(encodeSse({ a: 1 }), 'data: {"a":1}\n\n')
})

test('编了解得回来', () => {
  const msg = { type: 'x', text: '含"引号"' }
  const { payloads, rest } = decodeSse(encodeSse(msg))
  assert.deepEqual(payloads, [JSON.stringify(msg)])
  assert.equal(rest, '', '整帧吃掉之后不该剩下东西')
})

test('一次读到三帧，三帧都出来', () => {
  const { payloads } = decodeSse(encodeSse({ n: 1 }) + encodeSse({ n: 2 }) + encodeSse({ n: 3 }))
  assert.deepEqual(payloads, ['{"n":1}', '{"n":2}', '{"n":3}'])
})

test('半帧留在 rest 里，接上后半帧才成一条消息', () => {
  // ★ 这是**常态**而不是异常：TCP 怎么切和你写的帧边界没有任何关系。
  const full = encodeSse({ n: 1 })
  const cut = 5
  const a = decodeSse(full.slice(0, cut))
  assert.deepEqual(a.payloads, [], '半帧不该出消息')
  assert.equal(a.rest, full.slice(0, cut), '半帧要原样留着')

  const b = decodeSse(a.rest + full.slice(cut))
  assert.deepEqual(b.payloads, ['{"n":1}'])
  assert.equal(b.rest, '')
})

test('CRLF 结尾也算一帧', () => {
  assert.deepEqual(decodeSse('data: {"n":1}\r\n\r\n').payloads, ['{"n":1}'])
})

test('CRLF 被切在两个 chunk 中间也不丢', () => {
  // 第一段停在孤立的 `\r` 上 —— 那时它还不能被替换掉，要留到下一段
  const a = decodeSse('data: {"n":1}\r')
  assert.deepEqual(a.payloads, [])
  const b = decodeSse(`${a.rest}\n\n`)
  assert.deepEqual(b.payloads, ['{"n":1}'])
})

test('注释行不是消息', () => {
  // 服务端拿它做保活。它不是数据，不该变成一条（会解析失败的）消息。
  assert.deepEqual(decodeSse(': connected\n\n').payloads, [])
  assert.deepEqual(decodeSse(': keep-alive\n\ndata: 甲\n\n').payloads, ['甲'])
})

test('`data:` 后面没有空格也算（规范里空格可选）', () => {
  assert.deepEqual(decodeSse('data:{"n":1}\n\n').payloads, ['{"n":1}'])
})

test('一帧里的多行 data 拼成一条消息', () => {
  assert.deepEqual(decodeSse('data: 甲\ndata: 乙\n\n').payloads, ['甲\n乙'])
})

test('没有 data 行的帧不产生消息', () => {
  assert.deepEqual(decodeSse('event: ping\nid: 7\n\n').payloads, [])
})
