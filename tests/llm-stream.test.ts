/**
 * 流式生成。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * ══════════════════════════════════════════════════════════════
 *  这里起一个**真的 HTTP 服务**，而不是塞一个假的 `Response` 对象。
 *
 *  要验的恰恰是线协议那一层：`stream: true` 有没有真的发出去、
 *  `reset` 是不是每次尝试都发、usage 拿不拿得到、以及**服务端不理睬
 *  `stream` 时会不会坏**。塞假对象只能验「我的 mock 和我的代码一致」。
 * ══════════════════════════════════════════════════════════════
 *
 * @module JevLoop/llm-stream.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'

import { HttpGenerator, RetryingGenerator, ScriptedGenerator } from '../src/llm.ts'
import type { GenDelta } from '../src/events.ts'

/**
 * 一个只服务 `/chat/completions` 的假后端。
 *
 * `handler` 按收到的请求体决定怎么答；`seen` 里留下每次收到的请求体，
 * 用来断言我们**发出去**的东西（不只是我们收回来什么）。
 */
async function backend(handler: (body: any, res: ServerResponse, nth: number) => void) {
  const seen: any[] = []
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let raw = ''
    req.on('data', (c) => (raw += c))
    req.on('end', () => {
      const body = JSON.parse(raw || '{}')
      seen.push(body)
      handler(body, res, seen.length)
    })
  })
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  const { port } = server.address() as AddressInfo
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    seen,
    close: () => new Promise<void>((r) => server.close(() => r())),
  }
}

/** 按 SSE 吐一串载荷，然后收尾 */
function speakSse(res: ServerResponse, payloads: string[]): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8' })
  for (const p of payloads) res.write(`data: ${p}\n\n`)
  res.write('data: [DONE]\n\n')
  res.end()
}

/** 收增量的记录器 */
function recorder() {
  const deltas: GenDelta[] = []
  return { deltas, onDelta: (d: GenDelta) => deltas.push(d) }
}

const chunk = (text: string) => JSON.stringify({ choices: [{ delta: { content: text } }] })

test('给了 onDelta 就请求流式，并要 usage', async (t) => {
  const be = await backend((_b, res) => speakSse(res, [chunk('甲')]))
  t.after(() => be.close())

  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  await g.generate({ task: 't', evidence: 'e', onDelta: () => {} })

  assert.equal(be.seen[0].stream, true, '请求体里要有 stream: true')
  // 不显式要的话，流式的最后一个 chunk 上没有 usage —— 账目会从
  // 「provider 报的真值」悄悄退化成 0（界面上那是「没量到」，§8.10）
  assert.deepEqual(be.seen[0].stream_options, { include_usage: true })
})

test('不给 onDelta 就一个字节都不加（老路径原样）', async (t) => {
  const be = await backend((_b, res) => {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ choices: [{ message: { content: '完整回答' } }], usage: { prompt_tokens: 7, completion_tokens: 3 } }))
  })
  t.after(() => be.close())

  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e' })

  assert.equal('stream' in be.seen[0], false, '不该带 stream 字段')
  assert.equal(r.text, '完整回答')
  assert.equal(r.inputTokens, 7)
  assert.equal(r.outputTokens, 3)
})

test('增量按顺序到，第一段是 reset', async (t) => {
  const be = await backend((_b, res) => speakSse(res, [chunk('你'), chunk('好'), chunk('呀')]))
  t.after(() => be.close())

  const rec = recorder()
  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: rec.onDelta })

  assert.deepEqual(
    rec.deltas,
    [
      { text: '', reset: true },
      { text: '你', reset: false },
      { text: '好', reset: false },
      { text: '呀', reset: false },
    ],
    '第一段必须是 reset（重试/修订要靠它把上一版丢掉）',
  )
  assert.equal(rec.deltas.map((d) => d.text).join(''), '你好呀')
  assert.equal(r.text, '你好呀')
})

test('usage 在最后一个 chunk 上（那时 choices 是空的）', async (t) => {
  const be = await backend((_b, res) =>
    speakSse(res, [
      chunk('答'),
      JSON.stringify({ choices: [], usage: { prompt_tokens: 1234, completion_tokens: 56 } }),
    ]),
  )
  t.after(() => be.close())

  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: () => {} })

  assert.equal(r.text, '答', 'usage 那一帧没有内容，不该被当成文本')
  assert.equal(r.inputTokens, 1234)
  assert.equal(r.outputTokens, 56)
})

test('服务端不理睬 stream 时不报错，照样拿到完整回答', async (t) => {
  // 网关、代理、不认识 stream 字段的服务端都会这样。判据放在**响应**上，
  // 所以这两种情况都走得通（见 HttpGenerator 里那句注释）。
  const be = await backend((_b, res) => {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ choices: [{ message: { content: '我是完整的' } }] }))
  })
  t.after(() => be.close())

  const rec = recorder()
  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: rec.onDelta })

  assert.equal(r.text, '我是完整的')
  assert.deepEqual(rec.deltas, [], '没走流式就一个增量都不该发')
})

test('畸形载荷不致命，后面的内容照样收', async (t) => {
  const be = await backend((_b, res) =>
    speakSse(res, [chunk('前'), '{这不是 JSON', chunk('后')]),
  )
  t.after(() => be.close())

  const rec = recorder()
  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: rec.onDelta })

  assert.equal(r.text, '前后', '一行坏 JSON 不该把整个回答判死')
})

test('[DONE] 不是内容', async (t) => {
  const be = await backend((_b, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    res.write(`data: ${chunk('甲')}\n\n`)
    res.write('data: [DONE]\n\n')
    res.end()
  })
  t.after(() => be.close())

  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: () => {} })
  assert.equal(r.text, '甲')
})

test('一次尝试吐了一半就断，重试再发一个 reset —— 界面据此把半句丢掉', async (t) => {
  // ★ **这才是 reset 存在的那个场景。** 重试和流式并存时必须重置：
  //   不重置的话界面上会是两次尝试的文本首尾相接，而那句话模型从来没说过。
  const be = await backend((_b, res, nth) => {
    if (nth === 1) {
      // 已经开始吐了，然后连接**干净地**结束（不发 [DONE]）。
      // 干净结束在 `reader.read()` 上和「正常读完」长得一模一样 ——
      // 半个回答会被当成完整的交出去。见 HttpGenerator 里那段说明。
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      res.write(`data: ${chunk('半句话')}\n\n`)
      res.end()
      return
    }
    speakSse(res, [chunk('第二次'), chunk('才算数')])
  })
  t.after(() => be.close())

  const rec = recorder()
  const inner = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const g = new RetryingGenerator(inner, { maxRetries: 2, initialDelayMs: 1 })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: rec.onDelta })

  assert.equal(be.seen.length, 2, '应当试了两次')
  assert.equal(r.text, '第二次才算数')

  const resets = rec.deltas.filter((d) => d.reset)
  assert.equal(resets.length, 2, `两次尝试各要一个 reset，实际 ${JSON.stringify(rec.deltas)}`)
  // 而且第二个 reset 紧跟在半句之后 —— 界面正是在那一刻把它丢掉的
  const second = rec.deltas.findIndex((d, i) => i > 0 && d.reset)
  assert.equal(rec.deltas[second - 1].text, '半句话')
})

test('连接干净结束但没发 [DONE]：判失败，不当成完整回答', async (t) => {
  // 这条**单独拎出来**，因为它是流式特有的失败形状：不抛异常、不报错，
  // 只是「读完了」—— 而读到的是半个回答。交出去就是 §8.10 说的假装成功。
  const be = await backend((_b, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    res.write(`data: ${chunk('只有一半')}\n\n`)
    res.end()
  })
  t.after(() => be.close())

  const g = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  await assert.rejects(
    () => g.generate({ task: 't', evidence: 'e', onDelta: () => {} }),
    (err: Error) => {
      assert.match(err.message, /without \[DONE\]/, `错误信息要指着原因：${err.message}`)
      assert.match(err.message, /truncated/)
      return true
    },
  )
})

test('整次尝试一个字都没吐时不发 reset —— 没什么可重置的', async (t) => {
  // 和上一条相对。503 在 `#readStream` 之前就抛了，所以那次尝试
  // **没有向界面显示过任何东西**；这时发 reset 是无意义的动作。
  // 把这个语义写下来，是因为它容易被当成漏了（我自己第一版就断言错了）。
  const be = await backend((_b, res, nth) => {
    if (nth === 1) {
      res.writeHead(503, { 'content-type': 'text/plain' })
      res.end('overloaded')
      return
    }
    speakSse(res, [chunk('才'), chunk('算数')])
  })
  t.after(() => be.close())

  const rec = recorder()
  const inner = new HttpGenerator({ baseUrl: be.baseUrl, model: 'm' })
  const g = new RetryingGenerator(inner, { maxRetries: 2, initialDelayMs: 1 })
  const r = await g.generate({ task: 't', evidence: 'e', onDelta: rec.onDelta })

  assert.equal(r.text, '才算数')
  assert.deepEqual(rec.deltas, [{ text: '', reset: true }, { text: '才', reset: false }, { text: '算数', reset: false }])
})

// ═══════════════════════════════════════════════════════════
// 脚本生成器也分块 —— 离线路径（没 key、没网络）唯一能看见流式的地方
// ═══════════════════════════════════════════════════════════

test('脚本生成器分块吐，拼起来正好是它的完整输出', async () => {
  const rec = recorder()
  const g = new ScriptedGenerator({ latencyMs: 30 })
  const req = { task: '读一下 README', evidence: 'read_file(README.md)\n4016 字符' }

  const r = await g.generate({ ...req, onDelta: rec.onDelta })

  assert.equal(rec.deltas[0]?.reset, true, '第一段必须是 reset')
  assert.equal(rec.deltas.slice(1).every((d) => !d.reset), true, '之后都不该再 reset')
  assert.equal(
    rec.deltas.map((d) => d.text).join(''),
    r.text,
    '增量拼起来必须等于返回的文本 —— 否则界面上「流式看到的」和「最终答案」是两个东西',
  )
  assert.ok(rec.deltas.length > 3, `应当切成好几段，实际 ${rec.deltas.length} 段`)
  // 「游标必须前进」在这里的样子：除第一段的 reset 之外**没有空段**。
  // 空段意味着某一步没往前走（`size` 算成 0 的那个 bug 就长这样）。
  for (const d of rec.deltas.slice(1)) {
    assert.ok(d.text.length > 0, `每一段都要有内容：${JSON.stringify(d)}`)
  }
})

test('不给 onDelta 时脚本生成器行为不变：等一次，不分块', async () => {
  // ⚠️ 阈值别贴着实参写。第一版是 `latencyMs: 1` + `>= 1`，**五次里挂一次** ——
  //    `setTimeout(1)` 的实测耗时可以略小于 1ms（时钟精度），断言就翻。
  //    这不是产品的不稳定，是断言的不稳定，而两者在账面上长得一样。
  const g = new ScriptedGenerator({ latencyMs: 30 })
  const r = await g.generate({ task: 't', evidence: 'e' })
  assert.ok(r.text.includes('task: t'))
  assert.ok(r.latencyMs >= 15, `应当等了一次，实际 ${r.latencyMs}ms`)
})
