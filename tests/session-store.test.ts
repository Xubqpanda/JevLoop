/**
 * 会话存储的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里全是**磁盘行为**，`tsc` 通过不代表写对了。重点测四类：
 *
 *   · **盘上全留，截断只在读的那一侧** —— 以前是 `splice` 销毁历史，
 *     而且没有任何地方记着丢过东西。
 *   · **撕裂的尾行不能毁掉整个会话** —— 追加式格式的全部好处就在这，
 *     但它必须**被报出来**，不能悄悄少两轮。
 *   · **会话 id 是不可信输入**，它会变成文件名。`?session=../../x`
 *     必须被挡住。
 *   · **列表和读取分开** —— `list()` 不读正文，只回摘要。
 *
 * @module JevLoop/session-store.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { appendFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { SessionStore, assertSessionId, SESSION_FORMAT_VERSION } from '../src/session-store.ts'

/** 一个只属于这次测试的目录。用完删掉，免得互相看见 */
async function withStore(fn: (store: SessionStore, dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'jevstore-'))
  try {
    await fn(new SessionStore(dir), dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

const turn = (task: string, answer: string, at = 1000) => ({ task, answer, at })

// ═══════════════════════════════════════════════════════════
// 落盘与读回
// ═══════════════════════════════════════════════════════════

test('append 会先写文件头，再追加每一轮', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('问一', '答一'), { cwd: '/tmp/x' })
    await store.append('s1', turn('问二', '答二', 2000))

    const raw = await readFile(join(dir, 's1.jsonl'), 'utf8')
    const lines = raw.trim().split('\n')
    assert.equal(lines.length, 3, '一行文件头 + 两行轮次')

    const header = JSON.parse(lines[0]!)
    assert.equal(header.kind, 'header')
    assert.equal(header.id, 's1')
    // 格式版本必须在文件里 —— 将来换格式时读的人要靠它认出来
    assert.equal(header.v, SESSION_FORMAT_VERSION)
    assert.equal(header.cwd, '/tmp/x', '工作目录要记 —— 换了目录，上文就没有意义了')

    assert.deepEqual(await store.load('s1'), [turn('问一', '答一'), turn('问二', '答二', 2000)])
  })
})

test('第二次 append 不会重写文件头', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('a', 'b'))
    await store.append('s1', turn('c', 'd'))
    const raw = await readFile(join(dir, 's1.jsonl'), 'utf8')
    assert.equal(raw.split('\n').filter((l) => l.includes('"kind":"header"')).length, 1)
  })
})

test('load 带上限时只回最近几轮，但**盘上一条都不少**', async () => {
  await withStore(async (store) => {
    for (let i = 0; i < 20; i++) await store.append('s1', turn(`问${i}`, `答${i}`, 1000 + i))

    const last5 = await store.load('s1', 5)
    assert.equal(last5.length, 5)
    assert.equal(last5[0]!.task, '问15', '要的是最近的，不是最早的')
    assert.equal(last5.at(-1)!.task, '问19')

    // ★ 这是这条改动的全部意义：截断发生在**读**的一侧
    assert.equal((await store.load('s1')).length, 20, '盘上必须还是 20 轮')
  })
})

test('读一个不存在的会话返回空数组，不抛', async () => {
  await withStore(async (store) => {
    assert.deepEqual(await store.load('never'), [])
  })
})

// ═══════════════════════════════════════════════════════════
// 撕裂的尾行 / 认不出的行
// ═══════════════════════════════════════════════════════════

test('撕裂的尾行不会毁掉整个会话，但会被**报出来**', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('问一', '答一'))
    await store.append('s1', turn('问二', '答二', 2000))
    // 模拟写到一半断电：追加一段没有换行的半个 JSON
    await appendFile(join(dir, 's1.jsonl'), '{"kind":"turn","task":"问三","ans', 'utf8')

    assert.deepEqual(
      (await store.load('s1')).map((t) => t.task),
      ['问一', '问二'],
      '前面两轮必须完好',
    )
    const [summary] = await store.list()
    assert.equal(summary!.skipped, 1, '跳过了一行就要报 1 —— 不能悄悄少一轮')
  })
})

test('认不出的 kind 算跳过，不算轮次', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('问一', '答一'))
    // 将来加的 kind：老读法认不出来，但不该崩
    await appendFile(join(dir, 's1.jsonl'), '{"kind":"telemetry","x":1}\n', 'utf8')

    assert.equal((await store.load('s1')).length, 1)
    assert.equal((await store.list())[0]!.skipped, 1)
  })
})

test('空行不计入 skipped —— 文件末尾一定有换行', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('问一', '答一'))
    await appendFile(join(dir, 's1.jsonl'), '\n\n', 'utf8')
    assert.equal((await store.list())[0]!.skipped, 0)
  })
})

// ═══════════════════════════════════════════════════════════
// 列表
// ═══════════════════════════════════════════════════════════

test('list 按最后活动时间倒序，并且带上第一句当标题', async () => {
  await withStore(async (store) => {
    await store.append('old', turn('很久以前', 'a', 1000))
    await store.append('new', turn('刚刚', 'b', 9000))
    await store.append('mid', turn('中间', 'c', 5000))

    const list = await store.list()
    assert.deepEqual(list.map((s) => s.id), ['new', 'mid', 'old'])
    assert.equal(list[0]!.firstPrompt, '刚刚')
    assert.equal(list[0]!.turns, 1)
  })
})

test('updatedAt 跟着最后一轮走，createdAt 不跟着走', async () => {
  await withStore(async (store) => {
    await store.append('s1', turn('一', 'a', 1000))
    await store.append('s1', turn('二', 'b', 7000))
    const [s] = await store.list()
    assert.equal(s!.createdAt, 1000)
    assert.equal(s!.updatedAt, 7000)
    assert.equal(s!.turns, 2)
  })
})

test('目录不存在时 list 返回空数组 —— 那是「还没有会话」，不是错误', async () => {
  const dir = join(tmpdir(), `jevstore-missing-${Date.now()}`)
  assert.deepEqual(await new SessionStore(dir).list(), [])
})

test('list 不做正文搬运 —— 摘要里没有 answer 字段', async () => {
  await withStore(async (store) => {
    await store.append('s1', turn('问', '这是一段很长的回答'.repeat(100)))
    const [s] = await store.list()
    assert.equal('answer' in s!, false, '列表只回摘要')
  })
})

// ═══════════════════════════════════════════════════════════
// 会话 id 是不可信输入
// ═══════════════════════════════════════════════════════════

test('会话 id 变成文件名 —— 路径穿越必须被挡住', () => {
  for (const bad of ['../../etc/passwd', 'a/b', 'a\\b', '..', '', 'x'.repeat(65), 'a b', 'a.jsonl']) {
    assert.throws(() => assertSessionId(bad), /不合法/, `${JSON.stringify(bad)} 该被拒绝`)
  }
  for (const ok of ['default', 's1', 'abc-DEF_123', 'x'.repeat(64)]) {
    assert.doesNotThrow(() => assertSessionId(ok), `${ok} 该被放行`)
  }
})

test('非法的 id 在 append / load / list 路径上都进不去', async () => {
  await withStore(async (store) => {
    await assert.rejects(() => store.append('../escape', turn('a', 'b')), /不合法/)
    await assert.rejects(() => store.load('../escape'), /不合法/)
    await assert.rejects(() => store.remove('../escape'), /不合法/)
  })
})

// ═══════════════════════════════════════════════════════════
// 删除
// ═══════════════════════════════════════════════════════════

test('remove 删得掉，而且删不存在的返回 false 而不是抛', async () => {
  await withStore(async (store) => {
    await store.append('s1', turn('a', 'b'))
    assert.equal(await store.remove('s1'), true)
    assert.deepEqual(await store.list(), [])
    assert.equal(await store.remove('s1'), false, '再删一次是 false')
  })
})

test('list 忽略非 .jsonl 的文件', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('a', 'b'))
    await writeFile(join(dir, 'README.txt'), '不是我', 'utf8')
    assert.deepEqual((await store.list()).map((s) => s.id), ['s1'])
  })
})

test('是 .jsonl 但读不懂内容 —— 报成 0 轮 + 跳过计数，不消失', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('a', 'b'))
    await writeFile(join(dir, 'garbage.jsonl'), '这不是 JSON\n', 'utf8')
    const list = await store.list()
    // 文件**读到了**，只是内容认不出 —— 那是 skipped=1，不是 -1
    assert.deepEqual(list.map((s) => s.id).sort(), ['garbage', 's1'])
    const g = list.find((s) => s.id === 'garbage')!
    assert.equal(g.turns, 0)
    assert.equal(g.skipped, 1)
  })
})

test('整个文件读不了时 skipped = -1 —— 「读不到」和「读到了但是空的」要分得开', async () => {
  await withStore(async (store, dir) => {
    await store.append('s1', turn('a', 'b'))
    // 用目录占掉那个文件名：`readFile` 会抛 EISDIR，走 catch 那条路
    await mkdir(join(dir, 'weird.jsonl'))
    const list = await store.list()
    const w = list.find((s) => s.id === 'weird')!
    assert.ok(w, '读不了的文件也必须出现在列表里 —— 凭空少一个会话比报错更糟')
    assert.equal(w.skipped, -1)
  })
})
