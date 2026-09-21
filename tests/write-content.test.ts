/**
 * `write_file` 内容生成的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这些边界都是**真的会写到盘上**的东西，所以每一条都要钉住：
 * 围栏被写进文件、哨兵被当成正文、空内容覆盖掉原文件 —— 三种都不会报错，
 * 只会在盘上留下一个错的文件。
 *
 * @module JevLoop/write-content.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { writeContentVia, type WriteGeneration } from '../src/write-content.ts'
import type { GenerateRequest, GenerateResult } from '../src/llm.ts'

/** 一个脚本化生成器：记下收到的请求，回指定的文本 */
function fakeGen(text: string): { gen: { name: string; generate(r: GenerateRequest): Promise<GenerateResult> }; seen: GenerateRequest[] } {
  const seen: GenerateRequest[] = []
  return {
    seen,
    gen: {
      name: 'fake',
      generate: async (r: GenerateRequest): Promise<GenerateResult> => {
        seen.push(r)
        return { text, latencyMs: 7, inputTokens: 11, outputTokens: 22, model: 'fake' }
      },
    },
  }
}

/** 一段像样的历史：list_dir 加两次 read_file */
const HISTORY = [
  { tool: 'list_dir', input: '.', result: 'alpha.ts\nalpha.test.ts' },
  { tool: 'read_file', input: 'alpha.ts', result: 'export function totalOf(o: O[]): number { return 0 }' },
]

test('没有任何工具调用时**不写** —— 没证据只能编，而编出来的会真的落盘', async () => {
  const { gen, seen } = fakeGen('随便编的内容')
  const write = writeContentVia(gen)
  assert.equal(await write('notes.md', { task: '改 notes.md', history: [] }), undefined)
  assert.equal(seen.length, 0, '连生成器都不该调 —— 没东西可依据')
  assert.equal(await write('notes.md', { task: '改 notes.md' }), undefined)
})

test('证据来自 history 里的**工具输出**，和主生成路径同一个形状', async () => {
  const { gen, seen } = fakeGen('# 内容\n')
  const write = writeContentVia(gen)
  await write('summary.md', { task: '把函数写进 summary.md', history: HISTORY })

  assert.equal(seen.length, 1)
  const req = seen[0]!
  assert.match(req.evidence, /list_dir\(\.\) → alpha\.ts/, '每一步都带上了工具名、输入和输出')
  assert.match(req.evidence, /export function totalOf/, '★ 读到的文件内容就是证据')
  assert.equal(req.task, '把函数写进 summary.md')
  assert.match(String(req.instruction), /summary\.md/, '指令里点名了目标文件')
})

test('system prompt 透传 —— 那两条承重规则对写文件同样成立', async () => {
  const { gen, seen } = fakeGen('x')
  await writeContentVia(gen, '只用证据。用任务的语言回答。')('a.md', { task: 't', history: HISTORY })
  assert.equal(seen[0]!.system, '只用证据。用任务的语言回答。')
})

test('★ 整段输出是一个围栏时剥掉它 —— 否则 ``` 会被写进文件', async () => {
  const write = writeContentVia(fakeGen('```ts\nconst a = 1\n```\n').gen)
  assert.equal(await write('a.ts', { task: 't', history: HISTORY }), 'const a = 1\n')
})

test('★ 内容**本身**含围栏时不剥 —— 只认「整段恰好是一个围栏」', async () => {
  const body = '# 关于 markdown\n\n```sh\nnpm test\n```\n\n就这些。'
  const write = writeContentVia(fakeGen(body).gen)
  assert.equal(await write('a.md', { task: 't', history: HISTORY }), `${body}\n`, '原样保留')
})

test('模型说素材不够 → 不写 —— 不要把那句话本身当成文件内容', async () => {
  const write = writeContentVia(fakeGen('NOTHING').gen)
  assert.equal(await write('a.md', { task: 't', history: HISTORY }), undefined)
})

test('空输出 → 不写。空文件和「没写」在盘上分不出来，而调用方以为发生了前者', async () => {
  for (const blank of ['', '   \n\n  ']) {
    const write = writeContentVia(fakeGen(blank).gen)
    assert.equal(await write('a.md', { task: 't', history: HISTORY }), undefined, JSON.stringify(blank))
  }
})

test('★ 首尾空白去掉之后补回**恰好一个**换行', async () => {
  // 模型常常前后各留空行；盘上的文本文件该以恰好一个换行结尾
  const write = writeContentVia(fakeGen('\n\n# 标题\n\n正文\n\n\n').gen)
  assert.equal(await write('a.md', { task: 't', history: HISTORY }), '# 标题\n\n正文\n')
})

test('★ 判成「写不了」也是一次**真的调用**，onGenerate 照样要报', async () => {
  const { gen } = fakeGen('NOTHING')
  const got: WriteGeneration[] = []
  const write = writeContentVia(gen, undefined, (g) => got.push(g))

  assert.equal(await write('a.md', { task: 't', history: HISTORY }), undefined)
  assert.equal(got.length, 1, '调用发生了就要计数 —— 漏计会让「判定:模型」那个比例是错的')
  assert.equal(got[0]!.result.model, 'fake')
  assert.ok(got[0]!.estimatedInputTokens > 0, '估的输入 token 也要有')
})

test('没有历史时 onGenerate 不报 —— 那次生成根本没发生', async () => {
  const got: WriteGeneration[] = []
  await writeContentVia(fakeGen('x').gen, undefined, (g) => got.push(g))('a.md', { task: 't' })
  assert.equal(got.length, 0)
})
