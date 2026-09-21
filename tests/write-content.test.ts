/**
 * `write_file` 输入生成的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里每一条边界都对应一种**会写到盘上的错**：围栏被当正文、哨兵被当
 * 文件名、一整句话被当文件名、空内容覆盖掉原文件。四种都不会报错，
 * 只会在盘上留下一个错的文件 —— 所以每条都要钉住。
 *
 * @module JevLoop/write-content.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { parseWriteInput, writeInputVia, type WriteGeneration } from '../src/write-content.ts'
import type { GenerateRequest, GenerateResult } from '../src/llm.ts'

/** 一个脚本化生成器：记下收到的请求，回指定的文本 */
function fakeGen(text: string): {
  gen: { name: string; generate(r: GenerateRequest): Promise<GenerateResult> }
  seen: GenerateRequest[]
} {
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

/** 一段像样的历史：list_dir 加一次 read_file */
const HISTORY = [
  { tool: 'list_dir', input: '.', result: 'alpha.ts\nalpha.test.ts' },
  { tool: 'read_file', input: 'alpha.ts', result: 'export function totalOf(o: O[]): number { return 0 }' },
]

// ═══════════════════════════════════════════════════════════
// 解析：全文 → write_file 的输入
// ═══════════════════════════════════════════════════════════

test('★ 第一行是路径，其余是内容 —— 这正是 `write_file` 的输入格式', () => {
  assert.equal(parseWriteInput('summary.md\n# 标题\n正文'), 'summary.md\n# 标题\n正文\n')
})

test('★ 首尾空白去掉之后补回**恰好一个**换行', () => {
  assert.equal(parseWriteInput('\n\n  summary.md  \n\n# 标题\n\n正文\n\n\n'), 'summary.md\n# 标题\n\n正文\n')
})

test('模型说素材不够 → 不写 —— 不要把那句话当成文件名', () => {
  assert.equal(parseWriteInput('NOTHING'), undefined)
})

test('只有一行 → 不写。要么有路径没内容，要么没按格式来', () => {
  assert.equal(parseWriteInput('summary.md'), undefined)
})

test('有路径没内容 → 不写。空内容和「没写」在盘上分不出来', () => {
  assert.equal(parseWriteInput('summary.md\n\n  \n'), undefined)
})

test('全空 → 不写', () => {
  for (const blank of ['', '   \n', '\n\n']) assert.equal(parseWriteInput(blank), undefined, JSON.stringify(blank))
})

test('★ 一整句话被当成文件名时要挡住 —— 写出来的名字是错的，而没人会报错', () => {
  assert.equal(parseWriteInput('Here is the content of the file you asked for.\n# 标题'), undefined)
  assert.equal(parseWriteInput('好的，我来写这个文件。\n# 标题'), undefined)
})

test('★ markdown / 围栏的痕迹不是路径', () => {
  assert.equal(parseWriteInput('# summary.md\n内容'), undefined, '行首 # 是标题')
  assert.equal(parseWriteInput('- summary.md\n内容'), undefined, '行首 - 是列表项')
  assert.equal(parseWriteInput('`summary.md`\n内容'), undefined, '反引号是代码痕迹')
  assert.equal(parseWriteInput('> summary.md\n内容'), undefined, '行首 > 是引用')
})

test('路径可以含空格、可以带目录 —— 别把正常的挡掉', () => {
  assert.equal(parseWriteInput('my notes.md\nx'), 'my notes.md\nx\n')
  assert.equal(parseWriteInput('docs/api/summary.md\nx'), 'docs/api/summary.md\nx\n')
  assert.equal(parseWriteInput('a.b.c\nx'), 'a.b.c\nx\n')
})

test('路径过长 → 挡住（多半是模型把一整段当成了文件名）', () => {
  assert.equal(parseWriteInput(`${'x'.repeat(201)}\n内容`), undefined)
  assert.equal(parseWriteInput(`${'x'.repeat(200)}\n内容`), `${'x'.repeat(200)}\n内容\n`, '200 是允许的')
})

test('⚠️ 逃逸**不在这里挡** —— 那是 `safePath` 的事，这里只判格式', () => {
  // 这条测试是为了说清职责边界：`parseWriteInput` 放行，`safePath` 拒绝。
  // 两处都挡会让「谁负责安全」变得含糊，而只在这里挡则是个洞。
  assert.equal(parseWriteInput('../../etc/passwd\nx'), '../../etc/passwd\nx\n')
})

// ═══════════════════════════════════════════════════════════
// 围栏
// ═══════════════════════════════════════════════════════════

// 围栏在**生成器那一步**剥（`writeInputVia` 里调 `unwrapFence`），
// 不在解析那一步 —— 所以这两条要走完整条路，不能只测 `parseWriteInput`
test('★ 整段输出是一个围栏时剥掉它 —— 否则 ``` 会被写进文件', async () => {
  const write = writeInputVia(fakeGen('```\nsummary.md\n# 标题\n```').gen)
  assert.equal(await write({ task: 't', history: HISTORY }), 'summary.md\n# 标题\n')
})

test('★ 内容**本身**含围栏时不剥 —— 只认「整段恰好是一个围栏」', async () => {
  const body = 'a.md\n# 关于 markdown\n\n```sh\nnpm test\n```\n\n就这些。'
  const write = writeInputVia(fakeGen(body).gen)
  assert.equal(await write({ task: 't', history: HISTORY }), `${body}\n`, '原样保留')
})

// ═══════════════════════════════════════════════════════════
// 和生成器接起来
// ═══════════════════════════════════════════════════════════

test('没有任何工具调用时**不写** —— 没证据只能编，而编出来的会真的落盘', async () => {
  const { gen, seen } = fakeGen('a.md\n编的内容')
  const write = writeInputVia(gen)
  assert.equal(await write({ task: '改 a.md', history: [] }), undefined)
  assert.equal(seen.length, 0, '连生成器都不该调 —— 没东西可依据')
  assert.equal(await write({ task: '改 a.md' }), undefined)
})

test('证据来自 history 里的**工具输出**，和主生成路径同一个形状', async () => {
  const { gen, seen } = fakeGen('summary.md\n# 内容')
  await writeInputVia(gen)({ task: '把函数写进 summary.md', history: HISTORY })

  assert.equal(seen.length, 1)
  const req = seen[0]!
  assert.match(req.evidence, /list_dir\(\.\) → alpha\.ts/, '每一步都带上了工具名、输入和输出')
  assert.match(req.evidence, /export function totalOf/, '★ 读到的文件内容就是证据')
  assert.equal(req.task, '把函数写进 summary.md')
  assert.match(String(req.instruction), /first line is the path/i, '指令里说了格式：第一行是路径')
})

test('system prompt 透传 —— 那两条承重规则对写文件同样成立', async () => {
  const { gen, seen } = fakeGen('a.md\nx')
  await writeInputVia(gen, '只用证据。用任务的语言回答。')({ task: 't', history: HISTORY })
  assert.equal(seen[0]!.system, '只用证据。用任务的语言回答。')
})

test('★ 判成「写不了」也是一次**真的调用**，onGenerate 照样要报', async () => {
  const { gen } = fakeGen('NOTHING')
  const got: WriteGeneration[] = []
  const write = writeInputVia(gen, undefined, (g) => got.push(g))

  assert.equal(await write({ task: 't', history: HISTORY }), undefined)
  assert.equal(got.length, 1, '调用发生了就要计数 —— 漏计会让「判定:模型」那个比例是错的')
  assert.equal(got[0]!.result.model, 'fake')
  assert.ok(got[0]!.estimatedInputTokens > 0, '估的输入 token 也要有')
})

test('没有历史时 onGenerate 不报 —— 那次生成根本没发生', async () => {
  const got: WriteGeneration[] = []
  await writeInputVia(fakeGen('a.md\nx').gen, undefined, (g) => got.push(g))({ task: 't' })
  assert.equal(got.length, 0)
})
