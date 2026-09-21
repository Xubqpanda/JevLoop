/**
 * JevLoop · 一次调用里，「握手」和「算」各占多少
 *
 *   node --experimental-strip-types bench/latency.ts
 *
 * ══════════════════════════════════════════════════════════════
 *  `bench/compare.ts` 说「JevLoop 慢」。这个台子说**为什么**。
 *
 *  「一次判定 330ms」这句话本身不说明任何事 —— 没说那 330ms 里有多少
 *  是在算、多少是在等一个来回。而这两件事的优化方向正好相反：
 *  算得慢要换模型，等得久要换部署方式（搬到本地就没有来回）。
 * ══════════════════════════════════════════════════════════════
 *
 * @module JevLoop/latency
 */

import { loadEnv, resolveProvider, resolveGenerator, Decider, Meter, type Generator } from '../src/index.ts'
import { buildDecisions } from '../src/decisions.ts'
import { decisionEndpoint, generationEndpoint, measureFloor, type Floor } from './transport.ts'
import { C } from './util.ts'

const pad = (s: string | number, n: number) => {
  const str = String(s)
  return str + ' '.repeat(Math.max(0, n - [...str].length))
}
const ms = (x: number) => (Number.isFinite(x) ? `${Math.round(x)}ms` : '—')
const secs = (x: number) => `${(x / 1000).toFixed(2)}s`

/** 跑 n 次，丢掉第一次（含握手），返回中位数和样本 */
async function timeIt<T>(n: number, fn: () => Promise<T>): Promise<{ median: number; n: number }> {
  const xs: number[] = []
  for (let i = 0; i < n + 1; i++) {
    const t = performance.now()
    await fn()
    xs.push(performance.now() - t)
  }
  const warm = xs.slice(1).sort((a, b) => a - b)
  return { median: warm[warm.length >> 1] ?? NaN, n: warm.length }
}

/** 一次生成：返回耗时和 provider 报的输出 token */
async function timeGenerate(
  generator: Generator,
  req: { task: string; evidence: string },
  n: number,
): Promise<{ median: number; n: number; outputTokens: number }> {
  let outputTokens = 0
  let counted = 0
  const xs: number[] = []
  for (let i = 0; i < n + 1; i++) {
    const t = performance.now()
    const r = await generator.generate(req)
    const dt = performance.now() - t
    if (i > 0) {
      xs.push(dt)
      if (r.outputTokens > 0) {
        outputTokens += r.outputTokens
        counted++
      }
    }
  }
  const warm = xs.sort((a, b) => a - b)
  return { median: warm[warm.length >> 1] ?? NaN, n: warm.length, outputTokens: counted ? outputTokens / counted : 0 }
}

/*
  ★ **`loadEnv()` 必须在 `resolve*()` 之前。**
  这两个解析器是**按环境变量决定后端**的，key 还没进 `process.env` 就调用它们，
  拿到的是降级链的另一端 —— 第一版漏了这一句，于是台子量的是 mock 和脚本
  生成器（判定的名字直接印成 `laya→mock`，两边生成的耗时一模一样），
  而数字看起来完全正常。**一个测错对象的台子比没有台子更糟。**
*/
loadEnv()
const provider = resolveProvider()
const generator = resolveGenerator()
const specs = buildDecisions()
const meter = new Meter()
const decider = new Decider({ provider, meter })

const DECISION_N = 9
const GEN_N = 4

console.log(C.bold('\nJevLoop · 一次调用的耗时拆开'))
console.log(C.dim('  基线 = 同一条路径、同样的鉴权、body 缺字段 → 服务端校验层拒掉，走不到模型'))
console.log(C.dim('  所以它量的是：传输 + 鉴权 + 校验。连接是复用的，第一次（含 TCP/TLS）已丢掉。'))

// ── 判定 ──
const decEl = decisionEndpoint(process.env.JEV_BASE_URL ?? 'https://api.typesafe.ai')
const decFloor: Floor = await measureFloor({ who: provider.name, ...decEl })
const dec = await timeIt(DECISION_N, () =>
  decider.decide(specs.needsTool, {
    task: '列出工作目录里的文件，读取其中的 TypeScript 文件，说明它定义了哪些函数',
    cwd: '/tmp',
    files: ['invoice.ts', 'notes.md'],
    readFiles: [],
    history: [],
  }),
)

// ── 生成 ──
const genEl = generationEndpoint(process.env.LLM_BASE_URL ?? 'https://api.deepseek.com')
const genFloor = await measureFloor({ who: generator.name, ...genEl })
const genSmall = await timeGenerate(
  generator,
  { task: '用一句话说明什么是 HTTP', evidence: '(没有工具结果)' },
  GEN_N,
)
const genBig = await timeGenerate(
  generator,
  {
    task: '读一下文件，然后用 markdown 写 300 字总结它定义了哪些函数',
    evidence:
      'read_file(invoice.ts) → export interface Invoice { id: string; amount: number; paid: boolean }\nexport function outstanding(invoices: Invoice[]): number { ... }',
  },
  Math.max(2, GEN_N - 1),
)

const rows = [
  { who: `判定 · ${provider.name}`, floor: decFloor, total: dec.median },
  { who: '生成 · 一句话', floor: genFloor, total: genSmall.median },
  { who: '生成 · 300 字', floor: genFloor, total: genBig.median },
]

console.log('')
console.log(C.bold('  ── 拆开 ────────────────────────────────────────────────'))
console.log(C.dim(`  ${pad('', 26)}${pad('握手/校验', 12)}${pad('总计', 10)}${pad('其中在算', 12)}算占`))
for (const r of rows) {
  // ★ 基线不成立就**不算**，印 `—` 并说清楚为什么。
  //   硬算出来的会是负数（实测 -3268ms），而一个负的「计算耗时」比空着更糟：
  //   它看起来像个测量结果。
  if (!r.floor.ok) {
    console.log(`  ${pad(r.who, 26)}${pad('—', 12)}${pad(ms(r.total), 10)}${pad('—', 12)}—`)
    continue
  }
  const compute = r.total - r.floor.medianMs
  const share = r.total > 0 ? (compute / r.total) * 100 : 0
  console.log(
    `  ${pad(r.who, 26)}${pad(ms(r.floor.medianMs), 12)}${pad(ms(r.total), 10)}${pad(ms(compute), 12)}${share.toFixed(0)}%`,
  )
}
// 基线的**证据**要印出来：状态码和响应内容。没有它，读的人分不清
// 「服务端拒了」和「连接超时了」—— 而这两种情况下那个「握手」数毫无意义。
for (const f of [decFloor, genFloor]) {
  const mark = f.ok ? C.green('✓') : C.red('✗')
  console.log(C.dim(`  ${mark} ${f.who} · ${f.url} · HTTP ${f.status || '—'} · ${ms(f.medianMs)} (n=${f.samples})`))
  console.log(C.dim(`      ${f.detail || '(没有响应体)'}`))
  if (!f.ok) console.log(C.red(`      ✗ 这个基线不成立：${f.problem}`))
  console.log(C.dim(`      所以 ${f.who} 那一行的「其中在算」是空着的 —— 宁可不给数，也不给一个假的`))
}
if (genFloor.ok && genSmall.outputTokens > 0) {
  const compute = genSmall.median - genFloor.medianMs
  console.log(
    C.dim(
      `  生成吞吐 ≈ ${(genSmall.outputTokens / (compute / 1000)).toFixed(1)} token/s` +
        `（一句话那次输出 ${genSmall.outputTokens} token，算的部分 ${secs(compute)}）`,
    ),
  )
}

// ── 结论：单次计算时间 vs 每次都要付的握手 ──
console.log('')
console.log(C.bold('  ── 那这张台子想说什么 ──────────────────────────────────'))
const decCompute = decFloor.ok ? ms(dec.median - decFloor.medianMs) : '（基线不成立，没量到）'
console.log(
  C.dim(
    `  Jev 的**计算**只有 ${decCompute}，一次真实判定的大头是那 ${ms(decFloor.medianMs)} 的往返。\n` +
      '  所以它慢不是因为「判得慢」，是因为每个任务要问十几次，而**每次都要再付一遍往返**。\n' +
      '  搬到本地就没有那个往返 —— 那时判定的单次成本只剩「算」的那部分。',
  ),
)
console.log('')
