#!/usr/bin/env node
/**
 * JevLoop · demo
 *
 *   node --experimental-strip-types examples/demo.ts
 *   node --experimental-strip-types examples/demo.ts --laya     # 用本地 Laya
 *   node --experimental-strip-types examples/demo.ts --jev      # 用官方 Jev（需 TYPESAFE_API_KEY）
 *
 * 零依赖、零 key、离线可跑 —— 默认用 Mock 后端把 loop 走通。
 *
 * 重点看最后那行汇总：**判定 : 模型** 的比值。
 *
 * @module JevLoop/demo
 */

import { mkdir, writeFile, rm, mkdtemp } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { Decider, Meter, runAgent, loadEnv, resolveProvider, resolveGenerator, formatRatio } from '../src/index.ts'
import { RuleJudge } from './rule-judge.ts'

// 先加载 .env（有 TYPESAFE_API_KEY 就会自动用官方 Jev）
const env = loadEnv()
const argv = process.argv.slice(2)
const has = (f: string) => argv.includes(`--${f}`)
const prefer: 'jev' | 'laya' | 'mock' | 'rule' | 'scripted' | undefined =
  (['jev', 'laya', 'mock', 'rule', 'scripted'] as const).find(has)

const C = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  magenta: (s: string) => `\x1b[35m${s}\x1b[0m`,
}

// ── 造一个工作目录 ───────────────────────────────────────────

const cwd = await mkdtemp(join(tmpdir(), 'JevLoop-'))
await mkdir(cwd, { recursive: true })
await writeFile(
  join(cwd, 'invoice.ts'),
  `export interface Invoice {
  id: string
  amount: number
  paid: boolean
}

/** Total amount still unpaid */
export function outstanding(invoices: Invoice[]): number {
  return invoices
    .filter((i) => !i.paid)
    .reduce((sum, i) => sum + i.amount, 0)
}
`,
  'utf8',
)
await writeFile(join(cwd, 'notes.md'), '# Notes\n\nA demo directory.\n', 'utf8')

// ── 组装 ─────────────────────────────────────────────────────

// 降级只提示一次 —— 每次判定都打一遍会把 trace 淹掉
let warned = false
const notice = (err: unknown, from: string, to: string) => {
  if (warned) return
  warned = true
  console.error(`  ▲ ${from} unavailable (${(err as Error).message.slice(0, 60)}), falling back to ${to}`)
}

// 判定后端：--rule 强制规则表；--jev / --laya 强制指定；否则按可用性自动解析
const provider =
  prefer === 'rule'
    ? new RuleJudge()
    : resolveProvider({
        ...(prefer === 'jev' || prefer === 'laya' ? { prefer } : {}),
        ...(process.env.JEVOS_SIDECAR ? { layaUrl: process.env.JEVOS_SIDECAR } : {}),
        // ★ 链尾用规则表，不用 Mock。
        //   Mock 的保守答案会让 pickTool 走到 escalate —— 全新 clone 上
        //   `npm run demo` 会在第一步停下，而 README 承诺「无 key 也能跑」。
        //   规则表属于 examples（§8.6），所以在这里注入而不是写进内核。
        lastResort: new RuleJudge(),
        onFallback: notice,
      })

const meter = new Meter()
const decider = new Decider({
  provider,
  meter,
  // ★ 这两个选项以前**没有任何调用方**（`grep -rn "onWarn\|strict" examples/ server.ts`
  //   零命中）—— 于是预算校验只在 `decide()` 里算一遍就丢掉，算了没人看等于没算。
  //   这也是 C1（`headBudget` 死字段）修好之后仍然不生效的原因：只做 C1 不做这条，
  //   `error` 级别也只是一行没人读的字符串。
  onWarn: (id, warnings) => {
    for (const w of warnings) console.log(C.yellow(`  ⚠ budget [${id}] ${w.message}`))
    for (const w of warnings) if (w.hint) console.log(C.dim(`      ${w.hint}`))
  },
  // --strict：预算的 error 级别直接抛，在**发请求之前**拦住
  strict: process.argv.includes('--strict'),
})
const generator = resolveGenerator({ scripted: prefer === 'scripted' })

const TASK = 'List the files in the working directory, read the TypeScript file, and explain which functions it defines.'

console.log(C.bold('\nJevLoop · demo'))
console.log(C.dim(`  task      : ${TASK}`))
console.log(C.dim(`  cwd       : ${cwd}`))
console.log(C.dim(`  decision  : ${provider.name}`))
console.log(C.dim(`  generator : ${generator.name}${generator.name === 'scripted' ? ' — set DEEPSEEK_API_KEY for a real LLM' : ''}`))
if (env.loaded.length) console.log(C.dim(`  .env      : loaded ${env.loaded.join(', ')}`))
// 认不出来的行进 `skipped`，**必须显示** —— 一行 `.env` 写错就悄悄退回 Mock 的话，
// 排查方向会被完全带偏（以前 `export KEY=VALUE` 就是这个下场）。
if (env.skipped.length) {
  console.log(C.yellow(`  .env      : ⚠ skipped ${env.skipped.length} unrecognised line(s)`))
  for (const line of env.skipped) console.log(C.dim(`              ${line}`))
}
console.log('')
console.log(C.bold('  ── loop trace ──────────────────────────────────────────'))

const result = await runAgent({
  task: TASK,
  cwd,
  decider,
  generator,
  maxSteps: 8,
  onTrace: (line) => console.log(C.dim(line)),
})

// ── 输出 ─────────────────────────────────────────────────────

console.log('')
console.log(C.bold('  ── every decision ──────────────────────────────────────'))
console.log(C.dim(meter.trace()))

console.log('')
console.log(C.bold('  ── result ──────────────────────────────────────────────'))
console.log(`  halt      : ${C.cyan(result.halt)}`)
console.log(`  steps     : ${result.steps}`)
console.log('')
console.log(C.dim('  ' + result.answer.split('\n').join('\n  ').slice(0, 600)))

const s = meter.stats
console.log('')
console.log(C.bold('  ── accounting ──────────────────────────────────────────'))
console.log(
  `  decisions ${C.green(String(s.decisions).padStart(3))}     ${C.dim(`${s.decisionMs}ms (${s.avgDecisionMs}ms each)`)}`,
)
console.log(
  `  model     ${C.magenta(String(s.modelCalls).padStart(3))}     ${C.dim(`${s.modelMs}ms`)}`,
)
console.log('')
console.log(
  // 走 meter 的统一出口。以前这里自己拼，0 次模型调用时会报成 `3 : 1`（真相是 3:0）。
  `  ${C.bold('decisions : model =')} ${C.bold(C.green(formatRatio(s)))}` +
    C.dim(`   decisions are ${(s.decisionShare * 100).toFixed(1)}% of wall clock`),
)
console.log('')

await rm(cwd, { recursive: true, force: true })
