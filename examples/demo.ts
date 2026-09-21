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

import { Decider, Meter, runAgent, loadEnv, resolveProvider, resolveGenerator } from '../src/index.ts'
import { RuleJudge } from './rule-judge.ts'

// 先加载 .env（有 TYPESAFE_API_KEY 就会自动用官方 Jev）
const loaded = loadEnv()
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

/** 计算未付款总额 */
export function outstanding(invoices: Invoice[]): number {
  return invoices
    .filter((i) => !i.paid)
    .reduce((sum, i) => sum + i.amount, 0)
}
`,
  'utf8',
)
await writeFile(join(cwd, 'notes.md'), '# 说明\n\n这是一个演示目录。\n', 'utf8')

// ── 组装 ─────────────────────────────────────────────────────

// 降级只提示一次 —— 每次判定都打一遍会把 trace 淹掉
let warned = false
const notice = (err: unknown, from: string, to: string) => {
  if (warned) return
  warned = true
  console.error(`  ▲ ${from} 不可用（${(err as Error).message.slice(0, 60)}），改用 ${to}`)
}

// 判定后端：--rule 强制规则表；--jev / --laya 强制指定；否则按可用性自动解析
const provider =
  prefer === 'rule'
    ? new RuleJudge()
    : resolveProvider({
        ...(prefer === 'jev' || prefer === 'laya' ? { prefer } : {}),
        ...(process.env.JEVOS_SIDECAR ? { layaUrl: process.env.JEVOS_SIDECAR } : {}),
        // 有真实判定后端时，规则表当最后兜底：至少 loop 能跑完
        onFallback: (err, from, to) => {
          if (to === 'mock') return
          notice(err, from, to)
        },
      })

const meter = new Meter()
const decider = new Decider({ provider, meter })
const generator = resolveGenerator({ scripted: prefer === 'scripted' })

const TASK = '列出工作目录里的文件，读取其中的 TypeScript 文件，说明它定义了哪些函数。'

console.log(C.bold('\nJevLoop · demo'))
console.log(C.dim(`  task      : ${TASK}`))
console.log(C.dim(`  cwd       : ${cwd}`))
console.log(C.dim(`  判定后端  : ${provider.name}`))
console.log(C.dim(`  生成后端  : ${generator.name}${generator.name === 'scripted' ? '（脚本化，设 DEEPSEEK_API_KEY 可换真实 LLM）' : ''}`))
if (loaded.length) console.log(C.dim(`  .env      : 已加载 ${loaded.join(', ')}`))
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
console.log(C.bold('  ── 逐条明细 ────────────────────────────────────────────'))
console.log(C.dim(meter.trace()))

console.log('')
console.log(C.bold('  ── 结果 ────────────────────────────────────────────────'))
console.log(`  halt      : ${C.cyan(result.halt)}`)
console.log(`  steps     : ${result.steps}`)
console.log('')
console.log(C.dim('  ' + result.answer.split('\n').join('\n  ').slice(0, 600)))

const s = meter.stats
console.log('')
console.log(C.bold('  ── 记账 ────────────────────────────────────────────────'))
console.log(
  `  判定  ${C.green(String(s.decisions).padStart(3))} 次   ${C.dim(`${s.decisionMs}ms（均 ${s.avgDecisionMs}ms）`)}`,
)
console.log(
  `  模型  ${C.magenta(String(s.modelCalls).padStart(3))} 次   ${C.dim(`${s.modelMs}ms`)}`,
)
console.log('')
console.log(
  `  ${C.bold('判定 : 模型 =')} ${C.bold(C.green(s.modelCalls ? s.ratio.toFixed(1) : String(s.decisions)) + ' : 1')}` +
    C.dim(`   判定耗时只占 ${(s.decisionShare * 100).toFixed(1)}%`),
)
console.log('')

await rm(cwd, { recursive: true, force: true })
