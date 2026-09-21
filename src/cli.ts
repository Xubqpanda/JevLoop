#!/usr/bin/env node
/**
 * JevLoop · 命令行入口
 *
 * `examples/demo.ts` 回答的是「这条 loop 长什么样」——它写死一个任务、
 * 自带一个规则判定器，好让全新 clone 也能跑出数字。这个文件回答的是另一个
 * 问题：「拿它干活」。区别不是参数多少，是**任务从哪来**：demo 的任务是写死的，
 * 这里的任务由调用方给，工作目录由调用方给，判定后端按环境解析。
 *
 * 所以两者不合并 —— 合并会把 demo 里那个刻意的规则判定器（§8.6：规则属于
 * 场景不属于内核）带进一个用户以为在跑真实判定的命令里。
 *
 * ── 子命令为什么是这三个 ──────────────────────────────────────
 *
 * `run` / `serve` / `spec` 分别对应这个仓库能被用起来的三种方式：
 * 当库调（run）、当应用看（serve）、当格式检查（spec）。
 * `spec` 是只有这里能做的那一个 —— 它打印 `DECISION.md` 编译成了什么，
 * 包括**哪些谓词没编译出来**（那意味着一条不存在的闸门）。
 *
 * ── 为什么不能再拆 ──────────────────────────────────────────────
 *
 * **一句话说得完：把 argv 变成一个动作，把结果印出来。** 上面那三个子命令
 * **共用同一套输出词汇** —— 颜色、对齐、`── xxx ──` 那种分隔标题、
 * `dim()` / `bold()` 的用法。按子命令切开，那套词汇要再切出第三个文件，
 * 而读的人得从三个地方拼出「这个 CLI 能做什么」，恰恰丢掉了入口文件唯一的
 * 用处：**一眼看全**。
 *
 * 消费者也是同一个（终端前的人），而 §12 给的两条接缝（「输入输出形状变了」
 * 或「消费者不是同一批人」）在这里都不成立。
 *
 * ⚠️ 真正撑大它的是 `runTask` 里那段**给人看的账目**（约 50 行
 *    `console.log`）。哪天真要拆，接缝在那里 —— 把它做成一个「把 result
 *    印成账目」的函数，不是按子命令切。
 *
 * @module JevLoop/cli
 */

import { spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { USAGE, parseArgv } from './cli-args.ts'
import { Decider, Meter, formatRatio, loadEnv, resolveGenerator, resolveProvider, runAgent } from './index.ts'
import { compilePolicy, compileQuestions } from './decision-compile.ts'
import { headline, parseDecisionDoc, isGate, summarize } from './decisiondoc.ts'
import { describeGates, type GateOverrides } from './gates.ts'
import { resolveGates } from './decisions.ts'

/** 包根目录。编译后 `dist/cli.js` 与源码 `src/cli.ts` 都指回包根。 */
const PKG_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

/** 只在真的接终端时上色 —— 管道和重定向里不该混进转义序列。 */
const COLOR = process.stdout.isTTY === true
const paint = (code: number) => (s: string) => (COLOR ? `\x1b[${code}m${s}\x1b[0m` : s)
const dim = paint(2)
const bold = paint(1)
const yellow = paint(33)
const green = paint(32)

/**
 * 降级/重试的提示。
 *
 * 按**每一个转换**各报一次，不是一个全局标志位：重试也走这个回调
 * （`from === to`），全局标志会让先到的重试把后面真正的降级吃掉，
 * 而降级是这两件事里更重要的那个。
 *
 * 两种情况分开说 —— 同一个后端再试一次**不是**降级，套用降级的句子
 * 会印出「laya unavailable, falling back to laya」。
 */
function onceNotifier(): (err: unknown, from: string, to: string) => void {
  const notified = new Set<string>()
  return (err, from, to) => {
    const key = `${from}→${to}`
    if (notified.has(key)) return
    notified.add(key)
    const why = (err as Error).message.slice(0, 60)
    console.error(
      from === to
        ? yellow(`  ▲ ${from}: ${why}`)
        : yellow(`  ▲ ${from} unavailable (${why}), falling back to ${to}`),
    )
  }
}

async function runTask(task: string, options: Map<string, string>): Promise<number> {
  const cwd = resolve(options.get('cwd') ?? process.cwd())
  if (!existsSync(cwd)) {
    console.error(`✗ --cwd ${cwd} does not exist`)
    return 1
  }

  // 从**用户的工作目录**读 .env，不是从包目录 —— 这是调用方的项目，key 属于它。
  const env = loadEnv({ cwd })

  const prefer = options.has('jev') ? 'jev' : options.has('laya') ? 'laya' : undefined
  const provider = resolveProvider({
    ...(prefer ? { prefer } : {}),
    ...(process.env.JEVOS_SIDECAR ? { layaUrl: process.env.JEVOS_SIDECAR } : {}),
    onFallback: onceNotifier(),
  })
  const generator = resolveGenerator({})

  const meter = new Meter()
  const decider = new Decider({
    provider,
    meter,
    onWarn: (id, warnings) => {
      for (const w of warnings) console.log(yellow(`  ⚠ budget [${id}] ${w.message}`))
    },
    strict: options.has('strict'),
  })

  console.log(bold('\nJevLoop · run'))
  console.log(dim(`  task      : ${task}`))
  console.log(dim(`  cwd       : ${cwd}`))
  console.log(dim(`  decision  : ${provider.name}`))
  console.log(dim(`  generator : ${generator.name}`))
  if (env.loaded.length) console.log(dim(`  .env      : loaded ${env.loaded.join(', ')}`))

  // 没有判定模型时**先说出来**。不说的话表现为「第一步就 escalate」，
  // 而那看起来像 bug，不像缺配置 —— 排查方向会被完全带偏（§8.10）。
  //
  // ⚠️ 判据是**链头**，不是 `includes('mock')`。链尾永远是 Mock 兜底
  // （`resolveProvider` 的默认 `lastResort`），所以拿「含不含 mock」去判
  // 会在**配了 Jev 的时候也误报** —— 实测 `jev→laya→mock` 被判成「没有判定模型」。
  // 链头是 mock 才真的没有可用的判定后端。
  if (provider.name.split('→')[0] === 'mock') {
    console.log('')
    console.log(yellow('  ⚠ no decision model available — every step will escalate.'))
    console.log(dim('    Set TYPESAFE_API_KEY, or run a local Laya sidecar on :7789.'))
  }

  console.log('')
  console.log(bold('  ── loop trace ──────────────────────────────────────────'))

  /*
    门限覆盖。**在任何模型调用之前验完** —— 名字写错是致命的（见 `gates.ts`
    文件头：静默无效等于你以为加了一道闸门），而在这里验意味着写错不花钱。
  */
  let gates: GateOverrides = {}
  try {
    gates = resolveGates(options.get('gate') ?? process.env.JEVLOOP_GATES ?? '')
  } catch (err) {
    console.error(`✗ ${(err as Error).message}`)
    return 1
  }

  const result = await runAgent({
    task,
    cwd,
    decider,
    generator,
    maxSteps: Number(options.get('max-steps') ?? 8),
    onTrace: (line) => console.log(dim(line)),
    gates,
  })

  const s = meter.stats
  console.log('')
  console.log(bold('  ── result ──────────────────────────────────────────────'))
  console.log(`  halt      : ${result.halt}`)
  console.log(`  steps     : ${result.steps}`)
  console.log('')
  console.log(dim('  ' + result.answer.split('\n').join('\n  ').slice(0, 2000)))
  console.log('')
  console.log(bold('  ── accounting ──────────────────────────────────────────'))
  console.log(`  decisions ${green(String(s.decisions).padStart(3))}     ${dim(`${s.decisionMs}ms (${s.avgDecisionMs}ms each)`)}`)
  console.log(`  model     ${String(s.modelCalls).padStart(3)}     ${dim(`${s.modelMs}ms`)}`)
  console.log('')
  console.log(`  ${bold('decisions : model =')} ${bold(green(formatRatio(s)))}${dim(`   decisions are ${(s.decisionShare * 100).toFixed(1)}% of wall clock`)}`)
  // 覆盖过的门限**必须出现在给人看的那份账上**，不只在日志里 ——
  // 否则两次结果不同时，读的人会去怀疑模型，而不是怀疑自己改过的那个数
  const gateLine = describeGates(gates)
  if (gateLine) console.log(yellow(`  gates     : ${gateLine}  （覆盖了默认值）`))
  console.log('')

  /*
    退出码按「**答完了吗**」判，不按「走了哪条路」。

    ★ `answered_directly` 漏了是真 bug（2026-09-21 实测）：它和 `agent_done`
      / `task_done` 是**并列的成功出口** —— 三条都在 `agent.ts` 里 `break`
      出来走同一条尾路（生成 → 过交付闸门 → 返回答案）。漏掉的表现是
      `jevloop run "一句不用查资料的问题"` **打印一个好好的答案然后退出 1**，
      而退出码是 `jevloop run … && …` 唯一看的东西。

    ⚠️ 带 `+revise` 后缀的不算成功（交付闸门修订过一次后仍然没放行），
      所以这里精确匹配，不用前缀。
  */
  const DONE = new Set(['answered_directly', 'agent_done', 'task_done'])
  return DONE.has(result.halt) ? 0 : 1
}

/**
 * 起界面。
 *
 * 两条路，取决于这个 `cli.js` 是从哪跑起来的：
 *
 * - **装出来的包**：`dist/cli.js` 旁边就是 `dist/server.js`（`tsc` 一起编的），
 *   直接跑它。这条是 npm 路径，也是 `serve` 从包装出来能跑的原因。
 * - **clone**：`src/cli.ts` 旁边是 `src/server.ts`，按类型剥离跑。
 *
 * 两者都能跑，是因为 `server.ts` 的 `ROOT` 是**往上找 `package.json`**，
 * 不是「本文件所在目录」—— 从 `src/` 跑和从 `dist/` 跑都指回包根，
 * 而 `web/` 和 `DECISION.md` 都在那儿。
 */
function serve(options: Map<string, string>): Promise<number> {
  const here = dirname(fileURLToPath(import.meta.url))
  const compiled = join(here, 'server.js')
  const source = join(here, 'server.ts')

  let script: string
  let stripTypes = false
  if (existsSync(compiled)) {
    script = compiled
  } else if (existsSync(source)) {
    script = source
    // 类型剥离 v22.6 引入、v22.18 才默认开启。低版本必须显式带这个标志。
    const [major = 0, minor = 0] = process.versions.node.split('.').map(Number)
    stripTypes = major < 22 || (major === 22 && minor < 18)
  } else {
    console.error(`✗ neither ${compiled} nor ${source} exists — this package is incomplete`)
    return Promise.resolve(1)
  }

  const env = { ...process.env }
  const cwd = options.get('cwd')
  if (cwd) env.CWD_ROOT = resolve(cwd)
  const port = options.get('port')
  if (port) env.PORT = port
  const host = options.get('host')
  if (host) env.HOST = host

  const child = spawn(process.execPath, [...(stripTypes ? ['--experimental-strip-types'] : []), script], {
    stdio: 'inherit',
    env,
  })
  return new Promise((done) => {
    child.on('exit', (code) => done(code ?? 1))
    child.on('error', (err) => {
      console.error(`✗ could not start the server: ${err.message}`)
      done(1)
    })
  })
}

/**
 * 打印 `DECISION.md` 编译成了什么。
 *
 * 重点是**没编译出来的那部分**：一个认不出的谓词会退化成一条永不命中的规则，
 * 也就是一道不存在的闸门，而它是 fail open 的。人写这份文件，所以这里必须出声。
 */
function spec(fileArg: string | undefined): number {
  const local = join(process.cwd(), 'DECISION.md')
  const file = fileArg ? resolve(fileArg) : existsSync(local) ? local : join(PKG_ROOT, 'DECISION.md')

  let md: string
  try {
    md = readFileSync(file, 'utf8')
  } catch (err) {
    console.error(`✗ cannot read ${file}: ${(err as Error).message}`)
    return 1
  }

  const doc = parseDecisionDoc(md)
  const s = summarize(doc)

  console.log(bold('\nJevLoop · spec'))
  console.log(dim(`  file     : ${file}`))
  console.log(dim(`  headline : ${headline(doc)}`))
  console.log(dim(`  model    : ${s.modelDecisions} decisions reach the decision model, ${s.codeDecisions} are decided by code`))
  console.log('')

  let broken = 0
  for (const block of doc.blocks) {
    const questions = compileQuestions(block)
    const policy = compilePolicy(block)
    const uncompiled = policy?.problems ?? []
    broken += uncompiled.length

    const asks = questions ? Object.keys(questions).join(', ') : '—'
    const gate = isGate(block) ? `  ${yellow('gate')}` : ''
    console.log(`  ${bold(block.id.padEnd(14))} ${block.kind.padEnd(6)} ${dim(`asks: ${asks}`)}${gate}`)
    for (const problem of uncompiled) console.log(`    ${yellow('✗')} ${problem}`)
  }

  if (doc.problems.length > 0) {
    console.log('')
    console.log(yellow(`  ${doc.problems.length} parse problem(s):`))
    for (const p of doc.problems) console.log(yellow(`    L${p.line}: ${p.message}`))
  }

  console.log('')
  if (doc.problems.length === 0 && broken === 0) {
    console.log(green('  ✓ parses clean and every predicate compiles'))
  }
  console.log('')

  return doc.problems.length === 0 && broken === 0 ? 0 : 1
}

async function main(): Promise<number> {
  const argv = process.argv.slice(2)

  if (argv.includes('--help') || argv.includes('-h') || argv[0] === undefined) {
    console.log(USAGE)
    return argv[0] === undefined ? 1 : 0
  }
  if (argv.includes('--version')) {
    const pkg = JSON.parse(readFileSync(join(PKG_ROOT, 'package.json'), 'utf8')) as { version: string }
    console.log(pkg.version)
    return 0
  }

  const { command, positional, options } = parseArgv(argv)

  if (command === 'run') {
    const task = positional.join(' ').trim()
    if (!task) {
      console.error('✗ run needs a task: jevloop run "list the files and explain them"')
      return 1
    }
    return runTask(task, options)
  }
  if (command === 'serve') return serve(options)
  if (command === 'spec') return spec(positional[0])

  console.error(`✗ unknown command '${command}'\n`)
  console.log(USAGE)
  return 1
}

process.exitCode = await main()
