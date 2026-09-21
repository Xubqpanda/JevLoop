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
 * @module JevLoop/cli
 */

import { spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import { USAGE, parseArgv } from './cli-args.ts'
import { Decider, Meter, formatRatio, loadEnv, resolveGenerator, resolveProvider, runAgent } from './index.ts'
import { compilePolicy, compileQuestions } from './decision-compile.ts'
import { headline, parseDecisionDoc, isGate, summarize } from './decisiondoc.ts'

/** 包根目录。编译后 `dist/cli.js` 与源码 `src/cli.ts` 都指回包根。 */
const PKG_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

/** 只在真的接终端时上色 —— 管道和重定向里不该混进转义序列。 */
const COLOR = process.stdout.isTTY === true
const paint = (code: number) => (s: string) => (COLOR ? `\x1b[${code}m${s}\x1b[0m` : s)
const dim = paint(2)
const bold = paint(1)
const yellow = paint(33)
const green = paint(32)

/** 降级只提示一次 —— 每次判定都打一遍会把 trace 淹掉（同 `examples/demo.ts`）。 */
function onceNotifier(): (err: unknown, from: string, to: string) => void {
  let warned = false
  return (err, from, to) => {
    if (warned) return
    warned = true
    console.error(yellow(`  ▲ ${from} unavailable (${(err as Error).message.slice(0, 60)}), falling back to ${to}`))
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
  if (provider.name.includes('mock')) {
    console.log('')
    console.log(yellow('  ⚠ no decision model available — every step will escalate.'))
    console.log(dim('    Set TYPESAFE_API_KEY, or run a local Laya sidecar on :7789.'))
  }

  console.log('')
  console.log(bold('  ── loop trace ──────────────────────────────────────────'))

  const result = await runAgent({
    task,
    cwd,
    decider,
    generator,
    maxSteps: Number(options.get('max-steps') ?? 8),
    onTrace: (line) => console.log(dim(line)),
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
  console.log('')

  return result.halt === 'agent_done' || result.halt === 'task_done' ? 0 : 1
}

/**
 * 起界面。
 *
 * `server.ts` 留在包根、不编译进 `dist/`：它用 `import.meta.url` 定位 `web/`
 * 和 `DECISION.md`，编译到 `dist/server.js` 之后这两个路径会一起指错。
 *
 * ⚠️ **代价：从 npm 装出来时它跑不了。** Node 拒绝给 `node_modules` 下的文件
 * 剥离类型（`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`，§8.8 记的就是这条）。
 * 所以这里检测到包在 `node_modules` 下就直接给出指引 —— 把 Node 的栈丢给用户
 * 等于让人自己猜，而这一条猜不出来。
 */
function serve(options: Map<string, string>): Promise<number> {
  const serverPath = join(PKG_ROOT, 'server.ts')
  if (!existsSync(serverPath)) {
    console.error(`✗ ${serverPath} is missing — this package was published without server.ts`)
    return Promise.resolve(1)
  }

  if (PKG_ROOT.includes(`${sep}node_modules${sep}`)) {
    console.error('✗ `jevloop serve` needs a clone, not an installed package.')
    console.error('  Node cannot strip types for files under node_modules, and server.ts is TypeScript.')
    console.error('')
    console.error('    git clone https://github.com/zjunlp/JevLoop && cd JevLoop')
    console.error('    node --experimental-strip-types server.ts')
    console.error('')
    console.error('  `jevloop run` and `jevloop spec` work from the installed package.')
    return Promise.resolve(1)
  }

  // 类型剥离 v22.6 引入、v22.18 才默认开启。低版本必须显式带这个标志。
  const [major = 0, minor = 0] = process.versions.node.split('.').map(Number)
  const needsFlag = major < 22 || (major === 22 && minor < 18)

  const env = { ...process.env }
  const cwd = options.get('cwd')
  if (cwd) env.CWD_ROOT = resolve(cwd)
  const port = options.get('port')
  if (port) env.PORT = port
  const host = options.get('host')
  if (host) env.HOST = host

  const child = spawn(process.execPath, [...(needsFlag ? ['--experimental-strip-types'] : []), serverPath], {
    stdio: 'inherit',
    env,
  })
  return new Promise((done) => {
    child.on('exit', (code) => done(code ?? 1))
    child.on('error', (err) => {
      console.error(`✗ could not start server.ts: ${err.message}`)
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
