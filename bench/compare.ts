/**
 * JevLoop · 两种循环形状的对比台
 *
 *   node --experimental-strip-types bench/compare.ts
 *   node --experimental-strip-types bench/compare.ts --repeat 3
 *
 * ══════════════════════════════════════════════════════════════
 *  **它回答的是哪一句话。**
 *
 *  这个项目到处写着「14:1」。那个数字是**判定次数 : 生成次数** ——
 *  它说的是「这条 loop 里的岔路口有几个不花生成的钱」，**不是**
 *  「省了 14 次大模型调用」。同样一个任务，一个像样的 ReAct 循环大概只要
 *  3–4 次调用。两者不是一个量纲的东西，混着说就变成了口号。
 *
 *  所以这个台子跑同一个任务的两条路，量三个数：
 *
 *     大模型调用次数   最直接的那个 —— 「便宜」到底便宜在哪
 *     墙钟             判定的延迟优势在这儿（§8.11）
 *     输入/输出 token  钱是按这个算的
 *
 * ── 读这些数字时要记住的三件事 ──────────────────────────────────
 *
 *  ① **JevLoop 那一边多干了两件事**：风险分级（判定，不花钱）和交付闸门
 *     （过不了要**再生成一次** —— 那是实打实的成本）。所以它在生成次数上
 *     有时是 2 而不是 1。这不是不公，是这条 loop 真的多做了事。
 *
 *  ② **两边都没做「谁的答案更好」的判断**。验收只看任务要求的工具调用有没有
 *     发生、回答里该有的东西在不在、文件有没有真的写出来（`bench/oracle.ts`）。
 *     质量不在这把尺子的量程里。
 *
 *  ③ **一次运行一个样本。** 想看得住就 `--repeat`，看中位数。
 *
 * @module JevLoop/compare
 */

import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { Decider, Meter, runAgent, loadEnv, resolveProvider, resolveGenerator } from '../src/index.ts'
import { RuleJudge } from '../examples/rule-judge.ts'
import { TASKS, type BenchTask } from './tasks.ts'
import { answerOk, missing, checkArtifacts, matchesCall } from './oracle.ts'
import { runReact, REACT_SYSTEM } from './react.ts'
import { C, withRetry } from './util.ts'
import { decisionEndpoint, generationEndpoint, measureFloor, type Floor } from './transport.ts'

const argv = process.argv.slice(2)
const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : undefined
const repeat = argv.includes('--repeat') ? Math.max(1, Number(argv[argv.indexOf('--repeat') + 1]) || 1) : 1
const useRule = argv.includes('--rule')
const MAX_STEPS = 8

/** 一条路跑一条任务的结果 */
interface Sample {
  shape: 'JevLoop' | 'ReAct'
  task: string
  modelCalls: number
  /**
   * **判定次数**（ReAct 那一边恒为 0 —— 它没有判定）。
   *
   * ★ 它和 `modelCalls` 放在同一张表里，是因为**两者的比值才是关键**：
   *   一次判定比一次生成快好几倍（实测 329ms vs 2036ms），但这条 loop
   *   每个任务要问十几次，而 ReAct 只调三四次 —— 端到端谁快，取决于
   *   「快几倍」和「多几次」谁赢。少了这一列，那张表会被读成
   *   「JevLoop 慢」，而真正的结论是「**单次快 6 倍，次数多 4 倍**」。
   */
  decisions: number
  inputTokens: number
  outputTokens: number
  latencyMs: number
  /**
   * **纯调用耗时** = 成功那几次调用的 latency 之和（判定 + 生成）。
   *
   * ★ 它**不含重试**：`retryCall` 把内层结果原样返回，所以
   *   `DecisionResult.latencyMs` / `GenerateResult.latencyMs` 量的都是
   *   **成功那一次**。失败的尝试和退避等待不在里面。
   *
   *   这就是「把重试这些原因导致的时延去掉之后」的数 —— 和 `latencyMs`
   *   的差就是重试与框架开销。
   */
  pureMs: number
  /** 工具执行花了多少 */
  toolMs: number
  /**
   * **纯计算时间** = `pureMs` 减去每次调用都要付的那笔「握手/校验」。
   *
   * 减法用的两个数由 `bench/transport.ts` 现量（同路径、同鉴权、不做推理），
   * 而且**基线不成立时这里是 `NaN`** —— 那时表里印 `—`，不印一个硬算出来的数。
   */
  computeMs: number
  /**
   * 墙钟花在哪。**没有这个分解，`40.7s` 那样的数字是不可解释的** ——
   * 而不可解释的数字对读的人只是一句「好慢」，指不出该改哪里。
   */
  decisionMs: number
  /** meter 记的生成耗时 —— **成功那一次**的，不含重试 */
  modelMs: number
  /**
   * 生成那一步的**墙钟**（由事件时间戳量出来）。
   *
   * ★ 它和 `modelMs` 不是一回事，而差出来的那部分是真的：生成失败重试时，
   *   失败那次的耗时和退避等待都**不计入** `GenerateResult.latencyMs`。
   *   实测一条 `list` 任务：`generate` 那一步墙钟 **21.8s**，meter 只记了
   *   **11.0s** —— 另外 10.8s 是重试。拿 meter 的数当墙钟，会得出一个
   *   「JevLoop 比 ReAct 慢 10 倍」的结论，而其中一半是后端抖动。
   */
  modelWallMs: number
  /** 验收：工具调用齐了、回答对了、产物在盘上 */
  passed: boolean
  why: string[]
}

// ── 后端：两条路**必须**用同一个生成器，否则数字不可比 ──────────

loadEnv()
let warned = false
const provider = useRule
  ? new RuleJudge()
  : resolveProvider({
      lastResort: new RuleJudge(),
      onFallback: (err, from, to) => {
        if (warned) return
        warned = true
        console.error(C.yellow(`  ▲ ${from} 不可用（${(err as Error).message.slice(0, 60)}），改用 ${to}`))
      },
    })
const generator = resolveGenerator()

/**
 * 两个后端「每次调用都要付一遍」的那笔钱 —— 现量。
 *
 * 基线 = 同一条路径、同样的鉴权、body 缺字段 → 服务端校验层拒掉、走不到模型。
 * 拿不到成立的基线时 `ok` 为 false，下面所有 `computeMs` 就是 `NaN`，
 * 表里印 `—` —— **宁可不给数，也不给一个假的**（第一版在这里算出过 -3268ms）。
 */
const decFloor: Floor = await measureFloor({ who: provider.name, ...decisionEndpoint('https://api.typesafe.ai') })
const genFloor: Floor = await measureFloor({ who: generator.name, ...generationEndpoint('https://api.deepseek.com') })

/** 一次调用要付的往返；基线不成立就没有这个数 */
function computeOf(pureMs: number, decisions: number, modelCalls: number): number {
  if (!decFloor.ok || !genFloor.ok) return NaN
  return Math.max(0, pureMs - decisions * decFloor.medianMs - modelCalls * genFloor.medianMs)
}

/** 铺夹具，跑，然后把目录删掉。两条路**用同一个函数**铺，免得夹具分叉 */
async function inFixture<T>(task: BenchTask, fn: (cwd: string) => Promise<T>): Promise<T> {
  const cwd = await mkdtemp(join(tmpdir(), `jevcompare-${task.id}-`))
  try {
    for (const [name, content] of Object.entries(task.files)) {
      await writeFile(join(cwd, name), content, 'utf8')
    }
    return await fn(cwd)
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
}

/**
 * 工具调用齐了吗 —— **两条路同一套判据**，而且和判据机用的是同一个函数
 * （`matchesCall`）。自己再写一份的话，`write_file` 那种「输入是路径 + 内容」
 * 的形状会被判错，而错的方向是**把成功的判成失败**。
 */
function callsOk(calls: { tool: string; input?: string }[], task: BenchTask): string[] {
  const why: string[] = []
  for (const want of task.required) {
    if (!calls.some((c) => matchesCall(want, c))) why.push(`没调用 ${want.tool}${want.input ? `(${want.input})` : ''}`)
  }
  return why
}

async function runJev(task: BenchTask): Promise<Sample> {
  try {
    return await withRetry(`JevLoop/${task.id}`, () => runJevOnce(task))
  } catch (err) {
    return failedSample('JevLoop', task.id, err)
  }
}

/**
 * 这一轮跑不起来（网络、后端）→ 记一条失败样本，**不要**把整场对比带走。
 *
 * ★ 实测（2026-09-21）：一次 `api.deepseek.com` 连接超时让整场 7 任务的
 *   对比崩掉，前面已经跑出来的样本全部作废 —— 而那几分钟是真的花掉了。
 *   网络失败不是循环形状的差别，把它记成一条「跑不起来」比丢掉整场诚实得多。
 */
function failedSample(shape: 'JevLoop' | 'ReAct', task: string, err: unknown): Sample {
  return {
    shape,
    task,
    decisions: 0,
    modelCalls: 0,
    inputTokens: 0,
    outputTokens: 0,
    latencyMs: 0,
    pureMs: 0,
    toolMs: 0,
    computeMs: 0,
    decisionMs: 0,
    modelMs: 0,
    modelWallMs: 0,
    passed: false,
    why: [`跑不起来：${(err as Error).message.slice(0, 60)}`],
  }
}

async function runJevOnce(task: BenchTask): Promise<Sample> {
  return inFixture(task, async (cwd) => {
    const meter = new Meter()
    const decider = new Decider({ provider, meter })
    const calls: { tool: string; input?: string }[] = []
    // ★ **整轮墙钟**，和 ReAct 那边同一个口径。
    //   第一版这里写的是 `s.decisionMs + s.modelMs` —— 那只算判定和生成，
    //   不算工具执行，而 ReAct 那边量的是整轮。**两把尺子量出来的数不可比**，
    //   而它看起来完全正常（一边 13.6s、一边 3.7s），差点就成了结论。
    const t0 = performance.now()
    /*
      用**事件时间戳**量生成那一步的墙钟：`generate` 事件和它上一条之间的间隔
      就是那一步实际花掉的时间。这样重试和退避都被算进去 —— 而它们是用户
      真的等掉的时间。
    */
    let lastAt = t0
    let modelWallMs = 0
    let sawGenerate = false
    let toolMs = 0

    const result = await runAgent({
      task: task.task,
      cwd,
      decider,
      generator,
      maxSteps: MAX_STEPS,
      onAskHuman: async () => true,
      ...(task.writeInput !== undefined ? { provideWriteInput: () => task.writeInput } : {}),
      onEvent: (e) => {
        const now = performance.now()
        if (e.type === 'generate') {
          modelWallMs += now - lastAt
          sawGenerate = true
        }
        lastAt = now
        if (e.type === 'tool:call') calls.push({ tool: e.tool, input: e.input })
        if (e.type === 'tool:result') toolMs += e.ms
      },
    })

    /*
      ★ **停表就在这里** —— `runAgent` 一返回，验收之前。

      以前这个数是在下面的返回对象里取的，而那已经在 `await checkArtifacts`
      **之后**了 —— 于是 JevLoop 那边多算了读盘的时间，而 ReAct 那边
      （`runReact` 内部停表）没有。差的是零点几毫秒，但口径不一致就是不一致：
      同一张表里的两个数必须用同一把尺子量，否则哪天它长大了也没人发现。
    */
    const wallMs = performance.now() - t0

    const why = [
      ...callsOk(calls, task),
      ...missing(result.answer, task),
      ...(await checkArtifacts(task, cwd)),
    ]
    const s = meter.stats
    return {
      shape: 'JevLoop' as const,
      task: task.id,
      // **只数生成**：判定不花这个钱，那正是这条 loop 的主张
      modelCalls: s.modelCalls,
      decisions: s.decisions,
      // `MeterStats` 上的是**生成器**报的真值。判定那一边的记录里没有 token ——
      // 因为判定不花生成的钱，那正是这条 loop 的主张（判定模型那边确实也发
      // token，但走的是另一个后端、另一个价目表，不计在这一列里）
      inputTokens: s.inputTokens,
      outputTokens: s.outputTokens,
      latencyMs: wallMs,
      pureMs: s.decisionMs + s.modelMs,
      toolMs,
      computeMs: computeOf(s.decisionMs + s.modelMs, s.decisions, s.modelCalls),
      decisionMs: s.decisionMs,
      modelMs: s.modelMs,
      modelWallMs: sawGenerate ? modelWallMs : s.modelMs,
      passed: why.length === 0 && answerOk(result.answer, task),
      why,
    }
  })
}

async function runReAct(task: BenchTask): Promise<Sample> {
  try {
    return await withRetry(`ReAct/${task.id}`, () => runReActOnce(task))
  } catch (err) {
    return failedSample('ReAct', task.id, err)
  }
}

async function runReActOnce(task: BenchTask): Promise<Sample> {
  return inFixture(task, async (cwd) => {
    const r = await runReact({ task: task.task, cwd, generator, maxSteps: MAX_STEPS })
    if (r.failed) {
      return {
        shape: 'ReAct' as const,
        task: task.id,
        modelCalls: r.modelCalls,
        decisions: 0,
        inputTokens: r.inputTokens,
        outputTokens: r.outputTokens,
        latencyMs: r.latencyMs,
        pureMs: r.modelMs,
        toolMs: r.toolMs,
        computeMs: computeOf(r.modelMs, 0, r.modelCalls),
        decisionMs: 0,
        modelMs: r.modelMs,
        modelWallMs: r.modelMs,
        passed: false,
        why: [`跑不起来：${r.failed.slice(0, 60)}`],
      }
    }
    const why = [...callsOk(r.calls, task), ...missing(r.answer, task), ...(await checkArtifacts(task, cwd))]
    return {
      shape: 'ReAct' as const,
      task: task.id,
      modelCalls: r.modelCalls,
      decisions: 0,
      inputTokens: r.inputTokens,
      outputTokens: r.outputTokens,
      latencyMs: r.latencyMs,
      pureMs: r.modelMs,
      toolMs: r.toolMs,
      computeMs: computeOf(r.modelMs, 0, r.modelCalls),
      decisionMs: 0,
      modelMs: r.modelMs,
      modelWallMs: r.modelMs,
      passed: why.length === 0,
      why,
    }
  })
}

// ── 汇总 ─────────────────────────────────────────────────────

const median = (xs: number[]): number => {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2
}

const pad = (s: string | number, n: number) => {
  const str = String(s)
  return str + ' '.repeat(Math.max(0, n - [...str].length))
}
const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`

async function main(): Promise<void> {
  const tasks = only ? TASKS.filter((t) => t.id.includes(only)) : TASKS
  console.log(C.bold('\nJevLoop · 两种循环形状'))
  console.log(C.dim(`  判定后端 : ${provider.name}`))
  console.log(C.dim(`  生成后端 : ${generator.name}`))
  console.log(C.dim(`  任务     : ${tasks.length} 条 × ${repeat} 遍 · maxSteps ${MAX_STEPS}（两条路相同）`))
  console.log(C.dim('  验收     : 要求的工具调用 + 回答内容 + 产物在盘上（两边同一套判据）'))

  const all: Sample[] = []
  for (const task of tasks) {
    for (let i = 0; i < repeat; i++) {
      process.stdout.write(C.dim(`\r  跑 ${task.id} (${i + 1}/${repeat})…`.padEnd(60)))
      all.push(await runJev(task))
      process.stdout.write(C.dim(`\r  跑 ${task.id} (${i + 1}/${repeat})…  ReAct`.padEnd(60)))
      all.push(await runReAct(task))
    }
  }
  process.stdout.write('\r'.padEnd(62) + '\r')

  // 基线的证据 —— 没有它，读的人分不清「服务端拒了」和「连接超时了」
  for (const f of [decFloor, genFloor]) {
    console.log(
      C.dim(`  ${f.ok ? C.green('✓') : C.red('✗')} 握手基线 ${f.who} · HTTP ${f.status || '—'} · ${Math.round(f.medianMs)}ms (n=${f.samples})`),
    )
    if (!f.ok) {
      console.log(C.red(`      ✗ ${f.problem}`))
      console.log(C.dim('      → 「纯计算」那一列会是空的：减法需要一个成立的基线，宁可不给数'))
    }
  }

  // ── 逐任务 ──
  console.log('')
  console.log(C.bold('  ── 逐任务 ──────────────────────────────────────────────'))
  console.log(
    C.dim(
      `  ${pad('任务', 22)}${pad('形状', 9)}${pad('判定', 6)}${pad('大模型调用', 11)}${pad('纯调用', 9)}${pad('纯计算', 9)}${pad('墙钟', 8)}${pad('输出token', 10)}验收`,
    ),
  )
  for (const task of tasks) {
    for (const shape of ['JevLoop', 'ReAct'] as const) {
      const xs = all.filter((s) => s.task === task.id && s.shape === shape)
      if (!xs.length) continue
      console.log(
        `  ${pad(shape === 'JevLoop' ? task.id : '', 22)}${pad(shape, 9)}` +
          pad(median(xs.map((s) => s.decisions)), 6) +
          pad(median(xs.map((s) => s.modelCalls)), 11) +
          pad(secs(median(xs.map((s) => s.pureMs))), 9) +
          pad(Number.isFinite(median(xs.map((s) => s.computeMs))) ? secs(median(xs.map((s) => s.computeMs))) : '—', 9) +
          pad(secs(median(xs.map((s) => s.latencyMs))), 8) +
          pad(Math.round(median(xs.map((s) => s.outputTokens))), 10) +
          (xs.every((s) => s.passed) ? C.green('✓') : C.red(`✗ ${xs.find((s) => !s.passed)?.why.join('；').slice(0, 40)}`)),
      )
    }
    // 分解：读到「40.7s」时**下一步能查什么**，全在这一行里
    const jx = all.filter((s) => s.task === task.id && s.shape === 'JevLoop')
    const rx = all.filter((s) => s.task === task.id && s.shape === 'ReAct')
    const parts = (xs: Sample[], label: string) => {
      if (!xs.length) return ''
      const wall = median(xs.map((s) => s.latencyMs))
      const pure = median(xs.map((s) => s.pureMs))
      const tool = median(xs.map((s) => s.toolMs))
      // 余项 = 墙钟 − 纯调用 − 工具。**重试和框架开销都在这里**，
      // 而它是减出来的：前两项是量出来的，这一项是剩下的。
      const rest = Math.max(0, wall - pure - tool)
      return (
        `${label} 纯调用 ${secs(pure)}（判定 ${secs(median(xs.map((s) => s.decisionMs)))}` +
        ` + 生成 ${secs(median(xs.map((s) => s.modelMs)))}）· 工具 ${secs(tool)}` +
        ` · 重试/框架 ${secs(rest)}`
      )
    }
    console.log(
      C.dim(`  ${' '.repeat(22)}${parts(jx, 'JevLoop')}`) +
        (rx.length ? C.dim(`   │   ${parts(rx, 'ReAct')}`) : ''),
    )
    console.log('')
  }

  // ── 汇总 ──
  console.log(C.bold('  ── 汇总 ────────────────────────────────────────────────'))
  console.log(
    C.dim(
      `  ${pad('形状', 10)}${pad('判定/任务', 10)}${pad('大模型调用/任务', 16)}${pad('纯调用/任务', 14)}${pad('纯计算/任务', 14)}${pad('墙钟/任务', 12)}${pad('输出token/任务', 16)}验收`,
    ),
  )
  for (const shape of ['JevLoop', 'ReAct'] as const) {
    const xs = all.filter((s) => s.shape === shape)
    if (!xs.length) continue
    const okCount = xs.filter((s) => s.passed).length
    console.log(
      `  ${pad(shape, 10)}` +
        pad(median(xs.map((s) => s.decisions)).toFixed(1), 10) +
        pad(median(xs.map((s) => s.modelCalls)).toFixed(1), 16) +
        pad(secs(median(xs.map((s) => s.pureMs))), 14) +
        pad(Number.isFinite(median(xs.map((s) => s.computeMs))) ? secs(median(xs.map((s) => s.computeMs))) : '—', 14) +
        pad(secs(median(xs.map((s) => s.latencyMs))), 12) +
        pad(Math.round(median(xs.map((s) => s.outputTokens))), 16) +
        `${okCount}/${xs.length}`,
    )
  }

  const j = all.filter((s) => s.shape === 'JevLoop')
  const r = all.filter((s) => s.shape === 'ReAct')
  const jc = median(j.map((s) => s.modelCalls))
  const rc = median(r.map((s) => s.modelCalls))
  console.log('')
  const jd = median(j.map((s) => s.decisions))
  console.log(
    C.bold(`  同一个任务：大模型调用 JevLoop ${jc} 次 vs ReAct ${rc} 次`) +
      C.dim(`   （比值 ${(rc / jc).toFixed(1)}×）`),
  )
  /*
    ★ **把「单次快几倍」和「次数多几次」分开写。**

      只报「JevLoop 1 次 vs ReAct 4 次」会被读成「Jev 快」。而端到端
      墙钟上 JevLoop 反而是慢的那个（实测 5.5s vs 4.1s），因为判定要往返
      `jd` 次、单次约 330ms，而 ReAct 的三四次生成每次一千多毫秒。
      **单次快，不等于这件事快。**
  */
  console.log(
    C.dim(
      `  ★ JevLoop 每个任务还要 ${jd.toFixed(0)} 次**判定**（ReAct 一次都没有）。\n` +
        '    单次判定比一次生成快好几倍（实测 330ms vs 2000ms），但次数多 —— 端到端谁快\n' +
        '    取决于这两件事谁赢。上表的墙钟就是答案，别只看调用次数。',
    ),
  )
  console.log(
    C.dim(
      '  ★ 而「判定 : 生成」那个比值（README 里的 13:1）说的是**这条 loop 里有多少岔路口\n' +
        '    不花生成的钱**，不是「省了 13 次大模型调用」—— 同样任务 ReAct 只要 3–4 次。',
    ),
  )
  console.log('')

  // ── 对手拿到的指令 ──
  //
  // 「对比」如果说不清对手拿到的指令，读的人就只能信我。所以原样印出来。
  console.log(C.dim('  ── ReAct 那一边拿到的 system prompt（原样）────────────────'))
  for (const line of REACT_SYSTEM.split('\n')) console.log(C.dim(`  │ ${line}`))
  console.log('')
}

await main()
