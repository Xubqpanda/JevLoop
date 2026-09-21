/**
 * JevLoop · 判定标定台
 *
 *   node --experimental-strip-types bench/run.ts
 *   node --experimental-strip-types bench/run.ts --rule      # 离线规则表
 *   node --experimental-strip-types bench/run.ts --only pick # 只跑 id 含 pick 的
 *   node --experimental-strip-types bench/run.ts --repeat 5 # 每条任务跑 5 遍
 *
 * `--repeat` 存在的理由：**判定和生成都是随机的**，而 7 条任务跑一遍
 * 得到的命中率，样本小到分不清「这个节点 79% 对」和「这次恰好错了 3 个」。
 * 要拿这个数字做决定，就得知道它的抖动有多大。
 *
 * ══════════════════════════════════════════════════════════════
 *  **量一件事：七个判定点，各自判对了多少。**
 * ══════════════════════════════════════════════════════════════
 *
 * README 里那张表说托管 Jev「decisive and correct on every decision」——
 * 那是一次 demo 的观感。这个台子把它变成数字，而且**分开报**：
 * 每个节点各自的命中率，而不是一个笼统的「准确率」。
 *
 * ── 为什么是逐节点，而不是一个总分 ──────────────────────────────
 *
 * 一个总分没法用。如果 `pickTool` 只有 70% 对而 `isDone` 有 98%，
 * 总分 84% 既掩盖了问题也掩盖了可用性 —— 真正该做的是把**不可信的
 * 那个节点**换成别的判法（比如交给 LLM），而不是整体放弃或整体接受。
 * 逐节点才给得出这个结论。
 *
 * ── 置信度那一栏才是重点 ────────────────────────────────────────
 *
 * README 记过 Laya 的失败：四个判定点的答案**全落在 0.55–0.66**，
 * 错的比对的还高 —— 「No threshold fixes that — it's a capability gap,
 * not a calibration gap」。
 *
 * 所以这个表不只报命中率，还报**对的那些和错的那些各自的概率**。
 * 命中率高但两组概率重叠 → 阈值救不了；两组分得开 → 阈值能救。
 * 这两个结论指向完全不同的下一步。
 *
 * ── 它不测什么 ──────────────────────────────────────────────────
 *
 * 报告最后一节**必须**列出没测到的部分。一个不说自己覆盖面的
 * 测量结果，会被读成「全都测过了」。
 *
 * @module JevLoop/run
 */

import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  Decider,
  Meter,
  runAgent,
  loadEnv,
  resolveProvider,
  resolveGenerator,
  type AgentEvent,
} from '../src/index.ts'
import { RuleJudge } from '../examples/rule-judge.ts'
import { TASKS, type BenchTask } from './tasks.ts'
import { Oracle, answerOk, missing, checkArtifacts, type Judgement } from './oracle.ts'
import { C, withRetry } from './util.ts'

const argv = process.argv.slice(2)
const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : undefined
const repeat = argv.includes('--repeat') ? Math.max(1, Number(argv[argv.indexOf('--repeat') + 1]) || 1) : 1
const useRule = argv.includes('--rule')

// ── 组装后端 ─────────────────────────────────────────────────
//
// ★ **后端名字必须打进报告。** 一份不写后端名字的命中率没有意义 ——
//   规则表、本地 Laya、托管 Jev 是三个完全不同的东西。

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

// ── 跑一条任务 ───────────────────────────────────────────────

interface TaskRun {
  task: BenchTask
  /**
   * 这次运行**没跑起来**时的原因。
   *
   * ★ 跑失败**既不记对也不记错**，单独计。混进任何一边都是在编数字：
   *   记成「错」会把网络问题算成判定的能力问题，记成「对」更糟。
   *   第一版没有这个字段，一次 connect timeout 直接把整轮 21 次运行
   *   崩掉了 —— 测量工具必须先能活下来，才谈得上量得准。
   */
  failed?: string
  judgements: Judgement[]
  answer: string
  halt: string
  /** 任务的最终回答过没过 */
  answerPassed: boolean
  /** 为什么没过 */
  answerWhy: string[]
  /**
   * 产物验收：任务要求的文件真的在盘上吗。
   *
   * 和回答文本**分开报** —— 「说了」和「做了」是两件事，合成一个勾
   * 会让「说了但没做」看起来像成功（`write` 任务第一版就是这么骗过我的）。
   */
  artifactWhy: string[]
}

/**
 * 网络抖动重试。
 *
 * LLM 和判定后端都在网络上，而 `fetch failed` / `UND_ERR_CONNECT_TIMEOUT`
 * 是**常态**不是异常（这台机器上 github 和 api 都时常连不上）。没有重试
 * 的话，一轮十分钟的测量会被一次抖动清零。
 */
async function runTask(task: BenchTask): Promise<TaskRun> {
  const cwd = await mkdtemp(join(tmpdir(), `jevbench-${task.id}-`))
  try {
    for (const [name, content] of Object.entries(task.files)) {
      await writeFile(join(cwd, name), content, 'utf8')
    }

    const meter = new Meter()
    const decider = new Decider({ provider, meter })
    const oracle = new Oracle(task)
    const judgements: Judgement[] = []

    const result = await withRetry('跑这条任务', () =>
      runAgent({
      task: task.task,
      cwd,
      decider,
      generator,
      maxSteps: 8,
      // 台子上授权一律放行：这里量的是**判定的分级对不对**，
      // 不是「有没有人来点确认」。`gradeRisk` 该问而没问照样判错。
      onAskHuman: async () => true,
      // 写哪个文件、写什么内容，两者都是生成，不属于判定 —— 由台子给（三分法）
      ...(task.writeInput !== undefined
        ? { provideWriteInput: () => task.writeInput }
        : {}),
        onEvent: (e: AgentEvent) => {
          const j = oracle.feed(e)
          if (j) judgements.push(j)
        },
      }),
    )

    // 最终验收走**和 canDeliver 同一套**判据 —— 两处判据分开写必然分叉
    const passed = answerOk(result.answer, task)
    return {
      task,
      judgements,
      answer: result.answer,
      halt: result.halt,
      answerPassed: passed,
      answerWhy: missing(result.answer, task),
      artifactWhy: await checkArtifacts(task, cwd),
    }
  } catch (err) {
    // 空 catch 说明：吞的是**这条任务跑不起来**，把它变成一条失败的记录，
    // 而不是让它冒出去把整轮测量带走。
    return {
      task,
      failed: (err as Error).message,
      judgements: [],
      answer: '',
      halt: 'run_failed',
      answerPassed: false,
      answerWhy: [],
      artifactWhy: [],
    }
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
}

/**
 * 产物验收：说做了的，盘上到底有没有。
 *
 * 读不到就报「没写出来」；内容不匹配就报「写了但不对」——
 * 这两种失败的含义不同，合成一句会丢掉排查方向。
 */
// ── 汇总 ─────────────────────────────────────────────────────

const median = (xs: number[]): number | undefined => {
  if (!xs.length) return undefined
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2
}

const pct = (n: number, d: number) => (d === 0 ? '  —  ' : `${((n / d) * 100).toFixed(0).padStart(3)}%`)
const num = (x: number | undefined) => (x === undefined ? '  —  ' : x.toFixed(2))
const pad = (s: string, n: number) => s + ' '.repeat(Math.max(0, n - [...s].length))

/**
 * 先补齐宽度，再上色。
 *
 * ★ 顺序不能反：`pad` 数的是**字符**，而 ANSI 转义也是字符 ——
 *   把上了色的字符串交给 `pad`，它会把 `\x1b[31m` 这 5 个转义字符也当成
 *   占位宽度，于是列宽算多、数字挤在一起（实测：`3` 和 `1` 显示成 `31`，
 *   读起来像一个两位数）。**给字符串量宽度之前得先知道哪些字符不占位。**
 */
const cell = (v: string | number, n: number, color?: (s: string) => string) => {
  const plain = pad(String(v), n)
  return color ? color(plain) : plain
}

interface NodeStat {
  node: string
  right: number
  wrong: number
  unjudged: number
  rightProbs: number[]
  wrongProbs: number[]
  /** 出错的原话，报告里列出来 */
  why: string[]
}

async function main(): Promise<void> {
  const tasks = only ? TASKS.filter((t) => t.id.includes(only)) : TASKS
  if (tasks.length === 0) {
    console.error(`  没有匹配 --only ${only} 的任务`)
    process.exit(1)
  }

  console.log(C.bold('\nJevLoop · 判定标定台'))
  console.log(C.dim(`  判定后端 : ${provider.name}`))
  console.log(C.dim(`  生成后端 : ${generator.name}`))
  console.log(C.dim(`  任务     : ${tasks.length} 条${repeat > 1 ? ` × ${repeat} 遍` : ''}`))
  console.log('')

  const runs: TaskRun[] = []
  for (let pass = 1; pass <= repeat; pass++) {
    for (const t of tasks) {
      process.stdout.write(C.dim(`  跑 ${cell(t.id, 14)} ${cell(repeat > 1 ? `第 ${pass} 遍` : '', 8)} ${t.probes}\n`))
      runs.push(await runTask(t))
    }
  }

  // ── 逐任务 ─────────────────────────────────────────────────
  console.log('')
  const failed = runs.filter((r) => r.failed)
  if (failed.length) {
    console.log('')
    console.log(C.bold('  ── 没跑起来的 ──────────────────────────────────────────'))
    for (const r of failed) console.log(C.red(`  ✗ ${r.task.id}：${r.failed?.slice(0, 90)}`))
    console.log(C.dim(`  这 ${failed.length} 次**不计入**下面任何一个数字 —— 网络失败不是判定失败`))
  }

  console.log(C.bold(`  ── 逐任务${repeat > 1 ? '（多遍，见下面的逐节点汇总）' : ''} ──────────────────────`))
  console.log(C.dim(`  ${cell('id', 15)}${cell('判定', 6)}${cell('对', 5)}${cell('错', 5)}${cell('未判', 6)}验收`))
  const shown = repeat > 1 ? [] : runs.filter((r) => !r.failed)
  for (const r of shown) {
    const j = r.judgements
    const right = j.filter((x) => x.verdict === 'right').length
    const wrong = j.filter((x) => x.verdict === 'wrong').length
    const un = j.filter((x) => x.verdict === 'unjudged').length
    // 回答和产物**分开显示**：一个是「说对了」，一个是「做到了」
    const mark = [
      r.answerPassed ? C.green('回答✓') : C.red(`回答✗ ${r.answerWhy.join(' ')}`),
      ...(r.task.artifacts?.length
        ? [r.artifactWhy.length ? C.red(`产物✗ ${r.artifactWhy.join(' ')}`) : C.green('产物✓')]
        : []),
    ].join(' ')
    console.log(
      `  ${cell(r.task.id, 15)}${cell(j.length, 6)}${cell(right, 5)}` +
        `${cell(wrong, 5, wrong ? C.red : undefined)}${cell(un, 6, un ? C.yellow : undefined)}${mark}`,
    )
  }

  // ── 逐节点 ─────────────────────────────────────────────────
  const byNode = new Map<string, NodeStat>()
  for (const r of runs) {
    for (const j of r.judgements) {
      const s = byNode.get(j.node) ?? { node: j.node, right: 0, wrong: 0, unjudged: 0, rightProbs: [], wrongProbs: [], why: [] }
      if (j.verdict === 'right') {
        s.right++
        s.rightProbs.push(j.prob)
      } else if (j.verdict === 'wrong') {
        s.wrong++
        s.wrongProbs.push(j.prob)
        s.why.push(`${r.task.id}：${j.why}`)
      } else {
        s.unjudged++
      }
      byNode.set(j.node, s)
    }
  }
  const stats = [...byNode.values()].sort((a, b) => a.node.localeCompare(b.node))
  const ok = runs.filter((r) => !r.failed)
  const withArtifacts = ok.filter((r) => (r.task.artifacts?.length ?? 0) > 0)

  console.log('')
  console.log(C.bold('  ── 逐节点 ──────────────────────────────────────────────'))
  console.log(C.dim(`  ${cell('节点', 20)}${cell('判定', 6)}${cell('对', 5)}${cell('错', 5)}${cell('未判', 6)}命中率`))
  for (const s of stats) {
    const judged = s.right + s.wrong
    const rate = judged === 0 ? '  —  ' : pct(s.right, judged)
    const colored = s.wrong === 0 ? rate : C.yellow(rate)
    console.log(
      `  ${cell(s.node, 20)}${cell(judged + s.unjudged, 6)}${cell(s.right, 5)}` +
        `${cell(s.wrong, 5, s.wrong ? C.red : undefined)}` +
        `${cell(s.unjudged, 6, s.unjudged ? C.yellow : undefined)}${colored}`,
    )
  }

  // ── 分离度：这张表才是重点 ─────────────────────────────────
  //
  // 命中率高但「对的概率」和「错的概率」重叠 → 阈值救不了（capability gap）
  // 两组分得开 → 调阈值有用（calibration gap）
  console.log('')
  console.log(C.bold('  ── 分离度（对的 vs 错的，各自的概率） ──────────────────'))
  console.log(C.dim('     命中率高不等于判别力强：两组概率重叠的话，任何阈值都救不了'))
  console.log(C.dim(`  ${cell('节点', 20)}${cell('对的(中位)', 12)}${cell('错的(中位)', 12)}差`))
  for (const s of stats) {
    const a = median(s.rightProbs)
    const b = median(s.wrongProbs)
    const gap = a !== undefined && b !== undefined ? (a - b).toFixed(2) : '  —  '
    console.log(`  ${cell(s.node, 20)}${cell(num(a), 12)}${cell(num(b), 12)}${gap}`)
  }

  // ── 错在哪 ─────────────────────────────────────────────────
  const wrongs = stats.flatMap((s) => s.why)
  if (wrongs.length) {
    console.log('')
    console.log(C.bold('  ── 判错的那些 ──────────────────────────────────────────'))
    for (const w of wrongs) console.log(C.red(`  ✗ ${w}`))
  }

  // ── 没测到什么 ─────────────────────────────────────────────
  //
  // **这一节不能省。** 一个不说自己覆盖面的测量结果会被读成「全都测过了」。
  console.log('')
  console.log(C.bold('  ── 没能测到什么 ────────────────────────────────────────'))
  const notes: string[] = []
  for (const s of stats) {
    if (s.right + s.wrong === 0) notes.push(`${s.node}：一条判据都没命中（台子缺它的场景）`)
    else if (s.wrong === 0) {
      notes.push(`${s.node}：${s.right} 条全对，但**没有反例** —— 100% 只说明这批任务上没犯错，不说明它有判别力`)
    }
  }
  const seen = new Set(stats.map((s) => s.node))
  for (const t of ['loop.needsTool', 'loop.pickTool', 'loop.pickInput', 'loop.gradeRisk', 'loop.stepOk', 'loop.isDone', 'loop.canDeliver']) {
    if (!seen.has(t)) notes.push(`${t}：整轮下来一次都没被调用`)
  }
  notes.push('最终回答的正确性用 answerMust/answerMustNot 判，那是**代理**不是正确性本身 —— 回答可以含对标识符却把用途说反')
  notes.push('stepOk 的否定分支需要一个失败的调用才测得到，而 pickInput 的候选只来自真实存在的文件，正常路径上造不出失败调用')
  for (const n of notes) console.log(C.dim(`  · ${n}`))

  console.log('')
  console.log(C.dim(`  样本：${ok.length} 次有效运行（${tasks.length} 条任务${repeat > 1 ? ` × ${repeat} 遍` : ''}${failed.length ? `，另有 ${failed.length} 次没跑起来` : ''}）` +
    `；单节点最多 ${Math.max(0, ...stats.map((s) => s.right + s.wrong))} 次判定` +
    ' —— 这个量级只够看趋势，不够当置信区间用'))

  const totalWrong = stats.reduce((n, s) => n + s.wrong, 0)
  const totalRight = stats.reduce((n, s) => n + s.right, 0)
  console.log('')
  console.log(
    `  ${C.bold('合计')}  ${C.green(String(totalRight))} 对 / ${totalWrong ? C.red(String(totalWrong)) : '0'} 错` +
      C.dim(
        `   （回答合格 ${ok.filter((r) => r.answerPassed).length}/${ok.length}` +
          // ★ 分子分母都只数**有产物的那些任务**。第一版分子数了全部运行、
          //   分母只数有产物的，于是打出「产物合格 11/2」这种读不通的数 ——
          //   一个比例的两边必须来自同一个集合。
          (withArtifacts.length
            ? ` · 产物合格 ${withArtifacts.filter((r) => r.artifactWhy.length === 0).length}/${withArtifacts.length}`
            : '') +
          `${failed.length ? ` · 没跑起来 ${failed.length}` : ''}）`,
      ),
  )
  console.log('')
}

await main()
