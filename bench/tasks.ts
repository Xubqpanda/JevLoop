/**
 * JevLoop · 判定标定台：任务集
 *
 * ══════════════════════════════════════════════════════════════
 *  **这个文件是判据，不是被测对象。**
 * ══════════════════════════════════════════════════════════════
 *
 * README 里那张表说托管 Jev「decisive and correct on every decision」——
 * 那是一次 `npm run demo` 的观感，不是测量。七个判定点里哪个可信、
 * 哪个不可信，没人量过。这个台子就是去量它。
 *
 * ── 任务怎么设计才算「可判」 ────────────────────────────────────
 *
 * 一条任务要能机械地判对错，需要三样：
 *
 *   `required`      必须发生的调用（**顺序不限**）
 *   `answerMust`    最终回答必须包含什么
 *   `answerMustNot` 最终回答必须不含什么
 *
 * ★ **顺序不限是刻意的。** 「先 list 再 read」和「直接 read」都是合法走法，
 *   把其中一种判成错，量出来的就不是判定的质量，而是走法和你预期的一不一样。
 *   所以判据是「这件事做了没有」，不是「第几步做的」。
 *
 * ★ **`answerMust` 是正确性的代理，不是正确性本身。** 一个回答可以包含
 *   `totalOf` 却仍然把它的用途说反。这里选的都是**答案里有唯一标识符**的
 *   问题（函数名、文件名），就是为了让代理尽量紧。这条限制写在报告里。
 *
 * @module JevLoop/tasks
 */

/** 一次期望发生的工具调用 */
export interface ExpectedCall {
  /** 工具名，和 `tools.ts` 里的 `name` 逐字一致 */
  tool: string
  /** 需要挑输入的工具（`read_file` / `write_file`）要写明挑哪个 */
  input?: string
}

export interface BenchTask {
  id: string
  /** 交给 agent 的那句话 */
  task: string
  /** 工作目录里放什么。键是文件名，值是内容 */
  files: Record<string, string>
  /** 必须发生的调用。判定「对不对」全靠它 */
  required: ExpectedCall[]
  /**
   * 允许但不要求出现的工具。出现了**不算错**。
   *
   * 存在的理由：有些动作是「可做可不做」的（先列个目录看看），
   * 把它判成错会让命中率虚低 —— 而虚低的数字比没有数字更糟。
   */
  allowedTools?: string[]
  /** 最终回答必须**全部**匹配 */
  answerMust: RegExp[]
  /** 最终回答必须**全部不**匹配 */
  answerMustNot?: RegExp[]
  /**
   * 跑完之后**必须真的存在于工作目录里**的文件。
   *
   * ★ 这条是量出来的必要，不是想周全：`write` 任务第一版只查回答文本，
   *   而它判了「合格」—— 可那一次 `needsTool` 说的是 `answer`，写操作
   *   根本还没做，**文件从来没被创建**，回答却提到了 `summary.ts`。
   *   只查文本的判据会给一个「说了但没做」的回答打勾。
   */
  artifacts?: { path: string; must?: RegExp }[]
  /** 这条任务想量的是哪个判定点的哪一面。报告里按它分组 */
  probes: string
  /**
   * 这条任务需要 `write_file` 时，写进去的内容。
   *
   * 不给就说明这条任务不该写文件 —— 而 `runAgent` 在没有内容来源时
   * **根本不会把 `write_file` 放进候选**（见 `AgentOptions.provideWriteInput`），
   * 于是判定模型没有机会去选一个 loop 兑现不了的动作。
   */
  writeInput?: string
}

// ── 夹具：两个 TypeScript 文件 + 一个诱饵 markdown ─────────────
//
// 两个 TS 文件是**为了量 `pickInput`** —— 它当初的 bug 是
// `defaultInput` 永远返回 `files[0]`，配合 `toolsFor` 把用过的动作删掉，
// 导致一个生命周期内 `read_file` 只能读第一个文件（审计 N3）。
// 一条任务读不到第二个文件，那个 bug 就量不出来。

const ALPHA = `export interface Order {
  id: string
  total: number
}

/** 汇总一批订单的金额 */
export function totalOf(orders: Order[]): number {
  return orders.reduce((n, o) => n + o.total, 0)
}
`

const BETA = `/** 去掉重复项，保留**首次出现**的顺序 */
export function dedupe(items: string[]): string[] {
  return [...new Set(items)]
}
`

const NOTES = `# 说明

这个目录是判定标定台的夹具，里面有两个 TypeScript 文件。
`

const FIXTURE: Record<string, string> = {
  'alpha.ts': ALPHA,
  'beta.ts': BETA,
  'notes.md': NOTES,
}

export const TASKS: BenchTask[] = [
  {
    id: 'direct',
    task: '1 加 1 等于几？',
    files: FIXTURE,
    // 什么都不用做 —— 量的是 needsTool 的**否定分支**（能不能忍住不调工具）
    required: [],
    answerMust: [/\b2\b/],
    probes: 'needsTool 的否定分支（该直接答）',
  },
  {
    id: 'list',
    task: '工作目录里有哪些文件？只列文件名。',
    files: FIXTURE,
    required: [{ tool: 'list_dir' }],
    answerMust: [/alpha\.ts/, /beta\.ts/, /notes\.md/],
    probes: 'needsTool 的肯定分支 + pickTool 选 list_dir',
  },
  {
    id: 'read-one',
    task: 'alpha.ts 里导出的那个函数叫什么名字？',
    files: FIXTURE,
    // 「直接读」和「先列目录再读」都对
    allowedTools: ['list_dir'],
    required: [{ tool: 'read_file', input: 'alpha.ts' }],
    answerMust: [/totalOf/],
    // 诱饵：答案里出现 beta.ts 的函数说明它读错了文件
    answerMustNot: [/dedupe/],
    probes: 'pickInput 从多个候选里挑对文件',
  },
  {
    id: 'read-both',
    task: '这两个 TypeScript 文件里各导出了一个函数，分别叫什么名字？',
    files: FIXTURE,
    allowedTools: ['list_dir'],
    // ★ 两个都要读 —— 这条量的是 N3 那个 bug：第一个读完还会不会读第二个
    required: [
      { tool: 'read_file', input: 'alpha.ts' },
      { tool: 'read_file', input: 'beta.ts' },
    ],
    answerMust: [/totalOf/, /dedupe/],
    probes: 'pickInput 读完一个还会挑下一个（审计 N3 的回归）',
  },
  {
    id: 'discriminate',
    task: 'beta.ts 里的 dedupe 遇到重复项时保留哪一个？',
    files: FIXTURE,
    allowedTools: ['list_dir'],
    required: [{ tool: 'read_file', input: 'beta.ts' }],
    // 那条注释原文是「保留**首次出现**的顺序」
    answerMust: [/首|顺序|第一次/],
    answerMustNot: [/totalOf/],
    probes: 'pickInput 在两个同类候选里挑对那一个',
  },
  {
    id: 'count-ts',
    task: '这个目录里有几个 TypeScript 文件？它们分别导出了什么？',
    files: FIXTURE,
    allowedTools: ['list_dir'],
    required: [
      { tool: 'read_file', input: 'alpha.ts' },
      { tool: 'read_file', input: 'beta.ts' },
    ],
    answerMust: [/totalOf/, /dedupe/],
    // notes.md 不是 TS，它的内容不该出现在回答里
    answerMustNot: [/夹具/],
    probes: 'needsTool 的肯定分支 + 读完两个才 isDone',
  },
  {
    id: 'write',
    task: '把 alpha.ts 里的 totalOf 函数抄到一个新文件 summary.ts 里。',
    files: FIXTURE,
    allowedTools: ['list_dir'],
    required: [
      { tool: 'read_file', input: 'alpha.ts' },
      { tool: 'write_file', input: 'summary.ts' },
    ],
    // 路径和内容都由台子给（两者都是生成，不属于判定 —— 三分法）。
    // ★ 格式就是 `write_file` 的输入：第一行路径，其余内容
    writeInput: 'summary.ts\nexport function totalOf(orders: { total: number }[]): number {\n  return orders.reduce((n, o) => n + o.total, 0)\n}\n',
    answerMust: [/summary\.ts/],
    // ★ 光看回答不够 —— 写没写出来要看盘上有没有那个文件
    artifacts: [{ path: 'summary.ts', must: /totalOf/ }],
    probes: 'gradeRisk 对写操作的分级（唯一能测到它的任务）',
  },
]
