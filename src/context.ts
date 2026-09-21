/**
 * JevLoop · 上下文预算
 *
 * ══════════════════════════════════════════════════════════════
 *  判定帧被小心地 clip 到几百字符、发请求前还校验一遍预算；
 *  而**生成器的 evidence 一个字都不截** —— 那是这个项目里唯一
 *  贵的一步，却成了唯一没有预算的一步。这个文件补的就是这个不对称。
 * ══════════════════════════════════════════════════════════════
 *
 * 「一条结果怎么裁」在 `context-prune.ts`；这个文件管**整段**：
 * 到什么时候动手、压完留多少、丢了什么要怎么说出来。
 *
 * 分开的理由是两者的输入输出形状不同 —— 那边收一个字符串出一个字符串，
 * 对上下文一无所知；这边收一个数组出一份账。
 *

 * ── ⚠️ 状态：**和 `agent.ts` 里的一份实现重叠，待裁定** ──────────
 *
 * 写这个文件的同时，重构执行在 `96ebbfd` 里于 `agent.ts:255-295`
 * 内联解决了同一个问题（证据预算 + 报告省略/截断了什么）。
 * **两份都能跑，所以谁都不会报错** —— 这正是 AGENTS.md §3.1
 * 新加的那条规则说的情形。
 *
 * 按那条规则，这里**不合并**，只记差异、移交裁定：
 *
 * | | `agent.ts` 内联 | 本文件 |
 * |---|---|---|
 * | 位置 | ✅ 每条工具结果进 history 的那一层，粒度对 | ⚠️ 只想放在生成器外层 |
 * | 取哪些 | ✅ 从**最近**往回取 | 从最老的往下丢（结果等价，写法不同） |
 * | 截断方式 | 只留头（`budget.ts` 的 `clip`） | ✅ **留头也留尾**（DSH 的做法） |
 * | 预算自洽校验 | ✗ 没有 | ✅ `head + 标记 + tail ≤ 阈值`，不满足直接抛 |
 * | 双阈值 | ✗ 单阈值 6000 | ✅ trigger / retain 分开，且校验 retain < trigger |
 * | 压不下去时 | ✗ 不报 | ✅ `overRetain` 如实报（对应 DSH 的 fail-closed） |
 *
 * **要合并的话，建议以 `agent.ts` 的位置 + 本文件的截断规则**：
 * 位置是对的（粒度、且判定帧能共用同一份账），截断规则更细。
 * 本文件另外那两条校验是防「越裁越大」和「每轮都压、永远压不下去」的，
 * 单阈值那份挡不住这两种。
 *
 * **本文件目前没有被任何生产代码引用** —— 只被 `tests/context.test.ts` 引用。
 * 这是有意的：接上去会让证据被预算两次。
 *
 * ## 待拆
 *
 * 两件事：**单条结果怎么裁**（`prune*`，已切到 `context-prune.ts`）与
 * **整段预算怎么定**（`fit*`）。`context-prune.ts` 那一步已经做过，
 * 剩下这一半是否继续切排在 `AGENTS.md` §12 的待拆队列里。
 * **行数不在这里写** —— 它会漂（这里曾写 342，实际 441），见 §12。
 *
 * @module JevLoop/context
 */

import { estimateTokens } from './estimate.ts'
import {
  commitSurface,
  foldSeq,
  planReplace,
  surfaceChars,
  type FoldRecord,
  type SurfaceNode,
} from './surface.ts'
import {
  PRUNE_DEFAULTS,
  PRUNE_MARKER_MAX_CHARS,
  assertInt,
  pruneToolResult,
  type PruneBudget,
} from './context-prune.ts'

// 再导出：`agent.ts` 和测试原本从 `context.ts` 拿这几样。
// 拆文件不该让消费方跟着改 —— 那会把一次内部分层变成一次公开 API 变更（§8.12）。
export { pruneToolResult, resolvePruneBudget, PRUNE_DEFAULTS, PRUNE_MARKER_PREFIX } from './context-prune.ts'
export type { PruneBudget, PruneResult } from './context-prune.ts'

// ═══════════════════════════════════════════════════════════
// 整段证据的预算
//
// 这一层是**原型**：它只知道「有多少字符、上限是多少」，
// 不做摘要、不做分段压缩。DSH 在它之上还有一整套
// （surface 折叠 + 摘要式压缩 + provider 确认溢出触发），
// 那些等这里的形状稳定了再说。
// ═══════════════════════════════════════════════════════════

/**
 * 一次运行的工具结果总量策略。
 *
 * ── 两个数，不是一个 ──────────────────────────────────────────
 *
 * DSH 用的是**双阈值**：`thresholdRatio: 0.8` 决定**什么时候动手**，
 * `retainRatio: 0.16` 决定**压完留多少**。分开是必要的 —— 合成一个数
 * 就没法表达「到 80% 才开始压，但一次压到 16%」，而那是实际想要的形状：
 * 压缩本身有成本，不该频繁做；一旦做了就该留足余量，否则下一轮又要压。
 *
 * 这里抄的是同一个形状，只是作用对象不同：
 *
 *   DSH  **contextWindow** 是**每个模型**的容量（实测 262144 / 1000000），
 *        按路由配，管的是「整个上下文占模型多少」。
 *   这里 **triggerChars** 管的是「**一次运行的工具输出**能有多少」——
 *        因为我们还没有按消息分段，手上只有一段 evidence。
 *
 * ⚠️ 这三个数不是一回事，别混。DSH 那边还要再区分第三个：
 * `maxTokens` 是**输出**上限（deepseek-v4-flash 是 256000），
 * 它**不是**上下文预算。把输出上限当上下文预算用，会得到一个
 * 要么永远不触发、要么一触发就砍太狠的阈值。
 *
 * ── 取值 ──────────────────────────────────────────────────────
 *
 * 24000 / 4800 沿用的是 DSH 的比例（0.8 : 0.16，约 5:1）。
 * 绝对值的依据：`read_file` 单次上限 4000 字符、`maxSteps` 默认 12 ——
 * 不设总量上限时最坏约 48000 字符进一次生成调用，而那是唯一贵的一步。
 */
export const EVIDENCE_TRIGGER_CHARS = 24_000
export const EVIDENCE_RETAIN_CHARS = 6_000

export interface EvidencePolicy {
  /** 超过这个总量才动手 */
  triggerChars: number
  /** 压完要落到这个量以下 */
  retainChars: number
}

export const EVIDENCE_POLICY: EvidencePolicy = {
  triggerChars: EVIDENCE_TRIGGER_CHARS,
  retainChars: EVIDENCE_RETAIN_CHARS,
}

/**
 * 校验双阈值自洽。
 *
 * **这条是从 DSH 抄的**（`if (retainTokens >= thresholdTokens) throw`）。
 * 留得比触发线还多的话，「压缩」会在超预算时触发、压完仍然超预算 ——
 * 然后每一轮都再触发一次，每次都花一次钱，却永远压不下去。
 * 那种循环不会报错，只会一直烧。
 */
export function resolveEvidencePolicy(policy: Partial<EvidencePolicy> = {}): EvidencePolicy {
  const resolved: EvidencePolicy = {
    triggerChars: policy.triggerChars ?? EVIDENCE_POLICY.triggerChars,
    retainChars: policy.retainChars ?? EVIDENCE_POLICY.retainChars,
  }
  assertInt('triggerChars', resolved.triggerChars, 1)
  assertInt('retainChars', resolved.retainChars, 0)
  if (resolved.retainChars >= resolved.triggerChars) {
    throw new Error(
      `EvidencePolicy: retainChars (${resolved.retainChars}) 必须 < triggerChars (${resolved.triggerChars})。` +
        `否则压缩会在超预算时触发、压完仍然超预算，于是每一轮都再压一次 —— 永远压不下去，而且只会一直花钱。`,
    )
  }
  return resolved
}

/**
 * 目标线必须**容得下单条结果裁剪后的最大体积**。
 *
 * 这条是**实测发现的**：默认值原本是 retain 4800、单条裁剪后最大
 * 4096 + 标记 58 + 1024 = 5178 —— 于是「压到只剩一条」时仍然报
 * `overRetain`，而那是**永远不可能满足**的：最后一条不会被丢，
 * 而它自己就比目标大。
 *
 * 这是 `head + 标记 + tail ≤ 阈值`（`resolvePruneBudget`）的上一层：
 * 那条保证**单条裁剪会变小**，这条保证**单条裁剪能落进目标**。
 * 两条都要，否则目标线是个摆设 —— 每一轮都会报「压不到」，
 * 而看的人会以为只是这一轮数据太胖。
 */
export function assertCompatible(policy: EvidencePolicy, budget: PruneBudget): void {
  const biggestSingle = budget.headChars + PRUNE_MARKER_MAX_CHARS + budget.tailChars
  if (biggestSingle > policy.retainChars) {
    throw new Error(
      `EvidencePolicy: retainChars (${policy.retainChars}) 小于单条结果裁剪后的最大体积 ` +
        `(${biggestSingle} = head ${budget.headChars} + 标记 ${PRUNE_MARKER_MAX_CHARS} + tail ${budget.tailChars})。` +
        `最后一条永远不会被丢，所以这个目标**永远达不到** —— 每一轮都会报 overRetain。` +
        `要么抬高 retainChars，要么调小单条的 head/tail。`,
    )
  }
}

// 默认值在**模块加载时**自检一次：一组自相矛盾的默认值不该等到
// 第一次真跑起来才被发现，那时它已经在一次生成调用里了。
resolveEvidencePolicy()
assertCompatible(EVIDENCE_POLICY, PRUNE_DEFAULTS)

export interface ContextReport {
  /** 动手前 */
  rawChars: number
  /** 动手后 */
  keptChars: number
  /** 有多少个工具结果被剪了中间 */
  prunedCount: number
  /**
   * 有多少条被**折进了摘要**。
   *
   * 注意不是「丢了」—— 它们的内容还在日志里、轨迹里照样一条不少，
   * 只是不再逐条进生成请求。这个区别是 `surface.ts` 存在的全部理由。
   */
  foldedCount: number
  /** 触发线 */
  triggerChars: number
  /** 目标线 */
  retainChars: number
  /** 真的动手了吗（没到触发线就什么都不做） */
  acted: boolean
  /**
   * 压完仍然超目标线吗。
   *
   * **这个字段必须存在。** DSH 的摘要是 fail-closed 的：摘要不比被替换的
   * 内容小就抛（`summary is not smaller than the shadowed content`），
   * 被 token 上限截断的摘要也算失败。这里的对应物就是它 ——
   * 压不到目标就**如实报出来**，而不是假装压过了。
   */
  overRetain: boolean
}

/**
 * 交给生成器的一条证据。
 *
 * `label` 由**调用方**提供，因为 `context.ts` 不认识工具 —— 它只做预算。
 * 折叠时需要一个能代表这条的短名字，而只有调用方知道那是什么。
 */
export interface EvidencePart {
  /** 完整内容。会被逐条剪中间（`context-prune.ts`） */
  text: string
  /** 极短的名字，折叠成摘要时用。例如 `read_file(a.ts)` */
  label: string
}

/**
 * 把整段证据压进策略。
 *
 * 三档，按代价从低到高：
 *
 *   ① 没到触发线        → **什么都不做**（压缩本身有成本）
 *   ② 逐条剪中间        → 不去掉任何信息，只去掉冗余的中间段
 *   ③ 把最老的一段**折成一个摘要** → 细节看不到，但「做过什么」还在
 *
 * ── 第三档为什么不直接丢 ────────────────────────────────────────
 *
 * 直接丢会**永久抹掉「它做过那一步」**。模型于是不知道自己已经列过目录、
 * 读过哪些文件，就会重做一遍 —— 而重做要花钱。折叠保留的是这件事的**形状**：
 *
 *     [折叠 6 步：list_dir×1、read_file×5 —— a.ts、b.ts、c.ts、d.ts、e.ts]
 *
 * 日志（`ctx.history`）一个字节没动，所以轨迹视图里那 6 步照样一条不少。
 * 这就是 DSH 那个形状：**日志只追加，模型看的是一层折出来的表面**
 * （机制在 `surface.ts`）。
 *
 * @returns 折叠后的文本 + 折叠记录 + 一份账
 */
export function foldEvidence(
  parts: readonly EvidencePart[],
  policy: EvidencePolicy = EVIDENCE_POLICY,
  budget: PruneBudget = PRUNE_DEFAULTS,
): { text: string; folds: FoldRecord[]; report: ContextReport } {
  assertCompatible(policy, budget)

  const nodes: SurfaceNode[] = parts.map((p, i) => ({ seq: i, text: p.text }))
  const rawChars = surfaceChars(nodes)

  const base: ContextReport = {
    rawChars,
    keptChars: rawChars,
    prunedCount: 0,
    foldedCount: 0,
    triggerChars: policy.triggerChars,
    retainChars: policy.retainChars,
    acted: false,
    overRetain: false,
  }

  // ① 没到触发线就原样返回
  if (rawChars <= policy.triggerChars) {
    return { text: parts.map((p) => p.text).join('\n'), folds: [], report: base }
  }

  // ② 每条先剪中间：不去掉信息，只去掉冗余
  let prunedCount = 0
  for (let i = 0; i < nodes.length; i++) {
    const r = pruneToolResult(nodes[i]!.text, budget)
    if (r.pruned) {
      prunedCount++
      nodes[i] = { seq: i, text: r.text }
    }
  }

  // ③ 还超就从最老的开始**折**。吃多少取决于要压到多低，
  //    但**永远留最后一条** —— 最新的结果才是回答要用的（同 DSH 的
  //    「tail-most node is always retained」）。
  const folds: FoldRecord[] = []
  const afterPrune = surfaceChars(nodes)
  if (afterPrune > policy.retainChars && nodes.length > 1) {
    const digestOf = (end: number): string => digestFor(parts.slice(0, end + 1))
    let end = 0
    let digest = digestOf(0)
    while (end < nodes.length - 2) {
      const projected = afterPrune - removedChars(nodes, 0, end) + digest.length
      if (projected <= policy.retainChars) break
      end++
      digest = digestOf(end)
    }

    const removed = removedChars(nodes, 0, end)
    const seq = foldSeq(nodes, 0)
    const plan = planReplace(
      nodes,
      0,
      end,
      { seq, text: digest },
    )
    // 替换必须真的变小，否则那不是折叠（模块头第 3 条不变量）
    if (plan.deltaChars < 0) {
      folds.push({
        fromSeq: parts[0]!.label ? 0 : 0,
        toSeq: end,
        foldedNodes: end + 1,
        removedChars: removed,
        digestChars: digest.length,
      })
      commitSurface(nodes, plan)
    }
  }

  const keptChars = surfaceChars(nodes)
  const overRetain = keptChars > policy.retainChars
  const notes: string[] = []
  if (prunedCount > 0) notes.push(`其中 ${prunedCount} 步的工具输出被截断`)
  if (folds.length > 0) {
    notes.push(`更早的 ${folds[0]!.foldedNodes} 步已折叠成摘要（原文在轨迹里）`)
  } else if (overRetain) {
    // 折到只剩最后一条（或者本来就只有一条）还是超 —— 必须说出来。
    // 以前这里多一个 `&& nodes.length > 1`，于是「单条压不下去」这条
    // **最该被看见**的情形反而不写账：读的人以为没有预算问题。
    notes.push(`压完仍有 ${keptChars} 字符，超过目标 ${policy.retainChars}`)
  }
  const text = nodes.map((n) => n.text).join('\n') + (notes.length ? `\n\n[... context budget: ${notes.join('，')} ...]` : '')

  return {
    text,
    folds,
    report: {
      ...base,
      keptChars: text.length,
      prunedCount,
      foldedCount: folds.reduce((n, f) => n + f.foldedNodes, 0),
      acted: prunedCount > 0 || folds.length > 0,
      overRetain,
    },
  }
}

function removedChars(nodes: readonly SurfaceNode[], startIdx: number, endIdx: number): number {
  let n = 0
  for (let i = startIdx; i <= endIdx; i++) n += nodes[i]!.text.length
  return n
}

/**
 * 折叠摘要。
 *
 * 说清**几步**、**都是什么工具**、以及**碰过哪些目标** —— 前两个让模型
 * 知道自己做到哪儿了，第三个让「再读一遍那个文件」这种指代还能落地。
 *
 * 刻意**不**把内容摘要进来：那需要一次模型调用，而这一档存在的意义正是
 * 「先做不需要模型的那一步」（同 DSH：prune 在 summarise 之前）。
 */
function digestFor(parts: readonly EvidencePart[]): string {
  const byTool = new Map<string, number>()
  const targets: string[] = []
  for (const p of parts) {
    const tool = p.label.split('(')[0] ?? p.label
    byTool.set(tool, (byTool.get(tool) ?? 0) + 1)
    const m = /\(([^)]*)\)/.exec(p.label)
    if (m && m[1]) targets.push(m[1])
  }
  const counts = [...byTool].map(([t, n]) => (n > 1 ? `${t}×${n}` : t)).join('、')
  const seen = targets.slice(0, 20).join('、')
  return `[更早的 ${parts.length} 步已折叠：${counts}${seen ? ` —— 碰过 ${seen}` : ''}]`
}

// ═══════════════════════════════════════════════════════════
// 请求定价：我们估的 vs provider 数的
//
// 上面那套预算管的是**工具证据**。但一次生成请求里还有别的：
// 上文、当前这一句，以及生成器自己那段 system prompt。
// 证据被折到 6000 字符，可不等于整个请求就只有那么大 ——
// 所以「总共多大」必须**量出来**，不能只看被管住的那一块。
// ═══════════════════════════════════════════════════════════

/** 一次生成请求里**我们能控制**的那部分的价格 */
export interface RequestEstimate {
  /** 之前的轮次（问答各算） */
  historyTokens: number
  /** 当前这一句 */
  taskTokens: number
  /** 工具证据 —— 已经被 `foldEvidence` 管住的那一块 */
  evidenceTokens: number
  /** 合计：我们发出去的部分 */
  controlTokens: number
}

/**
 * 给一次生成请求定价。
 *
 * ── 为什么只算「我们能控制的那部分」 ────────────────────────────
 *
 * `system prompt` 住在生成器内部（`HttpGenerator` 的 `DEFAULT_INSTRUCTION`），
 * 外部看不见它 —— 硬要算就得把它复制出来，那份副本必然和真身漂移。
 * 所以这里**只算我们发出去的东西**，system prompt 的代价留在差值里。
 *
 * ── 和真值怎么比 ────────────────────────────────────────────────
 *
 * `GenerateResult.inputTokens` 是 **provider 报的真值**，包含 system prompt。
 * 两个数放在一起看：
 *
 *     provider 报的 inputTokens  −  这里的 controlTokens
 *       ≈ system prompt + 启发式的偏差
 *
 * **不要把这个差值当成 system prompt 的大小** —— 它是两者的和，而启发式
 * 的偏差是 ±30%（见 `estimate.ts`）。它的用途是**看趋势**：证据变大时
 * provider 报的数是不是也跟着涨。涨不动，说明预算没起作用；涨得比估算快，
 * 说明这条启发式在你的内容上偏了，该重新校准。
 *
 * 实测（2026-09-21，deepseek-flash，中文注释 + 英文代码）：
 * 3 文件时 估 37 / 报 103（差 +66，主要是 system prompt 约 60）；
 * 6 文件时 估 1606 / 报 2075（差 +469）—— 差值随规模涨，说明启发式
 * 在**代码**上偏低约 25%（英文代码不是 4 字符 1 token）。
 */
export function priceGenerateRequest(req: {
  task: string
  evidence: string
  history?: readonly { task: string; answer: string }[]
}): RequestEstimate {
  const historyTokens = (req.history ?? []).reduce(
    (n, t) => n + estimateTokens(t.task) + estimateTokens(t.answer),
    0,
  )
  const taskTokens = estimateTokens(req.task)
  const evidenceTokens = estimateTokens(req.evidence)
  return {
    historyTokens,
    taskTokens,
    evidenceTokens,
    controlTokens: historyTokens + taskTokens + evidenceTokens,
  }
}
