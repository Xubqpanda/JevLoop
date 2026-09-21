/**
 * JevLoop · 上下文预算
 *
 * ══════════════════════════════════════════════════════════════
 *  判定帧被小心地 clip 到几百字符、发请求前还校验一遍预算；
 *  而**生成器的 evidence 一个字都不截** —— 那是这个项目里唯一
 *  贵的一步，却成了唯一没有预算的一步。这个文件补的就是这个不对称。
 * ══════════════════════════════════════════════════════════════
 *
 * 机制照抄 DeepSeek Harness 的 `compaction-tool-result-pruner`
 * （MIT，Copyright © 2026 DeepSeek；见 web/LICENSE-DSH）。
 * 它的做法是**留头留尾、换掉中间**：
 *
 *     超过 thresholdChars 的结果
 *       → 保留前 headChars + 后 tailChars，中间换成 PRUNE_MARKER
 *
 * 为什么是头尾而不是只留头：工具输出里**最有用的两端**是开头（它是什么）
 * 和结尾（错误、汇总、最后的匹配）。砍中间比砍尾巴留下的信息多得多。
 *
 * ── 从 DSH 抄来的那条硬校验 ────────────────────────────────────
 *
 * `headChars + PRUNE_MARKER + tailChars` **必须 ≤ `thresholdChars`**，
 * 否则配置期直接抛。理由：裁剪必须**保证真的变小**。不校验的话，
 * 一组「阈值 100、头 80、尾 40」的配置会让裁剪后的文本比原来还长 ——
 * 而那不会报错，只会让上下文越裁越大。
 *
 * 同一个道理还有第二条：裁剪后如果没变小，就当没裁（见 `pruneToolResult`）。
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
 * @module JevLoop/context
 */

// ═══════════════════════════════════════════════════════════
// 单位：字符，不是 token
//
// DSH 也用字符。理由是这个数只需要**可比**：我们要判断的是
// 「这一个结果比那一个大一个量级」，而不是精确计量。真 tokenizer
// 要带一份词表，而零依赖是这个仓库的对外承诺。
//
// `budget.ts` 的 `estimateTokens` 负责把字符折算成 token 做上限判断，
// 那里已经处理了 CJK（中文 1 字 ≈ 1 token）；这里只管字符。
// ═══════════════════════════════════════════════════════════

/**
 * 被剪掉的那段用什么代替。
 *
 * 要能被一眼认出来，且**写明剪了多少** —— 只说「省略」的标记会让
 * 模型以为自己看到的是全部（§8.10 不假装成功）。
 */
export const PRUNE_MARKER_PREFIX = '\n\n[... tool result middle pruned: '

/** 标记的后缀，接在剪掉的字符数后面 */
const PRUNE_MARKER_SUFFIX = ' chars]\n\n'

/**
 * 标记的**最大**长度。
 *
 * 预留 15 位数字 —— 上限是 `Number.MAX_SAFE_INTEGER`（16 位），
 * 而一个字符串不可能有那么多字符。这个数是**上界**，不是估计：
 * 校验要么保证裁剪一定变小，要么就是假的。
 *
 * （第一版这里拍了 24 位，结果一组本来合法的预算被判非法。
 *   预留量拍大了不是「更安全」，是让校验误报。）
 */
const PRUNE_MARKER_MAX_CHARS =
  PRUNE_MARKER_PREFIX.length + 15 + PRUNE_MARKER_SUFFIX.length

export interface PruneBudget {
  /** 超过这个字符数就剪 */
  thresholdChars: number
  /** 保留开头多少字符 */
  headChars: number
  /** 保留结尾多少字符 */
  tailChars: number
}

/**
 * 默认预算。
 *
 * 和 DSH 的默认值一致（8192 / 4096 / 1024）—— 那是给 coding agent 的
 * 工具输出调的，而这里的工具也是读文件和列目录，形状相同。
 *
 * 为什么 head 是 tail 的四倍：开头决定「这是什么」，通常不可压缩；
 * 结尾给的是错误信息和汇总，几句话就够。
 */
export const PRUNE_DEFAULTS: PruneBudget = {
  thresholdChars: 8192,
  headChars: 4096,
  tailChars: 1024,
}

/**
 * 校验一组预算是否自洽。
 *
 * **这条是 DSH 教我抄的。** 没有它，`{threshold: 100, head: 80, tail: 40}`
 * 会让裁剪后的文本（80 + 标记 + 40 > 100）比原来还长 —— 而且不报错，
 * 只是上下文越裁越大。
 *
 * @throws 预算不自洽时
 */
export function resolvePruneBudget(budget: Partial<PruneBudget> = {}): PruneBudget {
  const resolved: PruneBudget = {
    thresholdChars: budget.thresholdChars ?? PRUNE_DEFAULTS.thresholdChars,
    headChars: budget.headChars ?? PRUNE_DEFAULTS.headChars,
    tailChars: budget.tailChars ?? PRUNE_DEFAULTS.tailChars,
  }
  assertInt('thresholdChars', resolved.thresholdChars, 1)
  assertInt('headChars', resolved.headChars, 0)
  assertInt('tailChars', resolved.tailChars, 0)

  const emitted = resolved.headChars + PRUNE_MARKER_MAX_CHARS + resolved.tailChars
  if (emitted > resolved.thresholdChars) {
    throw new Error(
      `PruneBudget: headChars + marker + tailChars (${emitted}) 必须 ≤ thresholdChars (${resolved.thresholdChars})。` +
        `否则「裁剪」会让文本变长 —— 上下文会越裁越大，而且不会报错。`,
    )
  }
  return resolved
}

function assertInt(name: string, value: number, min: number): void {
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`PruneBudget: ${name} (${value}) 必须是 ≥ ${min} 的整数`)
  }
}

export interface PruneResult {
  text: string
  /** 真的剪了吗 */
  pruned: boolean
  removedChars: number
}

/**
 * 一个工具结果进上下文之前过这里。
 *
 * @param text 工具返回的原文
 * @param budget 省略时用 `PRUNE_DEFAULTS`（已经过 `resolvePruneBudget` 校验）
 * @returns 裁剪后的文本；没超阈值就原样返回
 */
export function pruneToolResult(text: string, budget: PruneBudget = PRUNE_DEFAULTS): PruneResult {
  const full = String(text ?? '')
  // Array.from 数的是**码点**不是 UTF-16 单元 —— 中文和 emoji 不会被从中间劈开
  const chars = Array.from(full)
  if (chars.length <= budget.thresholdChars) {
    return { text: full, pruned: false, removedChars: 0 }
  }

  const removed = chars.length - budget.headChars - budget.tailChars
  const marker = `${PRUNE_MARKER_PREFIX}${removed}${PRUNE_MARKER_SUFFIX}`
  const head = chars.slice(0, budget.headChars).join('')
  const tail = budget.tailChars > 0 ? chars.slice(-budget.tailChars).join('') : ''
  const pruned = head + marker + tail

  // 第二条保险：真剪完还是没变小就当没剪。
  // 这能挡住「预算自洽但输入极短」之外的意外组合 —— 与其返回一个更长的
  // 字符串还声称「我帮你省了」，不如老实返回原文。
  if (Array.from(pruned).length >= chars.length) {
    return { text: full, pruned: false, removedChars: 0 }
  }
  return { text: pruned, pruned: true, removedChars: removed }
}

// ═══════════════════════════════════════════════════════════
// 整段上下文的预算
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
  /** 有多少条结果被整条丢了 */
  droppedCount: number
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
 * 把整段 evidence 压进策略。
 *
 * 三档，按代价从低到高：
 *
 *   ① 没到触发线        → **什么都不做**（压缩本身有成本）
 *   ② 逐条剪中间        → 不去掉任何信息，只去掉冗余的中间段
 *   ③ 从最老的整条丢    → 真的丢东西，所以丢最老的
 *
 * DSH 的顺序也是这样：先做不需要模型的那一步，**重新量**，还超才动摘要。
 * 这里没有摘要那一档（原型不做，见文件头）。
 *
 * @returns 压缩后的文本 + 一份账。账要显示给用户（§8.10：丢了什么必须说）
 */
export function fitEvidence(
  parts: readonly string[],
  policy: EvidencePolicy = EVIDENCE_POLICY,
  budget: PruneBudget = PRUNE_DEFAULTS,
): { text: string; report: ContextReport } {
  // 两样都在手上时才校验得动：目标线要容得下单条裁剪后的最大体积，
  // 否则「压到只剩一条」时永远报 overRetain，而那是**不可能满足**的
  assertCompatible(policy, budget)

  const size = (xs: readonly string[]) => xs.reduce((n, x) => n + Array.from(x).length, 0)
  const rawChars = size(parts)

  const base: ContextReport = {
    rawChars,
    keptChars: rawChars,
    prunedCount: 0,
    droppedCount: 0,
    triggerChars: policy.triggerChars,
    retainChars: policy.retainChars,
    acted: false,
    overRetain: false,
  }

  // ① 没到触发线就原样返回
  if (rawChars <= policy.triggerChars) return { text: parts.join('\n'), report: base }

  // ② 每条先剪中间
  let prunedCount = 0
  const trimmed = parts.map((p) => {
    const r = pruneToolResult(p, budget)
    if (r.pruned) prunedCount++
    return r.text
  })

  // ③ 还超目标线就从最老的整条丢。丢最老的，因为最近的结果才是回答要用的
  let kept = trimmed
  let droppedCount = 0
  while (kept.length > 1 && size(kept) > policy.retainChars) {
    kept = kept.slice(1)
    droppedCount++
  }

  const overRetain = size(kept) > policy.retainChars
  const notes: string[] = []
  if (prunedCount > 0) notes.push(`${prunedCount} 条结果剪了中间`)
  if (droppedCount > 0) notes.push(`${droppedCount} 条最早的结果被整条丢弃`)
  if (overRetain) notes.push(`压完仍有 ${size(kept)} 字符，超过目标 ${policy.retainChars}`)

  const text = kept.join('\n') + (notes.length ? `\n\n[... context budget: ${notes.join('，')} ...]` : '')

  return {
    text,
    report: {
      ...base,
      keptChars: Array.from(text).length,
      prunedCount,
      droppedCount,
      // ★ 两种动作都算「动过」。以前只写了 `prunedCount > 0` ——
      //   于是「一条都没剪中间、但整条丢了 39 条」时 `acted` 是 `false`：
      //   丢了条目却说没动过。
      //
      //   `text` 那一面是对的（末尾有「…条最早的结果被整条丢弃…」），
      //   错的是 `report` —— 而 `ContextReport` 存在的意义正是让**程序化消费方**
      //   不必去解析那句散文。谁读 `acted` 谁就得到相反的答案。
      acted: prunedCount > 0 || droppedCount > 0,
      overRetain,
    },
  }
}
