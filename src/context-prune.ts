/**
 * JevLoop · 单条工具结果的裁剪
 *
 * ══════════════════════════════════════════════════════════════
 *  超过阈值的结果**留头留尾、换掉中间**。
 *
 *  为什么是头尾而不是只留头：工具输出里**最有用的两端**是开头
 *  （它是什么）和结尾（错误、汇总、最后的匹配）。只截尾正好会把
 *  错误信息扔掉 —— 而那是最不该扔的。
 * ══════════════════════════════════════════════════════════════
 *
 * 机制照抄 DeepSeek Harness 的 `compaction-tool-result-pruner`
 * （MIT，Copyright © 2026 DeepSeek；见 web/LICENSE-DSH）。
 *
 * ── 这个文件只管**一条** ────────────────────────────────────────
 *
 * 「整段证据总共能有多少」是另一件事，在 `context.ts`。
 * 两者分开是因为它们的**输入输出形状不同**：这里收一个字符串、
 * 出一个字符串，对上下文一无所知；那边收一个数组、出一份账。
 * 合在一起时，两条自洽校验（见下）会互相纠缠。
 *
 * ── 两条自洽校验 ────────────────────────────────────────────────
 *
 * 1. `headChars + 标记 + tailChars` **必须 ≤ `thresholdChars`**，否则
 *    配置期直接抛。不校验的话「阈值 100、头 80、尾 40」会让裁剪后的
 *    文本**比原来还长** —— 而那不会报错，只会让上下文越裁越大。
 * 2. 真剪完如果没变小，就当没剪，返回原文。这是第 1 条的纵深防御。
 *
 * @module JevLoop/context-prune
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
export const PRUNE_MARKER_MAX_CHARS =
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

/**
 * 整数校验。两半都要用（单条的预算、整段的策略），所以放在这里 ——
 * 它跟着**单条**那一半，因为 `PRUNE_MARKER_MAX_CHARS` 是它的依据。
 */
export function assertInt(name: string, value: number, min: number): void {
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
// 上面是「一条怎么裁」。
// 「整段证据总共能有多少、超了丢谁、丢了怎么报」是 `context.ts`。
// ═══════════════════════════════════════════════════════════
