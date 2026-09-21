/**
 * JevLoop · 表面（surface）
 *
 * ══════════════════════════════════════════════════════════════
 *  **日志只追加；模型看到的是一层折出来的表面。**
 * ══════════════════════════════════════════════════════════════
 *
 * 这是从 DeepSeek Harness 的 session surface 抄来的形状
 * （`packages/llm/token-meter/src/surface-fold.ts`，MIT，Copyright © 2026 DeepSeek）。
 *
 * ── 为什么不是「丢掉」 ──────────────────────────────────────────
 *
 * 上下文超预算时，最直接的做法是把最老的东西扔掉。那会**永久抹掉**一件事：
 * 「它做过那一步」。模型于是不知道自己已经列过目录、读过哪些文件，
 * 就会**重做一遍** —— 而重做要花钱。
 *
 * 替换不一样：老的内容被一段**摘要**换掉，模型仍然知道「发生过什么」，
 * 只是看不到细节。日志里那些细节一个字节没少，所以人还能查。
 *
 *     nodes（表面）:  [A B C D E F G H]
 *     一次替换 0..4:  [Σ(A..E) F G H]      ← 表面短了，日志没动
 *
 * ── 三条不变量（都是 DSH 那边用错误换来的） ──────────────────────
 *
 * 1. **替换的范围必须真实存在于表面上。** 找不到就抛，不跳过 ——
 *    DSH 的原话是「committed logs are surface-validated at append time,
 *    so an unresolvable range is log corruption and must fail loud
 *    rather than skip the event」。跳过会让表面和日志悄悄分叉。
 * 2. **`plan` 只读，`commit` 只写。** 分成一对是为了让失败的规划
 *    不留下半改的表面 —— 同一个坏输入重试多少次，结果都一样。
 * 3. **替换必须真的变小。** 变大就不是折叠，是把问题推给下一轮。
 *    这一条由调用方在 `planReplace` 之前保证（它知道自己的预算）。
 *
 * 零 import，所以放 L0：谁都能依赖它，它不依赖谁。
 *
 * @module JevLoop/surface
 */

/** 表面上的一次操作：追加，或者用一个新节点**换掉**一段既有节点 */
export type SurfaceTarget = 'append' | { readonly startIdx: number; readonly endIdx: number }

export interface SurfaceNode {
  /**
   * 锚：这个节点在**日志**里的序号。
   *
   * 表面会变，日志不会 —— 所以指认一件事要靠它，不能靠下标。
   * 折叠掉的那一段在 `seq` 上仍然连续可查。
   */
  readonly seq: number
  /** 模型看到的内容 */
  readonly text: string
}

export interface SurfacePlan {
  readonly node: SurfaceNode
  readonly target: SurfaceTarget
  /** 这次操作净增（正）或净减（负）的字符数 —— 记账用 */
  readonly deltaChars: number
}

/** 一个表面上所有节点的字符总量 */
export function surfaceChars(nodes: readonly SurfaceNode[]): number {
  return nodes.reduce((n, x) => n + x.text.length, 0)
}

/** 计划一次追加。永远合法，所以不返回 null */
export function planAppend(seq: number, text: string): SurfacePlan {
  return { node: { seq, text }, target: 'append', deltaChars: text.length }
}

/**
 * 计划一次替换：用 `node` 换掉 `[startIdx, endIdx]`。
 *
 * @throws 范围不在表面上时。**这是刻意的** —— 见模块头的第 1 条不变量。
 */
export function planReplace(
  nodes: readonly SurfaceNode[],
  startIdx: number,
  endIdx: number,
  node: SurfaceNode,
): SurfacePlan {
  if (
    !Number.isInteger(startIdx) ||
    !Number.isInteger(endIdx) ||
    startIdx < 0 ||
    endIdx < startIdx ||
    endIdx >= nodes.length
  ) {
    throw new Error(
      `surface: 替换范围 ${startIdx}..${endIdx} 不在表面上（长度 ${nodes.length}）—— ` +
        `范围找不到说明表面和日志已经分叉，必须当场炸，不能跳过`,
    )
  }
  let removed = 0
  for (let i = startIdx; i <= endIdx; i++) removed += nodes[i]!.text.length
  return { node, target: { startIdx, endIdx }, deltaChars: node.text.length - removed }
}

/**
 * 落地上一步的计划。**不会失败** —— 所以不会留下改了一半的表面。
 *
 * 直接改 `nodes`（`splice` 是原地操作）：表面是热路径上的小数组，
 * 每次折叠都重建一份不划算。计划已经校验过范围，这里不再查。
 */
export function commitSurface(nodes: SurfaceNode[], plan: SurfacePlan): void {
  if (plan.target === 'append') {
    nodes.push(plan.node)
    return
  }
  const { startIdx, endIdx } = plan.target
  nodes.splice(startIdx, endIdx - startIdx + 1, plan.node)
}

/**
 * 对被折叠的那一段生成一个**摘要节点**的序号。
 *
 * 用被折掉那一段的**第一个** seq —— 于是「这个摘要代替了从 seq=N 开始的
 * 哪一段」是可查的：日志里 seq ≥ N 且不在表面上出现的那些，就是它盖住的。
 */
export function foldSeq(nodes: readonly SurfaceNode[], startIdx: number): number {
  return nodes[startIdx]!.seq
}

/**
 * 一段被折叠掉的日志，换了多大的摘要。
 *
 * 报出来是 §8.10 的要求：折了什么必须让人看见。只写「已压缩」的话，
 * 读的人分不清「本来就只有这些」和「有一半被折掉了」。
 */
export interface FoldRecord {
  /** 被折掉那一段的日志序号范围（含两端） */
  readonly fromSeq: number
  readonly toSeq: number
  /** 被折掉多少个节点 */
  readonly foldedNodes: number
  /** 折掉多少字符 */
  readonly removedChars: number
  /** 换成的摘要有多大 */
  readonly digestChars: number
}
