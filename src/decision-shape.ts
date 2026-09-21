/**
 * JevLoop · DECISION.md 的**形状**
 *
 * 从 `decisiondoc.ts` 切出来 —— 那个文件当时还剩「形状 + 解析」，而**形状可以独立**：
 * 它只描述 `DocBlock` 这一族类型，不认识 Markdown、也不认识策略。
 * 独立之后解析段更纯，而且形状成了 L0 的公共词汇，谁都能引用。
 *
 * **与 `decision-compile.ts` 的区别**：那边不能靠 re-export 兼容，
 * 因为它是 L2、`decisiondoc.ts` 是 L0（L0 不能 import L2）；这里两边都是 L0，
 * 所以 `decisiondoc.ts` **可以** re-export 本文件，消费方一行都不用改。
 *
 * @module JevLoop/decision-shape
 */

// ═══════════════════════════════════════════════════════════
// 形状
// ═══════════════════════════════════════════════════════════

/**
 * `choice` / `noul` / `score` 是单问题的判定，`mixed` 是一个判定问
 * 多件事（实测 `gradeRisk` 同时问危险度和是否需要授权），`rule` 是
 * 代码答的。
 */
export const KINDS = ['choice', 'noul', 'score', 'mixed', 'rule'] as const
export type BlockKind = (typeof KINDS)[number]

/** 问题的原语类型。由选项**写法**推出来，不靠声明 */
export type Primitive = 'choice' | 'noul' | 'score'

export interface DocOption {
  /** 选项 id。会原样进 `criteria` 的键，所以它就是对模型说的话 */
  name: string
  /** 什么情况下选它。这一栏决定判定质量 —— 帧里没有的，模型判不出来 */
  criteria: string
}

export interface DocQuestion {
  /** 问题 id。`prob:<id>` / `score:<id>` / `picked:<id>=..` 指的就是它 */
  id: string
  type: Primitive
  ask: string
  options: DocOption[]
  /** 起始行号（1-based），用于报错定位 */
  line: number
}

export interface DocPolicyRule {
  /** 谓词原文，如 `top >= 0.6`；兜底规则是 `else` */
  when: string
  action: string
}

export interface DocBlock {
  id: string
  kind: BlockKind
  /** 什么时候问。散文，只给人读 */
  when: string
  /**
   * 候选是每步算出来的，不是这份文件里列的。
   *
   * 选项必须每步重建（docs/CODE-STYLE.md §8.4）—— 固定的候选会让模型去选一个已经
   * 不适用的动作。有这一项时，文件里列的选项只是**默认全集或示例**，
   * 校验也不再要求至少两个。
   */
  dynamic: string
  questions: DocQuestion[]
  policy: DocPolicyRule[]
  /** 为什么这么设计。最该写的一段，也是这个文件作为文档的价值所在 */
  rationale: string
  /** 起始行号（1-based），用于报错定位 */
  line: number
}

export interface DocProblem {
  line: number
  message: string
}

export interface DecisionDoc {
  title: string
  /** 第一个 `##` 之前的散文 */
  intro: string
  blocks: DocBlock[]
  /** `## generator` 那一段，原样进 system prompt */
  generatorSection: string
  problems: DocProblem[]
  source: string
}

/** 它是不是一道授权闸门 —— 从策略里推，不要人声明 */
export function isGate(block: DocBlock): boolean {
  return block.policy.some((r) => r.action === 'ask_human')
}

/** 头条数字：这个 agent 有几个判定点、几个问题，其中几个要模型 */
export interface DocSummary {
  blocks: number
  byKind: Record<BlockKind, number>
  questions: number
  byPrimitive: Record<Primitive, number>
  /** 要过判定模型的判定点（非 rule） */
  modelDecisions: number
  /** 纯代码的判定点（rule） */
  codeDecisions: number
  /** 授权闸门 —— 从 policy 推出来的 */
  gates: number
  options: number
}
