/**
 * JevLoop · Policy Engine
 *
 * 概率 → 动作。
 *
 * 这一层是**纯代码**，不碰模型。所以：
 *   · 调一个阈值不需要重跑任何判定
 *   · 同一次运行的答案可以拿去反复试不同的策略
 *   · 策略可以被单元测试覆盖（模型不能）
 *
 * @module JevLoop/policy
 */

import type { AnswerSet } from './vocab.ts'
import type { PolicyRule } from './vocab-decision.ts'
import { confidenceOf } from './vocab.ts'

export interface PolicyOutcome {
  action: string
  reason: string
  /** 命中的规则下标，-1 = 没有规则命中 */
  ruleIndex: number
  /** 求值中发现的问题（策略函数抛异常、兜底规则位置不对） */
  warnings: PolicyWarning[]
}

export interface PolicyWarning {
  level: 'warn' | 'error'
  code: string
  message: string
}

/**
 * 按顺序求值，第一个 when 为真的规则胜出。
 * 没有 when 的规则 = 兜底，必须放最后。
 * 一条都没命中 → action = "escalate"（宁可交给上层，也不要瞎猜一个动作）。
 */
export function resolvePolicy<A extends AnswerSet>(
  rules: PolicyRule<A>[],
  answers: A,
  onWarn?: (w: PolicyWarning) => void,
): PolicyOutcome {
  const warnings: PolicyWarning[] = []
  const emit = (w: PolicyWarning) => {
    warnings.push(w)
    onWarn?.(w)
  }

  // 静态检查：「兜底必须放最后」如果不查，写错了是**静默**的
  const firstCatchAll = rules.findIndex((r) => !r.when)
  if (firstCatchAll >= 0 && firstCatchAll !== rules.length - 1) {
    emit({
      level: 'warn',
      code: 'catch_all_not_last',
      message: `第 ${firstCatchAll + 1} 条是无条件兜底，后面还有 ${rules.length - firstCatchAll - 1} 条规则 —— 那些永远不会被求值`,
    })
  }

  // ★ 静态检查：**兜底缺失同样是静默的**。
  //
  //   没有兜底时函数返回 escalate，但调用方从返回值上分不清
  //   「这组策略压根没打算兜底」和「兜底写了、条件没命中」——
  //   两者的排查方向完全相反。所以这里要主动报一句。
  //
  //   （这条在移植进 JevLoop 时掉过一次，见 REVIEWS 记录。）
  if (firstCatchAll < 0) {
    emit({
      level: 'warn',
      code: 'policy_no_catch_all',
      message: `这组策略（${rules.length} 条）没有无条件兜底规则。一条都没命中时会返回 escalate —— 没有调用方处理 escalate 的话，这一步就静默消失了`,
    })
  }

  for (let i = 0; i < rules.length; i++) {
    const r = rules[i]
    if (!r) continue
    if (!r.when) return { action: r.action, reason: r.reason ?? '兜底规则', ruleIndex: i, warnings }

    let hit = false
    try {
      hit = !!r.when(answers)
    } catch (err) {
      // 不静默：一个拼写错误（a.spamm.noul）和"模型判定不符合阈值"
      // 在日志上必须能区分开，否则排查是场灾难
      emit({
        level: 'warn',
        code: 'when_threw',
        message: `第 ${i + 1} 条（action=${r.action}）的 when 抛异常：${(err as Error)?.message ?? String(err)}。已按'条件不满足'处理`,
      })
    }
    if (hit) return { action: r.action, reason: r.reason ?? `命中第 ${i + 1} 条规则`, ruleIndex: i, warnings }
  }

  return {
    action: 'escalate',
    reason: '没有策略命中，且没有兜底规则 → 交回上层',
    ruleIndex: -1,
    warnings,
  }
}

// ── 策略里最常用的几个判断 ───────────────────────────────────

/**
 * 置信度门限：`when: gte("risky", 0.9)`
 *
 * ⚠️ **不要拿它卡 choice 问题。** Laya 的 `confidence` 是归一化熵
 * （`1 - H(p)/log(k)`，k = 选项个数），不是最大概率：
 *
 *     p = [0.80, 0.20]  →  confidence = 0.269
 *
 * 也就是说同一个阈值在 2 个选项和 20 个选项下含义完全不同 ——
 * 选项越少，要越过同一个门槛需要的概率就越极端。
 * choice 请用下面的 `topGte`。
 */
export const gte =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean =>
    confidenceOf(a[id]) >= threshold

/**
 * 选中项的概率门限：`when: topGte("tool", 0.6)`
 *
 * **choice 问题应该用这个。** 它直接可解释（"选中的那个拿到多少概率质量"），
 * 而且和选项个数无关 —— 加一个选项不会改变门槛的含义。
 */
export const topGte =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean => {
    const ans = a[id]
    if (!ans) return false
    if (ans.type === 'choice') return (ans.probabilities?.[ans.choice] ?? 0) >= threshold
    if (ans.type === 'noul') return Math.max(ans.noul, 1 - ans.noul) >= threshold
    return (ans.confidence ?? 0) >= threshold
  }

/** 布尔概率门限：`when: probGte("risky", 0.7)` */
export const probGte =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean => {
    const ans = a[id]
    return ans?.type === 'noul' ? ans.noul >= threshold : false
  }

/**
 * 布尔概率的「小于」门限。
 *
 * 用在「只有当它**不**成立时才怎样」的规则上，比写 `!probGte(...)` 可读：
 * `when: probLt("ok", 0.5)` 直接读成「不太可能成功」。
 *
 * ⚠️ **它不是 `probGte` 的补集，只在 `noul` 上成立。** 两者对非 `noul` 答案
 * 都返回 `false`（偏保守），所以对 `choice` / `score` 问题，`prob:x < v` 是一条
 * **永远不触发**的规则。这正是 `top < v` 不能用它的原因 —— 见 `topLt`。
 * （`decisiondoc` 侧还应该拒绝把 `prob:` 用在非 noul 问题上，见 REVIEWS-round5。）
 */
export const probLt =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean => {
    const ans = a[id]
    return ans?.type === 'noul' ? ans.noul < threshold : false
  }

/**
 * `topGte` 的**真补集**，对每种答案类型都成立。
 *
 * 为什么需要它：`top < v` 曾经被编译成 `probLt`，而后者只认 `noul`，且对 `noul`
 * 判的是 `p` 而不是 `max(p, 1-p)` —— 于是 `top >= v` 与 `top < v` 在 `p` 偏离 0.5 时
 * **同时为真**（实测 `p=0.05` 两条都真），在 `choice` / `score` 上 `top < v` **恒假**
 * （作者以为写了一道闸门，它一次都不会响）。
 *
 * 定义成 `!topGte(...)` 而不是另写一遍阈值比较：**补集必须从构造上成立**，
 * 靠两处代码各自正确是迟早会分叉的。
 */
export const topLt =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean =>
    !topGte(id, threshold)(a)

/** 分数门限：`when: scoreGte("risk", 2)` */
export const scoreGte =
  (id: string, threshold: number) =>
  (a: AnswerSet): boolean => {
    const ans = a[id]
    return ans?.type === 'score' ? ans.score >= threshold : false
  }

/** 选了某个选项：`when: picked("tool", "read_file")` */
export const picked =
  (id: string, option: string) =>
  (a: AnswerSet): boolean => {
    const ans = a[id]
    return ans?.type === 'choice' ? ans.choice === option : false
  }

/** 取某个选项的概率（做分级审批时用） */
export function probabilityOf(a: AnswerSet, id: string, option?: string): number {
  const ans = a[id]
  if (!ans) return 0
  if (ans.type === 'noul') return ans.noul
  return option ? (ans.probabilities?.[option] ?? 0) : (ans.confidence ?? 0)
}
