/**
 * demo 专用的规则判定器
 *
 * ══════════════════════════════════════════════════════════════
 *  这不是判定模型，是一张写死的规则表。
 * ══════════════════════════════════════════════════════════════
 *
 * 为什么需要它：`npm run demo` 必须**零 key、零网络、离线**就能跑出完整
 * 的 loop。真实判定模型（Jev / Laya）不可达时，用它顶上，否则 demo 会
 * 在第一帧就因为"置信度不足"而正确但无聊地停下。
 *
 * ★ 规则表放在 examples/ 而**不是** src/ —— 这是刻意的：
 *   内核不该内置任何领域启发式。规则属于场景。
 *   换成真实后端时，整个文件可以删掉，内核一行都不用改。
 *
 * 想跑真实判定：`--laya`（本地）或 `--jev`（官方，需 TYPESAFE_API_KEY）。
 */

import type { Provider, DecideRequest, DecideResponse, Answer, ChoiceQuestion, ScoreQuestion } from '../src/index.ts'

/** 工具 → 静态风险基线 */
const RISK: Record<string, number> = {
  list_dir: 0,
  read_file: 0,
  write_file: 1,
  done: 0,
}

const noul = (p: number): Answer => ({ type: 'noul', noul: p })

const choice = (pick: string, all: Record<string, string>): Answer => {
  const ids = Object.keys(all)
  const probabilities: Record<string, number> = {}
  // 选中的给高分，其余平分剩下的概率
  const rest = ids.length > 1 ? 0.3 / (ids.length - 1) : 0
  for (const id of ids) probabilities[id] = id === pick ? 0.7 : rest;
  if (!(pick in probabilities)) return { type: 'choice', choice: ids[0] ?? '', probabilities, confidence: 0 }
  return { type: 'choice', choice: pick, probabilities, confidence: 0.7 }
}

const score = (level: number, legend: string[]): Answer => {
  const probabilities: Record<string, number> = {}
  legend.forEach((_, i) => {
    probabilities[String(i)] = i === level ? 0.88 : 0.12 / Math.max(1, legend.length - 1)
  })
  return { type: 'score', score: level, legend: Object.fromEntries(legend.map((l, i) => [String(i), l])), probabilities, confidence: 0.88 }
}

export class RuleJudge implements Provider {
  readonly name = 'rule-judge'

  async decide(req: DecideRequest): Promise<DecideResponse> {
    const t0 = performance.now()
    await new Promise((r) => setTimeout(r, 4)) // 模拟一次判定的量级
    const state: any = req.state ?? {}
    const answers: Record<string, Answer> = {}

    for (const [id, q] of Object.entries(req.questions)) {
      answers[id] = this.#answer(id, q, state)
    }

    return { answers, provider: this.name, model: 'rules', latencyMs: performance.now() - t0, degraded: true,
      warnings: ['规则判定器（demo 用）：不是真实的判定模型，结论仅供演示 loop'] }
  }

  #answer(id: string, q: any, s: any): Answer {
    // 按**问题 id** 分派 —— 说明一下：真实场景里这段逻辑不存在，
    // 你只需要把 state 写好、把问题问清楚，判定模型自己会答。
    switch (id) {
      case 'needs_tool': {
        // 第一轮必然要动手；已经有结果了就停下来生成
        const steps = s.steps_done ?? 0
        return noul(steps >= 3 ? 0.2 : 0.95)
      }

      case 'tool': {
        const all = (q as ChoiceQuestion).criteria
          // ★ 读决策帧的**实际字段**。这里以前读 `s.recent`，而那个字段
          //   早就不存在了 —— didList 恒为 false，规则判定器只是一直选 list_dir，
          //   靠 `want in all` 的兜底意外地「能用」。toolsFor 不再永久移除
          //   read_file 之后兜底失效，才暴露出来。
          //   （同一类问题：状态帧里没有的信号，判定不出来。）
          const doneStr = String(s.already_done ?? '')
          const didList = doneStr.includes('list_dir')
          const read: string[] = Array.isArray(s.already_read) ? s.already_read : []
          const files: string[] = Array.isArray(s.files_known) ? s.files_known : []
          const unread = files.filter((f) => !read.includes(f))
          const want = !didList ? 'list_dir' : unread.length ? 'read_file' : 'done'
        // 想要的工具可能不在候选里（候选每步重建）—— 退到第一个可用的
        return choice(want in all ? want : Object.keys(all)[0]!, all)
      }

      // 审计 N3 新增：给选定的工具挑一个输入文件。
      // 规则：优先挑还没读过的第一个；都已经读过就挑候选里的第一个。
      case 'file': {
        const all = (q as ChoiceQuestion).criteria
        const ids = Object.keys(all)
        const read: string[] = Array.isArray(s.already_read) ? s.already_read : []
        const unread = ids.find((f) => !read.includes(f))
        return choice(unread ?? ids[0] ?? '', all)
      }

      case 'risk': {
        const lvl = RISK[s.tool] ?? 1
        return score(lvl, (q as ScoreQuestion).criteria)
      }

      case 'needs_auth':
        return noul((RISK[s.tool] ?? 1) >= 2 ? 0.95 : 0.05)

      case 'ok':
        // 输出里有"错误："才算失败
        return noul(String(s.output ?? '').startsWith('错误：') ? 0.05 : 0.92)

      case 'done': {
        const steps: string[] = Array.isArray(s.steps) ? s.steps : []
        return noul(steps.some((x) => String(x).startsWith('read_file')) ? 0.9 : 0.1)
      }

      case 'deliverable':
        return noul(String(s.answer ?? '').length > 20 ? 0.9 : 0.2)

      case 'unsupported':
        return noul(0.08)

      default: {
        // 未知问题：给保守答案，让 policy 的阈值去处理
        if (q.type === 'noul') return noul(0.5)
        if (q.type === 'score') {
          const c = (q as ScoreQuestion).criteria
          return score(Math.floor((c.length - 1) / 2), c)
        }
        return choice(Object.keys((q as ChoiceQuestion).criteria)[0] ?? '', (q as ChoiceQuestion).criteria)
      }
    }
  }
}
