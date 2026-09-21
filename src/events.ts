/**
 * JevLoop · 事件缝
 *
 * agent loop 把「发生了什么」发成事件，而不是直接打印或直接渲染。
 * 好处是**同一个 loop 可以喂给多个消费者**：
 *
 *     runAgent({ onEvent })  ──▶  终端 · SSE 到浏览器 · 测试断言 · 未来的标定台
 *
 * 这一层是只读的观察面：观察者拿不到取消、改写的句柄，
 * 所以一个观察者出错不会影响 loop 本身（见 {@link fanOut}）。
 *
 * ── 为什么不能再拆 ──────────────────────────────────────────────
 *
 * **这是词汇表，而词汇表的价值在于「一眼看全」。** 消费方（界面、日志、
 * 测试、未来的标定台）都是对 `AgentEvent` 做**穷尽匹配**的 —— 拆成两个
 * 文件之后，「一共有哪些事件」就从一次阅读变成一次翻目录，而漏看一个
 * 成员的代价是运行时才发现的分支缺失。
 *
 * 同样的论证已经用在 `decisions.ts` 上（§12 特意把它写成反例：「七个判定
 * 节点是同一件事，拆开会让『一共有哪些判定』从一眼可见变成翻目录」）。
 * 这个文件是那个论证的另一半：**判定**有哪些是一件事，**发生过什么**
 * 有哪些也是一件事。
 *
 * 它变长是**加流式那一段**造成的，而那三段类型（`GenDelta` / `RunDelta` /
 * `GenerateDelta`）恰恰只有放在一起读才讲得通 —— 分开正好把「每段只加自己
 * 知道的那个字段」这个设计拆散。
 *
 * @module JevLoop/events
 */

import type { GateOverrides } from './gates.ts'
import type { DecisionResult } from './vocab-decision.ts'
import type { QuestionSet, AnswerSet } from './vocab.ts'
import type { AuditRecord, MeterStats } from './vocab-records.ts'

/** 每一步里发生的一件事。判别式联合，新增成员时编译器会逼消费者处理。 */
export type AgentEvent =
  | {
      type: 'run:start'
      task: string
      cwd: string
      at: number
      /**
       * 这一轮用的**门限覆盖**（`<块>.<问题>` → 值）。没有覆盖时**整个字段不出现**。
       *
       * ★ 它必须在这里，因为它是**持久化**的那一份记录：同样的任务跑在默认门限
       *   和跑在覆盖门限上，日志里别的地方一个字节都不差 —— 少了这一条，
       *   事后没有任何办法分辨那一轮到底用了哪个数（§8.10 不假装成功）。
       */
      gates?: GateOverrides
    }
  /** 一次判定完成 —— 这是界面最主要的信息来源 */
  | {
      type: 'decision'
      step: number
      id: string
      /** 实际发给模型的决策帧 */
      state: unknown
      questions: QuestionSet
      answers: AnswerSet
      action: string
      reason: string
      latencyMs: number
      provider: string
      degraded: boolean
      escalate: boolean
      /**
       * 后端自己报的问题 —— 缺了哪几个答案、丢了哪几个、它那边的警告。
       *
       * ★ **`degraded: true` 必须能查到为什么。**
       *
       *   这个字段本来不存在，于是实测（2026-09-21）撞上：某个会话的
       *   **每一次**判定都是 `degraded: true`，而轨迹里、服务端日志里、
       *   事件里**都没有任何地方说缺了什么** —— 排查只能靠手工再发一次请求。
       *
       *   provider 那边一直算着这份清单（`provider-http.ts` 的 `notes`），
       *   `FallbackProvider` 也往里写「主 Provider 失败，降级到 X」。它们
       *   在 `DecideResponse` 上，只是**没有一个消费者**。
       *
       * §8.10：被丢掉的东西要报出来，否则读起来就像「本来就这些」。
       */
      warnings?: string[]
    }
  /**
   * 一步操作**开始了**。
   *
   * ══════════════════════════════════════════════════════════════
   *  ★ 为什么需要它：**上面所有事件都是「做完之后」才发的。**
   * ══════════════════════════════════════════════════════════════
   *
   * `decision` 带着 `latencyMs`、`generate` 带着 token 数 —— 它们都是
   * **记录**。于是「现在在干什么」这个问题，观察者只能拿「上一次干完的是
   * 什么」去猜，而这个猜法在**最长的那一步上错得最久**。
   *
   * 实测（2026-09-21，用户报的）：生成要 2 秒以上，那段时间界面显示的
   * 是「正在判定」—— 因为最后一条事件是 `isDone` 判定。等 `generate`
   * 到了，它把标签改成「正在生成回答」，可那时生成**已经结束**了，
   * 而且下一条 `canDeliver` 立刻又改回「正在判定」。
   *
   * `tool:call` 本来就在跑之前发，所以工具那一段一直是对的。判定和生成
   * 这两段缺「开始」，补上之后三段形状一致。
   *
   * ⚠️ **它刻意不进 `EVENT_META`**（界面的那张分类表）：那一张描述的是
   * **做完的事**，每一行都有耗时和泳道；「开始」没有耗时，进去只会多出
   * 一行空记录。界面在 `onEvent` 里单独处理它。
   */
  | {
      type: 'phase'
      step: number
      /** 这一段在干什么 */
      kind: 'decide' | 'generate'
      /** `kind: 'decide'` 时是判定节点 id，如 `loop.pickTool` */
      id?: string
    }
  /** 判定节点要求人工授权，以及结果 */
  | { type: 'authorize'; step: number; tool: string; reason: string; approved: boolean }
  /** `auto_audit` 承诺的留痕真的发生了 */
  | { type: 'audit'; step: number; record: AuditRecord }
  | { type: 'tool:call'; step: number; tool: string; input: string }
  | { type: 'tool:result'; step: number; tool: string; output: string; ms: number }
  /**
   * 一次生成调用。整个运行里通常只有一次，最多两次（修订）。
   *
   * 三个 token 数放在一起是**刻意的** —— 它们回答不同的问题：
   *
   *   `estimatedInputTokens`  我们按字符启发式估的「我们发出去的那部分」
   *   `inputTokens`           provider 报的**真值**（含生成器内部的 system prompt）
   *   `outputTokens`          provider 报的输出
   *
   * provider 不报 usage 时（脚本生成器）后两个是 0 —— 那是「没量到」，
   * 不是「量到了 0」，所以界面要能把两者分开显示（§8.10）。
   */
  | {
      type: 'generate'
      step: number
      kind: string
      latencyMs: number
      /** provider 报的**输入** token；0 = 没报 */
      inputTokens: number
      /** provider 报的**输出** token；0 = 没报 */
      outputTokens: number
      /** 我们估的「我们能控制的那部分」（上文 + 当前任务 + 证据）。见 `context.ts` */
      estimatedInputTokens: number
    }
  /**
   * 交给生成器的证据被预算压过。
   *
   * **只有真的动了才发**（没超触发线时什么都不做，那没什么可报的）。
   *
   * 为什么它是一个事件而不是一行日志：证据被压掉之后，回答里少了东西 ——
   * 而读的人**看不见少了什么**。没有这个事件，「这次答得不全」会被归因到
   * 模型身上，而不是归因到预算上（§8.10 不假装成功）。
   */
  | {
      type: 'context'
      step: number
      /** 动手前的字符数 */
      rawChars: number
      /** 动手后 */
      keptChars: number
      /** 有多少条结果被剪了中间 */
      prunedCount: number
      /**
       * 有多少条被**折进了摘要**（不是丢了 —— 原文还在轨迹里）。
       *
       * 这个区别是 `surface.ts` 存在的理由：直接丢会让模型不知道
       * 「它已经做过那一步」，于是重做一遍，而重做要花钱。
       */
      foldedCount: number
      /** 折了哪几段。轨迹视图据此标出「这几步被折过」 */
      folds: { toSeq: number; foldedNodes: number; removedChars: number }[]
      /** 压完仍然超过目标线吗 */
      overRetain: boolean
    }
  /**
   * 交给生成器的**上文**（多轮的问答）被折叠过。
   *
   * 和 `context` 是一对，但**单位不同**：`context` 管这一轮的**步**
   * （工具结果），这里管之前的**轮**（问答）。两块是独立预算 ——
   * 证据再大也不该把上文挤掉，反之亦然，所以各有各的事件。
   *
   * 同样**只有真的动了才发**。
   */
  | {
      type: 'conversation'
      /** 折叠发生在 loop 之前，所以步号恒为 0 */
      step: number
      rawChars: number
      keptChars: number
      rawTurns: number
      keptTurns: number
      /** 有几轮被折进摘要（**不是丢了** —— 原文还在服务端和轨迹里） */
      foldedTurns: number
      /** 折到只剩留尾那几轮仍然超线吗（留尾是故意的，所以超线可能是正确的） */
      overRetain: boolean
    }
  /**
   * 一次运行结束。
   *
   * ★ `budget` 是**每轮都发**的，不管预算有没有真的动手。
   *
   *   以前两块预算只在折叠真的发生时报（`context` / `conversation` 事件），
   *   于是**正常运行里界面对预算的感知是零** —— 分不清「证据 500 字符，
   *   离触发线远得很」和「23000，就差一点」。而这两者对「下一次会不会
   *   突然开始折叠」的含义完全不同：前者什么都不用管，后者说明你离
   *   一个会改变回答质量的行为只差一步。
   *
   * ★ 字段**就地写开**，不 import `ContextReport` / `ConversationReport` ——
   *   它们和 `events.ts` 同在 L1，同层不能互相依赖（§11）。这和上面
   *   `context` 事件的做法一致：事件的形状是**契约**，不是别人内部结构的转发。
   *
   * ★ **可选**，因为服务端异常那条路径发不出它（那份合成的事件是在
   *   `runAgent` 外面造的）。界面对缺失必须明说「这次没跑到生成，
   *   没有账目」，而不是画一堆 0（§8.10）。
   */
  | {
      type: 'run:end'
      halt: string
      steps: number
      answer: string
      stats: MeterStats
      budget?: RunBudget
    }

/**
 * 生成过程中的一段文本。**它不属于 `AgentEvent`，这是有意的。**
 *
 * 事件的日志和轨迹记的是**决定**（判定了几次、选了哪个工具）。一段流式文本
 * 不是决定 —— 一次回答有**上千段**（中文一两字一段），把它塞进事件流，等于
 * 用 1000 行把「判定 13 次、模型 1 次」那 13 行淹掉，而那正是这个项目要给人
 * 看的东西。三个理由，按重要性排：
 *
 * **① 它和 `run:end.answer` 冗余。** 日志里别的东西都**只有一份**（不记就
 *   没有了），只有它记不记都能从最终答案复原 —— 「不进日志」不丢信息，
 *   重放一轮会话，答案照样出现在 `run:end` 上。
 * **② 它没有可审计的语义。** 「第 7 段是『的』」不是一件发生过的事。
 * **③ 顺序价值为零。** 事件流的价值在顺序，而增量只有一个合法顺序：按到达次序拼。
 *
 * 所以它走**单独的通道**（`runAgent({ onDelta })`）。这样「不进日志」是
 * **结构上成立**的，而不是靠某个 `if (e.type === …) return` —— 过滤器会被
 * 忘掉，类型不会。
 */
export interface GenerateDelta extends RunDelta {
  type: 'generate:delta'
}

/**
 * loop 交出去的增量：{@link GenDelta} 加上**步号**。
 *
 * 三段拼起来正好对上「谁知道什么」：
 *
 *     GenDelta        生成器给的        （它不知道步号、也不知道往哪发）
 *     RunDelta        loop 补上步号     （它不知道往哪发）
 *     GenerateDelta   线上补上 type 标签（`server.ts` 才知道）
 *
 * 每段只加自己知道的那个字段 —— 合成一个类型的话，生成器就得拿到一个
 * 它无权知道的 `step`，而那种字段迟早会被某处填上一个假值。
 */
export interface RunDelta extends GenDelta {
  /** 属于哪一步的生成 */
  step: number
}

/**
 * 生成过程中的一段文本，**生成器看到的形状**。
 *
 * 和 {@link GenerateDelta} 分开，是因为两边知道的东西不一样：生成器不知道
 * 自己跑在第几步（那是 loop 的事），也不知道自己要往哪条线上发（那是
 * `server.ts` 的事）。它只产出一段文本、外加一个「要不要重来」的标记。
 *
 * 定义留在这里（L1）而不是 `llm.ts`（L2）：层次只许往上依赖，`llm.ts`
 * 能 import 这里，反过来不行。
 */
export interface GenDelta {
  /** 这一段新增的文本 */
  text: string
  /**
   * 真 = **丢掉已经收到的**，从 `text` 重新开始。
   *
   * 两件事会置它，而它们**不是一回事**（见 {@link GenDelta.resetWhy}）：
   * 重试重新发起了一次生成，或者交付闸门要求了一次修订。不重置的话界面上
   * 会是两次生成的文本首尾相接，而那句话模型从来没有说过（§8.10）。
   */
  reset: boolean
  /**
   * **为什么重置**（只在 `reset` 为真时有意义）。
   *
   *   `start`  这一轮第一次生成。**没什么可解释的** —— 画面上本来就是空的
   *   `retry`  同一次生成重试（上一次可能已经吐了半句）
   *   `revise` 交付闸门要求修订，这是一次**新的**生成
   *
   * ★ 为什么要分：**界面上已经流出来的字会被擦掉**。一声不吭地擦掉几百字
   *   正是「这个功能坏了」的样子（实测 2026-09-21 用户报的），而三种情况
   *   要说的话完全不同。给不出原因，界面就只能沉默。
   */
  resetWhy?: 'start' | 'retry' | 'revise'
  /**
   * 一句**给人看的**重置原因。目前只有修订会带：交付闸门给的那句话
   * （形如 `prob:unsupported >= 0.5 → revise`）。
   */
  resetNote?: string
}

/**
 * 一块预算的用量。**事件和界面用的形状**，不是哪份内部报告的转发。
 *
 * 两块预算（工具证据按**步**、上文按**轮**）都用它 —— 它们量的东西不同，
 * 但「用了多少 / 线在哪 / 动手没有」这三个问题是同一个。
 */
export interface BudgetLine {
  /** 动手前用了多少。**触发线比的就是它** */
  rawChars: number
  /** 动手后实际发出去多少；没动手时等于 `rawChars` */
  keptChars: number
  /** 超过它才会动手 */
  triggerChars: number
  /** 动手就压到它以下 */
  retainChars: number
  /** 真的动手了吗 */
  acted: boolean
  /** 压完仍然超目标线吗。**留尾是故意的，所以超线可能是正确行为** */
  overRetain: boolean
  /** 一句话说明动了什么（「剪了 3 条 · 折了 6 步」）。没动手是 `''` */
  note: string
}

/** 一次运行的上下文账：两块预算 + 这次请求的 token 构成 */
export interface RunBudget {
  /** 工具证据那一块（单位：步） */
  evidence: BudgetLine
  /** 上文那一块（单位：轮） */
  conversation: BudgetLine
  /**
   * 这次生成请求的 token 构成，**我们估的**。
   *
   * 和 `generate` 事件上 provider 报的 `inputTokens` 放在一起看才有意义：
   *
   *     报的 − 这里的 total ≈ system prompt + 启发式的偏差
   *
   * 实测那条启发式在**代码**上偏低约 25%（英文代码不是 4 字符 1 token），
   * 所以这个差值不能当成 system prompt 的大小 —— 它的用途是看趋势。
   */
  request: {
    /** 工具证据 */
    evidence: number
    /** 逐字那几轮 + 更早那些轮的摘要 */
    history: number
    /** 当前这一句 */
    task: number
    /** 三者之和 = 我们能控制的那部分 */
    total: number
  }
}

/** 观察者。返回值被忽略，抛出的异常被隔离。 */
export type AgentObserver = (event: AgentEvent) => void

/**
 * 把一次判定的结果转成事件。字段逐个搬，避免把 `DecisionResult` 的内部形状泄漏成事件契约。
 *
 * ★ 步号取自 `d.step`，**不再单独传一个 `step` 参数**。
 *   以前是 `decisionEvent(step, d)`，而两个参数说的是同一件事 ——
 *   同一个事实有两个出处就迟早会分叉（`Decider` 内部还维护着一个 `#step`），
 *   而事件里的步号一旦错位，界面上整条轨迹的对应关系就错了。
 *   `DecisionResult` 本来就带着 `step`，找它要就行。
 */
export function decisionEvent(d: DecisionResult<unknown>): AgentEvent {
  return {
    type: 'decision',
    step: d.step,
    id: d.id,
    state: d.state,
    questions: d.questions,
    answers: d.answers as AnswerSet,
    action: d.action,
    reason: d.reason,
    latencyMs: d.latencyMs,
    provider: d.provider,
    degraded: d.degraded,
    escalate: d.escalate,
    // `degraded` 和它的理由必须**一起**到 —— 分开送的话，读的人拿到一个
    // true 而没有任何下文（这正是它此前缺失的原因）
    ...(d.warnings ? { warnings: d.warnings } : {}),
  }
}

/**
 * 把若干观察者合成一个。
 *
 * **一个观察者抛异常不影响其它观察者，也不影响 loop。** UI 崩了不该让 agent 停下，
 * 但也不能静默 —— 异常通过 `onError` 报出来（默认打到 stderr）。
 */
export function fanOut(
  observers: readonly AgentObserver[],
  onError: (err: unknown, index: number) => void = (err, i) =>
    console.error(`  ▲ 事件观察者 #${i} 抛异常（已隔离）：${(err as Error)?.message ?? String(err)}`),
): AgentObserver {
  return (event) => {
    observers.forEach((o, i) => {
      try {
        o(event)
      } catch (err) {
        onError(err, i)
      }
    })
  }
}
