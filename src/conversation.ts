/**
 * JevLoop · 会话表面（多轮的上文）
 *
 * ══════════════════════════════════════════════════════════════
 *  **工具证据一直有预算，上文一直没有 —— 这个文件补上那一条。**
 * ══════════════════════════════════════════════════════════════
 *
 * `context.ts` 管住了**这一轮**的素材（工具结果，触发线 24k 字符）。
 * 但一次生成请求里还有**之前每一轮**的问答，而它是原样铺给生成器的：
 *
 *     messages: [ system, ...history.flatMap(问/答), 这一次的 ]
 *
 * 服务端留最近 12 轮（`server.ts` 的 `MAX_TURNS`），而**一轮的答案长度
 * 没有任何上界** —— 一次「讲讲这个文件」的回答就是几千字符。12 轮乘下来，
 * 光上文就能顶掉整个证据预算。它又**不进决策帧**，所以 `budget.ts` 的
 * `validate()` 同样管不到：帧有预算、证据有预算、上文没有。
 *
 * ── 为什么是折，不是丢 ──────────────────────────────────────────
 *
 * 多轮的全部意义是**指代**：「再读一遍那个文件」里的「那个」。丢了上文，
 * 这句话就无处可指 —— 模型会重新问一遍，或者猜一个（而猜错要花钱重来）。
 * 折成摘要是保留**形状**：问过什么、答出来什么，都还在，只是不再逐字。
 *
 *     recent（逐字）: […… 第 7 轮 第 8 轮]     ← 指代落在这里
 *     digest（折）  : [更早的 6 轮已折叠：…]     ← 只说发生过什么
 *
 * 机制复用 `surface.ts` —— 和证据折叠是**同一套**：日志只追加，模型看的
 * 是一层折出来的表面。**原文一个字节没少**，轨迹视图里照样一条不少。
 *
 * ── 和证据折叠的分工 ────────────────────────────────────────────
 *
 *     证据（`context.ts`）  这一轮**做了什么**    单位是**步**
 *     上文（这里）          之前**问过答过什么**  单位是**轮**
 *
 * 两块是**独立的预算**，不是一个池子：证据再大也不该把上文挤掉 ——
 * 上文是「这句话在说什么」的依据，不是素材。同理，上文再长也不该动证据：
 * 证据是「这次能凭什么回答」的依据。
 *
 * 只 import `surface.ts`（L0），所以放 L1 —— 和 `context.ts` 同层，
 * 但两者互不依赖（§11：L1 之间不许互相依赖）。
 *
 * ── 为什么不能再拆（§12） ──────────────────────────────────────
 *
 * 这个文件是**一件事**：给上文定一个上限，并按它折。里面的四块 ——
 * 策略常量、策略自检、折叠、摘要渲染 —— 谁也离不开谁：
 *
 * · 策略的四个数**只有在折叠的行为下才读得懂**（留尾几轮、从哪端折、
 *   摘要按行编号，都决定了那几个数合不合理）。拆出去读的人会看到一组
 *   没有依据的数字。
 * · 自检就是策略的一部分：它挡的正是「一组分开看都合法、合起来自相
 *   矛盾」的配置，跟常量放在一起才看得出它挡的是什么。
 * · 摘要是折叠的**出口**，不是独立能力 —— 它没有第二个消费者。
 *
 * 大半篇幅是**为什么**（这个是给读的人看的，也是 §3 要求的）。
 * 真到了要拆的那天，缝在「摘要渲染」和「折叠」之间：前者只认轮的数组，
 * 不认预算。但那要等它先长出第二个消费者，而不是先拆一个空壳出来。
 *
 * @module JevLoop/conversation
 */

import { commitSurface, foldSeq, planReplace, surfaceChars, type SurfaceNode } from './surface.ts'

/**
 * 一轮对话。**只有问答，没有中间的工具记录。**
 *
 * 中间过程（判定、工具调用、工具结果）故意不进这里：那是**这一轮**的素材，
 * 由 `evidence` 承载；进了这里的只有「问过什么、答了什么」，
 * 因为多轮要的是**指代关系**（"再读一遍那个文件"里的"那个"），
 * 不是把每一轮的完整过程堆进 prompt —— 那是上下文预算的事，
 * 而预算属于这里和 `context.ts`。
 */
export interface ConversationTurn {
  /** 用户那一句 */
  task: string
  /** agent 当时答的那一段 */
  answer: string
}

export interface ConversationPolicy {
  /** 超过这个字符数才动手 */
  triggerChars: number
  /** 压到这个字符数以下 */
  retainChars: number
  /** 摘要里每轮**问**留多少字符 */
  digestTaskChars: number
  /** 摘要里每轮**答**留多少字符 */
  digestAnswerChars: number
}

/**
 * 上文的触发线 / 目标线。
 *
 * ── 目标线是从**地板**推出来的，不是拍的 ──────────────────────
 *
 * 这个机制有一块**故意不折**的地板：留尾那 `CONVERSATION_KEEP_TAIL` 轮
 * 逐字保留（指代的落点），加上摘要自身的大小。所以
 *
 *     地板 ≈ 留尾 2 轮 × 长回答约 1200  +  摘要上界约 760  ≈ 3160
 *
 * **目标线必须在地板之上**，否则它永远达不成 —— 而「达不成」在账面上
 * 和「压不动」长得一模一样（都是 `overRetain: true`），读的人分不出
 * 是预算画错了还是内容压不下去。这是实测踩出来的：目标线原本是 2400，
 * 而 12 轮 × 1200 字符的留尾就已经是 2406。
 *
 * 取 3200 而不是紧贴 3160：留一点余量，让摘要按需增长时不至于立刻越线。
 *
 * · 它**小于 `EVIDENCE_RETAIN_CHARS`（6000）**，因为它不是素材，是
 *   「这句话在说什么」的依据 —— 一次生成里真正要读的是这次的工具输出。
 * · **触发线是目标线的 4 倍**，和证据那一对同比例（24000 / 6000）——
 *   比例一致意味着两者的「多久压一次」是同一个节奏。倍数太小会让压缩
 *   本身变成常态开销。
 *
 * ── 实测（2026-09-21，中文正文，deepseek-flash） ────────────────
 *
 * **一轮回答的长度跨度是十倍**，这决定了这条线画在哪：
 *
 * | 任务 | 回答 | 一轮合计 |
 * |---|---|---|
 * | 「列出文件、说明定义了哪些函数」 | 约 630 | 约 680 |
 * | 「把每个文件逐行解释一遍」 | 5852 / 6926 | 5927 / 6969 |
 *
 * 服务端 `MAX_TURNS = 12`，所以上文的**上限**是 12 轮：
 *
 * · 短任务下约 **8100 字符**，碰不到 12800 这条线 —— 这是有意的：
 *   压缩有成本，不该在没必要时发生。
 * · 长任务下**两轮就越线**（实测 2 轮 = 12896）。这时它是唯一的约束。
 *
 * ⚠️ **别把「短任务下很少触发」读成「可以删掉」。** 回答长度没有任何
 * 上界，而上面那张表里同一件事的两种问法差了十倍。删掉之后超长上文会
 * 原样进 prompt，而且**不报错** —— 只是账单变贵、指代变糊。
 *
 * ⚠️ **`MAX_TURNS` 和这条线不是一回事**：那边管「记多少轮」，
 * 管不了「一共多少字符」。轮数上限乘以**没有上界的**每轮长度，
 * 仍然是没上界。
 */
export const CONVERSATION_TRIGGER_CHARS = 12_800
export const CONVERSATION_RETAIN_CHARS = 3_200

/**
 * 摘要最多列几轮。
 *
 * ★ **这是摘要能收敛的唯一原因。** 逐轮罗列的话，摘要长度随折掉的轮数
 * **线性增长** —— 折 100 轮会得到一个一万多字符的摘要，比不折还糟，
 * 而「折了就该变小」那条不变量会拦住它，于是变成**永远折不动**：
 * 预算形同虚设，而且账面上看是「压过了」（`acted: false`）。
 *
 * `context.ts` 的 `digestFor` 早就避开了这个坑 —— 它**聚合计数**
 * （`read_file×5 —— 碰过 a.ts…`）并把目标截到 20 个，所以大小不随步数涨。
 * 那一套在轮上不适用（轮的问答不能聚合成计数，指代要靠原话），
 * 所以这里改成**只列最近几轮 + 报总数**。
 *
 * 折掉的东西并没有真的丢：原文在服务端和轨迹里都还在，只是不进这次请求。
 * 抬头会写明「这里只列最近 N 轮」，所以看到的人知道自己没看到全部（§8.10）。
 */
export const CONVERSATION_DIGEST_MAX_LINES = 4

/**
 * 摘要里每轮各留多少字符。
 *
 * 问那句留得多（80）：**指代的落点通常在提问里**（"那个文件呢？"）。
 * 答那句留得少（160）但要够长：它往往才是被指的东西（"它定义了
 * outstanding()"），压太短就查不到那个名字了。
 */
export const CONVERSATION_DIGEST_TASK_CHARS = 80
export const CONVERSATION_DIGEST_ANSWER_CHARS = 160

/**
 * **永远逐字保留的最近轮数。**
 *
 * 这是整个折叠的落点：指代（"那个"、"刚才那个"）指向的是**紧邻的上一轮**，
 * 折掉它等于把这句话的落点抽走 —— 而摘要里的截断文本救不回来，因为
 * 用户问的往往正是细节（"它接受什么参数"）。
 *
 * 留 2 而不是 1：用户可能连着两轮在同一个话题上（先问"定义了哪些函数"，
 * 再问"那它是干什么的"，然后才"它接受什么参数"）。
 */
export const CONVERSATION_KEEP_TAIL = 2

/**
 * 上面那几个数的**打包**，也是调用方唯一需要的那个默认值。
 *
 * 打包而不是让调用方自己拼：四个数是一组，分开传就有传错位的余地 ——
 * 而传错位（比如把 `digestAnswerChars` 传进 `retainChars`）不会报错，
 * 只会让压缩变得莫名其妙。
 */
export const CONVERSATION_POLICY: ConversationPolicy = {
  triggerChars: CONVERSATION_TRIGGER_CHARS,
  retainChars: CONVERSATION_RETAIN_CHARS,
  digestTaskChars: CONVERSATION_DIGEST_TASK_CHARS,
  digestAnswerChars: CONVERSATION_DIGEST_ANSWER_CHARS,
}

/**
 * 这一次折叠的账。
 *
 * 报出来是 §8.10 的要求：折了什么必须让人看见。只写「已压缩」的话，
 * 读的人分不清「本来就只有这些」和「有一半被折掉了」。
 */
export interface ConversationReport {
  /** 动手前，上文一共多少字符 */
  rawChars: number
  /** 动手后（逐字的那几轮 + 摘要） */
  keptChars: number
  /** 一共几轮 */
  rawTurns: number
  /** 还剩几轮是逐字的 */
  keptTurns: number
  /** 有几轮被折进了摘要（**不是丢了** —— 原文还在服务端和轨迹里） */
  foldedTurns: number
  triggerChars: number
  retainChars: number
  /** 真的动手了吗（没到触发线就什么都不做） */
  acted: boolean
  /**
   * 折到只剩 `CONVERSATION_KEEP_TAIL` 轮仍然超目标线吗。
   *
   * **这个字段必须存在**：留尾那几轮是**故意不折**的，所以超线可能是
   * 正确行为而不是失败。不报出来的话，读的人会以为预算没起作用；
   * 报出来才能区分「压不动」和「不该再压」。
   */
  overRetain: boolean
}

export interface FoldedConversation {
  /** 逐字铺给生成器的那几轮，**按时间顺序**（最老的在前） */
  recent: ConversationTurn[]
  /** 更早的若干轮折成的摘要；没折时为 `''` */
  digest: string
  report: ConversationReport
}

/**
 * 把上文折进预算。
 *
 * 两档，和证据折叠同形：
 *
 *   ① 没到触发线 → **什么都不做**（压缩本身有成本）
 *   ② 从最老的开始折成一个摘要，但**永远留最后 `CONVERSATION_KEEP_TAIL` 轮**
 *
 * @throws 策略自相矛盾时（`retainChars >= triggerChars`，或预算不是正整数）。
 *   这类配置错误必须**当场炸**：它们不会报错，只会让每一轮都在压、
 *   而且永远压不到目标线以下 —— 一个持续付费的空转。
 */
export function foldConversation(
  turns: readonly ConversationTurn[],
  policy: ConversationPolicy = CONVERSATION_POLICY,
): FoldedConversation {
  assertPolicy(policy)

  const nodes: SurfaceNode[] = turns.map((t, i) => ({ seq: i, text: turnText(t) }))
  const rawChars = surfaceChars(nodes)
  const base: ConversationReport = {
    rawChars,
    keptChars: rawChars,
    rawTurns: turns.length,
    keptTurns: turns.length,
    foldedTurns: 0,
    triggerChars: policy.triggerChars,
    retainChars: policy.retainChars,
    acted: false,
    overRetain: false,
  }

  // ① 没到触发线就原样返回
  if (rawChars <= policy.triggerChars) {
    return { recent: [...turns], digest: '', report: base }
  }

  // 轮数本来就不比留尾多 → 没有可折的。这不算失败：上文就只有这么点。
  const tail = Math.min(CONVERSATION_KEEP_TAIL, Math.max(1, turns.length))
  if (turns.length <= tail) {
    return {
      recent: [...turns],
      digest: '',
      report: { ...base, acted: false, overRetain: rawChars > policy.retainChars },
    }
  }

  // ② 从最老的开始折。吃多少取决于要压到多低，但**不越过留尾那条线**。
  let end = 0
  let digest = digestOf(turns.slice(0, end + 1), policy)
  while (end < turns.length - tail - 1) {
    const projected = rawChars - removedChars(nodes, 0, end) + digest.length
    if (projected <= policy.retainChars) break
    end++
    digest = digestOf(turns.slice(0, end + 1), policy)
  }

  // 替换必须真的变小，否则那不是折叠（`surface.ts` 模块头第 3 条不变量）。
  // 摘要比被替换的原文还大时**不折** —— 折了就是让问题更大。
  const plan = planReplace(nodes, 0, end, { seq: foldSeq(nodes, 0), text: digest })
  if (plan.deltaChars >= 0) {
    return {
      recent: [...turns],
      digest: '',
      report: { ...base, acted: false, overRetain: rawChars > policy.retainChars },
    }
  }
  commitSurface(nodes, plan)

  const keptTurns = turns.slice(end + 1)
  const keptChars = surfaceChars(nodes)
  return {
    recent: keptTurns,
    digest,
    report: {
      ...base,
      keptChars,
      keptTurns: keptTurns.length,
      foldedTurns: end + 1,
      acted: true,
      overRetain: keptChars > policy.retainChars,
    },
  }
}

/**
 * 折叠摘要。
 *
 * 说清**几轮**、每轮**问过什么**、**答出来什么** —— 前两个让模型知道
 * 自己做到哪儿了，第三个让「再读一遍那个文件」这种指代还有落点。
 *
 * 刻意**不**把内容真正摘要进来：那需要一次模型调用，而这一档存在的意义
 * 正是「先做不需要模型的那一步」（同 DSH：prune 在 summarise 之前，
 * 也同 `context.ts` 的 `digestFor`）。
 */
function digestOf(turns: readonly ConversationTurn[], policy: ConversationPolicy): string {
  const total = turns.length
  // 只列**最近**几轮：紧邻留尾的那几轮最可能被指代，最老的本来就没有落点了
  const shown = turns.slice(Math.max(0, total - CONVERSATION_DIGEST_MAX_LINES))
  const offset = total - shown.length

  const lines = shown.map((t, i) => {
    const task = oneLine(t.task, policy.digestTaskChars)
    const answer = oneLine(t.answer, policy.digestAnswerChars)
    // 编号用**它在折掉那一段里的原始序号**，不是列表里的下标 ——
    // 「第 9 轮」和「第 1 行」是两回事，混起来读的人对不上
    return `${offset + i + 1}. ${task}${answer ? ` → ${answer}` : ''}`
  })

  const head =
    offset > 0
      ? `[更早的 ${total} 轮已折叠成要点，完整原文不在这次请求里；这里只列最近 ${shown.length} 轮]`
      : `[更早的 ${total} 轮已折叠成要点，完整原文不在这次请求里]`

  return [head, ...lines].join('\n')
}

/**
 * 摘要里的一行。
 *
 * ★ **换行必须压平**：摘要是按行编号列出来的，答案里的换行会把列表冲散 ——
 * 模型读到的结构就没了（第 3 轮会看起来像第 2 轮的续行）。
 * 空白压成单个空格，超长照实截断并标出被截了多少（§8.10）。
 */
function oneLine(s: string, maxChars: number): string {
  const flat = String(s ?? '').replace(/\s+/g, ' ').trim()
  if (flat.length <= maxChars) return flat
  return `${flat.slice(0, maxChars)} …[+${flat.length - maxChars}]`
}

/**
 * 一轮的正文 —— 表面上存的就是**将要发出去的那段字**。
 *
 * 只算问答本身，**不算每条消息的固定开销**（role 包装、分隔符）：那部分
 * 与轮数成正比、与内容无关，而 `priceGenerateRequest` 同样不计它 ——
 * 两边口径一致，差值是常数，不会随会话变长而失真。
 */
function turnText(t: ConversationTurn): string {
  return `${t.task ?? ''}\n${t.answer ?? ''}`
}

function removedChars(nodes: readonly SurfaceNode[], startIdx: number, endIdx: number): number {
  let n = 0
  for (let i = startIdx; i <= endIdx; i++) n += nodes[i]!.text.length
  return n
}

/**
 * 策略自检。
 *
 * **为什么在这里而不是只在构造处**：策略是调用方能传的参数，
 * 而这两个数是**相乘**起作用的 —— 任何一对分开看都合法、合起来自相矛盾的
 * 配置都不会报错，只会让每一轮都在压缩。
 */
function assertPolicy(p: ConversationPolicy): void {
  const ints: [string, number][] = [
    ['triggerChars', p.triggerChars],
    ['retainChars', p.retainChars],
    ['digestTaskChars', p.digestTaskChars],
    ['digestAnswerChars', p.digestAnswerChars],
  ]
  for (const [name, value] of ints) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`ConversationPolicy: ${name} (${value}) 必须是 ≥ 1 的整数`)
    }
  }
  if (p.retainChars >= p.triggerChars) {
    throw new Error(
      `ConversationPolicy: retainChars (${p.retainChars}) 必须小于 triggerChars (${p.triggerChars}) —— ` +
        `否则每一轮都在折，而且永远折不到目标线以下`,
    )
  }
}
