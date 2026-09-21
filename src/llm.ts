/**
 * JevLoop · 生成（唯一贵的一步）
 *
 * ★ 这个文件是整个项目里**唯一**调用生成式大模型的地方。
 *
 * loop 里的每一个判断都走判定模型（10ms 量级、近乎免费），
 * 只有「写出一段给人看的文字」才走到这里。
 *
 * 接口故意做得很窄：输入是一段 state，输出是一段文本。
 * 想接 OpenAI / Anthropic / 本地模型都随意 —— 换的是这个文件，
 * 内核和判定一行都不用动。
 *
 * @module JevLoop/llm
 */

import type { ConversationTurn } from './conversation.ts'

/**
 * 一轮对话的形状住在 `conversation.ts`（L1）—— 那里才是**用它**的地方
 * （折叠按轮算预算），这里只转发。
 *
 * 定义留在这里的话，L1 的折叠模块就得反过来依赖 L2，分层方向会倒（§11）。
 */
export type { ConversationTurn }

export interface GenerateRequest {
  /** 任务描述 */
  task: string
  /** 工具执行的历史，作为素材 */
  evidence: string
  /**
   * **system prompt** —— 它现在来自 `DECISION.md` 的 `## generator` 段。
   *
   * ★ 和 `instruction` 分开是**修出来的**，不是一开始就这么设计的：
   *   以前只有一个 `instruction`，而 `HttpGenerator` 拿它当 system prompt、
   *   `ScriptedGenerator` 拿它接在正文后面 —— 同一个字段两种含义。
   *   后果在修订那条路径上：`agent.ts` 把交付闸门的反馈当 `instruction`
   *   传进来，于是**整个 system prompt 被那条反馈替换掉**。实测：
   *
   *     首次调用 system: "Answer the task using only the evidence provided…"
   *     修订调用 system: "上一次的回答没有通过交付闸门：不完整。"
   *
   *   模型于是丢掉了「只用证据」「用任务的语言回答」「不要编」，而它
   *   正要重写的就是那个没通过闸门的回答。
   */
  system?: string
  /**
   * **这一次的额外要求** —— 接在用户消息后面，**不动 system prompt**。
   *
   * 目前只有一处用它：交付闸门要求修订时，把闸门给的理由带过去。
   */
  instruction?: string
  /**
   * 之前的轮次。**多轮的全部意义就在这里** ——
   * 没有它，每一句都是孤立的任务，「再读一遍那个文件」里的"那个"无处可指。
   *
   * 调用方负责给出**有界**的份数（服务端只保留最近若干轮）。
   */
  history?: readonly ConversationTurn[]
  /**
   * 更早的若干轮折成的摘要（`conversation.ts`）。
   *
   * ★ 它和 `history` 是**同一个上文的两个部分**，不是两份东西 ——
   * 从老到新依次是：`historyDigest` → `history[0..]` → 本次的 `task`。
   *
   * 分成两个字段，是因为它们在请求里的**身份不同**：摘要不是任何一轮的
   * 真实发言，把它伪装成一对 user/assistant 消息会让模型以为那是真的
   * 对话记录，进而引用一段从未逐字出现过的话。所以它进 **system**。
   */
  historyDigest?: string
}

export interface GenerateResult {
  text: string
  latencyMs: number
  inputTokens: number
  outputTokens: number
  model: string
}

export interface Generator {
  readonly name: string
  generate(req: GenerateRequest): Promise<GenerateResult>
}

/**
 * 脚本化生成器 —— 让 demo **零 key、零网络、离线可跑**。
 *
 * 它不是假的：它真的根据素材生成文本，只是不做真正的语言推理。
 * 用途是让「判定:模型」这个比例可以被真实地跑出来 ——
 * 判定是**真的** Jev 调用，只有这一步是脚本。
 */
export class ScriptedGenerator implements Generator {
  readonly name = 'scripted'
  #latencyMs: number

  constructor(opts: { latencyMs?: number } = {}) {
    // 模拟一次真实生成调用的量级（几百毫秒到几秒）
    this.#latencyMs = opts.latencyMs ?? 600
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const t0 = performance.now()
    await new Promise((r) => setTimeout(r, this.#latencyMs))

    const prior = req.history ?? []
    const text = [
      ...(prior.length ? [`(earlier turns: ${prior.length} — ${prior.map((t) => t.task).join(' / ')})`, ``] : []),
      // 折叠摘要也摆出来 —— 脚本生成器要让「上文被折过」这件事**看得见**，
      // 否则它的输出和一个没折过的会话长得一模一样（§8.10）。
      ...(req.historyDigest ? [`(older history was folded:`, req.historyDigest, `)`, ``] : []),
      `task: ${req.task}`,
      ``,
      `done:`,
      ...req.evidence
        .split('\n')
        .filter(Boolean)
        .map((l) => `  · ${l}`),
      ``,
      req.instruction ?? '',
    ]
      .join('\n')
      .trim()

    return {
      text,
      latencyMs: performance.now() - t0,
      inputTokens: Math.ceil((req.task.length + req.evidence.length) / 4),
      outputTokens: Math.ceil(text.length / 4),
      model: this.name,
    }
  }
}

/**
 * HTTP 生成器的**兜底** system prompt。
 *
 * ⚠️ 正常路径上轮不到它：`agent.ts` 会把 `DECISION.md` 里 `## generator`
 * 那一段当 `system` 传进来。留着它是为了**直接使用这个类**的人
 * （不经过 `runAgent`、也不带 judging 规格的场景）拿到一句像样的指令，
 * 而不是一个空 system。
 *
 * 两者**不要求文字相同**（DECISION.md 是中英双语），但必须带着同一组
 * **承重规则** —— 只用证据、用任务的语言回答。少任何一条，交付闸门就会
 * 开始要求修订。`tests/decisiondoc.test.ts` 有一条断言盯着这两条。
 *
 * 三件事都是刻意的：**只用给到的证据**（判定节点会拿这条去查「回答里有没有
 * 证据不支持的内容」）、**用任务的语言回答**（问中文答英文会让 canDeliver 判不过）、
 * **不要客套**（客套话会被 canDeliver 判成不完整）。
 */
const DEFAULT_INSTRUCTION =
  'Answer the task using only the evidence provided. Reply in the same language as the task. ' +
  'State only what the evidence supports; do not invent files, functions or results. Be direct.'

/**
 * HTTP 生成器 —— 接任何 OpenAI 兼容的 /chat/completions。
 *
 *     new HttpGenerator({ baseUrl: "https://api.openai.com/v1", apiKey, model: "gpt-..." })
 *     new HttpGenerator({ baseUrl: "http://localhost:11434/v1", model: "qwen3" })   // ollama
 */
export class HttpGenerator implements Generator {
  readonly name: string
  #baseUrl: string
  #apiKey?: string
  #model: string
  #timeoutMs: number

  constructor(opts: { baseUrl: string; apiKey?: string; model: string; name?: string; timeoutMs?: number }) {
    this.#baseUrl = opts.baseUrl.replace(/\/+$/, '')
    this.#apiKey = opts.apiKey
    this.#model = opts.model
    this.name = opts.name ?? `http(${opts.model})`
    this.#timeoutMs = opts.timeoutMs ?? 60_000
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const t0 = performance.now()
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), this.#timeoutMs)

    // 折叠摘要进 **system**，不伪装成一问一答 —— 它不是任何一轮的真实发言。
    // 摘要在前、逐字轮次在后，顺序上就是真实的时间顺序。
    const base = req.system ?? DEFAULT_INSTRUCTION
    const system = req.historyDigest
      ? `${base}\n\nEarlier turns in this conversation, folded to their essentials. ` +
        `The full text is not in this request:\n${req.historyDigest}`
      : base

    try {
      const res = await fetch(`${this.#baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(this.#apiKey ? { authorization: `Bearer ${this.#apiKey}` } : {}),
        },
        body: JSON.stringify({
          model: this.#model,
          messages: [
            { role: 'system', content: system },
            // 之前的轮次按 user/assistant 成对铺开 —— 这是模型唯一能
            // 建立指代关系的方式。只有问答，没有中间过程（见 ConversationTurn）。
            ...(req.history ?? []).flatMap((t) => [
              { role: 'user', content: `Task:\n${t.task}` },
              { role: 'assistant', content: t.answer },
            ]),
            {
              role: 'user',
              // ★ 额外要求接在**用户消息**后面，不覆盖 system prompt（见 `GenerateRequest.system`）
              content:
                `Task:\n${req.task}\n\nWhat was done:\n${req.evidence}` +
                (req.instruction ? `\n\n${req.instruction}` : ''),
            },
          ],
        }),
        signal: ctrl.signal,
      })

      if (!res.ok) throw new Error(`generate HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
      const body: any = await res.json()
      const text = body?.choices?.[0]?.message?.content ?? ''

      return {
        text,
        latencyMs: performance.now() - t0,
        inputTokens: body?.usage?.prompt_tokens ?? 0,
        outputTokens: body?.usage?.completion_tokens ?? 0,
        model: this.#model,
      }
    } finally {
      clearTimeout(timer)
    }
  }
}
