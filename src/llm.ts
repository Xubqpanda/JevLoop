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
  /** 额外要求 */
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
      ...(prior.length ? [`（上文 ${prior.length} 轮：${prior.map((t) => t.task).join(' / ')}）`, ``] : []),
      // 折叠摘要也摆出来 —— 脚本生成器要让「上文被折过」这件事**看得见**，
      // 否则它的输出和一个没折过的会话长得一模一样（§8.10）。
      ...(req.historyDigest ? [`（更早的上文已折叠：`, req.historyDigest, `）`, ``] : []),
      `任务：${req.task}`,
      ``,
      `已完成：`,
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
 * HTTP 生成器的默认指令。
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
    const instruction = req.instruction ?? DEFAULT_INSTRUCTION
    const system = req.historyDigest
      ? `${instruction}\n\nEarlier turns in this conversation, folded to their essentials. ` +
        `The full text is not in this request:\n${req.historyDigest}`
      : instruction

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
            { role: 'user', content: `Task:\n${req.task}\n\nWhat was done:\n${req.evidence}` },
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
