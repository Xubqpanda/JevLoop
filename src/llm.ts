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
 * ── 待拆 ────────────────────────────────────────────────────────
 *
 * **这个文件确实还能拆，接缝也清楚**，只是那一步比加流式大，不该混在
 * 同一次改动里做。拆法：
 *
 *     seam-generate.ts   L2  **定义角**：GenerateRequest / GenerateResult /
 *                            Generator / DEFAULT_INSTRUCTION
 *     llm-scripted.ts    L2  ScriptedGenerator（离线路径）
 *     llm-http.ts        L2  HttpGenerator（含 #readStream）
 *     llm.ts             L2  RetryingGenerator（装饰器）+ 再导出旧名字
 *
 * **为什么现在没拆**：三个实现都要 import 契约类型，而 §11 不许同层互相
 * 依赖 —— 解法是先切出一个「定义角」（同 `seam-provider.ts` 之于那三个
 * provider 文件的先例，层表里写着）。所以那不是「移动代码」，而是**先决定
 * 契约住哪**，值得单独一次改动来做。
 *
 * 已经先切走的那一块是**线协议**（SSE 帧编解码 → `sse.ts`，L0）：它不
 * 属于任何一个后端，而且两端共用。
 *
 * 「待拆」的登记在伞仓库 `AGENTS.md` §9 —— 上游文件头写「待拆」而没有
 * 登记处，就等于自我豁免（那条规矩本身写在该节里）。
 *
 * @module JevLoop/llm
 */

import type { ConversationTurn } from './conversation.ts'
import type { GenDelta } from './events.ts'
import { ProviderError, httpFailure, transportFailure } from './http-error.ts'
import { decodeSse } from './sse.ts'
import { resolveRetry, retryCall, type RetryOptions } from './retry.ts'

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
  /**
   * **要不要边生成边把文本给我们。**
   *
   * ★ 它放在请求上，不是 `Generator` 接口上，有两个原因：
   *
   *   1. 接口保持「进一段 state、出一段文本」这个窄形状 —— 加一个方法就是
   *      两套调用路径，而重试、降级、记账都得各写两遍。
   *   2. **给不给是调用方的事。** 终端里只想要最后那一段，不需要流式；
   *      浏览器里最慢的那一步恰恰不能是黑的。同一个后端两种用法，
   *      差别就该在参数上。
   *
   * 不传 = 一次拿完整结果（**行为和不支持流式时完全一样**）。
   * 传了也只是「尽力」：后端不吐流式的时候不报错，照样把完整结果给你 ——
   * 所以调用方**不能**依赖回调一定被调用过。
   */
  onDelta?: (d: GenDelta) => void
  /**
   * 这是一次**修订**，值是**给人看的理由**（交付闸门给的那句话）。
   *
   * 不传 = 这一轮第一次生成。生成器自己分不出两者 —— 它只知道「你让我生成
   * 一段文本」，不知道前面有没有一次被闸门否掉的尝试。所以由 loop 说。
   *
   * 它的唯一用途是**让界面能解释那一次擦拭**：修订时已经流出来的字要作废，
   * 而一声不吭地擦掉几百字正是「这个功能坏了」的样子（见 `GenDelta.resetWhy`）。
   */
  revise?: string
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

  /** 拼出这次要「生成」的文本。分出来是为了能**分块发**（见 `generate`）。 */
  #compose(req: GenerateRequest): string {
    const prior = req.history ?? []
    return [
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
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const t0 = performance.now()
    const text = this.#compose(req)

    /*
      ★ **分块吐出去**，而不是等完了再一次性给。
      
      让离线路径（没 key、没网络）也能看见流式 —— 否则这个能力只有在
      配了真后端时才存在，而「clone 下来就能看到全部形状」是这个项目的
      立身之本（README）。总耗时仍然是 `#latencyMs`，只是切成了若干段。

      ⚠️ `size` 至少 1 是**防御性的，今天到不了**：`#compose` 任何情况下
         都至少产出 `task: ` 和 `done:`，所以文本不会是空的。
         留着它是因为 `for (i += size)` 里 size 为 0 就是死循环，而
         `Math.ceil(0 / n)` 恰好是 0 —— 同一天在 `markdown.js` 里刚踩过
         一模一样的坑（三个收集循环不推进游标）。**这个形状的错误在每一层
         都会重新出现一次**，所以宁可留一句不会被执行到的保护。
         （测试没有假装覆盖它 —— 覆盖不到，见 tests/llm-stream.test.ts。）
    */
    const onDelta = req.onDelta
    if (onDelta) {
      const chunks = Math.max(1, Math.ceil(text.length / 24))
      const size = Math.max(1, Math.ceil(text.length / chunks))
      const gap = this.#latencyMs / chunks
      onDelta({ text: '', reset: true, resetWhy: req.revise ? 'revise' : 'start', ...(req.revise ? { resetNote: req.revise } : {}) })
      for (let i = 0; i < text.length; i += size) {
        await new Promise((r) => setTimeout(r, gap))
        onDelta({ text: text.slice(i, i + size), reset: false })
      }
    } else {
      await new Promise((r) => setTimeout(r, this.#latencyMs))
    }

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
          // 不给 `onDelta` 就**一个字节都不加** —— 这条路径的行为和加流式之前
          // 逐字一样，风险为零（见 `GenerateRequest.onDelta`）。
          ...(req.onDelta ? { stream: true, stream_options: { include_usage: true } } : {}),
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

      // ★ **抛带分类的失败**，和判定那条缝**同一套**（`http-error.ts`）。
      //
      //   以前这里是 `throw new Error(\`generate HTTP ${res.status}: …\`)` ——
      //   状态码只活在给人看的那句话里，于是「这个 529 该不该重试」在两条缝上
      //   会得出不同的答案，而它们连的是同一类后端。
      if (!res.ok) throw httpFailure(this.name, res.status, await res.text(), res.headers.get('retry-after'))

      // ★ 按**响应的 content-type** 分流，不是按我们请求里写了什么。
      //
      //   有些部署会忽略 `stream: true`（网关、代理、不认识这个字段的服务端），
      //   那时它照常返回一个完整的 JSON —— 也能用，而且**不该报错**。
      //   判据放在响应上，这两种情况就都走得通。
      if (req.onDelta && res.headers.get('content-type')?.includes('text/event-stream')) {
        return await this.#readStream(req, res, t0)
      }

      const body: any = await res.json()
      const text = body?.choices?.[0]?.message?.content ?? ''

      return {
        text,
        latencyMs: performance.now() - t0,
        inputTokens: body?.usage?.prompt_tokens ?? 0,
        outputTokens: body?.usage?.completion_tokens ?? 0,
        model: this.#model,
      }
    } catch (err) {
      throw transportFailure(this.name, err, ctrl.signal.aborted, this.#timeoutMs)
    } finally {
      clearTimeout(timer)
    }
  }

  /**
   * 读一条 OpenAI 兼容的流式响应。
   *
   * 帧格式（就是 SSE）：
   *
   *     data: {"choices":[{"delta":{"content":"你"}}]}
   *     data: {"choices":[{"delta":{"content":"好"}}]}
   *     data: {"choices":[],"usage":{"prompt_tokens":…}}     ← 最后一个，只要 `include_usage`
   *     data: [DONE]
   *
   * ★ **开头先发一个 `reset`。** 每次 `generate()` 进来都发 —— 重试会再进来
   *   一次，而上次可能已经吐了半句。不重置的话界面上会是两次尝试的文本
   *   首尾相接，而那句话模型**从来没说过**（§8.10 不假装成功）。
   *   修订那条路径（`agent.ts` 的第二次 generate）同样受益。
   */
  async #readStream(req: GenerateRequest, res: Response, t0: number): Promise<GenerateResult> {
    const onDelta = req.onDelta
    if (!onDelta) throw new Error('readStream 只在给了 onDelta 时调用')
    if (!res.body) throw new Error(`流式响应没有 body（content-type 说是流，但读不到）`)

    /*
      ★ 每次 `generate()` 进来都先发一个 `reset`，而**为什么**要说清楚：
      `#readStream` 只在拿到 2xx 的流式响应之后才被调用，所以走到这里
      一定是「真的要开始生成了」。但开始的原因有三种，界面要说的话也不同：

        · 这一轮第一次生成  → 画面上本来就是空的，没什么可解释
        · 同一次生成重试    → 上一次可能已经吐了半句，那些字作废
        · loop 说这是修订  → 上一次整个回答被闸门否掉了，那些字作废
    */
    onDelta({
      text: '',
      reset: true,
      // 「重试」由 `RetryingGenerator` 改写（它才知道这是第几次）——
      // 这里只能分出「第一次」和「loop 说是修订」，见那个类里的说明。
      resetWhy: req.revise ? 'revise' : 'start',
      ...(req.revise ? { resetNote: req.revise } : {}),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    let text = ''
    let usage: any
    let sawDone = false

    for (;;) {
      const { done, value } = await reader.read()
      if (done) break

      // 帧的切分交给 `sse.ts` —— 两端共用一份，「一次 read 正好切在帧中间」
      // 在那里是常态而不是异常。
      const { payloads, rest } = decodeSse(buf + decoder.decode(value, { stream: true }))
      buf = rest

      for (const payload of payloads) {
        // `[DONE]` 是 **OpenAI 的哨兵，不是 SSE 的东西** —— 所以它归这里管，
        // 不归 `sse.ts`。
        if (payload === '[DONE]') {
          sawDone = true
          continue
        }

        let chunk: any
        try {
          chunk = JSON.parse(payload)
        } catch {
          /*
            吞的是：**一行解析不了的载荷**。
            
            为什么没有别的东西会到这里：能进这个循环的只有 `data:` 行的内容，
            而这条流是我们自己请求的、内容是 JSON。剩下的可能就两种 ——
            服务端夹带的非 JSON 行、以及被切断的尾帧。
            
            为什么不用 throw：这时 `text` 里通常已经有几百字的正文了，
            为一个畸形的尾帧把整次生成判失败是更坏的选择。真的少了内容，
            下一步的交付闸门（`canDeliver`）会看见「回答和证据对不上」
            并要求修订 —— 那是能修的那种失败，比整轮作废好。
          */
          continue
        }

        // `usage` 在**最后一个** chunk 上，那时 `choices` 是空的 ——
        // 所以这两件事要分开取，不能只读 `choices[0]`。
        if (chunk.usage) usage = chunk.usage
        const piece: unknown = chunk.choices?.[0]?.delta?.content
        if (typeof piece === 'string' && piece) {
          text += piece
          onDelta({ text: piece, reset: false })
        }
      }
    }

    /*
      ★ **没见到 `[DONE]` 就是被截断了，不许当成功返回。**

      这是流式特有的一种失败：连接**干净地**结束（FIN），而不是断开（RST）。
      前者在 `reader.read()` 上表现为 `done: true`，和「正常读完」长得一模一样 ——
      于是半个回答会被当成完整的回答交出去，而交付闸门拿到的是**残缺的证据**，
      它甚至可能判过。这正是 §8.10 说的「不假装成功」。

      为什么判据是 `[DONE]` 而不是「结束得早不早」：`[DONE]` 是 OpenAI 兼容
      流式的**明确终止符**，而「多长算短」是猜的。代价写在明处 ——
      **如果你的服务端不发 `[DONE]`，这条路会每次都失败**，而错误信息就写着
      这一句，所以它指着原因，不是一团迷雾。重试是可用的（TRANSPORT 可重试）：
      真正的截断是瞬时的，重试一次通常就好了。
    */
    if (!sawDone) {
      throw new ProviderError(
        `${this.name}: the stream ended without [DONE] after ${text.length} chars — the answer is truncated`,
        'TRANSPORT',
      )
    }

    return {
      text,
      latencyMs: performance.now() - t0,
      inputTokens: usage?.prompt_tokens ?? 0,
      outputTokens: usage?.completion_tokens ?? 0,
      model: this.#model,
    }
  }
}

/**
 * 给生成后端加一层有界重试。
 *
 * ★ **这才是 DSH 说的那个「model request」。**
 *
 *   用户问「模型重试该怎么做」时我去看 DSH，它的包叫 `llm-retry`、注释写的是
 *   「provider-routed **model-request** retry policy」—— 而 JevLoop 有两条
 *   通向后端的缝，**生成这条才是「模型请求」**。我先把重试加在了判定那条上
 *   （那条也确实需要，529 会把整轮打到 mock），而这条一直没接。
 *
 *   实测代价（2026-09-21）：`npm run demo` 在一次 `api.deepseek.com` 连接超时上
 *   **直接抛栈退出** —— 判定全都正常跑完了，最后那一次生成挂了，整个 demo 就
 *   失败了，而那是一次**瞬时故障**。
 *
 * 和判定那条缝的区别：生成没有降级链，所以重试是唯一的补救；用尽之后
 * 原样抛出（`agent.ts` 的两处 `generate()` 没有 try，见那里的说明）。
 *
 * ★ **重试和流式并存是安全的，靠的是 `reset`。** 这里原样重放同一个 `req`
 *   （包括 `onDelta`），于是第二次尝试会再发一个 `reset: true` ——
 *   界面把上一次的半句话丢掉重画。所以「重试」在界面上表现为文案**重来一遍**，
 *   而不是两段拼在一起。代价是第一次那半句的 token 白花了，那是真的。
 */
export class RetryingGenerator implements Generator {
  readonly name: string
  readonly #inner: Generator
  readonly #cfg

  constructor(inner: Generator, opts: RetryOptions = {}) {
    this.#inner = inner
    this.name = inner.name
    this.#cfg = resolveRetry(opts)
  }

  generate(req: GenerateRequest): Promise<GenerateResult> {
    let attempt = 0
    /** 到此刻为止，**有没有哪一次尝试真的吐出过字** */
    let emitted = false
    const onDelta = req.onDelta

    /*
      ★ **「重试」这件事只有这里知道。**

      `HttpGenerator.generate` 每次都是被重新调用的，它分不出「第一次」和
      「第三次」—— 而界面要对这两种情况说不同的话（一次是正常开头，一次是
      「刚才那些字作废了」）。所以标记在这里改写。

      ⚠️ **判据是「上一次真的吐出过字」，不是「这是第几次」。** 实测被自己
      的测试抓住过：第一次尝试返回 503，**一个增量都没有**，第二次的 reset
      却标成 `retry` —— 界面于是说「已经流出来的部分已作废」，而根本没有
      东西流出来过。次数不等于「屏幕上有东西可作废」。

      只包 `reset` 那一段：正常增量的热路径（一次回答几百段）不该多绕一层。
    */
    const wrapped: GenerateRequest =
      onDelta === undefined
        ? req
        : {
            ...req,
            onDelta: (d) => {
              const marked = d.reset && attempt > 1 && emitted ? { ...d, resetWhy: 'retry' as const } : d
              if (!d.reset) emitted = true
              onDelta(marked)
            },
          }

    return retryCall(this.name, this.#cfg, () => {
      attempt += 1
      return this.#inner.generate(wrapped)
    })
  }
}
