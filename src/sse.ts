/**
 * JevLoop · SSE 的帧，**两端共用**
 *
 * ══════════════════════════════════════════════════════════════
 *  这条线协议在这个项目里有**两个方向**，而在此之前它们各写各的：
 *
 *      src/server.ts   写    res.write(`data: ${JSON.stringify(e)}\n\n`)
 *      src/llm.ts      读    自己按 '\n' 切、自己认 'data:' 前缀
 *
 *  两个方向必须对同一份格式达成一致，而它们**没有任何东西逼它们一致** ——
 *  改了一边，另一边不报错，只是收不到东西（或者收到半截）。
 *  这就是把它们放进同一个文件的理由。
 * ══════════════════════════════════════════════════════════════
 *
 * 零 import、零 IO：两边都只是字符串进、字符串出。所以它是最底层（L0），
 * 谁都能依赖它，它不依赖谁（同 `context-prune.ts` / `estimate.ts` 的先例）。
 *
 * ── 为什么值得单独一个文件（而不是留在 llm.ts 里）────────────────
 *
 * **因为不抽出来它就测不了。** 帧的切分要处理的恰恰是那些边角 —— 一次
 * `read()` 拿到半帧、`\r\n` 结尾、多行 `data:`、保活注释行 —— 而它们
 * 只在真服务端上偶发。抽成纯函数之后，那些情况是一行输入的事
 * （见 `tests/sse.test.ts`）。留在 `HttpGenerator` 的私有方法里，
 * 验它就得起一个假 HTTP 服务，于是**没人会去验**。
 *
 * @module JevLoop/sse
 */

/**
 * 编一帧。
 *
 * 一条消息一行 `data:`，空行结尾 —— SSE 的帧格式。
 *
 * ★ **不做事件名分类**（不写 `event:` 字段）：前端只需要一个 `onmessage`，
 *   分流交给 JSON 里的 `type`。这样加新消息类型时协议层不用改。
 *   这个决定和 `server.ts` 的 `openStream` 是同一个，注释也留在那里。
 */
export function encodeSse(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`
}

/**
 * 从缓冲区里切出**完整的帧**，并返回还没收全的那半截。
 *
 * 调用方把新读到的字节接到上次的 `rest` 后面再调一次 —— 于是「一次 read
 * 正好切在帧中间」不需要任何特殊处理，那是常态而不是异常。
 *
 * 被跳过的帧（没有 `data:` 行、或者 `data:` 后面是空的）**不算消息**：
 * SSE 里的注释行（`: keep-alive`）和心跳长这样。它们不是数据。
 *
 * ⚠️ 只认 `\n` 和 `\r\n` 两种行尾，**不认单独的 `\r`**。SSE 规范允许它，
 *    但没有真实服务端这么发；真遇到了表现是**帧切不开**（一直收不到消息），
 *    是看得见的，不是静默丢数据。
 */
export function decodeSse(buffer: string): { payloads: string[]; rest: string } {
  // 先归一 `\r\n`。跨 chunk 的 `\r` 不会被误伤：它后面还没有 `\n`，
  // 替换不成立，它会留在 `rest` 里等下一段。
  let rest = buffer.replace(/\r\n/g, '\n')
  const payloads: string[] = []

  for (;;) {
    const cut = rest.indexOf('\n\n')
    if (cut < 0) return { payloads, rest }

    const frame = rest.slice(0, cut)
    rest = rest.slice(cut + 2)

    // 一帧可以有多行 `data:`，规范要求用 `\n` 拼起来当作一条消息。
    // `data:` 后面那个空格是可选的（`data:x` 和 `data: x` 等价）。
    const data = frame
      .split('\n')
      .filter((l) => l.startsWith('data:'))
      .map((l) => l.slice(5).replace(/^ /, ''))
      .join('\n')

    if (data) payloads.push(data)
  }
}
