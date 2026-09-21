/**
 * JevLoop · 会话日志的编解码
 *
 * ══════════════════════════════════════════════════════════════
 *  一行 JSON ↔ 一轮的运行。**纯函数，不碰磁盘。**
 * ══════════════════════════════════════════════════════════════
 *
 * ── 格式 ────────────────────────────────────────────────────────
 *
 *     {"kind":"header","v":2,"id":"…","createdAt":…,"cwd":"…"}
 *     {"kind":"event","run":0,"at":…,"e":{"type":"run:start",…}}
 *     {"kind":"event","run":0,"at":…,"e":{"type":"decision",…}}
 *     {"kind":"event","run":0,"at":…,"e":{"type":"run:end","answer":"…"}}
 *
 * 第一行是 header，之后**一行一个事件**。DSH 的 JSONL 日志也是这个形状
 * （一行 header + 一行一个事件，见它的 `toHeaderLine` / `eventLine`）。
 *
 * ── 轮次是推导出来的，不是存下来的 ──────────────────────────────
 *
 * `run:start` 给这一轮的任务，`run:end` 给回答和停机原因，中间的事件就是
 * 过程。**没有单独的「轮次」行** —— 存一份摘要加一份事件的话，两份会漂移，
 * 而漂移的方式是「界面上的过程和你看到的回答对不上」，没人会报错。
 *
 * 顺带一个好处：**没跑完的一轮自然可表达** —— 没有 `run:end` 就是没跑完。
 * 崩掉的那次恰恰最需要看它走到哪一步。
 *
 * ── 为什么从 `session-store` 里拆出来 ───────────────────────────
 *
 * 那边 446 行，超了 §12 的 300。而这里确实是一条干净的缝：
 * **「一行 JSON 怎么变成轮次」是纯函数，「文件在哪、怎么找」才有 IO。**
 * 拆开之后这一半能脱离磁盘测，而且 §11 的分层也更干净
 * （这个文件零 IO，所以是 L0）。
 *
 * @module JevLoop/session-log
 */

/**
 * 一轮 = 一次运行。全部从事件里推导，没有一个是单独存的字段。
 */
export interface StoredRun {
  /** 用户那一句 —— 来自这一轮的 `run:start` */
  task: string
  /** agent 那一段 —— 来自 `run:end`。**没有就是这一轮没跑完** */
  answer?: string
  /** 停机原因，同样来自 `run:end` */
  halt?: string
  /**
   * 这一轮的全部事件，按发生顺序。
   *
   * 界面拿它把轨迹重建出来 —— 判定帧、概率、命中的门限、工具输出，
   * 一个不少。**这是存事件而不是存摘要的理由。**
   */
  events: unknown[]
  /** 这一轮最后动过的时间 */
  at: number
}

/** 文件头。会话在磁盘上的**唯一**元数据落点 */
export interface SessionHeaderLine {
  v: number
  id: string
  createdAt: number
  cwd?: string
}

/** 盘上的一行。用判别式而不是 `any` —— 认不出的走 `skipped` */
export type SessionLine =
  | ({ kind: 'header' } & SessionHeaderLine)
  /** v2：一个事件 */
  | { kind: 'event'; run: number; at: number; e: unknown }
  /** v1 遗留：一行一轮，没有事件 */
  | { kind: 'turn'; task: string; answer: string; at: number }

/**
 * 一行 JSON。解析不出来、或者形状不认识，都返回 null。
 *
 * **返回 null 就是「这一行不算数」的信号**，由调用方计进 `skipped` ——
 * 在这里悄悄丢掉的话，一个「少了两轮」的会话看起来和「本来就只有这些」
 * 一模一样（§8.10）。
 */
export function parseSessionLine(raw: string): SessionLine | null {
  const s = raw.trim()
  if (!s) return null
  let v: unknown
  try {
    v = JSON.parse(s)
  } catch {
    // 吞的是「这一行不是完整 JSON」—— 撕裂的尾行就是这样
    return null
  }
  const o = v as Record<string, unknown>
  if (o.kind === 'header' && typeof o.id === 'string' && typeof o.createdAt === 'number') {
    return {
      kind: 'header',
      v: Number(o.v) || 0,
      id: o.id,
      createdAt: o.createdAt,
      cwd: typeof o.cwd === 'string' ? o.cwd : undefined,
    }
  }
  if (o.kind === 'event' && typeof o.run === 'number' && o.e !== undefined) {
    return { kind: 'event', run: o.run, at: Number(o.at) || 0, e: o.e }
  }
  if (o.kind === 'turn' && typeof o.task === 'string' && typeof o.answer === 'string') {
    return { kind: 'turn', task: o.task, answer: o.answer, at: Number(o.at) || 0 }
  }
  return null
}

/** 事件的某个字段。**文件是可以手改的**，所以逐字段验，不硬转 */
function field(e: unknown, key: string): unknown {
  return e && typeof e === 'object' ? (e as Record<string, unknown>)[key] : undefined
}

/**
 * 一个日志文件的全文 → 轮次。
 *
 * **一行坏了不会中断整个文件**：坏行计进 `skipped`，前后照读。这是追加式
 * 格式的全部好处 —— 撕裂最多毁掉最后一行。
 */
export function foldSessionLog(raw: string): {
  runs: StoredRun[]
  header?: SessionHeaderLine
  skipped: number
} {
  const runs: StoredRun[] = []
  let header: SessionHeaderLine | undefined
  let skipped = 0

  for (const line of raw.split('\n')) {
    const parsed = parseSessionLine(line)
    if (!parsed) {
      if (line.trim()) skipped++ // 空行是正常的（末尾一定有），不算跳过
      continue
    }
    if (parsed.kind === 'header') {
      header = { v: parsed.v, id: parsed.id, createdAt: parsed.createdAt, cwd: parsed.cwd }
      continue
    }
    if (parsed.kind === 'turn') {
      // v1 遗留：一轮问答，没有过程。**不假装它有** —— `events` 就是空的
      runs.push({ task: parsed.task, answer: parsed.answer, events: [], at: parsed.at })
      continue
    }

    // v2：把事件挂到它那一轮上。`run` 是服务端给的序号
    const r = (runs[parsed.run] ??= { task: '', events: [], at: parsed.at })
    r.events.push(parsed.e)
    r.at = parsed.at

    // 首尾那两个事件同时是这一轮的元数据 —— **从事件里读，不另存字段**
    const t = field(parsed.e, 'type')
    if (t === 'run:start') {
      const task = field(parsed.e, 'task')
      if (typeof task === 'string') r.task = task
    } else if (t === 'run:end') {
      const answer = field(parsed.e, 'answer')
      const halt = field(parsed.e, 'halt')
      if (typeof answer === 'string') r.answer = answer
      if (typeof halt === 'string') r.halt = halt
    }
  }

  // 序号跳了会留下洞 —— `filter` 把洞去掉，而**洞要报**：少了一轮和
  // 「本来就只有这些」看起来一模一样（§8.10）
  const holes = runs.length - runs.filter(Boolean).length
  return { runs: runs.filter(Boolean), ...(header ? { header } : {}), skipped: skipped + holes }
}
