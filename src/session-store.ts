/**
 * JevLoop · 会话存储
 *
 * ══════════════════════════════════════════════════════════════
 *  **会话落盘。日志只追加，有界的是「喂给模型的那一份」。**
 * ══════════════════════════════════════════════════════════════
 *
 * 在这之前会话只活在服务进程的内存里（一个 `Map`）：重启一次，聊过的
 * 东西全没了，而且**界面上看不出来** —— 列表是空的，和「从来没聊过」
 * 长得一模一样。
 *
 * ── 借 DSH 的三件事 ────────────────────────────────────────────
 *
 * 它的会话层比这里重得多（generation files / lease / zstd / 迁移校验），
 * 但三个想法直接可用：
 *
 * 1. **JSONL，只追加。** 一行一轮，加一轮就是 append 一行。崩在中间最多
 *    留一条**撕裂的尾行**，前面的一行都不会坏 —— 这是追加式格式的核心
 *    好处，也是它值得比「整个文件重写一遍」复杂一点点的原因。
 * 2. **文件头带格式版本。** 第一行是 header，里面有 `v`。将来换格式时
 *    读的人能认出来，而不是把旧文件解析成一堆垃圾。
 * 3. **列表和读取分开。** `list()` 只回摘要，不把每个会话的正文都搬出来。
 *
 * ── 和内核同一条原则 ───────────────────────────────────────────
 *
 * 服务端以前是这样的：
 *
 *     turns.push(...); if (turns.length > MAX_TURNS) turns.splice(0, ...)
 *
 * 也就是**把旧的对话销毁掉**。现在改成：**盘上全留，只在读给 agent 用时
 * 截到最近 N 轮**。日志是记录，有界的是模型看到的那一层 —— 和
 * `surface.ts` 那条「日志只追加，模型看的是一层折出来的表面」是同一件事。
 *
 * 区别很实际：以前你翻不回三天前那个会话里它到底答了什么，因为那几轮
 * 已经被 splice 掉了，而且**没有任何地方记着丢过东西**。
 *
 * ── 安全：会话 id 是不可信输入 ─────────────────────────────────
 *
 * `id` 从 query 来，而它会变成**文件名**。不过这一关的话
 * `?session=../../../../etc/passwd` 就能读写任意路径 —— 和这个服务端
 * 当初踩过的 `?cwd=/etc`（见 `server.ts:58`）是同一类洞。所以 id 的
 * 字符集是**白名单**，不是黑名单。
 *
 * 有 IO，所以放 L2；只依赖 vocab 层的形状，不认识 agent。
 *
 * @module JevLoop/session-store
 */

import { appendFile, mkdir, readdir, readFile, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'

/**
 * 会话 id 的字符白名单。
 *
 * **白名单不是洁癖**：这个字符串会成为文件名的一部分。黑名单（挡 `..`
 * 和 `/`）总会有漏网的编码方式，而白名单只需要回答「它能不能拼出一个
 * 我意料之外的路径」—— 答案是所有字符都出不了这个目录。
 */
const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

/** 当前的文件格式版本。换格式时 +1，并在这里写清怎么读旧的 */
export const SESSION_FORMAT_VERSION = 1

/** 一轮对话。`at` 是写进去的时间 —— 排序和「最后说话是什么时候」都要它 */
export interface StoredTurn {
  task: string
  answer: string
  /** Unix epoch 毫秒 */
  at: number
}

/** 列表里的一行。**不含正文** —— 见模块头第 3 条 */
export interface SessionSummary {
  id: string
  createdAt: number
  updatedAt: number
  /** 盘上有几轮（**不是**喂给模型的那几轮） */
  turns: number
  /** 建这个会话时的工作目录。会话换目录会让上文变得没有意义，所以要记 */
  cwd?: string
  /** 第一句问的是什么 —— 列表里拿它当标题 */
  firstPrompt: string
  /**
   * 读的时候跳过了几行。
   *
   * 撕裂的尾行（写到一半断电）和认不出的 `kind` 都算。**必须报出来**：
   * 悄悄跳过的话，一个「少了两轮」的会话看起来和「本来就只有这些」
   * 一模一样（§8.10）。
   */
  skipped: number
}

/** `list()` 读到的原始一行。用判别式而不是 `any` —— 认不出的走 `skipped` */
type Line =
  | { kind: 'header'; v: number; id: string; createdAt: number; cwd?: string }
  | { kind: 'turn'; task: string; answer: string; at: number }

/**
 * 会话 id 合法性。不合法就抛 —— **不静默替换成一个安全的 id**：
 * 那样调用方会以为自己拿到的还是原来那个会话。
 */
export function assertSessionId(id: string): void {
  if (!ID_PATTERN.test(id)) {
    throw new Error(
      `会话 id 不合法：${JSON.stringify(id.slice(0, 40))} —— ` +
        `只允许 A-Z a-z 0-9 _ -，长度 1–64。它会变成文件名，所以是白名单校验`,
    )
  }
}

/** 一行 JSON。解析不出来、或者形状不认识，都返回 null（由调用方计进 `skipped`） */
function parseLine(raw: string): Line | null {
  const s = raw.trim()
  if (!s) return null
  let v: unknown
  try {
    v = JSON.parse(s)
  } catch {
    // 吞的是「这一行不是完整 JSON」—— 撕裂的尾行就是这样。**由调用方
    // 计数并报出来**，不是在这里悄悄消失（本函数返回 null 就是那个信号）。
    return null
  }
  const o = v as Record<string, unknown>
  if (o.kind === 'header' && typeof o.id === 'string' && typeof o.createdAt === 'number') {
    return { kind: 'header', v: Number(o.v) || 0, id: o.id, createdAt: o.createdAt, cwd: typeof o.cwd === 'string' ? o.cwd : undefined }
  }
  if (o.kind === 'turn' && typeof o.task === 'string' && typeof o.answer === 'string') {
    return { kind: 'turn', task: o.task, answer: o.answer, at: Number(o.at) || 0 }
  }
  return null
}

/**
 * 一个目录里的所有会话。
 *
 * 没有索引文件：**列表直接读目录**。索引会漂移（写会话时忘了更新索引、
 * 手工删了文件），而漂移的索引比慢一点的列表糟得多。代价是 `list()` 要
 * 读每个文件的每一行 —— 本地原型规模下无所谓，真要长到几千个会话时，
 * 该加的是索引**加**一个校验，不是只加索引。
 */
export class SessionStore {
  readonly #dir: string

  constructor(dir: string) {
    this.#dir = dir
  }

  get dir(): string {
    return this.#dir
  }

  #path(id: string): string {
    assertSessionId(id)
    return join(this.#dir, `${id}.jsonl`)
  }

  /**
   * 加一轮。
   *
   * 文件不存在就先写 header —— 所以**一次调用就够**，调用方不需要先
   * 「创建会话」。少一个「忘了创建」的失败模式。
   */
  async append(id: string, turn: StoredTurn, meta: { cwd?: string } = {}): Promise<void> {
    const path = this.#path(id)
    await mkdir(this.#dir, { recursive: true })

    if (!(await exists(path))) {
      const header: Line = {
        kind: 'header',
        v: SESSION_FORMAT_VERSION,
        id,
        createdAt: turn.at,
        ...(meta.cwd ? { cwd: meta.cwd } : {}),
      }
      await appendFile(path, `${JSON.stringify(header)}\n`, 'utf8')
    }
    const line: Line = { kind: 'turn', task: turn.task, answer: turn.answer, at: turn.at }
    await appendFile(path, `${JSON.stringify(line)}\n`, 'utf8')
  }

  /**
   * 读一个会话的问答。
   *
   * @param limit 只要最近这么多轮。**盘上一条都不会少** —— 截断发生在这里，
   *   不在写入时（见模块头）。不传就是全要。
   */
  async load(id: string, limit?: number): Promise<StoredTurn[]> {
    // 文件不存在 = **一个还没有人说过话的会话**，不是错误。
    // 少了这一句，`readFile` 的 ENOENT 会一路抛到 HTTP 处理里，
    // 而调用方（界面刷新后恢复对话）拿到的是一条 500 而不是一段空对话。
    if (!(await exists(this.#path(id)))) return []
    const { turns } = await this.#read(id)
    return limit !== undefined && limit >= 0 ? turns.slice(-limit) : turns
  }

  /** 列表。按最后活动时间倒序 —— 最近说过的排最前 */
  async list(): Promise<SessionSummary[]> {
    let names: string[]
    try {
      names = await readdir(this.#dir)
    } catch {
      // 吞的是「目录还不存在」= 一个会话都还没有。那是**空列表**，
      // 不是错误 —— 第一次跑服务就是这个状态。
      return []
    }
    const out: SessionSummary[] = []
    for (const name of names) {
      if (!name.endsWith('.jsonl')) continue
      const id = name.slice(0, -'.jsonl'.length)
      if (!ID_PATTERN.test(id)) continue
      try {
        const { turns, header, skipped } = await this.#read(id)
        out.push({
          id,
          createdAt: header?.createdAt ?? turns[0]?.at ?? 0,
          updatedAt: turns.at(-1)?.at ?? header?.createdAt ?? 0,
          turns: turns.length,
          ...(header?.cwd ? { cwd: header.cwd } : {}),
          firstPrompt: turns[0]?.task ?? '',
          skipped,
        })
      } catch {
        // 吞的是「这一个文件读不了」（权限、正好被删）。**不跳过它** ——
        // 报成一条读不出来的记录，总比列表里凭空少一个会话好（§8.10）。
        out.push({
          id,
          createdAt: 0,
          updatedAt: 0,
          turns: 0,
          firstPrompt: '',
          skipped: -1,
        })
      }
    }
    return out.sort((a, b) => b.updatedAt - a.updatedAt)
  }

  /** 删一个会话。@returns 真的删掉了吗（不存在返回 false，不抛） */
  async remove(id: string): Promise<boolean> {
    const path = this.#path(id)
    if (!(await exists(path))) return false
    await rm(path)
    return true
  }

  /** 读 + 解析 + 计数。`list()` 和 `load()` 共用，所以两处的口径不会分叉 */
  async #read(id: string): Promise<{ turns: StoredTurn[]; header?: Line & { kind: 'header' }; skipped: number }> {
    const raw = await readFile(this.#path(id), 'utf8')
    const turns: StoredTurn[] = []
    let header: (Line & { kind: 'header' }) | undefined
    let skipped = 0

    // 逐行读，**不认识的行不中断整个文件** —— 一行坏了不该让整个会话消失
    for (const line of raw.split('\n')) {
      const parsed = parseLine(line)
      if (!parsed) {
        // 空行是正常的（末尾一定有），不算跳过
        if (line.trim()) skipped++
        continue
      }
      if (parsed.kind === 'header') header = parsed
      else turns.push({ task: parsed.task, answer: parsed.answer, at: parsed.at })
    }
    return { turns, ...(header ? { header } : {}), skipped }
  }
}

/** 文件在不在。`stat` 抛 ENOENT 就是不在 —— 那是**答案**，不是错误 */
async function exists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch {
    // 吞的只有 ENOENT。别的错误（权限等）在下面的 readFile/appendFile 上
    // 会照样抛出来 —— 这里把它们也报成「不存在」只是让错误晚一步出现，
    // 而晚一步的那个错误会带上路径，更好查。
    return false
  }
}
