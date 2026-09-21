/**
 * JevLoop · 会话存储
 *
 * ══════════════════════════════════════════════════════════════
 *  **会话日志 = 事件流本身。轮次是从事件里推导出来的。**
 * ══════════════════════════════════════════════════════════════
 *
 * 在这之前会话只活在服务进程的内存里（一个 `Map`）：重启一次，聊过的
 * 东西全没了，而且**界面上看不出来** —— 列表是空的，和「从来没聊过」
 * 长得一模一样。
 *
 * 后来落了盘，但只存问答。于是刷新之后**轨迹是空的** —— 而轨迹正是这个
 * 项目要展示的东西（「它凭什么这么判断」）。一次运行 50 个事件、约 22 KB，
 * 其中 `decision` 占 18 KB：那些帧、概率、命中的门限，才是最该留下来的。
 *
 * ── 布局：仿 DSH ────────────────────────────────────────────────
 *
 *     <root>/<projectKey(cwd)>/<encodeSegment(id)>/log.jsonl
 *
 * 「属于哪个项目」是**结构**，不是一次查询的结果。位置规则见
 * `session-path.ts`（那里也写了为什么不只记在 header 里）。
 *
 * ── 为什么存事件而不是存摘要 ────────────────────────────────────
 *
 * 存一份「问答摘要」+ 一份「事件」的话，两份会漂移，而漂移的方式是
 * 「界面上的过程和你看到的回答对不上」，没人会报错。
 *
 * 所以只存**事件**，轮次从里面读出来：
 *
 *     run:start  → 这一轮问的是什么
 *     …事件…      → 过程
 *     run:end    → 答的是什么、为什么停
 *
 * 这不是新发明的形状 —— 它就是这个 loop 已经在发的那个流（`events.ts`）。
 * DSH 的日志也是同一件事：一行 header + 一行一个事件。
 *
 * ── 三条不变量 ──────────────────────────────────────────────────
 *
 * 1. **日志只追加。** 崩溃最多留一条撕裂的尾行，前面一行都不会坏。
 *    所以一轮跑一半也留着 —— 恰恰是崩掉的那次最需要看它走到哪一步。
 * 2. **盘上全留，只在读给 agent 用时截断。**
 * 3. **跳过的行要报出来**（§8.10）。
 *
 * ── 为什么不能再拆（§12）────────────────────────────────────────
 *
 * 这个文件 300 行出头，而**能拆的已经拆走了**：
 *
 *     session-path.ts     (L0)  编码和布局，纯字符串函数
 *     session-log.ts      (L0)  一行 JSON ↔ 轮次，纯编解码
 *     session-migrate.ts  (L2)  v1→v2 的搬迁，服务端启动时调一次
 *
 * 剩下的六个方法是**一件事**：文件在哪（`#locate`）、怎么读（`#read`）、
 * 怎么写（`append`）、列出来（`list`）、删掉（`remove`）。它们共用
 * `#dir` 和同一套错误口径；拆开就要把 `#locate` 变成公开 API —— 那是
 * 拿暴露内部结构换行数，不划算。
 *
 * **下次要拆的话，缝在 `list()`**：它遍历两层目录并为每一行算摘要，
 * 和「读写某一个会话」不是同一件事，而且性能特征也不同（会话多了它先慢）。
 *
 * ── 迁移不在这里 ────────────────────────────────────────────────
 *
 * v1（扁平布局）→ v2（按项目分层）的搬迁在 `session-migrate.ts`，由
 * **服务端启动时**调一次。不放在这里有两个理由：它是**部署**的事不是
 * **读写**的事；以及 §11 不许同层互相依赖（两个都是 L2）。
 * store 因此完全不知道历史上还有过别的布局。
 *
 * @module JevLoop/session-store
 */

import { appendFile, mkdir, readdir, readFile, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'

import { LOG_FILENAME, encodeSegment, logPath, sessionDir } from './session-path.ts'
import { foldSessionLog, type SessionHeaderLine, type StoredRun } from './session-log.ts'

// 再导出：调用方（server、tests）只认 `session-store` 这一个入口，
// 拆出 codec 是实现细节，不该逼它们改 import 路径
export type { StoredRun } from './session-log.ts'

/**
 * 会话 id 的字符白名单。
 *
 * **白名单不是洁癖**：这个字符串会成为路径段。而这里挡在**最前面** ——
 * `encodeSegment` 已经能安全编任何字符串了，但一个 64 字符以内、只含
 * `[A-Za-z0-9_-]` 的 id 编出来还是原样，目录名可读。所以这一层挡的是
 * 「不该出现在这个系统里的 id」，不是路径安全（那由编码保证）。
 */
const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

/**
 * 当前的文件格式版本。
 *
 *     1  扁平 `<id>.jsonl`，一行一轮，只有问答
 *     2  按项目分目录，一行一个事件，轮次从事件里推导
 *
 * v2 上线时 `#migrateLegacy` 会把 v1 的文件**搬到新布局**（header 里有
 * `cwd`，所以搬得动），而不是拒绝它 —— 见那个方法的说明。
 */
export const SESSION_FORMAT_VERSION = 2

/** 列表里的一行。**不含正文也不含事件** —— 列表只要能排序和显示的东西 */
export interface SessionSummary {
  id: string
  createdAt: number
  updatedAt: number
  /** 盘上有几轮（**不是**喂给模型的那几轮） */
  turns: number
  /**
   * 这一轮属于哪个工作目录。
   *
   * **从 header 读的原值**，不是从目录名反推的 —— 目录名是有损的可读编码
   * （见 `projectKey`），`/a/b` 和 `/a-b` 会落进同一层。
   */
  cwd?: string
  /** 第一句问的是什么 —— 列表里拿它当标题 */
  firstPrompt: string
  /** 读的时候跳过了几行（撕裂的尾行、认不出的 kind、序号跳了） */
  skipped: number
}

/**
 * 会话 id 合法性。不合法就抛 —— **不静默替换成一个安全的 id**：
 * 那样调用方会以为自己拿到的还是原来那个会话。
 */
export function assertSessionId(id: string): void {
  if (!ID_PATTERN.test(id)) {
    throw new Error(
      `会话 id 不合法：${JSON.stringify(id.slice(0, 40))} —— ` +
        `只允许 A-Z a-z 0-9 _ -，长度 1–64。它会变成路径段，所以是白名单校验`,
    )
  }
}

/** 一个会话在盘上的位置 */
interface Located {
  path: string
  cwd?: string
  /** 项目目录名，报错和迁移都要用 */
  project: string
}

/**
 * 一个目录下的全部会话。
 */
export class SessionStore {
  readonly #dir: string

  constructor(dir: string) {
    this.#dir = dir
  }

  get dir(): string {
    return this.#dir
  }

  /**
   * 追加一个事件。
   *
   * **自己找位置**：会话已经存在就写在它原来的地方，不存在就按 `cwd` 新建。
   * 这样「会话属于哪个项目」在它出生时就定下来了 —— 之后换工作区不会把
   * 它搬走（`cwd` 记在 header 里，是这一轮的事实）。
   *
   * @param run 第几轮（从 0 开始）。服务端开跑时算一次，整轮都用它。
   */
  async append(id: string, run: number, event: unknown, meta: { cwd?: string } = {}): Promise<void> {
    const at = Date.now()
    const found = await this.#locate(id)
    const path = found?.path ?? logPath(this.#dir, meta.cwd, id)

    await mkdir(found ? join(this.#dir, found.project, encodeSegment(id)) : sessionDir(this.#dir, meta.cwd, id), {
      recursive: true,
    })

    if (!found) {
      // 第一次写：先落 header。它是**唯一的**会话元数据落点（id、创建时间、cwd）
      const header: SessionHeaderLine & { kind: 'header' } = {
        kind: 'header',
        v: SESSION_FORMAT_VERSION,
        id,
        createdAt: at,
        ...(meta.cwd ? { cwd: meta.cwd } : {}),
      }
      await appendFile(path, `${JSON.stringify(header)}\n`, 'utf8')
    }
    await appendFile(path, `${JSON.stringify({ kind: 'event', run, at, e: event })}\n`, 'utf8')
  }

  /**
   * 读一个会话的轮次（含事件）。
   *
   * @param limit 只要最近这么多轮。**盘上一条都不会少** —— 截断发生在这里，
   *   不在写入时（见模块头）。
   */
  async load(id: string, limit?: number): Promise<StoredRun[]> {
    const found = await this.#locate(id)
    if (!found) return [] // 不存在 = 一个还没人说过话的会话，不是错误
    const { runs } = await this.#read(found.path)
    return limit !== undefined && limit >= 0 ? runs.slice(-limit) : runs
  }

  /** 列表。按最后活动时间倒序 —— 最近说过的排最前 */
  async list(): Promise<SessionSummary[]> {
    
    let projects: string[]
    try {
      projects = (await readdir(this.#dir, { withFileTypes: true }))
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
    } catch {
      // 吞的是「根目录还不存在」= 一个会话都还没有。那是**空列表**，
      // 不是错误 —— 第一次跑服务就是这个状态。
      return []
    }

    const out: SessionSummary[] = []
    for (const project of projects) {
      let ids: string[]
      try {
        ids = (await readdir(join(this.#dir, project), { withFileTypes: true }))
          .filter((d) => d.isDirectory())
          .map((d) => d.name)
      } catch {
        continue // 这个项目目录读不了 —— 跳过它，别的项目照样列
      }
      for (const seg of ids) {
        const path = join(this.#dir, project, seg, LOG_FILENAME)
        try {
          const { runs, header, skipped } = await this.#read(path)
          if (!header && runs.length === 0) continue // 空目录 / 不是我们的会话
          out.push({
            id: header?.id ?? seg,
            createdAt: header?.createdAt ?? runs[0]?.at ?? 0,
            updatedAt: runs.at(-1)?.at ?? header?.createdAt ?? 0,
            turns: runs.length,
            ...(header?.cwd ? { cwd: header.cwd } : {}),
            firstPrompt: runs[0]?.task ?? '',
            skipped,
          })
        } catch {
          // 吞的是「这一个文件读不了」（权限、正好被删）。**不跳过它** ——
          // 报成一条读不出来的记录，总比列表里凭空少一个会话好（§8.10）。
          out.push({ id: seg, createdAt: 0, updatedAt: 0, turns: 0, firstPrompt: '', skipped: -1 })
        }
      }
    }
    return out.sort((a, b) => b.updatedAt - a.updatedAt)
  }

  /** 删一个会话（连同它独占的那个目录）。@returns 真的删掉了吗 */
  async remove(id: string): Promise<boolean> {
    const found = await this.#locate(id)
    if (!found) return false
    await rm(join(this.#dir, found.project, encodeSegment(id)), { recursive: true, force: true })
    return true
  }

  /**
   * 只用 id 找到会话。
   *
   * ★ **遍历所有项目目录** —— 路径按 cwd 分层之后，光有 id 是拼不出路径的。
   *   这是那个布局的代价，DSH 也一样（它的 `findLog` 干的就是这件事）。
   *   换来的好处是「属于哪个项目」是结构而不是一次查询。
   *
   * **撞了就抛，不挑一个。** 同一个 id 出现在两个项目目录下说明盘上有一份
   * 是我们不认识的副本（拷贝、恢复、人手工搬过）—— 挑一个等于**静默丢掉
   * 另一个**，而用户以为自己在看全部。
   */
  async #locate(id: string): Promise<Located | undefined> {
    assertSessionId(id)
    
    const seg = encodeSegment(id)
    const matches: Located[] = []
    let projects: string[]
    try {
      projects = (await readdir(this.#dir, { withFileTypes: true }))
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
    } catch {
      return undefined // 根目录还没有
    }

    for (const project of projects) {
      const path = join(this.#dir, project, seg, LOG_FILENAME)
      if (!(await exists(path))) continue
      // `cwd` 从 header 读原值 —— 目录名是有损的，反推不出真路径
      const { header } = await this.#read(path)
      matches.push({ path, project, ...(header?.cwd ? { cwd: header.cwd } : {}) })
    }

    if (matches.length > 1) {
      throw new Error(
        `会话 id ${id} 同时出现在 ${matches.length} 个项目目录下：` +
          `${matches.map((m) => m.project).join('、')} —— ` +
          `盘上有我们不认识的副本，挑一个等于静默丢掉另一个。请手工确认后删掉多余的`,
      )
    }
    return matches[0]
  }

  /**
   * 读一个日志文件。解析本身在 `session-log.ts`（纯函数，不碰磁盘），
   * 这里只负责 IO —— 两边共用同一个口径，不会分叉。
   */
  async #read(path: string): Promise<ReturnType<typeof foldSessionLog>> {
    return foldSessionLog(await readFile(path, 'utf8'))
  }
}

/** 文件在不在。`stat` 抛 ENOENT 就是不在 —— 那是**答案**，不是错误 */
async function exists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch {
    // 吞的只有 ENOENT。别的错误（权限等）在 readFile/appendFile 上会照样
    // 抛出来 —— 这里把它们也报成「不存在」只是让错误晚一步出现，
    // 而晚一步的那个错误会带上路径，更好查。
    return false
  }
}
