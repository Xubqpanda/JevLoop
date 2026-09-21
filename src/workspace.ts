/**
 * JevLoop · 工作区登记表
 *
 * 一个「路径 + 名字」的列表，落盘。**它只登记，不碰磁盘上的目录。**
 *
 * ── 为什么要有这张表 ────────────────────────────────────────────
 *
 * 会话已经记着自己的 `cwd` 了（见 `session-store.ts` 的文件头），所以
 * 「这个会话在哪个目录跑的」本来就有答案。那这张表解决什么？
 *
 * 1. **挑选**。`cwd` 只能来自「用户刚浏览过的那个目录」，而浏览是一个
 *    动作、不是一个状态 —— 关掉对话框就没了。表把它变成可以再选一次的
 *    东西。
 * 2. **命名**。`/home/x/work/acme-api-v2` 在列表里该叫什么，只有人能定。
 * 3. **跑任务那条路径上，请求发的是 id 不是路径**（见 `dir-browse.ts`
 *    的文件头）。没有这张表，客户端就得直接发路径，那正是我们不想开的
 *    口子。
 *
 * ── 和 DSH 的一处不同 ──────────────────────────────────────────
 *
 * DSH 的 `WorkspaceView` 带 `sessionIds[]`（手动排序的会话列表）。这里
 * **不做**：会话的 `cwd` 已经是事实，再维护一份列表就多一个会漂移的
 * 副本 —— 而漂移的后果是「这个会话从工作区里消失了」，且没人会报错。
 * 分组改成**查出来的**：会话的 `cwd` 等于哪个工作区的 `path`，就属于它。
 * 代价是没有手动排序，而那个可以以后再补。
 *
 * ── 为什么这次不是 JSONL ────────────────────────────────────────
 *
 * 会话用 JSONL 是因为它是**只追加的日志**，记录一旦写下就不该改。这张表
 * 不一样：改名、撤销都是**原地修改**，追加式表达不了。而它又小（几十条），
 * 整个文件重写一遍是最简单且不会写坏的写法。
 *
 * ⚠️ 代价：**没有跨进程并发保护**。两个请求同时写会互相覆盖。这个服务是
 * 单机原型、操作由一个人点出来，所以接受它 —— 但这条写在这里，而不是
 * 假装没有。真要多写者，该做的是文件锁或者换个存储。
 *
 * @module JevLoop/workspace
 */

import { mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises'
import { basename, dirname } from 'node:path'

import { WorkspaceError, assertAbsolute } from './vocab-workspace.ts'

/** 一个登记过的工作目录 */
export interface Workspace {
  id: string
  /** 规范化（`realpath` 过）的绝对路径。**这是唯一的真相**，`title` 只是给人看的 */
  path: string
  /** 显示名。默认取目录名 */
  title: string
  createdAt: number
  updatedAt: number
}

/** `create` 的结果。`created: false` 表示这个路径早就登记过了 */
export interface CreateResult {
  workspace: Workspace
  created: boolean
}

/**
 * 登记过的那些工作目录。
 *
 * 一个 `GET /api/workspaces` 背后的东西：**列表**给人挑，**id** 用来在
 * 跑任务时说「用那个目录」而不必再说一遍路径。
 *
 * 幂等的地方只有一处 —— `create` 同一个路径返回原来那条（`created: false`）。
 * 别的操作找不到就抛，不静默成功。
 */
export class WorkspaceStore {
  readonly #file: string
  /** 串行化写入：至少让同一个进程内的两次写不交叉 */
  #chain: Promise<unknown> = Promise.resolve()

  constructor(file: string) {
    this.#file = file
  }

  get file(): string {
    return this.#file
  }

  /** 按最近动过的排前面 —— 和会话列表同一个方向 */
  async list(): Promise<Workspace[]> {
    return (await this.#read()).sort((a, b) => b.updatedAt - a.updatedAt)
  }

  async get(id: string): Promise<Workspace | undefined> {
    return (await this.#read()).find((w) => w.id === id)
  }

  /**
   * 登记一个目录。**幂等**：同一个路径再登记一次返回原来那条
   * （`created: false`），不新建、不报错。
   *
   * 理由：用户点两次「添加同一个目录」是常事，而报错会让人以为做错了什么。
   * 但调用方要能分辨「新建了」和「早就有」，所以返回里带 `created`。
   *
   * @throws {WorkspaceError} `not-absolute` / `not-a-directory` / `unreadable`
   */
  async create(path: string, title?: string): Promise<CreateResult> {
    assertAbsolute(path)
    let real: string
    try {
      real = await realpath(path)
      if (!(await stat(real)).isDirectory()) {
        throw new WorkspaceError('not-a-directory', path, `${path} 不是目录`)
      }
    } catch (err) {
      if (err instanceof WorkspaceError) throw err
      throw new WorkspaceError('unreadable', path, `读不到 ${path}：${(err as Error).message}`)
    }

    return this.#mutate<CreateResult>((all) => {
      // 用**规范化之后**的路径比对：同一个目录经不同软链进来要认成同一个
      const found = all.find((w) => w.path === real)
      if (found) return { next: all, out: { workspace: found, created: false } }
      const now = Date.now()
      const ws: Workspace = {
        id: `w${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`,
        path: real,
        title: title?.trim() || basename(real) || real,
        createdAt: now,
        updatedAt: now,
      }
      return { next: [...all, ws], out: { workspace: ws, created: true } }
    })
  }

  /**
   * 改名。
   *
   * @throws {WorkspaceError} `not-found` —— **不静默成功**：那会让界面显示
   *   一个没保存的名字，而用户以为存上了。
   */
  async rename(id: string, title: string): Promise<Workspace> {
    const t = title.trim()
    if (!t) throw new WorkspaceError('bad-name', id, '名字不能为空')
    return this.#mutate<Workspace>((all) => {
      const i = all.findIndex((w) => w.id === id)
      if (i < 0) throw new WorkspaceError('not-found', id, `没有 id 为 ${id} 的工作区`)
      const updated = { ...all[i]!, title: t, updatedAt: Date.now() }
      const next = [...all]
      next[i] = updated
      return { next, out: updated }
    })
  }

  /**
   * 撤销登记。**只从表里去掉，不碰磁盘上的目录** ——
   * 「从列表里移除」和「删掉你的文件」是两件事，混起来是灾难。
   *
   * @returns 真的去掉了吗（本来就不在返回 false，不抛）
   */
  async remove(id: string): Promise<boolean> {
    return this.#mutate<boolean>((all) => {
      const next = all.filter((w) => w.id !== id)
      return { next, out: next.length !== all.length }
    })
  }

  /** 读 + 改 + 写，串行 */
  async #mutate<T>(fn: (all: Workspace[]) => { next: Workspace[]; out: T }): Promise<T> {
    const run = async (): Promise<T> => {
      const { next, out } = fn(await this.#read())
      await mkdir(dirname(this.#file), { recursive: true })
      await writeFile(this.#file, `${JSON.stringify(next, null, 2)}\n`, 'utf8')
      return out
    }
    // 串起来：前一次写完才做下一次。**成功失败都接着排** ——
    // 一次失败就卡死后续的话，界面会再也存不进东西，而且看不出原因
    const p = this.#chain.then(run, run)
    this.#chain = p.catch(() => {})
    return p
  }

  async #read(): Promise<Workspace[]> {
    let raw: string
    try {
      raw = await readFile(this.#file, 'utf8')
    } catch {
      // 吞的是「表还不存在」= 一个工作区都没登记过。**空表**，不是错误 ——
      // 第一次跑服务就是这个状态。
      return []
    }
    try {
      const v: unknown = JSON.parse(raw)
      if (!Array.isArray(v)) return []
      // 逐条验形状：表是可以手改的，一条坏记录不该让整张表消失
      return v.filter(
        (w): w is Workspace =>
          !!w &&
          typeof w === 'object' &&
          typeof (w as Workspace).id === 'string' &&
          typeof (w as Workspace).path === 'string' &&
          typeof (w as Workspace).title === 'string',
      )
    } catch {
      // 吞的是「文件不是合法 JSON」。**当空表**：服务起不来比少几条登记糟。
      // 代价是那几条看不见了 —— 但文件还在盘上，下次写入才会覆盖它。
      return []
    }
  }
}
