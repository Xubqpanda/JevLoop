/**
 * JevLoop · 目录浏览：选一个工作目录
 *
 * ══════════════════════════════════════════════════════════════
 *  **服务端第一次接受「路径」这个输入。**
 * ══════════════════════════════════════════════════════════════
 *
 * 在这之前工作目录只能来自启动参数 `CWD_ROOT`，请求指定不了 —— 那是
 * 一次实测换来的规矩：`?cwd=` 曾经可用，于是 `list_dir` + `read_file`
 * 能在 `/etc` 上跑通，这个端点等于一个**无鉴权的任意目录读取**。
 *
 * 现在要让人挑目录，就必须重新开这个口子。开法和上次不同，而且区别是
 * 关键：**跑任务那条路径上，请求发的是 workspaceId，不是路径** —— 路径
 * 由服务端从自己登记的表里取（见 `workspace.ts`）。客户端说不出一个
 * 服务端没登记过的目录。
 *
 * 而**这个文件**确实接受路径了。所以 `server.ts` 加了守卫：绑到非回环
 * 地址时浏览端点直接拒绝。浏览本机文件系统是**宿主级能力**，只在服务
 * 只能从本机访问时才成立。
 *
 * ── 形状抄自 DSH 的 directory-picker ────────────────────────────
 *
 * （`packages/host/directory-picker`，Apache-2.0。）几个决定直接可用：
 *
 * · **只列目录**，不列文件。这是选目录，不是文件浏览器。
 * · **每个条目带绝对路径**，客户端不自己拼。客户端于是不用处理 `..`、
 *   分隔符、Windows 盘符 —— 那些是最容易写出漏洞的地方。
 * · **`crumbs` 从根到当前，每一节都是跳转目标**。面包屑是导航，不是装饰。
 * · **`hidden` 逐条标，由客户端决定显不显示**。宿主只报自己知道的。
 * · **`truncated` 如实报**。截断了不说，读的人会以为就这些（§8.10）。
 * · **错误是封闭词表**（在 `vocab-workspace.ts`），不是字符串匹配。
 *
 * 有 IO，所以放 L2。
 *
 * @module JevLoop/dir-browse
 */

import { mkdir, readdir, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, join, normalize, parse, sep } from 'node:path'

import { WorkspaceError, assertAbsolute } from './vocab-workspace.ts'

/** 一次浏览里的一行：一个子目录，或者一节面包屑 */
export interface DirEntry {
  /** 显示用的名字。根那一节带完整路径 —— 它没有更好的短名字 */
  name: string
  /** **绝对**路径。客户端拿它直接跳，不自己拼 */
  path: string
  /** 按宿主平台的习惯判定（POSIX 上是以点开头）。**显不显示由客户端决定** */
  hidden: boolean
}

/** 一层目录加上它的祖先链 */
export interface DirListing {
  /** 被列出的目录（绝对路径） */
  path: string
  /** 宿主账户的家目录 —— 面包屑上「Home」那一节的落点 */
  home: string
  /** 从文件系统根到当前目录（含两端）。**每一节都可跳** */
  crumbs: DirEntry[]
  /** 直接子目录，按名字排序。**只有目录** */
  entries: DirEntry[]
  /** 子目录太多被截断了吗。截断的是按名字排序的**尾部** */
  truncated: boolean
}

/**
 * 一次浏览最多列几个子目录。
 *
 * 上限必须有：`/` 下面、或者一个 `node_modules` 的父目录下面，条目可以
 * 上万。截断了要**说出来**（`truncated`），不能悄悄少给几行。
 */
const MAX_ENTRIES = 500

function entryOf(p: string): DirEntry {
  const name = basename(p)
  return { name: name || p, path: p, hidden: name.startsWith('.') }
}

/** 从根到 `p`（含两端），每一节一个跳转目标 */
function crumbsOf(p: string): DirEntry[] {
  const out: DirEntry[] = []
  let cur = p
  for (;;) {
    const parent = dirname(cur)
    out.unshift({ ...entryOf(cur), hidden: false })
    // 根那一节：`dirname('/') === '/'`，用它当终止条件；
    // Windows 的盘符同理（`dirname('C:\\') === 'C:\\'`）
    if (parent === cur) break
    cur = parent
  }
  // 根那一节没有可读的短名字，用完整路径（`/` 或 `C:\`）
  if (out.length > 0) out[0] = { ...out[0]!, name: parse(out[0]!.path).root }
  return out
}

/**
 * 列一层目录。
 *
 * @param path 绝对路径。**不传就是家目录** —— 起点不该是服务进程的 cwd，
 *   那取决于怎么启动的，用户看不见也猜不到。
 * @throws {WorkspaceError} `not-absolute` / `not-a-directory` / `unreadable`
 */
export async function listDirs(path?: string): Promise<DirListing> {
  const home = homedir()
  const target = normalize(path ?? home)
  assertAbsolute(target)

  let real: string
  try {
    // `realpath` 而不是直接用：符号链接要解开，否则面包屑会停在链接的
    // 名字上，而用户以为自己在那儿、实际在别处
    real = await realpath(target)
    if (!(await stat(real)).isDirectory()) {
      throw new WorkspaceError('not-a-directory', target, `${target} 不是目录`)
    }
  } catch (err) {
    if (err instanceof WorkspaceError) throw err
    // 吞的是「读不到这个目录」（不存在、没权限）—— **换成有类型的错误**，
    // 让调用方能穷举处理，而不是去认 ENOENT / EACCES 这些字符串
    throw new WorkspaceError('unreadable', target, `读不到 ${target}：${(err as Error).message}`)
  }

  let dirents
  try {
    dirents = await readdir(real, { withFileTypes: true })
  } catch (err) {
    throw new WorkspaceError('unreadable', real, `读不到 ${real}：${(err as Error).message}`)
  }

  // 只留目录。符号链接要**试一下**才知道指向的是不是目录 ——
  // `withFileTypes` 对链接报的是 `isSymbolicLink()`，不会跟过去看
  const dirs: string[] = []
  for (const d of dirents) {
    if (d.isDirectory()) dirs.push(join(real, d.name))
    else if (d.isSymbolicLink()) {
      try {
        if ((await stat(join(real, d.name))).isDirectory()) dirs.push(join(real, d.name))
      } catch {
        // 吞的是「断链」—— 指不到的链接不是目录，跳过它。
        // 这不是错误，是「这个链接没用了」。
      }
    }
  }
  dirs.sort((a, b) => basename(a).localeCompare(basename(b)))

  return {
    path: real,
    home,
    crumbs: crumbsOf(real),
    entries: dirs.slice(0, MAX_ENTRIES).map(entryOf),
    truncated: dirs.length > MAX_ENTRIES,
  }
}

/**
 * 在已有的父目录下建一个子目录。
 *
 * @param parent 绝对路径，必须已经存在
 * @param name **单段**名字：不能有分隔符，不能是 `.` / `..`
 * @returns 新建目录的绝对路径
 * @throws {WorkspaceError} `bad-name` / `name-taken` / `unreadable`
 */
export async function createDir(parent: string, name: string): Promise<string> {
  assertAbsolute(parent)
  // 单段校验是**安全**性质不是洁癖：`../../x` 当名字用会写到父目录外面，
  // 而调用方以为自己只是在「新建一个子目录」
  if (!name.trim() || name === '.' || name === '..' || name.includes('/') || name.includes(sep)) {
    throw new WorkspaceError('bad-name', name, `名字要是单独一段，不能含分隔符、不能是 . 或 ..：${JSON.stringify(name)}`)
  }
  const target = join(parent, name)
  try {
    await mkdir(target)
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code
    if (code === 'EEXIST') throw new WorkspaceError('name-taken', target, `${target} 已经存在`)
    throw new WorkspaceError('unreadable', target, `建不了 ${target}：${(err as Error).message}`)
  }
  return target
}
