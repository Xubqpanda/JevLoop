/**
 * JevLoop · 工作区的词汇
 *
 * 失败词表 + 路径断言。**两半都要用，所以它得在最底下。**
 *
 * ── 为什么单独一个文件 ──────────────────────────────────────────
 *
 * 「选目录」（`dir-browse.ts`）和「登记表」（`workspace.ts`）是**两件事**，
 * 各约 100 行代码，所以拆开。但它们共用一套失败词表 —— 而「封闭词表」
 * 的意义就在于它**是一份**：拆成两套，调用方就得同时处理两种错误类型，
 * 那正是封闭词表想避免的。
 *
 * 两半都在 L2，而 §11 不允许同层互相依赖，所以词表下沉到 L0。
 * 这和 `vocab-records.ts` 装 `AuditRecord` / `MeterStats` 是同一个做法：
 * **纯形状和纯函数放最底下，谁都能用**。
 *
 * 零 import（除 node:path 的 `isAbsolute`），所以放 L0 名副其实。
 *
 * @module JevLoop/vocab-workspace
 */

import { isAbsolute } from 'node:path'

/**
 * 封闭的失败词表。**调用方穷举处理，不做字符串匹配。**
 *
 * 字符串匹配是这里最该避免的：错误信息是给人看的，随时会改措辞，
 * 而调用方按措辞分支的话，一改文案就静默失效。
 */
export type WorkspaceErrorCode =
  /** 收到的不是绝对路径。相对路径会落在**服务进程的 cwd** 上，那是没定义的 */
  | 'not-absolute'
  /** 路径存在，但不是目录 */
  | 'not-a-directory'
  /** 读不到：不存在、没权限、断链 */
  | 'unreadable'
  /** 名字不是单独一段（含分隔符、是 `.` / `..`、或者空的） */
  | 'bad-name'
  /** 建目录时重名 */
  | 'name-taken'
  /** 按 id 找东西没找到 */
  | 'not-found'

/** 带类型的失败。抛出它，而不是抛一个 `Error` 再让调用方去猜 */
export class WorkspaceError extends Error {
  readonly code: WorkspaceErrorCode
  /** 出问题的那个路径（或者 id）—— 报错要能直接定位 */
  readonly path: string

  /**
   * ⚠️ **字段显式声明 + 构造函数里赋值，不能用参数属性。**
   *
   * `constructor(readonly code: X)` 是合法 TS，`tsc` 也接受 —— 但 Node 的
   * **类型剥离模式剥不掉它**，运行时直接抛
   * `ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX: TypeScript parameter property is
   * not supported in strip-only mode`。
   *
   * 这个项目跑 .ts 靠的就是类型剥离，所以这条是硬约束，而**`tsc` 不会
   * 告诉你**：`npx tsc --noEmit` 全绿，一跑就炸。实测踩于 2026-09-21。
   */
  constructor(code: WorkspaceErrorCode, path: string, message: string) {
    super(message)
    this.name = 'WorkspaceError'
    this.code = code
    this.path = path
  }
}

/**
 * 路径必须是绝对路径。
 *
 * ★ 这条是**安全**性质，不是风格。相对路径会相对**服务进程的 cwd** 解析，
 *   而那个值取决于怎么启动的 —— 用户看不见、也猜不到，于是「我选的目录」
 *   和「实际用的目录」会不是同一个，而且没有任何地方会报错。
 *
 * 纯函数、零 IO，所以它住在 L0 而不是某一半里。
 */
export function assertAbsolute(p: string): void {
  if (!isAbsolute(p)) {
    throw new WorkspaceError('not-absolute', p, `只接受绝对路径，收到 ${JSON.stringify(p)}`)
  }
}
