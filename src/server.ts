#!/usr/bin/env node
/**
 * JevLoop · 开发服务器
 *
 *   npx jevloop serve                                       # 从包装出来的版本
 *   node --experimental-strip-types src/server.ts           # 从 clone
 *   CWD_ROOT=/path/to/project node --experimental-strip-types src/server.ts
 *
 * 把 agent loop 的事件流通过 SSE 推给浏览器，同时托管 web/ 下的界面。
 * 只用 Node 标准库 —— 没有 express，没有构建步骤。
 *
 * 工作目录由 `CWD_ROOT` 决定（启动参数），**不能由请求指定** —— 理由见
 * 「工作目录」一节。`HOST` 默认只监听本机；这个服务没有鉴权，
 * 改成 `0.0.0.0` 等于把「让 agent 在本机跑任务」开放给整个网络。
 *
 * 为什么是 SSE 而不是 WebSocket：事件是**单向**的（服务端 → 浏览器），
 * SSE 是浏览器原生的、会自动重连的、用 GET 就能开的单向通道。
 * 双向通信在这个界面里没有任何用途。
 *
 * 会话落盘在 `JEVLOOP_HOME/sessions/`（默认 `~/.jevloop/`），规则见
 * `session-store.ts`。**盘上全留，只在读给 agent 用时截到 `MAX_TURNS` 轮** ——
 * 以前是内存里的一个 Map，重启就没了，而且界面上看不出来。
 *
 * ── 为什么不能再拆 ────────────────────────────────────────────
 *
 * `docs/CODE-STYLE.md` §12 问的是能不能用一句话说完它负责什么。这里能：
 * **这个文件是开发服务器** —— 收请求、把事件流推给浏览器、托管 `web/`。
 * 路由按端点分节，每节都只是「解析请求 → 调内核 → 写响应」，而它们共享同一份
 * 状态（会话、工作区、正在跑的 run）；拆成 `server-routes.ts` 之类只会让
 * 「一共有哪些端点」从一眼可见变成要翻三个文件。
 *
 * 真的会长的那半已经在别处：会话存储、工作区登记、目录浏览各自成文件
 * （`session-store.ts` / `workspace.ts` / `dir-browse.ts`），本文件只做接线。
 *
 * ⚠️ **本文件在 `src/` 里，但它读的文件在包根。** `web/` 和 `DECISION.md`
 * 既不编译也不属于 `src/`，所以 `ROOT` 不是「本文件所在目录」，而是
 * **往上找到的第一个有 `package.json` 的目录** —— 从 `src/` 跑和从 `dist/` 跑
 * 都指回包根。写成前者的话，编译之后 `web/` 会指到 `dist/web`。
 *
 * @module JevLoop/server
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { realpathSync, existsSync, readdirSync, statSync } from 'node:fs'
import { basename, dirname, extname, isAbsolute, join, normalize, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

import { Decider } from './decide.ts'
import { Meter } from './meter.ts'
import { runAgent } from './agent.ts'
import { parseDecisionDoc, summarize, headline, isGate } from './decisiondoc.ts'
import { compilePredicate } from './decision-compile.ts'
import type { AgentEvent, GenerateDelta } from './events.ts'
import { SessionStore, assertSessionId } from './session-store.ts'
import { migrateLegacyLayout } from './session-migrate.ts'
import { createDir, listDirs } from './dir-browse.ts'
import { WorkspaceStore } from './workspace.ts'
import { WorkspaceError, type WorkspaceErrorCode } from './vocab-workspace.ts'
import { encodeSse } from './sse.ts'
// `backends` 和 `env` 都是 L6，和本文件同层 —— §11 只允许 L0 内部互相指涉、
// 以及 L2 指向定义角，同层直接 import 是违规的。走门面（`index.ts` 不受层约束），
// 这也是 `cli.ts` 的写法。
import { resolveGenerator, resolveProvider, loadEnv } from './index.ts'

/**
 * 包根目录：从本文件往上找到的第一个带 `package.json` 的目录。
 *
 * **不能写成「本文件所在目录」** —— `web/` 和 `DECISION.md` 在包根，
 * 而本文件从 `src/server.ts` 跑时所在的是 `src/`、编译后是 `dist/`，
 * 两者都指不到包根。往上找 `package.json` 对两种位置都成立。
 */
function findRoot(from: string): string {
  let dir = from
  for (;;) {
    if (existsSync(join(dir, 'package.json'))) return dir
    const up = dirname(dir)
    if (up === dir) throw new Error(`no package.json above ${from} — is this file installed correctly?`)
    dir = up
  }
}

const ROOT = findRoot(dirname(fileURLToPath(import.meta.url)))
const WEB_DIR = join(ROOT, 'web')
const DECISION_MD = join(ROOT, 'DECISION.md')
const PORT = Number(process.env.PORT ?? 7799)
const HOST = process.env.HOST ?? '127.0.0.1'

loadEnv({ cwd: ROOT })

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
}

// ═══════════════════════════════════════════════════════════
// 工作目录：由**启动服务的人**决定，不由调用方决定
//
// 这里曾经允许 `?cwd=` 从 query 带进来。那让这个端点变成了无鉴权的任意目录读取：
// 工具层的 `safePath(cwd, p)` 只保证「不逃出给定的 cwd」，而 cwd 本身是调用方指定的 ——
// 实测 `?cwd=/etc` 能让 `list_dir` + `read_file` 在 `/etc` 上成功，
// 结果原样进 SSE 流回到调用方。
//
// 选择权因此上移到进程边界：`CWD_ROOT` 是启动参数，不设就用临时演示目录。
// ═══════════════════════════════════════════════════════════

const DEFAULT_TASK = '列出工作目录里的文件，读取其中的 TypeScript 文件，说明它定义了哪些函数'

// ═══════════════════════════════════════════════════════════
// 会话
//
// 多轮的上文。一轮 = 用户说了什么 + agent 答了什么。
//
// ⚠️ **现在只做到「记录」**：会话会存下来、界面刷新能恢复、「新对话」
// 能清空 —— 但**生成器还看不到上文**。那需要内核侧的
// `GenerateRequest.history` 和 `AgentOptions.history`，而
// `src/llm.ts` `src/agent.ts` `src/decisions.ts` 正被重构顾问持有
// （见 docs/STATUS.md）。
//
// 所以下面 `runAgent` 的调用里**没有** history 参数。那不是漏了 ——
// 等内核收这个字段，接上只要一行。
// ═══════════════════════════════════════════════════════════


/** 每个会话保留的轮数。再多没有意义 —— 生成器的上下文不是无限的 */
const MAX_TURNS = 12

/**
 * 会话存在哪。
 *
 *     JEVLOOP_HOME/sessions/<id>.jsonl      默认 ~/.jevloop/sessions/
 *
 * 放**用户目录**而不是仓库里：会话是 agent 的状态，不是被分析的那个
 * 工作区的状态。放进工作区的话，`list_dir` 会把它们列出来、`read_file`
 * 能读到别的会话的对话。
 *
 * 上限**不再是「最多几个」**。以前是内存里的 LRU（`MAX_SESSIONS = 64`），
 * 因为不设上限的话每刷新一次页面就多一个永不释放的数组。落了盘之后，
 * 多出来的代价是磁盘而不是内存，而**悄悄删用户的会话**比占点磁盘糟 ——
 * 所以改成看得见、删得掉（界面上有列表，`DELETE /api/sessions?id=`）。
 */
const JEVLOOP_HOME = process.env.JEVLOOP_HOME ?? join(homedir(), '.jevloop')
const SESSIONS_DIR = join(JEVLOOP_HOME, 'sessions')
const store = new SessionStore(SESSIONS_DIR)
const workspaces = new WorkspaceStore(join(JEVLOOP_HOME, 'workspaces.json'))

/**
 * 绑到非回环地址时，**浏览和登记端点一律拒绝**。
 *
 * ══════════════════════════════════════════════════════════════
 *  ★ 这是加了工作区之后**新开的口子**，所以要有对应的闸门。
 * ══════════════════════════════════════════════════════════════
 *
 * 在这个功能之前，工作目录只能来自 `CWD_ROOT`（启动参数）—— 请求指定
 * 不了，因为 `?cwd=/etc` 曾经让这个端点变成**无鉴权的任意目录读取**。
 *
 * 现在跑任务那条路径仍然安全：请求发的是 **workspaceId**，路径由服务端
 * 从自己登记的表里取。但**浏览**（`/api/browse`）和**登记**
 * （`POST /api/workspaces`）确实开始接受路径了 —— 它们必须接受，否则
 * 没法挑目录。
 *
 * 「能浏览本机文件系统」是**宿主级能力**：它和「能在这台机器上跑命令」
 * 是一回事。所以它只在服务只能从本机访问时才成立。绑到 `0.0.0.0` 的话，
 * 这个端点等于把整个文件系统开放给网络，而那个后果不该由一个
 * 「我没注意 HOST 默认值」来承担。
 *
 * 要显式开放就设 `JEVLOOP_ALLOW_REMOTE_WORKSPACE=1` —— **让它是一次决定**。
 */
const LOOPBACK_HOSTS = new Set(['127.0.0.1', '::1', 'localhost'])
const ALLOW_REMOTE_WORKSPACE = process.env.JEVLOOP_ALLOW_REMOTE_WORKSPACE === '1'

function remoteWorkspaceRefusal(): string | null {
  if (LOOPBACK_HOSTS.has(HOST) || ALLOW_REMOTE_WORKSPACE) return null
  return (
    `服务绑在 ${HOST}（非本机），所以浏览目录和登记工作区被拒绝。\n` +
    `它们能读到这台机器上的任何目录 —— 那和「能在这台机器上跑命令」是一回事。\n` +
    `要么把 HOST 改回 127.0.0.1，要么明确接受这个后果：JEVLOOP_ALLOW_REMOTE_WORKSPACE=1`
  )
}

/**
 * 生成器**现在能不能看到上文**，如实报给界面。
 *
 * 这是这个服务端唯一一处「说了还没做」的地方，所以它必须是数据而不是
 * 注释 —— 界面拿它决定要不要提示读者「它不记得上一句」。
 *
 * **现在是 `true`**：`AgentOptions.history` 已落地（`de6889a`），
 * 判定帧也带上了有界的 `earlier`。界面那句提示会自动消失。
 *
 * 一个会撒谎的界面比一个缺功能的界面糟得多 —— 所以这个值是**跟着
 * 内核的实际能力走**的，不是跟着愿望走。内核要是哪天不收了，这里得翻回去。
 */
const MEMORY_WIRED = true

/**
 * 会话 id 来自 query，会变成文件名 —— 不合法就拒绝，**不静默换一个**。
 *
 * 抛出去会被下面的 `handleRun` 兜住（它已经有 try/catch），报成一条
 * `halt: 'error'` 的 `run:end` —— 而不是拿一个别的会话跑一轮。
 */
function resolveSessionId(raw: string | null): string {
  const id = raw?.trim() || 'default'
  assertSessionId(id)
  return id
}

/**
 * `run:end` 之后等多久再关连接。
 *
 * 给客户端收到 `run:end` 后自己 `close()` 的时间 —— 服务端的 FIN 先到
 * 会让浏览器报一次 error（见 handleRun 的 finally）。
 */
const STREAM_LINGER_MS = 200

/** `maxSteps` 的默认值与上限。上限不是装饰：query 是不花钱就能拧的旋钮 */
const DEFAULT_MAX_STEPS = 12
const MAX_MAX_STEPS = 50

/**
 * 解析 `maxSteps`。
 *
 * 不能写成 `Number(x) ?? 12` —— `Number('abc')` 是 `NaN` 而不是 `null`，
 * `??` 挡不住它，而 `while (step < NaN)` 恒为假：工具循环一次都不跑，
 * 在输出上和「模型判断不需要工具」完全一样（§8.10 不假装成功）。
 * 所以非法值一律回退并**出声**，不静默。
 */
function parseMaxSteps(raw: string | null): number {
  if (raw == null || raw.trim() === '') return DEFAULT_MAX_STEPS
  const n = Number(raw)
  if (!Number.isFinite(n) || n < 1) {
    console.warn(`  ⚠ maxSteps=${JSON.stringify(raw)} 不是正整数，回退到 ${DEFAULT_MAX_STEPS}`)
    return DEFAULT_MAX_STEPS
  }
  const clamped = Math.min(Math.floor(n), MAX_MAX_STEPS)
  if (clamped !== n) console.warn(`  ⚠ maxSteps=${raw} 被夹到上限 ${clamped}`)
  return clamped
}

const DEMO_FILES: Record<string, string> = {
  'invoice.ts': `export interface Invoice {
  id: string
  amount: number
  paid: boolean
}

/** 未付款总额 */
export function outstanding(invoices: Invoice[]): number {
  return invoices.filter((i) => !i.paid).reduce((sum, i) => sum + i.amount, 0)
}
`,
  'retry.ts': `/** 指数退避 */
export function backoff(attempt: number, base = 200): number {
  return Math.min(base * 2 ** attempt, 30_000)
}
`,
  'notes.md': '# 说明\n\n这个目录由 JevLoop 的开发服务器自动生成。\n',
}

/**
 * 演示目录。**固定在 `JEVLOOP_HOME/demo`，不再每次启动换一个。**
 *
 * ══════════════════════════════════════════════════════════════
 *  以前这里是 `mkdtemp`，于是**每次重启都是一个新目录**。
 * ══════════════════════════════════════════════════════════════
 *
 * 后果实测（2026-09-21）：39 个残留临时目录，而会话记着自己的 `cwd` ——
 * 于是重启之后，**所有历史会话都挂在一个已经不存在的目录上**。
 * 会话列表里 16 条，只有当前那一条是活的。
 *
 * 更要紧的是「工作区」这个概念在默认配置下因此**没有意义**：你去挑一个
 * 工作区，可选的那些全是死的。
 *
 * 固定下来之后，「会话属于哪个工作区」是稳定的，重启也不变。
 */
const DEMO_DIR = join(JEVLOOP_HOME, 'demo')

async function ensureDemoWorkspace(): Promise<string> {
  await mkdir(DEMO_DIR, { recursive: true })
  for (const [name, body] of Object.entries(DEMO_FILES)) {
    // ★ **只补缺的，不覆盖已有的。** agent 可以往这些文件里写东西，
    //   而每次启动都重新播种等于**悄悄回滚用户的工作**。
    if (!(await isDir(join(DEMO_DIR, name))) && !(await fileExists(join(DEMO_DIR, name)))) {
      await writeFile(join(DEMO_DIR, name), body, 'utf8')
    }
  }
  return DEMO_DIR
}

/** 这个路径存在吗（文件或目录都算）。空 catch 说明：吞的是「stat 失败 = 不存在」 */
async function fileExists(p: string): Promise<boolean> {
  try {
    await stat(p)
    return true
  } catch {
    return false
  }
}

function resolveCwdRoot(): string | null {
  const raw = process.env.CWD_ROOT
  if (!raw) return null
  try {
    return realpathSync(raw)
  } catch (err) {
    // 配错了要在启动时就知道，不能等到第一次工具调用才炸
    console.error(`✗ CWD_ROOT=${raw} 不可访问：${(err as Error).message}`)
    process.exit(1)
  }
}

const CWD_ROOT = resolveCwdRoot()

let demoWorkspace: Promise<string> | null = null

/** 演示目录只准备一次（这是**进程内**的缓存；目录本身现在跨重启稳定） */
function workspaceRoot(): Promise<string> {
  if (CWD_ROOT) return Promise.resolve(CWD_ROOT)
  demoWorkspace ??= ensureDemoWorkspace()
  return demoWorkspace
}

// ═══════════════════════════════════════════════════════════
// SSE
// ═══════════════════════════════════════════════════════════

/**
 * 这条 SSE 线上会出去的东西。
 *
 * 是**两者**，而不是 `AgentEvent` 一个 —— 因为这条线上跑的确实有两种东西：
 * 事件（进日志、进轨迹）和流式增量（**不进**，见 `events.ts`）。
 * 写成 `AgentEvent` 的话，类型在说谎，而说谎的类型会被绕过去
 * （`as any` 或者改 `openStream` 的签名）。
 */
type WireMessage = AgentEvent | GenerateDelta

/**
 * 开一条 SSE 通道。
 *
 * 每条事件单独一行 `data:`，用空行分隔 —— SSE 的帧格式。
 * **不做事件名分类**（`event:` 字段）：前端只需要一个 `onmessage`，
 * 分流交给 JSON 里的 `type`，这样加新事件类型时前端不用改协议层。
 */
function openStream(res: ServerResponse): (e: WireMessage) => void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    // 关掉 nginx 之类的缓冲，否则事件会被攒着一起发
    'x-accel-buffering': 'no',
  })
  // 客户端跑掉时 Node 会在 response 上发 `error`（ECONNRESET 之类）。
  // **没有监听器的 'error' 事件会抛** —— 一个断开的标签页不该能掀掉服务端。
  // 空 catch 说明：这里吞的就是「对面没了」，没有别的东西会到这里。
  res.on('error', () => {})
  res.write(': connected\n\n')
  return (e) => {
    if (res.writableEnded || res.destroyed) return
    res.write(encodeSse(e))
  }
}

// ═══════════════════════════════════════════════════════════
// 路由
// ═══════════════════════════════════════════════════════════

async function handleRun(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  // `cwd` 曾经是可用的。这里显式拒绝而不是静默忽略 ——
  // 静默忽略会让调用方以为它生效了，然后在错误的目录上解读结果（§8.10）。
  if (url.searchParams.has('cwd')) {
    res
      .writeHead(400, { 'content-type': 'text/plain; charset=utf-8' })
      .end(
        'cwd 不能由请求指定。要换工作目录，请在启动时设 CWD_ROOT：\n' +
          '  CWD_ROOT=/path/to/dir node --experimental-strip-types server.ts\n',
      )
    return
  }

  const task = url.searchParams.get('task')?.trim() || DEFAULT_TASK
  const maxSteps = parseMaxSteps(url.searchParams.get('maxSteps'))
  const session = resolveSessionId(url.searchParams.get('session'))
  // ★ 请求发的是 **id**，不是路径 —— 客户端说不出一个服务端没登记过的目录。
  //   没给就退回老路径（`CWD_ROOT` 或临时演示目录），行为不变。
  let cwd: string
  try {
    cwd = await resolveRunCwd(url)
  } catch (err) {
    if (err instanceof WorkspaceError) {
      sendJson(res, statusOf(err.code), { error: err.message, code: err.code })
      return
    }
    throw err
  }

  const send = openStream(res)

  // 客户端断开后不再往连接上写（写了会报错），但 **agent 仍会跑完** ——
  // `runAgent` 没有取消机制：`AgentOptions` 里没有 `signal`，`src/agent.ts` 里
  // `abort` 出现 0 次（审计 V2）。所以关掉页面只省下网络传输，
  // 省不下判定和生成的钱。真正的取消要等 `agent.ts` 交还后加 `AbortSignal`。
  let clientGone = false
  req.on('close', () => {
    clientGone = true
  })

  // 建在 try 外面：异常路径要报出**真实的**部分统计
  const meter = new Meter()
  /**
   * 事件写盘的串行链。**顺序是这条链存在的全部理由** —— 并发 append
   * 会让事件互相插队，而轨迹是按时序读的。
   */
  let writes: Promise<void> = Promise.resolve()

  try {
    /*
      ★ **降级和重试都要说出来。**

      以前这里是 `resolveProvider()` —— 一个参数都不传，于是后端从托管 Jev
      掉到兜底时，**服务端一个字都不打**。实测（2026-09-21）：托管 Jev
      返回 `529 system_overloaded`，整轮判定静默跑在 mock 的恒定 0.5 上，
      而界面上只表现为「这一次有点慢」。用那份数字得出的结论全是错的。

      `onFallback` 是这条链唯一的声音。不接它，链就是隐形的。
    */
    const provider = resolveProvider({
      onFallback: (err, from, to) => {
        // 走 stderr：这是**告警**不是常规输出，而且它不该混进那些
        // 「服务在哪个端口」的启动信息里
        console.error(`  ▲ 判定后端降级 ${from} → ${to}：${(err as Error).message}`)
      },
    })
    const generator = resolveGenerator()

    // 会话就是上文。**在 runAgent 之前读** —— 这一轮自己的事件要边跑边写，
    // 不该混进它自己的上文。
    //
    // ★ 只取最近 `MAX_TURNS` 轮，但**盘上一条都不少**：截断发生在读的这一侧，
    //   不在写入那一侧。以前是 `turns.splice(0, ...)`，也就是把旧对话**销毁**，
    //   而且没有任何地方记着丢过东西 —— 翻不回三天前那个会话里它答了什么。
    //
    // ★ 这一轮自己的序号 = 盘上已有几轮。**读一次就够**（下面 `load` 不再调），
    //   而且必须在写任何事件**之前**定下来 —— 边写边数会把自己数进去。
    const existing = await store.load(session)
    const runIndex = existing.length
    // 上文只要问答。**没跑完的那些轮不进上文** —— 它们没有回答，
    // 喂一句「问过但没答」给生成器只会让它困惑；但它们**留在盘上**，
    // 轨迹里照样看得见（见 `StoredRun.answer` 的注释）
    const history = existing
      .slice(-MAX_TURNS)
      .filter((r) => r.answer !== undefined)
      .map((r) => ({ task: r.task, answer: r.answer! }))
    await runAgent({
      task,
      cwd,
      history,
      decider: new Decider({
        provider,
        meter,
        // ★ 预算告警必须接上出口。第十轮 R4 报的是 `strict`/`onWarn` **没有任何调用方** ——
        //   于是 `validate()` 每次判定都算一遍，然后丢掉。`examples/demo.ts` 当时接上了，
        //   而**这里没接** —— 偏偏 server 才是走 Mock / 规则判定那条离线路径，
        //   也就是最需要看见预算告警的那条。
        onWarn: (id, warnings) => {
          for (const w of warnings) {
            console.warn(`[budget] ${w.level} ${id}: ${w.message}`)
            if (w.hint) console.warn(`[budget]   ↳ ${w.hint}`)
          }
        },
      }),
      generator,
      maxSteps,
      // 界面上「需要授权」一律先批准：这是一个演示环境，
      // 真拒绝会让 loop 在第一步就停，看不到后面的东西。
      // 真实的授权交互应该是前端弹一个确认框再回传。
      onAskHuman: async () => true,
      /*
        ★ **增量走这里，不走 `onEvent`。**

        它是独立的一条通道（见 `events.ts` 的 `GenerateDelta`），所以它
        **结构上不可能**被写进日志 —— 下面那段 `writes` 链里根本没有它。
        靠 `if (e.type === 'generate:delta') return` 过滤是会被忘掉的，
        而且忘掉的表现是日志悄悄涨几十倍、没有任何东西会报错。

        `step` 由 `runAgent` 注进来 —— 生成器不知道自己跑在第几步。
      */
      onDelta: (d) => {
        if (!clientGone) send({ type: 'generate:delta', ...d })
      },
      onEvent: (e) => {
        if (!clientGone) send(e)
        /*
          ★ **边跑边写盘**，不是等跑完再写。

          两个理由：

          1. **崩掉的那次最需要看它走到哪一步。** 生成后端抛异常时永远不会有
             `run:end`；等它才写的话，那次运行的轨迹（判定到哪、选了哪个工具）
             全部消失 —— 而排查问题恰恰只看得到那一刻。
          2. 日志**只追加**，所以半轮也是一条合法的记录。读的人从「没有
             `run:end`」就知道它没跑完。

          `onEvent` 是同步的而落盘是异步的，所以用一条链串起来 ——
          **不串的话并发 append 会互相插队**，而事件顺序是这个文件的全部意义。
          链上出错只记日志：写盘失败不该把正在跑的 agent 带停。
        */
        writes = writes
          .then(() => store.append(session, runIndex, e, { cwd }))
          .catch((err: unknown) => {
            console.error(`  ⚠ 会话事件写盘失败（这一轮仍会跑完）：${(err as Error).message}`)
          })
      },
    })

    // 把这一轮的事件写完再回响应 —— 不 await 的话，客户端收到 `run:end`
    // 立刻刷新会读到一条还没写完的日志
    await writes
  } catch (err) {
    if (!clientGone) {
      send({
        type: 'run:end',
        halt: 'error',
        steps: meter.decisions.length,
        answer: `服务端异常：${(err as Error).message}`,
        // 发真实的部分统计，不是 `new Meter()`：异常之前跑过的判定已经花过钱了，
        // 账不该被抹掉；而空 Meter 会让界面显示成一次「跑了但很快」的正常运行。
        stats: meter.stats,
      })
    }
  } finally {
    // **不要立刻 end。** EventSource 把服务端关连接当成错误：如果 FIN 比
    // `run:end` 的处理先到，浏览器就会报一次 error，界面于是显示「连接断开」
    // —— 而那次运行其实完全成功。
    //
    // 客户端收到 `run:end` 会自己 close()，那之后我们的 end 就是无害的收尾。
    // 留一拍是给那个 close() 的时间。
    if (!clientGone) setTimeout(() => { if (!res.writableEnded) res.end() }, STREAM_LINGER_MS)
  }
}

/**
 * 读一个 JSON 请求体。
 *
 * 有上限（64KB）：没有上限的话，一个 `POST` 就能让服务端把内存吃光。
 * 这几条端点的 body 都只有几十字节。
 */
async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const MAX = 64 * 1024
  const chunks: Buffer[] = []
  let size = 0
  for await (const c of req) {
    size += (c as Buffer).length
    if (size > MAX) throw new WorkspaceError('bad-name', '', `请求体超过 ${MAX} 字节`)
    chunks.push(c as Buffer)
  }
  const raw = Buffer.concat(chunks).toString('utf8').trim()
  if (!raw) return {}
  try {
    const v: unknown = JSON.parse(raw)
    return v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
  } catch {
    throw new WorkspaceError('bad-name', '', '请求体不是合法 JSON')
  }
}

/**
 * 这个路径现在还是个目录吗。
 *
 * 空 catch 说明：吞的是「stat 失败」= 不存在或读不到 —— 对这个调用方
 * 来说两者是同一件事（**不能在这儿跑**），而具体原因由运行失败时的
 * 工具错误去说。
 */
async function isDir(p: string): Promise<boolean> {
  try {
    return (await stat(p)).isDirectory()
  } catch {
    return false
  }
}

/**
 * 这次跑在哪个目录。
 *
 * `?workspace=<id>` 优先；没有就退回启动参数那条路（`CWD_ROOT`，或者
 * 自动造的演示目录）。**退回是为了不破坏已有用法** —— 以前不带这个参数
 * 是对的，现在也该是对的。
 */
async function resolveRunCwd(url: URL): Promise<string> {
  const id = url.searchParams.get('workspace')
  if (!id) return workspaceRoot()
  const ws = await workspaces.get(id)
  if (!ws) {
    throw new WorkspaceError(
      'not-found',
      id,
      `没有 id 为 ${id} 的工作区 —— 界面可能还拿着一个已经被移除的，刷新即可`,
    )
  }
  // 登记过不等于还在。**当场拒绝，不要进去再让工具层报「读不到 .」** ——
  // 那条消息和真正的原因（这个目录被删了）差着好几层
  if (!(await isDir(ws.path))) {
    throw new WorkspaceError('unreadable', ws.path, `工作区「${ws.title}」的目录不在了：${ws.path}`)
  }
  return ws.path
}

/**
 * 失败 → HTTP 状态。
 *
 * `not-found` 是 **404** 不是 400：请求本身没毛病，是那个东西不在了。
 * 混成 400 的话，界面没法区分「你发的请求不对」（改请求）和
 * 「它被删了」（刷新列表）—— 而这两件事该做的处理完全不同。
 */
function statusOf(code: WorkspaceErrorCode): number {
  return code === 'not-found' ? 404 : 400
}

/** 统一的 JSON 出口。`cache-control: no-store` —— 这些是**状态**，缓存没意义 */
function sendJson(res: ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
  res.end(JSON.stringify(body))
}

/**
 * 会话的历史：GET 读，POST 清空。
 *
 * 界面刷新后能用它把对话恢复出来 —— 会话落盘，不在浏览器里，也不在
 * 服务进程的内存里（以前是后者，重启就没了）。
 */
async function handleSession(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  const id = resolveSessionId(url.searchParams.get('id'))
  if (req.method === 'POST') {
    await store.remove(id)
    sendJson(res, 200, { runs: [], memory: MEMORY_WIRED })
    return
  }
  // ★ 返回的是**轮次**（含事件），不是问答 —— 界面据此把轨迹重建出来。
  //   一次运行约 22KB，所以这个响应比原来大得多，但它换回来的正是这个
  //   项目要展示的东西
  sendJson(res, 200, { runs: await store.load(id), memory: MEMORY_WIRED })
}

/**
 * 会话列表。**只回摘要，不回正文** —— 见 `SessionStore` 的模块头。
 *
 * `DELETE` 删一个会话。删是**看得见**的动作：界面上有列表，所以删掉
 * 什么用户是知道的（对比以前那个悄悄丢最老会话的 LRU）。
 */
async function handleSessions(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  if (req.method === 'DELETE') {
    const id = resolveSessionId(url.searchParams.get('id'))
    sendJson(res, 200, { removed: await store.remove(id) })
    return
  }
  /*
    ★ **每个会话属于哪个工作区，服务端算**（界面认不得两边）。

    关系是**查出来的**，不是存下来的：会话记着自己的 `cwd`，工作区记着
    自己的 `path`，两边相等就属于它。存一份 `sessionIds[]` 的话就多一个
    会漂移的副本 —— 而漂移的后果是「这个会话从工作区里消失了」，没人会
    报错。DSH 存那份列表是为了手动排序，我们还没有排序，所以不存。

    `workspaceId: null` 表示它的目录没有被登记过（目录被删了、或者是
    用 `CWD_ROOT` 临时指过来的）。这不是错误，是一种状态。
  */
  const ws = await workspaces.list()
  const idByPath = new Map(ws.map((w) => [w.path, w.id]))
  const sessions = (await store.list()).map((s) => ({
    ...s,
    workspaceId: s.cwd ? (idByPath.get(s.cwd) ?? null) : null,
  }))
  sendJson(res, 200, { sessions, home: JEVLOOP_HOME })
}

/**
 * 目录浏览。**只列目录**，每一条带绝对路径 —— 客户端不自己拼路径。
 *
 * 形状抄自 DSH 的 directory-picker，理由写在 `src/dir-browse.ts` 的文件头。
 */
async function handleBrowse(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  // **只认 GET**。不挡的话 `POST /api/browse` 也会去列家目录 ——
  // 请求方法写错却拿到 200，是最难发现的一类接口错误
  if (req.method !== 'GET') return sendJson(res, 405, { error: `浏览只支持 GET，收到 ${req.method}` })

  const refusal = remoteWorkspaceRefusal()
  if (refusal) return sendJson(res, 403, { error: refusal })

  const path = url.searchParams.get('path') ?? undefined
  try {
    sendJson(res, 200, await listDirs(path))
  } catch (err) {
    if (err instanceof WorkspaceError) {
      return sendJson(res, statusOf(err.code), { error: err.message, code: err.code, path: err.path })
    }
    throw err
  }
}

/**
 * 工作区登记表：GET 列表，POST 新增，PATCH 改名，DELETE 撤销。
 *
 * ⚠️ **`DELETE` 只从表里去掉，不碰磁盘上的目录。** 见 `WorkspaceStore.remove`。
 */
async function handleWorkspaces(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  const refusal = remoteWorkspaceRefusal()
  if (refusal) return sendJson(res, 403, { error: refusal })

  try {
    if (req.method === 'GET') {
      /*
        每条带上**目录还在不在**。

        ★ 这是实测撞出来的：登记过的目录会被删掉（用户自己删、临时目录被
          系统清），而列表照旧显示它们 —— 选中之后在一个不存在的地方跑
          任务，报出来的是工具层的「读不到 .」，看不出真正的原因。

        代价是每条一次 `stat`。几十条的量级无所谓。
      */
      const all = await workspaces.list()
      const withExists = await Promise.all(
        all.map(async (w) => ({ ...w, exists: await isDir(w.path) })),
      )
      return sendJson(res, 200, { workspaces: withExists })
    }

    if (req.method === 'POST') {
      const body = await readJsonBody(req)
      const path = typeof body.path === 'string' ? body.path : ''
      // `mkdir` 让界面能在浏览到的目录下就地新建一个 —— 没有它，用户
      // 得先切到终端建目录再回来。
      // 把**建出来的路径**回给界面：界面据此进入那一层（见 `mkDir()`）
      const newDirPath =
        typeof body.newDir === 'string' && body.newDir ? await createDir(path, body.newDir) : undefined
      const title = typeof body.title === 'string' ? body.title : undefined
      const out = await workspaces.create(path, title)
      return sendJson(res, 200, newDirPath ? { ...out, newDirPath } : out)
    }

    if (req.method === 'PATCH') {
      const body = await readJsonBody(req)
      const id = typeof body.id === 'string' ? body.id : ''
      const title = typeof body.title === 'string' ? body.title : ''
      return sendJson(res, 200, { workspace: await workspaces.rename(id, title) })
    }

    if (req.method === 'DELETE') {
      const id = url.searchParams.get('id') ?? ''
      return sendJson(res, 200, { removed: await workspaces.remove(id) })
    }

    sendJson(res, 405, { error: `不认识的方法 ${req.method}` })
  } catch (err) {
    if (err instanceof WorkspaceError) {
      return sendJson(res, statusOf(err.code), { error: err.message, code: err.code, path: err.path })
    }
    throw err
  }
}

/**
 * 把 `DECISION.md` 的解析结果给界面。
 *
 * 左栏显示的是**这个 agent 会问哪些问题**的规格，和右栏的记账是一对：
 * 规格说应该问什么，记账说实际花了多少。
 *
 * 解析出的问题也一并发出去。一个被悄悄忽略的判定块会让 agent 安静地
 * 少问一个问题，界面上不显示就等于没人会知道（§8.10 不假装成功）。
 */
async function handleSpec(res: ServerResponse): Promise<void> {
  let md: string
  try {
    md = await readFile(DECISION_MD, 'utf8')
  } catch (err) {
    res.writeHead(500, { 'content-type': 'application/json; charset=utf-8' })
    res.end(JSON.stringify({ error: `读不到 DECISION.md：${(err as Error).message}` }))
    return
  }

  const doc = parseDecisionDoc(md)
  res.writeHead(200, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
  res.end(
    JSON.stringify({
      headline: headline(doc),
      summary: summarize(doc),
      blocks: doc.blocks.map((b) => {
        // 这里的谓词编译**不是为了用它的结果，是为了它的失败**。
        // 编译不了的谓词会被编成永不命中的规则，那意味着这份规格里有一条
        // 闸门是空的 —— 界面必须把这件事显示出来，否则汇总看起来完全正常，
        // 而实际上少了一条规则（审计第十一轮 S3）。
        return {
          id: b.id,
          kind: b.kind,
          when: b.when,
          gate: isGate(b),
          questions: b.questions.map((q) => q.id),
          /** 没编译出来的谓词原文。空数组 = 这个块的策略全部可编译 */
          uncompiled: b.policy.filter((r) => compilePredicate(r.when, b) === null).map((r) => r.when),
        }
      }),
      problems: doc.problems,
    }),
  )
}

/**
 * `web/` 的当前版本 —— 各文件的**名字 + 大小 + 修改时间**揉成一个短串。
 *
 * ★ 为什么需要它：这个界面**没有构建步骤**，浏览器拿到什么就一直在跑什么。
 *   改了 `app.js` 之后，一个**一直开着的标签页**跑的还是旧代码 —— 而表现是
 *   「新功能没生效」，看起来完全像那个新功能本身坏了。
 *
 *   实测（2026-09-21）：流式生成做完之后，看到的仍然是「生成完了一次性出全文」。
 *   服务端、CSS、缓存头（`no-store` 全都是对的）挨个排查完，最后发现是那个
 *   标签页从改动之前就一直开着。
 *
 * **每次请求现算**，不缓存：静态文件本来就是每次从磁盘读的，所以改完文件
 * 不重启也生效 —— 版本号必须跟着一起变，缓存住反而会让「刚改完」那一次
 * 比对给出错误答案。
 */
function webBuild(): string {
  let newest = 0
  let bytes = 0
  for (const name of readdirSync(WEB_DIR).sort()) {
    const st = statSync(join(WEB_DIR, name))
    newest = Math.max(newest, st.mtimeMs)
    bytes += st.size
  }
  return `${Math.round(newest).toString(36)}-${bytes.toString(36)}`
}

async function serveStatic(res: ServerResponse, path: string): Promise<void> {
  const rel = path === '/' ? '/index.html' : path
  const safe = normalize(rel).replace(/^(\.\.[/\\])+/, '')
  const file = join(WEB_DIR, safe)
  // `join()` 把第二个参数一律当相对路径，所以 `safe` 里的 `..` 本来就到不了 WEB_DIR 之外 ——
  // 这一条其实是不会触发的。用 `relative()` 显式写出来，是为了让
  // 「文件必须在 WEB_DIR 里面」这个约束**可读**，并且在将来把 join 换成 resolve 时仍然成立。
  const escaped = relative(WEB_DIR, file)
  if (escaped.startsWith('..') || isAbsolute(escaped)) {
    res.writeHead(403).end('forbidden')
    return
  }
  try {
    const data = await readFile(file)
    res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' })
    /*
      index.html 里注入**当前版本**。前端把它和自己加载时那一份比对，
      对不上就提示刷新 —— 只有 HTML 需要注入，因为它是页面唯一的入口，
      别的资源都是它拉起来的（见 `webBuild`）。
    */
    if (file.endsWith('index.html')) {
      // 连 `</head>` 前面那两格缩进一起匹配，否则注入的行会缩进 6 格
      res.end(
        data
          .toString('utf8')
          .replace('\n  </head>', `\n    <meta name="jl-build" content="${webBuild()}" />\n  </head>`),
      )
      return
    }
    res.end(data)
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end('404')
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`)
  try {
    if (url.pathname === '/api/run') {
      await handleRun(req, res, url)
      return
    }
    if (url.pathname === '/api/build') {
      sendJson(res, 200, { build: webBuild() })
      return
    }
    if (url.pathname === '/api/spec') {
      await handleSpec(res)
      return
    }
    if (url.pathname === '/api/session') {
      await handleSession(req, res, url)
      return
    }
    if (url.pathname === '/api/sessions') {
      await handleSessions(req, res, url)
      return
    }
    if (url.pathname === '/api/browse') {
      await handleBrowse(req, res, url)
      return
    }
    if (url.pathname === '/api/workspaces') {
      await handleWorkspaces(req, res, url)
      return
    }
    await serveStatic(res, url.pathname)
  } catch (err) {
    // ★ **错误一律 JSON**，和成功路径同一个形状。
    //   以前这里是纯文本，而浏览/工作区那几条返回 JSON —— 界面统一按
    //   `res.json()` 解析就会在出错时炸在一个语法错误上，看不到真正的原因。
    //   「出错时接口变成另一种格式」是最典型的自伤。
    if (!res.headersSent) sendJson(res, 500, { error: (err as Error).message })
    else res.end()
  }
})

// 端口被占是**用户最可能遇到的第一个错误**（多半是自己已经起了一个）。
// 原始堆栈只说 EADDRINUSE，不说该怎么办 —— 那条信息对读的人没有用。
server.on('error', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  ✗ 端口 ${PORT} 已被占用 —— 多半是已经有另一个 JevLoop 在跑。`)
    console.error(`    占用它的进程：  ss -ltnp | grep :${PORT}`)
    console.error(`    换一个端口：    PORT=7800 node --experimental-strip-types server.ts\n`)
    process.exit(1)
  }
  // 别的监听错误没有「换个端口」这种统一解法，原样抛出去
  throw err
})

/*
  把 v1 布局的会话搬进 v2（按 cwd 分项目）。**启动时做一次**，不放在 store
  里 —— 那是部署的事，不是每次读写的事（见 `session-migrate.ts` 的文件头）。
  幂等，搬不动的留在原地。
*/
await migrateLegacyLayout(SESSIONS_DIR)

server.listen(PORT, HOST, async () => {
  const dir = await workspaceRoot()

  /*
    ★ **把演示目录登记成一个工作区。**

    没有这一步的话，默认配置下工作区列表是空的 —— 而会话挂在它们的 `cwd`
    上，于是「这个会话属于哪个工作区」答不出来，工作区那一栏也永远显示
    「演示目录（未登记）」。登记之后默认状态就是自洽的：一个工作区，
    会话挂在它下面。

    幂等（`create` 对同一个路径返回原来那条），所以重启几次都不会多。
    用户可以把它从列表里移除 —— 移除只是不登记，目录还在。
  */
  try {
    // ★ **服务器实际在用的那个目录也登记上。** 不登记的话，它上面的会话
    //   会全部落在「没有工作区」那一类里，而那个分类对用户没有意义 ——
    //   他明明正在这个目录里干活。
    await workspaces.create(dir, CWD_ROOT ? basename(dir) : '演示目录')
  } catch (err) {
    // 登记不上不该拦住服务启动 —— 它只是个便利，目录本身照样能用
    console.error(`  ⚠ ${dir} 没能登记成工作区：${(err as Error).message}`)
  }

  const count = (await workspaces.list().catch(() => [])).length
  console.log(`\n  JevLoop · http://${HOST}:${PORT}`)
  console.log(`  decision   : ${resolveProvider().name}`)
  // 提醒一句：这条链降级时是会打日志的，而**降级意味着数字不能用**
  console.log(`               （降级会打 ▲；重试说明主后端在过载，不是「这次慢」）`)
  console.log(`  generator  : ${resolveGenerator().name}`)
  console.log(`  cwd        : ${dir}${CWD_ROOT ? '' : '  (temporary demo directory, registered as a workspace)'}`)
  console.log(`  workspaces : ${count}   sessions ${JEVLOOP_HOME}/sessions/`)
  console.log(
    `  host       : ${HOST}` +
      (HOST === '127.0.0.1'
        ? '  (loopback only)'
        : '  ⚠ exposed to the network — this server has no auth, anyone can make it run tasks'),
  )
  console.log('')
})
