#!/usr/bin/env node
/**
 * JevLoop · 开发服务器
 *
 *   node --experimental-strip-types server.ts
 *   CWD_ROOT=/path/to/project node --experimental-strip-types server.ts
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
 * @module jevloop/server
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readFile, mkdtemp, writeFile } from 'node:fs/promises'
import { realpathSync } from 'node:fs'
import { extname, isAbsolute, join, normalize, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'

import { Decider } from './src/decide.ts'
import { Meter } from './src/meter.ts'
import { runAgent } from './src/agent.ts'
import { resolveProvider, resolveGenerator } from './src/backends.ts'
import { loadEnv } from './src/env.ts'
import type { AgentEvent } from './src/events.ts'

const ROOT = fileURLToPath(new URL('.', import.meta.url))
const WEB_DIR = join(ROOT, 'web')
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

async function makeWorkspace(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'jevloop-'))
  for (const [name, body] of Object.entries(DEMO_FILES)) {
    await writeFile(join(dir, name), body, 'utf8')
  }
  return dir
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

/** 演示目录只造一次：反复运行共用同一个，可预测，也不泄漏临时目录 */
function workspaceRoot(): Promise<string> {
  if (CWD_ROOT) return Promise.resolve(CWD_ROOT)
  demoWorkspace ??= makeWorkspace()
  return demoWorkspace
}

// ═══════════════════════════════════════════════════════════
// SSE
// ═══════════════════════════════════════════════════════════

/**
 * 开一条 SSE 通道。
 *
 * 每条事件单独一行 `data:`，用空行分隔 —— SSE 的帧格式。
 * **不做事件名分类**（`event:` 字段）：前端只需要一个 `onmessage`，
 * 分流交给 JSON 里的 `type`，这样加新事件类型时前端不用改协议层。
 */
function openStream(res: ServerResponse): (e: AgentEvent) => void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    // 关掉 nginx 之类的缓冲，否则事件会被攒着一起发
    'x-accel-buffering': 'no',
  })
  res.write(': connected\n\n')
  return (e) => res.write(`data: ${JSON.stringify(e)}\n\n`)
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
  const cwd = await workspaceRoot()

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

  try {
    const provider = resolveProvider()
    const generator = resolveGenerator()

    await runAgent({
      task,
      cwd,
      decider: new Decider({ provider, meter }),
      generator,
      maxSteps,
      // 界面上「需要授权」一律先批准：这是一个演示环境，
      // 真拒绝会让 loop 在第一步就停，看不到后面的东西。
      // 真实的授权交互应该是前端弹一个确认框再回传。
      onAskHuman: async () => true,
      onEvent: (e) => {
        if (!clientGone) send(e)
      },
    })
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
    if (!clientGone) res.end()
  }
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
    await serveStatic(res, url.pathname)
  } catch (err) {
    res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' }).end((err as Error).message)
  }
})

server.listen(PORT, HOST, async () => {
  console.log(`\n  JevLoop · http://${HOST}:${PORT}`)
  console.log(`  判定后端 : ${resolveProvider().name}`)
  console.log(`  生成后端 : ${resolveGenerator().name}`)
  console.log(
    `  工作目录 : ${await workspaceRoot()}${CWD_ROOT ? '' : '  （临时演示目录；设 CWD_ROOT 可换成真实目录）'}`,
  )
  console.log(
    `  监听地址 : ${HOST}` +
      (HOST === '127.0.0.1' ? '  （仅本机）' : '  ⚠ 已暴露到网络 —— 本服务没有鉴权，任何人都能让它跑任务'),
  )
  console.log('')
})
