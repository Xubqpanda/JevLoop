#!/usr/bin/env node
/**
 * JevLoop · 开发服务器
 *
 *   node --experimental-strip-types server.ts
 *
 * 把 agent loop 的事件流通过 SSE 推给浏览器，同时托管 web/ 下的界面。
 * 只用 Node 标准库 —— 没有 express，没有构建步骤。
 *
 * 为什么是 SSE 而不是 WebSocket：事件是**单向**的（服务端 → 浏览器），
 * SSE 是浏览器原生的、会自动重连的、用 GET 就能开的单向通道。
 * 双向通信在这个界面里没有任何用途。
 *
 * @module jevloop/server
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readFile, mkdtemp, writeFile } from 'node:fs/promises'
import { extname, join, normalize } from 'node:path'
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
}

// ═══════════════════════════════════════════════════════════
// 演示工作目录
//
// 界面要能一键跑，所以服务端自己造一个。真实的 cwd 可以由请求带进来。
// ═══════════════════════════════════════════════════════════

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
  const task = url.searchParams.get('task')?.trim() || '列出工作目录里的文件，读取其中的 TypeScript 文件，说明它定义了哪些函数'
  const cwd = url.searchParams.get('cwd') || (await makeWorkspace())
  const maxSteps = Number(url.searchParams.get('maxSteps') ?? 12)

  const send = openStream(res)

  // 浏览器关掉页面时停下来 —— 否则 loop 会继续跑完，白花判定和生成的钱
  let aborted = false
  req.on('close', () => {
    aborted = true
  })

  try {
    const provider = resolveProvider()
    const generator = resolveGenerator()
    const meter = new Meter()

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
        if (!aborted) send(e)
      },
    })
  } catch (err) {
    if (!aborted) {
      send({
        type: 'run:end',
        halt: 'error',
        steps: 0,
        answer: `服务端异常：${(err as Error).message}`,
        stats: new Meter().stats,
      })
    }
  } finally {
    if (!aborted) res.end()
  }
}

async function serveStatic(res: ServerResponse, path: string): Promise<void> {
  const rel = path === '/' ? '/index.html' : path
  const safe = normalize(rel).replace(/^(\.\.[/\\])+/, '')
  const file = join(WEB_DIR, safe)
  if (!file.startsWith(WEB_DIR)) {
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

server.listen(PORT, HOST, () => {
  console.log(`\n  JevLoop · http://${HOST}:${PORT}`)
  console.log(`  判定后端 : ${resolveProvider().name}`)
  console.log(`  生成后端 : ${resolveGenerator().name}\n`)
})
