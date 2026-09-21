/**
 * JevLoop · .env 加载器
 *
 * Node 内建 `--env-file` 但版本要求不一，而且缺文件时会直接报错。
 * 这里用一个 20 行的读取器：零依赖、任何 Node 22 都能用、文件不存在就静默跳过。
 *
 * 只做四件事：`KEY=VALUE`、`export KEY=VALUE`、`#` 注释、两端引号。
 * **不做变量插值、不做多行** —— 需要的自己去 dotenv。
 *
 * 认不出来的行**跳过并报出来**（返回值里的 `skipped`），不静默当成键名 ——
 * 边界上的误解析必须出声（§6 把配置解析点名为真实边界之一）。
 *
 * @module JevLoop/env
 */

import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'

export interface LoadEnvOptions {
  /** 从哪个目录开始往上找 .env，默认 process.cwd() */
  cwd?: string
  /** 已经存在的环境变量是否不覆盖，默认 true */
  keepExisting?: boolean
}

/**
 * 键名的合法形状。
 *
 * ★ 为什么必须校验：不校验时 `export KEY=VALUE`（shell 与 `.env` 里最常见的写法之一）
 * 会被当成**键名的一部分** —— 得到的是「键不存在」，外加一个凭空多出来的
 * `export KEY`。用户以为设了 key，实际没有，于是后续表现为「没有 key → 退回 Mock」，
 * 排查方向被完全带偏。**一个既不跳过也不报错的写法，比明确不支持更糟**：
 * 它产出的是一个看起来成功的错东西（§8.10）。
 */
const KEY_RE = /^[A-Za-z_][A-Za-z0-9_]*$/

export interface LoadEnvResult {
  /** 真正设上的键名。不包含值，避免把密钥打进日志 */
  loaded: string[]
  /** 认不出来、因而**跳过**的行（原文）。空数组 = 整份文件都懂了 */
  skipped: string[]
}

/** 找到并加载 .env。支持 `KEY=VALUE` / `export KEY=VALUE` / `#` 注释 / 两端引号。 */
export function loadEnv(opts: LoadEnvOptions = {}): LoadEnvResult {
  const keepExisting = opts.keepExisting ?? true
  const file = findEnvFile(opts.cwd ?? process.cwd())
  if (!file) return { loaded: [], skipped: [] }

  const loaded: string[] = []
  const skipped: string[] = []
  for (const raw of readFileSync(file, 'utf8').split('\n')) {
    let line = raw.trim()
    if (!line || line.startsWith('#')) continue

    // `export KEY=VALUE` —— 剥掉前缀再解析，与 dotenv 行为一致
    if (line.startsWith('export ')) line = line.slice('export '.length).trim()

    const eq = line.indexOf('=')
    if (eq <= 0) {
      skipped.push(raw.trim())
      continue
    }

    const key = line.slice(0, eq).trim()
    let value = line.slice(eq + 1).trim()
    // 去掉两端成对的引号
    const dq = '"'
    const sq = '\''
    const paired = (q: string) => value.startsWith(q) && value.endsWith(q)
    if (value.length >= 2 && (paired(dq) || paired(sq))) {
      value = value.slice(1, -1)
    }

    // ★ 键名不合法的行**跳过并报出来**，绝不 `process.env[key] = value`。
    //   以前这里没有这一道，于是 `export KEY` → 设了一个叫 `export KEY` 的变量，
    //   还把它列进返回值 —— 调用方打印「已加载 …」，看起来完全成功。
    if (!KEY_RE.test(key)) {
      skipped.push(raw.trim())
      continue
    }

    if (keepExisting && process.env[key] !== undefined) continue
    process.env[key] = value
    loaded.push(key)
  }
  return { loaded, skipped }
}

/** 从 cwd 开始往上找，最多找 3 层 */
function findEnvFile(start: string): string | undefined {
  let dir = resolve(start)
  for (let i = 0; i < 3; i++) {
    const candidate = resolve(dir, '.env')
    if (existsSync(candidate)) return candidate
    const parent = resolve(dir, '..')
    if (parent === dir) break
    dir = parent
  }
  return undefined
}
