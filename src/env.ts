/**
 * nanojev · .env 加载器
 *
 * Node 内建 `--env-file` 但版本要求不一，而且缺文件时会直接报错。
 * 这里用一个 20 行的读取器：零依赖、任何 Node 22 都能用、文件不存在就静默跳过。
 *
 * 只做三件事：`KEY=VALUE`、`#` 注释、两端引号。
 * **不做变量插值、不做多行** —— 需要的自己去 dotenv。
 */

import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

export interface LoadEnvOptions {
  /** 从哪个目录开始往上找 .env，默认 process.cwd() */
  cwd?: string
  /** 已经存在的环境变量是否不覆盖，默认 true */
  keepExisting?: boolean
}

/** 找到并加载 .env。返回加载到的键名（不返回值，避免把密钥打进日志）。 */
export function loadEnv(opts: LoadEnvOptions = {}): string[] {
  const keepExisting = opts.keepExisting ?? true
  const file = findEnvFile(opts.cwd ?? process.cwd())
  if (!file) return []

  const loaded: string[] = []
  for (const raw of readFileSync(file, 'utf8').split('\n')) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue

    const eq = line.indexOf('=')
    if (eq <= 0) continue

    const key = line.slice(0, eq).trim()
    let value = line.slice(eq + 1).trim()
    // 去掉两端成对的引号
    if (value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) {
      value = value.slice(1, -1)
    }

    if (keepExisting && process.env[key] !== undefined) continue
    process.env[key] = value
    loaded.push(key)
  }
  return loaded
}

/** 从 cwd 开始往上找，最多找 3 层 */
function findEnvFile(start: string): string | undefined {
  let dir = resolve(start)
  for (let i = 0; i < 3; i++) {
    const candidate = resolve(dir, '.env')
    try {
      readFileSync(candidate)
      return candidate
    } catch {
      /* 继续往上 */
    }
    const parent = resolve(dir, '..')
    if (parent === dir) break
    dir = parent
  }
  return undefined
}
