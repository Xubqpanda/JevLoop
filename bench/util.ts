/**
 * JevLoop · 两台标定台共用的两件小事
 *
 * 只有两样东西，而它们都是**已经在两个地方各写了一遍**的：
 * 颜色和重试。写两遍的后果不是「多几行」，是**只写对了一遍**。
 *
 * @module JevLoop/util
 */

/** ANSI 上色。两台台子用同一套，输出才对得起来 */
export const C = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
}

/**
 * 跑一次，失败就退避重试。
 *
 * ★ LLM 和判定后端都在网络上，而 `fetch failed` / `UND_ERR_CONNECT_TIMEOUT`
 *   是**常态**不是异常（这台机器上 github 和 api 都时常连不上）。
 *
 *   实测（2026-09-21）：没有它的话，一次 `api.deepseek.com` 连接超时会把
 *   一整场 7 任务的对比**打挂**，前面跑出来的样本全部作废 —— 而这个函数
 *   在 `run.ts` 里早就有了，只有**新写的** `compare.ts` 漏了。
 *   同一件事在两处各写一遍，结果就是只写对了一处。
 */
export async function withRetry<T>(what: string, fn: () => Promise<T>, tries = 3): Promise<T> {
  let last: unknown
  for (let i = 1; i <= tries; i++) {
    try {
      return await fn()
    } catch (err) {
      last = err
      if (i < tries) {
        const wait = i * 2000
        process.stdout.write(
          C.yellow(`      ${what} 第 ${i} 次失败（${(err as Error).message.slice(0, 50)}），${wait / 1000}s 后重试\n`),
        )
        await new Promise((r) => setTimeout(r, wait))
      }
    }
  }
  throw last
}
