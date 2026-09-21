/**
 * JevLoop · Fallback —— 主 Provider 挂了怎么办
 *
 * **从 `provider.ts` 切出来** —— 那个文件住着**四件事**
 * （HTTP 判定后端 / 响应归一化 / Mock / 降级链），按 docs/CODE-STYLE.md §12
 * 「一句话说不完就是住着两件以上」该拆。接缝是**各自的契约形状不同**：
 *  * Fallback 只做链式重试，不认识任何具体后端
 *
 * @module JevLoop/provider-fallback
 */

import type { Provider, DecideRequest, DecideResponse } from './seam-provider.ts'

// ═══════════════════════════════════════════════════════════
// Fallback —— 主 Provider 挂了怎么办
// ═══════════════════════════════════════════════════════════

/**
 * 链式兜底：按顺序试每一个后端，第一个成功的胜出。
 *
 * 全部失败时**记原计划用的那个后端名**（不覆盖成 `none`）再抛出，
 * 这样日志里看得出「本来想用谁」—— 覆盖掉就等于把排查线索删了（§8.10）。
 */
export class FallbackProvider implements Provider {
  readonly name: string
  #chain: Provider[]
  #onFallback?: (err: unknown, from: string, to: string) => void

  constructor(chain: Provider[], onFallback?: (err: unknown, from: string, to: string) => void) {
    if (!chain.length) throw new Error('FallbackProvider 至少需要一个 Provider')
    this.#chain = chain
    this.name = chain.map((p) => p.name).join('→')
    this.#onFallback = onFallback
  }

  async decide(req: DecideRequest): Promise<DecideResponse> {
    let lastErr: unknown
    for (let i = 0; i < this.#chain.length; i++) {
      const p = this.#chain[i]!
      try {
        const res = await p.decide(req)
        if (i > 0) {
          res.degraded = true
          res.warnings = [...(res.warnings ?? []), `主 Provider 失败，降级到 ${p.name}`]
        }
        return res
      } catch (err) {
        lastErr = err
        if (this.#chain[i + 1]) this.#onFallback?.(err, p.name, this.#chain[i + 1]!.name)
      }
    }
    throw lastErr
  }
}
