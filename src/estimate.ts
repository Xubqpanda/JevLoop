/**
 * JevLoop · token 估算
 *
 * ══════════════════════════════════════════════════════════════
 *  **这是启发式，不是 tokenizer。** 偏差约 ±30%，方向不可控。
 * ══════════════════════════════════════════════════════════════
 *
 * 它存在的原因：决策帧有硬上限（512/1024），而我们必须在**发请求之前**
 * 就知道一帧会不会超 —— 那需要一个不依赖网络的估算。真 tokenizer 要带
 * 一份词表，而零依赖是这个仓库的对外承诺。
 *
 * ── 中文按 1 字 1 token，英文按 4 字符 1 token ──────────────────
 *
 * 这条差别很大，别按英文估中文：同样 400 字符，英文约 100 token，
 * 中文约 400 token。所以 `clip()` 的注释里反复强调「中文 1 字 ≈ 1 token」。
 *
 * ── 估算值可信吗 ────────────────────────────────────────────────
 *
 * 生成侧**可以核对**：`HttpGenerator` 把 provider 报的 `inputTokens` 带回来了，
 * 那是真值。`context.ts` 的 `priceGenerateRequest` 会把它和我们的估算并排报出来
 * —— 两个数放在一起，偏差是可观测的，不必信这个文件里的任何断言。
 *
 * 判定侧核对不了（Jev 协议不回 usage），所以那里只能靠这条启发式，
 * 以及 `budget.ts` 的 `validate()` 在发请求之前拦。
 *
 * 放在 L0 是因为它**零依赖**且是纯函数 —— 谁都能用它，它不依赖谁。
 * 它以前住在 `budget.ts`（L1），而 `context.ts` 也是 L1，同层不许互相依赖
 * （docs/CODE-STYLE.md §11），所以抽出来。
 *
 * @module JevLoop/estimate
 */

/**
 * **字符启发式**，不是真 tokenizer。偏差约 ±30%，方向不可控。
 * 用途是「提前拦住明显超预算的帧」，不是精确计量 —— 名字里的 Rough 是刻意的。
 */
export function estimateTokens(value: unknown): number {
  const s = typeof value === 'string' ? value : JSON.stringify(value ?? '')
  if (!s) return 0
  let cjk = 0
  for (const ch of s) {
    const c = ch.codePointAt(0)!
    if ((c >= 0x2e80 && c <= 0x9fff) || (c >= 0xac00 && c <= 0xd7af) || (c >= 0x3040 && c <= 0x30ff)) cjk++
  }
  return Math.ceil(cjk + (s.length - cjk) / 4)
}
