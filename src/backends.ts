/**
 * 判定后端与生成后端的解析。
 *
 * 这两件事的决定依据不同，所以分成两个函数而不是一个「配置」：
 *
 *   - 判定后端看的是「哪里有可用的判定模型」——有 API key 用托管的，
 *     没有就用本地的，都没有就用确定性规则表，保证任何环境都能跑起来。
 *   - 生成后端看的是「用户想用哪个 LLM」，缺 key 时**不猜**，退回脚本生成器
 *     并把这件事说出来。
 *
 * 两者都遵守同一条规则：**永远有兜底，永远不静默**。降级会发生，
 * 但每一层降级都会往回调里报一次。
 *
 * @module JevLoop/backends
 */

import { HttpProvider } from './provider-http.ts'
import { MockProvider } from './provider-mock.ts'
import { FallbackProvider } from './provider-fallback.ts'
import { RetryingProvider } from './provider-retry.ts'
import { RetryingGenerator } from './llm.ts'
import type { RetryOptions } from './retry.ts'
import { HttpGenerator, ScriptedGenerator } from './llm.ts'
import type { Provider } from './seam-provider.ts'
import type { Generator } from './llm.ts'

/** 降级回调。每次从一层掉到下一层时调用一次，不是每次请求都调。 */
export type FallbackNotice = (error: unknown, from: string, to: string) => void

/** 判定后端的选择。 */
/**
 * 钉住的 Jev 版本。
 *
 * `docs.typesafe.ai/models` 里 `jev-latest` 当前解析到 `jev-1.13.0`。
 * **换它要当成一次决定**：改这里、重跑 bench、看数字动不动 —— 因为
 * `DECISION.md` 里那些门限是拿这个版本量出来的。
 */
export const PINNED_JEV_MODEL = 'jev-1.13.0'

export interface ProviderChoice {
  /** 官方 Jev 的 baseUrl，默认 `https://api.typesafe.ai` */
  jevUrl?: string
  /** 本地 Laya sidecar 的 baseUrl，默认 `http://127.0.0.1:7789` */
  layaUrl?: string
  /** TypeSafe key。缺省读 `TYPESAFE_API_KEY` */
  apiKey?: string
  /** 指定 checkpoint 或 model 名 */
  model?: string
  /** 强制只用某一个，跳过自动探测 */
  prefer?: 'jev' | 'laya' | 'mock'
  onFallback?: FallbackNotice
  /**
   * 链尾兜底。缺省是保守 {@link MockProvider}。
   *
   * 为什么要有这个口子：Mock 给的是保守答案（概率 0.5），
   * 于是 `pickTool` 的置信度门限会走到 `escalate` —— loop 在第一步就停下。
   * 对「无 key 也能跑完」的场景（`npm run demo`）那是错的兜底。
   * 但**内核不该知道什么才是对的兜底**（docs/CODE-STYLE.md §8.6：规则属于场景），
   * 所以由调用方注入，比如 `examples/rule-judge.ts`。
   */
  lastResort?: Provider
  /**
   * 每一跳的重试参数。缺省见 `provider-retry.ts`（3 次 / 300ms 起 / 上限 5s）。
   *
   * ★ 有了它，托管后端的一次**瞬时**过载（429 / 5xx / 超时）不会立刻把
   *   整轮打到兜底上。实测代价：那时 bench 报出来的数字全是 mock 的。
   */
  retry?: RetryOptions
}

/**
 * 挑一个判定后端。
 *
 * 顺序：官方 Jev（有 key 时）→ 本地 Laya → 保守 Mock。
 *
 * 没有 key 时**跳过** Jev 而不是先试一次再降级 —— 少一次注定失败的网络往返。
 */
export function resolveProvider(choice: ProviderChoice = {}): Provider {
  const mock = new MockProvider()
  const notice = choice.onFallback

  const last = choice.lastResort ?? mock

  if (choice.prefer === 'mock') return last

  const laya = new HttpProvider({
    baseUrl: choice.layaUrl ?? 'http://127.0.0.1:7789',
    name: 'laya',
    defaultModel: choice.model ?? 'typed-decisions',
    timeoutMs: 20_000,
  })

  if (choice.prefer === 'laya') {
    // `onFallback?:` 已经表达了「可以没有」这个语义，不需要在调用点再表达一遍。
    // 这里曾经写的是一个恒等三元（两个分支构造同一个对象）—— 它看起来在处理
    // 「notice 可能缺席」，实际什么都没处理，而读代码的人会以为有一层分支保护。
    return new FallbackProvider([laya, last], notice)
  }

  const apiKey = choice.apiKey ?? process.env.TYPESAFE_API_KEY
  const jev = new HttpProvider({
    baseUrl: choice.jevUrl ?? 'https://api.typesafe.ai',
    name: 'jev',
    /*
      ★ **钉住版本，不用 `jev-latest` 别名。**

      官方文档（`docs.typesafe.ai/models`）的原话：

        「An alias moves when a new release ships, so the answers behind it
          can change without a change on your side. … **If you have tuned
          confidence thresholds against a specific version, pin that
          version's ID instead of the alias** and move to the new one on
          your own schedule.」

      我们**确实调过**门限 —— `DECISION.md` 里现在有 8 条
      （`prob:needs_tool >= 0.5`、`top >= 0.6`、`prob:ok >= 0.6` …），
      每一条都是拿 bench 量出来的。别名一动，那些数字背后的模型就换了，
      而**我们这边一处都没改**，bench 的历史数字也不再可比。

      换版本是一次**决定**：改这里，重跑 bench，看数字动不动。
      `choice.model` 仍然可以覆盖（试验新版本时用）。
    */
    defaultModel: choice.model ?? PINNED_JEV_MODEL,
    ...(apiKey ? { apiKey } : {}),
  })

  /*
    ★ **重试在链里面，包住每一个会瞬时失败的后端。**

    重试和降级是两件事：

        RetryingProvider   这一跳**暂时**不行 → 等一下再问**同一个**
        FallbackProvider   这一跳**就是**不行 → 换下一个

    包在链**外面**是不行的，而且一试就知道：链的最后一级是 mock，
    它**永远不抛**，所以外面的重试包装器一次也不会触发。

    实测（2026-09-21）：托管 Jev 的 `529 system_overloaded` 因为当时
    没有这一层，直接把整轮判定打到 mock 的恒定 0.5 上。

    `last`（mock / 规则判定）不包 —— 它是本地兜底，不会瞬时失败。
  */
  const retry = (p: Provider): Provider =>
    new RetryingProvider(p, {
      ...(choice.retry ?? {}),
      // 重试**要说出来**：不说的话，界面上只表现为「这一次特别慢」，
      // 而真实情况是主后端在过载 —— 那两件事的排查方向完全不同
      onRetry: (info) =>
        notice?.(
          new Error(`${info.code}，等 ${Math.round(info.delayMs)}ms 后重试${info.fromServer ? '（服务端要求的）' : ''}`),
          info.who,
          info.who,
        ),
    })

  if (choice.prefer === 'jev') {
    return new FallbackProvider([retry(jev), last], notice)
  }

  const chain = apiKey ? [retry(jev), retry(laya), last] : [retry(laya), last]
  return new FallbackProvider(chain, notice)
}

/** 生成后端的选择。 */
export interface GeneratorChoice {
  /** OpenAI 兼容的 baseUrl，默认读 `DEEPSEEK_BASE_URL` */
  baseUrl?: string
  /** 缺省读 `DEEPSEEK_API_KEY` */
  apiKey?: string
  /** 缺省读 `DEEPSEEK_MODEL`，再缺省 `deepseek-flash` */
  model?: string
  /** 强制用脚本生成器（离线、确定性） */
  scripted?: boolean
  /** 脚本生成器模拟的延迟，毫秒 */
  scriptedLatencyMs?: number
  /**
   * 生成后端每一跳的重试参数。缺省见 `retry.ts`（3 次 / 300ms 起 / 上限 5s）。
   *
   * ★ 生成**没有降级链**，所以重试是唯一的补救。实测（2026-09-21）：
   *   `npm run demo` 在一次 `api.deepseek.com` 连接超时上直接抛栈退出 ——
   *   判定全都正常跑完了，最后那一次生成挂了，整个 demo 就失败。
   */
  retry?: RetryOptions
}

/**
 * 挑一个生成后端。
 *
 * 有 key 就用 HTTP，没有就用 {@link ScriptedGenerator}。
 *
 * 脚本生成器不是「假的」——它真的根据素材产出文本，只是不做语言推理。
 * 它存在的理由是让 `npm run demo` 在没有 key、没有网络时也能跑完整个 loop，
 * 并且让「判定 : 模型」这个比例是真实的（判定是真的，只有这一步是脚本）。
 */
export function resolveGenerator(choice: GeneratorChoice = {}): Generator {
  const apiKey = choice.apiKey ?? process.env.DEEPSEEK_API_KEY
  if (choice.scripted || !apiKey) {
    return new ScriptedGenerator(
      choice.scriptedLatencyMs === undefined ? {} : { latencyMs: choice.scriptedLatencyMs },
    )
  }

  /*
    ★ **包一层重试，不含糊。**

    生成是整轮里**唯一**一次真正花钱的调用，也是**唯一没有降级**的一步 ——
    判定挂了有 `FallbackProvider` 兜底，生成挂了整轮就没了。一次连接超时
    不该把跑完的 loop 全丢掉。

    脚本生成器不包：它不碰网络，不会瞬时失败。
  */
  return new RetryingGenerator(
    new HttpGenerator({
      baseUrl: choice.baseUrl ?? process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com/v1',
      apiKey,
      model: choice.model ?? process.env.DEEPSEEK_MODEL ?? 'deepseek-flash',
    }),
    choice.retry ?? {},
  )
}
