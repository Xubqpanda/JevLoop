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

import { HttpProvider, MockProvider, FallbackProvider } from './provider.ts'
import { HttpGenerator, ScriptedGenerator } from './llm.ts'
import type { Provider } from './types.ts'
import type { Generator } from './llm.ts'

/** 降级回调。每次从一层掉到下一层时调用一次，不是每次请求都调。 */
export type FallbackNotice = (error: unknown, from: string, to: string) => void

/** 判定后端的选择。 */
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

  if (choice.prefer === 'mock') return mock

  const laya = new HttpProvider({
    baseUrl: choice.layaUrl ?? 'http://127.0.0.1:7789',
    name: 'laya',
    defaultModel: choice.model ?? 'typed-decisions',
    timeoutMs: 20_000,
  })

  if (choice.prefer === 'laya') {
    return notice ? new FallbackProvider([laya, mock], notice) : new FallbackProvider([laya, mock])
  }

  const apiKey = choice.apiKey ?? process.env.TYPESAFE_API_KEY
  const jev = new HttpProvider({
    baseUrl: choice.jevUrl ?? 'https://api.typesafe.ai',
    name: 'jev',
    defaultModel: choice.model ?? 'jev-latest',
    ...(apiKey ? { apiKey } : {}),
  })

  if (choice.prefer === 'jev') {
    return notice ? new FallbackProvider([jev, mock], notice) : new FallbackProvider([jev, mock])
  }

  const chain = apiKey ? [jev, laya, mock] : [laya, mock]
  return notice ? new FallbackProvider(chain, notice) : new FallbackProvider(chain)
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

  return new HttpGenerator({
    baseUrl: choice.baseUrl ?? process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com/v1',
    apiKey,
    model: choice.model ?? process.env.DEEPSEEK_MODEL ?? 'deepseek-flash',
  })
}
