/**
 * JevLoop · 门限覆盖
 *
 * `DECISION.md` 里的门限是我们量出来的**默认值**（每个数字怎么来的写在那个
 * 文件里）。但默认值不一定适合你的任务 —— 比如你的交付标准更严，想让
 * `unsupported` 从 0.5 提到 0.7。这个文件管的就是那件事：**怎么覆盖，
 * 以及怎么保证覆盖不撒谎**。
 *
 * ── 两条设计约束 ────────────────────────────────────────────────
 *
 * **① 覆盖必须留痕。** 一个跑在 0.7 上的运行，日志里不能写着
 *   `prob:unsupported >= 0.5 → revise` —— 那是**假装**（§8.10）。
 *   所以覆盖会一路带到规则的 `reason` 和 `run:start` 事件里去。
 *
 * **② 覆盖没生效必须是响的。** 名字写错一个字母（`can_deliver.unsupport`
 *   少个 ed）而静默无效，是这个项目最怕的那类错：**你以为加了一道闸门，
 *   其实什么都没发生**，而且没有任何东西报错。所以解析层不认识的名字
 *   全部留下，由调用方当场炸掉（同 `decisions.ts` 里 `block()` 的做法：
 *   「写错名字要当场知道，不是静默给一个空问题集」）。
 *
 * 零 import、零 IO —— 最底层（L0），谁都能用它，它不依赖谁。
 *
 * @module JevLoop/gates
 */

/**
 * 覆盖表：`<块 id>.<问题 id>` → 新门限。
 *
 *     can_deliver.unsupported = 0.7
 *     grade_risk.risk         = 3
 *
 * ★ **键是「块 + 问题」，不是行号。** 按行号（`can_deliver#0`）在
 *   `DECISION.md` 里插一条规则之后就会指到别的门限上 —— 而那**不会报错**，
 *   只会悄悄地让另一个判断换一个阈值。块和问题的名字是人写出来的、
 *   也是人读得懂的，改文件的时候它们会跟着一起改。
 *
 * ⚠️ 同一个块里**同一个问题有多条规则**时（比如两档风险），一次覆盖
 *    同时作用于它们 —— 键里没有区分它们的东西。这是有意的：那个键是
 *    给人写的，而写的人想的是「这个问题的门限」，不是「第几条规则」。
 */
export type GateOverrides = Readonly<Record<string, number>>

/** 覆盖的一项：解析之后的形状（`reason` 里报给人看的就是它） */
export interface GateSetting {
  readonly key: string
  readonly value: number
}

/**
 * 解析 `can_deliver.unsupported=0.7, grade_risk.risk=3`。
 *
 * 分隔符 `,` `;` 换行都行（命令行和环境变量里都顺手）。
 * 认不出来的条目**不丢**，落在 `problems` 里 —— 见文件头第 ② 条。
 */
export function parseGates(spec: string): { overrides: GateOverrides; problems: string[] } {
  const overrides: Record<string, number> = {}
  const problems: string[] = []

  for (const raw of (spec ?? '').split(/[,;\n]/)) {
    const item = raw.trim()
    if (!item) continue

    const eq = item.indexOf('=')
    if (eq <= 0) {
      problems.push(`认不出 '${item}'（应当是 块.问题=数值）`)
      continue
    }
    const key = item.slice(0, eq).trim()
    const text = item.slice(eq + 1).trim()
    const value = Number(text)

    if (!/^[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*$/.test(key)) {
      problems.push(`认不出问题名 '${key}'（应当是 块.问题）`)
      continue
    }
    if (!Number.isFinite(value)) {
      problems.push(`'${key}' 的门限 '${text}' 不是数`)
      continue
    }
    overrides[key] = value
  }

  return { overrides, problems }
}

/**
 * 把整份覆盖表**按块**切开，交给那个块的编译。
 *
 * 返回 `{ forBlock, unused }`：`unused` 是这份表里没有任何块认领的项。
 * 调用方拿到非空的 `unused` 就该停下 —— 它意味着有人以为改了一个门限。
 */
export function splitGates(
  overrides: GateOverrides,
): { keys: string[]; byBlock: Record<string, Record<string, number>>; unused: string[] } {
  const byBlock: Record<string, Record<string, number>> = {}
  const unused: string[] = []

  for (const [key, value] of Object.entries(overrides)) {
    const dot = key.indexOf('.')
    const block = key.slice(0, dot)
    const question = key.slice(dot + 1)
    if (dot <= 0 || !question) {
      unused.push(key)
      continue
    }
    const perBlock = (byBlock[block] ??= {})
    perBlock[question] = value
  }

  return { keys: Object.keys(overrides), byBlock, unused }
}

/**
 * 覆盖表 → 一行给人看的字。`null` = 一项都没有（那就什么都不显示）。
 *
 * 排序过：同一份覆盖在日志和界面上每次都长一样，比对的才有意义。
 */
export function describeGates(overrides: GateOverrides): string | null {
  const keys = Object.keys(overrides).sort()
  if (keys.length === 0) return null
  return keys.map((k) => `${k}=${overrides[k]}`).join(', ')
}
