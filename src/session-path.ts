/**
 * JevLoop · 会话在磁盘上的**位置**
 *
 * ══════════════════════════════════════════════════════════════
 *  布局仿照 DSH：**按工作目录分项目，一个会话一个目录。**
 * ══════════════════════════════════════════════════════════════
 *
 *     <root>/<projectKey(cwd)>/<encodeSegment(id)>/log.jsonl
 *            └ 项目：一个 cwd 一个目录 ┘  └ 会话：一个 id 一个目录 ┘
 *
 * `cwd` 是 `undefined` 时落 `_no-cwd`（DSH 同名）。
 *
 *     /home/x/proj  →  --home-x-proj--
 *     s3f9a2        →  s3f9a2
 *     a/b           →  a~002Fb
 *
 * ── 为什么把 cwd 放进**路径**里，而不是只记在 header ────────────
 *
 * 我第一版只在 header 里记 `cwd`，然后拿它去和「工作区登记表」**匹配**，
 * 算出这个会话属于哪个工作区。那有三个毛病，而路径分组一个都没有：
 *
 * · 匹配要读**两**张表才能回答一个问题；
 * · 登记表被清空之后，所有会话就"不属于任何工作区"了 —— 而它们的目录
 *   明明还在那儿；
 * · 列一个项目的会话要遍历**所有**会话，读每一个的 header。
 *
 * 放进路径之后，「属于哪个项目」是**结构**，不是一次查询的结果。
 *
 * ── 编码为什么是白名单 + 转义，不是简单替换 ────────────────────
 *
 * 会话 id 和 cwd 都是**外部来的**（query 参数、启动参数），而它们会变成
 * 路径段。DSH 的 `encodeSegment` 只放行 `[A-Za-z0-9._-]`，其余一律
 * `~XXXX`（四位十六进制码元），并把 `.` / `..` 单独转义。
 *
 * 这比「把 `/` 换成 `-`」强的地方在于**它是可逆的**：`a/b` 和 `a-b`
 * 不会撞成同一个目录。`projectKey` 是可读性优先的**有损**编码（分隔符
 * 统一成 `-`，超长截断），因为它是给人看的项目名；两者用途不同，
 * 所以规则也不同 —— 这一点 DSH 的注释里写明了。
 *
 * 零 IO、零依赖，所以放 L0。
 *
 * @module JevLoop/session-path
 */

import { join } from 'node:path'

/** 会话日志的文件名。DSH 用 `vN.jsonl` 表示不可变的格式世代；我们只有一份，所以名字固定 */
export const LOG_FILENAME = 'log.jsonl'

/** 没有 `cwd` 的会话落在这一层。名字抄 DSH */
export const NO_CWD_DIR = '_no-cwd'

/**
 * 把一个字符串编成**单个安全的路径段**。
 *
 * 放行 `[A-Za-z0-9._-]`，其余每个码元写成 `~XXXX`。`.` 和 `..` 单独处理 ——
 * 它们是路径段里唯一有特殊含义的两个值。
 *
 * **可逆**：不同输入不会编成同一个输出（`~` 本身也被转义）。
 */
export function encodeSegment(raw: string): string {
  if (raw.length === 0) throw new Error('路径段不能是空字符串')
  if (raw === '.') return '~002E'
  if (raw === '..') return '~002E~002E'
  let out = ''
  for (let i = 0; i < raw.length; i++) {
    const code = raw.charCodeAt(i)
    const ch = String.fromCharCode(code)
    out += ch !== '~' && /^[A-Za-z0-9._-]$/.test(ch) ? ch : `~${code.toString(16).toUpperCase().padStart(4, '0')}`
  }
  return out
}

/** 项目目录名的长度上限（文件系统单个组件通常 255 字节，留一点余量） */
const PROJECT_KEY_MAX = 251

/**
 * 把 cwd 变成一个**人能看懂**的项目目录名。
 *
 * `/home/x/proj` → `--home-x-proj--`
 *
 * ⚠️ **这个编码是有损的**（和 `encodeSegment` 不同）：分隔符统一压成 `-`，
 * 连续的压成一个，超长截断。所以 `/a/b` 和 `/a-b` 会落到同一个目录 ——
 * 那一层里混着两个项目的会话。
 *
 * 这是**有意的取舍**，DSH 的注释里也写明了：这一层是给人翻目录看的，
 * 可读性优先。真正区分身份的是每个会话 header 里的 `cwd`（原值），
 * 所以混在一起也不会把会话认错。
 */
export function projectKey(cwd: string): string {
  if (cwd.length === 0) throw new Error('项目路径不能是空字符串')
  let readable = ''
  let separators = false
  for (let i = 0; i < cwd.length; i++) {
    const code = cwd.charCodeAt(i)
    const ch = String.fromCharCode(code)
    if (ch === '/' || ch === '\\' || ch === ':') {
      // 连续的分隔符压成一个 `-`（`//` 不该变成 `--`）
      if (!separators) readable += '-'
      separators = true
    } else if (ch !== '~' && /^[A-Za-z0-9._-]$/.test(ch)) {
      readable += ch
      separators = false
    } else {
      readable += `~${code.toString(16).toUpperCase().padStart(4, '0')}`
      separators = false
    }
  }
  const slug = readable.replace(/^-+/, '') || 'root'
  return `--${slug.slice(0, PROJECT_KEY_MAX)}--`
}

/** 一个 cwd 对应的项目目录 */
export function projectDir(root: string, cwd: string | undefined): string {
  return join(root, cwd === undefined ? NO_CWD_DIR : projectKey(cwd))
}

/** 一个会话独占的目录（将来放会话自己的产物也用这里） */
export function sessionDir(root: string, cwd: string | undefined, id: string): string {
  return join(projectDir(root, cwd), encodeSegment(id))
}

/** 一个会话的日志文件 */
export function logPath(root: string, cwd: string | undefined, id: string): string {
  return join(sessionDir(root, cwd, id), LOG_FILENAME)
}

/**
 * 把 `encodeSegment` 编出来的段还原。
 *
 * 认不出的 `~XXXX` 原样留着 —— 一个手改坏了的目录名不该让整个列表崩，
 * 顶多那个 id 看起来怪。
 */
export function decodeSegment(encoded: string): string {
  let out = ''
  for (let i = 0; i < encoded.length; i++) {
    if (encoded[i] === '~' && /^[0-9A-Fa-f]{4}$/.test(encoded.slice(i + 1, i + 5))) {
      out += String.fromCharCode(parseInt(encoded.slice(i + 1, i + 5), 16))
      i += 4
    } else {
      out += encoded[i]
    }
  }
  return out
}
