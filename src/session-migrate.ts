/**
 * JevLoop · 把 v1 的会话日志搬进 v2 的布局
 *
 * ══════════════════════════════════════════════════════════════
 *  **这是一次性的历史包袱，做完了就该删掉这一整个文件。**
 * ══════════════════════════════════════════════════════════════
 *
 *      v1   <root>/<id>.jsonl                      扁平的，一行一轮，只有问答
 *      v2   <root>/<projectKey(cwd)>/<id>/log.jsonl  按 cwd 分项目，一行一个事件
 *
 * 为什么是**搬**而不是像 DSH 那样**拒绝**：v1 的 header 里记着 `cwd`，
 * 所以新位置是确定的，搬得动。DSH 的旧布局是「项目目录下平铺」，位置本身
 * 有歧义，所以它选择抛错 —— 那是它的取舍。
 *
 * ── 三条纪律 ────────────────────────────────────────────────────
 *
 * · **内容一字不改。** 搬的是文件本身（`rename`），不是重新序列化 ——
 *   重写一遍就有机会把一个字段写丢，而那是静默的。
 * · **搬不动就留在原地。** 读不出 `cwd` 的落到 `_no-cwd`；连读都读不了的
 *   原样留着，下次启动再试。**绝不删** —— 那是用户的对话。
 * · **说不搬了什么。** 每搬一个打一行，搬不动的报原因。
 *
 * 有 IO，所以放 L2。
 *
 * @module JevLoop/session-migrate
 */

import { mkdir, readdir, readFile, rename } from 'node:fs/promises'
import { join } from 'node:path'

import { foldSessionLog } from './session-log.ts'
import { logPath, sessionDir } from './session-path.ts'

/** 会话 id 的白名单。和 `session-store` 里那个是同一个规则，见那里的说明 */
const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

/**
 * 把 `dir` 下所有 v1 的扁平文件搬进 v2 布局。
 *
 * **幂等**：搬过的文件不在原地了，下次扫描就找不到它。失败的那些留在原地，
 * 下次再试。
 *
 * @param dir 会话根目录
 * @param note 报告往哪写。默认 `console.error` —— 和内核别处的告警同一条流
 */
export async function migrateLegacyLayout(
  dir: string,
  note: (line: string) => void = (l) => console.error(l),
): Promise<void> {
  let entries: string[]
  try {
    entries = (await readdir(dir, { withFileTypes: true }))
      .filter((d) => d.isFile() && d.name.endsWith('.jsonl'))
      .map((d) => d.name)
  } catch {
    // 吞的是「根目录还不存在」= 一个会话都没有。**没什么可搬的**，不是错误
    return
  }

  for (const name of entries) {
    const id = name.slice(0, -'.jsonl'.length)
    if (!ID_PATTERN.test(id)) continue

    const from = join(dir, name)
    try {
      // 读 header 拿 `cwd` —— 这就是「新位置是确定的」的依据
      const { header } = foldSessionLog(await readFile(from, 'utf8'))
      const cwd = header?.cwd
      const to = logPath(dir, cwd, id)
      // `sessionDir` 一次把项目目录和会话目录都建出来（`recursive`）
      await mkdir(sessionDir(dir, cwd, id), { recursive: true })
      // ★ **搬文件本身**，不重新序列化 —— 重写一遍就有机会静默丢字段
      await rename(from, to)
      note(`  · 会话 ${id} 从 v1 布局搬到 v2：${to}`)
    } catch (err) {
      // 吞的是「这一个文件搬不动」。**不删它、也不跳过整轮** —— 留在原地
      // 下次再试。搬一半比不搬糟，所以循环继续。
      note(`  ⚠ 会话 ${id} 没能搬到 v2 布局（留在原地）：${(err as Error).message}`)
    }
  }
}
