/**
 * nanojev · 工具
 *
 * 工具是唯一产生真实副作用的地方，所以这里有两道约束：
 *   1. **路径锁死在工作目录内** —— 任何 `..` 逃逸都被拒绝
 *   2. 每个工具声明一个**静态**风险基线，作为判定节点的参考
 *
 * 「这个调用到底危不危险」由 loop.gradeRisk 判定，不由工具自己决定 ——
 * 静态基线只是给判定模型看的提示。
 */

import { readFile, writeFile, readdir, mkdir, stat } from 'node:fs/promises'
import { resolve, relative, dirname, sep } from 'node:path'

export interface Tool {
  name: string
  description: string
  /** 静态风险基线 0..3。判定节点会参考它，但最终判定由模型做 */
  baseRisk: number
  run(input: string, cwd: string): Promise<string>
}

/** 把用户给的路径解析到 cwd 内，逃逸就抛 */
function safePath(cwd: string, p: string): string {
  const full = resolve(cwd, p)
  const rel = relative(cwd, full)
  if (rel.startsWith('..') || (rel !== '' && rel.startsWith(sep))) {
    throw new Error(`路径逃出工作目录：${p}`)
  }
  return full
}

export const TOOLS: Record<string, Tool> = {
  list_dir: {
    name: 'list_dir',
    description: '列出工作目录里的文件（不含子目录内容）',
    baseRisk: 0,
    async run(_input: string, cwd: string): Promise<string> {
      const entries = await readdir(cwd, { withFileTypes: true })
      const out = entries
        .filter((e) => !e.name.startsWith('.'))
        .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
        .sort()
      return out.length ? out.join('\n') : '(目录为空)'
    },
  },

  read_file: {
    name: 'read_file',
    description: '读取一个文件的内容。输入是相对工作目录的路径',
    baseRisk: 0,
    async run(input: string, cwd: string): Promise<string> {
      const path = safePath(cwd, input.trim())
      const info = await stat(path)
      if (info.isDirectory()) return `(这是一个目录，请用 list_dir)`
      const text = await readFile(path, 'utf8')
      return text.length > 4000 ? text.slice(0, 4000) + `\n…[+${text.length - 4000} chars]` : text
    },
  },

  write_file: {
    name: 'write_file',
    description: '写入或覆盖一个文件。输入格式：`路径\n内容`（第一行是路径，其余是内容）',
    baseRisk: 1,
    async run(input: string, cwd: string): Promise<string> {
      const nl = input.indexOf('\n')
      if (nl < 0) throw new Error('write_file 需要两行输入：第一行路径，其余内容')
      const path = safePath(cwd, input.slice(0, nl).trim())
      const content = input.slice(nl + 1)
      await mkdir(dirname(path), { recursive: true })
      await writeFile(path, content, 'utf8')
      return `已写入 ${relative(cwd, path)}（${content.length} 字符）`
    },
  },

  done: {
    name: 'done',
    description: '任务已完成，不需要再调用任何工具',
    baseRisk: 0,
    async run(): Promise<string> {
      return '任务标记为完成'
    },
  },
}

export const toolNames = (): string[] => Object.keys(TOOLS)

/**
 * 执行一次工具调用。
 *
 * 判定节点已经决定「放行 / 需要授权」了，这里只负责执行 ——
 * 但**执行结果永远要回传**，因为判定「成功了吗」需要看到它。
 */
export async function callTool(name: string, input: string, cwd: string): Promise<string> {
  const tool = TOOLS[name]
  if (!tool) return `错误：没有这个工具 '${name}'`
  try {
    return await tool.run(input, cwd)
  } catch (err) {
    return `错误：${(err as Error).message}`
  }
}
