/**
 * JevLoop · 命令行参数层
 *
 * 从 `cli.ts` 切出来，接缝是**输入输出的形状变了**：这一半把 `argv` 变成一个
 * 决定（哪个子命令、什么位置参数、哪些选项），另一半把那个决定变成行为。
 * 这不是按行数拆的 —— 两半的依赖面可测量地不同：这里零 import、无 IO，
 * 所以它落在 L1 并且可以单独测；`cli.ts` 要起进程、要读盘，落在 L6。
 *
 * 取值的选项用**白名单**，不是「下一个 token 不以 `--` 开头就吃掉」——
 * 后者会让 `jevloop run --jev list the files` 把 `list` 当成 `--jev` 的值，
 * 于是任务名少一个词而没有任何东西报错。
 *
 * @module JevLoop/cli-args
 */

/** 用法说明。`jevloop --help` 和「命令打错」两条路径共用，避免分叉。 */
export const USAGE = `
JevLoop — the agent loop where decisions don't cost a model call

  jevloop run "<task>"      run the loop once and print the accounting
  jevloop serve             start the UI
  jevloop spec [file]       print what DECISION.md compiles to

run options
  --cwd <dir>               working directory (default: the current one)
  --jev | --laya            force a decision backend
  --max-steps <n>           step limit (default 8)
  --strict                  throw on a budget error before sending the request

serve options
  --cwd <dir>               working directory (default: a temporary demo dir)
  --port <n>                default 7799
  --host <h>                default 127.0.0.1; this server has no auth
`.trim()

/** 取值的选项。其余一律当开关（存在即真）。 */
const VALUE_OPTIONS = new Set(['cwd', 'max-steps', 'port', 'host'])

/** `argv` 解析之后的样子。 */
export interface Parsed {
  /** 子命令名，没给就是空串 */
  command: string
  /** 位置参数，按出现顺序。`run` 的任务名由调用方拼回来（用户可能不加引号） */
  positional: string[]
  /** 选项。开关型选项的值是空串 —— 判断用 `has`，不用值的真假 */
  options: Map<string, string>
}

/**
 * 把 `argv` 切成子命令、位置参数和选项。
 *
 * 支持 `--key=value` 与 `--key value` 两种写法；只有 {@link VALUE_OPTIONS}
 * 里的键会吃掉下一个 token。
 *
 * @param argv 不含 `node` 和脚本路径的参数列表（即 `process.argv.slice(2)`）
 * @throws 取值的选项后面没有值时抛 —— 静默当成开关会让命令少一个参数而照常跑
 */
export function parseArgv(argv: string[]): Parsed {
  const [command = '', ...rest] = argv
  const positional: string[] = []
  const options = new Map<string, string>()

  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i]!
    if (!arg.startsWith('--')) {
      positional.push(arg)
      continue
    }
    const eq = arg.indexOf('=')
    if (eq > 0) {
      options.set(arg.slice(2, eq), arg.slice(eq + 1))
      continue
    }
    const name = arg.slice(2)
    if (VALUE_OPTIONS.has(name)) {
      const value = rest[++i]
      if (value === undefined) throw new Error(`--${name} needs a value`)
      options.set(name, value)
      continue
    }
    options.set(name, '')
  }

  return { command, positional, options }
}
