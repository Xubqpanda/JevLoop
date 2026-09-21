/**
 * 代码规范的机械检查。
 *
 * JevLoop/AGENTS.md §1/§3 写了一套格式规则，但在写这个脚本之前**没有任何东西执行它** ——
 * 规范是在代码写完之后补的，于是每轮审计都要手工重查一遍，而新代码可以继续偏离。
 *
 * 为什么不是 oxlint / eslint：这个仓库的对外承诺是零运行时依赖（见 README）。
 * 这里要检查的只是本仓库自己那几条规则，而 `typescript` 已经是 devDependency
 * （`npm run typecheck` 需要它），所以直接用它的**解析器**，不引入新的依赖树。
 *
 * **不要自己写词法器。** 试过两版都失败，而且失败方式很隐蔽：
 *
 *   1. 逐字符扫引号和注释：遇到「字符串里含注释结束符」（glob 模式 `src/**` 那种）
 *      以及「注释里含反引号」就会丢状态，之后整段文件被误判。
 *   2. `ts.createScanner`：在模板字符串相邻处把 token 范围整体错位，
 *      于是一大段代码被当成字符串刷掉 —— **漏报**。
 *
 * 漏报比误报更危险：没人会去查一个显示"通过"的检查。
 * `ts.createSourceFile` 没有这些问题，而且注释、字符串、模板都由它处理，这里一行都不用管。
 *
 * **检查什么**（全部来自 §1/§3）：
 *   · 行尾分号
 *   · 双引号字符串
 *   · 制表符 / CRLF / 文件末尾恰好一个换行
 *   · 相对导入带 `.ts` 扩展名
 *   · 模块头部有 `@module JevLoop/<文件名>`
 *
 * **不检查什么**（规则本身不精确，硬查会误伤）：
 *   · 缩进是不是恰好 2 空格 —— 续行、模板字符串、对齐注释都会让逐行判定失真
 *   · 注释措辞、命名、类型设计 —— 那些靠 review，不靠脚本
 *
 * @module JevLoop/scripts/check
  *
 * @module JevLoop/check
 */

import { readFileSync, globSync } from 'node:fs'
import { basename } from 'node:path'
import ts from 'typescript'

interface Violation {
  file: string
  line: number
  rule: string
  detail: string
}

const ROOTS = ['src/**/*.ts', 'examples/**/*.ts', 'tests/**/*.ts', 'scripts/**/*.ts']

/**
 * 从 AST 找出两类词法违规。
 *
 * **行尾分号**：任何节点只要它结束位置的前一个字符是 `;`，那个分号就是语句终止符。
 * `for (;;)` 里的分号在节点**内部**，不在 `end` 位置，所以不会误报。
 * 父子节点会看到同一个分号，所以最后按 (行, 规则) 去重。
 *
 * **双引号字符串**：`StringLiteral` 节点的起始字符是不是 `"`。
 *
 * @param sf 已解析的源文件
 * @returns 违规列表，行号 1-based
 */
function lexicalViolations(sf: ts.SourceFile): { line: number; rule: string; detail: string }[] {
  const raw = sf.getFullText()
  const out: { line: number; rule: string; detail: string }[] = []
  const lineOf = (pos: number) => sf.getLineAndCharacterOfPosition(pos).line + 1

  const walk = (node: ts.Node): void => {
    const end = node.getEnd()
    // ★ 只看**语句**。类型字面量和接口成员也用 `;` 分隔（`{ a: string; b: number }`），
    //   那不是语句终止符 —— 不加这一层会误报一堆类型定义（试过）。
    if (end > 0 && raw[end - 1] === ';' && ts.isStatement(node)) {
      out.push({ line: lineOf(end - 1), rule: 'semicolon', detail: '行尾分号' })
    }
    if (ts.isStringLiteral(node) && raw[node.getStart(sf)] === '"') {
      out.push({ line: lineOf(node.getStart(sf)), rule: 'quote', detail: '双引号字符串（规范要求单引号）' })
    }
    ts.forEachChild(node, walk)
  }
  walk(sf)

  const seen = new Set<string>()
  return out.filter((v) => {
    const key = `${v.line}:${v.rule}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

/** 检查一个文件，返回它违反的规则 */
function checkFile(file: string): Violation[] {
  const raw = readFileSync(file, 'utf8')
  const found: Violation[] = []
  const add = (line: number, rule: string, detail: string) => found.push({ file, line, rule, detail })

  if (raw.includes('\r\n')) add(0, 'crlf', '含 CRLF 行尾（规范要求 LF）')
  if (!raw.endsWith('\n')) add(0, 'eof-newline', '文件末尾没有换行')
  else if (raw.endsWith('\n\n')) add(0, 'eof-newline', '文件末尾多于一个换行')

  const sf = ts.createSourceFile(file, raw, ts.ScriptTarget.Latest, false)
  for (const v of lexicalViolations(sf)) add(v.line, v.rule, v.detail)

  raw.split('\n').forEach((line, i) => {
    if (line.includes('\t')) add(i + 1, 'tab', '含制表符（规范要求 2 空格缩进）')
  })

  // 相对导入必须带 .ts —— 这条词法层面查不出（说明符本身合法），要看 AST
  for (const stmt of sf.statements) {
    if (!ts.isImportDeclaration(stmt) && !ts.isExportDeclaration(stmt)) continue
    const spec = stmt.moduleSpecifier
    if (!spec || !ts.isStringLiteral(spec)) continue
    const path = spec.text
    if (!path.startsWith('./') && !path.startsWith('../')) continue
    if (!path.endsWith('.ts')) {
      add(sf.getLineAndCharacterOfPosition(spec.getStart(sf)).line + 1, 'import-ext', `相对导入缺少 .ts 扩展名：${path}`)
    }
  }

  const stem = basename(file, '.ts')
  if (!raw.includes(`@module JevLoop/${stem}`)) {
    add(0, 'module-tag', `头部缺少 \`@module JevLoop/${stem}\``)
  }

  return found
}

const files = ROOTS.flatMap((pattern) => [...globSync(pattern)]).sort()
if (files.length === 0) {
  console.error('没有匹配到任何文件 —— glob 模式写错了？')
  process.exit(1)
}

const violations = files.flatMap(checkFile)

if (violations.length === 0) {
  console.log(`check: ${files.length} 个文件，全部通过`)
  process.exit(0)
}

const byRule = new Map<string, Violation[]>()
for (const v of violations) byRule.set(v.rule, [...(byRule.get(v.rule) ?? []), v])

for (const [rule, list] of [...byRule.entries()].sort()) {
  console.error(`\n✖ ${rule} —— ${list.length} 处`)
  for (const v of list) console.error(`    ${v.file}${v.line ? `:${v.line}` : ''}  ${v.detail}`)
}
console.error(`\n共 ${violations.length} 处违规。规则见 AGENTS.md §1/§3。`)
process.exit(1)
