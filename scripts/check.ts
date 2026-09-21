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
 *   · `src/` 内的 import 方向符合分层（见 DESIGN-layers-2026-09-21.md）
 *   · `src/index.ts` 公开导出的**值**有 JSDoc（类型/接口不查，见下）
 *   · 公开值的签名引用到的本仓库类型也在导出面上（契约不能只导出一半）
 *
 * **不检查什么**（规则本身不精确，硬查会误伤）：
 *   · 缩进是不是恰好 2 空格 —— 续行、模板字符串、对齐注释都会让逐行判定失真
 *   · 注释措辞、命名、类型设计 —— 那些靠 review，不靠脚本
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
  // ★ 只在**文件头部那个文档块**里找，不是全文件子串匹配。
  //
  //   以前是 `raw.includes(...)`：实测把标签从头部文档块挪到文件末尾一条普通
  //   `//` 注释里，检查照样全绿 —— 而这条规则的文档（本文件头部、AGENTS.md §3）
  //   写的是「**模块头部**有 `@module …`」。§7 不允许检查比规范松：
  //   一个比自己的文档弱的检查会让人以为规范已经被守住了。
  //   `#!` 那一行要放行：可执行脚本的 shebang 必须是文件第一行，
  //   文档块只能跟在它后面 —— 那是正确写法，不是违规。
  //   （第一版收紧时没放行，于是 examples/demo.ts 被误报了。）
  const header = /^(?:#![^\n]*\n)?\s*\/\*\*([\s\S]*?)\*\//.exec(raw)?.[1]
  if (!header || !header.includes(`@module JevLoop/${stem}`)) {
    add(0, 'module-tag', `头部文档块里缺少 \`@module JevLoop/${stem}\``)
  }

  return found
}

// ═══════════════════════════════════════════════════════════
// 分层方向
//
// 完整设计见 docs/DESIGN-layers-2026-09-21.md。规则一句话：
// **一个文件只能 import 编号严格更小的层。**
//
// 为什么需要机器检查：第七轮 C1（`headBudget` 声明未用）与第八轮 P1
// （`policy_no_catch_all` 分支在移植时消失）是同一类 bug 的两个实例 ——
// 边界靠人记就会漏。这条规则把「哪层能用哪层」变成退不掉的检查。
// ═══════════════════════════════════════════════════════════

/**
 * `src/*.ts` → 层号。**新增文件必须同时加进这张表**，否则算违规。
 *
 * 分层的依据是「知道什么」，不是「放在哪个目录」：
 *
 *   L0 词汇    什么都不知道（不依赖任何东西）
 *   L1 机制    只知道词汇；纯函数，没有 IO，没有领域知识
 *   L2 接缝    定义 / 提供者 / 消费三角；有 IO，不知道 agent
 *   L3 编译    ctx → 帧（frame），Markdown → 问题与策略（decisiondoc）
 *   L4 节点    JevLoop 的产品主张：七个判定节点
 *   L5 循环    驱动器
 *   L6 组合    具体后端的选择与拼装
 */
const LAYER: Record<string, number> = {
  // L0 —— 词汇。彼此可以互相引用（词汇天然互相指涉）
  vocab: 0,
  'vocab-decision': 0,
  'vocab-records': 0,
  util: 0,
  // L1 —— 机制。纯函数，不许有 IO
  policy: 1,
  budget: 1,
  meter: 1,
  events: 1,
  // `context` 放 L1 而不是 L3：它与 `frame.ts`（L3）都做「裁剪/编译」，
  // 但 `frame.ts` 知道 `AgentCtx`，而 `context.ts` 只收 `readonly string[]` ——
  // 它比 L3 更不知道上下文。而且它**零 import**，放哪层都不会产生依赖问题。
  context: 1,
  // L2 —— 接缝
  'seam-provider': 2,
  provider: 2,
  decide: 2,
  llm: 2,
  tools: 2,
  // L3 —— 编译器
  frame: 3,
  decisiondoc: 3,
  // L4 —— 判定节点
  decisions: 4,
  // L5 —— 循环
  agent: 5,
  // L6 —— 组合
  backends: 6,
  env: 6,
}

/** 层号 → 一句话，报错时要说清两边各是什么 */
const LAYER_NAME = ['L0 词汇', 'L1 机制', 'L2 接缝', 'L3 编译', 'L4 节点', 'L5 循环', 'L6 组合']

/** 门面，不受层约束 */
const FACADE = 'index'

/**
 * L2 内部的合法方向：提供者与消费者 → **定义角**。
 *
 * 定义角是 `seam-provider.ts`。反向（定义 import 某个具体提供者）永远违规 ——
 * 那会让「换一个后端」重新需要改内核，也就是能力缝失效。
 */
const SEAM_DEFINITION = 'seam-provider'

/**
 * 检查 `src/` 内所有相对 import 的方向。
 *
 * @param dir `src` 目录
 * @returns 违规列表；`file` 为相对路径
 */
function layerViolations(dir: string): Violation[] {
  const out: Violation[] = []
  const stems = new Set<string>()

  for (const f of globSync(`${dir}/*.ts`)) {
    const stem = basename(f, '.ts')
    stems.add(stem)
    if (stem === FACADE) continue

    const me = LAYER[stem]
    if (me === undefined) {
      out.push({
        file: f,
        line: 0,
        rule: 'layers',
        detail: `新文件没有登记层号 —— 请把它加进 scripts/check.ts 的 LAYER（见 DESIGN-layers-2026-09-21.md）`,
      })
      continue
    }

    const sf = ts.createSourceFile(f, readFileSync(f, 'utf8'), ts.ScriptTarget.Latest, false)
    for (const stmt of sf.statements) {
      if (!ts.isImportDeclaration(stmt) && !ts.isExportDeclaration(stmt)) continue
      const spec = stmt.moduleSpecifier
      if (!spec || !ts.isStringLiteral(spec)) continue
      const m = /^\.\/([a-z0-9-]+)\.ts$/.exec(spec.text)
      if (!m) continue
      const target = m[1]
      if (target === FACADE || target === stem) continue

      const their = LAYER[target]
      if (their === undefined) {
        out.push({ file: f, line: 0, rule: 'layers', detail: `依赖了未登记层号的文件 ${target}.ts` })
        continue
      }
      if (their < me) continue

      // 同层：只有两种合法情形
      if (their === me) {
        if (me === 0) continue // 词汇互相指涉
        if (me === 2 && target === SEAM_DEFINITION) continue // 提供者/消费者 → 定义角
      }

      const where = sf.getLineAndCharacterOfPosition(spec.getStart(sf)).line + 1
      out.push({
        file: f,
        line: where,
        rule: 'layers',
        detail: `${LAYER_NAME[me]} 依赖 ${LAYER_NAME[their]}（${target}.ts）—— 依赖只能指向编号更小的层`
          + (their === me ? `；同层只允许 L0 内部、以及 L2 指向 ${SEAM_DEFINITION}.ts` : ''),
      })
    }
  }

  return out
}

// ═══════════════════════════════════════════════════════════
// 公开面的 JSDoc
//
// 第九轮 R7 报的是「JSDoc 覆盖率 38%，且**对外主入口正好是缺的那几个**」。
// 执行把口径**收窄**了，理由必须写清楚：
//
//   原始的 126 个导出里混着大量内部类型别名（`export type BlockKind = …`），
//   按那个口径补文档只会产出填充物 —— 而 §7 禁止「为了通过检查而降低检查标准」，
//   附录也写明「文档讲**为什么**和**契约**」。
//
//   收窄后的口径是「**`src/index.ts` 导出的值**必须有 JSDoc」：
//   公开的值是使用者唯一会去查的东西，也是 `.d.ts` 里唯一会显示成 IDE 提示的东西
//   （`//` 横幅注释不进 `.d.ts`，所以不算）。
//
//   **类型与接口不查** —— 它们的契约由字段自己说明，逼着写只会得到同义反复。
// ═══════════════════════════════════════════════════════════

/** 取一个语句声明的名字。不是带名字的声明（import / export / if …）就返回 undefined */
function declaredName(st: ts.Statement): string | undefined {
  if (
    ts.isFunctionDeclaration(st) ||
    ts.isClassDeclaration(st) ||
    ts.isInterfaceDeclaration(st) ||
    ts.isTypeAliasDeclaration(st) ||
    ts.isEnumDeclaration(st)
  ) {
    return st.name?.text
  }
  if (ts.isVariableStatement(st)) {
    const d = st.declarationList.declarations[0]
    return d && ts.isIdentifier(d.name) ? d.name.text : undefined
  }
  return undefined
}

/** 类型/接口 —— 只有类型空间里有意义，运行时不存在 */
const isTypeOnly = (st: ts.Statement): boolean =>
  ts.isInterfaceDeclaration(st) || ts.isTypeAliasDeclaration(st)

/**
 * 检查 `index.ts` 公开导出的**值**是否都有前置 JSDoc。
 *
 * 判定方法是「声明前的 trivia 是否以一个块注释的结束符收尾」——
 * 也就是它前面紧挨着一个块注释。
 * 宽松但有意义：它拦的是「公开的东西一句话说明都没有」，不是评注的质量。
 *
 * @param dir `src` 目录
 */
function publicSurfaceViolations(dir: string): Violation[] {
  const documented = new Set<string>()
  const valueNames = new Set<string>()

  for (const f of globSync(`${dir}/*.ts`)) {
    const raw = readFileSync(f, 'utf8')
    const sf = ts.createSourceFile(f, raw, ts.ScriptTarget.Latest, false)
    for (const st of sf.statements) {
      const name = declaredName(st)
      if (!name) continue
      if (!isTypeOnly(st)) valueNames.add(name)
      if (raw.slice(st.getFullStart(), st.getStart(sf)).trimEnd().endsWith('*/')) documented.add(name)
    }
  }

  const facade = `${dir}/${FACADE}.ts`
  const raw = readFileSync(facade, 'utf8')
  const sf = ts.createSourceFile(facade, raw, ts.ScriptTarget.Latest, false)
  const out: Violation[] = []

  for (const st of sf.statements) {
    if (!ts.isExportDeclaration(st) || !st.exportClause || !ts.isNamedExports(st.exportClause)) continue
    for (const el of st.exportClause.elements) {
      // `export { local as public }` —— 要查的是**本地**那个名字的声明
      const name = (el.propertyName ?? el.name).text
      if (!valueNames.has(name) || documented.has(name)) continue
      out.push({
        file: facade,
        line: sf.getLineAndCharacterOfPosition(el.getStart(sf)).line + 1,
        rule: 'public-jsdoc',
        detail: `公开导出的值 '${name}' 没有 JSDoc —— 它是使用者唯一会查的东西，也是 .d.ts 里唯一会显示成提示的东西`,
      })
    }
  }
  return out
}

// ═══════════════════════════════════════════════════════════
// 公开面的**类型**：契约不能只导出一半
//
// 第十六轮 V1：`loadEnv` 导出了，它的返回类型 `LoadEnvResult` 没有 ——
// 使用者调用得了、解构得了，却**命名不了这个类型**（写一个接收它的辅助函数、
// 或在消费 `.d.ts` 的项目里声明一个变量，都没有名字可用）。
//
// 为什么上面那条 `public-jsdoc` 拦不住：它**明确只查值**（类型/接口不查）。
// 所以「公开函数的返回类型必须也在公开面上」这条约束，
// 既不在规范里、也不在检查里 —— 它靠人记得，而人这次没记得。
//
// 判据是**文本级**的：把签名里出现的标识符与本仓库声明的类型名取交集。
// 宽松（不认识泛型约束、映射类型之类），但足以挡住"导出函数忘了导出它的类型"。
// ═══════════════════════════════════════════════════════════

/** 从一段类型文本里取出所有标识符。必须显式传 `sf` —— 节点没有 parent，`getText()` 找不到源码 */
const identifiersIn = (t: ts.TypeNode, sf: ts.SourceFile): string[] =>
  [...t.getText(sf).matchAll(/[A-Za-z_$][\w$]*/g)].map((m) => m[0]!)

/**
 * `index.ts` 导出的**值**，其签名里引用到的本仓库类型，也必须在导出面上。
 *
 * @param dir `src` 目录
 */
function publicTypeSurfaceViolations(dir: string): Violation[] {
  const typeNames = new Set<string>()
  const referenced = new Map<string, Set<string>>()

  for (const f of globSync(`${dir}/*.ts`)) {
    const sf = ts.createSourceFile(f, readFileSync(f, 'utf8'), ts.ScriptTarget.Latest, false)

    for (const st of sf.statements) {
      if (ts.isInterfaceDeclaration(st) || ts.isTypeAliasDeclaration(st)) {
        if (st.name?.text) typeNames.add(st.name.text)
        continue
      }

      // 值：`function f(...): T` 与 `const f = (...): T => …`
      const sigs: (ts.TypeNode | undefined)[] = []
      let name: string | undefined
      if (ts.isFunctionDeclaration(st) && st.name) {
        name = st.name.text
        sigs.push(st.type, ...st.parameters.map((p) => p.type))
      } else if (ts.isVariableStatement(st)) {
        const d = st.declarationList.declarations[0]
        if (d && ts.isIdentifier(d.name)) {
          name = d.name.text
          sigs.push(d.type)
          const init = d.initializer
          if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
            sigs.push(init.type, ...init.parameters.map((p) => p.type))
          }
        }
      }
      if (!name) continue

      const set = referenced.get(name) ?? new Set<string>()
      for (const t of sigs) if (t) for (const id of identifiersIn(t, sf)) set.add(id)
      if (set.size) referenced.set(name, set)
    }
  }

  const facade = `${dir}/${FACADE}.ts`
  const sf = ts.createSourceFile(facade, readFileSync(facade, 'utf8'), ts.ScriptTarget.Latest, false)
  const exportedValues = new Set<string>()
  const exportedTypes = new Set<string>()

  for (const st of sf.statements) {
    if (!ts.isExportDeclaration(st) || !st.exportClause || !ts.isNamedExports(st.exportClause)) continue
    for (const el of st.exportClause.elements) {
      const name = (el.propertyName ?? el.name).text
      if (st.isTypeOnly || el.isTypeOnly) exportedTypes.add(name)
      else exportedValues.add(name)
    }
  }

  const out: Violation[] = []
  for (const [value, types] of referenced) {
    if (!exportedValues.has(value)) continue
    for (const t of [...types].sort()) {
      if (!typeNames.has(t) || exportedTypes.has(t)) continue
      out.push({
        file: facade,
        line: 0,
        rule: 'public-types',
        detail: `公开导出的值 '${value}' 的签名引用了 '${t}'，但 '${t}' 不在导出面上 —— 使用者命名不了这个类型`,
      })
    }
  }
  return out
}

const files = ROOTS.flatMap((pattern) => [...globSync(pattern)]).sort()
if (files.length === 0) {
  console.error('没有匹配到任何文件 —— glob 模式写错了？')
  process.exit(1)
}

const violations = [
  ...files.flatMap(checkFile),
  ...layerViolations('src'),
  ...publicSurfaceViolations('src'),
  ...publicTypeSurfaceViolations('src'),
]

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
