/**
 * 回答的 markdown 渲染。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * ══════════════════════════════════════════════════════════════
 *  这个文件是**补票**：`web/markdown.js` 上线时一行测试都没有，
 *  而它是整个项目里**唯一用户直接盯着看**的代码。
 * ══════════════════════════════════════════════════════════════
 *
 * 后果实测（2026-09-21，用户报的「回答是空白的」）：六处「收集连续的同类行」
 * 里有四处忘了推进游标，其中三处（引用 / 无序列表 / 有序列表）**永远不会
 * 退出** —— 数组涨到 `2^32-1`，浏览器抛 `RangeError: Invalid array length`，
 * 整个回答渲染不出来。只要回答里有一个列表就必炸。
 *
 * 为什么以前测不到：`tsconfig.json` 的 `include` 里没有 `web/`，测试文件
 * `import '../web/markdown.js'` 直接是 **TS7016**（找不到声明文件）。
 * 「测不了」和「不用测」在账面上长得一样，而它是前者。
 *
 * ── 这个测试为什么不碰浏览器 ────────────────────────────────────
 *
 * `markdown.js` 需要 `document`，所以下面有一个**最小的 DOM 桩**。
 * 桩只实现 `markdown.js` 真正用到的那几个方法 —— 它不假装自己是浏览器，
 * 也**不测样式和布局**（那要真浏览器，见 §11 的验收脚本）。
 * 这里测的是**纯函数性质**：给定一段 markdown，出来什么结构、会不会挂。
 * 而那个 bug 恰好就是纯函数性质 —— 和 DOM 实现得对不对无关。
 *
 * @module JevLoop/markdown.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { renderMarkdown } from '../web/markdown.js'

// ═══════════════════════════════════════════════════════════
// DOM 桩
// ═══════════════════════════════════════════════════════════

/** 桩节点的基类。它存在的**唯一**理由是 `el()` 里那句 `c instanceof Node`。 */
class DomNode {
  kids: DomNode[]
  constructor() {
    this.kids = []
  }
  append(...cs: DomNode[]): void {
    for (const c of cs) this.kids.push(c)
  }
}

/** 文本节点 */
class DomText extends DomNode {
  data: string
  constructor(data: string) {
    super()
    this.data = data
  }
}

/** 元素节点 */
class DomEl extends DomNode {
  tag: string
  cls: string
  attrs: Record<string, string>
  text: string | null
  constructor(tag: string) {
    super()
    this.tag = tag
    this.cls = ''
    this.attrs = {}
    this.text = null
  }
  set className(v: string) {
    this.cls = v
  }
  set textContent(v: string) {
    this.text = v
    this.kids = []
  }
  setAttribute(k: string, v: string): void {
    this.attrs[k] = v
  }
}

// `markdown.js` 在**函数体里**用 `document` / `Node`，模块顶层不碰它们，
// 所以这个赋值在 import 之后做也来得及。
Object.assign(globalThis, {
  Node: DomNode,
  document: {
    createDocumentFragment: () => new DomEl('#fragment'),
    createElement: (tag: string) => new DomEl(tag),
    createTextNode: (data: string) => new DomText(data),
  },
})

// ═══════════════════════════════════════════════════════════
// 断言用的形状
// ═══════════════════════════════════════════════════════════

/** 转义 —— 只在**序列化**这一步做。`textContent` 本身存的是不转义的原文，
    所以「是个文本节点」和「是个元素」的区别只有在压成 HTML 时才看得出来。 */
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/** 空元素：没有闭合标签，也没有子节点 */
const VOID = new Set(['hr', 'br', 'img'])

/** 把桩树压成一行 HTML，便于精确断言结构 */
function shape(n: DomNode): string {
  if (n instanceof DomText) return esc(n.data)
  const e = n as DomEl
  const cls = e.cls ? ` class="${e.cls}"` : ''
  const attrs = Object.entries(e.attrs)
    .map(([k, v]) => ` ${k}="${esc(v)}"`)
    .join('')
  const open = `<${e.tag}${cls}${attrs}>`
  if (VOID.has(e.tag)) return open
  if (e.text !== null) return `${open}${esc(e.text)}</${e.tag}>`
  return `${open}${e.kids.map(shape).join('')}</${e.tag}>`
}

/** 渲染一段 markdown，返回压平后的结构串 */
function html(text: string): string {
  const frag: DomEl = renderMarkdown(text)
  return frag.kids.map(shape).join('')
}

// ═══════════════════════════════════════════════════════════
// 终止性 —— 这几条就是那次事故
// ═══════════════════════════════════════════════════════════

test('列表吃掉整个输入也不会不返回（曾经死循环）', () => {
  // 每一类「收集连续行」的块各来一次。以前列表和引用这三条
  // `while (pred(lines[i]))` 不推进 i，这里会挂住或者抛 RangeError。
  assert.equal(html('- 甲\n- 乙\n- 丙'), '<ul class="md-ul"><li>甲</li><li>乙</li><li>丙</li></ul>')
  assert.equal(html('1. 甲\n2. 乙'), '<ol class="md-ol"><li>甲</li><li>乙</li></ol>')
  assert.equal(html('> 甲\n> 乙'), '<blockquote class="md-quote">甲\n乙</blockquote>')
})

test('块类型交替出现时每一块都收得住', () => {
  // 「收集」类循环写错时，**下一个块**是第一个被吞掉的 —— 只测单块不够。
  const out = html(['# 标题', '- 甲', '> 乙', '```', '代码', '```', '正文'].join('\n'))
  assert.equal(
    out,
    '<h1 class="md-h">标题</h1>' +
      '<ul class="md-ul"><li>甲</li></ul>' +
      '<blockquote class="md-quote">乙</blockquote>' +
      '<pre class="md-pre"><code class="md-block">代码</code></pre>' +
      '<p class="md-p">正文</p>',
  )
})

test('最容易触发的那次事故：围栏 + 列表 + 行内代码', () => {
  // 实测崩掉的那个回答的形状（它就是在 run:end 上抛的）。
  const answer = [
    '## 怎么跑',
    '',
    '```bash',
    'git clone git@github.com:zjunlp/JevLoop && cd JevLoop',
    'npm run demo',
    '```',
    '',
    '要点：',
    '',
    '- 先 `npm install`',
    '- 再 `npm run demo`',
    '',
    '| 命令 | 作用 |',
    '| --- | --- |',
    '| `npm test` | 跑测试 |',
  ].join('\n')
  const out = html(answer)
  assert.match(out, /<pre class="md-pre"><code class="md-block" data-lang="bash">/)
  assert.match(out, /<ul class="md-ul"><li>先 <code class="md-code">npm install<\/code><\/li>/)
  assert.match(out, /<table class="md-table">/)
})

// ═══════════════════════════════════════════════════════════
// 结构
// ═══════════════════════════════════════════════════════════

test('标题按级别出 h1..h6', () => {
  assert.equal(html('### 三级'), '<h3 class="md-h">三级</h3>')
  assert.equal(html('####### 七个井号'), '<p class="md-p">####### 七个井号</p>')
})

test('分隔线', () => {
  assert.equal(html('---'), '<hr class="md-hr">')
})

test('表格：表头 + 表体，缺的单元格补空', () => {
  const out = html('| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |')
  assert.equal(
    out,
    '<table class="md-table">' +
      '<thead><tr><th>A</th><th>B</th><th>C</th></tr></thead>' +
      '<tbody><tr><td>1</td><td>2</td><td></td></tr></tbody>' +
      '</table>',
  )
})

test('连续的普通行合成一段，空行分段', () => {
  assert.equal(html('甲\n乙\n\n丙'), '<p class="md-p">甲\n乙</p><p class="md-p">丙</p>')
})

test('行内：代码 / 粗 / 斜 / 链接', () => {
  assert.equal(
    html('看 `a.js` 和 **重要** 和 *斜* 和 [文档](https://x.dev/d)'),
    '<p class="md-p">看 <code class="md-code">a.js</code> 和 <strong>重要</strong>' +
      ' 和 <em>斜</em> 和 <a class="md-a" href="https://x.dev/d" target="_blank"' +
      ' rel="noreferrer noopener">文档</a></p>',
  )
})

// ═══════════════════════════════════════════════════════════
// 不可信输入
// ═══════════════════════════════════════════════════════════

test('HTML 是文本，不是节点 —— 全程没有 innerHTML', () => {
  // 这是这个文件的**安全性质**：靠 createElement + textContent 在结构上成立，
  // 不是靠一个 sanitizer 去追。所以 `<img onerror>` 只会被读出来。
  assert.equal(
    html('<img src=x onerror=alert(1)>'),
    '<p class="md-p">&lt;img src=x onerror=alert(1)&gt;</p>',
  )
})

test('javascript: 链接不生成 <a>，原样显示', () => {
  const out = html('[点我](javascript:alert(1))')
  assert.ok(!out.includes('<a '), `不该生成链接：${out}`)
  assert.equal(out, '<p class="md-p">[点我](javascript:alert(1))</p>')
})

test('http/https/mailto/相对路径放行', () => {
  for (const href of ['https://x.dev', 'http://x.dev', 'mailto:a@b.c', './notes.md']) {
    assert.match(html(`[x](${href})`), /<a class="md-a" /, href)
  }
})

test('不支持的语法原样显示，不丢内容', () => {
  // 「宁可看到源码，也不要看到一段被悄悄吃掉的文本」—— 文件头第 24 行。
  assert.equal(html('![图](x.png)'), '<p class="md-p">![图](x.png)</p>')
})

test('空输入不产生任何块', () => {
  assert.equal(html(''), '')
  assert.equal(html('\n\n\n'), '')
})

test('没有收尾围栏的代码块把剩下的都吃掉，不吞掉自己', () => {
  assert.equal(
    html('```\n甲\n乙'),
    '<pre class="md-pre"><code class="md-block">甲\n乙</code></pre>',
  )
})
