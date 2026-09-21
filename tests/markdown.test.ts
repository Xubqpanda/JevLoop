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

import { renderMarkdown, unstableFrom, createStreamRenderer } from '../web/markdown.js'

// ═══════════════════════════════════════════════════════════
// DOM 桩
// ═══════════════════════════════════════════════════════════

/** 桩节点的基类。它存在的**唯一**理由是 `el()` 里那句 `c instanceof Node`。 */
class DomNode {
  kids: DomNode[]
  parent: DomNode | null
  text: string | null
  constructor() {
    this.kids = []
    this.parent = null
    this.text = null
  }
  append(...cs: DomNode[]): void {
    for (const c of cs) {
      // 真实 DOM 里 append 一个已有的子节点是**移动**，不是复制
      if (c.parent) c.parent.drop(c)
      c.parent = this
      this.kids.push(c)
    }
  }
  drop(c: DomNode): void {
    const k = this.kids.indexOf(c)
    if (k >= 0) this.kids.splice(k, 1)
    c.parent = null
  }
  remove(): void {
    if (this.parent) this.parent.drop(this)
    else this.parent = null
  }
  replaceChildren(...cs: DomNode[]): void {
    for (const c of this.kids) c.parent = null
    this.kids = []
    this.text = null
    this.append(...cs)
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
  constructor(tag: string) {
    super()
    this.tag = tag
    this.cls = ''
    this.attrs = {}
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

test('中文紧挨着的粗体和斜体 —— 答案大多是中文，这是最常见的形状', () => {
  // ★ DSH 为此专门写了一个 `cjkFriendlyStrong` 语法扩展（CommonMark 的
  //   flanking 规则按空格判词边界，而中文没有空格）。这边**不需要**它：
  //   这个渲染器压根不实现完整的 flanking 规则。所以这条测试是**守着
  //   那个简化**的 —— 哪天有人把规则补全，中文的粗体会当场失效。
  assert.equal(html('这是**粗体**文字'), '<p class="md-p">这是<strong>粗体</strong>文字</p>')
  assert.equal(html('中文**粗体**。句号'), '<p class="md-p">中文<strong>粗体</strong>。句号</p>')
  assert.equal(html('**开头就粗**后面'), '<p class="md-p"><strong>开头就粗</strong>后面</p>')
  assert.equal(html('中文*斜体*中文'), '<p class="md-p">中文<em>斜体</em>中文</p>')
})

test('标记内侧有空格就不算强调 —— `2 * 3 * 4` 不该变成斜体', () => {
  // 少了这条，乘法式会渲染成 `2 <em> 3 </em> 4`。那是**错的输出**，
  // 不是「不支持的语法」，所以它不能靠「原样显示」那条兜底。
  assert.equal(html('2 * 3 * 4'), '<p class="md-p">2 * 3 * 4</p>')
  assert.equal(html('a ** b ** c'), '<p class="md-p">a ** b ** c</p>')
  // 但正常的还是正常
  assert.equal(html('2 * 3 和 *斜*'), '<p class="md-p">2 * 3 和 <em>斜</em></p>')
})

test('不支持内联 HTML —— 全程没有 innerHTML', () => {
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

// ═══════════════════════════════════════════════════════════
// 流式渲染
//
// ★ 这一节里最重要的是一条**不变量**：流式渲染的结果必须和一次性渲染
//   **逐字相同**。冻结尾部是个优化，优化错了的表现是「流式时看到的和
//   跑完之后看到的不一样」—— 那种错不会有任何东西报错，只有人眼能发现。
// ═══════════════════════════════════════════════════════════

/** 一个能当 `host` 用的容器 */
function container(): DomEl {
  return new DomEl('div')
}

/** host 里现在是什么 */
function inHost(host: DomEl): string {
  return host.kids.map(shape).join('')
}

test('逐字喂进去，结果和一次性渲染**完全相同**', () => {
  const src = [
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
    '> 需要 Node 22',
    '',
    '| 命令 | 作用 |',
    '| --- | --- |',
    '| `npm test` | 跑测试 |',
    '',
    '收尾。',
  ].join('\n')

  const once = html(src)
  const host = container()
  const stream = createStreamRenderer(host)
  // 每个字符一帧 —— 比真实流式（按 token）还碎，是最坏情况
  for (let i = 1; i <= src.length; i++) stream.update(src.slice(0, i))

  assert.equal(inHost(host), once)
})

test('按 token 大小的块喂，结果也一样', () => {
  const src = '# 标题\n\n一段话。\n\n- 甲\n- 乙\n\n```\n码\n```\n\n表格：\n\n| A | B |\n| --- | --- |\n| 1 | 2 |'
  const once = html(src)
  const host = container()
  const stream = createStreamRenderer(host)
  for (let i = 3; i < src.length; i += 3) stream.update(src.slice(0, i))
  stream.update(src)

  assert.equal(inHost(host), once)
})

test('已经定稿的块**不重建** —— 同一个节点对象还在那儿', () => {
  // 这是「冻结」这件事本身的证据。只断言输出相同是看不出差别的：
  // 整篇重画也能得到相同的输出，只是每帧抖一次、而且代价对最终长度是平方级的。
  //
  // 12 个段落 → 24 个块，尾部只留 2 个不稳定，所以前面 11 个必须原样留着。
  const head = `${Array.from({ length: 12 }, (_, i) => `第 ${i} 段`).join('\n\n')}\n`
  const host = container()
  const stream = createStreamRenderer(host)
  stream.update(head)

  const before = host.kids.slice()
  assert.equal(before.length, 12, '12 个段落各出一个节点')

  stream.update(`${head}\n收尾那一段`)

  // 第 0..9 段离定稿线远得很，必须**一个对象都没换**
  for (let k = 0; k < 10; k++) {
    assert.equal(host.kids[k], before[k], `第 ${k} 段应当是同一个节点对象（不是重建的）`)
  }
  assert.equal(shape(host.kids[12]), '<p class="md-p">收尾那一段</p>')
})

test('尾部那一块每帧重画 —— 列表接着长的时候看得见', () => {
  const host = container()
  const stream = createStreamRenderer(host)
  stream.update('- 甲')
  assert.equal(inHost(host), '<ul class="md-ul"><li>甲</li></ul>')
  stream.update('- 甲\n- 乙')
  assert.equal(inHost(host), '<ul class="md-ul"><li>甲</li><li>乙</li></ul>')
  stream.update('- 甲\n- 乙\n- 丙')
  assert.equal(inHost(host), '<ul class="md-ul"><li>甲</li><li>乙</li><li>丙</li></ul>')
})

test('文本**不是追加**时自己发现并重来（不靠对端发 reset）', () => {
  // 重试重新发起生成时，文本会从头再来。`generate:delta` 上带 `reset`，
  // 但界面正确性不该**只**依赖对端记得发它 —— 这里验的是自己看得出来。
  const host = container()
  const stream = createStreamRenderer(host)
  stream.update('第一版：这是半句话')
  stream.update('第二版：完全换了一句')
  assert.equal(inHost(host), '<p class="md-p">第二版：完全换了一句</p>')
})

test('reset() 清空，重来一遍不带上一版的残留', () => {
  const host = container()
  const stream = createStreamRenderer(host)
  stream.update('# 甲\n\n一段\n\n又一段')
  stream.reset()
  assert.equal(inHost(host), '')
  stream.update('# 乙')
  assert.equal(inHost(host), '<h1 class="md-h">乙</h1>')
})

test('unstableFrom：块太少时一行都不冻结', () => {
  assert.equal(unstableFrom(''), 0)
  assert.equal(unstableFrom('就一段'), 0)
  assert.equal(unstableFrom('# 标题'), 0)
})

test('unstableFrom 指向倒数第二个块的起点', () => {
  // 行：甲 / 空 / 乙 / 空 / 丙 / 空 —— 六个块，倒数第二个从第 4 行开始
  assert.equal(unstableFrom('甲\n\n乙\n\n丙\n'), 4)
  // 追加只可能让这条线**往后**走，不会往回退
  assert.ok(unstableFrom('甲\n\n乙\n\n丙\n\n丁') >= 4)
})
