/**
 * JevLoop · 把模型生成的回答渲染成 DOM
 *
 * ══════════════════════════════════════════════════════════════
 *  回答是 markdown，而界面一直用 `textContent` 显示 —— 于是 `##`、`**`
 *  `` ` ``、代码围栏全都以**字面**出现。实测（2026-09-21，用户报的）：
 *  一段正常的 README 总结在界面里是一屏带井号和星号的原文。
 * ══════════════════════════════════════════════════════════════
 *
 * ── 两条约束决定了这个文件长什么样 ──────────────────────────────
 *
 * **① 不引依赖。** 这个项目的立身之本之一是「无依赖、无构建步骤」
 *   （`package.json` 的 `dependencies` 是空的）。为了渲染回答引入
 *   `marked` + `DOMPurify` 两个包，换来的东西不值得 —— 所以手写。
 *
 * **② 绝不用 `innerHTML`。** 渲染的是**模型生成的文本** —— 那是**不可信
 *   输入**。这个文件全程 `createElement` + `textContent`，所以
 *   `<script>`、`onerror=`、`javascript:` 这些**在结构上无法生效**，
 *   不需要再挂一个 sanitizer 去追。（链接是唯一的例外，见 `safeHref`。）
 *
 *   ⚠️ 所以：**将来也不要为了「支持内联 HTML」把 `innerHTML` 加回来。**
 *   那一步会把这里唯一的安全性质扔掉。
 *
 * ── 支持什么，不支持什么 ────────────────────────────────────────
 *
 *   支持  `#` 标题 · 围栏代码块 · `-`/`1.` 列表 · `>` 引用 · `---` 分隔线
 *         · 表格 · `` `代码` `` · **粗** · *斜* · [链接](url)
 *   不支持 内联 HTML（有意）· 嵌套列表 · 图片 · 引用式链接
 *
 *   不支持的那些**原样显示**，不是丢掉 —— 宁可看到源码，也不要看到
 *   一段被悄悄吃掉的文本。
 *
 * @module JevLoop/web-markdown
 */

/** 建一个元素。`h()` 在 app.js 里，这里是**自带的**一份 —— 见文件头第 ① 条 */
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue
    if (k === 'class') node.className = v
    else if (k === 'text') node.textContent = v
    else node.setAttribute(k, v)
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue
    node.append(c instanceof Node ? c : document.createTextNode(String(c)))
  }
  return node
}

/**
 * 只放行这几种协议的链接。
 *
 * ★ 这是全文件**唯一**需要判断的地方。`[点我](javascript:alert(1))` 是一段
 *   在**点击时**执行的代码，而它不经过任何我们构造的 DOM —— 浏览器执行的是
 *   `href` 里那个协议。所以白名单挡在前面。
 *
 * 相对链接（`./x.md`）也放行：它们不会执行代码。
 */
function safeHref(href) {
  const h = href.trim()
  if (/^(https?:|mailto:)/i.test(h)) return true
  // 相对路径：不含协议分隔符，且不是 `javascript:` 那种伪协议
  return !/^[a-z][a-z0-9+.-]*:/i.test(h)
}

/**
 * 行内：`` `代码` `` · `**粗**` · `*斜*` · `[文字](链接)`
 *
 * `(?<!!)` 是**图片语法**的护栏。图片这个文件不支持（见文件头），而
 * `![图](x.png)` 里的 `[图](x.png)` 单独看就是一个合法链接 —— 于是
 * 「原样显示」变成了「一个感叹号后面跟一个链接」，实测（2026-09-21）。
 * 断言写在前面，它就不成其为链接，整段按普通文本出去。
 */
const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\n]+\*)|((?<!!)\[[^\]\n]*\]\([^)\s]+\))/g

function inline(text) {
  const out = []
  let last = 0
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push(document.createTextNode(text.slice(last, m.index)))
    const tok = m[0]
    if (tok.startsWith('`')) {
      out.push(el('code', { class: 'md-code' }, tok.slice(1, -1)))
    } else if (tok.startsWith('**') || tok.startsWith('__')) {
      out.push(el('strong', {}, tok.slice(2, -2)))
    } else if (tok.startsWith('*')) {
      out.push(el('em', {}, tok.slice(1, -1)))
    } else {
      const link = /^\[([^\]]*)\]\(([^)\s]+)\)$/.exec(tok)
      if (link && safeHref(link[2])) {
        out.push(el('a', { class: 'md-a', href: link[2], target: '_blank', rel: 'noreferrer noopener' }, ...inline(link[1])))
      } else {
        // 不放行就**原样显示**，不要丢掉 —— 读的人该看到模型写了什么
        out.push(document.createTextNode(tok))
      }
    }
    last = m.index + tok.length
  }
  if (last < text.length) out.push(document.createTextNode(text.slice(last)))
  return out
}

const FENCE = /^\s*```(\S*)\s*$/
const HEADING = /^(#{1,6})\s+(.*)$/
const UL = /^\s*[-*+]\s+(.*)$/
const OL = /^\s*\d+[.)]\s+(.*)$/
const QUOTE = /^\s*>\s?(.*)$/
const HR = /^\s*([-*_])\s*(\1\s*){2,}$/
/** 表格的分隔行：`|---|:--:|` 这种 */
const TABLE_SEP = /^\s*\|?[\s:|-]+\|[\s:|-]*$/

/** 一行的单元格 */
function cells(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
}

/**
 * 整段 markdown → `DocumentFragment`。
 *
 * 不认识的块**按段落原样渲染**，所以任何 markdown 至少不会丢内容。
 */
export function renderMarkdown(text) {
  const frag = document.createDocumentFragment()
  const lines = String(text ?? '').split('\n')
  let i = 0

  /**
   * 连续吃掉满足 `pred` 的行，返回每行经 `pick` 之后的结果；游标停在第一个
   * 不满足的行上。
   *
   * ★ **游标由这里推进，调用方拿不到 `i`** —— 这就是这个函数存在的全部理由。
   *
   * 原来六处「收集连续的同类行」各写各的 `while`，其中三处（引用 / 无序列表 /
   * 有序列表）忘了 `i++`：`pred(lines[i])` 永远为真，数组一直涨到 `2^32-1`，
   * 浏览器抛 `RangeError: Invalid array length`（实测 2026-09-21，回答里只要有
   * 一个列表或引用就整段渲染不出来，界面上是空白）。
   *
   * 当时**只修了段落那一处**（它也忘了），另外三处原样留着 —— 所以第二次
   * 仍然炸。教训不是「记得写 `i++`」，而是**修好一处不等于修好一类**：
   * 把推进游标这件事收进一个地方，让忘掉它变得不可能。
   */
  const takeWhile = (pred, pick) => {
    const out = []
    while (i < lines.length && pred(lines[i])) out.push(pick(lines[i++]))
    return out
  }

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i++
      continue
    }

    // ── 围栏代码块 ────────────────────────────────────────
    const fence = FENCE.exec(line)
    if (fence) {
      const lang = fence[1] ?? ''
      i++
      const buf = takeWhile(
        (l) => !FENCE.test(l),
        (l) => l,
      )
      i++ // 吃掉收尾的 ```
      frag.append(
        el('pre', { class: 'md-pre' }, el('code', { class: 'md-block', ...(lang ? { 'data-lang': lang } : {}) }, buf.join('\n'))),
      )
      continue
    }

    // ── 标题 ──────────────────────────────────────────────
    const head = HEADING.exec(line)
    if (head) {
      const lvl = head[1].length
      frag.append(el(`h${lvl}`, { class: 'md-h' }, ...inline(head[2])))
      i++
      continue
    }

    // ── 分隔线 ────────────────────────────────────────────
    if (HR.test(line)) {
      frag.append(el('hr', { class: 'md-hr' }))
      i++
      continue
    }

    // ── 引用 ──────────────────────────────────────────────
    if (QUOTE.test(line)) {
      const buf = takeWhile(
        (l) => QUOTE.test(l),
        (l) => QUOTE.exec(l)[1],
      )
      frag.append(el('blockquote', { class: 'md-quote' }, ...inline(buf.join('\n'))))
      continue
    }

    // ── 表格：本行有 `|`，下一行是分隔行 ──────────────────
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) && lines[i + 1].includes('-')) {
      const head = cells(line)
      i += 2
      const body = takeWhile((l) => l.includes('|') && l.trim(), cells)
      frag.append(
        el(
          'table',
          { class: 'md-table' },
          el('thead', {}, el('tr', {}, ...head.map((c) => el('th', {}, ...inline(c))))),
          el('tbody', {}, ...body.map((r) => el('tr', {}, ...head.map((_, k) => el('td', {}, ...inline(r[k] ?? '')))))),
        ),
      )
      continue
    }

    // ── 无序列表 ──────────────────────────────────────────
    if (UL.test(line)) {
      const items = takeWhile(
        (l) => UL.test(l),
        (l) => UL.exec(l)[1],
      )
      frag.append(el('ul', { class: 'md-ul' }, ...items.map((t) => el('li', {}, ...inline(t)))))
      continue
    }

    // ── 有序列表 ──────────────────────────────────────────
    if (OL.test(line)) {
      const items = takeWhile(
        (l) => OL.test(l),
        (l) => OL.exec(l)[1],
      )
      frag.append(el('ol', { class: 'md-ol' }, ...items.map((t) => el('li', {}, ...inline(t)))))
      continue
    }

    // ── 段落：连续的普通行合成一段 ────────────────────────
    // 这个谓词在 `line` 上**必然为真**（上面每一类都已经排除了，而空行在循环
    // 顶上就 `continue` 了），所以 `takeWhile` 一定至少吃掉一行 —— 主循环
    // 因此每轮都前进。这是「不会死循环」的论证，不是观察。
    const para = takeWhile(
      (l) =>
        l.trim() &&
        !FENCE.test(l) &&
        !HEADING.test(l) &&
        !HR.test(l) &&
        !QUOTE.test(l) &&
        !UL.test(l) &&
        !OL.test(l),
      (l) => l,
    )
    frag.append(el('p', { class: 'md-p' }, ...inline(para.join('\n'))))
  }

  return frag
}
