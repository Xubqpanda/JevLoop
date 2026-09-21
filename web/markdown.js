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
 * ── 为什么不能再拆 ──────────────────────────────────────────────
 *
 * **一句话说得完：把 markdown 变成 DOM。** 拆不出接缝，理由有三条，
 * 按分量排：
 *
 * **① 三件事共用同一个块扫描器，而那个共用是承重的。** 一次性渲染、
 *   定稿线（`unstableFrom`）、增量渲染（`createStreamRenderer`）都建在
 *   `scanBlock` 上 —— 它们不是三个消费者，是**同一个判断的三种用法**。
 *   把增量那半挪出去，就得把 `scanBlock` / `drawBlocks` 变成跨文件的
 *   内部 API，而「块到哪结束」这句话又从「一个地方」变回「一份约定」。
 *
 *   ★ 这不是假设：这个文件里**同一个 bug 出现过两次** —— 收集连续行的
 *     循环忘了推进游标，答案里有列表就整段渲染不出来。两次都是因为
 *     「怎么算一个块」被写了两遍以上。共用是修法，不是风格。
 *
 * **② 消费者是同一批。** §12 说接缝是「这里的输入输出形状变了」或者
 *   「这里的消费者不是同一批人」。这里两者都不成立：输入永远是一段文本，
 *   输出永远是一组 DOM 节点，而调用方只有 `app.js` 一个。
 *
 * **③ 它是唯一用户直接盯着看的代码。** 一个文件翻得完，改渲染的时候
 *   不用在几个文件之间来回跳。
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
 * ★ **`(?<!!)` 是图片语法的护栏。** 图片这个文件不支持（见文件头），而
 *   `![图](x.png)` 里的 `[图](x.png)` 单独看就是一个合法链接 —— 于是
 *   「原样显示」变成了「一个感叹号后面跟一个链接」（实测 2026-09-21）。
 *   断言写在前面，它就不成其为链接，整段按普通文本出去。
 *
 * ★ **`(?!\s)` 和 `[^\s*]` 是「标记内侧不能是空格」** —— CommonMark 的
 *   flanking 规则里最要紧的那半条。少了它 `2 * 3 * 4` 会渲染成
 *   `2 <em> 3 </em> 4`，那是**错的输出**，不是「不支持的语法」。
 *   完整的 flanking 规则要认标点类别和 Unicode，那是 DSH 引入 micromark
 *   的原因（见文件头：这里不引依赖）；这半条挡住了实际会碰到的那类。
 *
 * ⚠️ 故意的偏差：CommonMark 不许**词内**的 `__粗__`（`中文__粗__中文`
 *    应该原样），这里放行。理由是目标读者写中文，没有词边界可言 ——
 *    按 CommonMark 判就永远是「原样显示」，对写的人没有任何帮助。
 */
const INLINE = /(`[^`\n]+`)|(\*\*(?!\s)[^*\n]*[^\s*]\*\*)|(__(?!\s)[^_\n]*[^\s_]__)|(\*(?!\s)[^*\n]*[^\s*]\*)|((?<!!)\[[^\]\n]*\]\([^)\s]+\))/g

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
 * 扫一个顶层块：从 `lines[from]` 开始，返回它**到哪一行为止**（不含），
 * 以及「拿到那几行之后怎么画」。
 *
 * ── 为什么把「扫」和「画」分开 ──────────────────────────────────
 *
 * **因为流式渲染只重画尾部，而它要的是尾部的起点行号 —— 那要扫，不要画。**
 * 合成一个「扫到哪就画到哪」的函数，算边界就等于把整篇画了一遍，
 * 省下来的东西一分不剩。
 *
 * ★ 它**不碰任何游标**，只从 `from` 算出 `end` 还回去。这是 `takeWhile`
 *   那条教训的延续：调用方（`renderMarkdown` / `unstableFrom` /
 *   `createStreamRenderer`）各自持有自己的 `i`，而「块到哪结束」**只有
 *   这一个说法**。三个地方各写一遍边界判断就会有三个版本，而它们分叉的
 *   表现是「流式渲染和一次性渲染长得不一样」—— 那不会有任何东西报错。
 *
 * `end` 保证 `> from`；`drawBlocks` 会断言这一点。
 */
function scanBlock(lines, from) {
  const line = lines[from]

  // 空行：不画东西，但仍然是一个「块」（占一行），否则调用方会卡住
  if (!line.trim()) return { end: from + 1, draw: () => null }

  // ── 围栏代码块 ──────────────────────────────────────────
  const fence = FENCE.exec(line)
  if (fence) {
    const lang = fence[1] ?? ''
    let close = from + 1
    while (close < lines.length && !FENCE.test(lines[close])) close++
    const body = lines.slice(from + 1, close).join('\n')
    return {
      // 收尾的 ``` 也算这个块的一行；没找到就吃到结尾（没闭合的围栏也是合法的）
      end: close < lines.length ? close + 1 : close,
      draw: () =>
        el('pre', { class: 'md-pre' }, el('code', { class: 'md-block', ...(lang ? { 'data-lang': lang } : {}) }, body)),
    }
  }

  // ── 标题 ────────────────────────────────────────────────
  const head = HEADING.exec(line)
  if (head) {
    const lvl = head[1].length
    return { end: from + 1, draw: () => el(`h${lvl}`, { class: 'md-h' }, ...inline(head[2])) }
  }

  // ── 分隔线 ──────────────────────────────────────────────
  if (HR.test(line)) return { end: from + 1, draw: () => el('hr', { class: 'md-hr' }) }

  // ── 引用 ────────────────────────────────────────────────
  if (QUOTE.test(line)) {
    const take = takeWhile(lines, from, (l) => QUOTE.test(l))
    return {
      end: take.end,
      draw: () => el('blockquote', { class: 'md-quote' }, ...inline(take.items.map((l) => QUOTE.exec(l)[1]).join('\n'))),
    }
  }

  // ── 表格：本行有 `|`，下一行是分隔行 ────────────────────
  if (line.includes('|') && from + 1 < lines.length && TABLE_SEP.test(lines[from + 1]) && lines[from + 1].includes('-')) {
    const cols = cells(line)
    const take = takeWhile(lines, from + 2, (l) => l.includes('|') && l.trim())
    return {
      end: take.end,
      draw: () =>
        el(
          'table',
          { class: 'md-table' },
          el('thead', {}, el('tr', {}, ...cols.map((c) => el('th', {}, ...inline(c))))),
          el(
            'tbody',
            {},
            ...take.items.map((r) => el('tr', {}, ...cols.map((_, k) => el('td', {}, ...inline(cells(r)[k] ?? ''))))),
          ),
        ),
    }
  }

  // ── 无序列表 ────────────────────────────────────────────
  if (UL.test(line)) {
    const take = takeWhile(lines, from, (l) => UL.test(l))
    return {
      end: take.end,
      draw: () => el('ul', { class: 'md-ul' }, ...take.items.map((t) => el('li', {}, ...inline(UL.exec(t)[1])))),
    }
  }

  // ── 有序列表 ────────────────────────────────────────────
  if (OL.test(line)) {
    const take = takeWhile(lines, from, (l) => OL.test(l))
    return {
      end: take.end,
      draw: () => el('ol', { class: 'md-ol' }, ...take.items.map((t) => el('li', {}, ...inline(OL.exec(t)[1])))),
    }
  }

  // ── 段落：连续的普通行合成一段 ──────────────────────────
  // 这个谓词在 `line` 上**必然为真**（上面每一类都已经排除了，而空行在开头
  // 就返回了），所以它一定至少吃掉一行。这是「不会死循环」的论证，不是观察。
  const take = takeWhile(
    lines,
    from,
    (l) =>
      l.trim() && !FENCE.test(l) && !HEADING.test(l) && !HR.test(l) && !QUOTE.test(l) && !UL.test(l) && !OL.test(l),
  )
  return { end: take.end, draw: () => el('p', { class: 'md-p' }, ...inline(take.items.join('\n'))) }
}

/**
 * 把 `lines[from..to)` 之间的顶层块画进 `parent`，返回画出来的节点。
 *
 * ★ 那句 `end <= i` 的断言是**一个死循环的墓碑**。
 *
 *   2026-09-21：三处「收集连续同类行」的 `while` 忘了推进游标，`pred` 永远
 *   为真，数组涨到 `2^32-1`，浏览器抛 `RangeError: Invalid array length`，
 *   整个回答渲染成空白。当时修法是让 `takeWhile` 独占游标。
 *
 *   现在结构又变了一次（边界挪进了 `scanBlock`），所以**那个保证要重新
 *   建立一次** —— 而且这次是**当场抛一个说得出名字的错**，而不是等着
 *   内存涨到 32GB。同一个坑在同一个文件里出现过两次，第三次不该再靠自觉。
 */
function drawBlocks(parent, lines, from, to) {
  const nodes = []
  let i = from
  // `i < to` 之外还要 `i < lines.length` —— `to` 允许是 `Infinity`（「画到底」），
  // 而 `Infinity` 会把原来 `while (i < lines.length)` 的边界弄丢，
  // 于是 `lines[i]` 是 `undefined`，`scanBlock` 里第一句 `.trim()` 就抛。
  while (i < to && i < lines.length) {
    const block = scanBlock(lines, i)
    const node = block.draw()
    if (node) {
      parent.append(node)
      nodes.push(node)
    }
    if (block.end <= i) throw new Error(`markdown 块扫描在第 ${i} 行没有前进 —— 这是死循环`)
    i = block.end
  }
  return { nodes, end: i }
}

/** 连续吃掉满足 `pred` 的行。**游标由这里推进，调用方拿不到 `i`。** */
function takeWhile(lines, from, pred) {
  const items = []
  let i = from
  while (i < lines.length && pred(lines[i])) items.push(lines[i++])
  return { items, end: i }
}

/**
 * 整段 markdown → `DocumentFragment`。
 *
 * 不认识的块**按段落原样渲染**，所以任何 markdown 至少不会丢内容。
 */
export function renderMarkdown(text) {
  const frag = document.createDocumentFragment()
  drawBlocks(frag, String(text ?? '').split('\n'), 0, Infinity)
  return frag
}

/**
 * 尾部留几个块**不冻结**。
 *
 * 追加的文本只可能改变最后一个块，留一个是够的。留两个是**余量** ——
 * 省得去推理「块边界本身会不会动」（一个段落会因为下一行是 `|---|` 而
 * 变成表格，一个列表会因为多了一行而变长）。这和 DSH 的 `incremental.ts`
 * 取同样的值、同样的理由（那边叫 `UNSTABLE_TAIL_BLOCKS`）。
 */
const UNSTABLE_TAIL_BLOCKS = 2

/**
 * 从第几行开始**还没定稿**。它之前的行画出来就不会再变了。
 *
 * 返回的是**倒数第二个块的起始行**（块不够多时返回 0 = 什么都不冻结）。
 */
export function unstableFrom(text) {
  const lines = String(text ?? '').split('\n')
  const starts = []
  for (let i = 0; i < lines.length; ) {
    starts.push(i)
    const end = scanBlock(lines, i).end
    if (end <= i) throw new Error(`markdown 块扫描在第 ${i} 行没有前进 —— 这是死循环`)
    i = end
  }
  return starts.length <= UNSTABLE_TAIL_BLOCKS ? 0 : starts[starts.length - UNSTABLE_TAIL_BLOCKS]
}

/**
 * 流式渲染器：把**追加式增长**的回答画进 `host`，只重画尾部。
 *
 * ══════════════════════════════════════════════════════════════
 *  参考 DSH 的 `packages/client/ui-primitives/src/markdown/incremental.ts`。
 *  那份文件的核心判断是这一句：
 *
 *    「CommonMark 的块解析是按行的，而追加的文本只能改变**解析边界** ——
 *      最后一个顶层块（一个段落变成 setext 标题或表格，一个列表在空行之后
 *      继续）—— 所以前面的块已经定稿。」
 *
 *  它因此冻结尾部两个块、只重解析尾部，把「每个 chunk 重解析整篇」
 *  （对最终长度是**平方级**的）变成「每个源区间只解析有限次」。
 * ══════════════════════════════════════════════════════════════
 *
 * **抄的是判断，不是代码。** 那份建在 micromark 的 `position` 偏移上；
 * 这边的渲染器本来就是按行扫的，所以边界是**行号**，不需要位置偏移。
 * 依赖也抄不过来：这个文件的前提是**零依赖**（见文件头）。
 *
 * 除了快，它解决的是**观感**：整篇重画会让已经读过的部分每帧抖一次，
 * 而且没闭合的围栏会把后面所有内容吞进代码块里（一闪一闪）。
 * 冻住的部分一个字节都不动，所以抖的只有最后那两块。
 *
 * @param host 容器元素。它的内容**由这个渲染器独占** —— 调用方别再往里塞东西。
 */
export function createStreamRenderer(host) {
  /** 已经画进 `host` 且不会再变的行数 */
  let frozen = 0
  /** `host` 里 `frozen` 之后的那几个节点，每帧重画 */
  let tail = []
  /** 上一帧的完整文本，用来判「还是追加吗」 */
  let last = ''

  const dropTail = () => {
    for (const n of tail) n.remove()
    tail = []
  }
  const reset = () => {
    host.replaceChildren()
    tail = []
    frozen = 0
    last = ''
  }

  const update = (text) => {
    const now = String(text ?? '')

    /*
      ★ **判据是「还是追加吗」，而不是「调用方说重置了吗」。**

      `generate:delta` 上的 `reset` 是**意图**：重试重新发起生成时会带上它。
      但界面正确性不该只依赖对端记得发那个标记 —— 文本**不是**上一帧的延长
      时，这里自己就看得出来（重试、改口、变短都会让这条不成立）。
      两边都做：`reset` 让重画立刻发生，这一句保证它**一定**发生。
    */
    if (!now.startsWith(last)) reset()

    const lines = now.split('\n')
    const from = unstableFrom(now)

    // 定稿线往回退了。在「只追加」的前提下不该发生（`startsWith` 已经保证
    // 前缀没变），真发生了就以它为准 —— 宁可多画一遍，也不要画错。
    if (from < frozen) reset()

    // 尾部每帧重画：它可能还在长（列表接一行、段落变成表格）
    dropTail()
    // 新定稿的部分画一次，之后不再碰
    frozen = drawBlocks(host, lines, frozen, from).end
    // 新的尾部
    tail = drawBlocks(host, lines, frozen, Infinity).nodes
    last = now
  }

  return { update, reset }
}
