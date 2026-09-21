/**
 * JevLoop · 界面
 *
 * 两个视图，**同一串事件**：
 *
 *     对话   你说了什么、它答了什么。过程默认折成一行
 *     轨迹   每一次判定、每一次工具调用、每一次生成，一张卡一张卡铺开
 *
 * 分开不是为了藏东西，是因为它们回答的问题不同。对话回答「它做成了吗」，
 * 轨迹回答「它凭什么这么判断」。切过去看不需要重新跑 —— 事件全在内存里，
 * 轨迹是**回放**出来的，重跑就是为同一个答案付两次钱。
 *
 * 视觉照搬自 DeepSeek Harness（MIT，Copyright © 2026 DeepSeek）：
 * 只有用户消息有气泡，助手正文落在背景上，过程折叠成一行。
 *
 * 主张只有一条：**橙色（模型生成）必须稀少且扎眼。**
 *
 * ## 待拆
 *
 * 开头的「两个视图，同一串事件」是一句话，但文件内部是十三节：
 * DOM 小工具 / 状态 / 视图切换 / 渲染：对话 / 滚动 / 跳转轨 /
 * 渲染：轨迹 / 右栏：记账 / 事件分发 / 运行 / DECISION.md 面板 /
 * 主题 / 会话。
 *
 * 接缝就是这些节，而且**三条链几乎不交叉**：`渲染：对话`、`渲染：轨迹`
 * 各自闭包（都只读同一份事件数组），`事件分发 + 运行` 是另一块。
 * 拆的时候按这三块走，不按行数。
 *
 * `滚动` 和 `跳转轨` 是后加的：两者都只操作 `.stream` 一个元素，
 * 既不碰事件流也不碰渲染，合起来是一块独立的「视口控制」——
 * 要摘的话先摘它。
 */

import { renderMarkdown, createStreamRenderer } from './markdown.js'

// ═══════════════════════════════════════════════════════════
// DOM 小工具（不引框架：这个界面只有两种交互，画卡片和切视图）
// ═══════════════════════════════════════════════════════════

/**
 * 一次生成调用的 token 账。
 *
 * 三个数回答不同的问题，所以**并排显示**：
 *
 *     估 N      我们按字符启发式估的「我们发出去的那部分」
 *     入 M       provider 报的输入（含生成器内部的 system prompt）
 *     出 K       provider 报的输出
 *
 * `入/出` 是 0 的时候**不说 0，说「未报」** —— 脚本生成器不报 usage，
 * 那是「没量到」而不是「量到了 0」（§8.10）。把两者显示成同一个东西
 * 会让人以为一次调用真的没花 token。
 */
/** 上下文账目的一句话。两个调用点共用 —— 分两处写必然分叉 */
function contextBits(e) {
  const bits = [`${e.rawChars} → ${e.keptChars} 字符`]
  if (e.prunedCount > 0) bits.push(`剪中间 ${e.prunedCount} 条`)
  if (e.foldedCount > 0) bits.push(`折叠 ${e.foldedCount} 条`)
  if (e.overRetain) bits.push('⚠ 压完仍超目标')
  return bits
}

/** 上文（多轮问答）的账目。和 `contextBits` 是一对，但**单位是轮**，不是步 */
function conversationBits(e) {
  const bits = [`${e.rawTurns} → ${e.keptTurns} 轮`, `${e.rawChars} → ${e.keptChars} 字符`]
  if (e.foldedTurns > 0) bits.push(`折叠 ${e.foldedTurns} 轮`)
  if (e.overRetain) bits.push('⚠ 压完仍超目标')
  return bits
}

function tokenText(e) {
  const reported = e.inputTokens > 0 || e.outputTokens > 0
  if (!reported) return `估 ${e.estimatedInputTokens} tok · 用量未报`
  return `估 ${e.estimatedInputTokens} / 入 ${e.inputTokens} / 出 ${e.outputTokens} tok`
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue
    if (k === 'class') el.className = v
    else if (k === 'text') el.textContent = v
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v)
    else el.setAttribute(k, v)
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue
    el.append(c instanceof Node ? c : document.createTextNode(String(c)))
  }
  return el
}

/**
 * 概率的显示。
 *
 * **四舍五入绝不能跨过 0 和 1 这两个端点。** 先前的写法是
 * `n >= 0.995 ? toFixed(0) : toFixed(1)`，于是 0.995 画成 `100%`、
 * 0.004 画成 `0%` —— 把一个**不确定**的判定画成了**确定**的。
 * 这个界面的全部主张就是「判定带概率」，所以端点必须保住：
 * 只有恰好等于 1 才配显示 100%。
 */
const pct = (n) => {
  if (n >= 1) return '100%'
  if (n <= 0) return '0%'
  const s = (n * 100).toFixed(1)
  return s === '100.0' ? '>99.9%' : s === '0.0' ? '<0.1%' : `${s}%`
}
const ms = (n) => `${Math.round(n)}ms`

// ═══════════════════════════════════════════════════════════
// 状态
// ═══════════════════════════════════════════════════════════

const state = {
  decisions: 0,
  models: 0,
  tools: 0,
  rules: 0,
  running: false,
  /**
   * 最后一次 `generate` 事件。
   *
   * 留着它是因为 **provider 报的 token 真值只在那上面**，而记账面板要
   * 把它和我们估的构成并排显示 —— 那个差值是校准启发式的唯一依据
   * （见 `context.ts` 的 `priceGenerateRequest`）。
   */
  lastGenerate: null,
  /** 登记过的工作区。空数组 = 一个都没登记，这时服务端用它自己的演示目录 */
  workspaces: [],
  /** 会话列表（摘要，不含正文 —— 见 `SessionStore` 的模块头） */
  sessions: [],
  /** 「其他 N 个会话」那一组展开着吗 */
  showOthers: false,
}

/**
 * 对话记录。
 *
 * 每一轮 = 用户一句话 + 一次运行。事件留在轮里，所以切到轨迹
 * 可以直接回放，不需要重跑。
 */
const turns = []
let current = null

const $ = (id) => document.getElementById(id)
const chat = $('chat')

function updateTally() {
  $('t-decide').textContent = state.decisions
  $('t-model').textContent = state.models
  $('t-tool').textContent = state.tools
  $('t-rules').textContent = state.rules
}

// ═══════════════════════════════════════════════════════════
// 视图切换
// ═══════════════════════════════════════════════════════════

const VIEWS = ['chat', 'trace', 'spec']

function selectView(id) {
  for (const v of VIEWS) {
    const on = v === id
    $(`tab-${v}`).setAttribute('aria-selected', String(on))
    $(`view-${v}`).hidden = !on
  }
  try {
    localStorage.setItem('jl-view', id)
  } catch {
    /* 隐私模式下 localStorage 会抛，忽略即可 —— 视图选择不是关键功能 */
  }
}

for (const v of VIEWS) $(`tab-${v}`).addEventListener('click', () => selectView(v))

// ═══════════════════════════════════════════════════════════
// 渲染：对话
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// 滚动：跟着最新走，但不跟用户抢
// ═══════════════════════════════════════════════════════════

const stream = $('stream')

/**
 * 多靠近底部才算「在底部」。
 *
 * 这个判断是整段滚动逻辑的核心 —— 没有它，两条路都是错的：
 * 每来一行就滚到底，你往回翻第 3 步时页面自己在动，翻不上去；
 * 完全不自动滚，长任务就得手动追着拖。
 *
 * 所以规则是：**你离开底部我就停手，你回到底部我接着跟。**
 *
 * 阈值不取 0，是因为「底部」在像素上永远差一两像素（行高取整、
 * 平滑滚动还没停稳）。精确比较会把它误判成「用户滚走了」而永久停手 ——
 * 表现就是自动滚动彻底失效，且看不出原因。
 */
const STICK_GAP_PX = 32

let stick = true

function nearBottom() {
  return stream.scrollHeight - stream.scrollTop - stream.clientHeight <= STICK_GAP_PX
}

/**
 * 跟着尾巴滚 —— 只在用户没有主动离开时。
 *
 * 用 `scrollTop = scrollHeight`，**不用 `scrollIntoView`**。后者有两个毛病，
 * 合起来正好就是「页面完全不动」：
 *
 *   1. 它会一路滚动**所有**可滚动的祖先，包括页面本身；
 *   2. 连续调用时，后一次平滑动画会取消前一次 —— 每 390ms 来一行，
 *      上一段动画还没走完就被取消，于是永远停在原地。
 */
function followTail() {
  if (!stick) return
  stream.scrollTop = stream.scrollHeight
  syncToBottom()
}

function syncToBottom() {
  $('to-bottom').hidden = nearBottom()
}

/** 正在做程序化的平滑滚动。期间不接受滚动事件的判决。 */
let smoothJump = false

/**
 * 回到底部（点按钮时）。这里用平滑滚动，所以需要一个标志位：
 * 动画途中每一帧都「不靠近底部」，不加标志的话每一帧都会把 stick
 * 判成 false —— 按钮自己把自己的自动跟随关掉了。
 */
function backToBottom() {
  stick = true
  smoothJump = true
  stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' })
  clearTimeout(backToBottom.timer)
  backToBottom.timer = setTimeout(() => {
    smoothJump = false
    stick = nearBottom()
    syncToBottom()
    syncActiveMark()
  }, 400)
}

stream.addEventListener('scroll', () => {
  if (smoothJump) return
  stick = nearBottom()
  syncToBottom()
  syncActiveMark()
})

$('to-bottom').addEventListener('click', backToBottom)

// ═══════════════════════════════════════════════════════════
// 跳转轨：每一次输入一个刻度
// ═══════════════════════════════════════════════════════════

/**
 * 屏幕右边那条轨道，一个刻度 = 一次用户输入。
 *
 * 它解决的是「往回翻的成本随长度增长」：跑十几步之后想回到「我最早说的是
 * 什么」要滚很久，而且滚的过程没有落点。
 *
 * 刻度按**内容里的位置比例**摆，不是等距排 —— 等距只能说明「有 5 次输入」，
 * 按比例才能说明「那 5 次分别在多深的地方」。
 */
const turnMarks = []
let railEls = []

function syncTurnRail() {
  $('turn-rail').hidden = turnMarks.length < 2
  if (turnMarks.length < 2) return

  const scroller = $('turn-rail-scroller')

  // 刻度只在数量变化时重建。位置每次都要重算 —— 内容变长，比例就变了。
  if (railEls.length !== turnMarks.length) {
    railEls = turnMarks.map((mark, i) => {
      const el = h('button', {
        class: 'turn-mark',
        type: 'button',
        title: mark.text.length > 60 ? `${mark.text.slice(0, 60)}…` : mark.text,
        'aria-label': `第 ${i + 1} 次输入`,
      })
      el.addEventListener('click', () => scrollToTurn(i))
      return el
    })
    scroller.replaceChildren(...railEls)
  }

  const total = stream.scrollHeight || 1
  const top = stream.getBoundingClientRect().top
  for (const [i, el] of railEls.entries()) {
    // 视口坐标 + 已滚距离 = 内容坐标
    const y = turnMarks[i].el.getBoundingClientRect().top - top + stream.scrollTop
    el.style.top = `${(Math.min(1, Math.max(0, y / total)) * 100).toFixed(3)}%`
  }
  syncActiveMark()
}

/**
 * 跳到第 i 次输入。
 *
 * 同样不用 `scrollIntoView` —— 它连整个页面一起滚。这里要动的只有
 * `.stream` 一个元素，所以按「目标相对滚动视口的偏移」算差值。
 */
function scrollToTurn(i) {
  const mark = turnMarks[i]
  if (!mark) return
  const delta = mark.el.getBoundingClientRect().top - stream.getBoundingClientRect().top
  // 手动跳转意味着用户不是要看最新的，把自动跟随关掉
  stick = false
  smoothJump = true
  stream.scrollTo({ top: Math.max(0, stream.scrollTop + delta - 12), behavior: 'smooth' })
  clearTimeout(scrollToTurn.timer)
  scrollToTurn.timer = setTimeout(() => {
    smoothJump = false
    stick = nearBottom()
    syncToBottom()
    syncActiveMark()
  }, 400)
}

/** 高亮「现在读到的是哪一次输入」：最后一个已经越过视口顶部的刻度 */
function syncActiveMark() {
  if (!railEls.length) return
  const top = stream.getBoundingClientRect().top
  let active = 0
  for (const [i, mark] of turnMarks.entries()) {
    if (mark.el.getBoundingClientRect().top - top <= 16) active = i
  }
  for (const [i, el] of railEls.entries()) el.classList.toggle('active', i === active)
}

function userTurn(text) {
  const el = h('div', { class: 'msg-user' }, h('div', { class: 'bubble' }, text))
  chat.append(el)
  turnMarks.push({ el, text })
  syncTurnRail()
  followTail()
}

/**
 * 一次运行是「一次模型调用 + 很多次判定」，所以**过程本身就是内容**。
 *
 * 每一行事件追加到列表里，不覆盖 —— 覆盖的话你只看得见最后一个，
 * 而「判定 13 次、模型 1 次」这件事恰恰只有铺开才看得见。
 *
 * 列表末尾永远有一行运行指示器，**它的秒数在走**。这是「慢」和「断」
 * 的分界线：一次判定在 hosted Jev 上要 ~390ms，所以几秒没动静是正常的，
 * 但秒数一直涨而不出下一行，就说明真的卡住了。
 */
function assistantTurn() {
  const list = h('div', { class: 'process-list' })
  const running = h('div', { class: 'running-row' }, h('span', { class: 'pulse' }), '开始…')
  const answer = h('div', { class: 'answer' }, '')
  const foot = h('div', { class: 'msg-foot' })
  const el = h('div', { class: 'msg-assistant' }, list, running, answer, foot)
  chat.append(el)
  followTail()
  return {
    el,
    list,
    running,
    answer,
    foot,
    rows: 0,
    finished: false,
    stopped: false,
    // 回答是**边生成边长**的：`draft` 是到目前为止收到的全文，
    // `stream` 只重画尾部（见 `markdown.js` 的 `createStreamRenderer`）。
    draft: '',
    stream: createStreamRenderer(answer),
  }
}

/** 事件 → 过程里的一行 */
function processRow(e, dur) {
  const meta = EVENT_META[e.type]
  if (!meta) return null

  const detail = []
  switch (e.type) {
    case 'decision': {
      const picked = Object.entries(e.answers ?? {})[0]
      detail.push(h('span', { class: 'hit' }, `→ ${e.action}`))
      if (picked) {
        const [qid, a] = picked
        if (a.type === 'choice') detail.push(` · ${qid}=${a.choice} ${pct(a.probabilities?.[a.choice] ?? 0)}`)
        else if (a.type === 'noul') detail.push(` · ${qid}=${a.noul >= 0.5 ? '真' : '假'} ${pct(a.noul)}`)
        else if (a.type === 'score') detail.push(` · ${qid}=档位 ${a.score}`)
      }
      break
    }
    case 'tool:call':
      detail.push(e.input ? `${e.tool}(${e.input})` : `${e.tool}()`)
      break
    case 'tool:result':
      detail.push(`${String(e.output ?? '').length} 字符`)
      break
    case 'generate':
      detail.push(`${e.kind} · ${tokenText(e)} · 唯一贵的一步`)
      break
    case 'authorize':
      detail.push(`${e.tool} · ${e.approved ? '已授权' : '已拒绝'} · ${e.reason ?? ''}`)
      break
    case 'audit':
      detail.push(`${e.record?.tool ?? ''} · risk ${e.record?.risk ?? '?'}`)
      break
    case 'context': {
      // 压了什么必须写清楚 —— 否则「这次答得不全」会被归因到模型身上
      detail.push(contextBits(e).join(' · '))
      break
    }
    case 'conversation': {
      // 同理：上文被折过而没说，模型答不出「那个文件」时会被归因到它自己身上
      detail.push(conversationBits(e).join(' · '))
      break
    }
    default:
      break
  }

  return h(
    'div',
    { class: `prow ${meta.cls}` },
    h('span', { class: 'pdot' }),
    h('span', { class: 'pkind' }, meta.label),
    h('span', { class: 'pname' }, e.id ?? e.tool ?? ''),
    h('span', { class: 'pdetail' }, ...detail),
    h('span', { class: 'ptime' }, dur && dur.measured ? ms(dur.ms) : ''),
  )
}

/**
 * 运行指示器：显示「现在在等什么」+ 已经等了多久。
 *
 * 没有这个，一次 37 秒的运行看起来和挂掉没有区别 —— 而它其实一直在跑，
 * 只是每次判定之间隔着几百毫秒的静默。
 */
/**
 * 「比平时久」的界线，**按阶段分开**。
 *
 * ★ 一个数管三段是错的：托管 Jev 一次判定约 **390ms**（§8.11），而一次
 *   生成 **2 秒起、几十秒都正常**（生成器自己的超时是 60 秒）。用同一个
 *   6 秒，既会把正常的生成报成「比平时久」，又对真卡住的判定太宽容。
 *
 * 数字的来源：判定 6 秒 ≈ 托管 Jev 的 15 倍；生成 30 秒 ≈ 实测常见的
 * 10 倍，同时留在 60 秒超时之内 —— 超过它多半是真出问题了，但**还可能
 * 正常返回**，所以文案是「比平时久」而不是「失败」。
 */
const STALL_AFTER_MS = { decide: 6000, tool: 6000, generate: 30000 }
let ticker = null

function startTicker() {
  stopTicker()
  ticker = setInterval(() => {
    if (!current || current.finished || current.stopped) return
    const waited = Date.now() - (current.lastEventAt ?? Date.now())
    const secs = (waited / 1000).toFixed(0)
    const stalled = waited >= (STALL_AFTER_MS[current.phaseKind] ?? 6000)
    current.running.className = `running-row${stalled ? ' stalled' : ''}`
    current.running.replaceChildren(
      h('span', { class: 'pulse' }),
      `${current.waitingOn}  ·  已等待 ${secs}s${stalled ? '（比平时久）' : ''}`,
    )
  }, 500)
}

function stopTicker() {
  if (ticker !== null) {
    clearInterval(ticker)
    ticker = null
  }
}

/**
 * 换「现在在干什么」，并把秒数从头数起。
 *
 * **只有「开始」信号该调它** —— `phase` 事件和 `tool:call`。其余事件都是
 * 做完之后才发的，拿它们改标签就是原来那个 bug（生成期间显示「正在判定」）。
 */
function setPhase(text, kind) {
  if (!current) return
  current.waitingOn = text
  current.phaseKind = kind
  current.lastEventAt = Date.now()
  if (!current.stopped) {
    current.running.className = 'running-row'
    current.running.replaceChildren(h('span', { class: 'pulse' }), `${text}  ·  已等待 0s`)
  }
}

/**
 * 又来了一条事件 —— **只把秒数从头数起，不动标签**。
 *
 * 两件事分开是因为它们回答不同的问题：标签是「在做什么」（由开始信号决定），
 * 秒数是「上一次有动静是多久以前」（任何事件都算）。
 */
function touch() {
  if (current) current.lastEventAt = Date.now()
}

/** `phase` 事件 → 那句话。判定节点 id 不进去 —— 轨迹里有，这里加只会变吵 */
function phaseText(e) {
  return e.kind === 'generate' ? '正在生成回答' : '正在判定'
}

/**
 * 「判定 : 模型」的显示文本。**两个调用点共用这一个函数。**
 *
 * 以前消息页脚（`finishAssistant`）和右侧统计栏（`renderEndStats`）各写了一份，
 * 而且**已经不一致**：`decisions` 缺失时一个渲染 `? : 0`、另一个渲染 `— : 0` ——
 * 同一次运行、同屏可见，两种文本。旁边那段「不在这里重算，两处实现会分叉」
 * 的注释防住的是**除法**，没防住**格式化**，而分叉恰恰发生在格式化上。
 *
 * `ratio === null` 是服务端的「没有模型调用」（见 `MeterStats.ratio`）——
 * 以前那里是 `Infinity`，而 JSON 把它变成 `null`，于是前端那条
 * `!Number.isFinite` 分支**永远不可达**，显示成 `?` 而不是它想显示的 `N : 0`。
 * 恰恰是「跑了 N 次判定、一次模型调用都没成」这个最需要看清的情形。
 */
function ratioText(s) {
  const r = s?.ratio
  if (r === null) return `${s?.decisions ?? '?'} : 0`
  if (typeof r !== 'number' || !Number.isFinite(r)) return '?'
  return `${r.toFixed(1)} : 1`
}

function finishAssistant(text, stats, halt) {
  if (!current) return
  current.finished = true
  stopTicker()
  current.el.removeChild(current.running)
  /*
    ★ **回答按 markdown 渲染，不再当纯文本。**

    以前这里是 `textContent = text` —— 于是模型输出的 `##`、`**`、代码围栏
    全都以**字面**出现，一段正常的总结是一屏带井号和星号的原文
    （实测 2026-09-21，用户报的）。

    渲染器在 `markdown.js`：手写（不引依赖）、**全程构造 DOM 不用
    `innerHTML`**（渲染的是模型生成的文本，即不可信输入）。
  */
  current.answer.replaceChildren(text ? renderMarkdown(text) : '（没有回答）')

  const s = stats ?? {}
  const ratio = ratioText(s)
  current.foot.className = `msg-foot${halt === 'error' ? ' failed' : ''}`
  current.foot.replaceChildren(
    h('span', {}, `${s.decisions ?? '?'} 判定`),
    h('span', {}, `${s.modelCalls ?? '?'} 模型`),
    h('span', {}, `${ratio} 判定:模型`),
    h('span', {}, `停于 ${halt}`),
  )
  followTail()
}

// ═══════════════════════════════════════════════════════════
// 渲染：轨迹
//
// 结构照搬 DSH 的 ui-trajectory：顶部时间轴 + 事件账本 + 详情面板。
//
// 时间轴横轴是**真实耗时**，三条泳道是判定 / 工具 / 模型。它一眼要说的
// 就是那个主张：蓝的一堆、橙的只有一个，而橙的那个最宽。
//
// 账本一行一个事件，30px。点一行，右边显示它的全部内容 ——
// 判定卡里那些概率条搬到了那里，因为一行 30px 放不下四个选项，
// 而概率恰恰是这个界面最不该省略的东西。
// ═══════════════════════════════════════════════════════════

/** 泳道号。顺序和 .plot-labels 里的三行标签一一对应 */
const LANE = { decide: 0, tool: 1, model: 2 }

/**
 * 事件 → 它在轨迹里的样子。
 *
 * 时间轴和账本共用这张表，所以两边的分类永远不会不一致 ——
 * 分两处写的话，加一种事件时必然只改一处。
 */
const EVENT_META = {
  decision: { label: '判定', cls: 'decide', lane: LANE.decide },
  // 授权单独一类 —— 它是**停下来等人**，不是一次判定，也不该像出错
  authorize: { label: '授权', cls: 'authorize', lane: LANE.decide },
  audit: { label: '审计', cls: 'audit', lane: LANE.decide },
  // 上下文账目是**代码做的决定**（预算是纯代码），所以归「规则」那一档的灰
  context: { label: '预算', cls: 'audit', lane: LANE.decide },
  // 上文折叠同理，也是纯代码 —— 但**单位不同**（轮 vs 步），所以徽章分开写
  conversation: { label: '上文', cls: 'audit', lane: LANE.decide },
  // 工具那两行用等宽字体：它们的内容是命令和输出，不是句子
  'tool:call': { label: '工具', cls: 'tool', lane: LANE.tool, mono: true },
  'tool:result': { label: '工具结果', cls: 'tool', lane: LANE.tool, mono: true },
  generate: { label: '模型', cls: 'model', lane: LANE.model },
}

/**
 * 图例的说明文字 —— 按**类**写，不按事件写。
 *
 * 同一档颜色下可能有好几种徽章（灰色那一档就是「审计」和「预算」），
 * 它们在图里长得一模一样，所以在图例里也该并排出现、共用一句说明。
 */
const LEGEND_NOTES = {
  decide: { text: '一次前向就出答案，不生成文本' },
  audit: { text: '代码做的决定，不过模型' },
  tool: { text: '唯一有真实副作用的地方' },
  model: { text: '唯一贵的一步，整条轨迹上最宽的那一条', strong: true },
  authorize: { text: '停下来等人' },
}

/** 图例的排列顺序。不在这里的类会被追加到末尾并标出来，不会被丢掉。 */
const LEGEND_ORDER = ['decide', 'audit', 'tool', 'model', 'authorize']

/**
 * 图例是 EVENT_META 的**第三个**消费者（另两个是时间轴和账本）。
 *
 * 上一版图例是手写的 HTML，于是和轨迹漂移了个彻底：轨迹早就是 19px 的
 * 文字徽章，图例还在画 3px 色块，连「规则」那一档的类名（`.swatch.rule`）
 * 在轨迹侧都不存在。三样全错，却没有任何东西会报错。
 *
 * 从同一张表生成就没有这个问题：加一种事件，图例自己会多出一行。
 */
function renderLegend() {
  const byCls = new Map()
  for (const meta of Object.values(EVENT_META)) {
    if (!byCls.has(meta.cls)) byCls.set(meta.cls, [])
    byCls.get(meta.cls).push(meta.label)
  }

  // EVENT_META 里出现了、但 LEGEND_ORDER 没收录的类，追加到末尾。
  // 说明文字缺失时报「还没写说明」，**不静默少一行** —— 少一行是看不出来的。
  const extra = [...byCls.keys()].filter((cls) => !LEGEND_ORDER.includes(cls))

  const rows = []
  for (const cls of [...LEGEND_ORDER, ...extra]) {
    const labels = byCls.get(cls)
    if (!labels) continue
    const note = LEGEND_NOTES[cls]
    rows.push(
      h(
        'div',
        { class: 'legend-row' },
        h(
          'span',
          { class: 'legend-badges' },
          ...labels.map((label) => h('span', { class: `kind-tag ${cls}` }, label)),
        ),
        h('span', {}, note?.strong ? h('b', {}, note.text) : (note?.text ?? '（还没写说明）')),
      ),
    )
  }
  $('legend-rows').replaceChildren(...rows)
}

/**
 * 事件自己报的耗时。
 *
 * 时间轴按它排布。没有耗时的那些（工具发起、授权、审计）给一个标称宽度 ——
 * 它们在时间轴上只是"发生过"的记号，不是可测量的区间。
 * 标称值不参与总时长计算，否则会把它撑大，让真正贵的那个显得没那么宽。
 */
function durationOf(e) {
  switch (e.type) {
    case 'decision':
      return { ms: e.latencyMs ?? 0, measured: true }
    case 'tool:result':
      return { ms: e.ms ?? 0, measured: true }
    case 'generate':
      return { ms: e.latencyMs ?? 0, measured: true }
    default:
      return { ms: 0, measured: false }
  }
}

/** 一行里「内容」列写什么。返回一串节点，让调用方决定怎么排版 */
function contentOf(e) {
  const mono = (t) => h('span', { class: 'mono' }, t)
  const dim = (t) => h('span', { class: 'dim' }, t)

  switch (e.type) {
    case 'decision': {
      const picked = Object.entries(e.answers ?? {})[0]
      const bits = [h('span', { class: 'hit' }, e.action)]
      if (picked) {
        const [qid, a] = picked
        if (a.type === 'choice') {
          bits.push(dim(' · '), mono(qid), ' = ', mono(a.choice), dim(` ${pct(a.probabilities?.[a.choice] ?? 0)}`))
        } else if (a.type === 'noul') {
          bits.push(dim(' · '), mono(qid), ' ', dim(a.noul >= 0.5 ? '真' : '假'), dim(` ${pct(a.noul)}`))
        } else if (a.type === 'score') {
          bits.push(dim(' · '), mono(qid), ' ', dim(`档位 ${a.score}`))
        }
      }
      return bits
    }
    case 'tool:call':
      return [mono(e.tool), dim('('), dim(e.input || '无输入'), dim(')')]
    case 'tool:result': {
      const body = String(e.output ?? '')
      const first = body.split('\n')[0].slice(0, 120)
      return [dim(`${body.length} 字符 · ${ms(e.ms)} · `), first]
    }
    case 'generate':
      return [
        h('span', { class: 'warn' }, e.kind),
        dim(` · ${ms(e.latencyMs)} · ${tokenText(e)} · 整个 loop 里唯一贵的一步`),
      ]
    case 'authorize':
      return [e.approved ? '已授权' : '已拒绝', dim(` · ${e.tool} · ${e.reason}`)]
    case 'audit':
      return [mono(e.record?.tool ?? ''), dim(` 记了审计留痕 · risk ${e.record?.risk ?? '?'}`)]
    case 'context': {
      const bits = contextBits(e)
      if (e.overRetain) {
        return [h('span', { class: 'warn' }, '⚠ 上下文压不到目标线'), dim(' · ' + bits.join(' · '))]
      }
      return [dim('工具证据被预算压过 · '), bits.join(' · ')]
    }
    case 'conversation': {
      const bits = conversationBits(e)
      if (e.overRetain) {
        return [h('span', { class: 'warn' }, '⚠ 上文压不到目标线'), dim(' · ' + bits.join(' · '))]
      }
      return [dim('上文被折叠过 · '), bits.join(' · ')]
    }
    default:
      return [e.type]
  }
}

// ── 账本 ──────────────────────────────────────────────────

const trajBody = $('traj-body')
const plotLanes = $('plot-lanes')
const plotEmpty = $('plot-empty')
const detailTitle = $('detail-title')
const detailLocation = $('detail-location')
const detailTabs = $('detail-tabs')
const detailBody = $('detail-body')

/** 时间轴上的条。`start`/`dur` 是累计毫秒 */
const spans = []
let clock = 0
let selected = null
let rowsAdded = 0

function addSpan(e, dur) {
  const meta = EVENT_META[e.type]
  if (!meta) return
  // 有耗时的按真实区间排；没有的给 1ms 标称宽度，且不推进时钟
  const width = dur.measured ? Math.max(dur.ms, 1) : 1
  spans.push({ lane: meta.lane, cls: meta.cls, start: clock, dur: width, event: e })
  if (dur.measured) clock += dur.ms
  renderPlot()
}

/**
 * 重画时间轴。
 *
 * 每次事件都整体重画：一次运行几十个条，重画比维护增量便宜得多，
 * 而且总时长还在长 —— 百分比定位必须跟着总长走。
 */
function renderPlot() {
  if (spans.length === 0) {
    plotEmpty.hidden = false
    plotLanes.replaceChildren()
    return
  }
  plotEmpty.hidden = true
  const total = clock > 0 ? clock : 1
  plotLanes.replaceChildren(
    ...spans.map((s) => {
      const el = h('div', {
        class: `span ${s.cls}`,
        style: `--lane:${s.lane};--left:${(s.start / total) * 100}%;--width:${Math.max(0.4, (s.dur / total) * 100)}%`,
        title: `${EVENT_META[s.event.type]?.label ?? s.event.type} · ${Math.round(s.dur)}ms`,
      })
      return el
    }),
  )
}

function addRow(e, dur, step) {
  const meta = EVENT_META[e.type]
  if (!meta) return

  // 步骤边界：行顶一条 2px 细线（CSS 的 ::before），左上角一个 8px 的步骤号徽章。
  // 第一次见到某个 step 时打上标记 —— DSH 的 data-turn-start 是同一个做法。
  const stepStart = e.type === 'decision' && step && step !== lastStep
  if (stepStart) lastStep = step

  const tr = h('tr', {
    tabindex: '0',
    ...(stepStart ? { 'data-step-start': 'true' } : {}),
    ...(meta.mono ? { 'data-mono': 'true' } : {}),
  })

  // 事件列：36px 左留白里放步骤号，76px 的槽里右对齐一个文字徽章
  tr.append(
    h(
      'td',
      { class: 'col-event' },
      stepStart ? h('span', { class: 'step-label' }, `step ${step}`) : null,
      h(
        'div',
        { class: 'event-inner' },
        h('span', { class: 'kind-slot' }, h('span', { class: `kind-tag ${meta.cls}` }, meta.label)),
        e.id || e.tool ? h('span', { class: 'event-id' }, e.id ?? e.tool) : null,
      ),
    ),
    h('td', { class: 'content-cell' }, h('span', { class: 'content-text' }, ...contentOf(e))),
  )

  const select = () => selectRow(tr, e)
  tr.addEventListener('click', select)
  tr.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault()
      select()
    }
  })

  trajBody.append(tr)
  rowsAdded++
}

function selectRow(tr, e) {
  if (selected) selected.removeAttribute('aria-selected')
  tr.setAttribute('aria-selected', 'true')
  selected = tr
  showDetail(e)
}

// ── 详情 ──────────────────────────────────────────────────

/** 右侧详情有两个 tab：概览看结论，决策帧看模型当时看到了什么 */
const DETAIL_TABS = ['概览', '决策帧']
let detailEvent = null
let detailTab = 0

function showDetail(e) {
  detailEvent = e
  detailTab = 0
  const meta = EVENT_META[e.type]
  detailTitle.textContent = `${meta?.label ?? e.type}${e.id ? ` ${e.id}` : ''}`
  detailLocation.textContent = e.tool ?? e.kind ?? ''
  detailTabs.replaceChildren(
    ...DETAIL_TABS.map((name, i) =>
      h(
        'button',
        {
          class: 'detail-tab',
          role: 'tab',
          'aria-selected': String(i === detailTab),
          onclick: () => {
            detailTab = i
            for (const [j, b] of [...detailTabs.children].entries()) {
              b.setAttribute('aria-selected', String(j === i))
            }
            renderDetail()
          },
        },
        name,
      ),
    ),
  )
  renderDetail()
}

const field = (k, v) => h('div', { class: 'field' }, h('div', { class: 'field-k' }, k), h('div', { class: 'field-v' }, v))

function renderDetail() {
  const e = detailEvent
  if (!e) return

  if (detailTab === 1) {
    // 决策帧：真正发给模型的东西。判定质量的上限由这一帧决定
    detailBody.replaceChildren(
      e.state
        ? field('发给模型的 state', h('pre', {}, JSON.stringify(e.state, null, 2)))
        : h('div', { class: 'detail-empty' }, '这个事件没有决策帧'),
    )
    return
  }

  const kids = []
  if (e.reason) kids.push(field('为什么是这个动作', e.reason))
  if (e.latencyMs != null) kids.push(field('判定耗时', ms(e.latencyMs)))

  // 每个问题：选项 + 概率条。这是这个界面最不该省略的东西
  for (const [qid, q] of Object.entries(e.questions ?? {})) {
    const a = e.answers?.[qid]
    if (!a) continue
    const qa = h('div', { class: 'qa' })
    qa.append(h('div', { class: 'q-instruction' }, h('span', { class: 'q-id' }, qid), q.instructions ?? ''))

    if (a.type === 'choice') {
      const opts = Object.entries(a.probabilities ?? {}).sort((x, y) => y[1] - x[1])
      for (const [name, p] of opts) {
        qa.append(
          h(
            'div',
            { class: `opt${name === a.choice ? ' picked' : ''}` },
            h('span', { class: 'opt-name', title: q.criteria?.[name] ?? '' }, name),
            h('span', { class: 'opt-track' }, h('span', { class: 'opt-fill', style: `width:${Math.max(1, p * 100)}%` })),
            h('span', { class: 'opt-pct' }, pct(p)),
          ),
        )
      }
    } else if (a.type === 'noul') {
      qa.append(
        h(
          'div',
          { class: 'noul-bar' },
          h('span', { class: 'noul-false', style: `flex:${Math.max(0.001, 1 - a.noul)}` }),
          h('span', { class: 'noul-true', style: `flex:${Math.max(0.001, a.noul)}` }),
        ),
        h(
          'div',
          { class: 'noul-legend' },
          h('span', {}, `假 ${pct(1 - a.noul)}`),
          h('span', {}, `真 ${pct(a.noul)}`),
        ),
      )
    } else if (a.type === 'score') {
      const legend = a.legend ?? {}
      for (const [k, p] of Object.entries(a.probabilities ?? {})) {
        qa.append(
          h(
            'div',
            { class: 'opt' },
            h('span', { class: 'opt-name', title: legend[k] ?? '' }, `${k} ${(legend[k] ?? '').slice(0, 8)}`),
            h('span', { class: 'opt-track' }, h('span', { class: 'opt-fill', style: `width:${Math.max(1, p * 100)}%` })),
            h('span', { class: 'opt-pct' }, pct(p)),
          ),
        )
      }
      qa.append(h('div', { class: 'scale' }, `期望档位 ${a.score}`))
    }
    kids.push(qa)
  }

  if (e.type === 'tool:call') kids.push(field('输入', h('pre', {}, e.input || '(无输入)')))
  if (e.type === 'tool:result') {
    kids.push(field('耗时', ms(e.ms)))
    kids.push(field('输出', h('pre', {}, String(e.output ?? ''))))
  }
  if (e.type === 'generate') {
    kids.push(field('耗时', ms(e.latencyMs)))
    kids.push(field('我们估的输入', `${e.estimatedInputTokens} token（上文 + 当前任务 + 证据）`))
    if (e.inputTokens > 0 || e.outputTokens > 0) {
      kids.push(field('provider 报的输入', `${e.inputTokens} token（含生成器内部的 system prompt）`))
      kids.push(field('provider 报的输出', `${e.outputTokens} token`))
      // 差值是 system prompt + 启发式偏差的和 —— 分开说，别假装能算清
      const gap = e.inputTokens - e.estimatedInputTokens
      kids.push(
        field(
          '差值',
          `${gap > 0 ? '+' : ''}${gap} token = system prompt + 启发式偏差（±30%），两者分不开`,
        ),
      )
    } else {
      kids.push(field('provider 用量', '未报 —— 这个后端不返回 usage，不是 0'))
    }
  }
  if (e.type === 'audit') {
    kids.push(field('目标', h('pre', {}, e.record?.target ?? '')))
    kids.push(field('风险档位', String(e.record?.risk ?? '?')))
  }
  if (e.type === 'authorize') kids.push(field('结果', e.approved ? '已授权' : '已拒绝'))
  if (e.type === 'context') {
    kids.push(field('证据总量', `${e.rawChars} → ${e.keptChars} 字符`))
    kids.push(field('剪了中间的', `${e.prunedCount} 条`))
    // **不是「丢弃」** —— 原文还在轨迹里，只是没进这次生成请求。
    // 这两个词的区别就是 surface 折叠存在的理由，界面上不能混。
    kids.push(field('折叠成摘要的', `${e.foldedCount ?? 0} 条（原文在轨迹里，一条不少）`))
    for (const f of e.folds ?? []) {
      kids.push(field(`折叠段 seq 0–${f.toSeq}`, `${f.foldedNodes} 条，省下 ${f.removedChars} 字符`))
    }
    if (e.overRetain) {
      kids.push(
        field(
          '⚠ 压不到目标线',
          '最后一条永远不会被丢，而它自己就超过目标 —— 这是如实报告，不是 bug',
        ),
      )
    }
  }
  if (e.type === 'conversation') {
    kids.push(field('上文总量', `${e.rawTurns} → ${e.keptTurns} 轮`))
    kids.push(field('字符', `${e.rawChars} → ${e.keptChars}`))
    kids.push(field('折叠成摘要的', `${e.foldedTurns} 轮（原文在服务端和轨迹里，一轮不少）`))
    // 留尾是**故意的**，所以超线可能是正确行为而不是失败 —— 这句要写出来，
    // 否则读的人分不清「预算画错了」和「这几轮本来就不该折」
    if (e.overRetain) {
      kids.push(
        field('⚠ 压不到目标线', '留尾那几轮永远不会被折，而它们自己就超过目标 —— 如实报告，不是 bug'),
      )
    }
  }

  detailBody.replaceChildren(...(kids.length ? kids : [h('div', { class: 'detail-empty' }, '没有更多内容')]))
}

/** 百分比，夹在 0–100 —— 用量条不该溢出，也不该因为除零变成 NaN */
function pctOf(n, total) {
  return Math.min(100, Math.max(0, (n / Math.max(1, total)) * 100))
}

/**
 * 一块预算的用法条。
 *
 * 条上那道刻度是**目标线**（越过去就会动手，动手就压到它以下）。
 * 刻度画在已用之上，所以「还要涨多少才越线」是看得见的长度差 ——
 * 而这个差值正是这一块存在的理由。
 */
function budgetRow(label, line) {
  /*
    三件事**互相独立**，所以一条条加上去，**不写成互斥分支**。

    以前这里是 `overRetain ? A : acted ? B : C` —— 而 `overRetain` 为真时
    会把「折前折后各多少字符」整句吞掉，偏偏那几个数正是理解「为什么压不
    下去」要看的（轨迹账本里有，这里没有，两处就对不上了）。互斥的写法
    让两个同时成立的事实只能显示一个。
  */
  const parts = []
  if (line.acted) {
    parts.push(`${line.note || '压过'} · ${line.rawChars} → ${line.keptChars} 字符`)
  } else {
    // 没动手时说「离触发线还有多远」—— 这句话以前在界面上根本不存在
    parts.push(`距触发线 ${line.triggerChars - line.rawChars} 字符`)
  }
  // 留尾是**故意的**，所以超线可能是正确行为而不是失败 —— 这句要说出来
  if (line.overRetain) parts.push(`⚠ 压完仍超目标线 ${line.retainChars}`)

  const note = h('div', { class: `budget-note${line.overRetain ? ' warn' : ''}` }, parts.join(' · '))

  return h(
    'div',
    { class: 'budget-row' },
    h(
      'div',
      { class: 'budget-head' },
      h('span', { class: 'budget-k' }, label),
      h('span', { class: 'budget-v' }, `${line.rawChars} / ${line.triggerChars} 字符`),
    ),
    h(
      'div',
      { class: 'budget-track' },
      h('div', {
        class: `budget-fill${line.acted ? ' acted' : ''}`,
        style: `width: ${pctOf(line.rawChars, line.triggerChars).toFixed(1)}%`,
      }),
      h('div', { class: 'budget-mark', style: `left: ${pctOf(line.retainChars, line.triggerChars).toFixed(1)}%` }),
    ),
    note,
  )
}

/**
 * 上下文账：两块预算 + 这次请求的 token 构成。
 *
 * **每轮都画，不管预算有没有动手。** 轨迹里那两个折叠事件只在真的折了
 * 才发，所以正常运行时这一块是唯一的预算视图 —— 它回答的是
 * 「离下一次折叠还有多远」。
 */
function budgetBlock(b, lastGen) {
  if (!b) {
    // 服务端异常那条路径发不出账目。**明说没有**，而不是画一堆 0 ——
    // 一堆 0 读起来像「什么都没用」，那是两回事（§8.10）。
    return h(
      'div',
      { class: 'budget-block' },
      h('div', { class: 'budget-title' }, '上下文账'),
      h('div', { class: 'budget-hint' }, '这次没跑到生成那一步，没有账目。'),
    )
  }

  const r = b.request
  const reported = lastGen && lastGen.inputTokens > 0 ? lastGen.inputTokens : 0

  return h(
    'div',
    { class: 'budget-block' },
    h('div', { class: 'budget-title' }, '上下文账'),
    h('div', { class: 'budget-hint' }, '两块独立预算（步 / 轮）。刻度是目标线，越过去就会折叠。'),
    budgetRow('证据 / 步', b.evidence),
    budgetRow('上文 / 轮', b.conversation),
    h(
      'div',
      {
        class: 'budget-compose',
        title:
          '差值 = 生成器内部的 system prompt + 启发式的偏差（实测那条启发式在代码上偏低约 25%）。' +
          '它的用途是看趋势，不要当成 system prompt 的大小。',
      },
      h('div', {}, '估 ', h('b', {}, String(r.total)), ` = 证据 ${r.evidence} + 上文 ${r.history} + 本句 ${r.task}`),
      h('div', { class: 'dim' }, reported > 0 ? `报 ${reported} · 差 ${reported - r.total}` : 'provider 用量未报'),
    ),
  )
}

// ═══════════════════════════════════════════════════════════
// 右栏：记账
//
// 它和左栏是一对 —— 左栏的 DECISION.md 说**应该**问哪些问题，
// 这里说**实际**花了多少。
// ═══════════════════════════════════════════════════════════

function renderEndStats(e) {
  const s = e.stats ?? {}

  // `?? 0` 会把「没有数据」显示成「0」：一次失败的运行看起来就和一次
  // 跑得很快的正常运行一样。缺数据就明说缺数据。
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? String(v) : '—')
  const dur = (v) => (typeof v === 'number' && Number.isFinite(v) ? ms(v) : '—')

  // 判定 : 模型 —— 走和消息页脚**同一个**函数（`ratioText`）。
  // 以前这里抄了一份，两处在 `decisions` 缺失时给出不同文本；
  // 那段「不在这里重算」的注释防住了除法，没防住格式化。
  const ratio = ratioText(s)

  $('side-stats').replaceChildren(
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '判定次数'), h('div', { class: 'metric-v decide' }, num(s.decisions))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '判定总耗时'), h('div', { class: 'metric-v decide' }, dur(s.decisionMs))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '模型调用'), h('div', { class: 'metric-v model' }, num(s.modelCalls))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '模型总耗时'), h('div', { class: 'metric-v model' }, dur(s.modelMs))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '工具执行'), h('div', { class: 'metric-v' }, num(state.tools))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '审计留痕'), h('div', { class: 'metric-v' }, num(s.audits))),
    h(
      'div',
      { class: 'ratio' },
      h('div', { class: 'ratio-v' }, ratio),
      h('div', { class: 'ratio-k' }, '判定 : 模型'),
    ),
    budgetBlock(e.budget, state.lastGenerate),
  )
}

// ═══════════════════════════════════════════════════════════
// 事件分发
//
// 一个事件做三件事：更新对话那行状态、往折叠的过程里加一行、
// 往轨迹里放一张卡。三条路互不依赖 —— 只有轨迹在前台时它也照常
// 构建，因为切过去要能立刻看到全貌，而不是从切换那一刻才开始记。
// ═══════════════════════════════════════════════════════════

let lastStep = 0
/** 本轮是否收到过 run:end。服务端正常收尾时关连接也会触发 onerror */
let sawEnd = false

/**
 * 正在**重放**历史事件，不是实时。
 *
 * 重放走的是**同一个 `onEvent`**（见 `restoreConversation`）—— 另写一份
 * 渲染必然会和实时那份分叉，而分叉的表现是「刷新前后的轨迹长得不一样」，
 * 不会有任何东西报错。所以只把两处**实时专有**的东西关掉：
 *
 *   · 「正在… · 已等待 Ns」那一行 —— 历史没有「正在」
 *   · `state.running` / 按钮禁用 —— 重放不是一次运行
 */
let replaying = false

function onEvent(e) {
  /*
    ══════════════════════════════════════════════════════════════
      ★ 增量**最先处理，而且处理完就 return** —— 它不是「事件」。
    ══════════════════════════════════════════════════════════════

    它不进 `current.events`（那是重放用的），不占过程列表的一行，不进右栏
    的账本。一次回答有**上千段**，混进去会把这轮那十几条**决定**淹掉 ——
    而「判定 13 次、模型 1 次」正是这个界面要给人看的唯一一件事。

    服务端那条路上它同样没进日志（见 `src/events.ts` 的 `GenerateDelta`），
    所以「实时看到的」和「刷新之后重放的」是同一个口径。

    `e.reset` 为真时丢掉已经收到的 —— 生成重试会重新开始（见 `llm.ts`）。
    渲染器自己也会发现「文本不是接着上一帧」，所以这里即使漏了也不会画错。
  */
  if (e.type === 'generate:delta') {
    if (!current || replaying) return
    if (e.reset) current.stream.reset()
    current.draft = (e.reset ? '' : current.draft) + e.text
    current.stream.update(current.draft)
    // 秒数从头数起：**它在动**，不是在卡住
    touch()
    followTail()
    return
  }

  if (current) current.events.push(e)

  // 对话侧：**追加**一行到过程列表，不覆盖。
  // 覆盖的话你只看得见最后一个，而「判定 13 次、模型 1 次」这件事
  // 恰恰只有一行行铺开才看得见 —— 那就是这个项目的全部主张。
  if (current && e.type !== 'run:end') {
    const row = processRow(e, durationOf(e))
    if (row) {
      current.list.append(row)
      current.rows++
      // 新行让内容变长了，刻度的比例位置跟着变
      followTail()
      syncTurnRail()
    }
    // ★ 「现在在干什么」**只由开始信号决定**：`phase` 和 `tool:call`。
    //   其余事件（`decision` / `generate` / `tool:result`）都是**做完之后**
    //   才发的，拿它们猜就会在最长的那一步上错得最久 —— 实测生成那 2 秒
    //   显示的是「正在判定」，因为最后一条事件是 `isDone`。
    if (!replaying) {
      if (e.type === 'phase') setPhase(phaseText(e), e.kind)
      else if (e.type === 'tool:call') setPhase(`正在执行 ${e.tool}`, 'tool')
      else touch()
    }
  }

  // 轨迹侧：账本加一行，时间轴加一条
  const dur = durationOf(e)
  addSpan(e, dur)
  addRow(e, dur, e.step)

  // 计数
  if (e.type === 'decision') {
    state.decisions++
    updateTally()
  } else if (e.type === 'tool:result') {
    state.tools++
    updateTally()
  } else if (e.type === 'generate') {
    state.models++
    state.lastGenerate = e
    updateTally()
  } else if (e.type === 'audit') {
    // 审计留痕也是「代码做的决定」，要进计数
    state.rules++
    updateTally()
  }

  if (e.type === 'run:end') {
    renderEndStats(e)
    finishAssistant(e.answer, e.stats, e.halt)
    if (!replaying) {
      state.running = false
      $('run').disabled = false
    }
  }
}

/**
 * 把轨迹侧清空，准备画一轮。
 *
 * 时间轴按**这一轮**的真实耗时排布，混进上一轮的条会让「橙色的那个最宽」
 * 这个结论失真。实时和重放共用这一个函数 —— 两处各写一遍就会分叉，
 * 而「刷新之后轨迹比例不一样」是没人会报错的那种错。
 */
function resetTrace() {
  trajBody.replaceChildren()
  spans.length = 0
  clock = 0
  rowsAdded = 0
  selected = null
  detailEvent = null
  renderPlot()
  // 计数也是「这一轮」的口径 —— 和上面那条理由一样
  state.decisions = state.models = state.tools = state.rules = 0
  lastStep = 0
  updateTally()
}

/** 一轮**没有跑完**（没有 `run:end`）。说清楚，不要留一个「正在…」在那里转 */
function abortedTurn() {
  if (!current) return
  current.finished = true
  stopTicker()
  current.el.removeChild(current.running)
  /*
    ★ 流过的正文**留着**。

    以前这里无条件写 `（这一轮没有跑完）`。没有流式时那句话是对的（本来就
    什么都没有），有流式之后它等于**把已经生成出来的几百字擦掉** ——
    而那些字是真的、是花过钱换来的，而且正是排查「它断在哪」唯一能看的
    东西。下面那句 `msg-foot` 已经说清楚了这一轮没跑完。
  */
  if (!current.answer.childNodes.length) current.answer.textContent = '（这一轮没有跑完）'
  current.foot.className = 'msg-foot failed'
  current.foot.replaceChildren(h('span', {}, '没有 run:end —— 这一轮中途断了，上面是它走到的位置'))
  followTail()
}

// ═══════════════════════════════════════════════════════════
// 运行
// ═══════════════════════════════════════════════════════════

function run(task) {
  if (state.running || !task) return
  state.running = true
  sawEnd = false

  if (turns.length === 0) chat.replaceChildren()
  userTurn(task)
  current = { task, events: [], lastEventAt: Date.now(), waitingOn: '正在判定', ...assistantTurn() }

  resetTrace()
  turns.push(current)

  $('run').disabled = true
  $('task').value = ''
  $('task').style.height = 'auto'

  startTicker()

  // ★ 带的是 **workspace id，不是路径** —— 客户端说不出一个服务端没登记过的
  //   目录。没选工作区时不带这个参数，服务端退回它的演示目录。
  const ws = workspaceId ? `&workspace=${encodeURIComponent(workspaceId)}` : ''
  const es = new EventSource(
    `/api/run?task=${encodeURIComponent(task)}&session=${encodeURIComponent(sessionId)}${ws}`,
  )

  es.onmessage = (msg) => {
    // 防御性：服务端会先发一行 `: connected` 注释，但按 SSE 规范注释行
    // 由解析器消费、**不会派发 message 事件**，正常收不到它。留着是为了
    // 万一中间有代理把它当数据转发时，不至于报一个假的解析错误。
    if (msg.data.startsWith(':')) return

    let e
    try {
      e = JSON.parse(msg.data)
    } catch (err) {
      console.error('事件解析失败', err, msg.data)
      return
    }
    if (e.type === 'run:start') return

    // ★ 收尾标记必须设在 `onEvent` **之前**。
    //
    //   以前这三句在同一个 `try` 里、且没有 `finally`：`onEvent` 在 `run:end`
    //   上抛异常就整块跳走 —— `sawEnd` 保持 false、`es.close()` 也不执行，
    //   于是 EventSource 还开着，服务端在 STREAM_LINGER_MS 之后关连接触发
    //   `onerror`，界面把一次**成功**的运行显示成「连接断开」。
    //
    //   分开之后两件事各自成立：「这次运行结束了没有」是**协议状态**，
    //   「渲染成功了没有」是**界面问题** —— 后者不该改写前者。
    if (e.type === 'run:end') {
      // 记下来：正常收尾时服务端关闭连接也会触发 onerror，那不是故障
      sawEnd = true
      es.close()
    }

    try {
      onEvent(e)
    } catch (err) {
      console.error('事件渲染失败', err, e)
    }
  }

  // 这里**故意**在第一次错误就 close()，和 EventSource 默认的自动重连相反：
  // `/api/run` 不是幂等的 —— 自动重连就是重新发一次 GET，服务端会**再跑一遍
  // agent**，也就是再花一次钱。所以断线时宁可放弃接收，也不能让它自己重来。
  es.onerror = () => {
    es.close()
    if (sawEnd) return // 正常收尾时服务端关连接也会走到这里，不是故障

    stopTicker()
    state.running = false
    $('run').disabled = false
    if (current && !current.finished) {
      current.stopped = true
      current.running.className = 'running-row dropped'
      current.running.replaceChildren(
        h('span', { class: 'pulse' }),
        '连接断开，已停止接收。服务端那次运行可能仍在继续（本版本没有取消机制）。',
        h(
          'button',
          {
            class: 'badge action',
            onclick: () => {
              // 重发一次是**再花一次钱**，所以让人自己按，不自动重试
              const t = current.task
              turns.pop()
              current.el.remove()
              run(t)
            },
          },
          '重跑',
        ),
      )
    }
  }
}

$('composer').addEventListener('submit', (ev) => {
  ev.preventDefault()
  run($('task').value.trim())
})

// Enter 发送，Shift+Enter 换行；输入框随内容长高（上限在 CSS 里）
$('task').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault()
    run($('task').value.trim())
  }
})
$('task').addEventListener('input', (ev) => {
  ev.target.style.height = 'auto'
  ev.target.style.height = `${ev.target.scrollHeight}px`
})

// ═══════════════════════════════════════════════════════════
// DECISION.md 面板
//
// 左边这一栏是「这个 agent 会问哪些问题」的规格。它和右边那栏
// 是一对：规格说应该问什么，记账说实际花了多少。
// ═══════════════════════════════════════════════════════════

async function loadSpec() {
  const box = $('spec')
  try {
    const res = await fetch('/api/spec')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const spec = await res.json()

    const kids = [h('div', { class: 'spec-headline' }, spec.headline)]

    // 有策略没编译出来时，先说这一句。
    // 编译不了的谓词会变成永不命中的规则 —— 也就是一条**空闸门**。
    // 不显示的话，上面那句汇总看起来完全正常（审计第十一轮 S3）。
    const broken = spec.blocks.filter((b) => (b.uncompiled ?? []).length > 0)
    if (broken.length > 0) {
      kids.push(
        h(
          'div',
          { class: 'spec-alert' },
          `⚠ ${broken.length} 个判定块有策略没编译出来，那些规则永远不会命中：` +
            broken.map((b) => `${b.id}（${b.uncompiled.join(' / ')}）`).join('；'),
        ),
      )
    }

    for (const b of spec.blocks) {
      const bad = (b.uncompiled ?? []).length > 0
      kids.push(
        h(
          'div',
          { class: `spec-block k-${b.kind}${bad ? ' broken' : ''}` },
          h(
            'div',
            {},
            h('span', { class: 'spec-id' }, b.id),
            h('span', { class: 'spec-kind' }, b.kind),
            b.gate ? h('span', { class: 'spec-gate' }, '闸门') : null,
            bad ? h('span', { class: 'spec-broken' }, `${b.uncompiled.length} 条未编译`) : null,
          ),
          b.when ? h('div', { class: 'spec-when' }, b.when) : null,
        ),
      )
    }
    // 解析这份文件时发现的问题必须显示出来 —— 一个被悄悄忽略的判定块
    // 会让 agent 安静地少问一个问题，界面上不显示就等于没人会知道
    for (const p of spec.problems ?? []) {
      kids.push(h('div', { class: 'spec-problem' }, `L${p.line}: ${p.message}`))
    }
    box.replaceChildren(...kids)
  } catch (err) {
    box.replaceChildren(h('div', { class: 'spec-problem' }, `读不到 DECISION.md：${err.message}`))
  }
}

// ═══════════════════════════════════════════════════════════
// 主题
// ═══════════════════════════════════════════════════════════

function applyTheme(dark) {
  document.body.toggleAttribute('data-jl-dark', dark)
  $('theme').textContent = dark ? '浅色' : '深色'
  try {
    localStorage.setItem('jl-theme', dark ? 'dark' : 'light')
  } catch {
    /* 隐私模式下 localStorage 会抛，忽略即可 —— 主题不是关键功能 */
  }
}

$('theme').addEventListener('click', () => applyTheme(!document.body.hasAttribute('data-jl-dark')))

/**
 * 新对话。
 *
 * 换一个会话 id 并清空两个视图。**这会真的断开服务端的上文** ——
 * 不是只把屏幕擦干净（那样等内核接上 history 之后就会变成一个谎：
 * 界面看着是新的，agent 却还记得）。
 */
$('new-session').addEventListener('click', () => void newSession())

/**
 * 开一段新对话。
 *
 * **清空服务端的那一份，而不是只换一个 id。** 只换 id 的话服务端那边
 * 对应的历史就断了（那也行），但这一段旧的会留在会话列表里 ——
 * 而用户点的是「新对话」，不是「把这段扔掉」。所以这里是真的清，
 * 要保留的走「新会话」那个 `+`（它只换 id，旧的留在列表里）。
 */
async function newSession() {
  if (state.running) return
  try {
    await fetch(`/api/session?id=${encodeURIComponent(sessionId)}`, { method: 'POST' })
  } catch (err) {
    // 清不掉服务端的就**不要**换 id —— 否则界面是新的、上下文还在，
    // 两边不一致比不清更糟。说清楚并停在这里。
    console.error('清空会话失败', err)
    return
  }
  await startFreshSession()
}

/** 换一个 id 并清空两个视图。**旧的那些留在服务端**，列表里还看得见 */
async function startFreshSession() {
  sessionId = newSessionId()
  remember('jl-session', sessionId)

  turns.length = 0
  current = null
  chat.replaceChildren(h('div', { class: 'empty' }, '新对话。说点什么。'))
  turnMarks.length = 0
  railEls = []
  syncTurnRail()
  trajBody.replaceChildren(h('tr', {}, h('td', { colspan: '2', class: 'empty' }, '还没有跑过。')))
  spans.length = 0
  clock = 0
  selected = null
  detailEvent = null
  renderPlot()
  state.lastGenerate = null
  state.decisions = state.models = state.tools = state.rules = 0
  updateTally()
  $('side-stats').replaceChildren(h('div', { class: 'empty' }, '运行结束后显示'))
  detailBody.replaceChildren(h('div', { class: 'detail-empty' }, '点左边任意一行'))
  detailTitle.textContent = '详情'
  detailLocation.textContent = ''
  detailTabs.replaceChildren()
  await loadSessions()
}

function restore(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    /* 同上 */
    return null
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* 同上 */
  }
}

// ═══════════════════════════════════════════════════════════
// 工作区与会话：左栏那一列
//
// 两级，抄 DSH 的侧边栏（`ui-sidebar` + `ui-workspace`）：
//
//     工作区   一个登记过的目录。跑任务时发的是它的 **id**，不是路径
//     会话     一段对话。挂在哪个工作区下，看它自己的 `cwd`
//
// **一段会话属于哪个工作区，是查出来的**：会话记着自己的 `cwd`，工作区
// 记着自己的 `path`，服务端把两边对上（它同时认得两个表），界面拿到的是
// `session.workspaceId`。
//
// ★ 用查的而不是存一份 `sessionIds[]`，因为存的那份会漂移 —— 而漂移的后果
//   是「这个会话从工作区里消失了」，且没人会报错。DSH 存那份是为了手动
//   排序，我们还没有排序，所以不存。
//
// （上一轮这里写着「会话的 cwd 没回传到界面」—— 那句是错的。`cwd` 一直在
//   `/api/sessions` 的返回里，只是界面没有用它。）
// ═══════════════════════════════════════════════════════════

function newSessionId() {
  // 不追求唯一性，只要同一台机器上两次对话不撞 —— 服务端还会做 LRU 淘汰
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
}

let sessionId = restore('jl-session') || newSessionId()
remember('jl-session', sessionId)

// ═══════════════════════════════════════════════════════════
// 工作区：选一个目录，然后登记它
// ═══════════════════════════════════════════════════════════

/** 当前工作区 id。`null` = 没选，服务端用它自己的演示目录 */
let workspaceId = restore('jl-workspace')
/** 选目录对话框当前停在哪一层 */
let browseAt = null

/**
 * 一次 API 调用。
 *
 * **错误一律是 JSON**（服务端那边统一过），所以这里能拿到 `error` 原文。
 * 以前有的端点返纯文本、有的返 JSON，统一按 JSON 解析就会在出错时炸在
 * 一个语法错误上，看不到真正的原因。
 */
async function api(path, opts) {
  const res = await fetch(path, opts)
  let body = {}
  try {
    body = await res.json()
  } catch {
    // 吞的是「响应体不是 JSON」。**不当成成功** —— 下面 `res.ok` 为假时
    // 会带着状态码报出来，而不是把一个空对象当结果用
  }
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`)
  return body
}

/** 「3 分钟前」这种。会话列表按时间排，读的人要能一眼比出来 */
function ago(ms) {
  if (!ms) return ''
  const s = Math.max(0, (Date.now() - ms) / 1000)
  if (s < 60) return '刚刚'
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`
  return `${Math.floor(s / 86400)} 天前`
}

/**
 * 读工作区，并决定**当前选哪个**。
 *
 * ★ 顺序抄 DSH：**显式选中的 → 当前会话的 → 最近活动的**。
 *
 * 第三档不是锦上添花，是修一个实测出来的毛病：以前这里选不出来时
 * `workspaceId` 就是 `null`，而 `renderWorkspace` 那一栏在 `null` 时**也**
 * 显示「演示目录」（自动登记的那个工作区刚好叫这个名字）—— 于是
 * 「没选」和「选了演示目录」两种状态在屏幕上一模一样，而会话列表按
 * `workspaceId` 筛，两者筛出来的东西完全不同（实测：16 条 vs 1 条）。
 *
 * 现在只要有工作区就一定会选中一个，`null` 只可能出现在**一个都没登记**
 * 的时候，而那种时候屏幕上写的是「未选择工作区」，不会再撞名。
 */
async function loadWorkspaces() {
  try {
    const { workspaces } = await api('/api/workspaces')
    state.workspaces = workspaces
    // 记住的那个可能已经被移除了 —— 别拿着一个服务端不认识的 id 一路 404
    if (workspaceId && !workspaces.some((w) => w.id === workspaceId)) workspaceId = null
  } catch (err) {
    console.error('读工作区失败', err)
    state.workspaces = []
  }

  /*
    ★ **别落在一个用不了的工作区上。**

    实测：localStorage 里记着一个工作区，它的目录被删了而登记还在 ——
    于是每次打开界面都选中它，会话列表空着（那些会话属于别的工作区），
    一发消息就是 400「目录不在了」。**那个 400 是对的，错的是选中了它。**

    所以「记住的那个」只有在**还能用**的时候才作数；不能用就按下面这个
    顺序找一个能用的。
  */
  const remembered = state.workspaces.find((w) => w.id === workspaceId)
  if (!remembered?.exists) {
    const cur = state.sessions.find((s) => s.id === sessionId)
    const curWs = state.workspaces.find((w) => w.id === cur?.workspaceId)
    workspaceId =
      (curWs?.exists ? curWs.id : null) ?? // ① 当前会话那个（还活着的话）
      state.workspaces.find((w) => w.exists)?.id ?? // ② 最近活动的、还能用的
      remembered?.id ?? // ③ 一个能用的都没有 —— 用它，但界面会标「目录不在了」
      null
  }
  remember('jl-workspace', workspaceId ?? '')
  renderWorkspace()
}

function renderWorkspace() {
  const cur = state.workspaces.find((w) => w.id === workspaceId)
  // 「没选」要**看起来就是没选** —— 以前这里回退成「演示目录」，而自动登记
  // 的那个工作区也叫这个名字，两种状态就撞在一起了（见 `loadWorkspaces`）
  $('ws-name').textContent = cur ? cur.title : '未选择工作区'
  $('ws-path').textContent = cur
    ? `${cur.path}${cur.exists ? '' : '  ⚠ 目录不在了'}`
    : '会话不会归到任何工作区下'
  renderWsMenu()
}

/**
 * 从列表里移除一个工作区。
 *
 * **只从列表里去掉，不碰磁盘上的目录** —— 服务端那一侧的 `remove` 也是
 * 这么做的，按钮的 title 写明了，免得有人以为它在删文件。
 */
async function forgetWorkspace(id) {
  try {
    await api(`/api/workspaces?id=${encodeURIComponent(id)}`, { method: 'DELETE' })
  } catch (err) {
    console.error('移除工作区失败', err)
    return
  }
  if (workspaceId === id) {
    workspaceId = null
    remember('jl-workspace', '')
  }
  await loadWorkspaces()
}

function renderWsMenu() {
  const menu = $('ws-menu')
  if (menu.hidden) return
  menu.replaceChildren(
    ...(state.workspaces.length
      ? state.workspaces.map((w) =>
          h(
            'div',
            { class: `ws-item${w.id === workspaceId ? ' on' : ''}${w.exists ? '' : ' gone'}` },
            h(
              'button',
              {
                class: 'ws-item-main',
                type: 'button',
                onclick: () => {
                  workspaceId = w.id
                  remember('jl-workspace', w.id)
                  menu.hidden = true
                  $('ws-current').setAttribute('aria-expanded', 'false')
                state.showOthers = false
                renderWorkspace()
                  void loadSessions()
                },
              },
              h('span', { class: 'ws-item-name' }, w.title),
              h('span', { class: 'ws-item-path' }, w.path),
              // 目录不在了就**当场说**，不要等选中之后跑起来才报错
              ...(w.exists ? [] : [h('span', { class: 'ws-item-gone' }, '⚠ 目录不在了')]),
            ),
            h('button', {
              class: 'ws-item-del',
              type: 'button',
              title: '从列表里移除（不删目录）',
              'aria-label': '从列表里移除',
              onclick: (ev) => {
                ev.stopPropagation()
                void forgetWorkspace(w.id)
              },
            }, '×'),
          ),
        )
      : [h('div', { class: 'ws-empty' }, '还没有登记过目录')]),
  )
}

$('ws-current').addEventListener('click', () => {
  const menu = $('ws-menu')
  menu.hidden = !menu.hidden
  $('ws-current').setAttribute('aria-expanded', String(!menu.hidden))
  renderWsMenu()
})

$('ws-add').addEventListener('click', () => void openPicker())

// ── 会话列表 ───────────────────────────────────────────────

async function loadSessions() {
  try {
    const { sessions } = await api('/api/sessions')
    state.sessions = sessions
  } catch (err) {
    console.error('读会话列表失败', err)
    state.sessions = []
  }
  renderSessions()
}

/**
 * 一段会话一行。
 */
function sessionRow(s) {
  return h(
    'div',
    {
      class: `sess-item${s.id === sessionId ? ' on' : ''}`,
      role: 'button',
      tabindex: '0',
      onclick: () => void openSession(s.id),
      onkeydown: (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') void openSession(s.id)
      },
    },
    h(
      'div',
      { class: 'sess-body' },
      h('span', { class: 'sess-title' }, s.firstPrompt || '（空会话）'),
      h('span', { class: 'sess-meta' }, `${s.turns} 轮 · ${ago(s.updatedAt)}${s.skipped ? ' · ⚠ 有读不出的行' : ''}`),
    ),
    h('button', {
      class: 'sess-del',
      type: 'button',
      title: '删除这个会话',
      'aria-label': '删除这个会话',
      onclick: (ev) => {
        ev.stopPropagation()
        void removeSession(s.id)
      },
    }, '×'),
  )
}

/**
 * 会话列表。**按当前工作区筛**，不属于它的收在「其他」里。
 *
 * ★ 以前这里是**全部会话平铺**，和上面选的工作区没有任何关系。实测那意味着
 *   16 条混在一起，其中大多数属于已经不存在的临时目录 —— 你选了工作区 A，
 *   屏幕上的会话却是 B、C、D 的。**界面上两处互相矛盾**，而没有任何东西会说。
 *
 * 「其他」那一组**不删掉**是有意的：那里面是你真的聊过的内容，只是它的目录
 * 没被登记（被删了，或者以前是每次重启都换的临时目录）。藏起来比混在一起更糟。
 */
function renderSessions() {
  const box = $('sess-list')
  if (!state.sessions.length) {
    box.replaceChildren(h('div', { class: 'empty' }, '还没有会话'))
    return
  }

  // 没选工作区时，「我的」= 也没有工作区的那些（目录没被登记）
  const key = (s) => s.workspaceId ?? null
  const mine = state.sessions.filter((s) => key(s) === (workspaceId ?? null))
  const others = state.sessions.filter((s) => key(s) !== (workspaceId ?? null))

  const kids = mine.length
    ? mine.map(sessionRow)
    : [h('div', { class: 'empty' }, workspaceId ? '这个工作区下还没有会话' : '没有未归属的会话')]

  if (others.length) {
    kids.push(
      h(
        'button',
        {
          class: 'sess-more',
          type: 'button',
          onclick: () => {
            state.showOthers = !state.showOthers
            renderSessions()
          },
        },
        `${state.showOthers ? '▾' : '▸'} 其他 ${others.length} 个会话`,
      ),
    )
    if (state.showOthers) kids.push(...others.map(sessionRow))
  }

  box.replaceChildren(...kids)
}

/**
 * 切到另一个会话。
 *
 * **连服务端的正文一起读回来** —— 会话是落盘的，刷新页面也该看得到。
 * 以前这里只换 id，界面永远是空的，而服务端一直存着。
 */
async function openSession(id) {
  if (state.running || id === sessionId) return
  sessionId = id
  remember('jl-session', id)

  // ★ **跟着切工作区。** 不切的话屏幕上两处是矛盾的：对话属于目录 A，
  //   而上面那一栏写着 B —— 接着发一句就会在 B 里跑，和眼前这段对话无关。
  const s = state.sessions.find((x) => x.id === id)
  const wsId = s?.workspaceId ?? null
  if (wsId !== (workspaceId ?? null)) {
    workspaceId = wsId
    remember('jl-workspace', wsId ?? '')
    renderWorkspace()
  }

  await restoreConversation()
  renderSessions()
}

async function removeSession(id) {
  if (state.running) return
  try {
    await api(`/api/sessions?id=${encodeURIComponent(id)}`, { method: 'DELETE' })
  } catch (err) {
    console.error('删会话失败', err)
    return
  }
  // 删掉的正是当前这段 → 换一段新的，否则界面还显示着一段已经不存在的对话
  if (id === sessionId) await startFreshSession()
  else await loadSessions()
}

/**
 * 把那一段对话读回来铺到屏幕上 —— **连过程一起**。
 *
 * ★ 走的是**和实时同一个 `onEvent`**（`replaying` 那个开关就是为它加的）。
 *   另写一份渲染的话，刷新前后的轨迹迟早长得不一样，而没有任何东西会报错。
 *
 * ★ 轨迹只画**最后一轮**：实时那边每开一轮都会 `resetTrace()`，因为时间轴
 *   是按这一轮的真实耗时排的，混进上一轮会让「哪个最宽」这个结论失真。
 *   聊天侧**每一轮的过程都在**（每轮有自己的过程列表）。
 *
 * 没有 `run:end` 的那一轮**照样显示**，并写明它没跑完 —— 崩掉的那次恰恰
 * 最需要看它走到哪一步。
 */
async function restoreConversation() {
  turns.length = 0
  current = null
  chat.replaceChildren()
  turnMarks.length = 0
  railEls = []
  syncTurnRail()

  let runs = []
  try {
    const r = await api(`/api/session?id=${encodeURIComponent(sessionId)}`)
    runs = r.runs ?? []
  } catch (err) {
    console.error('读会话失败', err)
  }

  if (!runs.length) {
    chat.replaceChildren(h('div', { class: 'empty' }, '说点什么，它会边判定边做。'))
    return
  }

  replaying = true
  try {
    for (const [i, r] of runs.entries()) {
      if (i === runs.length - 1) resetTrace()
      userTurn(r.task)
      current = { task: r.task, events: [], lastEventAt: Date.now(), ...assistantTurn() }
      turns.push(current)
      for (const e of r.events) onEvent(e)
      if (r.answer === undefined) abortedTurn()
    }
  } finally {
    replaying = false
  }
  current = null
  followTail()
}

$('sess-new').addEventListener('click', () => void startFreshSession())

// ═══════════════════════════════════════════════════════════
// 选目录
//
// 形状抄 DSH 的 browse 后端（`ui-directory-picker-browse`）：
//
//   · **只列目录** —— 这是选目录，不是文件浏览器
//   · **面包屑每一节都可跳** —— 它是导航，不是装饰
//   · **每个条目带绝对路径**，界面不自己拼 `../` —— 拼路径是路径穿越
//     最容易发生的地方，把它全留在服务端
//   · **可以就地新建子目录** —— 没有它，用户得切到终端建完再回来
//
// 服务端那边有一道守卫：绑到非回环地址时这些端点直接拒绝（能浏览本机
// 文件系统 = 能在这台机器上跑命令）。所以这里要**如实显示**那条 403。
// ═══════════════════════════════════════════════════════════

async function openPicker() {
  $('picker-mask').hidden = false
  $('picker-note').textContent = ''
  $('picker-note').className = 'picker-note'
  $('picker-new').value = ''
  await browseTo(null) // 不传路径 = 家目录，起点不该是服务端的 cwd
}

function closePicker() {
  $('picker-mask').hidden = true
  browseAt = null
}

$('picker-x').addEventListener('click', closePicker)
// 点遮罩关掉。**只在点遮罩本身时** —— 点对话框内部不该关
$('picker-mask').addEventListener('click', (ev) => {
  if (ev.target === $('picker-mask')) closePicker()
})
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && !$('picker-mask').hidden) closePicker()
})

/** 跳到某一层。`path` 为 null 就是家目录 */
async function browseTo(path) {
  let d
  try {
    d = await api(`/api/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`)
  } catch (err) {
    // 403 是服务端绑到了非回环地址 —— 那条消息本身就是给用户看的，
    // 原样显示，不要换一句「出错了」
    $('picker-note').textContent = err.message
    $('picker-note').className = 'picker-note err'
    return
  }
  browseAt = d
  renderPicker()
}

function renderPicker() {
  if (!browseAt) return
  const d = browseAt

  // 面包屑：从根到当前，每一节都是跳转目标
  const crumbs = []
  for (const [i, c] of d.crumbs.entries()) {
    if (i) crumbs.push(h('span', { class: 'crumb-sep' }, '/'))
    crumbs.push(
      h('button', {
        class: `crumb${i === d.crumbs.length - 1 ? ' last' : ''}`,
        type: 'button',
        onclick: () => void browseTo(c.path),
      }, c.name),
    )
  }
  // 「家」是常用的落点，单独给一个 —— 从根一层层点回来太费事
  if (d.path !== d.home) {
    crumbs.unshift(
      h('button', { class: 'crumb', type: 'button', onclick: () => void browseTo(d.home) }, '家'),
      h('span', { class: 'crumb-sep' }, '·'),
    )
  }
  $('picker-crumbs').replaceChildren(...crumbs)

  $('picker-entries').replaceChildren(
    ...(d.entries.length
      ? d.entries.map((e) =>
          h(
            'button',
            {
              class: `entry${e.hidden ? ' hidden-dir' : ''}`,
              type: 'button',
              // 双击进目录太隐蔽，单击就进 —— 选是底部那个按钮的事
              onclick: () => void browseTo(e.path),
            },
            h('span', { class: 'entry-icon' }, e.hidden ? '·' : '▸'),
            h('span', { class: 'entry-name' }, e.name),
          ),
        )
      : [h('div', { class: 'ws-empty' }, '这一层没有子目录')]),
  )

  /*
    ★ **路径永远显示，截断警告是加在它后面的第二句。**

    以前这两件事是二选一：截断了就只显示警告，于是**你在哪一层看不见了**
    —— 而 `/tmp` 刚好就有 500 多个子目录（实测），也就是说这个「少见情形」
    在最常见的一个目录上就会发生。两个都该说的时候，一个不能挤掉另一个。
  */
  const note = $('picker-note')
  note.replaceChildren(
    h('span', {}, d.path),
    ...(d.truncated
      ? [h('span', { class: 'warn' }, '　子目录太多，只列了前 500 个（按名字排序）—— 里面没有就往下新建')]
      : []),
  )
  note.className = 'picker-note'
}

async function mkDir() {
  const name = $('picker-new').value.trim()
  if (!name || !browseAt) return
  let created
  try {
    const r = await api('/api/workspaces', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ path: browseAt.path, newDir: name }),
    })
    created = r.newDirPath
  } catch (err) {
    $('picker-note').textContent = err.message
    $('picker-note').className = 'picker-note err'
    return
  }
  $('picker-new').value = ''
  /*
    ★ **新建之后直接进那一层。**

    这一步和 DSH 不一样，是**有意的**。它有两栏（列表 + 选中项），所以
    新建之后停在父目录、把新目录当成**选中项**是对的 —— 列表和选中是
    两件事。我们只有一栏，「列出的那一层就是你要选的那一层」是唯一的
    规则；照搬 DSH 会造出一个「选中了但没进去」的状态，而那个状态在这
    一栏里没法表达。

    进入之后「新建 → 用这个目录」是两步；停在父目录的话要三步。
    建一个目录的意图，绝大多数时候就是「我要用它」。
  */
  await browseTo(created ?? browseAt.path)
}

$('picker-mk').addEventListener('click', () => void mkDir())
$('picker-new').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') {
    ev.preventDefault()
    void mkDir()
  }
})

/** 用当前这一层。登记是**幂等**的，所以同一个目录再登记一次不会出问题 */
async function useCurrentDir() {
  if (!browseAt) return
  let r
  try {
    r = await api('/api/workspaces', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ path: browseAt.path }),
    })
  } catch (err) {
    $('picker-note').textContent = err.message
    $('picker-note').className = 'picker-note err'
    return
  }
  workspaceId = r.workspace.id
  remember('jl-workspace', workspaceId)
  closePicker()
  await loadWorkspaces()
  await loadSessions()
}

$('picker-pick').addEventListener('click', () => void useCurrentDir())

// 跟随系统，除非用户手动选过
const saved = restore('jl-theme')
applyTheme(saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches)
const savedView = restore('jl-view')
selectView(VIEWS.includes(savedView) ? savedView : 'chat')
loadSpec()
renderLegend()
syncToBottom()
/*
  顺序有依赖，所以不能并行：
    ① 先读会话 —— 选哪个工作区要看当前这段会话属于哪个
    ② 再读工作区 —— 它据此决定 `workspaceId`
    ③ 最后恢复对话
  （以前三个都是 `void` 并行，于是 `loadWorkspaces` 读 `state.sessions`
    时它还是空的，第三档选取永远走不到。）
*/
void (async () => {
  await loadSessions()
  await loadWorkspaces()
  await restoreConversation()
  renderSessions()
})()
