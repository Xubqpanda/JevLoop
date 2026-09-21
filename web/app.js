/**
 * JevLoop · 界面
 *
 * 两个视图，**同一串事件**：
 *
 *     对话   你说了什么、它答了什么。运行中只有一行状态，不刷屏
 *     轨迹   每一次判定、每一次工具调用、每一次生成，一张卡一张卡铺开
 *
 * 分开不是为了藏东西，是因为它们回答的问题不同。对话回答「它做成了吗」，
 * 轨迹回答「它凭什么这么判断」。切过去看不需要重新跑 —— 事件全在内存里，
 * 轨迹是**回放**出来的。
 *
 * 视觉主张只有一条：**橙色（模型生成）必须稀少且扎眼。**
 * 判定是蓝的、工具是绿的、规则是灰的 —— 它们铺满整页也不心疼，
 * 而橙色出现一次就该让人意识到「这里花钱了」。
 */

// ═══════════════════════════════════════════════════════════
// DOM 小工具（不引框架：这个界面只有两种交互，画卡片和切视图）
// ═══════════════════════════════════════════════════════════

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

const state = { decisions: 0, models: 0, tools: 0, rules: 0, running: false }

/**
 * 对话记录。
 *
 * 每一轮 = 用户一句话 + 一次运行。事件留在轮里，所以切到轨迹
 * 可以直接回放，不需要重跑 —— 重跑就是重新花钱。
 */
const turns = []
let current = null

const $ = (id) => document.getElementById(id)
const chat = $('chat')
const timeline = $('timeline')

function updateTally() {
  $('t-decide').textContent = state.decisions
  $('t-model').textContent = state.models
  $('t-tool').textContent = state.tools
  $('t-rules').textContent = state.rules
}

// ═══════════════════════════════════════════════════════════
// 视图切换
// ═══════════════════════════════════════════════════════════

const VIEWS = ['chat', 'trace']
const NOTES = {
  chat: '',
  trace: '同一串事件，不折叠',
}

function selectView(id) {
  for (const v of VIEWS) {
    const on = v === id
    $(`tab-${v}`).setAttribute('aria-selected', String(on))
    $(`view-${v}`).hidden = !on
  }
  $('tab-note').textContent = NOTES[id] ?? ''
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

function userMessage(text) {
  chat.append(
    h(
      'div',
      { class: 'msg msg-user' },
      h('div', { class: 'msg-role' }, '你'),
      h('div', { class: 'msg-body' }, text),
    ),
  )
}

function assistantMessage() {
  const status = h('div', { class: 'msg-status decide' }, h('span', { class: 'dot' }), '开始判定…')
  const body = h('div', { class: 'msg-body' }, '')
  const foot = h('div', { class: 'msg-foot' })
  const el = h('div', { class: 'msg msg-assistant' }, h('div', { class: 'msg-role' }, 'JevLoop'), status, body, foot)
  chat.append(el)
  el.scrollIntoView({ block: 'end', behavior: 'smooth' })
  return { el, status, body, foot, finished: false }
}

/** 运行中的一行状态。颜色说明**现在在花哪种钱** */
function setStatus(kind, text) {
  if (!current) return
  current.status.className = `msg-status ${kind}`
  current.status.replaceChildren(h('span', { class: 'dot' }), text)
}

function finishAssistant(text, stats, halt) {
  if (!current) return
  current.finished = true
  current.el.removeChild(current.status)
  current.body.textContent = text || '（没有回答）'
  if (halt === 'error') current.el.classList.add('error')

  const s = stats ?? {}
  const r = s.ratio
  const ratio =
    typeof r !== 'number' ? '?'
    : !Number.isFinite(r) ? `${s.decisions ?? '?'} : 0`
    : `${r.toFixed(1)} : 1`
  current.foot.replaceChildren(
    h('span', {}, `${s.decisions ?? '?'} 判定`),
    h('span', {}, `${s.modelCalls ?? '?'} 模型`),
    h('span', {}, `${ratio} 判定:模型`),
    h('span', {}, `停于 ${halt}`),
  )
  current.el.scrollIntoView({ block: 'end', behavior: 'smooth' })
}

// ═══════════════════════════════════════════════════════════
// 渲染：轨迹
//
// 这是界面的主体。一张判定卡要回答四个问题：
//   问了什么 · 有哪些选项 · 各多少概率 · 命中了哪条策略
// ═══════════════════════════════════════════════════════════

function renderDecision(e) {
  state.decisions++
  updateTally()

  const card = h('div', { class: 'card decide' })
  card.append(
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge kind-decide' }, '判定'),
      h('span', { class: 'card-id' }, e.id),
      h('span', { class: `badge action${/escalate/.test(e.action) ? ' warn' : ''}` }, e.action),
      h('span', { class: 'card-lat' }, ms(e.latencyMs)),
    ),
  )
  card.append(h('div', { class: 'card-reason' }, e.reason))

  for (const [qid, q] of Object.entries(e.questions ?? {})) {
    const a = e.answers?.[qid]
    if (!a) continue
    const qa = h('div', { class: 'qa' })
    qa.append(h('div', { class: 'q-instruction' }, h('span', { class: 'q-id' }, qid), q.instructions ?? ''))

    if (a.type === 'choice') {
      // 按概率降序 —— 模型认定的最优解应该在最上面
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
      // noul 只看一端就够 —— 画成两段，右边是「真」的概率
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
      qa.append(h('div', { class: 'card-reason' }, `期望档位 ${a.score}`))
    }
    card.append(qa)
  }

  // 原始决策帧 —— 折叠着。想看「模型到底看到了什么」时才展开，
  // 但它的存在本身很重要：判定质量的上限由这一帧决定。
  card.append(
    h(
      'details',
      { class: 'raw' },
      h('summary', {}, '决策帧（真正发给模型的东西）'),
      h('pre', {}, JSON.stringify(e.state, null, 2)),
    ),
  )
  return card
}

function renderToolCall(e) {
  return h(
    'div',
    { class: 'card tool' },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge kind-tool' }, '工具'),
      h('span', { class: 'card-id' }, e.tool),
    ),
    h('div', { class: 'io' }, e.input || '(无输入)'),
  )
}

function renderToolResult(e) {
  state.tools++
  updateTally()
  const body = String(e.output ?? '')
  const shown = body.length > 700 ? `${body.slice(0, 700)}…` : body
  return h(
    'div',
    { class: 'card tool' },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge kind-tool' }, '工具结果'),
      h('span', { class: 'card-id' }, e.tool),
      h('span', { class: 'card-lat' }, ms(e.ms)),
    ),
    h('div', { class: 'io' }, shown),
  )
}

function renderGenerate(e) {
  state.models++
  updateTally()
  return h(
    'div',
    { class: 'card model' },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge kind-model' }, '模型生成'),
      h('span', { class: 'card-id' }, e.kind),
      h('span', { class: 'card-lat' }, `${ms(e.latencyMs)} · ${e.tokens} tokens`),
    ),
    h('div', { class: 'card-reason' }, '整个 loop 里唯一贵的一步'),
  )
}

function renderAuthorize(e) {
  return h(
    'div',
    { class: 'card authorize' },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge action warn' }, e.approved ? '已授权' : '已拒绝'),
      h('span', { class: 'card-id' }, e.tool),
    ),
    h('div', { class: 'card-reason' }, e.reason),
  )
}

function renderAudit(e) {
  // 审计留痕也是「代码做的决定」，要进计数
  state.rules++
  updateTally()
  const r = e.record ?? {}
  return h(
    'div',
    { class: 'card audit' },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge kind-decide' }, '审计留痕'),
      h('span', { class: 'card-id' }, r.tool ?? ''),
      h('span', { class: 'card-lat' }, `risk ${r.risk ?? '?'}`),
    ),
    h('div', { class: 'io' }, r.target ?? ''),
  )
}

function renderEndStats(e) {
  const s = e.stats ?? {}

  // `?? 0` 会把「没有数据」显示成「0」：一次失败的运行看起来就和一次
  // 跑得很快的正常运行一样。缺数据就明说缺数据。
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? String(v) : '—')
  const dur = (v) => (typeof v === 'number' && Number.isFinite(v) ? ms(v) : '—')

  // 判定 : 模型 —— **直接用服务端的值**，不在这里重算。
  // `meter.ts` 已经算过一遍，两处实现会在「没有模型调用」这个边界上分叉。
  const r = s.ratio
  const ratio =
    typeof r !== 'number' ? '?'
    : !Number.isFinite(r) ? `${num(s.decisions)} : 0`
    : `${r.toFixed(1)} : 1`

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
  )
}

// ═══════════════════════════════════════════════════════════
// 事件分发
//
// 一个事件做两件事：更新对话那行状态，往轨迹里放一张卡。
// 两条路互不依赖 —— 只有轨迹在前台时它也照常构建，因为切过去
// 要能立刻看到全貌，而不是从切换那一刻才开始记。
// ═══════════════════════════════════════════════════════════

let lastStep = 0

function onEvent(e) {
  if (current) current.events.push(e)

  // 对话侧：只留一行状态，不刷屏
  switch (e.type) {
    case 'decision':
      setStatus('decide', `判定 ${e.id} → ${e.action}`)
      break
    case 'tool:call':
      setStatus('tool', `调用 ${e.tool}`)
      break
    case 'tool:result':
      setStatus('tool', `${e.tool} 返回 ${String(e.output ?? '').length} 字符`)
      break
    case 'authorize':
      setStatus('decide', e.approved ? `${e.tool} 已授权` : `${e.tool} 被拒绝`)
      break
    case 'audit':
      setStatus('decide', `${e.record?.tool ?? ''} 记了审计留痕`)
      break
    case 'generate':
      setStatus('model', `生成中…（${e.kind}）`)
      break
    default:
      break
  }

  // 轨迹侧：一张卡
  if (e.step && e.step !== lastStep && e.type === 'decision') {
    timeline.append(h('div', { class: 'step-head' }, `step ${e.step}`))
    lastStep = e.step
  }

  const card =
    e.type === 'decision' ? renderDecision(e)
    : e.type === 'tool:call' ? renderToolCall(e)
    : e.type === 'tool:result' ? renderToolResult(e)
    : e.type === 'generate' ? renderGenerate(e)
    : e.type === 'authorize' ? renderAuthorize(e)
    : e.type === 'audit' ? renderAudit(e)
    : null

  if (card) timeline.append(card)

  if (e.type === 'run:end') {
    renderEndStats(e)
    finishAssistant(e.answer, e.stats, e.halt)
    state.running = false
    $('run').disabled = false
    $('run').textContent = '发送'
  }
}

// ═══════════════════════════════════════════════════════════
// 运行
// ═══════════════════════════════════════════════════════════

function run(task) {
  if (state.running || !task) return
  state.running = true
  state.decisions = state.models = state.tools = state.rules = 0
  lastStep = 0
  updateTally()

  if (turns.length === 0) chat.replaceChildren()
  userMessage(task)
  current = { task, events: [], ...assistantMessage() }
  turns.push(current)

  $('run').disabled = true
  $('run').textContent = '运行中…'
  $('task').value = ''

  const es = new EventSource(`/api/run?task=${encodeURIComponent(task)}`)

  es.onmessage = (msg) => {
    // 防御性：服务端会先发一行 `: connected` 注释，但按 SSE 规范注释行
    // 由解析器消费、**不会派发 message 事件**，正常收不到它。留着是为了
    // 万一中间有代理把它当数据转发时，不至于报一个假的解析错误。
    if (msg.data.startsWith(':')) return
    try {
      const e = JSON.parse(msg.data)
      if (e.type === 'run:start') return
      onEvent(e)
      if (e.type === 'run:end') es.close()
    } catch (err) {
      console.error('事件解析失败', err, msg.data)
    }
  }

  // 这里**故意**在第一次错误就 close()，和 EventSource 默认的自动重连相反：
  // `/api/run` 不是幂等的 —— 自动重连就是重新发一次 GET，服务端会**再跑一遍
  // agent**，也就是再花一次钱。所以断线时宁可放弃接收，也不能让它自己重来。
  es.onerror = () => {
    es.close()
    state.running = false
    $('run').disabled = false
    $('run').textContent = '发送'
    if (current && !current.finished) {
      setStatus('error', '连接断开；服务端那一次运行可能仍在继续（本版本没有取消机制）')
    }
  }
}

$('composer').addEventListener('submit', (ev) => {
  ev.preventDefault()
  run($('task').value.trim())
})

// Enter 发送，Shift+Enter 换行；输入框随内容长高
$('task').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault()
    run($('task').value.trim())
  }
})
$('task').addEventListener('input', (ev) => {
  ev.target.style.height = 'auto'
  ev.target.style.height = `${Math.min(ev.target.scrollHeight, 180)}px`
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
    for (const b of spec.blocks) {
      kids.push(
        h(
          'div',
          { class: `spec-block k-${b.kind}` },
          h(
            'div',
            {},
            h('span', { class: 'spec-id' }, b.id),
            h('span', { class: 'spec-kind' }, b.kind),
            b.gate ? h('span', { class: 'spec-gate' }, '闸门') : null,
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

function restore(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    /* 同上 */
    return null
  }
}

// 跟随系统，除非用户手动选过
const saved = restore('jl-theme')
applyTheme(saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches)
selectView(VIEWS.includes(restore('jl-view')) ? restore('jl-view') : 'chat')
loadSpec()
