/**
 * JevLoop · 界面
 *
 * 消费 `server.ts` 推来的事件流，把每个事件渲染成时间线上的一张卡片。
 *
 * 整个界面的视觉主张只有一条：**橙色（模型生成）必须稀少且扎眼。**
 * 判定是蓝的、工具是绿的、规则是灰的 —— 它们铺满整页也不心疼，
 * 而橙色出现一次就该让人意识到「这里花钱了」。
 */

// ═══════════════════════════════════════════════════════════
// DOM 小工具（不引框架：这个界面只有一种交互，就是「画卡片」）
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

const pct = (n) => `${(n * 100).toFixed(n >= 0.995 || n <= 0.005 ? 0 : 1)}%`
const ms = (n) => `${Math.round(n)}ms`

// ═══════════════════════════════════════════════════════════
// 状态
// ═══════════════════════════════════════════════════════════

const state = { decisions: 0, models: 0, tools: 0, rules: 0, running: false }

const $ = (id) => document.getElementById(id)
const timeline = $('timeline')

function updateTally() {
  $('t-decide').textContent = state.decisions
  $('t-model').textContent = state.models
  $('t-tool').textContent = state.tools
}

// ═══════════════════════════════════════════════════════════
// 渲染：判定
//
// 这是界面的主体。一张卡要回答四个问题：
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
    qa.append(
      h('div', { class: 'q-instruction' }, h('span', { class: 'q-id' }, qid), q.instructions ?? ''),
    )

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

// ═══════════════════════════════════════════════════════════
// 渲染：工具 / 模型 / 授权 / 审计
// ═══════════════════════════════════════════════════════════

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
    { class: `card tool${body.startsWith('错误：') ? ' error' : ''}` },
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
  state.rules++
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

function renderEnd(e) {
  const s = e.stats ?? {}
  const ratio = s.modelCalls ? (s.decisions / s.modelCalls).toFixed(1) : String(s.decisions)

  $('side-stats').replaceChildren(
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '判定次数'), h('div', { class: 'metric-v decide' }, s.decisions ?? 0)),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '判定总耗时'), h('div', { class: 'metric-v decide' }, ms(s.decisionMs ?? 0))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '模型调用'), h('div', { class: 'metric-v model' }, s.modelCalls ?? 0)),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '模型总耗时'), h('div', { class: 'metric-v model' }, ms(s.modelMs ?? 0))),
    h('div', { class: 'metric' }, h('div', { class: 'metric-k' }, '工具执行'), h('div', { class: 'metric-v' }, state.tools)),
    h(
      'div',
      { class: 'ratio' },
      h('div', { class: 'ratio-v' }, `${ratio} : 1`),
      h('div', { class: 'ratio-k' }, '判定 : 模型'),
    ),
  )

  return h(
    'div',
    { class: `card end${e.halt === 'error' ? ' error' : ''}` },
    h(
      'div',
      { class: 'card-head' },
      h('span', { class: 'badge action' }, '结束'),
      h('span', { class: 'card-id' }, e.halt),
      h('span', { class: 'card-lat' }, `${e.steps} steps`),
    ),
    e.answer ? h('div', { class: 'answer' }, e.answer) : null,
  )
}

// ═══════════════════════════════════════════════════════════
// 事件分发
// ═══════════════════════════════════════════════════════════

let lastStep = 0

function onEvent(e) {
  // 每个 step 前面加一个分隔标题 —— 时间线要能看出「第几步」
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
    : e.type === 'run:end' ? renderEnd(e)
    : null

  if (card) {
    card.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    timeline.append(card)
  }
  if (e.type === 'run:end') {
    state.running = false
    $('run').disabled = false
    $('run').textContent = '运行'
  }
}

// ═══════════════════════════════════════════════════════════
// 运行
// ═══════════════════════════════════════════════════════════

function run() {
  if (state.running) return
  state.running = true
  state.decisions = state.models = state.tools = 0
  lastStep = 0
  updateTally()

  timeline.replaceChildren(h('div', { class: 'empty' }, '连接中…'))
  $('side-stats').replaceChildren(h('div', { class: 'empty' }, '运行结束后显示'))
  $('run').disabled = true
  $('run').textContent = '运行中…'

  const task = $('task').value.trim()
  const es = new EventSource(`/api/run?task=${encodeURIComponent(task)}`)

  es.onmessage = (msg) => {
    if (msg.data.startsWith(':')) return
    try {
      const e = JSON.parse(msg.data)
      if (e.type === 'run:start') {
        timeline.replaceChildren()
        return
      }
      onEvent(e)
      if (e.type === 'run:end') es.close()
    } catch (err) {
      console.error('事件解析失败', err, msg.data)
    }
  }

  es.onerror = () => {
    es.close()
    state.running = false
    $('run').disabled = false
    $('run').textContent = '运行'
    if (!timeline.querySelector('.card')) {
      timeline.replaceChildren(h('div', { class: 'empty' }, '连接断开，检查服务端日志'))
    }
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

$('run').addEventListener('click', run)
$('theme').addEventListener('click', () => applyTheme(!document.body.hasAttribute('data-jl-dark')))

// 跟随系统，除非用户手动选过
let saved = null
try {
  saved = localStorage.getItem('jl-theme')
} catch {
  /* 同上 */
}
applyTheme(saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches)
