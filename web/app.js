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

const VIEWS = ['chat', 'trace']

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

function userTurn(text) {
  chat.append(h('div', { class: 'msg-user' }, h('div', { class: 'bubble' }, text)))
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
  el.scrollIntoView({ block: 'end', behavior: 'smooth' })
  return { el, list, running, answer, foot, rows: 0, finished: false, stopped: false }
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
      detail.push(`${e.kind} · ${e.tokens} tokens · 唯一贵的一步`)
      break
    case 'authorize':
      detail.push(`${e.tool} · ${e.approved ? '已授权' : '已拒绝'} · ${e.reason ?? ''}`)
      break
    case 'audit':
      detail.push(`${e.record?.tool ?? ''} · risk ${e.record?.risk ?? '?'}`)
      break
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
const STALL_AFTER_MS = 6000
let ticker = null

function startTicker() {
  stopTicker()
  ticker = setInterval(() => {
    if (!current || current.finished || current.stopped) return
    const waited = Date.now() - (current.lastEventAt ?? Date.now())
    const secs = (waited / 1000).toFixed(0)
    const stalled = waited >= STALL_AFTER_MS
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

function setWaiting(text) {
  if (!current) return
  current.waitingOn = text
  current.lastEventAt = Date.now()
  if (!current.stopped) {
    current.running.className = 'running-row'
    current.running.replaceChildren(h('span', { class: 'pulse' }), `${text}  ·  已等待 0s`)
  }
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
  current.answer.textContent = text || '（没有回答）'

  const s = stats ?? {}
  const ratio = ratioText(s)
  current.foot.className = `msg-foot${halt === 'error' ? ' failed' : ''}`
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
  // 工具那两行用等宽字体：它们的内容是命令和输出，不是句子
  'tool:call': { label: '工具', cls: 'tool', lane: LANE.tool, mono: true },
  'tool:result': { label: '工具结果', cls: 'tool', lane: LANE.tool, mono: true },
  generate: { label: '模型', cls: 'model', lane: LANE.model },
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
        dim(` · ${ms(e.latencyMs)} · ${e.tokens} tokens · 整个 loop 里唯一贵的一步`),
      ]
    case 'authorize':
      return [e.approved ? '已授权' : '已拒绝', dim(` · ${e.tool} · ${e.reason}`)]
    case 'audit':
      return [mono(e.record?.tool ?? ''), dim(` 记了审计留痕 · risk ${e.record?.risk ?? '?'}`)]
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
    kids.push(field('tokens', String(e.tokens)))
  }
  if (e.type === 'audit') {
    kids.push(field('目标', h('pre', {}, e.record?.target ?? '')))
    kids.push(field('风险档位', String(e.record?.risk ?? '?')))
  }
  if (e.type === 'authorize') kids.push(field('结果', e.approved ? '已授权' : '已拒绝'))

  detailBody.replaceChildren(...(kids.length ? kids : [h('div', { class: 'detail-empty' }, '没有更多内容')]))
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

function onEvent(e) {
  if (current) current.events.push(e)

  // 对话侧：**追加**一行到过程列表，不覆盖。
  // 覆盖的话你只看得见最后一个，而「判定 13 次、模型 1 次」这件事
  // 恰恰只有一行行铺开才看得见 —— 那就是这个项目的全部主张。
  if (current && e.type !== 'run:end') {
    const row = processRow(e, durationOf(e))
    if (row) {
      current.list.append(row)
      current.rows++
      row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
    if (e.type === 'tool:call') setWaiting(`正在执行 ${e.tool}`)
    else if (e.type === 'generate') setWaiting('正在生成回答')
    else setWaiting('正在判定')
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
    updateTally()
  } else if (e.type === 'audit') {
    // 审计留痕也是「代码做的决定」，要进计数
    state.rules++
    updateTally()
  }

  if (e.type === 'run:end') {
    renderEndStats(e)
    finishAssistant(e.answer, e.stats, e.halt)
    state.running = false
    $('run').disabled = false
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
  sawEnd = false
  updateTally()

  if (turns.length === 0) chat.replaceChildren()
  userTurn(task)
  current = { task, events: [], lastEventAt: Date.now(), waitingOn: '正在判定', ...assistantTurn() }

  // 轨迹侧清空。时间轴按**这一轮**的真实耗时排布，混进上一轮的条
  // 会让「橙色的那个最宽」这个结论失真
  trajBody.replaceChildren()
  spans.length = 0
  clock = 0
  rowsAdded = 0
  selected = null
  detailEvent = null
  renderPlot()
  turns.push(current)

  $('run').disabled = true
  $('task').value = ''
  $('task').style.height = 'auto'

  startTicker()

  const es = new EventSource(`/api/run?task=${encodeURIComponent(task)}&session=${encodeURIComponent(sessionId)}`)

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
$('new-session').addEventListener('click', async () => {
  if (state.running) return
  try {
    await fetch(`/api/session?id=${encodeURIComponent(sessionId)}`, { method: 'POST' })
  } catch (err) {
    // 清不掉服务端的就**不要**换 id —— 否则界面是新的、上下文还在，
    // 两边不一致比不清更糟。说清楚并停在这里。
    console.error('清空会话失败', err)
    return
  }
  sessionId = newSessionId()
  remember('jl-session', sessionId)

  turns.length = 0
  current = null
  chat.replaceChildren(h('div', { class: 'empty' }, '新对话。说点什么。'))
  trajBody.replaceChildren(h('tr', {}, h('td', { colspan: '2', class: 'empty' }, '还没有跑过。')))
  spans.length = 0
  clock = 0
  selected = null
  detailEvent = null
  renderPlot()
  state.decisions = state.models = state.tools = state.rules = 0
  updateTally()
  $('side-stats').replaceChildren(h('div', { class: 'empty' }, '运行结束后显示'))
  detailBody.replaceChildren(h('div', { class: 'detail-empty' }, '点左边任意一行'))
  detailTitle.textContent = '详情'
  detailLocation.textContent = ''
  detailTabs.replaceChildren()
})

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
// 会话
//
// 一个 id 代表一段对话。服务端按这个 id 记住每一轮的问答 ——
// 现在**只记不读**（生成器还看不到上文，见 server.ts 的 MEMORY_WIRED）。
//
// 界面这边先把它接上：「新对话」换一个 id，服务端那边对应的历史就断了。
// 这样等内核接上 history，不需要再动这里。
// ═══════════════════════════════════════════════════════════

function newSessionId() {
  // 不追求唯一性，只要同一台机器上两次对话不撞 —— 服务端还会做 LRU 淘汰
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
}

let sessionId = restore('jl-session') || newSessionId()
remember('jl-session', sessionId)

// 跟随系统，除非用户手动选过
const saved = restore('jl-theme')
applyTheme(saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches)
const savedView = restore('jl-view')
selectView(VIEWS.includes(savedView) ? savedView : 'chat')
loadSpec()
