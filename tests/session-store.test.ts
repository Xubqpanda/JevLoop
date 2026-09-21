/**
 * 会话存储的回归测试。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里全是**磁盘行为**，`tsc` 通过不代表写对了。重点测五类：
 *
 *   · **日志 = 事件流**，轮次从 `run:start` / `run:end` 推导 —— 不要另存摘要。
 *     这是这个模块存在的理由：刷新之后轨迹还得看得见。
 *   · **布局按 cwd 分项目**（仿 DSH），所以光有 id 要找得到、撞了要报。
 *   · **撕裂的尾行不能毁掉整个会话**，但必须**被报出来**，不能悄悄少两轮。
 *   · **会话 id 是不可信输入**，它会变成路径段。`?session=../../x` 要被挡住。
 *   · **旧布局要搬过来**，搬不动就留在原地 —— 不删、不假装成功。
 *
 * @module JevLoop/session-store.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { appendFile, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { SessionStore, assertSessionId, SESSION_FORMAT_VERSION } from '../src/session-store.ts'
import { migrateLegacyLayout } from '../src/session-migrate.ts'
import { encodeSegment, projectKey } from '../src/session-path.ts'

/** 造一段事件流。和内核发的是同一个形状（`events.ts`） */
const ev = {
  start: (task: string) => ({ type: 'run:start', task, cwd: '/x', at: 0 }),
  decide: (n: number) => ({
    type: 'decision',
    step: n,
    id: 'loop.needsTool',
    action: 'use_tool',
    reason: 'r',
    latencyMs: 3,
    provider: 'p',
    degraded: false,
    escalate: false,
    state: { task: 'x' },
    questions: {},
    answers: {},
  }),
  end: (answer: string, halt = 'task_done') => ({ type: 'run:end', halt, steps: 1, answer, stats: {} }),
}

async function withStore(fn: (store: SessionStore, root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'jevstore-'))
  try {
    await fn(new SessionStore(root), root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

/** 一个会话跑一轮：start → 两个判定 → end */
async function oneRun(store: SessionStore, id: string, task: string, answer: string, cwd = '/proj', run = 0) {
  await store.append(id, run, ev.start(task), { cwd })
  await store.append(id, run, ev.decide(1))
  await store.append(id, run, ev.decide(2))
  await store.append(id, run, ev.end(answer))
}

// ═══════════════════════════════════════════════════════════
// 布局：仿 DSH，按 cwd 分项目
// ═══════════════════════════════════════════════════════════

test('按 cwd 分项目：<root>/<projectKey>/<id>/log.jsonl', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '问', '答', '/home/x/proj')

    const projects = await readdir(root)
    assert.deepEqual(projects, [projectKey('/home/x/proj')])
    const sessions = await readdir(join(root, projects[0]!))
    assert.deepEqual(sessions, ['s1'])
    assert.deepEqual(await readdir(join(root, projects[0]!, 's1')), ['log.jsonl'])
  })
})

test('没有 cwd 的会话落 _no-cwd', async () => {
  await withStore(async (store, root) => {
    await store.append('s1', 0, ev.start('问'))
    assert.deepEqual(await readdir(root), ['_no-cwd'])
  })
})

test('同一个 id 在别的 cwd 下继续跑，**不会被搬走**', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '问一', '答一', '/a')
    // 第二次带一个不同的 cwd —— 会话属于它**出生**时那个项目
    await oneRun(store, 's1', '问二', '答二', '/b', 1)

    assert.deepEqual(await readdir(root), [projectKey('/a')], '位置没变')
    assert.equal((await store.load('s1')).length, 2, '两轮都在')
  })
})

test('只用 id 找得到会话 —— 路径按 cwd 分层之后，得遍历项目目录', async () => {
  await withStore(async (store) => {
    await oneRun(store, 'a', '甲的会话', '答', '/proj/one')
    await oneRun(store, 'b', '乙的会话', '答', '/proj/two')
    assert.equal((await store.load('a'))[0]!.task, '甲的会话')
    assert.equal((await store.load('b'))[0]!.task, '乙的会话')
  })
})

test('★ 同一个 id 出现在两个项目目录下 → 抛，不挑一个', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 'dup', '原始', '答', '/a')
    // 手工造一份「我们不认识的副本」（拷贝、恢复、人搬过）
    const copy = join(root, projectKey('/b'), encodeSegment('dup'))
    await mkdir(copy, { recursive: true })
    await writeFile(join(copy, 'log.jsonl'), '{"kind":"header","v":2,"id":"dup","createdAt":1,"cwd":"/b"}\n', 'utf8')

    await assert.rejects(
      () => store.load('dup'),
      /同时出现在 2 个项目目录下/,
      '挑一个等于静默丢掉另一个，而用户以为自己在看全部',
    )
  })
})

// ═══════════════════════════════════════════════════════════
// 轮次从事件里推导
// ═══════════════════════════════════════════════════════════

test('轮次的任务和回答**从事件里读**，不另存字段', async () => {
  await withStore(async (store) => {
    await oneRun(store, 's1', '问的是什么', '答的是什么')
    const [r] = await store.load('s1')
    assert.equal(r!.task, '问的是什么', '来自 run:start')
    assert.equal(r!.answer, '答的是什么', '来自 run:end')
    assert.equal(r!.halt, 'task_done')
    assert.equal(r!.events.length, 4, '四个事件一个不少')
  })
})

test('★ 事件是完整存下来的 —— 判定帧、概率、门限都在', async () => {
  await withStore(async (store) => {
    await store.append('s1', 0, ev.start('问'), { cwd: '/p' })
    await store.append('s1', 0, {
      type: 'decision',
      step: 1,
      id: 'loop.pickTool',
      state: { task: '问', already_done: 'nothing yet' },
      questions: { tool: { type: 'choice', instructions: 'pick', criteria: { read_file: 'x' } } },
      answers: { tool: { type: 'choice', choice: 'read_file', probabilities: { read_file: 0.91 }, confidence: 0.2 } },
      action: 'call',
      reason: 'top >= 0.6 → call',
      latencyMs: 264,
      provider: 'jev',
      degraded: false,
      escalate: false,
    })

    const [r] = await store.load('s1')
    const d = r!.events.find((e) => (e as { type: string }).type === 'decision') as Record<string, unknown>
    // ★ 这三样才是这个项目要展示的东西（「它凭什么这么判断」）
    assert.deepEqual(d.state, { task: '问', already_done: 'nothing yet' })
    const answers = d.answers as { tool: { probabilities: { read_file: number } } }
    assert.equal(answers.tool.probabilities.read_file, 0.91)
    assert.equal(d.reason, 'top >= 0.6 → call')
  })
})

test('一轮没跑完时**没有 answer** —— 那就是「它没跑完」的信号', async () => {
  await withStore(async (store) => {
    await store.append('s1', 0, ev.start('崩掉的那一轮'), { cwd: '/p' })
    await store.append('s1', 0, ev.decide(1))
    const [r] = await store.load('s1')
    assert.equal(r!.task, '崩掉的那一轮')
    assert.equal(r!.answer, undefined, '没有 run:end')
    assert.equal(r!.events.length, 2, '但过程留着 —— 崩掉的那次最需要看它走到哪一步')
  })
})

test('多轮各自归位', async () => {
  await withStore(async (store) => {
    await oneRun(store, 's1', '第一问', '第一答', '/p', 0)
    await oneRun(store, 's1', '第二问', '第二答', '/p', 1)
    const runs = await store.load('s1')
    assert.deepEqual(runs.map((r) => r.task), ['第一问', '第二问'])
    assert.deepEqual(runs.map((r) => r.answer), ['第一答', '第二答'])
  })
})

test('load 带上限时只回最近几轮，但**盘上一条都不少**', async () => {
  await withStore(async (store) => {
    for (let i = 0; i < 20; i++) await oneRun(store, 's1', `问${i}`, `答${i}`, '/p', i)
    const last5 = await store.load('s1', 5)
    assert.equal(last5.length, 5)
    assert.equal(last5[0]!.task, '问15', '要的是最近的')
    assert.equal((await store.load('s1')).length, 20, '盘上必须还是 20 轮')
  })
})

test('读一个不存在的会话返回空数组，不抛', async () => {
  await withStore(async (store) => {
    assert.deepEqual(await store.load('never'), [])
  })
})

// ═══════════════════════════════════════════════════════════
// 坏行：撕裂、认不出、序号跳了
// ═══════════════════════════════════════════════════════════

test('撕裂的尾行不会毁掉整个会话，但会被**报出来**', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '第一问', '第一答', '/p')
    await oneRun(store, 's1', '第二问', '第二答', '/p', 1)
    // 写到一半断电
    await appendFile(join(root, projectKey('/p'), 's1', 'log.jsonl'), '{"kind":"event","run":2,"at":3,"e":{"type":"deci', 'utf8')

    assert.deepEqual((await store.load('s1')).map((r) => r.task), ['第一问', '第二问'], '前两轮完好')
    assert.equal((await store.list())[0]!.skipped, 1, '跳过一行就要报 1')
  })
})

test('认不出的 kind 算跳过，不算轮次', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '问', '答', '/p')
    await appendFile(join(root, projectKey('/p'), 's1', 'log.jsonl'), '{"kind":"telemetry","x":1}\n', 'utf8')
    assert.equal((await store.load('s1')).length, 1)
    assert.equal((await store.list())[0]!.skipped, 1)
  })
})

test('★ 轮次序号跳了要报 —— 少一轮和「本来就只有这些」看起来一模一样', async () => {
  await withStore(async (store) => {
    await oneRun(store, 's1', '第 0 轮', '答', '/p', 0)
    await oneRun(store, 's1', '第 2 轮', '答', '/p', 2) // 跳过 1
    const runs = await store.load('s1')
    assert.equal(runs.length, 2, '两轮都在（洞被去掉）')
    assert.equal((await store.list())[0]!.skipped, 1, '跳过的那一轮要报出来')
  })
})

test('空行不计入 skipped —— 文件末尾一定有换行', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '问', '答', '/p')
    await appendFile(join(root, projectKey('/p'), 's1', 'log.jsonl'), '\n\n', 'utf8')
    assert.equal((await store.list())[0]!.skipped, 0)
  })
})

// ═══════════════════════════════════════════════════════════
// 旧布局：搬过来，不是拒绝
//
// ★ 迁移**不在 store 里** —— 它在 `session-migrate.ts`，由服务端启动时
//   调一次。放在 store 里会同时踩两个坑：那是「部署」的事不是「每次读写」
//   的事；以及 §11 不许同层互相依赖（两个都是 L2）。
//   所以这里直接测那个模块，而不是绕道 store。
// ═══════════════════════════════════════════════════════════

test('★ v1 的扁平文件会被搬进新布局，内容一字不改', async () => {
  await withStore(async (store, root) => {
    // 造一个旧布局的文件：<root>/old.jsonl
    const legacy =
      `{"kind":"header","v":1,"id":"old","createdAt":100,"cwd":"/legacy/proj"}\n` +
      `{"kind":"turn","task":"旧的一问","answer":"旧的一答","at":200}\n`
    await writeFile(join(root, 'old.jsonl'), legacy, 'utf8')
    await migrateLegacyLayout(root, () => {})

    const runs = await store.load('old')
    assert.equal(runs.length, 1)
    assert.equal(runs[0]!.task, '旧的一问')
    assert.equal(runs[0]!.answer, '旧的一答')
    assert.deepEqual(runs[0]!.events, [], 'v1 没有过程 —— **不假装它有**')

    // 已经在新位置了，旧文件不在了
    const moved = join(root, projectKey('/legacy/proj'), 'old', 'log.jsonl')
    assert.equal(await readFile(moved, 'utf8'), legacy, '内容一字不改')
    assert.deepEqual(
      (await readdir(root)).filter((n) => n.endsWith('.jsonl')),
      [],
      '旧位置不该再留着',
    )
  })
})

test('读不出 cwd 的旧文件落到 _no-cwd —— 内容不丢', async () => {
  const root = await mkdtemp(join(tmpdir(), 'jevstore-mig-'))
  try {
    const legacy = '{"kind":"turn","task":"没有 header 的一轮","answer":"答","at":1}\n'
    await writeFile(join(root, 'noheader.jsonl'), legacy, 'utf8')
    await migrateLegacyLayout(root, () => {})

    const runs = await new SessionStore(root).load('noheader')
    assert.equal(runs.length, 1, '**读得到** —— 只是不知道它属于哪个项目')
    assert.equal(runs[0]!.task, '没有 header 的一轮')
    assert.equal(await readFile(join(root, '_no-cwd', 'noheader', 'log.jsonl'), 'utf8'), legacy, '内容一字不改')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('迁移是**幂等**的，跑两次不会出问题', async () => {
  const root = await mkdtemp(join(tmpdir(), 'jevstore-mig2-'))
  try {
    const legacy = '{"kind":"header","v":1,"id":"a","createdAt":1,"cwd":"/p"}\n{"kind":"turn","task":"问","answer":"答","at":2}\n'
    await writeFile(join(root, 'a.jsonl'), legacy, 'utf8')
    const said: string[] = []
    await migrateLegacyLayout(root, (l) => said.push(l))
    await migrateLegacyLayout(root, (l) => said.push(l))
    assert.equal(said.filter((l) => l.includes('搬到')).length, 1, '只搬一次')
    assert.equal(await readFile(join(root, projectKey('/p'), 'a', 'log.jsonl'), 'utf8'), legacy)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('迁移会**说出来**搬了什么', async () => {
  const root = await mkdtemp(join(tmpdir(), 'jevstore-mig3-'))
  try {
    await writeFile(join(root, 'x.jsonl'), '{"kind":"header","v":1,"id":"x","createdAt":1,"cwd":"/q"}\n', 'utf8')
    const said: string[] = []
    await migrateLegacyLayout(root, (l) => said.push(l))
    assert.equal(said.length, 1, '搬了一个就报一行 —— 静默搬文件不该发生')
    assert.match(said[0]!, /会话 x 从 v1 布局搬到 v2/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// 列表与删除
// ═══════════════════════════════════════════════════════════

test('列表按最近活动倒序，带上 cwd 和第一句', async () => {
  await withStore(async (store) => {
    await oneRun(store, 'old', '很久以前', 'a', '/p')
    await new Promise((r) => setTimeout(r, 5))
    await oneRun(store, 'new', '刚刚', 'b', '/q')

    const list = await store.list()
    assert.deepEqual(list.map((s) => s.id), ['new', 'old'])
    assert.equal(list[0]!.firstPrompt, '刚刚')
    assert.equal(list[0]!.cwd, '/q', 'cwd 从 header 读原值')
    assert.equal(list[0]!.turns, 1)
  })
})

test('目录不存在时 list 返回空数组', async () => {
  const root = join(tmpdir(), `jevstore-missing-${Date.now()}`)
  assert.deepEqual(await new SessionStore(root).list(), [])
})

test('list 不做正文搬运 —— 摘要里没有 answer 也没有 events', async () => {
  await withStore(async (store) => {
    await oneRun(store, 's1', '问', '很长的一段回答', '/p')
    const [s] = await store.list()
    assert.equal('answer' in s!, false)
    assert.equal('events' in s!, false)
  })
})

test('remove 删掉的是整个会话目录', async () => {
  await withStore(async (store, root) => {
    await oneRun(store, 's1', '问', '答', '/p')
    assert.equal(await store.remove('s1'), true)
    assert.deepEqual(await store.list(), [])
    assert.equal((await readdir(join(root, projectKey('/p')))).includes('s1'), false, '目录也清掉')
    assert.equal(await store.remove('s1'), false, '再删一次是 false')
  })
})

// ═══════════════════════════════════════════════════════════
// 路径与 id
// ═══════════════════════════════════════════════════════════

test('encodeSegment 是**可逆**的 —— 不同输入不会编成同一个', () => {
  assert.equal(encodeSegment('abc-DEF_123'), 'abc-DEF_123')
  assert.equal(encodeSegment('a/b'), 'a~002Fb')
  assert.notEqual(encodeSegment('a/b'), encodeSegment('a-b'), '这是它比「把 / 换成 -」强的地方')
  assert.equal(encodeSegment('.'), '~002E')
  assert.equal(encodeSegment('..'), '~002E~002E')
  assert.throws(() => encodeSegment(''))
})

test('projectKey 是**有损**的可读编码 —— 这一点写在注释里了', () => {
  assert.equal(projectKey('/home/x/proj'), '--home-x-proj--')
  assert.equal(projectKey('/a//b'), '--a-b--', '连续分隔符压成一个')
  // 有损：这两个落进同一层。**可接受**，因为身份由 header 里的 cwd 原值决定
  assert.equal(projectKey('/a/b'), projectKey('/a-b'))
  assert.equal(projectKey('/'), '--root--')
  assert.ok(projectKey(`/${'x'.repeat(400)}`).length <= 260, '超长要截断')
})

test('会话 id 变成路径段 —— 穿越必须被挡住', () => {
  for (const bad of ['../../etc/passwd', 'a/b', 'a\\b', '..', '', 'x'.repeat(65), 'a b']) {
    assert.throws(() => assertSessionId(bad), /不合法/, `${JSON.stringify(bad)} 该被拒绝`)
  }
  for (const ok of ['default', 's1', 'abc-DEF_123', 'x'.repeat(64)]) {
    assert.doesNotThrow(() => assertSessionId(ok), `${ok} 该被放行`)
  }
})

test('非法的 id 在每个入口都进不去', async () => {
  await withStore(async (store) => {
    await assert.rejects(() => store.append('../escape', 0, ev.start('x')), /不合法/)
    await assert.rejects(() => store.load('../escape'), /不合法/)
    await assert.rejects(() => store.remove('../escape'), /不合法/)
  })
})

test('header 里写的是当前格式版本', async () => {
  await withStore(async (store, root) => {
    await store.append('s1', 0, ev.start('问'), { cwd: '/p' })
    const raw = await readFile(join(root, projectKey('/p'), 's1', 'log.jsonl'), 'utf8')
    assert.equal(JSON.parse(raw.split('\n')[0]!).v, SESSION_FORMAT_VERSION)
  })
})

test('并发的两次写不会互相吃掉', async () => {
  await withStore(async (store) => {
    await Promise.all(
      ['a', 'b', 'c', 'd', 'e'].map(async (n) => {
        await store.append(n, 0, ev.start(`问-${n}`), { cwd: '/p' })
        await store.append(n, 0, ev.end(`答-${n}`))
      }),
    )
    assert.equal((await store.list()).length, 5, '五个都要在')
  })
})
