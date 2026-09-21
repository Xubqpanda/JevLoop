/**
 * 工作区的回归测试：选目录（浏览）+ 登记表。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里全是磁盘和路径行为，`tsc` 通过不代表写对了。重点测四类：
 *
 *   · **只列目录**，而且每条带绝对路径 —— 客户端不自己拼路径，
 *     因为拼路径是路径穿越最容易发生的地方。
 *   · **相对路径必须被拒**。它会相对服务进程的 cwd 解析，而那个值
 *     用户看不见，于是「我选的目录」和「实际用的目录」会不是同一个。
 *   · **撤销登记不碰磁盘**。「从列表里移除」和「删掉你的文件」混起来是灾难。
 *   · **封闭的错误词表**，调用方穷举处理而不是认字符串。
 *
 * @module JevLoop/workspace.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readdir, rm, symlink, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, join } from 'node:path'
import { tmpdir } from 'node:os'

import { createDir, listDirs } from '../src/dir-browse.ts'
import { WorkspaceStore } from '../src/workspace.ts'
import { WorkspaceError, assertAbsolute } from '../src/vocab-workspace.ts'

/** 造一个临时目录树，用完删掉 */
async function withTree(
  build: (root: string) => Promise<void>,
  fn: (root: string) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'jevws-'))
  try {
    await build(root)
    await fn(root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

/** 断言抛的是带指定 code 的 WorkspaceError —— **不匹配措辞** */
async function rejects(fn: () => Promise<unknown>, code: string, what: string): Promise<void> {
  await assert.rejects(fn, (err: unknown) => {
    assert.ok(err instanceof WorkspaceError, `${what}：抛的应当是 WorkspaceError，实际 ${String(err)}`)
    assert.equal(err.code, code, `${what}：code 应当是 ${code}`)
    return true
  })
}

// ═══════════════════════════════════════════════════════════
// 浏览
// ═══════════════════════════════════════════════════════════

test('只列目录，文件一个都不出现', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, 'src'))
      await mkdir(join(root, 'docs'))
      await writeFile(join(root, 'README.md'), 'x')
      await writeFile(join(root, 'a.txt'), 'x')
    },
    async (root) => {
      const l = await listDirs(root)
      assert.deepEqual(l.entries.map((e) => e.name), ['docs', 'src'], '文件不该出现')
      assert.equal(l.path, root)
    },
  )
})

test('每个条目带**绝对**路径 —— 客户端不自己拼', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, 'sub'))
    },
    async (root) => {
      const l = await listDirs(root)
      for (const e of [...l.entries, ...l.crumbs]) {
        assert.ok(e.path.startsWith('/') || /^[A-Za-z]:/.test(e.path), `${e.name} 的 path 不是绝对路径：${e.path}`)
      }
      assert.equal(l.entries[0]!.path, join(root, 'sub'))
    },
  )
})

test('hidden 逐条标出来，但**不替客户端决定**显不显示', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, '.git'))
      await mkdir(join(root, 'src'))
    },
    async (root) => {
      const l = await listDirs(root)
      assert.equal(l.entries.find((e) => e.name === '.git')!.hidden, true)
      assert.equal(l.entries.find((e) => e.name === 'src')!.hidden, false)
      // 两条**都返回了** —— 过滤是客户端的事
      assert.equal(l.entries.length, 2)
    },
  )
})

test('crumbs 从根到当前，每一节都是跳转目标', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, 'a', 'b'), { recursive: true })
    },
    async (root) => {
      const l = await listDirs(join(root, 'a', 'b'))
      const names = l.crumbs.map((c) => c.name)
      assert.equal(names[names.length - 1], 'b')
      assert.deepEqual(names.slice(-3), [basename(root), 'a', 'b'])
      assert.equal(names[0], '/', '根那一节用完整路径当名字')
      // 最后一节的 path 就是被列的目录 —— 面包屑能原地刷新
      assert.equal(l.crumbs[l.crumbs.length - 1]!.path, l.path)
    },
  )
})

test('不传路径就从家目录开始 —— 起点不该是服务进程的 cwd', async () => {
  const l = await listDirs()
  assert.equal(l.home, homedir())
  assert.equal(l.path, homedir())
})

test('相对路径必须被拒 —— 它会相对服务进程的 cwd 解析，而用户看不见那个值', async () => {
  await rejects(() => listDirs('some/relative'), 'not-absolute', 'listDirs')
  assert.throws(() => assertAbsolute('./x'), /绝对路径/)
  assert.throws(() => assertAbsolute('x'), /绝对路径/)
  assert.doesNotThrow(() => assertAbsolute('/x'))
})

test('路径不存在 / 不是目录 —— 两种失败分得开', async () => {
  await withTree(
    async (root) => {
      await writeFile(join(root, 'file.txt'), 'x')
    },
    async (root) => {
      await rejects(() => listDirs(join(root, 'nope')), 'unreadable', '不存在')
      await rejects(() => listDirs(join(root, 'file.txt')), 'not-a-directory', '是文件')
    },
  )
})

test('符号链接指向目录就算目录；断链跳过，不报错', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, 'real'))
      await symlink(join(root, 'real'), join(root, 'link'))
      await symlink(join(root, 'gone'), join(root, 'broken'))
    },
    async (root) => {
      const l = await listDirs(root)
      const names = l.entries.map((e) => e.name)
      assert.ok(names.includes('real'), '真目录在')
      assert.ok(names.includes('link'), '指向目录的链接也算目录')
      assert.ok(!names.includes('broken'), '断链不是目录，跳过它')
    },
  )
})

test('条目超过上限时要**说出来**，不能悄悄少给几行', async () => {
  const root = await mkdtemp(join(tmpdir(), 'jevws-many-'))
  try {
    // 501 个目录，刚好越过 500 的上限
    await Promise.all(Array.from({ length: 501 }, (_, i) => mkdir(join(root, `d${String(i).padStart(3, '0')}`))))
    const l = await listDirs(root)
    assert.equal(l.truncated, true, '截断了就要报')
    assert.equal(l.entries.length, 500)
    // 截掉的是按名字排序的**尾部**
    assert.equal(l.entries[0]!.name, 'd000')
    assert.equal(l.entries[499]!.name, 'd499')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

// ═══════════════════════════════════════════════════════════
// 新建目录
// ═══════════════════════════════════════════════════════════

test('createDir 建出来，返回绝对路径', async () => {
  await withTree(
    async (root) => {
      const made = await createDir(root, 'fresh')
      assert.equal(made, join(root, 'fresh'))
      assert.deepEqual(await readdir(root), ['fresh'])
    },
    async () => {},
  )
})

test('createDir 的名字必须是**单独一段** —— 这是安全性质不是洁癖', async () => {
  await withTree(
    async (root) => {
      for (const bad of ['', '   ', '.', '..', 'a/b', '../escape']) {
        await rejects(() => createDir(root, bad), 'bad-name', `名字 ${JSON.stringify(bad)}`)
      }
      // 一条都没建出来
      assert.deepEqual(await readdir(root), [])
      // 也没写到父目录去
      assert.ok(!(await readdir(tmpdir())).some((n) => n === 'escape'))
    },
    async () => {},
  )
})

test('createDir 撞名报 name-taken，不覆盖', async () => {
  await withTree(
    async (root) => {
      await mkdir(join(root, 'dup'))
      await rejects(() => createDir(root, 'dup'), 'name-taken', '重名')
    },
    async () => {},
  )
})

test('createDir 的父目录必须是绝对路径', async () => {
  await rejects(() => createDir('relative', 'x'), 'not-absolute', '父目录')
})

// ═══════════════════════════════════════════════════════════
// 登记表
// ═══════════════════════════════════════════════════════════

async function withStore(fn: (store: WorkspaceStore, root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'jevws-store-'))
  try {
    await fn(new WorkspaceStore(join(root, 'workspaces.json')), root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('登记一个目录，再登记一次是**幂等**的', async () => {
  await withStore(async (store, root) => {
    const dir = join(root, 'proj')
    await mkdir(dir)

    const a = await store.create(dir)
    assert.equal(a.created, true)
    assert.equal(a.workspace.path, dir)
    assert.equal(a.workspace.title, 'proj', '默认名字取目录名')

    const b = await store.create(dir)
    assert.equal(b.created, false, '同一个路径第二次是 false')
    assert.equal(b.workspace.id, a.workspace.id, '返回的是原来那条，不是新建一条')
    assert.equal((await store.list()).length, 1)
  })
})

test('符号链接进来的同一个目录认成同一个 —— 比的是 realpath', async () => {
  await withStore(async (store, root) => {
    const dir = join(root, 'real')
    await mkdir(dir)
    await symlink(dir, join(root, 'alias'))

    const a = await store.create(dir)
    const b = await store.create(join(root, 'alias'))
    assert.equal(b.created, false, '经软链进来的同一个目录不该新建一条')
    assert.equal(b.workspace.id, a.workspace.id)
  })
})

test('登记时的路径校验：相对路径 / 不存在 / 是文件', async () => {
  await withStore(async (store, root) => {
    await writeFile(join(root, 'f.txt'), 'x')
    await rejects(() => store.create('relative/path'), 'not-absolute', '相对路径')
    await rejects(() => store.create(join(root, 'nope')), 'unreadable', '不存在')
    await rejects(() => store.create(join(root, 'f.txt')), 'not-a-directory', '是文件')
  })
})

test('列表按最近动过的排前面', async () => {
  await withStore(async (store, root) => {
    for (const n of ['a', 'b', 'c']) await mkdir(join(root, n))
    const a = await store.create(join(root, 'a'))
    await new Promise((r) => setTimeout(r, 5))
    await store.create(join(root, 'b'))
    await new Promise((r) => setTimeout(r, 5))
    // 给 a 改个名 —— 它就成了最近动过的
    await store.rename(a.workspace.id, '甲')
    assert.deepEqual((await store.list()).map((w) => w.title), ['甲', 'b'])
  })
})

test('改名：改得掉、找不到要抛、空名字要抛', async () => {
  await withStore(async (store, root) => {
    await mkdir(join(root, 'p'))
    const { workspace } = await store.create(join(root, 'p'))
    assert.equal((await store.rename(workspace.id, '我的项目')).title, '我的项目')
    await rejects(() => store.rename('w-nope', 'x'), 'not-found', '找不到')
    await rejects(() => store.rename(workspace.id, '   '), 'bad-name', '空名字')
  })
})

test('★ 撤销登记**不碰磁盘** —— 「从列表里移除」和「删掉你的文件」是两件事', async () => {
  await withStore(async (store, root) => {
    const dir = join(root, 'keepme')
    await mkdir(dir)
    const { workspace } = await store.create(dir)

    assert.equal(await store.remove(workspace.id), true)
    assert.deepEqual(await store.list(), [])
    // ★ 目录还在，而且里面原来的东西也在
    assert.deepEqual(await readdir(dir), [])
    await writeFile(join(dir, 'still-here.txt'), 'x')
    assert.deepEqual(await readdir(dir), ['still-here.txt'])
  })
})

test('撤销不存在的返回 false，不抛', async () => {
  await withStore(async (store) => {
    assert.equal(await store.remove('w-nope'), false)
  })
})

test('表文件坏了当空表，不把服务带崩', async () => {
  await withStore(async (store, root) => {
    await writeFile(store.file, '这不是 JSON{{{', 'utf8')
    assert.deepEqual(await store.list(), [])
    // 而且还能继续用
    await mkdir(join(root, 'after'))
    const r = await store.create(join(root, 'after'))
    assert.equal(r.created, true)
    assert.equal((await store.list()).length, 1)
  })
})

test('表里的一条坏记录不会让整张表消失', async () => {
  await withStore(async (store, root) => {
    await mkdir(join(root, 'ok'))
    await store.create(join(root, 'ok'))
    const good = await store.list()
    // 手改表：塞一条缺字段的进去（表是可读的，人会去改它）
    await writeFile(store.file, JSON.stringify([...good, { id: 'x' }, null, 'nope']), 'utf8')
    assert.equal((await store.list()).length, 1, '好的那条要留下')
  })
})

test('并发的两次写不会互相吃掉 —— 串行化', async () => {
  await withStore(async (store, root) => {
    for (const n of ['a', 'b', 'c', 'd', 'e']) await mkdir(join(root, n))
    await Promise.all(['a', 'b', 'c', 'd', 'e'].map((n) => store.create(join(root, n))))
    assert.equal((await store.list()).length, 5, '五条都要在 —— 后写的不能覆盖先写的')
  })
})
