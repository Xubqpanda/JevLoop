/**
 * 命令行参数层。
 *
 *   node --experimental-strip-types --test "tests/*.test.ts"
 *
 * 这里钉的是**白名单**那条规则：只有取值的选项会吃掉下一个 token。
 * 没有它，`jevloop run --jev list the files` 会把 `list` 当成 `--jev` 的值，
 * 于是任务名少一个词，而没有任何东西报错。
 *
 * @module JevLoop/cli-args.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { parseArgv } from '../src/cli-args.ts'

test('位置参数按顺序收集，选项单独拿出来', () => {
  const p = parseArgv(['run', 'list', 'the', 'files', '--cwd', '/tmp'])
  assert.equal(p.command, 'run')
  assert.deepEqual(p.positional, ['list', 'the', 'files'])
  assert.equal(p.options.get('cwd'), '/tmp')
})

test('开关型选项不吃下一个 token —— 这是白名单存在的理由', () => {
  const p = parseArgv(['run', '--jev', 'list', 'the', 'files'])
  assert.ok(p.options.has('jev'), '--jev 应当是一个开关')
  assert.equal(p.options.get('jev'), '')
  assert.deepEqual(p.positional, ['list', 'the', 'files'], '任务名一个词都不能少')
})

test('--key=value 和 --key value 等价', () => {
  assert.equal(parseArgv(['serve', '--port=7812']).options.get('port'), '7812')
  assert.equal(parseArgv(['serve', '--port', '7812']).options.get('port'), '7812')
})

test('取值的选项后面没有值要抛，不能静默当成开关', () => {
  assert.throws(() => parseArgv(['run', 'task', '--cwd']), /--cwd needs a value/)
})

test('没有子命令时 command 是空串，不抛', () => {
  const p = parseArgv([])
  assert.equal(p.command, '')
  assert.deepEqual(p.positional, [])
  assert.equal(p.options.size, 0)
})

test('spec 的路径参数照常当位置参数收', () => {
  const p = parseArgv(['spec', './my/DECISION.md'])
  assert.equal(p.command, 'spec')
  assert.deepEqual(p.positional, ['./my/DECISION.md'])
})
