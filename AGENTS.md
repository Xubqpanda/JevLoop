# nanojev 的代码规范

这份文件是给**改这个仓库的人（和 agent）**看的。它管的是代码怎么写，不是这个项目是什么 ——
那在 [README.md](README.md) 里。

规范参考 DeepSeek Harness（MIT）。摘的是**有实际约束力的那几条**，不是格式洁癖。

---

## 1. 格式

| | |
|---|---|
| 引号 | 单引号 `'` |
| 分号 | 不写 |
| 缩进 | 2 空格 |
| 行尾 | LF，文件以**恰好一个**换行结尾 |
| 模块系统 | ESM（`"type": "module"`） |
| 相对导入 | **必须带 `.ts` 扩展名** |

最后一条不是风格问题：Node 的类型剥离和 `allowImportingTsExtensions` 都要求它。

## 2. 注释写事实和契约，不叙述代码

**要写的**：这个模块解决什么问题、为什么这么设计、有什么不变量、调用方要注意什么。

**不要写的**：逐行解释控制流、复述代码、写思考过程。

**不要用比喻。** 写「响应字段」不写「响应形态」，写「ESM 导出」不写「模块形态」。
（`contract` 只留给前置条件、后置条件、不变量、兼容性承诺这类**义务**。）

## 3. 每个模块开头一段 JSDoc

第一段讲**问题**，第二段讲**设计决定**，结尾 `@module nanojev/<文件名>`。

```ts
/**
 * 判定后端的位置解析。
 *
 * 上游传的是一个「想要什么」的请求，不是一个已经定好的连接参数 ——
 * 因为超时和重试属于调用方，baseUrl 和鉴权属于部署。把两者混在一个类型里，
 * 换部署方式就得改调用方。
 *
 * @module nanojev/provider
 */
```

## 4. 每个导出都要有 JSDoc

非显然的契约必须写。函数型导出带 `@param` / `@returns`。

## 5. 封闭联合的 switch 必须以 `assertNever` 结尾

这样新增一个成员时编译器会逼你处理它，而不是静默走到 default。

```ts
function describe(a: Answer): string {
  switch (a.type) {
    case 'noul': return `P=${a.noul}`
    case 'choice': return a.choice
    case 'score': return String(a.score)
    default: return assertNever(a)
  }
}
```

## 6. 只在**真实边界**做运行时校验

进程内、类型化、同进程的边界**信任 TypeScript**。不要为了「静态类型已经保证的东西」再加一层
运行时校验、兜底行为、或者防御性测试。

该校验的地方只有四类：

- 模型/工具返回的 JSON
- 配置文件
- 从磁盘读的文件
- 网络响应

## 7. 显式 > 隐式

默认值必须是一个**显式的解析步骤**（`resolve(request): Spec`），
不能藏在 `run()` 里的 `?? default`。读代码的人要能一眼看出「这个值是哪来的」。

## 8. 没有硬编码的可调项

部署方式会变的量，都是显式的选项字段。写死的 `DEFAULT_*` 常量不算「可配置」。

**例外**：协议常量、外部规范、安全不变量 —— 这些就该写死。

## 9. 空的 catch 必须说明它吞了什么

而且要说明**为什么没有别的东西能到达这里**。`try` 里只包一条语句。

```ts
try {
  readFileSync(candidate)
} catch {
  // 文件不存在是正常情况：继续往上一层找 .env
  continue
}
```

## 10. 能力缝是三角，不能只有一角

一个能力由三部分组成，要么齐全，要么不叫能力缝：

```
Service Definition   接口（契约）
Service Provider     实现（可以有多个）
Consumer             使用者
```

nanojev 现有的两条缝：`Provider`（判定后端）和 `Generator`（生成后端）。
新增能力请照这个形状切。

---

## 验证

改完必须跑：

```sh
npx tsc --noEmit                  # 0 error
npm test                          # 全绿
npm run demo                      # 跑完，打印记账
node --experimental-strip-types examples/demo.ts --laya   # 有 sidecar 时
```

**注意 Node 版本**：类型剥离是 v22.6 引入、v22.18 才默认开启的。
v22.16 上必须带 `--experimental-strip-types`，否则报一个完全看不出原因的
`SyntaxError: Missing initializer in const declaration`。
