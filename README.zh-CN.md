# JevLoop

**你的 agent loop 里每一个岔路口都是一次完整的大模型调用。而它们没有一个是「生成」。**

*要不要动手？用哪个工具？这个操作安全吗？成功了吗？做完了吗？这个回答能发出去吗？* —— 常规 agent 对每一个的回答方式，都是让大模型写一段话，再由代码解析回来。但这六件事分别是**挑选、打分、是非题**：一次前向传播，在固定的候选集上给出答案，约 10–40 ms，不生成任何 token。

JevLoop 把它们交给判定模型（[Jev](https://typesafe.ai) / [Laya](https://github.com/NandaKishorM/laya)），把大模型留给它唯一不可替代的那件事：**写**。

JevLoop 是一个独立项目，与 TypeSafe AI 没有隶属关系，也未获其背书 —— 名字只是指向它所路由的那个模型，仅此而已。

[English](README.md) · **中文**

![JevLoop：一次 demo 运行，12 次判定、1 次模型调用](docs/demo.gif)

```
$ npm run demo          # 全新 clone：无 key、无网络、不用 npm install

JevLoop · demo
  decision  : laya→rule-judge
  generator : scripted — set DEEPSEEK_API_KEY for a real LLM

  ── loop trace ──────────────────────────────────────────
  ▲ laya unavailable (fetch failed), falling back to rule-judge
  cleared: list_dir (auto)
  cleared: read_file (auto)

  ── every decision ──────────────────────────────────────
  step 1
     decide  loop.needsTool         use_tool           4.3ms  needs_tool=0.95
     decide  loop.pickTool          call               4.3ms  tool=list_dir
     decide  loop.gradeRisk         auto               4.2ms  risk=0.0 needs_auth=0.05
     decide  loop.stepOk            continue           4.1ms  ok=0.92
     decide  loop.isDone            keep_going         4.0ms  done=0.10
  ...
  step 3
     decide  loop.canDeliver        deliver            4.3ms  deliverable=0.90 unsupported=0.08
     model   generate (scripted)                       601ms

  ── accounting ──────────────────────────────────────────
  decisions  12     50ms (4.2ms each)
  model       1     601.2ms

  decisions : model = 12.0:1   decisions are 7.7% of wall clock
```

零依赖、零构建步骤、无 key 无网络也能跑完整条 loop。

> 上面这次运行的判定来自**规则表，不是模型** —— 它演示的是 loop 的**形状**，不是判定的质量。
> 真实后端与它们的实际开销见 [诚实的数字](#诚实的数字)。

---

## 问题在哪

拿一个需要两次工具调用的任务来说，常规 agent 会为下面每一件事各烧掉一次模型调用：

| loop 要问的问题 | 常规 agent | JevLoop |
|---|---|---|
| 现在需要动手吗？ | 大模型调用 | 判定 |
| 用哪个工具？ | 大模型调用 | 判定 |
| 这个调用安全吗？ | 大模型调用，或者干脆不问 | 判定 |
| 成功了吗？ | 大模型调用 | 判定 |
| 做完了吗？ | `max_iter` 计数器 | 判定 |
| 这个回答能发出去吗？ | **什么都没有** | 判定 |

**你一直在用生成的价格，买判定的答案。**

## 快速开始

需要 **Node ≥ 22.6**（它直接跑 TypeScript，没有构建步骤）。

```bash
git clone https://github.com/zjunlp/JevLoop
cd JevLoop
npm run demo
```

就这样 —— 不用 `npm install`、不用 key、不用网络。demo 会退回到一个确定性的规则判定器，让整条 loop 离线跑通。

**换成真实的判定模型：**

```bash
npm run demo -- --laya     # 本地 Laya sidecar（:7789，开放权重，免费）
npm run demo -- --jev      # 官方 Jev API（需要 TYPESAFE_API_KEY）
```

## DECISION.md —— 被编译的决策文件

每一代 agent 框架都会留下一个 `.md`。`AGENTS.md` 放约定，`SKILL.md` 放能力 —— 而这两者都是**给大模型读的散文**。模型每轮都为此付 token，它可以不听，而且没有任何东西会告诉你它到底听没听。

`DECISION.md` 是第一个**被编译**的。

> **它不是 decision *record*。** 记录（record）是事后写下来解释 agent 做过什么的。`DECISION.md` 声明的是这条 loop **将要**下哪些判断，由程序把它编译成发给判定模型的问题。

一份文件，两个消费者：

```
结构块  →  问题 + 策略  →  判定模型（几十毫秒，不花 token）
散文    →  system prompt  →  大模型（整个 loop 唯一贵的一步）
```

**不能编译的是帧。** 每个判定还需要一段 `state` 投影 —— 从 agent 的状态里挑哪几个字段给模型、各截多长。那是个**函数**，而 markdown 表达不了函数。它留在 [`src/decisions.ts`](src/decisions.ts) 里：

```ts
export const needsTool = defineDecision({
  id: 'loop.needsTool',
  state: ctx => ({ task: clip(ctx.task, 400), earlier: …, steps_done: … }),  // 代码
  ...compiled('needs_tool', ['needs_tool']),                                  // 文件
})
```

硬把帧塞进 markdown 只有两条路：发明一门真正的 DSL，或者让帧退化成「把整个上下文塞进去」—— 后者装不进判定模型 512/1024 token 的窗口。**这条边界是刻意的，不是没做完。**

两半各有各的把关方向。问题和动作来自文件，代码在**加载时**被文件校验：问题名改成代码不认识的样子，启动就失败并同时列出两边的名字，而不是跑三步之后 `answers.risk` 变成 `undefined`。帧没有这道校验 —— 因为文件里没有能拿来校验它的东西，这也正是它留在代码里的原因。

文件同样没法悄悄腐烂：它在加载时被解析，任何问题 —— 动作名不在封闭词汇表里、谓词指向了错误的问题类型 —— 都会带着行号抛出来，而不是编译成一条永不命中的规则。

所以它是**减法**：每搬一个块进文件，就是少问大模型一个问题。[`headline()`](src/decisiondoc.ts) 会**从文件本身**把它们数出来 —— 改一个 `kind`，那句话就跟着变。

（这句话以前写的是「tests 双向断言它和代码是同一件事」—— 那个说法现在**反了**：文件是正本，没有第二份可供比对。原来那条对账测试留下的是仍然成立的那部分：原语类型、`score` 的档位标签、`noul` 的 true/false 说明。）

```markdown
## grade_risk
kind: mixed
when: 每次真正调用工具之前

### risk
ask: How risky is this tool call?
- read-only
- reversible write
- irreversible
- destructive

### needs_auth
ask: This call must be explicitly authorised by a human before it runs
- true — it can destroy data, spend money, or leave the machine
- false — it only reads or writes inside the working directory

policy:
  - score:risk >= 2 → ask_human
  - prob:needs_auth >= 0.5 → ask_human
  - score:risk >= 1 → auto_audit
  - else → auto
```

**问题的原语类型靠选项的写法推出来，从不声明。** 两个选项叫 `true` / `false` 就是 `noul`；全都带名字就是 `choice`；全都不带名字就是 `score`；混着写会**报错而不是猜**。然后 `kind` 必须和写法推出来的结果对得上。

**谓词是封闭词汇表。** `else`、`top >= n` / `top < n`（仅限只有一个问题的块）、`prob:<id>`（用在 `noul` 上）、`score:<id> >= n`（用在 `score` 上）、`picked:<id> = <选项>`（用在 `choice` 上）。故意不支持 `>` 和 `<=`：**写不出来的条件，就是该写进代码的信号。**

**谓词写错了问题类型会被拒绝，而不是被编译。** 放着不管的话它会变成一条**永不命中的规则** —— 作者以为自己写了一道闸门，实际没有，而且它是 fail open 的。动作名同样是封闭表，写错会带行号报出来。文件里没有任何东西会被静默丢弃：认不出来的一律进 `problems`，并附上它来自哪一行。

## 它是怎么工作的

```
 step ─┬─ loop.needsTool   ↗ 需要动手吗？  ──否──▶ 生成
       │
       ├─ loop.pickTool    ↗ 用哪个工具？（候选每步重建）
       │
       ├─ loop.gradeRisk   ↗ 这个操作多危险？  ──▶ 问人
       │
       ├─ [ 工具执行 ]      ← 全流程唯一产生真实副作用的地方
       │
       ├─ loop.stepOk      ↗ 成功了吗？
       │
       └─ loop.isDone      ↗ 做完了吗？  ──否──▶ 下一个 step
                            │
                            ▼
                       [ 大模型生成 ]   ← 唯一昂贵的一次调用
                            │
                       loop.canDeliver  ↗ 这个回答能发出去吗？
```

**六个判定点全在一个文件里：[`src/decisions.ts`](src/decisions.ts)。** 如果你只看这个仓库的一个文件，看那个 —— 它就是全部主张。

### 一个判定由三样东西组成

```ts
export const pickTool = defineDecision({
  id: "loop.pickTool",

  // ① 把 agent 状态投影成一个**有界**的决策帧。
  //    它决定了判定的上限：帧里没有的东西，判不出来。
  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    files_known: (ctx.files ?? []).slice(0, 20),
    recent: (ctx.history ?? []).slice(-3).map(h => `${h.tool}(${h.input}) → ${clip(h.result, 120)}`),
  }),

  // ② 类型化的问题。一次前向传播全部答完。
  questions: (ctx: AgentCtx) => ({
    tool: choice("Which tool should the agent call next?", toolsFor(ctx)),
  }),

  // ③ 策略：答案 → 动作。纯代码，没有模型参与。
  policy: [
    { when: gte("tool", 0.6), action: "call" },
    { action: "escalate", reason: "not confident enough — hand back, don't guess" },
  ],
});
```

三个原语，直接来自 Jev 的线协议：

| 原语 | 答案 | 用来做 |
|---|---|---|
| `noul` | P(true)，0–1 | **闸门** —— 放行 / 拦住 |
| `choice` | 一个选项 + 每个选项的概率 | **路由** —— 走哪条路 |
| `score` | 有序刻度上的期望档位 | **评级** —— 有多糟 |

### 两件值得偷走的东西

**候选每一步都重建。** 固定的候选列表会让模型去选一个已经不适用的动作 —— 写完文件之后 `write_file` 不该还在候选里。所以 `questions` 可以是上下文的函数。

**绝不让概率绕过授权。** 风险闸门是硬规则，不是阈值：

```ts
policy: [
  // 不可逆 ⇒ 必须显式授权。没有任何置信度能覆盖这一条。
  { when: scoreGte("risk", 2), action: "ask_human" },
  // 模型自己的判断是第二道、独立的闸门
  { when: probGte("needs_auth", 0.5), action: "ask_human" },
  { action: "auto" },
]
```

判定模型可以决定**要不要问人**。它绝不能决定**要不要跳过授权**。

## 诚实的数字

同一条 loop 跑在三种判定后端上。真正重要的比值是「判定 : 模型调用」，而让我们意外的是判定占了多少墙钟时间。

| 判定后端 | 每次判定 | 判定 : 模型 | 判定占墙钟 | 质量 |
|---|---:|---:|---:|---|
| `examples/rule-judge.ts`（离线 demo） | 4 ms | 12 : 1 | **7.7 %** | 规则表，不是模型 |
| Laya `typed-decisions`，本地 A100 | 30–85 ms | 8 : 1 | ~38 % | **零样本不够用**（见下） |
| Jev `jev-latest`，托管 API | ~390 ms | 13 : 1 | **79 %** | 每个判定都果断且正确 |

两条我们不会软化的结论：

- **整个主张在「本地部署判定模型」时成立。** 30 ms 一次的判定，让这条 loop 的思考相对于一次生成调用基本免费。
- **走托管 API 时不成立。** 每次判定约 390 ms 是网络往返，13 次判定换 1 次生成，判定就成了墙钟的大头。它仍然比一次前沿大模型调用快 5–8 倍、便宜几个数量级，但在那个延迟下说「判定不要钱」就是撒谎。

明显的甜点是本地部署的强判定模型。我们测过的两个都不是：一个快但不够准，一个准但要走网络。

### 两个我们踩过的坑

两个都是拿真实 Laya 权重在 A100 上跑出来的，不是读文档读出来的。

**`confidence` 不是选中项的概率。** Laya 对 choice 的 `confidence` 是归一化香农熵（`1 - H(p)/log(k)`）—— `p = [0.80, 0.20]` 算出来是 `confidence = 0.269`。所以同一个阈值在 2 个选项和 20 个选项下含义完全不同。请改用**选中项的概率**来卡 `choice`，也就是 `topGte()`。（[官方文档](https://docs.typesafe.ai/confidence)把 `confidence` 定位成「够用的默认值」，并把完整的 `probabilities` 一并给你，正是为了这种情况。）

**基座 checkpoint 做不了没见过的判定任务。** 问它「下一步用哪个工具」，`laya-typed-decisions` 在 step 2 以 **0.660** 选了 `done`，而 step 1 的正确答案只有 **0.646** —— 错的比对的还高，而且四个判定点的答案全落在 0.55–0.66 这个带子里，毫无区分度。**没有阈值能修这个**，这是能力缺口。开放权重的 checkpoint 是一个**用来微调的快速基座**，不是一个开箱即用的判定器。

两条其实是同一个教训，来自 [Jev Engineering](https://madewithjev.com/what-is-jev-engineering)：*调用是最简单的一步 —— 功夫全在你送进去的 state，和你据以行动的那个阈值。*

## 换成你自己的后端

**判定后端** —— 任何能回答 `{state, questions} → {answers}` 的东西：

```ts
import { Decider, HttpProvider, FallbackProvider, MockProvider } from 'jevloop';

const decider = new Decider({
  provider: new FallbackProvider([
    new HttpProvider({ baseUrl: "https://api.typesafe.ai", apiKey: process.env.TYPESAFE_API_KEY, name: "jev" }),
    new HttpProvider({ baseUrl: "http://127.0.0.1:7789", name: "laya" }),
    new MockProvider(),   // 永不失败
  ]),
});
```

**生成后端** —— 任何能把 prompt 变成文本的东西：

```ts
import { HttpGenerator } from 'jevloop';
// 任何 OpenAI 兼容的 /chat/completions 端点
new HttpGenerator({ baseUrl: "https://api.openai.com/v1", apiKey, model: "gpt-5" });
new HttpGenerator({ baseUrl: "http://localhost:11434/v1", model: "qwen3" });  // ollama
```

换掉任何一个都只动一个文件。loop 和判定规格一步都不用挪。

> 还没发到 npm。从 git 装 —— `prepare` 脚本会自动帮你构建 `dist/`：
> `npm install github:zjunlp/JevLoop`

## 它不是什么

- **不是大模型的替代品。** 起草、写代码、总结仍然需要它。
- **不是「零幻觉」。** 判定模型给不出你问的类型之外的答案，但答案仍然可能是错的。阈值就是为这个准备的。
- **还没有和常规 agent 在同一任务上做过对比。** 上面的 `12:1` 来自自带的 demo。那个对比是显而易见的下一步，而它还没做。
- **没有做过生产加固。** 工具沙箱只覆盖路径逃逸。把它指向任何你在乎的东西之前，先读 [`src/tools.ts`](src/tools.ts)。

## 目录结构

```
DECISION.md      ★ 决策文件 —— 被编译（问题与策略；帧仍在代码里）
src/
  vocab.ts       Question / Answer / Decision —— 全部词汇
  decisions.ts   ★ 这个 agent 的全部判定，一个文件
  decisiondoc.ts DECISION.md 的解析器（什么都不静默丢）
  decision-compile.ts  块 → 问题 + 策略
  decide.ts      一次判定的六个步骤
  policy.ts      答案 → 动作（纯代码，可单元测试）
  seam-provider.ts     判定后端的接口
  provider-http.ts     Jev / Laya —— 换 baseUrl 就换后端
  provider-mock.ts     一个从不猜测的替身
  provider-fallback.ts 按顺序试，每次降级都报出来
  meter.ts       ★ 判定 vs 模型调用
  agent.ts       ★ 这条 loop
  tools.ts       list / read / write，路径锁在 cwd
  llm.ts         唯一生成文本的地方
examples/
  demo.ts        离线可跑
  rule-judge.ts  确定性的判定模型替身
```

## 参与贡献

先读 [`CONTRIBUTING.md`](CONTRIBUTING.md)（英文）。里面写了我们收什么样的改动、什么样的会直接关掉，以及现在真正开着的活在哪。

## License

Apache-2.0
