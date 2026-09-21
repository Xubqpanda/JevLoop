# experiments/ —— 实验台

**这是七个人共用的地方。动手之前先读完这份,再读 `docs/PLAN-confidence-2026-09-21.md`。**

计划、数据集清单、baseline 清单、指标清单都在伞仓库：
`../../docs/PLAN-confidence-2026-09-21.md`（决定）+ `../../docs/NOTES-benchmark-rationale-2026-09-21.md`（理由）+
`../../docs/SURVEY-agent-benchmarks-2026-09.md`（事实底稿，每条带 URL）。

**这份只管「东西放哪、叫什么名、怎么跑」。**

| 要做什么 | 看哪份 |
|---|---|
| 东西放哪、文件叫什么 | **这份**（§1–§3）|
| 结果长什么样 | **这份** §4 |
| **怎么跑、留下什么 log** | **`PROTOCOL.md`** ← ★ 跑之前必读 |
| 数据从哪来、什么版本 | **`dataset/MANIFEST.md`** |
| 为什么选这些数据集 / baseline | 伞仓库 `docs/PLAN-*.md` §0–§2 |

> ## ★ 一条不能破的：**没有 log 的运行 = 没跑过。**
> 见 `PROTOCOL.md` §0。**拿不出 log 的数字不许进 `result/`。**

---

## 1. 六个目录

| 目录 | 放什么 | 进 git？ |
|---|---|---|
| `dataset/` | 数据集、Wikipedia 索引、下载缓存。**上百 GB** | ❌ **只有 `MANIFEST.md` + `.gitkeep` 进** |
| `benchmark/` | 每个 benchmark 一个 loader / adapter，产出统一的 `Result[]` | ✅ |
| `baseline/` | 每个范式**一份**实现（Direct / Act / ReAct / Reflexion / Plan / ReWOO） | ✅ |
| `scripts/` | 入口命令：跑一个数据集、跑一张表、出图 | ✅ |
| `log/` | 每次运行的原始产物（见 `PROTOCOL.md` §1） | ❌ |
| `result/` | 汇总后的表与图 —— **论文里的数字从这里出** | ✅ |

**`dataset/` 里只放索引，不放数据。** 数据下落不明没关系，
`dataset/MANIFEST.md` 写清了**每一个数据集的 id / 版本 / 划分 / Wikipedia dump 版本**，
照着能重建。**拿不准的标 `待核`，不许用记忆里的数字填。**

**为什么 `dataset/` 和 `log/` 不进 git**：一个 Wikipedia 索引 + 19 个数据集是上百 GB，
一次跑出来的逐条日志是几十 MB。**仓库不是硬盘。**
**为什么 `result/` 进 git**：它是**论文的数字本身** —— 必须能 diff、能追溯是哪次改动改出来的。

⚠️ **log 不进 git ≠ log 可以不存** —— 跑完要打包归档，见 `PROTOCOL.md` §3.6。


---

## 2. ★ 开工之前必须先冻住三件（P1 负责）

七个人各写各的，最后合不起来**不报错**，只是第 4 周发现七套 CSV 拼不到一起。
所以这三件**由 P1 先落地并提交，其他人等它**：

| 冻什么 | 落在哪 | 内容 |
|---|---|---|
| **① runner 接口** | `scripts/run.ts` | 所有 arm 走同一个入口：`runDataset(spec) → Result[]` |
| **② 结果 schema** | `scripts/spec.ts` | `docs/PLAN-*.md` §3 那套字段**一个不少**；**缺字段的结果拒收** |
| **③ 各 arm 的实现** | `baseline/*.ts` | **一份 ReAct、一份 Reflexion……不是七份。** arm 是共享代码，**数据集才是各人 own 的** |

**在 ① ② 落地之前，先照 §4 的字段写，别自己加列。**

---

## 3. 文件命名 —— **先定死，别自己发挥**

七个 `bfcl` 会长成七个不同的名字，然后 import 全错。**按这张表：**

### `benchmark/<name>.ts`

```
bfcl-v4        fever          hotpotqa       triviaqa       strategyqa
alfworld       mcp-agentbench 2wiki          musique        ragtruth
truthfulqa     tau2-bench     gaia2          appworld       terminal-bench
gsm8k          sports         physics        bfcl-v3        api-bank
```

（`bfcl-v3` 并进 `bfcl-v4`、`api-bank` 已砍 —— **见 `PLAN-*.md` §0.6，别照旧名字建。**）

### `baseline/<name>.ts`

```
direct   act   react   reflexion   planthenexecute   rewoo   jevloop
```

⚠️ **`react` 现在名不副实**：`bench/react.ts` 的 system prompt 里**没有 thought 字段**
（`grep -c thought bench/react.ts` = **0**），所以它跑出来的是 **`act`**。
**新写的 `baseline/react.ts` 必须补上 thought**，否则标签本身就是审稿风险。

### 其他

- 测试放 `tests/<name>.test.ts`（**平的**，测试运行器的 glob 是 `tests/*.test.ts`）
- 一个数据集需要 fixture 时，才升级成 `benchmark/<name>/` 目录

---

## 4. 结果 schema —— **缺字段的结果不收**

字段来源是 `PLAN-*.md` §3。**一个 run 一行**，至少这些：

```
dataset  task_id  arm  seed  model{id,version,provider}  thinking_budget
temperature  top_p  max_tokens  prompt_hash  dataset_version  commit  region  cold_start

# 效果
success  steps  first_divergence_step  escalated  gate_false_reject  gate_false_deny
failure_class            # §3.8 的分类，逐条打标，不许只报总数

# 成本（usage）
llm_calls  decision_requests  questions_per_request  tool_calls
tool_calls_necessary  tool_calls_exploratory
input_tokens_cached  input_tokens_uncached  output_tokens_reasoning  output_tokens_visible
usd

# 时间（每一层分开，不许只报墙钟）
wall_ms  model_ms{handshake,ttft,after_ttft}  decision_ms{handshake,compute}
tool_ms  framework_ms  retry_ms  round_trips

# 置信度（RQ2）
confidence  correct      # 每条判定一行 → 分桶 / ECE / Brier / AUC / 分离度 / 风险-覆盖
```

**★ `framework_ms` 和 `retry_ms` 必须有。** 不留这两行，就永远不知道差距里有多少是自己的开销、
多少是对面在退避 —— 我们踩过：`generate` 那步墙钟 21.8s，meter 只记 11.0s，
差掉的一半被误读成「JevLoop 慢 10 倍」。
**★ `confidence` + `correct` 要逐判定落，不是逐任务。** RQ2 全靠这一对。

---

## 5. 怎么跑

**现在还没有 ① ②，所以先按这个形状写，等 P1 的入口：**

```sh
cd JevLoop                       # 注意：在 JevLoop/ 里，不是伞仓库
node --experimental-strip-types experiments/scripts/<你的脚本>.ts
```

**数据集和索引一律落 `experiments/dataset/`，日志一律落 `experiments/log/`。**
不要写到 `/tmp`（别人找不到），不要写到仓库别的目录（会被 `npm run check` 数进去）。

## 6. 提交之前必须跑（报告只写你**实际跑过**的）

```sh
cd JevLoop
npx tsc --noEmit      # 0 错
npm run check         # 格式 / 分层 / 文件体量 —— experiments/ 已纳入
npm test              # 当前 311 个
npm run demo          # 退出码 0
```

`experiments/**/*.ts` **已加进 `check.ts` 的 `ROOTS`**，所以你的文件会被查：
单引号、不写分号、2 空格缩进、LF、文件末尾恰好一个换行。
**但不需要登记层号** —— 分层只查 `src/`。

其余协作规矩见伞仓库 `AGENTS.md`：**单写者**（开工前在 `docs/STATUS.md` 声明）、
`git add` 显式列路径、不用 `--amend`、commit message 英文。
