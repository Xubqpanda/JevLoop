# `dataset/` —— **只放索引，不放数据**

> ## ★ 这个目录里进 git 的，只有这份文件和 `.gitkeep`。
>
> **数据一个字节都不上传。** 19 个数据集 + 一个 Wikipedia 索引是**上百 GB**，
> 而 `experiments/dataset/*` 在 `.gitignore` 里（`JevLoop/.gitignore`）。
>
> **这份文件是「怎么把它拿回来」的索引。** 别人 clone 之后照着跑，就能重建出同样的数据 ——
> 而**版本必须逐字一致**，否则数字和文献对不上。

---

## 1. 三条规矩

| | |
|---|---|
| **① 数据只落 `experiments/dataset/`** | 不许放 `/tmp`（别人找不到 = 没跑过），不许放仓库别处（会被 `check` 数进去）|
| **② 版本写进这里才能用** | 数据集版本、划分、**Wikipedia dump 版本** —— 三样缺一不可 |
| **③ 拿不准的标 `待核`** | **不许用记忆里的数字填上去。** 这一列是这份文件存在的理由 |

---

## 2. 数据集清单

`benchmark/` 下的文件名见 `experiments/README.md` §3。**规模一列：`子集` 是我们实际要跑的条数，
`总量` 是数据集本身的规模 —— 两者不是一个数，别混。**

| 名称 | 取哪些 | 规模 | 版本 / 划分 | 状态 |
|---|---|---|---|---|
| **BFCL-v4** | 计分 **5,088** + 不计分 **5,218** | Multi-Turn **800**（`base` / `long_context` / `miss_func` / `miss_param` 各 200）· Memory **465**（`kv` / `rec_sum` / `vector` 各 155）· Web Search **200**（`base` / `no_snippet` 各 100）· **Format Sensitivity 5,200 = 26 × 200（不计分）** | 钉 `bfcl-eval`；**两处版本号不一致**：EvalScope 文档 `2025.10.27.1`，官方 leaderboard `2025.12.17` / commit `f7cf735`。**用哪个写哪个** | ✅ 口径已核<br>⚠️ 版本待定 |
| **GAIA-2** | 先 `mini`，再全量 | **mini 160**；全量 **800 + 320 = 1,120**；环境 **101 个工具** | ARE 平台；HF `meta-agents-research-environments/gaia2`；pass@1 取 **3 次平均** | ✅ |
| **τ²-bench** | 3 个域全取 | **279** = retail 115 + airline 50 + **telecom 114**（生成池 2,285 抽样）| repo `sierra-research/tau2-bench`；**pass^k = 4 次/任务，temperature 0** | ✅ |
| **τ³-bench**（扩展）| 只取 `banking_knowledge` | **375** = airline 50 + retail 114 + telecom 114 + **banking 97**；banking 带 **698** 份政策文档 | 作为 τ² 的扩展域，不单独跑全套 | ✅ |
| **AppWorld** | `Test-N` 迭代，`Test-C` 头条 | **750** = 250 场景 × 3；**Train 105 / Dev 60 / Test-N 168 / Test-C 417**；**9 个 app，457 个 API，101 张表** | repo `stonybrooknlp/appworld`；**离线 + 固定种子 + 冻结时间** | ✅ |
| **Terminal-Bench** | 一小块当成本仪器 + 边界反例 | TB **2.0 = 89 任务**；**4.0 删了 8 题、修了 19 题、统一 8 小时超时** | ★ **语义化版本**（major = 必须重跑 / minor = 只重判 / patch = 结果可复用）；家族 15 个月从 **1.0（2025-05-19）走到 4.0（2026-08-28）** | ⚠️ 3.0 / 4.0 题数**待核** |
| **HotpotQA** | ReAct 原设置 | **总量 113k** 问答对，**带句级支撑事实**；ReWOO 用的是 **1000 条子集** | ReAct 原仓库的 whoosh 索引脚本 | ✅ 总量<br>⚠️ 划分待核 |
| **TriviaQA** | 同 HotpotQA 工具集 | ReWOO 用 **1000 条子集** | ⚠️ **他们的设置是「把 reading context 藏起来强迫检索」** —— 复现时必须照做，否则不是同一个任务 | ⚠️ 总量待核 |
| **StrategyQA** | 二值 → `noul` 标签 | ReWOO 用 **300 条子集** | ⚠️ 总量待核 | ⚠️ 待核 |
| **ALFWorld** | 官方 eval = unseen | **3,553** train；**140 seen + 134 unseen** eval；**6 类任务**，120 个房间 | 纯文本模式（TextWorld）；**官方带专家轨迹** → 步级标签 | ✅ |
| **2WikiMultiHopQA** | 多跳，带推理路径 | 总量 `待核`（常见引用的 192,606 **未在原始摘要里核到**）| ★ **带 evidence / reasoning path 标注** → 步级标签免费 | ⚠️ 待核 |
| **MuSiQue** | 多跳 | **MuSiQue-Ans 25K**（2–4 跳）；**MuSiQue-Full** 另加**不可回答**的对照题 | 自底向上合成，**构造上保证不是捷径可解** | ✅ |
| **FEVER** | 3 类裁决 → `canDeliver` | `待核` | ⚠️ **官方标注本身有噪声**（`NOT ENOUGH INFO` 尤其），跑之前想清楚怎么处理 | ⚠️ 待核 |
| **MCP-AgentBench** | **只取 single-server** | `待核`（multi-server ~128 个工具 → **超出候选上界，不取**）| | ⚠️ 待核 |
| **RAGTruth** | span 级「有没有证据支持」→ `canDeliver` | `待核` | | ⚠️ 待核 |
| **TruthfulQA** | 拒答 / 幻觉标签 | `待核` | | ⚠️ 待核 |
| **GSM8K** | **预期 direct 很高**，用来让附录那张表有下行 | ReWOO 用 **1000 条子集** | EvalScope 里也有一份，可交叉核对 | ⚠️ 总量待核 |
| **SportsUnderstanding** | 同上 | ReWOO 用 **300 条子集** | BigBench | ⚠️ 待核 |
| **PhysicsQuestions** | **53 条，低于 ≥300 底线** → 只用于 direct 分档 | ReWOO 用 **53 条** | 这一行标注「不足以画逐节点曲线」| ⚠️ 待核 |

**已砍**：`API-Bank`（太小：314 段评测对话 / 753 次调用，做不了 2026 头条 ——
但它的 `Call → Retrieve+Call → Plan+Retrieve+Call` 三级能力梯是一条干净的诊断轴，记一笔）。

---

## 3. 派生索引（**最大的一块，也最不能上传**）

| 索引 | 服务谁 | 规矩 |
|---|---|---|
| **Wikipedia `search` / `lookup`** | HotpotQA · TriviaQA · 2Wiki · MuSiQue · FEVER 检索 | ★ **四个数据集共用一套，必须由一个人建**（见 `PLAN-*.md` §0.7），否则会长出四个索引 |

**必须固定 dump 版本并写进这一行。** 现在是 `待核` ——
**这是全项目最长的前置（几小时 + 上百 GB），版本没定就不要开工。**

参考：ReAct 原实现用 whoosh 建索引；Search-R1 那条线用的是 **2018 年 Wikipedia dump + E5 retriever、top-3 passages**。
**两条线不是同一个索引，选哪条要写明理由**，因为它决定了我们和谁的数字可比。

**建库步骤也要落在这里**（`scripts/` 下的脚本路径 + 一条能跑的命令）。

---

## 4. 跑之前，把 `待核` 清掉

上面有 **11 处 `待核`**。清的方式是**去一手来源核**，不是问人也不是凭印象：

```
数据集条数 / 划分   → 官方 repo 的 README 或 datasets 页
版本号              → 官方 PyPI / release / commit
Wikipedia dump      → 原仓库的建索引脚本里写死的那个
```

**核完把证据（URL）写进这一行的备注。** 没有 URL 的数字不进这份文件 ——
理由和伞仓库 `AGENTS.md` §8 一样：**没有依据的只是偏好。**
