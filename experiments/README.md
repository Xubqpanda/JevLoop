# experiments/

跑评测的管道。**Python 侧,和 `../src/` 的 TS 内核互不依赖。**

| 目录 | 放什么 |
|---|---|
| `dataset/` | 数据集与检索索引。**上百 GB，不进 git** |
| `benchmark/` | 各数据集的加载与评分（实现 `core/bench.py` 的 `Benchmark`）|
| `baseline/` | 各 agent loop 范式的实现（实现 `core/agent.py` 的 `Agent`）|
| `core/` | ★ **三个协议 + 记账**，见下 |
| `scripts/` | 入口命令 |
| `log/` | 每次运行的原始产物。**不进 git** |
| `result/` | 汇总后的表和图 |

`.gitignore` 用的是 `dir/*` 加否定 `.gitkeep`，**不是**直接忽略整个目录 ——
后者连占位文件一起丢掉，于是别人 clone 出来这些目录**根本不存在**，而这份文件说它们存在。

---

## 三个协议

```
Benchmark  数据集 + 工具 + 评分      benchmark/<name>.py
Agent      拿工具解题                baseline/<name>.py
runner     把两者组合，别的什么都不做  core/runner.py
```

**为什么是三个而不是两个。** 如果管道只知道 `benchmark × baseline` 两轴，
那么任何跑在**别的语言**里的实现（本项目的 JevLoop 内核是 TypeScript）就只能变成
runner 里的一个 `if arm == ...` 特例 —— 而特例的记账、计时、失败处理会慢慢和别的臂不一样，
最后比的就不是方法，是两套记账。把 `Agent` 提成协议之后，「另一个实现」就只是另一个实现。

**记账归 `Session`（`core/agent.py`），不归 agent。** agent 只说自己做了什么，
`usage` / `decisions` / 计时由 `Session` 统一收 —— 否则每个 baseline 各记一遍账，
就会长出各不相同的口径，而我们要比较的恰恰是这些数。

**结果 schema 在 `core/spec.py`，必填字段没有默认值** —— 少一个字段是**构造时**的
`TypeError`，不是事后校验。理由同上：等到汇总时才发现，已经跑完一批了。

---

## 跑

```sh
cd ..                       # 到 JevLoop/

# 自检：离线，不要 key，不要网
python3 -m experiments.scripts.run --benchmark toy --agent direct --seed 0 --model offline

# 汇总 log/ → result/（单向，不许手改 result）
python3 -m experiments.scripts.summarize

# 接口的契约测试
python3 -m pytest experiments/tests -q
```

接真模型：`--model <id> --base-url <url>`，key 从 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` 读。

## 加一个数据集 / 加一个 arm

1. 在 `benchmark/` 或 `baseline/` 下建文件，实现对应协议；
2. 在 `scripts/run.py` 里**显式** import 并注册。

第 2 步是刻意的：注册**不做自动发现**。七个人并行时，`benchmark/` 下任何一个文件
写上语法错，自动发现会让全场的运行一起挂 —— 而那个人可能只是刚存了半行。

## 写 loader 时容易漏的四件事

1. **评分尽量调官方评分器。** 自己重写一遍官方判分逻辑等于换判分器，
   而换判分器之后的准确率不可比。
2. **抽样必须用传进来的 `seed`。** 否则「我们跑的是哪 300 条」说不清。
3. **`gold` 不许泄进 prompt。**
4. **工具声明了就要有实现**，名字对不上会在开跑前抛 —— 这是有意的。
