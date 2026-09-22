# `benchmark/` —— 数据集接入

**一个数据集一个包。** 每个包实现 `core/bench.py` 的 `Benchmark` 协议。

```
benchmark/
├── toy/toy.py         接口自检用（不是实验的一部分，照它看形状）
└── gsm8k/gsm8k.py     ★ 第一个真 loader —— 后面照着它写
```

---

## ★ 两段式：**先把 loader 写出来跑 `direct`，再决定要不要写评估逻辑**

这是所有者定的流程（`docs/PLAN-*.md` §0.9）。**不要一口气写完。**

```
① 写 loader（tasks / tools / tool_impls / score 的最简版）
② 只跑 direct —— 便宜、不用工具、不依赖任何基础设施
③ 看分数：
     过高（已刷爆）  → 舍弃这个数据集，不要再投入
     分数合适        → 才去接它的完整评估逻辑与官方评分器
```

**跑法**（在 `JevLoop/` 下）：

```sh
python3 -m experiments.scripts.run --benchmark <名字> --agent direct \
    --seed 0 --limit 300 --model deepseek-chat --base-url https://api.deepseek.com/v1
```

`--limit 300` 是**必须的**，理由在下面「四条容易写错的」第 2 条。

---

## 要实现什么

| 成员 | 干什么 | 最容易漏的地方 |
|---|---|---|
| `name` | 数据集标识，**同时是 `log/<name>/` 的目录名** | 必须和 `docs/PLAN-*.md` §0.7 的命名表一致，别自己起 |
| `dataset_version` | ★ 版本，逐字对上 `MANIFEST-datasets-*.md` | **能拿到 fingerprint 就用它**（见 `gsm8k`） |
| `tasks(split, limit, seed)` | 产出 `Task` | 抽样必须用传进来的 `seed` |
| `tools()` | 这一批题可用的工具 | **声明了就要有实现**，否则开跑前就抛 |
| `tool_impls()` | 工具的实现 | 同上 |
| `check(task, answer) -> bool` | **可选**，只回答对不对，给 Reflexion 的 Evaluator | 没实现时那类臂**拿不到成败信号**，会抛而不是静默降级 |
| `score(task, trajectory)` | 完整判分 | **尽量调官方评分器**，别自己重写 |

---

## ★ 四条容易写错的（都是踩过或马上就踩的）

### 1. 判分口径**必须写明**，而且**只报一个数是不够的**

`gsm8k` 是活例子：它的分数对「怎么从模型输出里抠出那个数」**极其敏感**，
而两种抠法都有出处（`strict` 要 `####`，`flexible` 取最后一个数）。

**两个都算，报的时候写明报的是哪个。** 只报一个还不说口径 = 别人复现不出你的数。
其它数据集同理：**先去找它的官方评分器，找不到才自己写，写了就在文件头写明依据。**

### 2. 抽样必须由传进来的 `seed` 决定

否则「我们跑的是哪 300 条」**说不清**，而那意味着这个数不可复现。

⚠️ 顺带一个刚踩的坑：写 `seed % 1` 这种「用一下 seed」的样子货很容易 ——
**它对 int 恒等于 0，种子等于没起作用。** 我们的第一个版本就是这么写的。

### 3. `gold` 不许泄进 prompt

`Task.gold` 只给评分器看。**`direct + 金标证据` 那一臂读的是 `oracle_context`，不是 `gold`。**

### 4. 抠不出答案要**单独报**，不要混进「答错」

`gsm8k` 把「抠不出数值」记成 `no_numeric_answer`，和 `wrong_answer` 分开。
混在一起就看不出**是模型不会算，还是它没按格式给答案** —— 而这两件事的修法完全不同。

---

## 加一个数据集的清单

```
□ 建 benchmark/<name>/<name>.py 和 __init__.py（__init__.py 一行 re-export）
□ 文件头写：出处（论文号 + 数据 id）、**判分口径**、以及它验不到什么
□ 实现 tasks / tools / tool_impls / check / score
□ dataset_version 用能自动拿到的东西（fingerprint / commit），别手写
□ 在 scripts/run.py 的 _builtin_benchmarks 里注册一行
□ 在 tests/ 里加测试 —— **用喂 rows 的方式，不联网**
□ 跑一次 direct（--limit 300），把分数记下来，再决定要不要继续投入
```

---

## ★ 现在的覆盖情况（写第二个的时候看这里）

| 验到了 | 没验到 |
|---|---|
| 取数据 / 缓存 / 版本号 | **工具定义（`tools()` + `tool_impls()`）** |
| 按 seed 抽样 | 多轮交互（每步一个新观察） |
| 从答案字段抠金标 | 带标注的判定（`needsTool` 这类**该选哪个**的标签） |
| 对齐判分口径（strict / flexible） | 官方评分器（`gsm8k` 是纯数值比较，没有独立评分器） |
| 失败分类的两种分法 | 检索 / 环境 / 模拟器 |

**`gsm8k` 一个工具都没有**（`tools()` 返回空）。所以**第二个 loader 必须挑一个带工具的**，
把「工具怎么定义才让判定模型能枚举」这条验掉 —— 那是我们和六个 baseline 共用的前提。
