# `baseline/` —— 六个范式的实现

**一个范式一个包。** 每个包里的 `<name>.py` 头部写清了**对齐原文的哪几条**,
以及**任何偏离都要在那里写明** —— 没有出处就没法核对「含义一致」。

```
baseline/
├── common.py              ★ 六个臂共用的基础设施（见下）
├── direct/direct.py       zero-shot，下界
├── act/act.py             只有动作
├── react/react.py         Thought + Action + Observation
├── reflexion/reflexion.py 失败后写反思，带进下一次
├── plan_then_execute/     先规划，再带计划执行
└── rewoo/rewoo.py         一次盲规划 → 收集证据 → 综合
```

`figures/` 放各论文的方法插图。**不进 git** —— 见下。

---

## ★ 论文出处（写论文时的引用表）

| 臂 | 出处 |
|---|---|
| `direct` | ReWOO §3.1 Baselines 对 zero-shot 的定义 · [arXiv:2305.18323](https://arxiv.org/abs/2305.18323) |
| `act` | ReAct 自己的消融（去掉 Thought）· [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| `react` | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023 · [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| `reflexion` | Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023 · [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) |
| `plan_then_execute` | ⚠️ **见下方更正** |
| `rewoo` | Xu et al., *ReWOO: Decoupling Reasoning from Observations...* · [arXiv:2305.18323](https://arxiv.org/abs/2305.18323) |

### ⚠️ `plan_then_execute` 的出处更正

**不是** *Plan-Then-Execute: An Empirical Study of User Trust and Team Performance
When Using LLM Agents As A Daily Assistant*（He, Demartini, Gadiraju，CHI 2025，
[arXiv:2502.01390](https://arxiv.org/abs/2502.01390)）。

那一篇是 **cs.HC 的实证用户研究（N=248）**，研究的是**人对 agent 的信任**和团队表现。
它自己写的是「**we adopted** LLM agents in a plan-then-execute manner」——
**它采用这个范式做实验，不是提出这个范式。** 引它可以，但只能引成
「有人用这个范式做过人机协作研究」，不能引成方法出处。

**架构出处是这两条：**

1. **Plan-and-Solve Prompting** —— Wang, Xu, Fang, Liu, Zhang, Yang.
   *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models.*
   ACL 2023 · [arXiv:2305.04091](https://arxiv.org/abs/2305.04091)。
   原话是「Let's first understand the problem and devise a plan to solve the problem.
   Then, let's carry out the plan and solve the problem step by step.」
2. **Plan-and-Execute agents** 的工程写法（LangChain，2023）——
   把这个提示法变成「planner 出计划、executor 逐步执行、必要时 replan」的结构。

**这一族的其他成员**（写 related work 会用到，都不是本文）：HuggingGPT（2303.17580）、
BabyAGI（2023）、ADaPT（2311.05772）、LLM+P（2304.11477）、
以及 **ReWOO（2305.18323，本目录里那个：它是「规划一次、但不带观察反馈」的特例）**。

> **教训**：名字撞车不等于同一件事。**引用前核一次摘要**，
> 「Plan-Then-Execute」在两个社区里指的是两样东西。

---

## `figures/` 与版权

`baseline/<arm>/figures/` 放论文的方法插图，**在 `.gitignore` 里，不进仓库**。两个理由：

1. **版权**：arXiv 的授权**因文而异**（有的 CC-BY 可以带署名转载，有的不是）。
   这些图不是我们画的，也不加任何许可证就能公开分发。
2. **它本来就是读书材料**，不是代码的一部分。

**要引用方法图，就引论文本身**，不要把图搬进我们的仓库。

---

## 六个臂只该差在被测变量上 —— 靠 `common.py` 保证

| 共享什么 | 不共享会怎样 |
|---|---|
| `render_tools()` | **工具描述必须逐字一致**。各写一份措辞，措辞差别就混进方法差别 |
| `parse_step()` | 解析器不同 = 同一个模型输出在 A 臂被认成工具调用、在 B 臂被认成答案 |
| `render_history()` | 上下文形状不同 = **token 数不可比** |
| `run_loop()` | 循环骨架相同，**差别只在配置** |

**`LoopConfig.with_thought` 就是 `act` 与 `react` 的全部区别**，有测试盯着
「截断点 / 重试 / 记账必须逐项相同，只有它不同」。

---

## 加一个臂

1. 建 `baseline/<name>/`，里面放 `<name>.py`（实现 `core/agent.py` 的 `Agent`）和 `__init__.py`；
2. **文件头写清对齐哪篇论文的哪一节**，偏离也要写；
3. 能在 `common.py` 里复用的一律复用 —— 新写一个循环之前先问「这真的不一样吗」；
4. 在 `scripts/run.py` 的 `_BUILTIN_ARMS` 里注册；
5. 在 `tests/test_baselines.py` 里加测试。**解析和边界条件是重点**，不是准确率。

**尽量用 `run_loop`。** `plan_then_execute` 是这么做的：它的执行段就是
「`run_loop` 加一段 preamble」，**一行循环都没新写** —— 这是 `common.py` 划得对不对的检验。
