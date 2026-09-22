"""跨模块共用的词汇。

**这里只有数据形状,没有行为。** 谁都可以 import 它,它不 import 任何人 ——
理由和 `src/vocab.ts` 一样:词汇互相指涉是天然的,但机制之间不该互相依赖。

一个刻意的取舍:**字段用 `tuple` 而不是 `list`**。轨迹是**已经发生的事**,
不该在评分阶段被人悄悄改一个元素 —— 那会让「这个分数是哪次跑出来的」失去意义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    """一个工具。**它属于 benchmark,不属于 baseline。**

    为什么强调这一点:`docs/PLAN-*.md` §2 要求所有 arm「同一批工具、同一份工具描述」。
    把工具的定义权放在 benchmark 一侧,这条要求就是**结构性**的,而不是靠自觉。
    """

    name: str
    description: str
    # JSON Schema 形状。判定模型的候选枚举要靠它，所以**必须是闭集**
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Task:
    """一道题。`gold` 只给评分器看,不给 agent 看。"""

    task_id: str
    prompt: str
    gold: Any = None
    # 数据集自带的、可供裁判的证据（金标检索结果等）。`direct + 金标证据` 那一臂用它
    oracle_context: str | None = None
    # ★ **这一道题自己的工具集。**
    #   有些数据集每题给的函数不同（BFCL 就是:每条自带 `function`）,
    #   给一个全局 `tools()` 会让模型去调一个这道题根本没给它的函数。
    #   空元组 = 用 `Benchmark.tools()` 那个全局兜底。
    tools: tuple[Tool, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """agent 做的一件事。`kind` 只有三种,别加第四种。

    - `tool`  调一个工具。`name` + `arguments`
    - `answer` 交答案。`content`
    - `ask`    该问人 / 该弃答。**这是我们的设计独有的一种动作**,
              评测里必须能被看见,否则「拒答率」这个指标算不出来
    """

    kind: str
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""


@dataclass(frozen=True)
class Step:
    """一步 = 一个动作 + 它的观察。`decision_ms` / `model_ms` 逐层留,理由见 PROTOCOL §3.4。

    ★ `thought` 是 ReAct 的 `Thought:` 那一行。**它必须存在这个类型里**,
    否则 scratchpad 渲染不出来,而「ReAct 的 thought 到底有没有用」这个消融
    就无从做起 —— 我们现有的 `bench/react.ts` 正是漏了它,于是跑出来的是 act。
    """
    index: int
    action: Action
    observation: str = ""
    thought: str = ""
    model_ms: float = 0.0
    decision_ms: float = 0.0
    tool_ms: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    """一次运行的完整记录。**评分只看它** —— 所以它必须够全,足以重算任何指标。"""

    task_id: str
    arm: str
    steps: tuple[Step, ...] = ()
    final_answer: str | None = None
    # 逐判定一行：{node, answer, confidence, correct, batch, latency_ms}
    # ★ `confidence` 和 `correct` 必须在同一行 —— RQ2 全部指标从这一对算
    decisions: tuple[dict[str, Any], ...] = ()
    # 逐调用一行，见 PROTOCOL §3.4
    usage: tuple[dict[str, Any], ...] = ()
    escalated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Judgment:
    """评分结果。`correct` 是我们唯一需要的真假,其余是给人看的。"""

    correct: bool
    score: float = 0.0
    detail: str = ""
    # 失败分类（PLAN §3.8）。**逐条打标,不许只报总数**
    failure_class: str | None = None
