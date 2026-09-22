"""`act` —— 只有动作,没有推理。

**出处**:这一臂有两个来源,都标出来,因为「Act 是什么」在文献里指过两样东西:

1. **ReAct 自己的消融** —— Yao et al., arXiv:2210.03629 的 ablation,
   把 Thought 去掉只留 Action/Observation（原文 §3 把它记为 *Act*）。
2. **ReWOO 的 baseline 表**里那一行 —— Xu et al., arXiv:2305.18323 §3.1。

两者是同一个形状:**同一个循环,只是不产生 `Thought`。**

★ **为什么这个文件必须存在**：我们现有的 `bench/react.ts` 的 system prompt 里
`thought` 出现 **0** 次 —— 也就是说它跑出来的本来就是 act,而标签写着 ReAct。
把两条臂拆成两个文件、两个名字之后,这件事就再也不能悄悄发生。

★ 它和 `react` 的实现差别是**一个布尔值**（`LoopConfig.with_thought`）,
这正是 `baseline/common.py` 存在的意义:两条臂只差在被测的那个变量上,
不差在格式、解析或计时的实现上。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.baseline.common import LoopConfig, run_loop
from experiments.core.controller import Controller  # noqa: F401  (类型注解用)
from experiments.core.agent import AgentOutcome, Session

INSTRUCTION = (
    "Solve the task by calling the tools below. "
    "You may call them as many times as needed, then give the final answer."
)


@dataclass
class Act:
    """Act 臂。"""

    exemplars: str = ""
    max_parse_retries: int = 2
    name: str = "act"
    # ★ **控制器** —— 论文第一根轴:同一个循环,换一个「谁来回答下一步」。
    #   `None` = `LLMController`（生成 + 解析,决定藏在生成里）。
    controller: "Controller | None" = None
    config: LoopConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = LoopConfig(
            name="act",
            instruction=INSTRUCTION,
            # ★ 与 react 的唯一区别
            with_thought=False,
            exemplars=self.exemplars,
            max_parse_retries=self.max_parse_retries,
            # ★ **控制器** —— 论文第一根轴。默认 `None` = `LLMController`
            #   （生成 + 解析,决定藏在生成里,也就是今天在用的那个）。
            #   传 `TypedController` 就是同一循环的另一个格子。
            controller=self.controller,
        )

    def solve(self, session: Session) -> AgentOutcome:
        return run_loop(session, self.config)
