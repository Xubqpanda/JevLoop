"""`plan-then-execute` —— 先规划,再照着计划执行。

**出处**:两条,都标出来。⚠️ **`baseline/README.md` 里记了一处引用更正,别引错。**

1. **Plan-and-Solve prompting** —— Wang, Xu, Fang, Liu, Zhang, Yang.
   *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning*.
   ACL 2023, arXiv:2305.04091。「先理解问题并制定计划,然后按计划执行」。
2. **Plan-and-Execute 的工程写法**（LangChain 等）—— planner 出计划,
   executor 逐步执行并**看得到每步的观察**,必要时重新规划。


★ **它和 `rewoo` 的区别,是这两条臂必须分开的理由**:

| | ReWOO | plan-then-execute |
|---|---|---|
| 计划里有没有变量 | 有（`#E1 = Tool[...]`,后面可以引用）| 没有,只是子任务排序 |
| 执行时看不看观察 | **不看**（盲规划到底,最后一次性给 Solver）| **看**（执行段是有观察的循环）|
| 能否中途改计划 | 不能 | 本实现：**默认也不重新规划**（见下）|

★ **默认不重新规划**是有意的:所有者要的是「最简原型」,加了 replan 就等于把
planner/executor 的边界又模糊掉一次。`replan=True` 留着,但**默认关**,
这样两条臂的差别始终清楚。

★ 实现上它几乎全部复用 `common`:**执行段就是 `run_loop` 加一段 preamble**
（把计划塞进去）。一处循环也不用重写 —— 这就是 `baseline/common.py` 该有的样子。
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.baseline.common import (
    LoopConfig,
    parse_plan,
    render_plan,
    render_tools,
    run_loop,
)
from experiments.core.agent import AgentOutcome, Session, action_answer
from experiments.core.controller import Controller  # noqa: F401  (类型注解用)
from experiments.core.models import Message
from experiments.core.types import Step

PLANNER_INSTRUCTION = """Break the task into an ordered list of subtasks, then stop.

Rules:
- Each subtask is one concrete thing to find out or do.
- Order them so that later subtasks can use what earlier ones produce.
- Do not answer the task, and do not guess any result.
- Output the list only, one subtask per line, numbered.
"""

EXECUTOR_INSTRUCTION = (
    "Work through the plan below in order. "
    "You may call the tools as many times as needed, then give the final answer."
)


@dataclass
class PlanThenExecute:
    """规划一次 → 带着计划执行（执行段是有观察的循环）。"""

    replan: bool = False
    exemplars: str = ""
    name: str = "plan-then-execute"
    # ★ **控制器** —— 论文第一根轴:同一个循环,换一个「谁来回答下一步」。
    #   `None` = `LLMController`（生成 + 解析,决定藏在生成里）。
    controller: "Controller | None" = None

    def solve(self, session: Session) -> AgentOutcome:
        steps: list[Step] = []

        # ── Planner ────────────────────────────────────────────
        plan_prompt = "\n".join([
            PLANNER_INSTRUCTION,
            "",
            "# Tools",
            render_tools(session.tools),
            "",
            "# Task",
            session.task.prompt,
        ])
        reply = session.call_model([Message(role="user", content=plan_prompt)])
        items = parse_plan(reply.text, [t.name for t in session.tools])

        if not items:
            # 计划解析不出来 → **仍然去执行**(带一个空计划),
            # 而不是弃答。理由:plan-then-execute 的执行段本身是有观察的循环,
            # 它有能力自己把任务做掉 —— 这一点和 ReWOO 不同
            # （ReWOO 的执行段是盲的,没计划就真的没得跑）。
            steps.append(Step(index=0, action=action_answer("(plan unparsed)"),
                              observation="Planner 没给出可解析的计划,直接执行",
                              thought=reply.text))
        else:
            steps.append(Step(index=0, action=action_answer(render_plan(items)),
                              observation=f"{len(items)} 步计划", thought=reply.text))

        # ── Executor：**同一个 run_loop,只是多一段 preamble** ──
        preamble = ""
        if items:
            preamble = "# Plan\n" + render_plan(items)
        if self.replan:
            preamble += (
                "\n\nIf the plan turns out to be wrong, say so explicitly and "
                "revise it before continuing."
            )

        cfg = LoopConfig(
            name=self.name,
            instruction=EXECUTOR_INSTRUCTION,
            # 执行段带 thought:这是「执行器」,不是 act —— 它要在每一步决定下一步做什么
            with_thought=True,
            exemplars=self.exemplars,
            preamble=preamble,
            # ★ **控制器** —— 论文第一根轴。默认 `None` = `LLMController`
            #   （生成 + 解析,决定藏在生成里,也就是今天在用的那个）。
            #   传 `TypedController` 就是同一循环的另一个格子。
            controller=self.controller,
        )
        outcome = run_loop(session, cfg)
        # 把规划那一步并进轨迹,顺序不能反
        return AgentOutcome(
            steps=[*steps, *outcome.steps],
            final_answer=outcome.final_answer,
            escalated=outcome.escalated,
            error=outcome.error,
        )


__all__ = ["PlanThenExecute", "PLANNER_INSTRUCTION", "EXECUTOR_INSTRUCTION"]
