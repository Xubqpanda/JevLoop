"""工具执行器。**工具由 benchmark 定义,由这里派发。**

★ 为什么计时放在这里而不是 baseline 里:六个 baseline 各写一遍计时,
就会有六种口径 —— 而我们恰恰要靠这个数去比较它们。
放在共享的一处,「同一把尺子」就是结构性的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from experiments.core.spec import UsageRecord
from experiments.core.types import Tool

ToolImpl = Callable[..., str]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolNotFound(KeyError):
    """调了一个不存在的工具。

    **不许静默返回空串** —— 那会让「选错了工具」看起来像「工具返回了空」,
    正好把我们要测的那件事抹掉。
    """


class ToolExecutor:
    """按名字派发,并记下**每一次**调用的耗时与用量。

    `necessary` 那一列留给 baseline 自己标:同样是「调了三次工具」,
    「三次都是必需的」和「两次在探索」在成本分析里含义完全不同
    （有的评分器**明确不惩罚探索性调用**,所以要和它可比就必须拆开）。
    """

    def __init__(self, tools: list[Tool], impls: dict[str, ToolImpl]) -> None:
        self.tools = list(tools)
        self._impls = dict(impls)
        missing = [t.name for t in self.tools if t.name not in self._impls]
        if missing:
            # 声明了工具却没给实现 —— 早炸。留着它会让 agent 在运行时才撞墙
            raise ValueError(f"工具声明了但没有实现: {missing}")
        self.records: list[UsageRecord] = []
        self.calls: list[ToolCall] = []

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def call(self, run_id: str, task_id: str, name: str, arguments: dict[str, Any],
             *, necessary: bool = True) -> str:
        if name not in self._impls:
            raise ToolNotFound(f"{name}（可用: {sorted(self._impls)}）")

        t0 = time.perf_counter()
        try:
            observation = self._impls[name](**arguments)
        finally:
            elapsed = (time.perf_counter() - t0) * 1000

        self.calls.append(ToolCall(name=name, arguments=dict(arguments)))
        self.records.append(
            UsageRecord(
                run_id=run_id,
                task_id=task_id,
                kind="tool",
                name=name if necessary else f"{name}?exploratory",
                compute_ms=elapsed,
            )
        )
        return observation

    def total_ms(self) -> float:
        return sum(r.compute_ms for r in self.records)

    def counts(self) -> tuple[int, int]:
        """(必要, 探索性)。"""
        exploratory = sum(1 for r in self.records if r.name.endswith("?exploratory"))
        return len(self.records) - exploratory, exploratory
