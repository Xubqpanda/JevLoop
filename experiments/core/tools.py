"""工具执行器。**工具由 benchmark 定义,由这里派发。**

★ 为什么计时放在这里而不是 baseline 里:六个 baseline 各写一遍计时,
就会有六种口径 —— 而我们恰恰要靠这个数去比较它们。
放在共享的一处,「同一把尺子」就是结构性的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from experiments.core.types import Tool

ToolImpl = Callable[..., str]

# ★ 每次调用报一次:`(名字, 参数, 是否必要, 耗时ms, 错误)`。
#   形状**对齐 Inspect 的 ToolEvent** —— 由 `Session` 转成事件落盘。
#   为什么不在这里直接写文件:执行器不知道 run 的时钟、也拿不到 span 栈。
#: 每次工具调用的回报。**参数顺序就是事件的字段顺序。**
#:
#: ⚠️ 第 6 个 `observation` 是补上的 —— 见 `call()` 里那段说明。
#: 它是**位置参数**,改签名前先 `grep -rn on_call experiments/`:
#: 少传一个参数不会报错,只会让日志里那一栏永远是空的。
OnCall = Callable[[str, dict, bool, float, "str | None", str], None]


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
    """按名字派发,并把**每一次**调用报给 `Session`。

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
        # ★ 不自己攒记录 —— 报给 `Session`,由它发事件（见 core/events.py）。
        #   各攒各的正是「三份平行列表」那个坑的开始。
        self.on_call: OnCall | None = None
        self._total_ms = 0.0
        self._necessary = 0
        self._exploratory = 0
        self.calls: list[ToolCall] = []

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def call(self, run_id: str, task_id: str, name: str, arguments: dict[str, Any],
             *, necessary: bool = True) -> str:
        if name not in self._impls:
            raise ToolNotFound(f"{name}（可用: {sorted(self._impls)}）")

        t0 = time.perf_counter()
        error: str | None = None
        try:
            observation = self._impls[name](**arguments)
        except Exception as exc:  # noqa: BLE001 —— 工具自己抛了,要报出去而不是吞掉
            # ★ 不吞:错误原文进事件。**「是模型选错了还是工具坏了」靠这个分。**
            observation = f"Tool error: {type(exc).__name__}: {exc}"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            elapsed = (time.perf_counter() - t0) * 1000

        self.calls.append(ToolCall(name=name, arguments=dict(arguments)))
        self._total_ms += elapsed
        if necessary:
            self._necessary += 1
        else:
            self._exploratory += 1
        if self.on_call is not None:
            # ★★★ **`observation` 必须一起报出去 —— 它曾经被漏在这里,而没人发现。**
            #
            #   实测（2026-09-23）:`ToolCallEvent` 有 `observation` 字段,
            #   而这条回调**没有那个参数**,于是它永远取默认值 `""`。
            #   后果不是报错,是**日志里那一栏一直是空的** ——
            #   于是「agent 当时看到了什么」这件事**从任何一批跑里都查不到**。
            #
            #   ★ 我因此差点写下一个错的结论:看到 BFCL 的 observation 全是 `""`,
            #     以为「BFCL 的工具不返回任何东西」。**其实返回了一整句话**
            #     （见 `benchmark/bfcl/bfcl.py::_not_executed` 那段 —— 那句话的措辞
            #     本身就是一次修好的事故）。**是日志漏了,不是工具空。**
            #
            #   ⇒ §8.10 那个形状的又一例:**一个字段静默地永远是默认值,
            #     和这个字段不存在,在读数的人眼里完全一样。**
            self.on_call(name, arguments, necessary, elapsed, error, observation)
        return observation

    def total_ms(self) -> float:
        return self._total_ms

    def counts(self) -> tuple[int, int]:
        """(必要, 探索性)。★ 分开数 —— 有的评分器**明确不惩罚探索性调用**,
        不拆的话我们的调用数和它不可比。"""
        return self._necessary, self._exploratory
