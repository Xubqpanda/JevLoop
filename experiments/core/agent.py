"""★ **第三个协议:`Agent`。**

一个 pipeline 如果只知道 `benchmark × baseline` 两个轴,那么「我们的方法」
（跑在 TS 那一侧）就只能变成 runner 里的一个 `if arm == "jevloop"` 特例 ——
而那正是「两把尺子」长出来的地方:特例的记账、计时、失败处理都会慢慢和别的臂不一样。

所以这里是三个协议:

    Benchmark  ── 数据集 + 工具 + 评分        （实习生写）
    Agent      ── 拿工具解题                   （见下）
    runner     ── 把两者组合,别的什么都不做

**收益不是洁癖,是这一条**:以下五件事全都只是「另一个 `Agent` 实现」——
六个范式 baseline · 我们的 JevLoop（子进程过 JSON 协议）· B3 消融（同一个 loop、
问题改由大模型答）· 帧消融的各个变体 · 「一步一请求」变体。
**跑哪个只是配置,不是代码分支。**

★ **记账归 `Session`,不归 agent。** agent 只说它做了什么;
`usage` / `decisions` / 计时由 `Session` 统一收 —— 六个 baseline 各写一遍记账,
就会长出六种口径,而我们要比较的恰恰是这些数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from experiments.core.models import Message, ModelClient, ModelReply
from experiments.core.spec import DecisionRecord, UsageRecord
from experiments.core.tools import ToolExecutor
from experiments.core.types import Action, Step, Task, Tool, Trajectory


@dataclass
class AgentOutcome:
    """agent 的产出。**只有它做了什么,没有它花了多少** —— 花费由 `Session` 记。"""

    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    escalated: bool = False
    error: str | None = None


class Agent(Protocol):
    """拿一批工具解一道题。

    **实现里不许自己计时、不许自己数 token** —— 用 `session.call_model()` /
    `session.call_tool()` / `session.record_decision()`,它们会记账。
    """

    name: str

    # ★ 声明「这个臂要吃环境给的成败信号」。默认 False。
    #   runner 只在它为 True 时才把 `session.check_answer` 接上 ——
    #   于是「谁偷看了答案」这件事在代码里是显式的,而不是靠自觉。
    needs_success_signal: bool = False

    def solve(self, session: "Session") -> AgentOutcome:
        ...


class Session:
    """一次 (任务 × arm × seed) 的共享设施 + 唯一记账处。

    生命周期:runner 为每道题建一个,交给 agent,回来时已经把账收齐。
    """

    def __init__(
        self,
        *,
        run_id: str,
        task: Task,
        arm: str,
        tools: list[Tool],
        executor: ToolExecutor,
        model: ModelClient,
        max_steps: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.arm = arm
        self.tools = tools
        self.executor = executor
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model_calls: list[UsageRecord] = []
        self.decisions: list[DecisionRecord] = []
        # ★ 每一步真正发出去的 prompt。
        #   两个臂的差别如果只能靠读代码确认,那「baseline 含义一致」就没有证据。
        #   而 ReAct 每步重发整个 scratchpad,所以这一份会长得很快 —— 它是 log,不进 git。
        self.prompts: list[dict] = []
        # ★ 每步模型的**原始输出**。
        #   只记 prompt 不记输出,就查不出「为什么它只输出了 1 个 token」这类问题 ——
        #   我们刚踩过:两条互相冲突的指令,从 prompt 上看不出来,只有输出证明它听了哪条。
        self.completions: list[dict] = []

        # ★ 成功信号 —— **只有声明需要它的臂才拿得到。**
        #
        #   Reflexion 原文的 Evaluator 用的是环境的成败信号（ALFWorld 那种）,
        #   而在这里「成败」只有 benchmark 知道。所以它必须是一条**显式的接缝**,
        #   不能靠 agent 偷看 `task.gold`（那就成了每个臂都能作弊）。
        #
        #   谁用了它,谁的名字里就写清楚（`reflexion` vs `reflexion-selfeval`）——
        #   **一个吃了金标信号的臂,读数时不能和没吃的一样。**
        self.check_answer: "Callable[[str], bool] | None" = None
        self._retry_ms = 0.0
        self._batch = 0

    # —— 给 agent 用的三件事 ──────────────────────────────────

    def call_model(self, messages: list[Message]) -> ModelReply:
        """一次对话补全,并记账。"""
        self.prompts.append(
            {
                "run_id": self.run_id,
                "task_id": self.task.task_id,
                "call": len(self.model_calls),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            }
        )
        t0 = time.perf_counter()
        reply = self.model.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        elapsed = (time.perf_counter() - t0) * 1000

        self.completions.append(
            {
                "run_id": self.run_id,
                "task_id": self.task.task_id,
                "call": len(self.model_calls),
                "text": reply.text,
                "reasoning_content": reply.reasoning_content,
                "tool_calls": list(reply.tool_calls),
                "output_tokens_visible": reply.output_tokens_visible,
                "output_tokens_reasoning": reply.output_tokens_reasoning,
            }
        )

        self.model_calls.append(
            UsageRecord(
                run_id=self.run_id,
                task_id=self.task.task_id,
                kind="llm",
                name=self.model.name,
                input_tokens_cached=reply.input_tokens_cached,
                input_tokens_uncached=reply.input_tokens_uncached,
                output_tokens_reasoning=reply.output_tokens_reasoning,
                output_tokens_visible=reply.output_tokens_visible,
                handshake_ms=reply.handshake_ms,
                ttft_ms=reply.ttft_ms,
                after_ttft_ms=reply.after_ttft_ms or elapsed,
            )
        )
        return reply

    def call_tool(self, name: str, arguments: dict[str, Any], *, necessary: bool = True) -> str:
        return self.executor.call(self.run_id, self.task.task_id, name, arguments, necessary=necessary)

    def next_batch(self) -> int:
        """开一批判定请求。

        ★ **按批计时,不按记录。** 一次请求判多路时,每一路都记同一份 latency
        再求和,会多算几倍 —— 实测 `decisionMs` 一度比整轮墙钟还大。
        """
        self._batch += 1
        return self._batch

    def record_decision(
        self,
        *,
        step: int,
        node: str,
        answer: str,
        confidence: float,
        correct: bool,
        latency_ms: float,
        batch: int | None = None,
    ) -> None:
        """★ `confidence` 和 `correct` **记在同一行**。RQ2 全靠这一对。"""
        self.decisions.append(
            DecisionRecord(
                run_id=self.run_id,
                task_id=self.task.task_id,
                step=step,
                node=node,
                answer=answer,
                confidence=confidence,
                correct=correct,
                batch=self._batch if batch is None else batch,
                latency_ms=latency_ms,
            )
        )

    def note_retry(self, ms: float) -> None:
        """重试/退避的耗时。**单独攒着** —— 混进模型时间里会让失败的臂看起来更慢。"""
        self._retry_ms += ms

    # —— 收账 ────────────────────────────────────────────────

    def usage_records(self) -> list[UsageRecord]:
        return [*self.model_calls, *self.executor.records]

    def trajectory(self, outcome: AgentOutcome) -> Trajectory:
        return Trajectory(
            task_id=self.task.task_id,
            arm=self.arm,
            steps=tuple(outcome.steps),
            final_answer=outcome.final_answer,
            decisions=tuple(d.__dict__ for d in self.decisions),
            usage=tuple(r.__dict__ for r in self.usage_records()),
            escalated=outcome.escalated,
            error=outcome.error,
        )

    def retry_ms(self) -> float:
        return self._retry_ms


def action_tool(name: str, arguments: dict[str, Any]) -> Action:
    return Action(kind="tool", name=name, arguments=arguments)


def action_answer(content: str) -> Action:
    return Action(kind="answer", content=content)


def action_ask(reason: str) -> Action:
    """★ 我们的设计独有的一种动作。**必须能被评测看见** ——
    否则「拒答率」和「闸门假拒」这两个指标算不出来,而它们只有我们有。
    """
    return Action(kind="ask", content=reason)
