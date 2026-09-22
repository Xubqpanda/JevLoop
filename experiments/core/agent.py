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

from experiments.core.events import (
    Clock,
    DecisionBatchEvent,
    Event,
    ModelCallEvent,
    SpanEvent,
    Timing,
    ToolCallEvent,
    Usage,
    hash_text,
    now_iso,
)
from experiments.core.models import Message, ModelClient, ModelReply
from experiments.core.spec import DecisionRecord
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

        # ★★ **一条事件流,不是三份平行列表。**
        #
        #   移植自 Inspect 的 transcript 模型（`ModelEvent` / `SpanBeginEvent` …,
        #   见 `core/events.py` 的说明与 `docs/PROBE-inspect-2026-09-22.md`）:
        #   **一次调用一条记录,自己装齐请求 + 响应 + 用量 + 时间 + 重试 + 错误。**
        #
        #   在这之前是 `prompts.jsonl` / `completions.jsonl` / `usage.jsonl`
        #   三个文件靠 `call` 序号 join —— 任何一处漏写,那次调用就**看起来像没发生过**,
        #   或者更糟:token 数和文本对不上,而账面看不出来。
        self.events: list[Event] = []
        self.clock = Clock()
        self._span_stack: list[str] = []
        self._batch_uuid: str | None = None
        self._batch_open_ms: float = 0.0
        self._batch_decisions: list[DecisionRecord] = []
        # 工具执行器把每次调用报回来 —— 见 tools.ToolExecutor.on_call
        self.executor.on_call = self._on_tool_call

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
        working_start = self.clock.elapsed()
        started = now_iso()
        t0 = time.perf_counter()
        reply = self.model.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        elapsed = (time.perf_counter() - t0) * 1000
        event = ModelCallEvent(
            run_id=self.run_id,
            task_id=self.task.task_id,
            step=self.call_index(),
            # ★ 请求和响应**装在同一条记录里** —— 这是从 Inspect 搬来的那条
            request={
                "model": self.model.model_id,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "prompt_hash": hash_text("\n".join(m.content for m in messages)),
            },
            response={
                "text": reply.text,
                "reasoning_content": reply.reasoning_content,
                "tool_calls": list(reply.tool_calls),
            },
            usage=Usage(
                input_tokens_cache_read=reply.input_tokens_cached,
                input_tokens=reply.input_tokens_cached + reply.input_tokens_uncached,
                output_tokens=reply.output_tokens_visible + reply.output_tokens_reasoning,
                reasoning_tokens=reply.output_tokens_reasoning,
            ),
            timing=Timing(
                handshake_ms=reply.handshake_ms,
                ttft_ms=reply.ttft_ms,
                after_ttft_ms=reply.after_ttft_ms or elapsed,
            ),
            model=self.model.name,
            timestamp=started,
            completed=now_iso(),
            working_start=working_start,
            parent=self._span_stack[-1] if self._span_stack else None,
        )
        event.working_time = event.timing.total_ms
        self.events.append(event)
        return reply

    def call_tool(self, name: str, arguments: dict[str, Any], *, necessary: bool = True) -> str:
        return self.executor.call(self.run_id, self.task.task_id, name, arguments, necessary=necessary)

    def next_batch(self) -> int:
        """开一批判定请求。**上一批在这里收口。**

        ★ **按批计时,不按记录。** 一次请求判多路时,每一路都记同一份 latency
        再求和,会多算几倍 —— 实测 `decisionMs` 一度比整轮墙钟还大。
        """
        self._close_batch()
        self._batch += 1
        self._batch_open_ms = time.perf_counter()
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
        self._batch_decisions.append(
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

    def span(self, name: str):
        """给一段工作计时。**`with session.span("decision"):`** —— 成对发事件。

        移植自 `SpanBeginEvent` / `SpanEndEvent`。有了它,一步的耗时不用手写计时。
        """
        return _Span(self, name)

    # —— 事件 ────────────────────────────────────────────────

    def call_index(self) -> int:
        return sum(1 for e in self.events if isinstance(e, ModelCallEvent))

    def model_events(self) -> list[ModelCallEvent]:
        return [e for e in self.events if isinstance(e, ModelCallEvent)]

    def tool_events(self) -> list[ToolCallEvent]:
        return [e for e in self.events if isinstance(e, ToolCallEvent)]

    def decision_batches(self) -> list[DecisionBatchEvent]:
        return [e for e in self.events if isinstance(e, DecisionBatchEvent)]

    def decision_records(self) -> list[DecisionRecord]:
        """把各批摊平 —— RQ2 的输入就是这一串。"""
        return [DecisionRecord(**d) for b in self.decision_batches() for d in b.decisions]

    @property
    def prompts(self) -> list[dict]:
        """兼容视图:每一步真正发出去的 prompt。

        两个臂的差别如果只能靠读代码确认,那「baseline 含义一致」就没有证据。
        底层已经是事件了,这里只是把它摊成旧形状给测试和调试用。
        """
        return [
            {"call": i, "task_id": e.task_id, **e.request}
            for i, e in enumerate(self.model_events())
        ]

    @property
    def completions(self) -> list[dict]:
        """兼容视图:每一步模型的**原始输出**。

        只记 prompt 不记输出,就查不出「为什么它只输出了 1 个 token」这类问题 ——
        我们刚踩过:两条互相冲突的指令,从 prompt 上看不出来,只有输出证明它听了哪条。
        """
        return [{"call": i, "task_id": e.task_id, **e.response}
                for i, e in enumerate(self.model_events())]

    def _close_batch(self) -> None:
        if not self._batch_decisions:
            return
        working_start = self.clock.elapsed()
        self.events.append(
            DecisionBatchEvent(
                run_id=self.run_id,
                task_id=self.task.task_id,
                step=self._batch_decisions[0].step,
                questions_in_batch=len(self._batch_decisions),
                latency_ms=(time.perf_counter() - self._batch_open_ms) * 1000 if self._batch_open_ms else 0.0,
                decisions=[d.__dict__ for d in self._batch_decisions],
                working_start=working_start,
                parent=self._span_stack[-1] if self._span_stack else None,
            )
        )
        self._batch_decisions = []

    def _on_tool_call(self, name: str, arguments: dict, necessary: bool, ms: float, error: str | None) -> None:
        self.events.append(
            ToolCallEvent(
                run_id=self.run_id,
                task_id=self.task.task_id,
                step=self.call_index(),
                name=name,
                arguments=dict(arguments),
                necessary=necessary,
                error=error,
                working_start=self.clock.elapsed(),
                working_time=ms,
                parent=self._span_stack[-1] if self._span_stack else None,
            )
        )

    def finish_events(self) -> None:
        """收尾:把最后一批判定封口。**不封口最后一批就会丢** —— 而它常常正是决策性的那批。"""
        self._close_batch()

    def trajectory(self, outcome: AgentOutcome) -> Trajectory:
        return Trajectory(
            task_id=self.task.task_id,
            arm=self.arm,
            steps=tuple(outcome.steps),
            final_answer=outcome.final_answer,
            decisions=tuple(d.__dict__ for d in self.decision_records()),
            usage=(),  # 逐调用明细在 events.jsonl 里,不在这里复制一份
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


class _Span:
    """`with session.span("name"):` —— 成对发 begin / end 事件。"""

    def __init__(self, session: Session, name: str) -> None:
        self.session = session
        self.name = name
        self.uuid = ""
        self.t0 = 0.0

    def __enter__(self) -> "_Span":
        from experiments.core.events import new_uuid

        self.uuid = new_uuid()
        self.t0 = self.session.clock.elapsed()
        self.session._span_stack.append(self.uuid)
        self.session.events.append(SpanEvent(
            run_id=self.session.run_id, task_id=self.session.task.task_id,
            name=self.name, phase="begin", uuid=self.uuid, working_start=self.t0,
        ))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.session._span_stack.pop()
        end = self.session.clock.elapsed()
        self.session.events.append(SpanEvent(
            run_id=self.session.run_id, task_id=self.session.task.task_id,
            name=self.name, phase="end", uuid=self.uuid,
            working_start=end, working_time=(end - self.t0) * 1000,
            metadata={"error": f"{exc_type.__name__}: {exc}" if exc else None},
        ))
        return False  # 不吞异常
