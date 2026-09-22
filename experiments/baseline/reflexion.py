"""`reflexion` —— 失败之后**用语言反思**,把反思带进下一次尝试。

**出处**:Shinn, Cassano, Gopinath, Narasimhan, Yao.
*Reflexion: Language Agents with Verbal Reinforcement Learning*.
NeurIPS 2023, arXiv:2303.11366。仓库 `noahshinn/reflexion`。

对齐原文的三个模块:

| 模块 | 原文职责 | 这里 |
|---|---|---|
| **Actor** | 用 ReAct 式循环做事 | `common.run_loop`（**和 `react` 臂同一个循环**）|
| **Evaluator** | 判断这次成没成 | `session.check_answer`（环境信号）或 LLM 自评,见下 |
| **Self-Reflection** | 把失败**写成一段话** | 一次 `call_model` |
| **Memory** | 反思存起来,下次带进 prompt | 折进下一次 trial 的 `preamble` |

★ **和 `react` 臂的唯一区别是「有几次 trial、以及上一次的反思进不进 prompt」。**
Actor 本身是同一个循环 —— 这正是 `baseline/common.py` 存在的意义:
两条臂只差在被测变量上,不差在循环实现上。

## ★★ 一个必须说清楚的取舍:Evaluator 的信号从哪来

原文的 Evaluator 用**环境的成败信号**（ALFWorld 那种:任务过没过）。
在 HotpotQA 这类没有环境反馈的设置里,论文用的是**对照标准答案的 EM**。

**两种都行,但读数时必须分得开** —— 一个吃了金标信号的臂,和一个自己判断成没成的臂,
强度完全不同。所以这里**用臂名把这件事写死**:

| 构造 | 臂名 | Evaluator |
|---|---|---|
| `Reflexion(uses_success_signal=True)`（默认）| `reflexion` | **benchmark 的 `check()`** —— 忠于原文 |
| `Reflexion(uses_success_signal=False)` | `reflexion-selfeval` | **LLM 自评** —— 不碰金标 |

★ **接缝是显式的**:只有声明 `needs_success_signal = True` 的臂才会被 runner 接上
`session.check_answer`（见 `core/agent.py`）。**agent 不许自己去读 `task.gold`** ——
那会让每个臂都能作弊,而且是悄悄作。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.baseline.common import LoopConfig, render_tools, run_loop
from experiments.core.agent import AgentOutcome, Session, action_ask
from experiments.core.models import Message
from experiments.core.types import Step

ACTOR_INSTRUCTION = (
    "Solve the task. You may use the tools below as many times as needed, "
    "then give the final answer."
)

REFLECT_INSTRUCTION = """Your previous attempt failed. Write a short reflection.

- Say what you did, why it did not work, and what to do differently next time.
- Be specific about the tools and the arguments you used.
- Do not restate the task and do not give the answer.
- Output the reflection only, in a few sentences.
"""

SELF_EVAL_INSTRUCTION = """Decide whether the answer below actually answers the task.

- Reply with exactly one word: YES or NO.
- Answer NO if the answer is empty, evasive, or does not address the task.
"""


@dataclass
class Reflexion:
    """重试若干次,每次把上一次的反思带进 prompt。

    `max_trials=3` 对齐原文（他们最多反思 3 轮）。
    """

    max_trials: int = 3
    uses_success_signal: bool = True
    exemplars: str = ""
    needs_success_signal: bool = field(init=False)

    def __post_init__(self) -> None:
        # ★ 类属性,给 runner 看 —— 声明「我要吃环境信号」
        self.needs_success_signal = self.uses_success_signal

    @property
    def name(self) -> str:
        # ★ **臂名把 Evaluator 的选择写死。** 一个吃了金标信号的臂
        #   不能和没吃的同列一张表,而表里的主键就是这个名字。
        return "reflexion" if self.uses_success_signal else "reflexion-selfeval"

    def solve(self, session: Session) -> AgentOutcome:
        reflections: list[str] = []
        all_steps: list[Step] = []
        last: AgentOutcome | None = None

        for trial in range(self.max_trials):
            cfg = LoopConfig(
                name=self.name,
                instruction=ACTOR_INSTRUCTION,
                with_thought=True,          # Actor 就是 ReAct 式循环
                exemplars=self.exemplars,
                preamble=self._memory_block(trial, reflections),
            )
            last = run_loop(session, cfg)
            all_steps.extend(last.steps)

            if last.final_answer is not None and self._evaluate(session, last.final_answer):
                return AgentOutcome(steps=all_steps, final_answer=last.final_answer,
                                    escalated=False, error=last.error)

            if trial == self.max_trials - 1:
                break

            text, step = self._reflect(session, last)
            all_steps.append(step)      # ★ 反思是方法的产物,必须进最终轨迹
            reflections.append(text)

        # 用完了 trial 次数仍没过 → **弃答,并把最后一次的答案丢掉**。
        # 留着一个「我们自己判定它没答对」的答案当结果,是在假装成功。
        assert last is not None
        return AgentOutcome(
            steps=all_steps, final_answer=None, escalated=True,
            error=f"{self.max_trials} 次尝试都没通过 Evaluator；最后一次: "
                  f"{(last.final_answer or last.error or '无输出')[:160]!r}",
        )

    # ── 内部三件事 ──────────────────────────────────────────

    def _memory_block(self, trial: int, reflections: list[str]) -> str:
        if not reflections:
            return "" if trial == 0 else "# Memory\n(no reflection recorded)"
        joined = "\n".join(f"- {r}" for r in reflections)
        return f"# Memory (your own reflections from earlier attempts)\n{joined}"

    def _evaluate(self, session: Session, answer: str) -> bool:
        """Evaluator。**用哪条路取决于臂名,不取决于运行时能不能拿到信号。**"""
        if self.uses_success_signal and session.check_answer is not None:
            return session.check_answer(answer)
        if self.uses_success_signal and session.check_answer is None:
            # ★ 声明要吃信号,benchmark 却没实现 `check()` ——
            #   **不许静默退回自评**,那会让两个臂在同一个表里混着不同强度的 Evaluator
            raise RuntimeError(
                "reflexion: 这个 benchmark 没有实现 `check()`,拿不到成败信号。"
                "要么实现它,要么改用 Reflexion(uses_success_signal=False)"
                "（臂名会变成 reflexion-selfeval）。"
            )
        reply = session.call_model([Message(role="user", content="\n".join([
            SELF_EVAL_INSTRUCTION, "", "# Task", session.task.prompt,
            "", "# Answer", answer,
        ]))])
        return "YES" in reply.text.upper()

    def _reflect(self, session: Session, outcome: AgentOutcome) -> tuple[str, Step]:
        history = "\n".join(
            f"{s.index}. {s.action.kind}:{s.action.name or s.action.content[:80]}"
            f" -> {s.observation[:120]}"
            for s in outcome.steps
        )
        reply = session.call_model([Message(role="user", content="\n".join([
            REFLECT_INSTRUCTION, "", "# Tools", render_tools(session.tools),
            "", "# Task", session.task.prompt,
            "", "# What you did", history or "(nothing)",
            "", "# Your last answer", outcome.final_answer or "(none)",
        ]))])
        # ★ 反思也占轨迹里的一步 —— 它是这个方法的**产物**,不是中间变量。
        #   不记它的话,「Reflexion 到底反思出了什么」在 log 里查不到,
        #   而那是审稿人最会问的一样东西。
        return (
            reply.text.strip()[:500],
            Step(index=-1, action=action_ask(f"reflection: {reply.text.strip()[:80]}"),
                 thought=reply.text),
        )
