"""★ **决策接口 —— 「下一步做什么」由谁回答。**

## 这个文件是论文那句话的代码形态

论文名:**`JevLoop: Decoupling Decision from Generation for Efficient Language Model Agents`**。
「解耦」在代码里就是这一个接口。

★★ **区别不是「哪个更好」,是「决定和生成是不是同一次调用」。**
在未解耦的形态里,模型吐出的那段文本**既是它的推理,也是它的决定** ——
先 `call_model`,再从文本里 `parse_step`。把这两件事拆开,就是这篇论文主张的全部。

## 分层（这一条踩过）

**`core/` 只放词汇与协议,不放实现。** 具体的 `LLMController` 住在 `baseline/common.py`,
因为它实现的 `Action: tool[arg]` 是 **ReAct 的交互协议** —— 那是某个范式的事,不是词汇。

我第一版把 `LLMController` 写在这里,于是 `core/` 要 import `baseline/` ——
**依赖方向反了**。分层不是洁癖:反过来的话,以后加一个不依赖 ReAct 文本协议的范式,
它也得把 `baseline/` 拖进来。

## 为什么**现在**就该抽这个接口,哪怕还没有第二种实现

因为**基线一旦写完再抽,就得把六个臂全部重跑一遍**才敢说「两条臂只差决策者」。
现在抽、并钉一条等价性测试,以后加 `TypedController` 时,
**两个格子之间的差只可能来自决策者**。

★ **不强行给每个范式都加第二种实现。** 所有者定了:`ReWOO` 不用硬加 ——
它的 Worker 根本没有决策（见 `docs/NOTES-*.md` §2.10.1）。
**能被替换多少,按范式差得很远,而那个差别本身就是要量的东西。**

## 与 `Agent` 协议的关系

`Agent`（`core/agent.py`）管「整个臂长什么样」;`Controller` 管「臂里那个岔路口由谁回答」。
一个 `Agent` 持有一个 `Controller` —— 于是 `loop × controller` 是**两层组合**,不是两套代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from experiments.core.types import Step, Tool


@dataclass(frozen=True)
class Decision:
    """控制器给出的下一步。

    `kind` 只有三种,**没有第四种**:

    - `tool` 调一个工具
    - `answer` 给最终答案
    - `unparsed` **没答出来** —— 由循环决定重试还是弃答

    ★ 第三种的措辞很重要:**「没答出来」不等于「答错了」**。
    把它当成答案是最坏的处理,它把「格式错了」变成「模型答了」。
    """

    kind: str
    thought: str = ""
    tool: str = ""
    arguments: dict = field(default_factory=dict)
    answer: str = ""
    # 用的是哪种写法（bracket / action-input / json / bare-answer）—— 失败分类要用
    syntax: str = ""
    raw: str = ""


@dataclass(frozen=True)
class DecisionView:
    """岔路口上**看得到的东西**。

    ★ **控制器只能看这些。** 这是 `AGENTS.md` §8.2「帧里没有的,它判不出来」
    在接口上的形状:想让一种决策者能判,就得先把材料放进 `view`。
    把 `session` 整个交给它,会让「它到底看到了什么」无从核对 —— 而那正是要报的东西。
    """

    prompt: str
    tools: tuple[Tool, ...]
    # ★★ **这里只有 `task_id`,没有 `Task`** —— 因为 `Task` 带着 `gold`。
    #
    #   第一版我放的是整个 `Task`,于是上面那句「控制器只能看这些」是**空话**:
    #   控制器可以 `view.task.gold` 把答案读出来,而没有任何东西拦得住。
    #   真正要防的是**偷看金标**:`reflexion` 那类臂是**声明**了才拿到成败信号的
    #   （见 `core/agent.py` 的 `check_answer`）,其余的一律拿不到。
    #   **接口上拿不到,比「约定不许」强** —— 约定会忘,类型不会。
    task_id: str = ""
    step: int = 0
    history: tuple[Step, ...] = ()
    extra: dict = field(default_factory=dict)

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def first_arg(self) -> dict[str, str | None]:
        """每个工具的第一个位置参数名。

        ★ `tool[arg]` 这种写法只有一个位置参数,**名字按 `parameters` 里的定义顺序取** ——
        所以 loader 写工具定义时键的顺序是有意义的。
        """
        out: dict[str, str | None] = {}
        for tool in self.tools:
            props = tool.parameters.get("properties", {}) or {}
            out[tool.name] = next(iter(props), None)
        return out


class Controller(Protocol):
    """回答「下一步做什么」。

    ★ 只拿得到 `view` —— **不许它自己去读 `task.gold`**。谁吃了金标信号,
    谁的名字里就要写清楚（见 `baseline/reflexion.py` 的 `reflexion` vs `reflexion-selfeval`）。
    """

    name: str

    def decide(self, session, view: DecisionView) -> Decision:
        """`session` 只用来 `call_model`。"""
        ...


__all__ = ["Controller", "Decision", "DecisionView"]
