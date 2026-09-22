"""我们把 JevLoop 这条臂。**一个 `decide()` = 一次节点序列。**

## 这个文件是论文那张表的第一个格子

`docs/PLAN-*.md` 的 Table 2 按两根轴切臂:**（一）谁来决定**、**（二）历史怎么到达下一步**。
这个文件动的是第一根轴 —— 循环还是 ReAct 那个循环（`common.run_loop`）,
**换掉的只有控制器**。

所以 `react × typed` 和 `react × llm` 的差**只可能来自决策者**。这不是声明,是构造:
两个臂共用同一个 `run_loop`、同一个 `build_prompt`、同一批工具。

## 逐步是什么

```
decide()
 ├─ needsTool  noul    「这个任务还有没有没做的动作?」
 │    否 ─────────────────────────────► 生成答案（LLM）→ Decision(answer)
 │    是
 ├─ pickTool   choice  在**这一步重建的**候选里挑一个
 └─ pickInput  choice  该工具的参数候选（枚举得出时）
       └──────────────────────────────► Decision(tool)
```

★ **七个节点里这条臂只驱动三个**（`needsTool` / `pickTool` / `pickInput`）。
另外四个（`gradeRisk` / `stepOk` / `isDone` / `canDeliver`）属于**完整 JevLoop**
那条臂 —— 它有自己的循环和授权闸门。这里不假装驱动了它们:
拿一个用不上的节点去说「我们也解耦了」,是**用帧的数量冒充解耦的深度**。

## 三分法的落点（§8.1）

| 这一步是 | 交给谁 | 在这里 |
|---|---|---|
| 生成文本 | LLM | `_generate()` —— 只有这一处 |
| 挑选 / 打分 / 是否 | Jev | `_ask()` —— 一次请求判一批 |
| 遵循精确规则 | 代码 | 候选重建、预算校验、`Request.check()` |

## ★ 两个从事故里长出来的约束,在这里**是构造性的**

**候选每步重建**（§8.4）:`_candidates()` 每次都从 `view.tools` 减去 `ctx.records()`
里做过的动作。固定候选会让模型去选一个已经不适用的动作 —— 实测写完文件后
`write_file` 还在候选里,模型会**再选它**。

**违规必须有人接**（第 10 轮 R2/R5）:`_ask()` 里的 `Request.check()` 不是装饰,
`fatal` 会让这一步**发不出去**,并退回一个 `unparsed` + 原因。
那条失效链是「选项超限 → 校验发现 → **无人接收** → 请求照发 → 判定静默掉点」——
断在第三步还是第四步,区别就是这里有没有人接。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from experiments.core.agent import Session
from experiments.core.controller import Decision, DecisionView
from experiments.core.deciding import Answer, DecideRequest, DecisionClient
from experiments.core.frame import (
    AgentCtx,
    Question,
    Request,
    candidate_provider,
    compile_frame,
    ctx_from_steps,
    frame_for,
)
from experiments.core.models import Message

#: **「是」的概率**门限（TS 的 `probGte`）。`needsTool` 用它:
#: 「还要不要动作」—— 一个果断的「否」就该去生成答案。
DEFAULT_YES = 0.5
#: **果断程度**门限（TS 的 `topGte`）。`choice` 用它,判的是「答得确定吗」——
#: `noul` 上是 `max(p, 1-p)`,**不是** `p`。
#: ★ 这两个数是两件事,而 §8.3 那条「别用 `confidence` 卡 choice」说的是
#: **别用归一化熵**,两个都不是它。
DEFAULT_TOP = 0.5

#: 生成答案时的格式要求。★ **不是 ReAct 的 `Action:` 格式** —— 那要求模型吐一个
#: 动作行,而这里要的是最终答案本身。
ANSWER_FORMAT = "Reply with the final answer only. Do not emit an action."


def _noul_question(node: str, ask: str, *, threshold: float) -> Question:
    return Question(node=node, kind="noul", ask=ask, threshold=threshold)


def _choice_question(node: str, ask: str, options: list[str],
                     *, threshold: float) -> Question:
    return Question(node=node, kind="choice", ask=ask, options=tuple(options),
                    threshold=threshold)


def candidates(session: Session, ctx: AgentCtx) -> list[str]:
    """**这一步**能选的工具。§8.4 的落点。

    ★ 做过的动作**在这里删掉,不是在帧里提示一句** —— 提示是可以被无视的,
    而候选列表是模型唯一能选的东西。

    ⚠️ 注意这只是**工具级**的去重。同一个工具做两次常常是合理的
    （读两个不同的文件）,所以删的是**已经做过的那一次动作**,
    而 `pickInput` 会在参数那一层再算一次候选。
    """
    done = {r.tool for r in ctx.records()}
    return [t.name for t in session.tools if t.name not in done]


@dataclass
class TypedController:
    """判定与生成分开的控制器。

    `client` 是判定后端（`core/deciding.py`）;生成仍然走 `session.call_model`。
    **一次 `decide()` 里的所有判定走同一批** —— `session.next_batch()` 开批,
    按批计时,不按条（同一次请求判多路时按条求和会多算几倍,实测过）。
    """

    client: DecisionClient
    yes: float = DEFAULT_YES
    top: float = DEFAULT_TOP
    name: str = "typed"
    #: 每一步的判定轨迹（节点 / 答案 / 置信度）。**进日志用,不是给 policy 读的。**
    trace: list[dict] = field(default_factory=list)

    # —— 对外 ────────────────────────────────────────────────

    def decide(self, session: Session, view: DecisionView) -> Decision:
        ctx = ctx_from_steps(view.task_prompt, list(view.history))
        self.trace = []

        if not view.tools:
            # 没有工具的任务（GSM8K 那种）不该问「要不要用工具」——
            # 那个问题的答案恒为否,问它只是白花一次判定。
            return self._generate(session, view, ctx, why="这个任务没有工具")

        # ① 还要不要动作
        batch = session.next_batch()
        need = self._ask(session, batch, ctx, view.step, "needsTool",
                         _noul_question("needsTool",
                                        "Does this task still require a tool call "
                                        "before it can be answered?",
                                        threshold=self.top))
        if need is None:
            return self._blocked(session, view, "needsTool")

        if need.prob_true() < self.yes:
            # ★ 用 `prob_true()`（「是」的概率）,不是 `top()`（果断程度）——
            #   这里问的是「还要不要动作」,答案本身是「是/否」。
            #   一个**果断的「否」**（noul=0.05）在 `top()` 上是 0.95,
            #   拿它来比就会得出「很确定还要工具」—— 正好反了。
            return self._generate(session, view, ctx, why=f"needsTool={need.noul:.3f}",
                                  batch=batch)

        # ② 挑哪个工具 —— **候选在这一步重建**
        options = candidates(session, ctx)
        if not options:
            # 候选空了:做过的都做过了,那就是可以答了。
            # ★ 但 `needsTool` 刚说「还要一个动作」—— 这两句是**矛盾的**,
            #   矛盾本身要记下来,否则它看起来像一次顺利的收尾。
            self.trace.append({"step": view.step, "node": "pickTool",
                               "violation": "no_candidate_left", "fatal": False,
                               "detail": (f"needsTool={need.noul:.3f} 说还要动作,"
                                          f"但候选已空（做过的都做过了）")})
            # ★ 收尾不是「模型判的」,是**代码按精确规则判的**（§8.1 第三行）——
            #   所以要说明来源,不能让日志看起来像一次判定。
            return self._generate(session, view, ctx, why="候选已空（做过的都做过了）",
                                  batch=batch)

        if len(options) == 1:
            # ★ 只有一个候选就**不问** —— 「要不要用工具」刚由 `needsTool` 判过,
            #   再问「要哪一个（而只有一个）」是白花一次判定。
            #   这正是 §8.1 第三行:能由精确规则定的,不交给判定模型。
            picked_choice = options[0]
            tool = next(t for t in view.tools if t.name == picked_choice)
        else:
            picked = self._ask(session, batch, ctx, view.step, "pickTool",
                               _choice_question("pickTool",
                                                "Which tool should be called next?",
                                                options, threshold=self.top))
            if picked is None:
                return self._blocked(session, view, "pickTool")
            picked_choice = picked.choice
            tool = next((t for t in view.tools if t.name == picked_choice), None)
            if tool is None:
                # 判定模型选了一个**不在候选里**的工具 —— 那是它的错,
                # 但必须能被看见,不能悄悄退回一个默认值。
                return self._blocked(session, view,
                                     f"pickTool 选了候选外的 {picked_choice!r}",
                                     answer=picked)

        # ③ 参数
        arguments, err = self._arguments(session, batch, ctx, view, tool)
        if err is not None:
            return self._blocked(session, view, err)

        return Decision(kind="tool", tool=tool.name, arguments=arguments,
                        syntax="typed", raw=f"picked={picked_choice}")

    # —— 节点 ────────────────────────────────────────────────

    def _arguments(self, session: Session, batch: int, ctx: AgentCtx,
                   view: DecisionView, tool) -> tuple[dict, str | None]:
        """参数从哪来。**能枚举的就判,不能枚举的只能生成** —— 而且要说明是哪一种。

        ★ 这是 §8.2 那条硬约束的落点:判定模型只能从枚举里挑
        （实测 77 个候选时选中概率掉到 0.425）。所以一个参数**自由文本**的工具
        （`search[query]` 那种）**不能被 `pickInput` 选中** ——
        把它做成 `choice` 是在骗自己,它只能靠猜。

        那种情况走「生成提候选、判定排序」里的**前半段**:让 LLM 生成。
        ★ 于是**哪一步是判的、哪一步是生成的**必须记清楚 ——
        那正是论文第一根轴要量的东西,含糊过去这个臂就没有意义了。
        """
        provider = candidate_provider(tool)
        params = list((tool.parameters.get("properties") or {}).keys())

        if not params:
            return {}, None

        first = params[0]
        if provider is None:
            # 不可枚举 → 生成。**记下来 —— 这一步不是判定。**
            session.record_decision(step=view.step, node=f"pickInput:{first}",
                                    answer="(generated)", confidence=0.0,
                                    correct=True, latency_ms=0.0, batch=batch)
            return {first: self._generate_argument(session, view, tool, first)}, None

        options = [o for o in provider(ctx)]
        if not options:
            return {}, None
        if len(options) == 1:
            # 只有一个候选 —— 问它是在浪费一次判定,而且答案必然是它。
            return {first: options[0]}, None

        picked = self._ask(session, batch, ctx, view.step, "pickInput",
                           _choice_question("pickInput",
                                            f"Which {first} should `{tool.name}` be called with?",
                                            options, threshold=self.top))
        if picked is None:
            return {}, "pickInput"
        if picked.choice not in options:
            return {}, f"pickInput 选了候选外的 {picked.choice!r}"
        return {first: picked.choice}, None

    def _generate(self, session: Session, view: DecisionView, ctx: AgentCtx,
                  *, why: str, batch: int = 0) -> Decision:
        """生成答案。**这条臂唯一一次让 LLM 生成的地方**（另一个是自由文本参数）。

        ★ 用的是 `view.task_prompt` + 自己的格式要求,**不是** `view.prompt` ——
        后者是 ReAct 的提示词,它要求模型吐一个 `Action: <tool>[<arg>]` 行。
        拿它去生成答案,模型会照着格式吐一个动作,而我们把它当答案收下。

        ★★ **这里不写 `record_decision`。** 第一版写了,记成
        `node="needsTool", confidence=1.0, correct=True` —— 那是**一条假判定**:
        同一个节点、同一步,前面刚记过一条 `0.1 / False` 的真判定,
        读日志的人会看到两个相反的结论。

        更糟的是 `correct` 那一列**正是 RQ2 要量的东西**
        （「置信度说 0.8 的那批判定,到底对了多少」）——
        往里塞一个恒真的值,那条曲线就永远是 45°。

        「这一步走了生成」这件事由轨迹里的 `Action(kind="answer")` 表达,
        不属于判定账。**§8.1:代码按精确规则做的分支不是判定。**
        """
        del batch  # 只为了和 `_ask` 的调用形状一致;这里不记账
        return Decision(kind="answer",
                        answer=self._answer_text(session, view),
                        thought=f"生成答案（{why}）", syntax="generated")

    def _answer_text(self, session: Session, view: DecisionView) -> str:
        prompt = "\n".join([
            ANSWER_FORMAT,
            "",
            "# Task",
            view.task_prompt,
            "",
            "# What was done",
            "; ".join(f"{s.action.name or s.action.kind}" for s in view.history) or "(nothing)",
        ])
        return session.call_model([Message(role="user", content=prompt)]).text

    def _generate_argument(self, session: Session, view: DecisionView, tool, name: str) -> str:
        prompt = "\n".join([
            f"Give only the value for `{name}` — nothing else, no quotes, no label.",
            "",
            "# Task",
            view.task_prompt,
            f"# Tool\n{tool.name}: {tool.description}",
        ])
        return session.call_model([Message(role="user", content=prompt)]).text.strip()

    # —— 判定本身 ────────────────────────────────────────────

    def _ask(self, session: Session, batch: int, ctx: AgentCtx, step: int,
             node: str, question: Question) -> Answer | None:
        """发一次判定请求。**返回 `None` = 这一步发不出去。**

        ★★ 这里是「有人接收」那一步（第 10 轮 R2/R5）。

        那条失效链是:**选项超限 → 校验发现 → 无人接收 → 请求照发 → 判定静默掉点**。
        断在第三步和断在第四步,对这个函数来说就是 `return None` 和 `continue` 的区别。

        两种违规分开处理:
        - **fatal**（缺判定依据 / 超预算）→ **不发**。发出去只会得到一个
          凭空生成的答案,而日志上它和一次正常判定长得一样（§8.10）。
        - **非 fatal**（候选里有做过的动作）→ 发,但记下来 ——
          §8.4 那条不变量要有人**看见**,不是有人拦住。
        """
        req = Request(frame=compile_frame(frame_for(node), ctx), question=question)
        t0 = time.perf_counter()

        for v in req.check(ctx):
            self.trace.append({"step": step, "node": node, "violation": v.code,
                               "fatal": v.fatal, "detail": v.detail})
            # ★★ **违规必须落到事件流里,否则不算有人接收。**
            #
            #   第 10 轮那条失效链是「校验发现 → **无人接收** → 请求照发 → 静默掉点」。
            #   写进内存里一个 list 仍然不是接收 —— 跑完之后没人读得到它,
            #   于是下一个人重建现场时,看到的是「这里怎么少了一次判定」。
            #   所以按「一次没成立的判定」记账:`correct=False` + 说明原因。
            session.record_decision(
                step=step, node=node, answer=f"({v.code})", confidence=0.0,
                correct=False, latency_ms=0.0, batch=batch,
                frame_digest=req.frame.digest(), note=v.detail,
            )
            if not v.fatal:
                continue
            # ★ 不发。为什么发不出去已经记在账上了 —— 否则调用方只知道「没判定」。
            return None

        resp = self.client.decide(DecideRequest(
            state={"frame": req.frame.render()}, questions={node: _wire(question)},
        ))
        answer = resp.answers.get(node)
        if answer is None:
            # 后端没给 / 给了畸形的 —— §8.10:报出来,不假装
            detail = f"dropped={resp.dropped} missing={resp.missing} {resp.warnings}"
            self.trace.append({"step": step, "node": node, "violation": "answer_unusable",
                               "fatal": True, "detail": detail})
            session.record_decision(
                step=step, node=node, answer="(answer_unusable)", confidence=0.0,
                correct=False, latency_ms=(time.perf_counter() - t0) * 1000, batch=batch,
                frame_digest=req.frame.digest(), note=detail,
            )
            return None

        self.trace.append({
            "step": step, "node": node, "answer": answer.choice or answer.noul,
            "top": answer.top(), "provider": resp.provider,
            "frame_digest": req.frame.digest(),
            "truncations": dict(req.frame.truncations),
            "missing": list(req.frame.missing),
        })

        # ★★ **果断程度不够就不许往下走。**
        #
        #   这是一个**被迫的选择**:选项是穷尽的,判定模型必须挑一个 ——
        #   所以「挑了一个」不等于「挑得对」。`top()` 判的正是这件事
        #   （`choice` 看被选中那项的概率;`noul` 看 `max(p,1-p)`）。
        #
        #   §8.6 要求 Mock 必须让 policy **自己走到 `escalate`** ——
        #   而 Mock 给的是平均分布（8 个选项时每项 0.125）。
        #   没有这道门,`_arguments` 会照拿第一个候选,
        #   于是「不确定就别猜」在代码里**从来没有被走到过**。
        decided = answer.top() >= question.threshold
        session.record_decision(
            step=step, node=node, answer=str(answer.choice or f"{answer.noul:.3f}"),
            confidence=answer.top(),
            # ★ 帧的指纹进日志（§8.14）—— 两次运行帧不一样而没人发现,
            #   「同一个方法」这句话就不成立。
            frame_digest=req.frame.digest(),
            # ★ `correct` 是「这次判定**成立**吗」,不是「护身符」。
            #   全填 True 的话,`correct` 那一列恒为真,
            #   于是「置信度和正确性同现」这句话**在账面上永远成立** —— 而它本该被检验。
            correct=decided,
            # ★★ **一条判定只记一行。**
            #
            #   第一版这里在「不果断」时**又记了一条** —— 于是同一次判定在账上出现两遍,
            #   而**判定次数正是论文的头条指标之一**。这和 §8.10「按批计时、按记录求和
            #   会多算几倍」是同一类错:账目上的一个数被数了两遍,而账面上看不出来。
            #
            #   所以原因写在**同一行的 `note` 里**,不是另起一行。
            note="" if decided else f"top={answer.top():.3f} < {question.threshold}",
            latency_ms=(time.perf_counter() - t0) * 1000, batch=batch,
        )
        if not decided:
            self.trace.append({
                "step": step, "node": node, "violation": "below_threshold", "fatal": True,
                "detail": (f"{node}: top={answer.top():.3f} < {question.threshold}"
                           f"（答得不果断,不许拿它往下走）"),
            })
            return None
        return answer

    def _blocked(self, session: Session, view: DecisionView, why: str,
                 *, answer: Answer | None = None) -> Decision:
        """这一步发不出去 / 判定不成立。

        ★ 返回 `unparsed` 而不是硬凑一个动作 —— `unparsed` 的措辞是
        **「没答出来」不是「答错了」**（`core/controller.py`）。循环会重试,
        重试用完就 `escalate`,而 `escalate` 是一种**能被看见**的结果。

        ⚠️ 弃答的**原因**放进 `raw`,而且开头就写明是 `typed:` ——
        不写的话,日志上它和「模型格式错了」长得一样,而那是完全不同的事。
        """
        detail = ""
        if answer is None and self.trace:
            detail = self.trace[-1].get("detail", "")
        return Decision(kind="unparsed", syntax="typed",
                        raw=f"typed: {why}{' — ' + detail if detail else ''}")


def _wire(question: Question) -> dict:
    """`Question` → 线协议的问题形状。"""
    if question.kind == "choice":
        return {"kind": "choice", "ask": question.ask,
                "options": {o: 0.0 for o in question.options}}
    if question.kind == "score":
        return {"kind": "score", "ask": question.ask}
    return {"kind": "noul", "ask": question.ask}


__all__ = ["TypedController", "candidates", "DEFAULT_YES", "DEFAULT_TOP", "ANSWER_FORMAT"]
