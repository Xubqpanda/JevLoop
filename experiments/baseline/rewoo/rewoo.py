"""`rewoo` —— Plan / Work / Solve:**推理与观察解耦**。

**出处**:Xu, Peng, Lei, Mukherjee, Liu, Xu.
*ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*.
arXiv:2305.18323。仓库 `billxbf/ReWOO`。

★ **这是六个 baseline 里最危险的一个**（`docs/PLAN-*.md` §0.6）,
因为它打的是**同一条轴**——token / 调用效率——而且它赢我们的方式很直接:
**它只规划一次**。原文 Eq.(1) 证明 ReAct 的 prompt 开销随步数 **二次**增长
（每步重发整个 scratchpad）,Eq.(2) 证明它是**线性**的。
在 HotpotQA 上他们报 **5× token 效率、+4% 准确率**。

对齐原文的三段结构:

| 模块 | 原文职责 | 这里 |
|---|---|---|
| **Planner** | 任务 + 工具 + exemplar → 一串 `(Plan, #E)` 蓝图,**看不到任何观察** | 一次 `call_model` |
| **Worker** | 执行 `#E_i = Tool[input]`,输入里的 `#E_j` 换成上一步的真实结果 | `substitute_evidence` + `call_tool` |
| **Solver** | 任务 + 计划 + 证据 → 最终答案 | 一次 `call_model` |

★ **「看不到观察」是它的本质,不是实现细节。** 计划是**盲**的:规划时不知道检索会返回什么。
这既带来线性开销,也带来它固有的失败模式 ——
**计划可以引用一个根本不存在的 `#E`**（`substitute_evidence` 会把这类变量名返回出来,
必须当失败上报,不能把 `#E1` 原样喂给工具)。

★ **和我们自己的区别**（review 时会用到）:
我们**每一步**都重新编译有界状态再判定,所以能对观察作出反应;
ReWOO **一次规划到底**,中途不看了。**它省的是「重复读历史」,我们省的是「用生成做决定」。**
两条路不冲突 —— 这正是 `PLAN-*.md` §0.6 表 2 里「ReWOO(类型化判定)」那一格要测的东西。
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.baseline.common import (
    extract_answer,
    first_arg_name,
    parse_plan,
    render_exemplars,
    render_plan,
    render_tools,
    substitute_evidence,
)
from experiments.core.agent import AgentOutcome, Session, action_answer, action_ask, action_tool
from experiments.core.models import Message
from experiments.core.types import Step

INSTRUCTION = """Devise a plan that solves the task step by step, then stop.

Rules:
- Each step names the evidence it produces as a variable, and says how to obtain it.
- A step may refer to evidence produced earlier by its variable name.
- Use only the tools listed above, or the pseudo-tool LLM[<question>] for reasoning
  over evidence you already have.
- Do not attempt to answer the task directly, and do not guess what any tool
  will return. Emit the plan only.

Format, one step per line:
Plan: <what this step is for>
#E1 = <tool>[<input>]
Plan: <what this step is for>
#E2 = <tool>[<input or #E1>]
"""

# ★ Solver 那段只写 **ReWOO 自己的协议**（用计划和证据）。
#   「答案长什么样」是**输出契约**,归 benchmark（见 core/bench.py）——
#   搬到这里就等于 ReWOO 臂自己改任务陈述,而各臂该只差在**交互**上。
SOLVER_INSTRUCTION = """Answer the task using the plan and the evidence collected for it.

- Use only the evidence below. If it is insufficient, say so rather than guessing.
"""

# 原文在 HotpotQA 用 6 条 exemplar（ReWOO 为了可比照抄了 ReAct 的那 6 条）。
# ⚠️ 我们是 zero-shot —— 想和原文数字对齐就得补上,想和「零样本」那条线比就留空。**两种都行,但必须写明。**
DEFAULT_EXEMPLARS = ""

# 原文蓝图里的伪工具 —— **不是我们提供的工具,但也**不是**失败。
# `LLM[<question>]` 的活由 Solver 干,`finish`/`answer` 是收尾。
PSEUDO_TOOLS = {"llm", "finish", "answer", "final_answer"}


@dataclass
class ReWOO:
    """盲规划 → 收集证据 → 综合答案。**中途不重新规划** —— 那是 `plan-then-execute` 的事。"""

    exemplars: str = DEFAULT_EXEMPLARS
    name: str = "rewoo"

    def solve(self, session: Session) -> AgentOutcome:
        steps: list[Step] = []

        # ── Planner：一次调用，看不到任何观察 ──────────────────
        plan_prompt = "\n".join(p for p in [
            INSTRUCTION,
            "",
            "# Tools",
            render_tools(session.tools),
            # ★ 伪工具 LLM[...] 原文里有,我们没提供 —— **必须说明**,
            #   否则模型计划出来的 LLM[...] 会被当成未知工具,而那是我们的口径问题不是它的
            "# Note",
            "`LLM[<question>]` is available for reasoning over evidence you already have.",
            render_exemplars(self.exemplars),
            "",
            "# Task",
            session.task.prompt,
        ] if p is not None).strip()

        with session.span("rewoo/plan"):
            reply = session.call_model([Message(role="user", content=plan_prompt)])
        items = parse_plan(reply.text, [t.name for t in session.tools])

        if not items:
            # ★ 计划都解析不出来 → 弃答。**不许把整段当计划硬跑。**
            return AgentOutcome(
                steps=steps, final_answer=None, escalated=True,
                error=f"Planner 没给出可解析的计划: {reply.text[:200]!r}",
            )

        steps.append(Step(index=0, action=action_answer(render_plan(items)),
                          observation=f"{len(items)} 步计划", thought=reply.text))

        # ── Worker：逐步执行，`#E` 换成真实观察 ────────────────
        evidence: dict[str, str] = {}
        unresolved: list[str] = []
        # ★ Worker 阶段 —— 三段各自一个 span,于是「Plan 慢还是 Work 慢」从日志直接读得出
        with session.span("rewoo/work"):
            for item in items:
                if not item.tool:
                    continue
                tool_names = [t.name for t in session.tools]
                if not item.known:
                    # ★ 分两种,别混:
                    #
                    # ① **伪工具**（`LLM[...]` / `finish[...]`）—— 原文里就有,
                    #    我们在 prompt 里明确说过 `LLM[<question>]` 可用。
                    #    **它不是失败**,它的活由 Solver 干。早先把这类记成 unresolved,
                    #    于是 error 被置上,而 runner 一看到 error 就把答案判否 ——
                    #    **一个正确答案就这么丢了**（实测)。
                    # ② **我们不认识的工具名** —— 那才是计划错了,要记。
                    if item.tool.lower() not in PSEUDO_TOOLS:
                        unresolved.append(f"{item.evidence_var or '?'}={item.tool}")
                    if item.evidence_var:
                        evidence[item.evidence_var] = (
                            f"(not executed here: {item.tool}[...] is handled by the solver)"
                        )
                    steps.append(Step(
                        index=len(steps),
                        action=action_answer(f"plan step {item.evidence_var or ''}: {item.tool}[...]"),
                        observation=f"skipped: {item.tool} 不是工具,交给 Solver",
                    ))
                    continue

                argument, missing = substitute_evidence(item.argument, evidence)
                if missing:
                    # ★ ReWOO 固有的失败模式:盲计划引用了一个不存在的变量。
                    #   原样把 `#E9` 传给工具会变成一次莫名其妙的检索失败,
                    #   而真正的原因是计划错了。
                    unresolved.extend(missing)
                    if item.evidence_var:
                        evidence[item.evidence_var] = f"(unresolved: {' '.join(missing)})"
                    continue

                tool = next(t for t in session.tools if t.name == item.tool)
                first = first_arg_name(tool)
                if first is None:
                    unresolved.append(f"{item.evidence_var or item.tool}: 工具没有可用的参数名")
                    continue
                observation = session.call_tool(item.tool, {first: argument})
                if item.evidence_var:
                    evidence[item.evidence_var] = observation
                steps.append(
                    Step(index=len(steps), action=action_tool(item.tool, {first: argument}),
                         observation=observation)
                )

        # ── Solver：任务 + 计划 + 证据 → 答案 ──────────────────
        evidence_block = "\n".join(f"{k} = {v}" for k, v in evidence.items()) or "(no evidence)"
        solver_prompt = "\n".join([
            SOLVER_INSTRUCTION,
            "",
            "# Task",
            session.task.prompt,
            "",
            "# Plan",
            render_plan(items),
            "",
            "# Evidence",
            evidence_block,
        ])
        with session.span("rewoo/solve"):
            solved = session.call_model([Message(role="user", content=solver_prompt)])
        answer = extract_answer(solved.text) or solved.text.strip()

        steps.append(Step(index=len(steps), action=action_answer(answer), thought=solved.text))

        # ★ `error` 的语义是「这次运行坏了」,**不是「计划有瑕疵」**。
        #   runner 见到 error 就把 correct 判否,所以把警告塞进 error
        #   等于因为计划有一处小毛病就丢掉一个正确答案（实测踩过）。
        #   计划里跳过的步骤留在 step 里,失败分类要抓就从那里抓。
        if not answer.strip():
            return AgentOutcome(steps=steps, final_answer=None, escalated=True,
                                error="Solver 没给出答案")
        if unresolved:
            steps.append(Step(
                index=len(steps),
                action=action_ask(f"计划里有 {len(unresolved)} 处未知工具"),
                observation=str(unresolved),
            ))
        return AgentOutcome(steps=steps, final_answer=answer)
