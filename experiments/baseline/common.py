"""六个 baseline 共用的基础设施。

**它们相同的地方比不同的地方多得多。** 每个臂各写一遍「工具怎么渲染、动作怎么解析、
历史怎么拼回 prompt」,结果就是六个臂差在格式上而不是差在方法上 ——
而那正好毁掉比较的意义。

★ 共享的四件事:

| 函数 | 为什么必须共享 |
|---|---|
| `render_tools()` | **工具描述必须逐字一致**（`docs/PLAN-*.md` §2 的公平性要求）。各写一遍 = 各写一份措辞,而措辞会改判定 |
| `parse_step()` | 解析器不同 = 同一个模型输出在 A 臂被认成工具调用、在 B 臂被认成答案 |
| `render_history()` | 上下文形状不同 = token 数不可比 |
| `run_loop()` | 循环骨架相同,**差别只在配置**（`LoopConfig.with_thought` 就是 act 与 react 的全部区别）|

★ 一条刻意的选择:**工具靠 prompt 描述、动作靠文本解析,不用 API 原生的 tool calling。**
这是忠于 ReAct 原文的做法（Yao et al. 2022 用的是文本 scratchpad）,
也正是 ReWOO 攻击的那个形状（每步重发整个 scratchpad）。
「ReAct vs 原生 tool calling」是**另一个** baseline,不是这一个 —— 别混。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from experiments.core.agent import AgentOutcome, Session, action_answer, action_ask, action_tool
from experiments.core.models import Message
from experiments.core.types import Action, Step, Tool

# ReAct 的截断点。**没有它,模型会自己把 Observation 也编出来** ——
# 那不叫幻觉,叫我们没告诉它该停在哪。
REACT_STOP: tuple[str, ...] = ("\nObservation", "\nObservation:")

ANSWER_MARKERS = ("finish", "answer", "final_answer", "final answer")


# ═══════════════════════════════════════════════════════════
# 一、工具描述 —— **全臂逐字一致**
# ═══════════════════════════════════════════════════════════


def render_tools(tools: Sequence[Tool]) -> str:
    """把工具列表渲染成 prompt 里那一段。

    ★ **所有 baseline 都必须调它。** 理由:`docs/PLAN-*.md` §2 要求
    「同一批工具、同一份工具描述」—— 各写一份措辞,措辞的差别就会混进方法差别里。
    """
    if not tools:
        return "(no tools available)"
    lines = []
    for tool in tools:
        params = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        args = []
        for name, spec in params.items():
            mark = "" if name in required else "?"
            enum = spec.get("enum")
            hint = f" one of {enum}" if enum else ""
            args.append(f"{name}{mark}: {spec.get('type', 'any')}{hint}")
        lines.append(f"- {tool.name}({', '.join(args)}) — {tool.description}")
    return "\n".join(lines)


def first_arg_name(tool: Tool) -> str | None:
    """工具的第一个位置参数名。

    ★ `tool[arg]` 这种写法只有一个位置参数,所以**名字必须按 `parameters` 里的
    定义顺序取** —— 这就是为什么 loader 写工具定义时键的顺序是有意义的。
    取不到就返回 None,让调用方自己决定（而不是猜一个 `"input"`）。
    """
    props = tool.parameters.get("properties", {}) or {}
    return next(iter(props), None)


def render_exemplars(exemplars: str) -> str:
    return f"\n{exemplars.strip()}\n" if exemplars.strip() else ""


# ═══════════════════════════════════════════════════════════
# 二、解析 —— **一个解析器,不是六个**
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParsedStep:
    """模型这一步说了什么。`kind` 只有三种,**没有第四种**。

    解析不出来时必须是 `unparsed`,**不许猜**。
    把 `unparsed` 当成答案是最坏的一种处理:它把「模型格式错了」变成「模型答了」。
    """

    kind: str  # "tool" | "answer" | "unparsed"
    thought: str = ""
    tool: str = ""
    arguments: dict = field(default_factory=dict)
    answer: str = ""
    syntax: str = ""  # 用的是哪种写法 —— 失败分类要用
    raw: str = ""


# ★ 函数名里的 **`.` 必须允许** —— BFCL 的函数合法地带点
#   （`math.hcf` / `triangle_properties.get` / `history_api.get_president_by_year`）。
#   早先这里是 `[\w-]*`,于是那些调用全被判成「解析不出动作」→ 重试 → 弃答,
#   而模型输出其实**完全正确**。实测:bfcl-simple × act 的失败里
#   **8/23 全是这一个字符造成的**,不是模型的问题。
_BRACKET = re.compile(r"^(?P<name>[A-Za-z_][\w.-]*)\s*\[(?P<arg>.*)\]\s*$", re.S)
_ACTION = re.compile(r"^\s*Action\s*:\s*(?P<body>.+?)\s*$", re.I | re.M)
_ACTION_INPUT = re.compile(r"^\s*Action\s*Input\s*:\s*(?P<body>.+?)\s*$", re.I | re.M)
_THOUGHT = re.compile(r"^\s*Thought\s*:\s*(?P<body>.*?)(?=\n\s*(?:Action|Final|Answer)\s*:|\Z)",
                      re.I | re.S)
_FENCED = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.S)


def parse_step(text: str, tool_names: Sequence[str], *, first_arg: str | None = None) -> ParsedStep:
    """把模型的输出解析成一步。

    ★ **容错但不猜。** 认不出来就是 `unparsed`,由循环去重试或弃答 ——
    悄悄把它当答案会让「格式错」这个失败模式在数据里消失。

    支持三种写法,按 ReAct 原文的优先级:
    1. `Action: search[entity]` —— 原文写法
    2. `Action: search` + `Action Input: entity` —— 常见变体
    3. JSON（含围栏）,给原生 tool calling 的模型留的路
    """
    thought = ""
    m = _THOUGHT.search(text)
    if m:
        thought = m.group("body").strip()

    # ③ JSON
    candidate = _FENCED.search(text)
    blob = candidate.group("body") if candidate else (text if text.strip().startswith("{") else None)
    if blob:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and ("action" in obj or "name" in obj):
            name = str(obj.get("action") or obj.get("name") or "")
            args = obj.get("action_input", obj.get("arguments", {}))
            if not isinstance(args, dict):
                args = {first_arg: args} if first_arg else {"input": args}
            return _finish_or_tool(
                name, args, thought=thought, text=text,
                tool_names=tool_names, first_arg=first_arg, syntax="json",
            )

    # ①② 文本写法
    action = _ACTION.search(text)
    if action:
        body = action.group("body").strip()
        bracket = _BRACKET.match(body)
        if bracket:
            name = bracket.group("name")
            raw_arg = bracket.group("arg").strip()
            args = _arg_dict(raw_arg, first_arg)
            return _finish_or_tool(
                name, args, thought=thought, text=text,
                tool_names=tool_names, first_arg=first_arg, syntax="bracket",
            )
        # `Action: search` + `Action Input: ...`
        name = body.splitlines()[0].strip()
        arg = _ACTION_INPUT.search(text)
        if arg:
            return _finish_or_tool(
                name, _arg_dict(arg.group("body").strip(), first_arg),
                thought=thought, text=text, tool_names=tool_names,
                first_arg=first_arg, syntax="action-input",
            )
        if name in tool_names:
            # ★ 光有 `Action: name`、没有参数 —— 这**是一个工具调用**,只是缺参数。
            #   交给工具自己去报错（错误会成为观察,模型下一轮能改）,
            #   而不是在解析层把它丢掉。丢掉的话「模型忘了给参数」这个失败模式
            #   在数据里就看不见了,只表现为一次莫名的重试。
            return ParsedStep(kind="tool", thought=thought, tool=name, arguments={},
                              syntax="bare-action", raw=text)

    # 直接给了答案（有些模型不写 Action: finish[...]）
    answer = extract_answer(text)
    if answer:
        return ParsedStep(kind="answer", thought=thought, answer=answer, syntax="bare-answer", raw=text)

    return ParsedStep(kind="unparsed", thought=thought, raw=text)


def _finish_or_tool(
    name: str, args: dict, *, thought: str, text: str,
    tool_names: Sequence[str], first_arg: str | None, syntax: str,
) -> ParsedStep:
    if name.lower().replace("_", "") in {m.replace("_", "") for m in ANSWER_MARKERS}:
        # `finish[Paris]` / `answer[Paris]` —— 取第一个参数的值
        answer = next(iter(args.values()), "") if args else ""
        return ParsedStep(kind="answer", thought=thought, answer=str(answer).strip(),
                          syntax=syntax, raw=text)
    if name not in tool_names:
        # ★ 认不出来也要说清楚它想调什么 —— 这是「选错工具」的原始证据
        return ParsedStep(kind="unparsed", thought=thought,
                          raw=f"{text}\n[unknown tool: {name}]")
    return ParsedStep(kind="tool", thought=thought, tool=name, arguments=args,
                      syntax=syntax, raw=text)


def _arg_dict(raw: str, first_arg: str | None) -> dict:
    """`search[Arthur's Magazine]` 里的那个参数。

    ★ 单个位置参数要用工具自己的 `first_arg` 名字 —— 猜名字会让参数名对不上,
    而 `ToolExecutor` 会因此抛。**名字由 `Tool.parameters` 里排在第一个的键决定**,
    所以 loader 写工具定义时的顺序是有意义的。
    """
    raw = raw.strip().strip('"').strip("'")
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {first_arg: raw} if first_arg else {"input": raw}


_ANSWER_PATTERNS = (
    re.compile(r"^\s*Action\s*:\s*(?:finish|answer)\s*\[(?P<a>.*)\]\s*$", re.I | re.M),
    re.compile(r"^\s*(?:Final\s+Answer|Answer)\s*:\s*(?P<a>.+?)\s*$", re.I | re.M),
)


def extract_answer(text: str) -> str:
    """从一段文本里抠最终答案。**抠不到就返回空串,不返回整段。**

    返回整段是另一种「猜」:评测器会拿一整段话去和标准答案比,然后判错 ——
    而真正的失败原因（模型没按格式给答案）就看不见了。
    """
    for pattern in _ANSWER_PATTERNS:
        found = pattern.findall(text)
        if found:
            return str(found[-1]).strip().strip('"').strip()
    return ""


# ═══════════════════════════════════════════════════════════
# 三、历史 —— 拼回 prompt
# ═══════════════════════════════════════════════════════════


def render_step(step: Step, *, with_thought: bool) -> str:
    """把一步渲染成 scratchpad 里的一段。**`with_thought=False` 就是 act。**"""
    lines: list[str] = []
    # ★ act 与 react 在**渲染**上的唯一区别就是这一行
    if with_thought and step.thought:
        lines.append(f"Thought: {step.thought}")
    if step.action.kind == "tool":
        lines.append(f"Action: {step.action.name}[{_render_arg(step.action.arguments)}]")
        if step.observation:
            lines.append(f"Observation: {step.observation}")
    elif step.action.kind == "answer":
        lines.append(f"Action: finish[{step.action.content}]")
    return "\n".join(lines)


def _render_arg(arguments: dict) -> str:
    if len(arguments) == 1:
        return str(next(iter(arguments.values())))
    return json.dumps(arguments, ensure_ascii=False)


def render_history(steps: Sequence[Step], *, with_thought: bool) -> str:
    """整个 scratchpad。★ ReAct **每步重发它** —— 那正是 ReWOO 攻击的形状,
    也是这个臂该有的 token 增长行为。别偷偷换成多轮 messages。"""
    return "\n".join(render_step(s, with_thought=with_thought) for s in steps)


# ═══════════════════════════════════════════════════════════
# 四、循环骨架 —— **act 与 react 只差一个开关**
# ═══════════════════════════════════════════════════════════


@dataclass
class LoopConfig:
    """一个臂的全部配置。

    ★ `with_thought` 就是 **act 与 react 的唯一区别**。
    （我们现有的 `bench/react.ts` 里 `thought` 出现 0 次,所以它跑出来的是 act ——
    这里把它变成一个显式开关,免得再靠读 prompt 才发现。）
    """

    name: str
    instruction: str
    with_thought: bool = True
    exemplars: str = ""
    stop: tuple[str, ...] = REACT_STOP
    max_parse_retries: int = 2
    # ★ 插在 Tools 之前的一段固定文本。`plan-then-execute` 的执行段就是
    #   「把计划塞进 preamble,然后跑同一个 run_loop」—— 一处也不用重写。
    preamble: str = ""


def run_loop(session: Session, cfg: LoopConfig) -> AgentOutcome:
    """ReAct 式循环。**所有臂共用它** —— 差的只是 `LoopConfig`。

    三步:拼 prompt（含整个 scratchpad）→ 调模型 → 解析并执行。
    解析不出来时最多重试 `max_parse_retries` 次,再不行就**弃答**（`action_ask`）——
    弃答是一种**能被看见**的结果,比硬凑一个答案诚实。
    """
    tools = session.tools
    tool_names = [t.name for t in tools]
    first_arg = {t.name: first_arg_name(t) for t in tools}
    steps: list[Step] = []
    retries = 0

    while len(steps) < session.max_steps:
        prompt = build_prompt(session, cfg, steps)
        reply = session.call_model([Message(role="user", content=prompt)])
        parsed = parse_step(reply.text, tool_names, first_arg=first_arg.get(parsed_tool_guess(reply.text)))

        if parsed.kind == "unparsed":
            retries += 1
            # ★ 重试也要留下痕迹 —— 不然后面分不清「一次就对」和「纠了两次才对」
            session.note_retry(0.0)
            if retries <= cfg.max_parse_retries:
                steps.append(Step(index=len(steps), action=action_tool("__parse_error__", {}),
                                  observation=PARSE_NUDGE))
                continue
            return AgentOutcome(
                steps=steps, final_answer=None, escalated=True,
                error=f"连续 {retries} 步解析不出动作（最后一段: {parsed.raw[-200:]!r}）",
            )

        if parsed.kind == "answer":
            steps.append(Step(index=len(steps), action=action_answer(parsed.answer),
                              thought=parsed.thought))
            return AgentOutcome(steps=steps, final_answer=parsed.answer)

        # 工具
        try:
            observation = session.call_tool(parsed.tool, parsed.arguments)
        except Exception as exc:  # noqa: BLE001 —— 工具自己抛了,当观察喂回去让模型改
            observation = f"Tool error: {type(exc).__name__}: {exc}"
        steps.append(
            Step(index=len(steps), action=action_tool(parsed.tool, parsed.arguments),
                 observation=observation, thought=parsed.thought, model_ms=reply.model_ms)
        )

    # 步数用完 —— ★ 弃答,不是硬答
    return AgentOutcome(
        steps=steps, final_answer=None, escalated=True,
        error=f"用满 {session.max_steps} 步仍未给出答案",
    )


def parsed_tool_guess(text: str) -> str:
    """给 `_arg_dict` 找「这一个位置参数该叫什么名字」。

    ★ 猜错名字 → `ToolExecutor` 抛 `TypeError` → 被当成工具错误喂回模型。
    这里只做一次尽力而为的匹配,真正的名字以工具定义里的第一个键为准。
    """
    m = _ACTION.search(text)
    if not m:
        return ""
    bracket = _BRACKET.match(m.group("body").strip())
    return bracket.group("name") if bracket else m.group("body").strip().splitlines()[0].strip()


def build_prompt(session: Session, cfg: LoopConfig, steps: Sequence[Step]) -> str:
    """整个 prompt。**每一步重建一次**（ReAct 原文就是重发 scratchpad）。"""
    scratchpad = render_history(steps, with_thought=cfg.with_thought)
    parts = [
        cfg.instruction,
        cfg.preamble,
        "",
        "# Tools",
        render_tools(session.tools),
        "",
        "# Format",
        FORMAT_WITH_THOUGHT if cfg.with_thought else FORMAT_WITHOUT_THOUGHT,
        render_exemplars(cfg.exemplars),
        "",
        "# Task",
        session.task.prompt,
        "",
        scratchpad,
    ]
    return "\n".join(p for p in parts if p is not None).strip()


PARSE_NUDGE = (
    "Could not parse an action. Reply with exactly one of:\n"
    "  Action: <tool_name>[<argument>]\n"
    "  Action: finish[<answer>]"
)

# ★ 参数的两种写法都要说。ReAct 原文的 `tool[arg]` 是为**单参数**工具设计的
#   （search[entity] / lookup[string]），而真实 benchmark 的函数常常要多个命名参数
#   （BFCL 的 `calculate_triangle_area[base, height]`）。不说清楚,模型只能瞎猜格式,
#   而那是**我们的格式问题**,不是它能力问题。
_ARG_NOTE = (
    "  · One argument:      Action: <tool_name>[<value>]\n"
    "  · Several arguments: Action: <tool_name>[{\"arg1\": <v1>, \"arg2\": <v2>}]\n"
    "  · No arguments:      Action: <tool_name>[]"
)

FORMAT_WITH_THOUGHT = (
    "Use exactly this format, one block per step:\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <tool_name>[...]      (or)  Action: finish[<final answer>]\n"
    + _ARG_NOTE + "\n"
    "Stop after the Action line. Do not write the Observation yourself."
)

FORMAT_WITHOUT_THOUGHT = (
    "Use exactly this format, one block per step:\n"
    "Action: <tool_name>[...]      (or)  Action: finish[<final answer>]\n"
    + _ARG_NOTE + "\n"
    "Stop after the Action line. Do not write the Observation yourself.\n"
    "Do not output a Thought line."
)


# ═══════════════════════════════════════════════════════════
# 五、计划 —— ReWOO 与 plan-then-execute 共用
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlanItem:
    """计划里的一步。

    `evidence_var` 是 ReWOO 的 `#E1` 那种变量名;**只有 ReWOO 会填它**。
    `plan-then-execute` 只用 `text` —— 两条臂的区别之一就在这里:
    **ReWOO 的计划里显式声明了「这一步的产出叫什么」,于是后面的步骤可以引用它;
    plan-then-execute 只是把子任务排个序。**
    """

    text: str
    evidence_var: str = ""
    tool: str = ""
    argument: str = ""
    # 工具名不在 `tool_names` 里 → False。**名字保留,不塞进 text** ——
    # ReWOO 的蓝图里本来就有 `LLM[...]` 这种我们没提供的伪工具,
    # 把名字抹掉会让「它计划了一个我们没有的工具」这件事查不出来。
    known: bool = True


_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(?P<body>.+?)\s*$")
_PLAN_LINE = re.compile(r"^\s*Plan\s*\d*\s*:\s*(?P<body>.+?)\s*$", re.I)
_EVIDENCE = re.compile(
    r"^\s*(?P<var>#E\d*)\s*=\s*(?P<tool>[A-Za-z_][\w-]*)\s*\[(?P<arg>.*)\]\s*$"
)


def parse_plan(text: str, tool_names: Sequence[str]) -> list[PlanItem]:
    """把 Planner 的输出解析成步骤列表。

    认三种写法,足够覆盖 ReWOO 原文和常见的 plan-then-execute 提示:
    1. ReWOO 原文:`Plan: <描述>` + `#E1 = Tool[input]`
    2. 编号列表:`1. <描述>`
    3. 项目符号:`- <描述>`

    ★ **认不出来就返回空列表,不返回「把整段当成一步」。** 后者会让一个
    解析失败看起来像一次很长的单个计划,而失败模式就此消失。
    """
    items: list[PlanItem] = []
    pending_text = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        ev = _EVIDENCE.match(line)
        if ev:
            items.append(
                PlanItem(
                    text=pending_text or f"{ev.group('tool')}[{ev.group('arg')}]",
                    evidence_var=ev.group("var"),
                    tool=ev.group("tool"),
                    argument=ev.group("arg").strip(),
                )
            )
            pending_text = ""
            continue

        plan = _PLAN_LINE.match(line)
        if plan:
            pending_text = _strip_plan_prefix(plan.group("body"))
            continue

        num = _NUMBERED.match(line)
        if num:
            items.append(PlanItem(text=_strip_plan_prefix(num.group("body"))))
            pending_text = ""
            continue

    if pending_text:
        # `Plan:` 后面没有跟 `#E = ...` —— 还没成一步,补上
        items.append(PlanItem(text=pending_text))

    # 工具名不在我们的工具表里 —— **标出来,但不改内容**。
    # 调用方（ReWOO 的 Worker）自己决定:是报错,还是当成「这一步该由 Solver 做」。
    return [
        item if not item.tool or item.tool in tool_names
        else PlanItem(text=item.text, evidence_var=item.evidence_var, tool=item.tool,
                      argument=item.argument, known=False)
        for item in items
    ]


def _strip_plan_prefix(body: str) -> str:
    return re.sub(r"^\s*Plan\s*\d*\s*:\s*", "", body, flags=re.I).strip()


def render_plan(items: Sequence[PlanItem]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        prefix = f"{item.evidence_var} = " if item.evidence_var else ""
        lines.append(f"{i}. {prefix}{item.text}")
    return "\n".join(lines)


_UNRESOLVED = re.compile(r"#E\d*")


def substitute_evidence(text: str, evidence: dict[str, str]) -> tuple[str, list[str]]:
    """把 `#E1` 换成上一步的真实观察。**返回 (结果, 没解析出来的变量名)。**

    ★ 返回第二个值是关键:ReWOO 的计划是**盲规划**的,它可以引用一个
    根本不存在的 `#E`。**把 `#E1` 原样传给工具,会变成一次莫名其妙的检索失败** ——
    而真正的原因是「计划引用了一个不存在的变量」。
    这一类必须被看见,它是 ReWOO 这一族的固有失败模式。
    """
    missing: list[str] = []

    def repl(m: re.Match) -> str:
        name = m.group(0)
        if name in evidence:
            return evidence[name]
        missing.append(name)
        return name

    return _UNRESOLVED.sub(repl, text), missing


__all__ = [
    "LoopConfig", "ParsedStep", "PlanItem", "parse_step", "parse_plan", "render_plan",
    "substitute_evidence", "extract_answer", "render_tools", "render_exemplars",
    "render_history", "first_arg_name",
    "render_step", "build_prompt", "run_loop", "REACT_STOP",
]
