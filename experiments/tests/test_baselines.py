"""`baseline/` 的测试。**解析器和循环骨架**是 bug 最爱藏的地方。

不联网、不要 key。跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]  # JevLoop/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.baseline import common  # noqa: E402
from experiments.baseline.act import Act  # noqa: E402
from experiments.baseline.direct import Direct  # noqa: E402
from experiments.baseline.react import ReAct  # noqa: E402
from experiments.benchmark.toy import ToyCapitals  # noqa: E402
from experiments.core.agent import AgentOutcome, Session  # noqa: E402
from experiments.core.models import ScriptedModel  # noqa: E402
from experiments.core.runner import Cell, run_cell  # noqa: E402
from experiments.core.tools import ToolExecutor  # noqa: E402

TOOL_NAMES = ["lookup_capital"]


# ═══════════════════════════════════════════════════════════
# 解析 —— 容错,但**不许猜**
# ═══════════════════════════════════════════════════════════


def test_parses_the_paper_bracket_form() -> None:
    parsed = common.parse_step(
        "Thought: I need to look this up.\nAction: lookup_capital[Peru]",
        TOOL_NAMES, first_arg="lookup_capital",
    )
    assert parsed.kind == "tool"
    assert parsed.tool == "lookup_capital"
    assert parsed.arguments == {"lookup_capital": "Peru"} or parsed.arguments == {"Peru": "Peru"} or "Peru" in str(parsed.arguments)
    assert parsed.thought == "I need to look this up."
    assert parsed.syntax == "bracket"


def test_finish_is_an_answer_not_a_tool() -> None:
    parsed = common.parse_step("Thought: done.\nAction: finish[Lima]", TOOL_NAMES)
    assert parsed.kind == "answer"
    assert parsed.answer == "Lima"


def test_action_input_variant_is_accepted() -> None:
    """`Action: lookup_capital` + `Action Input: Peru` —— 常见变体,得认。"""
    parsed = common.parse_step(
        "Action: lookup_capital\nAction Input: Peru", TOOL_NAMES, first_arg="lookup_capital",
    )
    assert parsed.kind == "tool" and parsed.syntax == "action-input"


def test_json_form_is_accepted() -> None:
    parsed = common.parse_step(
        'Action: ```json\n{"action": "lookup_capital", "action_input": {"country": "Peru"}}\n```',
        TOOL_NAMES, first_arg="lookup_capital",
    )
    assert parsed.kind == "tool" and parsed.syntax == "json"
    assert parsed.arguments == {"country": "Peru"}


def test_unknown_tool_is_unparsed_but_keeps_the_evidence() -> None:
    """★ 认不出来也要说清楚它想调什么 —— 那是「选错工具」的原始证据。

    静默丢掉它,失败分类里就永远看不到这一类。
    """
    parsed = common.parse_step("Action: no_such_tool[x]", TOOL_NAMES)
    assert parsed.kind == "unparsed"
    assert "no_such_tool" in parsed.raw


def test_garbage_is_unparsed_and_never_guessed_as_an_answer() -> None:
    """★ 最坏的一种处理是把认不出来的东西当成答案 ——
    它把「模型格式错了」变成「模型答了」,而失败模式就此消失。"""
    parsed = common.parse_step("I think the answer might be Paris, but I am not sure.", TOOL_NAMES)
    assert parsed.kind == "unparsed"
    assert parsed.answer == ""


def test_extract_answer_returns_empty_rather_than_the_whole_text() -> None:
    """返回整段 = 让评测器拿一整段话去比标准答案,**真正的失败原因就看不见了**。"""
    assert common.extract_answer("no marker here") == ""
    assert common.extract_answer("Final Answer: Paris") == "Paris"
    assert common.extract_answer("Action: finish[Paris]") == "Paris"


# ═══════════════════════════════════════════════════════════
# 工具描述 —— 全臂逐字一致
# ═══════════════════════════════════════════════════════════


def test_render_tools_is_stable_and_marks_optional_and_enum() -> None:
    """工具描述必须逐字一致（PLAN §2 的公平性要求）,而且可选参数与闭集要标出来。"""
    rendered = common.render_tools(ToyCapitals().tools())
    assert "lookup_capital" in rendered
    assert "one of" in rendered, "闭集要标出来 —— 那是判定模型能枚举的前提"
    assert common.render_tools(ToyCapitals().tools()) == rendered, "同样的输入必须给同样的串"


def test_render_tools_handles_no_tools() -> None:
    assert "no tools" in common.render_tools([])


# ═══════════════════════════════════════════════════════════
# act 与 react —— **只差一个 Thought**
# ═══════════════════════════════════════════════════════════


def test_react_and_act_differ_only_in_the_thought_flag() -> None:
    """★ 这是这个文件存在的理由。

    我们现有的 `bench/react.ts` 里 `thought` 出现 0 次 —— 也就是说它跑出来的是 act,
    而标签写着 ReAct。拆成两个文件、两个名字之后,这件事再也不能悄悄发生。
    """
    react, act = ReAct(), Act()
    assert react.config.with_thought is True
    assert act.config.with_thought is False
    assert react.config.stop == act.config.stop, "截断点必须一样,否则 token 数不可比"
    assert react.config.max_parse_retries == act.config.max_parse_retries


def test_react_prompt_asks_for_a_thought_and_act_forbids_it(tmp_path: Path) -> None:
    session = _session(tmp_path, arm="react")
    react_prompt = common.build_prompt(session, ReAct().config, [])
    act_prompt = common.build_prompt(session, Act().config, [])
    assert "Thought:" in react_prompt and "Do not output a Thought line" not in react_prompt
    assert "Do not output a Thought line" in act_prompt
    # 工具那一段必须逐字相同 —— 否则两条臂差的不只是 thought
    assert common.render_tools(session.tools) in react_prompt
    assert common.render_tools(session.tools) in act_prompt


# ═══════════════════════════════════════════════════════════
# 循环 —— 走完整路径,并且**把该留的留下**
# ═══════════════════════════════════════════════════════════


def test_react_loop_calls_the_tool_then_finishes(tmp_path: Path) -> None:
    model = ScriptedModel([
        "Thought: look it up.\nAction: lookup_capital[Peru]",
        "Thought: got it.\nAction: finish[Lima]",
    ])
    outcome = _run(tmp_path, ReAct(), model)
    assert outcome.final_answer == "Lima"
    assert not outcome.escalated
    assert [s.action.kind for s in outcome.steps] == ["tool", "answer"]
    assert outcome.steps[0].observation == "Lima", "工具的观察要落进轨迹"
    assert outcome.steps[0].thought == "look it up.", "★ thought 必须存进 Step,否则消融做不了"


def test_act_loop_also_works_and_records_no_thought(tmp_path: Path) -> None:
    model = ScriptedModel([
        "Action: lookup_capital[Peru]",
        "Action: finish[Lima]",
    ])
    outcome = _run(tmp_path, Act(), model)
    assert outcome.final_answer == "Lima"
    assert all(s.thought == "" for s in outcome.steps)


def test_scratchpad_contains_the_thought_only_when_asked(tmp_path: Path) -> None:
    from experiments.core.types import Action, Step

    steps = [Step(index=0, action=Action(kind="tool", name="lookup_capital"), observation="Lima",
                  thought="look it up")]
    assert "Thought: look it up" in common.render_history(steps, with_thought=True)
    assert "Thought:" not in common.render_history(steps, with_thought=False)
    assert "Observation: Lima" in common.render_history(steps, with_thought=False)


def test_running_out_of_steps_escalates_instead_of_answering(tmp_path: Path) -> None:
    """★ 步数用完 → **弃答**,不是硬凑一个答案。弃答是一种能被看见的结果。"""
    model = ScriptedModel(["Action: lookup_capital[Peru]"] * 20)
    outcome = _run(tmp_path, ReAct(), model, max_steps=3)
    assert outcome.escalated
    assert outcome.final_answer is None
    assert "3 步" in (outcome.error or "")


def test_unparsable_output_retries_then_escalates(tmp_path: Path) -> None:
    """解析不出来最多重试 `max_parse_retries` 次,再不行就弃答 —— **不许猜**。"""
    model = ScriptedModel(["我觉得可能是利马", "嗯……", "还是不知道"])
    outcome = _run(tmp_path, ReAct(), model)
    assert outcome.escalated
    assert outcome.final_answer is None
    assert "解析不出动作" in (outcome.error or "")


def test_tool_error_is_fed_back_as_an_observation(tmp_path: Path) -> None:
    """工具自己抛了 → 当成观察喂回去让模型改,**不是让整轮崩掉**。

    但也**不许吞**:错误原文要进观察,后面才看得出「是模型选错了还是工具坏了」。
    """
    model = ScriptedModel([
        "Action: lookup_capital",           # 缺参数 → 工具实现会抛 TypeError
        "Action: finish[Lima]",
    ])
    outcome = _run(tmp_path, ReAct(), model)
    assert outcome.steps[0].observation.startswith("Tool error:")


def test_react_arm_end_to_end_through_the_runner(tmp_path: Path) -> None:
    """整条链路:benchmark × react → log/,和 direct 走的是同一套记账。"""
    model = ScriptedModel(
        ["Thought: t.\nAction: lookup_capital[Peru]", "Thought: t.\nAction: finish[Lima]"] * 10
    )
    results = run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: ReAct(), model=model,
        cell=Cell(dataset="toy", arm="react", seed=0), log_root=tmp_path,
    )
    assert results, "至少跑出一条"
    first = results[0]
    assert first.cost.llm_calls == 2, "react 两步:查工具 + 收尾"
    assert first.cost.tool_calls == 1
    assert first.timing.round_trips == 2


def test_direct_still_makes_exactly_one_call(tmp_path: Path) -> None:
    """回归:`direct` 不该被循环那套东西影响。"""
    model = ScriptedModel(["Paris"])
    results = run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: Direct(), model=model,
        cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path,
    )
    assert all(r.cost.llm_calls == 1 for r in results)
    assert all(r.cost.tool_calls == 0 for r in results)


# ═══════════════════════════════════════════════════════════
# 夹具
# ═══════════════════════════════════════════════════════════


def _session(tmp_path: Path, *, arm: str, max_steps: int = 20) -> Session:
    bench = ToyCapitals()
    tools = list(bench.tools())
    task = next(iter(bench.tasks(split="test", limit=1, seed=0)))
    return Session(
        run_id=f"test/{arm}", task=task, arm=arm, tools=tools,
        executor=ToolExecutor(tools, bench.tool_impls()),
        model=ScriptedModel([]), max_steps=max_steps, temperature=0.0, max_tokens=256,
    )


def _run(tmp_path: Path, agent, model, *, max_steps: int = 20) -> AgentOutcome:
    session = _session(tmp_path, arm=agent.name, max_steps=max_steps)
    session.model = model
    return agent.solve(session)
