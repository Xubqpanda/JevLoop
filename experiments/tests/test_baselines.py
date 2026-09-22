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
from experiments.baseline.plan_then_execute import PlanThenExecute  # noqa: E402
from experiments.baseline.react import ReAct  # noqa: E402
from experiments.baseline.reflexion import Reflexion  # noqa: E402
from experiments.baseline.rewoo import ReWOO  # noqa: E402
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


# ═══════════════════════════════════════════════════════════
# 计划解析 —— ReWOO 与 plan-then-execute 共用
# ═══════════════════════════════════════════════════════════


def test_parses_rewoo_blueprint_with_evidence_variables() -> None:
    items = common.parse_plan(
        "Plan: Find the author.\n#E1 = lookup_capital[Peru]\nPlan: Answer it.\n#E2 = LLM[#E1]",
        TOOL_NAMES,
    )
    assert len(items) == 2
    assert items[0].evidence_var == "#E1" and items[0].tool == "lookup_capital"
    assert items[0].known is True
    # ★ `LLM` 是原文的伪工具,我们没有 —— 名字要**保留**,不能抹掉塞进 text
    assert items[1].tool == "LLM" and items[1].known is False


def test_parses_numbered_and_bulleted_plans() -> None:
    assert len(common.parse_plan("1. first\n2. second", TOOL_NAMES)) == 2
    assert len(common.parse_plan("- first\n- second", TOOL_NAMES)) == 2


def test_unparsable_plan_is_empty_not_one_big_step() -> None:
    """★ 把整段当成一步,会让一次解析失败看起来像一个很长的计划 —— 失败模式就此消失。"""
    assert common.parse_plan("嗯……我想想", TOOL_NAMES) == []


def test_evidence_substitution_reports_missing_variables() -> None:
    """★ ReWOO 是**盲规划**,可以引用一个根本不存在的 `#E`。

    原样把 `#E9` 传给工具会变成一次莫名其妙的检索失败,而真正的原因是计划错了。
    所以第二个返回值（没解析出来的变量名）必须被用上。
    """
    text, missing = common.substitute_evidence("answer is #E1 and #E9", {"#E1": "Lima"})
    assert text == "answer is Lima and #E9"
    assert missing == ["#E9"]


# ═══════════════════════════════════════════════════════════
# ReWOO —— Plan / Work / Solve
# ═══════════════════════════════════════════════════════════


def test_rewoo_plans_once_then_solves(tmp_path: Path) -> None:
    """★ 它的本质:**一次规划到底,中途不看观察。** 两次调用,不是每步一次。"""
    model = ScriptedModel([
        "Plan: Look it up.\n#E1 = lookup_capital[Peru]",
        "Lima",
    ])
    session = _session(tmp_path, arm="rewoo")
    session.model = model
    outcome = ReWOO().solve(session)

    assert outcome.final_answer == "Lima"
    assert len(session.model_calls) == 2, "Planner 一次 + Solver 一次"
    assert [s.action.kind for s in outcome.steps if s.action.kind == "tool"] == ["tool"]


def test_rewoo_pseudo_tool_is_not_a_failure(tmp_path: Path) -> None:
    """★★ 回归:早先 `LLM[...]` 被记成 unresolved → `error` 被置上 →
    **runner 一看到 error 就把答案判否,一个正确答案就这么丢了**（实测）。

    `LLM[...]` 是原文的伪工具,而且我们在 prompt 里明确承诺过它可用 ——
    它是 Solver 的活,不是失败。
    """
    model = ScriptedModel([
        "Plan: Look it up.\n#E1 = lookup_capital[Peru]\nPlan: Read it.\n#E2 = LLM[#E1]",
        "Lima",
    ])
    session = _session(tmp_path, arm="rewoo")
    session.model = model
    outcome = ReWOO().solve(session)

    assert outcome.final_answer == "Lima"
    assert outcome.error is None, "伪工具不该让整次运行变成 error"
    skipped = [s for s in outcome.steps if "skipped" in s.observation]
    assert len(skipped) == 1, "跳过的步骤要留在轨迹里,失败分类从那里抓"


def test_rewoo_unknown_tool_is_recorded_but_does_not_discard_the_answer(tmp_path: Path) -> None:
    """不认识的工具名 = 计划错了 —— **记下来,但别因此丢掉 Solver 给出的答案**。"""
    model = ScriptedModel([
        "Plan: Use a tool we do not have.\n#E1 = imaginary_tool[x]",
        "Lima",
    ])
    session = _session(tmp_path, arm="rewoo")
    session.model = model
    outcome = ReWOO().solve(session)
    assert outcome.final_answer == "Lima"
    assert outcome.error is None
    assert any("imaginary_tool" in s.observation for s in outcome.steps)


def test_rewoo_unparsable_plan_escalates(tmp_path: Path) -> None:
    """计划都解析不出来 → **弃答**。不许把整段当计划硬跑。"""
    session = _session(tmp_path, arm="rewoo")
    session.model = ScriptedModel(["嗯……我不想做这个"])
    outcome = ReWOO().solve(session)
    assert outcome.escalated and outcome.final_answer is None
    assert "解析" in (outcome.error or "")


# ═══════════════════════════════════════════════════════════
# plan-then-execute —— 执行段**看得到观察**
# ═══════════════════════════════════════════════════════════


def test_plan_then_execute_shows_the_plan_to_the_executor(tmp_path: Path) -> None:
    """★ 与 ReWOO 的关键差别:执行段是**有观察的循环**,而且计划进了 prompt。"""
    model = ScriptedModel([
        "1. Look up the capital.\n2. Report it.",
        "Action: lookup_capital[Peru]",
        "Action: finish[Lima]",
    ])
    session = _session(tmp_path, arm="plan-then-execute")
    session.model = model
    outcome = PlanThenExecute().solve(session)

    assert outcome.final_answer == "Lima"
    assert len(outcome.steps) >= 3  # 计划 + 工具 + 收尾
    # 计划确实进了执行段的 prompt
    exec_prompt = session.prompts[1]["messages"][-1]["content"]
    assert "# Plan" in exec_prompt and "Look up the capital" in exec_prompt


def test_plan_then_execute_still_executes_when_the_plan_is_unparsable(tmp_path: Path) -> None:
    """解析不出计划 → **仍然执行**（带空计划）。

    理由:它的执行段本身是有观察的循环,有能力自己把任务做掉 ——
    这一点和 ReWOO 不同（ReWOO 的执行段是盲的,没计划就真的没得跑）。
    """
    model = ScriptedModel(["嗯……", "Action: lookup_capital[Peru]", "Action: finish[Lima]"])
    session = _session(tmp_path, arm="plan-then-execute")
    session.model = model
    outcome = PlanThenExecute().solve(session)
    assert outcome.final_answer == "Lima"


# ═══════════════════════════════════════════════════════════
# Reflexion —— 臂名把 Evaluator 的选择写死
# ═══════════════════════════════════════════════════════════


def test_reflexion_retries_with_its_own_reflection(tmp_path: Path) -> None:
    model = ScriptedModel([
        "Action: finish[Wrong]",       # trial 1：答错
        "我查错了地方，下次直接查工具。",  # 反思
        "Action: finish[Lima]",        # trial 2：答对
    ])
    session = _session(tmp_path, arm="reflexion")
    session.model = model
    session.check_answer = lambda answer: answer == "Lima"

    outcome = Reflexion(max_trials=3).solve(session)

    assert outcome.final_answer == "Lima" and not outcome.escalated
    # ★ 反思是这个方法的**产物**,必须进轨迹 —— 审稿人最会问的就是它反思出了什么
    assert any(s.action.kind == "ask" and "reflection" in s.action.content for s in outcome.steps)
    # 第二次 trial 的 prompt 里带着上次的反思
    second_trial = session.prompts[-1]["messages"][-1]["content"]
    assert "# Memory" in second_trial


def test_reflexion_exhausting_trials_discards_the_last_answer(tmp_path: Path) -> None:
    """★ 三次都没过 Evaluator → **弃答,并把最后一次的答案丢掉**。

    留着一个「我们自己判定它没答对」的答案当结果,是在假装成功。
    """
    model = ScriptedModel(["Action: finish[Wrong]"] * 1 + ["反思"] * 1 + ["Action: finish[Wrong]"] * 1
                          + ["反思"] * 1 + ["Action: finish[Wrong]"] * 1)
    session = _session(tmp_path, arm="reflexion")
    session.model = model
    session.check_answer = lambda answer: False

    outcome = Reflexion(max_trials=3).solve(session)
    assert outcome.escalated and outcome.final_answer is None
    assert "3 次尝试" in (outcome.error or "")


def test_reflexion_arm_name_encodes_which_evaluator_was_used() -> None:
    """★★ **一个吃了金标信号的臂,不能和没吃的同列一张表。**
    表里的主键就是臂名,所以臂名必须把 Evaluator 的选择写死。
    """
    assert Reflexion(uses_success_signal=True).name == "reflexion"
    assert Reflexion(uses_success_signal=False).name == "reflexion-selfeval"
    assert Reflexion().needs_success_signal is True
    assert Reflexion(uses_success_signal=False).needs_success_signal is False


def test_reflexion_refuses_to_silently_fall_back_to_self_evaluation(tmp_path: Path) -> None:
    """★ 声明要吃信号、benchmark 却没实现 `check()` → **抛,不许静默退回自评**。

    静默退回会让两个臂在同一张表里混着两种强度的 Evaluator,
    而表上看不出来 —— 那比直接报错糟得多。
    """
    session = _session(tmp_path, arm="reflexion")
    session.model = ScriptedModel(["Action: finish[Lima]"])
    session.check_answer = None
    with pytest.raises(RuntimeError, match="check\\(\\)"):
        Reflexion(uses_success_signal=True).solve(session)


def test_runner_only_wires_the_signal_for_arms_that_ask(tmp_path: Path) -> None:
    """★ 接缝是显式的:没声明的臂**拿不到** `check_answer`,而不是靠自觉不读。"""
    seen: dict[str, object] = {}

    class Snoop:
        name = "snoop"

        def solve(self, session: Session) -> AgentOutcome:  # type: ignore[override]
            seen["signal"] = session.check_answer
            return AgentOutcome(final_answer="x")

    run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: Snoop(), model=ScriptedModel(["x"]),
        cell=Cell(dataset="toy", arm="snoop", seed=0), log_root=tmp_path,
    )
    assert seen["signal"] is None, "没声明 needs_success_signal 的臂不该拿到成败信号"


def test_dotted_tool_names_parse() -> None:
    """★★ 回归:BFCL 的函数名**合法地带点**（`math.hcf` / `triangle_properties.get`）。

    早先 `_BRACKET` 的名字部分是 `[\\w-]*`，`.` 不在里面 —— 于是那些调用全被判成
    「解析不出动作」→ 重试 → 弃答。**实测 bfcl-simple × act 的失败里 23 条有 8 条
    是这一个字符造成的，而模型输出完全正确。**

    解析器拒掉一个合法名字，在读数上看起来就是「模型不会用工具」。
    """
    names = ["math.hcf", "triangle_properties.get", "history_api.get_president_by_year"]
    parsed = common.parse_step(
        'Action: math.hcf[{"number1": 36, "number2": 24}]', names, first_arg="math.hcf",
    )
    assert parsed.kind == "tool", "带点的函数名被拒了"
    assert parsed.tool == "math.hcf"
    assert parsed.arguments == {"number1": 36, "number2": 24}

    plain = common.parse_step("Action: history_api.get_president_by_year[1940]", names,
                              first_arg="history_api.get_president_by_year")
    assert plain.kind == "tool" and plain.syntax == "bracket"


def test_dotted_name_that_is_not_a_tool_is_still_unparsed_with_evidence() -> None:
    parsed = common.parse_step("Action: no.such[tool]", ["math.hcf"])
    assert parsed.kind == "unparsed" and "no.such" in parsed.raw
