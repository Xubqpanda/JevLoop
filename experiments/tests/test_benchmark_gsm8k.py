"""`benchmark/gsm8k` 的测试。**不联网** —— rows 直接喂进去。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q

重点在**判分口径**和**抽样可复现**，不在准确率。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.gsm8k.gsm8k import (  # noqa: E402
    Gsm8k,
    _calculate,
    extract_gold,
    extract_pred,
    numeric_equal,
)
from experiments.core.runner import Cell, run_cell  # noqa: E402
from experiments.core.models import CallableModel  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402

ROWS = {
    "test": [
        {"question": "Janet's ducks lay 16 eggs per day.", "answer": "She sells them. #### 18"},
        {"question": "A robe takes 2 bolts of blue fiber.", "answer": "#### 3"},
        {"question": "Josh decides to try flipping a house.", "answer": "#### 70000"},
    ]
}


def bench() -> Gsm8k:
    return Gsm8k(rows=ROWS)


# ═══════════════════════════════════════════════════════════
# 抠金标 / 抠预测 —— 判分口径全在这里
# ═══════════════════════════════════════════════════════════


def test_gold_is_the_part_after_the_marker() -> None:
    assert extract_gold("lots of reasoning #### 18") == "18"


def test_gold_without_the_marker_returns_everything_not_empty() -> None:
    """★ 金标字段不规范时返回整段,不是空串。

    返回空串会把「数据有问题」变成「模型答错了」—— 那是最难查的一类。
    """
    assert extract_gold("no marker here") == "no marker here"


def test_strict_requires_the_marker_and_does_not_guess() -> None:
    """★ `strict` 下没写 `####` 就是**没答**,不是「取最后一个数」。"""
    assert extract_pred("the answer is 18", "strict") == ""
    assert extract_pred("reasoning #### 18", "strict") == "18"


def test_flexible_takes_the_last_number() -> None:
    assert extract_pred("first 3 then 18", "flexible") == "18"
    assert extract_pred("so $1,234.5 total", "flexible") == "1234.5"


def test_both_modes_return_empty_rather_than_guessing() -> None:
    assert extract_pred("I cannot solve this", "flexible") == ""
    assert extract_pred("", "strict") == ""


def test_numeric_equal_is_total_not_raising() -> None:
    """判分器不许因为一个畸形输入把整轮带走。"""
    assert numeric_equal("18", "18")
    assert numeric_equal("18.0", "18")
    assert not numeric_equal("18", "19")
    assert not numeric_equal("", "18")
    assert not numeric_equal("abc", "18")


def test_headline_extract_must_be_one_of_the_two_documented_modes() -> None:
    with pytest.raises(ValueError, match="strict / flexible"):
        Gsm8k(rows=ROWS, headline_extract="vibes")


# ═══════════════════════════════════════════════════════════
# 抽样 —— 必须由 seed 决定
# ═══════════════════════════════════════════════════════════


def test_sampling_is_reproducible_for_a_seed() -> None:
    ids = [t.task_id for t in bench().tasks(split="test", limit=2, seed=3)]
    again = [t.task_id for t in bench().tasks(split="test", limit=2, seed=3)]
    assert ids == again


def test_sampling_actually_depends_on_the_seed() -> None:
    """★★ 回归:第一版写的是 `seed % 1` —— 对 int **恒等于 0**,种子等于没起作用。

    这个测试就是为它写的:换个 seed 必须换一批题。
    """
    a = [t.meta["gold_raw"] for t in bench().tasks(split="test", limit=2, seed=0)]
    b = [t.meta["gold_raw"] for t in bench().tasks(split="test", limit=2, seed=1)]
    assert a != b, "换 seed 拿到同一批题 → 抽样根本没用到 seed"


def test_limit_none_returns_everything() -> None:
    assert len(list(bench().tasks(split="test", limit=None, seed=0))) == len(ROWS["test"])


def test_limit_larger_than_the_split_returns_everything() -> None:
    assert len(list(bench().tasks(split="test", limit=99, seed=0))) == len(ROWS["test"])


# ═══════════════════════════════════════════════════════════
# 工具 —— 默认没有,开了就得能跑
# ═══════════════════════════════════════════════════════════


def test_no_tools_by_default() -> None:
    """★ 默认不给工具是有意的:这三个工具**没有带标注的「该选哪个」岔路**,

    `needsTool` 会恒为否、零方差。理由在 docs/PLAN-*.md §1。
    所以这个 loader **验不到工具那条路** —— 第二个 loader 必须验。
    """
    assert bench().tools() == ()
    assert bench().tool_impls() == {}


def test_calculator_is_opt_in_and_has_an_impl() -> None:
    b = Gsm8k(rows=ROWS, with_calculator=True)
    assert [t.name for t in b.tools()] == ["calculate"]
    assert set(b.tool_impls()) == {"calculate"}, "声明了就要有实现,否则 ToolExecutor 开跑前就抛"


@pytest.mark.parametrize("expr,expect", [
    ("16 - 3 - 4", "9"),
    ("2 * (3 + 4)", "14"),
    ("10 / 4", "2.5"),
    ("7 / 2", "3.5"),
])
def test_calculator_does_arithmetic(expr: str, expect: str) -> None:
    assert _calculate(expr) == expect


def test_calculator_refuses_anything_but_arithmetic() -> None:
    """★ **不是 `eval`。** 这条路径的输入来自模型输出,
    直接 eval 等于把任意代码执行权交出去 —— 必须挡住,而且要说清挡了什么。
    """
    assert _calculate("__import__('os').getcwd()").startswith("Error:")
    assert _calculate("open('/etc/passwd')").startswith("Error:")
    assert _calculate("1 / 0").startswith("Error:")


# ═══════════════════════════════════════════════════════════
# 判分 —— 两种失败要分得开
# ═══════════════════════════════════════════════════════════


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="gsm8k/test/0", arm="direct", final_answer=answer)


def test_correct_when_the_number_matches() -> None:
    b, task = bench(), next(iter(bench().tasks(split="test", limit=None, seed=0)))
    assert b.score(task, _traj("18")).correct


def test_unparsable_is_its_own_failure_class() -> None:
    """★ 「抠不出数」和「算错了」必须分开 —— 修法完全不同。"""
    b = bench()
    task = next(iter(b.tasks(split="test", limit=None, seed=0)))
    got = b.score(task, _traj("I cannot solve this"))
    assert not got.correct and got.failure_class == "no_numeric_answer"


def test_wrong_number_is_a_different_failure_class() -> None:
    b = bench()
    task = next(iter(b.tasks(split="test", limit=None, seed=0)))
    got = b.score(task, _traj("19"))
    assert not got.correct and got.failure_class == "wrong_answer"


def test_empty_answer_is_no_answer_not_a_wrong_number() -> None:
    b = bench()
    task = next(iter(b.tasks(split="test", limit=None, seed=0)))
    assert b.score(task, _traj("   ")).failure_class == "no_answer"


def test_strict_mode_reports_a_format_failure_where_flexible_succeeds() -> None:
    """★ 这就是「口径必须写明」的实证:同一段输出,两种口径给出相反的结论。"""
    b = bench()
    task = next(iter(b.tasks(split="test", limit=None, seed=0)))
    text = "Let me work it out. The answer is 18."
    assert Gsm8k(rows=ROWS, headline_extract="flexible").score(task, _traj(text)).correct
    assert not Gsm8k(rows=ROWS, headline_extract="strict").score(task, _traj(text)).correct


# ═══════════════════════════════════════════════════════════
# 整条链路 —— 和别的 arm 走同一套记账
# ═══════════════════════════════════════════════════════════


def test_end_to_end_through_the_runner(tmp_path: Path) -> None:
    b = bench()
    model = CallableModel(lambda messages: "The answer is 18.", model_id="fixed")

    class Blind:
        name = "blind"

        def solve(self, session):  # type: ignore[no-untyped-def]
            from experiments.core.agent import AgentOutcome, action_answer
            from experiments.core.types import Step

            reply = session.call_model([__import__(
                "experiments.core.models", fromlist=["Message"]
            ).Message(role="user", content=session.task.prompt)])
            return AgentOutcome(steps=[Step(index=0, action=action_answer(reply.text))],
                                final_answer=reply.text)

    results = run_cell(
        bench=b, make_agent=lambda task, tools: Blind(), model=model,
        cell=Cell(dataset=b.name, arm="blind", seed=0), log_root=tmp_path,
    )
    assert len(results) == len(ROWS["test"])
    assert results[0].correct
    assert all(r.failure_class in ("none", "no_numeric_answer", "wrong_answer") for r in results)
