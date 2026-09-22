"""接口的自检。**不联网、不要 key、不碰真数据集。**

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q

它验的是**契约**,不是准确率:字段缺了会不会炸、工具名字对不上会不会炸、
拒答有没有被看见、`confidence` 和 `correct` 有没有落在同一行。
这些正是七个人并行时最先分叉的地方。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]  # JevLoop/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.baseline.direct import Direct  # noqa: E402
from experiments.benchmark.toy import CAPITALS, ToyCapitals  # noqa: E402
from experiments.core.agent import AgentOutcome, Session, action_ask, action_answer  # noqa: E402
from experiments.core.models import CallableModel, Message  # noqa: E402
from experiments.core.registry import AGENTS, BENCHMARKS, known, register_agent  # noqa: E402
from experiments.core.runner import Cell, run_cell  # noqa: E402
from experiments.core.spec import REQUIRED_RESULT_FIELDS, Result, missing_fields  # noqa: E402
from experiments.core.tools import ToolExecutor, ToolNotFound  # noqa: E402
from experiments.core.types import Judgment, Step, Trajectory  # noqa: E402
from experiments.scripts import summarize  # noqa: E402


def half_knowledge_model() -> CallableModel:
    """知道一半国家的假模型 —— 让「对」和「错」两条路都被走到。"""
    known_half = set(sorted(CAPITALS)[: len(CAPITALS) // 2])

    def responder(messages: list[Message]) -> str:
        text = messages[-1].content
        for country, capital in CAPITALS.items():
            if f"capital of {country}" in text:
                return capital if country in known_half else "I don't know."
        return "I don't know."

    return CallableModel(responder, model_id="half")


# ═══════════════════════════════════════════════════════════
# schema —— 「缺字段的结果不收」
# ═══════════════════════════════════════════════════════════


def test_result_missing_a_field_is_a_type_error() -> None:
    """★ 这是整份 schema 的核心机制:少一个字段 = **构造时**就炸。

    不是事后校验 —— 事后校验给「先写下来回头补」留了余地。
    """
    with pytest.raises(TypeError):
        Result(run_id="r", task_id="t")  # type: ignore[call-arg]


def test_missing_fields_lists_exactly_what_is_absent() -> None:
    gaps = missing_fields({"run_id": "r", "task_id": "t"}, REQUIRED_RESULT_FIELDS)
    assert "correct" in gaps and "timing" in gaps
    assert "run_id" not in gaps and "task_id" not in gaps


def test_summarize_rejects_incomplete_rows(tmp_path: Path) -> None:
    """★ 残缺的行**不进表**,而且**要说出来** —— 悄悄跳过 = 表格看着是全的,而它不是。"""
    run = tmp_path / "toy" / "direct" / "2026-01-01T00-00-00Z-seed0"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        json.dumps({"run_id": "x", "task_id": "y", "correct": True}) + "\n", encoding="utf-8"
    )
    rows, rejected = summarize.load_rows(tmp_path)
    assert rows == []
    assert len(rejected) == 1
    assert "score" in rejected[0][1]


# ═══════════════════════════════════════════════════════════
# 工具 —— 名字对不上必须炸,不许静默返回空
# ═══════════════════════════════════════════════════════════


def test_declared_tool_without_impl_fails_at_construction() -> None:
    """声明了却没实现 → **开跑前**就炸,不是运行时才撞墙。"""
    bench = ToyCapitals()
    with pytest.raises(ValueError, match="没有实现"):
        ToolExecutor(list(bench.tools()), {})


def test_unknown_tool_raises_instead_of_returning_empty() -> None:
    """★ 静默返回空串会把「选错了工具」变成「工具返回了空」——
    正好抹掉我们要测的那件事。"""
    bench = ToyCapitals()
    executor = ToolExecutor(list(bench.tools()), bench.tool_impls())
    with pytest.raises(ToolNotFound):
        executor.call("r", "t", "no_such_tool", {})


def test_exploratory_calls_are_counted_separately() -> None:
    """有的评分器**明确不惩罚探索性调用** —— 不拆开,调用数就不可比。"""
    bench = ToyCapitals()
    executor = ToolExecutor(list(bench.tools()), bench.tool_impls())
    executor.call("r", "t", "lookup_capital", {"country": "Peru"}, necessary=True)
    executor.call("r", "t", "lookup_capital", {"country": "Japan"}, necessary=False)
    assert executor.counts() == (1, 1)


# ═══════════════════════════════════════════════════════════
# 全链路 —— benchmark × agent → log/
# ═══════════════════════════════════════════════════════════


def test_pipeline_end_to_end_writes_a_complete_log(tmp_path: Path) -> None:
    results = run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: Direct(), model=half_knowledge_model(),
        cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path,
    )
    assert len(results) == len(CAPITALS)
    assert any(r.correct for r in results) and any(not r.correct for r in results)

    run_dir = next(tmp_path.glob("toy/direct/*"))
    for name in ("cmd.txt", "meta.json", "exit.json", "results.jsonl", "usage.jsonl"):
        assert (run_dir / name).exists(), f"缺 {name}"

    exit_record = json.loads((run_dir / "exit.json").read_text(encoding="utf-8"))
    assert exit_record["completed"] is True
    assert exit_record["finished_tasks"] == len(CAPITALS)

    # 每一行都要能指回这一次 run
    ids = {json.loads(line)["run_id"] for line in (run_dir / "results.jsonl").read_text().splitlines()}
    assert ids == {results[0].run_id}


def test_meta_stamps_dirty_so_unreproducible_numbers_are_visible(tmp_path: Path) -> None:
    """★ 脏工作区跑出来的数字别人复现不了,连跑它的人自己都复现不了 —— 必须盖章。"""
    run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: Direct(), model=half_knowledge_model(),
        cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path,
    )
    meta = json.loads(next(tmp_path.glob("toy/direct/*/meta.json")).read_text(encoding="utf-8"))
    assert "dirty" in meta and isinstance(meta["dirty"], bool)
    assert meta["commit"]


def test_direct_makes_no_typed_decisions_so_no_trace(tmp_path: Path) -> None:
    """`direct` 不做类型化判定 → 没有 trace.jsonl。

    如果它有,说明有人在 direct 里塞了判定 —— 那这一臂就不是下界了。
    """
    run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: Direct(), model=half_knowledge_model(),
        cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path,
    )
    assert not list(tmp_path.glob("toy/direct/*/trace.jsonl"))


class DecideOnce:
    """假的「带判定的臂」—— 只为验证 `trace.jsonl` 的形状。"""

    name = "decide-once"

    def solve(self, session: Session) -> AgentOutcome:
        batch = session.next_batch()
        # ★ confidence 和 correct 必须落在同一行 —— RQ2 全靠这一对
        session.record_decision(
            step=0, node="isDone", answer="yes", confidence=0.83, correct=True,
            latency_ms=12.0, batch=batch,
        )
        return AgentOutcome(steps=[Step(index=0, action=action_answer("Paris"))], final_answer="Paris")


def test_trace_pairs_confidence_with_correctness(tmp_path: Path) -> None:
    run_cell(
        bench=ToyCapitals(), make_agent=lambda task, tools: DecideOnce(), model=half_knowledge_model(),
        cell=Cell(dataset="toy", arm="decide-once", seed=0), log_root=tmp_path,
    )
    lines = [json.loads(s) for s in next(tmp_path.glob("toy/decide-once/*/trace.jsonl")).read_text().splitlines()]
    assert lines, "带判定的臂必须留下 trace"
    for row in lines:
        assert {"node", "answer", "confidence", "correct", "batch", "latency_ms"} <= set(row)
        assert 0.0 <= row["confidence"] <= 1.0


def test_escalation_is_visible_in_the_result(tmp_path: Path) -> None:
    """★ 拒答必须是**能被看见**的一种结果。

    否则「拒答率」和「闸门假拒」算不出来 —— 而这两个指标只有我们的设计有。
    """

    class AlwaysAsks:
        name = "asker"

        def solve(self, session: Session) -> AgentOutcome:
            return AgentOutcome(
                steps=[Step(index=0, action=action_ask("证据不足"))],
                final_answer=None, escalated=True,
            )

    bench = ToyCapitals()
    results = run_cell(
        bench=bench, make_agent=lambda task, tools: AlwaysAsks(), model=half_knowledge_model(),
        cell=Cell(dataset="toy", arm="asker", seed=0), log_root=tmp_path,
    )
    assert all(r.escalated for r in results)
    assert all(not r.correct for r in results)
    assert all(r.failure_class == "no_answer" for r in results)


# ═══════════════════════════════════════════════════════════
# 注册表 —— 显式,不自动发现
# ═══════════════════════════════════════════════════════════


def test_duplicate_registration_fails() -> None:
    """重名会让两个数据集的 log 混在一个目录里 —— 那时候已经跑完一批了。"""
    register_agent("dup-test", lambda: Direct())
    with pytest.raises(ValueError, match="重名"):
        register_agent("dup-test", lambda: Direct())
    AGENTS.pop("dup-test")


def test_known_reports_empty_rather_than_pretending() -> None:
    """空列表本身也是有用的信息 —— 别让人以为是他参数写错了。"""
    text = known()
    assert "benchmark:" in text and "agent:" in text
    assert isinstance(BENCHMARKS, dict)


def test_toy_sampling_is_seed_reproducible() -> None:
    """★ 同一个 seed 必须永远得到同一批题 —— 否则「跑的是哪 300 条」说不清。"""
    bench = ToyCapitals()
    first = [t.task_id for t in bench.tasks(split="test", limit=4, seed=3)]
    second = [t.task_id for t in bench.tasks(split="test", limit=4, seed=3)]
    assert first == second
    assert len(first) == 4


def test_toy_scoring_normalizes_case_and_punctuation() -> None:
    bench = ToyCapitals()
    task = next(iter(bench.tasks(split="test", limit=None, seed=0)))
    assert isinstance(task.gold, str)
    messy = Trajectory(task_id=task.task_id, arm="x", final_answer=f"  {task.gold.upper()}. ")
    assert bench.score(task, messy).correct


def test_scorer_exception_does_not_become_a_pass(tmp_path: Path) -> None:
    """★ 判分器坏了**不许当成对** —— 那会把「判分器坏了」变成「我们赢了」。"""

    class Broken(ToyCapitals):
        name = "broken"

        def score(self, task, trajectory) -> Judgment:  # type: ignore[override]
            raise RuntimeError("官方评分器没装")

    results = run_cell(
        bench=Broken(), make_agent=lambda task, tools: Direct(), model=half_knowledge_model(),
        cell=Cell(dataset="broken", arm="direct", seed=0), log_root=tmp_path,
    )
    assert all(not r.correct for r in results)
    assert all(r.failure_class == "scorer_error" for r in results)
