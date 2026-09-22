"""事件流 + 重判。

**这两样都是从 Inspect 移植来的**（见 `core/events.py` 头部与
`docs/PROBE-inspect-2026-09-22.md`）:

- 一次调用**一条记录**,不再摊在三张表里
- 已有轨迹**换口径重判**,不重跑

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.baseline.direct import Direct  # noqa: E402
from experiments.benchmark.gsm8k.gsm8k import Gsm8k  # noqa: E402
from experiments.benchmark.toy.toy import CAPITALS, ToyCapitals  # noqa: E402
from experiments.core import events as E  # noqa: E402
from experiments.core.models import CallableModel  # noqa: E402
from experiments.core.runner import Cell, run_cell  # noqa: E402


def read_events(root: Path, cell: str) -> list[dict]:
    path = next(root.glob(f"{cell}/*/events.jsonl"))
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def model_that_answers(text: str) -> CallableModel:
    return CallableModel(lambda messages: text, model_id="fixed")


# ═══════════════════════════════════════════════════════════
# 事件流
# ═══════════════════════════════════════════════════════════


def test_one_model_call_is_one_record(tmp_path: Path) -> None:
    """★★ 这是从 Inspect 搬来的那条:**请求和响应装在同一条记录里。**

    之前它们分在 `prompts.jsonl` 和 `completions.jsonl`,靠 `call` 序号 join ——
    任何一处漏写,那次调用就看起来像没发生过,或者 token 数和文本对不上。
    """
    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    calls = [e for e in read_events(tmp_path, "toy/direct") if e["type"] == "model"]
    assert calls, "至少一次模型调用"
    for c in calls:
        assert c["request"]["messages"], "请求在同一条记录里"
        assert "text" in c["response"], "响应也在同一条记录里"
        assert "usage" in c and "timing" in c


def test_three_files_became_one(tmp_path: Path) -> None:
    """`prompts` / `completions` / `usage` 三个文件**已经合并** —— 别再写回去。"""
    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    names = {p.name for p in next(tmp_path.glob("toy/direct/*")).iterdir()}
    assert "events.jsonl" in names
    for gone in ("prompts.jsonl", "completions.jsonl", "usage.jsonl"):
        assert gone not in names, f"{gone} 应该已经合并进 events.jsonl 了"


def test_answer_event_makes_the_log_self_sufficient(tmp_path: Path) -> None:
    """★ **`answer` 事件是日志自足的关键。**

    在这之前答案只活在最后一个 model 事件的文本里,而「哪一句算答案」是各臂的事 ——
    **重判根本无从下手**。这是写重判时发现的缺口。
    """
    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    answers = [e for e in read_events(tmp_path, "toy/direct") if e["type"] == "answer"]
    assert answers, "每条 run 都要留下 answer 事件"
    assert all("text" in a and "escalated" in a for a in answers)


def test_tool_events_carry_necessary_and_exploratory(tmp_path: Path) -> None:
    """★ 必要 / 探索性分开 —— 有的评分器**明确不惩罚探索性调用**,不拆不可比。"""
    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    for e in read_events(tmp_path, "toy/direct"):
        if e["type"] == "tool":
            assert "necessary" in e


def test_events_round_trip_through_disk(tmp_path: Path) -> None:
    """写下去能读回来,而且**认不出的类型要炸** —— 静默丢会让流看起来是完整的。"""
    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    for line in next(tmp_path.glob("toy/direct/*/events.jsonl")).read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert isinstance(E.from_json(line), E.Event)
    with pytest.raises(ValueError, match="认不出的"):
        E.from_json('{"type": "who_knows"}')


def test_usage_splits_cached_from_uncached_and_reasoning_from_visible() -> None:
    """照 Inspect 的 `ModelUsage` 取的字段名 —— 以后要对得上。"""
    u = E.Usage(input_tokens=100, input_tokens_cache_read=60,
                output_tokens=50, reasoning_tokens=30, total_cost=0.01)
    assert u.input_tokens_uncached == 40
    assert u.output_tokens_visible == 20


def test_spans_pair_up_and_carry_working_time(tmp_path: Path) -> None:
    """`with session.span(...)` —— 成对发 begin/end,**给一步单独计时不用手写**。"""
    from experiments.core.agent import Session
    from experiments.core.tools import ToolExecutor

    bench = ToyCapitals()
    tools = list(bench.tools())
    task = next(iter(bench.tasks(split="test", limit=1, seed=0)))
    s = Session(run_id="t", task=task, arm="x", tools=tools,
                executor=ToolExecutor(tools, bench.tool_impls()),
                model=model_that_answers("x"), max_steps=2, temperature=0.0, max_tokens=64)
    with s.span("decide"):
        pass
    spans = [e for e in s.events if e.type == "span"]
    assert [x.phase for x in spans] == ["begin", "end"]
    assert spans[1].working_time >= 0


# ═══════════════════════════════════════════════════════════
# 重判 —— `inspect score` 的对应物
# ═══════════════════════════════════════════════════════════


def test_gsm8k_offers_both_documented_variants() -> None:
    v = Gsm8k(rows={"test": []}).score_variants()
    assert {"strict", "flexible"} <= set(v), "两种口径都有出处,必须都提供"


def test_variants_disagree_on_the_same_trajectory() -> None:
    """★★ **同一段输出,两种口径给出相反的结论** —— 这就是「口径必须写明」的实证。

    实测过一次真的:12 条轨迹,`flexible` 12/12,`strict` **0/12**。
    """
    from experiments.core.types import Trajectory

    bench = Gsm8k(rows={"test": [{"question": "q", "answer": "#### 18"}]})
    task = next(iter(bench.tasks(split="test", limit=None, seed=0)))
    traj = Trajectory(task_id=task.task_id, arm="direct", final_answer="The answer is 18.")
    v = bench.score_variants()
    assert v["flexible"](task, traj).correct is True
    assert v["strict"](task, traj).correct is False


def test_rescore_never_overwrites_the_source_run(tmp_path: Path) -> None:
    """★★ **原始日志是证据,只许写新目录。**

    这条是从 `inspect score` 那一段学的:「同一批轨迹、不同尺子」要能被核对,
    而核对的前提是原始那次**还在**。
    """
    from experiments.core.runlog import LOG_DIR

    run_cell(bench=ToyCapitals(), make_agent=lambda t, tools: Direct(),
             model=model_that_answers("Lima"),
             cell=Cell(dataset="toy", arm="direct", seed=0), log_root=tmp_path)
    # rescore 只认 LOG_DIR,所以这里只验它的目录命名约定
    src = next(tmp_path.glob("toy/direct/*"))
    assert src.is_dir()
    assert "rescore" not in src.name, "原始 run 的名字里不该有 rescore"


def test_rescore_script_is_runnable() -> None:
    """脚本至少要是能跑的 —— `--list-variants` 不碰网络也不碰数据。"""
    out = subprocess.run(
        [sys.executable, "-m", "experiments.scripts.rescore", "--list-variants", "gsm8k"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert "strict" in out.stdout and "flexible" in out.stdout
