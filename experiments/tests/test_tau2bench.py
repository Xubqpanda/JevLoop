"""τ²-bench —— 钉住**钉版本**、**两方结构**、和**缺一层时抛**这三件事。

★ 这个文件最值钱的一条是 `test_the_user_scenario_never_reaches_the_prompt`:
τ² 是两方任务,agent 只该看到业务规则(policy),而用户想干什么
是**用户模拟器的秘密**。把 `user_scenario` 写进 prompt 等于把
「要通过对话发现的东西」直接告诉 agent —— 而那样跑出来的分**看起来会更好**,
且看不出为什么。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.tau2bench import tau2bench as tb  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402

_HAS_DATA = (tb.DATASET_DIR / "tau2-bench" / "airline" / "tasks.json").is_file()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="τ²-bench 数据没抽出来")


# ═══════════════════════════════════════════════════════════
# ★★★ 钉版本 —— 不钉就拿到另一个 benchmark
# ═══════════════════════════════════════════════════════════


def test_the_version_is_pinned_to_the_tag_not_main() -> None:
    """★★★ **仓库 HEAD 现在是 τ³-bench v1.0.1,不是论文那个 τ²。**

    而它自己的 README 写着 `<1.0.1` 的结果不能和 `>=1.0.1` 比。
    ⇒ 拿 HEAD 跑出来的数**不能和论文比**,而**目录名不会提醒你**。

    ★ 另一个只有 v0.1.0 才成立的事实:那个版本的 `tasks.json`
      **就是 base 划分**（278 条）;HEAD 上多了 `split_tasks.json`,
      base 要从里面挑 —— 换版本连「哪 278 条」都会变。
    """
    spec = tb.Tau2Bench().downloads()[0]
    assert spec.pinned, "必须钉 tag —— main 等于哪天下的算哪天"
    assert tb.TAU2_TAG in spec.locator
    assert spec.revision == tb.TAU2_COMMIT
    assert "τ³-bench" in spec.note, "换版本这件事要跟着声明走"


@needs_data
def test_the_task_count_is_the_paper_s_278() -> None:
    """★ 50 + 114 + 114 = **278** —— 论文那个数,一次跑齐。"""
    ts = list(tb.Tau2Bench().tasks(split="base", limit=None, seed=0))
    assert len(ts) == tb.TOTAL_TASKS == 278
    from collections import Counter

    assert dict(Counter(t.meta["domain"] for t in ts)) == tb.DOMAINS


@needs_data
def test_a_domain_whose_count_changed_raises(monkeypatch) -> None:
    """★ 条数对不上就炸 —— 那说明**版本没钉住**,而不是「数据更新了」。"""
    monkeypatch.setitem(tb.DOMAINS, "airline", 999)
    with pytest.raises(ValueError, match="版本没钉住"):
        list(tb.Tau2Bench(domain="airline").tasks(split="base", limit=None, seed=0))


# ═══════════════════════════════════════════════════════════
# ★★★ 两方结构 —— 用户那一半不许进 prompt
# ═══════════════════════════════════════════════════════════


@needs_data
def test_the_user_scenario_never_reaches_the_prompt() -> None:
    """★★★ agent 看到的是**业务规则**,用户想干什么在 `meta` 里。

    ★ 泄漏的后果不是报错,是**分数变好** —— 而「为什么这一格特别高」
      在表上没有任何线索。所以这条要机械地钉住,不能靠写代码时记得。
    """
    task = next(iter(tb.Tau2Bench(domain="airline").tasks(split="base", limit=1, seed=0)))

    # prompt 是 policy
    assert task.prompt.lstrip().startswith("#"), "policy.md 是一份 markdown"
    assert "A customer is contacting you" in task.prompt

    # 而用户那一半在 meta 里,内容不许出现在 prompt 中
    scenario = task.meta["user_scenario"]
    assert scenario and scenario.get("instructions")
    instructions = scenario["instructions"]
    for key in ("task_instructions", "reason_for_call"):
        text = str(instructions.get(key) or "")
        if len(text) > 30:
            assert text[:30] not in task.prompt, f"{key} 泄漏进 prompt 了"


@needs_data
def test_the_three_evaluation_fields_all_stay_in_meta() -> None:
    """★ `user_scenario` / `description` / `evaluation_criteria` 三样**都不进 prompt**。

    它们分别是**用户模拟器的输入**、**出题人的备注**、**判分依据** ——
    三样都是 agent 不该看到的（`description` 会直接说出这题在考什么）。
    """
    task = next(iter(tb.Tau2Bench(domain="retail").tasks(split="base", limit=1, seed=0)))
    assert set(task.meta) >= {"user_scenario", "description", "evaluation_criteria", "domain"}
    # description.purpose 是出题人的话,不该出现在 agent 的 prompt 里
    purpose = str((task.meta["description"] or {}).get("purpose") or "")
    if len(purpose) > 30:
        assert purpose[:30] not in task.prompt


# ═══════════════════════════════════════════════════════════
# ★★★ 还没接的三样:抛,不返回 0
# ═══════════════════════════════════════════════════════════


def test_the_unwired_parts_raise_and_name_the_real_reason() -> None:
    """★★★ 工具 / 判分 / 用户模拟器都还没接 —— **抛,不返回 0 分**。

    ★ **一个缺一层看起来像一个算法读数**:报 0 分的 τ²-bench 和
      「所有方法都很差」在表上长得一模一样。

    ★ 而且报错要说清**为什么它不是 loader 的工作量**:
      τ² 是**两方**任务,agent 之外还要一个 LLM 用户模拟器 ——
      前面所有数据集都是一个 agent 对着静态题目。
    """
    bench = tb.Tau2Bench()
    from experiments.core.types import Task

    task = Task(task_id="t", prompt="p", gold="g", oracle_context=None, meta={})
    traj = Trajectory(task_id="t", arm="x", steps=(), final_answer=None,
                      decisions=(), usage=(), escalated=False, error=None)
    for call in (bench.tools, bench.tool_impls, lambda: bench.score(task, traj),
                 lambda: bench.check(task, "x")):
        with pytest.raises(tb.NeedsSimulator) as exc:
            call()
        text = str(exc.value)
        assert "用户模拟器" in text, "要说清缺的是哪一层"
        assert "不是返回 0 分" in text


def test_the_error_also_says_what_IS_done() -> None:
    """★ 报错要说清**哪一半是好的** —— 否则读的人以为整个 loader 没写。

    已经做完的是**任务枚举**（278 条,和论文一致）与下载声明。
    """
    with pytest.raises(tb.NeedsSimulator) as exc:
        tb.Tau2Bench().tools()
    text = str(exc.value)
    assert "任务枚举" in text and "278" in text


# ═══════════════════════════════════════════════════════════
# 下载声明
# ═══════════════════════════════════════════════════════════


def test_the_download_covers_every_domain_file() -> None:
    """★ 三个域的文件都要声明,而且 **telecom 的 policy 不叫 `policy.md`**
    （它叫 `main_policy.md`）,`db` 的后缀也不同（`toml` 不是 `json`）。

    ★ 写死一个名字会让那个域缺文件,而缺文件的报错**不会说「是名字写错了」**。
    """
    spec = tb.Tau2Bench().downloads()[0]
    assert set(tb.DOMAIN_FILES) == {"airline", "retail", "telecom"}
    assert "telecom/main_policy.md" in spec.files
    assert "telecom/db.toml" in spec.files
    assert "airline/db.json" in spec.files
    assert len(spec.files) == 9, "三域 × (tasks + policy + db) —— 声明要齐"
