"""FEVER 与 SOTU_QA —— 一个没有坑,一个有**判分器本身**的坑。

★ 这个文件最值钱的一条是 `test_judge_reports_scorer_error_instead_of_passing`:
SOTU_QA 的 74 条里有 **40 条（54%）既没有判定词也没有数字**,
而我第一版的规则判官对那种金标**返回「判对」** ——
于是过半的行上**任何回答都是正确的**,而账面上看不出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.fever import fever as fv  # noqa: E402
from experiments.benchmark.sotuqa import sotuqa as sq  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="test", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)


def _task(gold: str, **meta):
    from experiments.core.types import Task

    return Task(task_id="t", prompt="p", gold=gold, oracle_context=None, meta=dict(meta))


# ═══════════════════════════════════════════════════════════
# FEVER —— 三选一,没有 HotpotQA 那个坑
# ═══════════════════════════════════════════════════════════


def test_fever_declares_three_way_options() -> None:
    """★ FEVER 是**三选一**,不是「真/假」—— 别把它和 `noul` 混了。

    ⚠️ 拆成两个 `noul`（「可验证吗」+「支持还是反驳」）是一个**变体**,
      它值不值得做要量了才知道 —— **别默认它更好**。
    """
    assert fv.LABELS == ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def test_fever_label_is_the_official_metric() -> None:
    """★ 官方口径就是 label accuracy。默认必须是它。"""
    bench = fv.Fever()
    task = _task("REFUTES", options=list(fv.LABELS))
    assert bench.score(task, _traj("REFUTES")).correct is True
    assert bench.score(task, _traj("SUPPORTS")).correct is False
    assert bench.headline == "label"


def test_fever_label_matches_the_official_scorer_including_its_strictness() -> None:
    """★★ **官方判据只做 `.upper()`,不去标点** —— 比手写的版本严。

    `scorer.py::is_correct_label` 的原文就是
    `instance["label"].upper() == instance["predicted_label"].upper()`。

    ⇒ 模型答 `NOT ENOUGH INFO.`（**多一个句号**）**判错**。

    ★ 这不是我实现得糙,是**官方就是这样**。我第一版按「归一化后比较」写,
      那会给出一个**比官方高**的数 —— 而把它和文献里的数并排就是在比判分器。
      **宁可跟着官方严,也不要自己「修好」它** —— 要修就得两边一起修。
    """
    bench = fv.Fever()
    task = _task("NOT ENOUGH INFO", options=list(fv.LABELS))
    assert bench.score(task, _traj("not enough info")).correct is True, "大小写不该算错"
    assert bench.score(task, _traj("NOT ENOUGH INFO.")).correct is False, \
        "官方不去标点 —— 多一个句号就是判错（这是它的性质,不是 bug）"


def test_fever_contains_is_loose_and_flags_two_sided_answers() -> None:
    """★★ `contains` 是**宽松口径**,它会把**两头堵**的回答判成对 —— 必须在说明里点名。

    「SUPPORTS 和 REFUTES 都有可能」这种回答在 `contains` 下算命中,
    在 `label` 下算错。**两个都报,但表里必须写明是哪个。**
    """
    bench = fv.Fever(headline="contains")
    task = _task("SUPPORTS", options=list(fv.LABELS))
    two_sided = "It could be SUPPORTS or REFUTES depending on the source."
    assert bench.score(task, _traj(two_sided)).correct is True
    assert "两头堵" in bench.score(task, _traj(two_sided)).detail
    # 同一条回答在严格口径下是错的
    assert fv.Fever(headline="label").score(task, _traj(two_sided)).correct is False


def test_fever_refuses_a_split_with_no_labels(monkeypatch) -> None:
    """★ 和 HotpotQA / TriviaQA 同一道守卫。

    实测 FEVER 的三个划分**都带标签**,所以它不会触发 —— 留着是为了
    **划分被换掉**时能挡住,而不是因为它现在有问题。
    """
    bench = fv.Fever()
    monkeypatch.setattr(bench, "_rows",
                        lambda split: [{"id": f"x{i}", "claim": "c", "label": None} for i in range(20)])
    with pytest.raises(ValueError, match="没有一条带标签"):
        list(bench.tasks(split="test", limit=3, seed=2024))


def test_fever_prompt_lists_the_labels_verbatim() -> None:
    """★ 契约里必须**逐字**列出三个标签。

    `NOT ENOUGH INFO` 是带空格的字符串 —— 不写清楚的话模型会答 `NEI`
    或 `Not enough information`,而判分会判错,且错得看起来像模型不行。
    """
    row = {"claim": "Kareena Kapoor was a commercial failure."}
    prompt = fv.Fever()._prompt(row)
    for label in fv.LABELS:
        assert label in prompt
    assert row["claim"] in prompt


# ═══════════════════════════════════════════════════════════
# ★★★ SOTU_QA —— 判分器自己的坑
# ═══════════════════════════════════════════════════════════


def test_judge_reports_scorer_error_instead_of_passing() -> None:
    """★★★ **金标里没有可判的要素时,不许判对。**

    实测:74 条里 **40 条（54%）** 既没有判定词也没有数字
    （金标是 `The new leader of the House Democrats is Hakeem Jeffries. ...`
    这种整句话）。我第一版的 `judge` 对那种金标
    `verdict=None` + `numbers=∅` → **两边都「没问题」→ 判对**。

    ⇒ 过半的行上**任何回答都是正确的**,包括「我不知道」。
    而账面上只有一行 `correct=True`,看不出判分器根本没工作。

    ★ 修法是返回 `scorer_error` —— 它和 `wrong_answer` 是**不同的行动**:
      前者说明**判分器**在这条上失效了,后者说明模型答错了。
    """
    bench = sq.SotuQa(headline="judge")
    task = _task("The new leader of the House Democrats is Hakeem Jeffries.")
    j = bench.score(task, _traj("I have no idea"))
    assert j.correct is False, "不可判的金标绝不能判对"
    assert j.failure_class == "scorer_error"


def test_recall_is_the_default_and_works_on_every_row() -> None:
    """★★ 默认换成 `recall` —— 它对**全部 74 条**都算得出,而且是确定性的。

    `em` 在这一列必然接近 0（金标是一整句话）,`judge` 只在 34/74 条上有效。
    `recall` 是唯一一个既覆盖全部行、又不调 LLM 的口径。
    """
    bench = sq.SotuQa()
    assert bench.headline == "recall"
    task = _task("No, Kevin is 58 while Nancy is 83")
    assert bench.score(task, _traj("No, Kevin is 58 while Nancy is 83")).score == 1.0
    assert bench.score(task, _traj("Yes, Kevin is 40")).correct is False


def test_em_is_kept_only_as_a_contrast() -> None:
    """★ `em` / `f1` 留着是**对照**,不是成绩 —— 它们证明「为什么不能用它们」。

    金标 `No, Kevin is 58 while Nancy is 83` 不可能被逐字复现,
    所以 `em` 恒为 0。**报它等于报了一个所有方法都是 0 的列。**
    """
    bench = sq.SotuQa()
    task = _task("No, Kevin is 58 while Nancy is 83")
    assert set(bench.score_variants()) == {"recall", "judge", "em", "f1"}
    # 答对意思但换了措辞 —— em 判错,recall 判对
    paraphrase = "No. Kevin McCarthy is 58 and Nancy Pelosi is 83"
    assert bench.score_variants()["em"](task, _traj(paraphrase)).correct is False
    assert bench.score_variants()["recall"](task, _traj(paraphrase)).correct is True


def test_the_small_n_warning_travels_with_each_task() -> None:
    """★★ **74 条这个事实要跟着结果走。**

    n=74 时准确率的 95% 置信区间宽到 ±11 个百分点 —— 比我们想量的差距还大。
    所以这个数据集的价值在**机制归因**,不在「谁比谁高两个点」。

    ★ 写进 `meta` 而不是只写在文档里:读表的人不一定读得到那份文档,
      但他一定会读到结果行。
    """
    task = next(iter(sq.SotuQa().tasks(split="test", limit=1, seed=2024)))
    assert task.meta["n_total"] == 74
    assert task.meta["small_n_warning"] is True


def test_both_declare_where_their_data_comes_from() -> None:
    """★ 下载声明:FEVER 走 HF,`sotuqa` 走 ReWOO 的 tarball（和三个 BigBench 同源）。"""
    f = fv.Fever().downloads()[0]
    s = sq.SotuQa().downloads()[0]
    assert f.kind == "hf-dataset" and f.dataset == "fever"
    assert s.kind == "http" and s.pinned
    assert s.files == ("SOTU_QA.csv",)
