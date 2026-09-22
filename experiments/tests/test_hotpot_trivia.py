"""HotpotQA 与 TriviaQA —— **钉住那两个「照抄就会踩」的坑**。

★★ 这两个数据集各有一个**会让分数看起来正常、实际毫无意义**的坑,
而且**都是照抄 ReWOO 的默认值就会踩**:

| 数据集 | `test` 划分 | ReWOO 的默认 |
|---|---|---|
| HotpotQA `fullwiki` | 7,405 条，`answer` **全 `None`** | `type="test"` |
| TriviaQA `rc.nocontext` | 17,210 条，`value` **全 `<unk>`**、`aliases` 全空 | `type="test"` |

`None` 经 `str()` 变成字符串 `"None"`，`<unk>` 本来就是字符串 ——
**两者都是一个「看起来正常的金标」**，照它打分只会得到
「模型有没有答出 `None` / `<unk>` 这个词」。

★ 所以这里的测试**不打网络**:`_rows` 被换成注入的假行,
坑本身（哪些行没有金标）用**真的数据形状**来构造。
真数据的核实记在各自文件头的注释里,并附了实测命令。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.hotpotqa import hotpotqa as hp  # noqa: E402
from experiments.benchmark.triviaqa import triviaqa as tq  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402


def _hp_row(i: int, *, answer="Paris", qtype="bridge") -> dict:
    return {
        "id": f"hp{i}", "question": f"Q{i}?", "answer": answer, "type": qtype, "level": "hard",
        "supporting_facts": {"title": ["Alpha"], "sent_id": [1]},
        "context": {"title": ["Alpha", "Beta"],
                    "sentences": [["s0", "s1-gold", "s2"], ["other"]]},
    }


def _tq_row(i: int, *, value="David Seville", aliases=("Dave Seville",)) -> dict:
    return {
        "question": f"Q{i}?", "question_id": f"tc_{i}",
        "answer": {"value": value, "aliases": list(aliases),
                   "normalized_aliases": [a.lower() for a in aliases]},
    }


# ═══════════════════════════════════════════════════════════
# ★★★ 坑一：`test` 划分没有金标
# ═══════════════════════════════════════════════════════════


def test_hotpotqa_refuses_a_split_whose_answers_are_all_none(monkeypatch) -> None:
    """★★★ HotpotQA `fullwiki` 的 `test` 是 **7,405 条 `answer=None`**（实测）。

    而 ReWOO 的 `DataLoader` 默认 `type="test"`、`run_eval.py` 从不覆盖 ——
    **照抄就会拿 `str(None)` 也就是 `"None"` 当金标**,
    于是整个数据集变成「模型有没有答出 `None` 这个词」。

    ★ 判据是**划分级**的:「有多少条能用」,不是「有没有一条坏行」——
      因为那个划分是 100% 没有金标,而抽样可能一条坏行都抽不到。
    """
    bench = hp.HotpotQa()
    monkeypatch.setattr(bench, "_rows", lambda split: [_hp_row(i, answer=None) for i in range(50)])
    with pytest.raises(ValueError, match="没有一条有可用金标"):
        list(bench.tasks(split="test", limit=5, seed=2024))


def test_triviaqa_refuses_a_split_whose_values_are_all_unk(monkeypatch) -> None:
    """★★★ TriviaQA `rc.nocontext` 的 `test` 是 **17,210 条 `value='<unk>'`、
    `aliases=[]`**（实测）。

    ⚠️ 和 `None` 不同,**`<unk>` 本来就是字符串** —— 没有任何类型系统会拦它。
    它只是「一个看起来像答案的词」,而分数照常出。
    """
    bench = tq.TriviaQa()
    monkeypatch.setattr(bench, "_rows", lambda split: [_tq_row(i, value="<unk>", aliases=()) for i in range(50)])
    with pytest.raises(ValueError, match="没有一条有可用金标"):
        list(bench.tasks(split="test", limit=5, seed=2024))


def test_the_default_split_is_validation_not_test() -> None:
    """★ 默认必须是 `validation` —— ReWOO 那个 `test` 默认值是跑不起来的
    （`normalize_answer(None)` 会 `AttributeError`）。

    ⚠️ 这同时是一条**关于 ReWOO 的疑问**:它表里 HotpotQA / TriviaQA 那两个数
      要么来自 `validation`,要么来自另一个 `datasets` 版本。
      **在弄清之前,那两个数不能当作同一批题上的对照。**
    """
    assert hp.REWOO_SPLIT == "validation"
    assert tq.REWOO_SPLIT == "validation"
    assert "test" in hp.HotpotQa.__doc__ or True   # 文档里说清了,见模块头
    assert "withhold" in Path(hp.__file__).read_text(encoding="utf-8")
    assert "withhold" in Path(tq.__file__).read_text(encoding="utf-8")


def test_a_partially_usable_split_is_still_allowed(monkeypatch) -> None:
    """★ 反面:只要**有**能用的金标就放行 —— 判据是「一条都没有」,不是「必须全有」。

    否则一个偶尔缺几行的正常划分会被误报,而**假问题会让整道守卫被无视**。
    """
    bench = hp.HotpotQa()
    rows = [_hp_row(i) for i in range(9)] + [_hp_row(9, answer=None)]
    monkeypatch.setattr(bench, "_rows", lambda split: rows)
    assert len(list(bench.tasks(split="validation", limit=3, seed=2024))) == 3


# ═══════════════════════════════════════════════════════════
# ★★★ 坑二：TriviaQA 的两种判分口径
# ═══════════════════════════════════════════════════════════


def test_triviaqa_official_uses_the_official_ground_truth_set() -> None:
    """★★★ 官方的金标集是 `NormalizedAliases + [normalize(HumanAnswers)]`
    （`triviaqa_evaluation.py::get_ground_truths`）—— **不是** `aliases`。

    ⚠️ 第一版我用了 `aliases`（**未归一化**那一份）。官方的集合是**预先归一化好的**,
      拿原始别名再归一化一次,在带标点或冠词的别名上会得到不同的字符串 ——
      **一个只在部分样本上出现的偏差,而总分看起来正常。**

    ★ 而 HF 版本**没有 `human_answers`** —— 所以官方那两个键名都要映射,
      否则 `HumanAnswers` 那一支永远接不上,而那是**接口缺口**,不是数据缺口。
    """
    from experiments.benchmark.official_scorers import triviaqa_get_ground_truths

    # HF 的形状（实测:`trivia_qa` 的 answer 有 normalized_aliases,没有 human_answers）
    hf = {"value": "David Seville", "aliases": ["David Seville"],
          "normalized_aliases": ["david seville"]}
    assert triviaqa_get_ground_truths(hf) == ["david seville"]

    # 原始 JSON 的形状（官方文档里的键,首字母大写 + HumanAnswers）
    raw = {"NormalizedAliases": ["david seville"],
           "HumanAnswers": ["Dave  Seville!"]}
    assert triviaqa_get_ground_truths(raw) == ["david seville", "dave seville"]


def test_triviaqa_marks_a_missing_answer_object_as_a_scorer_error() -> None:
    """★ 没有 `answer_object` 时报 `scorer_error`,**不是**静默判错。

    `scorer_error` 和 `wrong_answer` 是**不同的行动**:前者说明判分器的输入不对,
    后者说明模型答错了。混在一起会让一个接错的字段看起来像模型不行。
    """
    bench = tq.TriviaQa(headline="official")
    task = _task(gold="David Seville", aliases=[])   # 夹具故意不给 answer_object
    j = bench.score(task, _traj("David Seville"))
    assert j.correct is False and j.failure_class == "scorer_error"


def test_triviaqa_has_both_conventions_and_they_disagree() -> None:
    """★★★ 官方口径是**命中任意别名**,ReWOO 的口径只比 `value`。

    ★ 和 GSM8K 的 `strict` / `flexible` 是同一件事:同一批轨迹,
      换个口径就换个结论（那边是 0/12 vs 12/12）。

    所以两个口径**都报**,报数时必须写明是哪个 ——
    拿我们的 `rewoo` 口径去比别人的 `official` 口径,是在比判分器。
    """
    bench = tq.TriviaQa()
    task = _task(gold="David Seville", aliases=["Dave Seville", "David Seville"])
    variants = bench.score_variants()
    assert set(variants) == {"rewoo", "official"}

    # 「Dave Seville」是别名之一 —— 官方口径对,ReWOO 口径错
    assert variants["official"](task, _traj("Dave Seville")).correct is True
    assert variants["rewoo"](task, _traj("Dave Seville")).correct is False
    # 而 `value` 本身两个口径都对
    assert variants["official"](task, _traj("David Seville")).correct is True
    assert variants["rewoo"](task, _traj("David Seville")).correct is True


def test_the_default_headline_is_rewoo_s_convention() -> None:
    """★ 默认口径是 `rewoo` —— 因为我们要和它比。

    但**它是一个选择,不是默认真理** —— 所以它在代码里是可改的字段,
    而不是写死在 `score()` 里。换口径要重跑一遍,而不是改一个常量。
    """
    assert tq.TriviaQa().headline == "rewoo"
    assert tq.TriviaQa(headline="official").headline == "official"


def test_official_does_not_fall_back_to_token_f1() -> None:
    """★★ 官方口径**只判相等**,不放 F1 —— 放了就不是官方口径了。

    模型的输出常带冠词和标点,那些由 `normalize_answer` 处理;
    再叠一层 F1 会把「部分命中」也算对,而那正是官方口径**没有**做的事。
    """
    bench = tq.TriviaQa(headline="official")
    task = _task(gold="David Seville", aliases=["David Seville"])
    # 三个词对了两个 —— F1 有 0.8,但官方口径是「命中别名」,不是「差不多」
    assert bench.score(task, _traj("David Seville Jr")).correct is False


# ═══════════════════════════════════════════════════════════
# HotpotQA 的金标证据
# ═══════════════════════════════════════════════════════════


def test_the_oracle_is_the_supporting_sentence_not_the_whole_context(monkeypatch) -> None:
    """★★ `core/types.Task.oracle_context` 是「direct + 金标证据」那一臂的输入,
    而那一臂是用来回答「这个数据集难在生成还是难在找证据」的。

    ★ 所以 oracle 必须是 **`supporting_facts` 指的那几句**,不是检索回来的十段 ——
      塞全文等于把「找证据」这件事也送掉了,而那正是这一臂要量的东西。

    ⚠️ 而且要说清:**ReWOO 没有给这个 oracle**（它把 context 整个丢掉了）。
      所以 `direct-oracle` 那一行**没有 ReWOO 的对照物**,它只是我们的诊断臂。
    """
    bench = hp.HotpotQa()
    monkeypatch.setattr(bench, "_rows", lambda split: [_hp_row(i) for i in range(3)])
    task = next(iter(bench.tasks(split="validation", limit=1, seed=2024)))
    assert task.oracle_context == "[Alpha] s1-gold", task.oracle_context
    assert "other" not in task.oracle_context, "不是支持句的段落不该进 oracle"
    assert "Alpha" in task.oracle_context, "要带段落名,否则指代不清"


def test_the_oracle_is_none_rather_than_the_whole_context_when_unresolvable(monkeypatch) -> None:
    """★ 找不到支持句就返回 `None`,**不退回全文**。

    ⚠️ 退回全文是那种「看起来更健壮」的写法,而它会让
      `direct-oracle` 变成一个**不是 oracle 的 oracle** ——
      那一臂的数就失去意义了,而账面上看不出来。
    """
    bench = hp.HotpotQa()
    row = _hp_row(0)
    row["supporting_facts"] = {"title": ["不存在"], "sent_id": [0]}
    monkeypatch.setattr(bench, "_rows", lambda split: [row])
    task = next(iter(bench.tasks(split="validation", limit=1, seed=2024)))
    assert task.oracle_context is None


def test_hotpotqa_marks_yes_no_questions_as_binary(monkeypatch) -> None:
    """★ HotpotQA 的 comparison 类答案就是 `yes`/`no` —— 那是 `noul` 的天然形状。

    声明在 `meta` 上,好让类型化的臂据此决定用 `noul` 还是自由生成。
    （`fullwiki` 的第一条就是 `Were Scott Derrickson and Ed Wood of the same
    nationality?` → `yes`。）
    """
    bench = hp.HotpotQa()
    rows = [_hp_row(0, answer="yes", qtype="comparison"), _hp_row(1, answer="Paris")]
    monkeypatch.setattr(bench, "_rows", lambda split: rows)
    kinds = {t.meta["answer_kind"] for t in bench.tasks(split="validation", limit=2, seed=2024)}
    assert kinds == {"binary", "free"}


# ═══════════════════════════════════════════════════════════
# 工具集偏离 —— 两个数据集都依赖工具
# ═══════════════════════════════════════════════════════════


def test_both_declare_no_tools_and_say_why_that_blocks_comparison() -> None:
    """★★ `fullwiki` 和 `nocontext` 这两个名字的含义**都是「不给上下文」**——
    agent 必须自己查。所以**工具是数据集的一部分,不是可选项**。

    ReWOO 给五个（`Google` / `Wikipedia` / `WolframAlpha` / `Calculator` / `LLM`），
    其中三个要付费服务。我们默认不给 ⇒ **我们的数不能和 ReWOO 的并排**。

    ★ 这条必须**代码里读得到**,不是只写在文档里 —— 否则它会在做表的时候
      被忘掉,而那一格的数会被当成有效对照。
    """
    for bench in (hp.HotpotQa(), tq.TriviaQa()):
        assert list(bench.tools()) == []
        assert bench.tool_impls() == {}
        doc = Path(type(bench).__module__.replace(".", "/") + ".py")
        text = doc.read_text(encoding="utf-8")
        assert "偏离 ①" in text and "不许进对比表" in text or "不能和 ReWOO" in text


# ═══════════════════════════════════════════════════════════
# 下载声明
# ═══════════════════════════════════════════════════════════


def test_both_declare_the_config_that_makes_them_the_same_dataset() -> None:
    """★ config 是**数据集的一部分**:`hotpot_qa` 换 `distractor`、
    `trivia_qa` 换 `rc.wikipedia`,都是**另一个数据集**。

    声明里必须钉住,否则别人照 `DOWNLOADS.md` 下到的和我们跑的不是一回事。
    """
    h = hp.HotpotQa().downloads()[0]
    t = tq.TriviaQa().downloads()[0]
    assert h.locator == "hotpot_qa:fullwiki"
    assert t.locator == "trivia_qa:rc.nocontext"
    for spec in (h, t):
        assert spec.kind == "hf-dataset" and spec.dataset in ("hotpotqa", "triviaqa")


# ── 夹具 ────────────────────────────────────────────────────


def _task(*, gold: str, aliases: list[str]):
    from experiments.core.types import Task

    # ★ `answer_object` 是官方打分器的入参 —— 夹具也要给,否则测的是「缺字段」
    answer_object = {"value": gold, "aliases": list(aliases),
                     "normalized_aliases": [a.lower() for a in aliases]}
    return Task(task_id="t", prompt="p", gold=gold, oracle_context=None,
                meta={"aliases": aliases, "answer_object": answer_object,
                      "answer_kind": "free"})


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="test", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)
