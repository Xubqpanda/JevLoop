"""三个官方打分器的对拍 —— **「我照着抄了」这句话没有保障,一条测试才有。**

★ 参照物是**官方原文**,逐字抄在本文件里（不 import 官方包 —— 那要装一堆依赖,
而且「装上了哪个版本」本身又变成一个变量）。

★★ 本文件最值钱的一条是 `test_the_headline_metrics_refuse_to_run_without_evidence`:
HotpotQA 的排行榜报 **joint**、FEVER 的**主**指标是 **strict**,而两者都要模型输出证据。
我们的答案契约里没有那个字段 —— 所以它们**抛**,而不是拿 answer-only 的数冒充。

**一个接口缺口看起来像一个算法读数,是这类工作里最难发现的一种错。**
"""

from __future__ import annotations

import re
import string
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark import official_scorers as os_  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 参照物：官方原文，逐字抄
# ═══════════════════════════════════════════════════════════


def _ref_normalize(s: str) -> str:
    """`hotpot_evaluate_v1.py::normalize_answer` 与
    `triviaqa_evaluation.py::normalize_answer` —— 两份逐字相同。"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _ref_hotpot_f1(prediction: str, ground_truth: str):
    """`hotpot_evaluate_v1.py::f1_score`，逐字（**返回三元组**）。"""
    p, g = _ref_normalize(prediction), _ref_normalize(ground_truth)
    ZERO = (0, 0, 0)
    if p in ["yes", "no", "noanswer"] and p != g:
        return ZERO
    if g in ["yes", "no", "noanswer"] and p != g:
        return ZERO
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = 1.0 * num_same / len(pt)
    recall = 1.0 * num_same / len(gt)
    return (2 * precision * recall) / (precision + recall), precision, recall


def _ref_hotpot_update_sp(prediction, gold):
    """`hotpot_evaluate_v1.py::update_sp`，逐字。"""
    cur, want = set(map(tuple, prediction)), set(map(tuple, gold))
    tp = sum(1 for e in cur if e in want)
    fp = sum(1 for e in cur if e not in want)
    fn = sum(1 for e in want if e not in cur)
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    # ★ 官方返回的是 **(em, prec, recall) 三个值** —— f1 只累加进 metrics。
    return em, prec, recall


def _ref_fever_is_strictly_correct(label, predicted_label, evidence, predicted_evidence,
                                   max_evidence=5):
    """`src/fever/scorer.py::is_strictly_correct`，逐字（去掉断言）。"""
    correct = str(label).upper() == str(predicted_label).upper()
    if str(label).upper() != "NOT ENOUGH INFO" and correct:
        for group in evidence:
            actual = [[e[2], e[3]] for e in group]
            if all(s in predicted_evidence[:max_evidence] for s in actual):
                return True
        return False
    if str(label).upper() == "NOT ENOUGH INFO" and correct:
        return True
    return False


# ═══════════════════════════════════════════════════════════
# 一、对拍
# ═══════════════════════════════════════════════════════════

NORMALIZE_CASES = ["The Answer is Paris", "a cat", "an Apple!", "", "YES", "  spaced  out  "]
F1_CASES = [("Paris", "Paris"), ("Paris", "London"), ("yes", "yes"), ("yes", "no"),
            ("the cat sat", "cat sat"), ("", "Paris"), ("noanswer", "no"), ("a b c", "a b")]


def test_hotpot_normalize_matches_the_official_one() -> None:
    for s in NORMALIZE_CASES:
        assert os_.hotpot_normalize_answer(s) == _ref_normalize(s), s


def test_hotpot_f1_matches_including_the_three_tuple_shape() -> None:
    """★ 官方的 `f1_score` 返回 **`(f1, precision, recall)`** —— 不是标量。

    ReWOO 只取了第一个（`f1, prec, recall = f1_score(...)` 然后只用 f1）。
    我们保留官方的形状,因为 joint 那一层要用 prec 和 recall。
    """
    for pred, gold in F1_CASES:
        mine = os_.hotpot_f1_score(pred, gold)
        ref = _ref_hotpot_f1(pred, gold)
        assert len(mine) == 3, "官方是三元组"
        assert mine == pytest.approx(ref), (pred, gold)


def test_hotpot_sp_em_matches_and_is_all_or_nothing() -> None:
    """★ `sp_em` 判据是 `fp + fn == 0` —— **集合完全一致,不是 F1**。

    所以漏一个或错一个都是 0。这个「全对才给分」的性质要钉住:
    用 F1 代替它会让这个指标整体偏高。
    """
    gold = [["A", 0], ["B", 1]]
    assert os_.hotpot_update_sp([["A", 0], ["B", 1]], gold).em == 1.0
    assert os_.hotpot_update_sp([["A", 0]], gold).em == 0.0, "漏一个也是 0"
    assert os_.hotpot_update_sp([["A", 0], ["B", 1], ["C", 2]], gold).em == 0.0, "多一个也是 0"

    # ★ 逐**字段**比,不比元组 —— 官方的顺序是 (em, prec, recall),
    #   而我们多给一个 f1。第一版我按位置抄,把 f1 放进了 prec 的位置,
    #   而两种都是「三个 float 的元组」,类型系统拦不住。
    for pred in ([["A", 0], ["B", 1]], [["A", 0]], [["C", 2]]):
        ref_em, ref_prec, ref_recall = _ref_hotpot_update_sp(pred, gold)
        mine = os_.hotpot_update_sp(pred, gold)
        assert (mine.em, mine.prec, mine.recall) == pytest.approx((ref_em, ref_prec, ref_recall))


def test_hotpot_joint_is_a_product_not_an_average() -> None:
    """★★ `joint_em = em × sp_em` —— **乘,不是平均**。

    ★ 这不是细节:平均会让「答对了但没找到证据」拿到 0.5,
      而官方判它 0。HotpotQA 排行榜报的就是这个 joint,
      而正文里常引 answer-only —— **两个数差很远,而表上看不出区别**。
    """
    from experiments.benchmark.official_scorers import SpScore

    zero = SpScore(em=0.0, f1=0.0, prec=0.0, recall=0.0)
    one = SpScore(em=1.0, f1=1.0, prec=1.0, recall=1.0)
    assert os_.hotpot_joint(1.0, 0.8, 0.8, zero)[0] == 0.0, "答对但没证据 = 0"
    assert os_.hotpot_joint(1.0, 1.0, 1.0, one)[0] == 1.0
    # joint_prec = 1×1, joint_recall = 0.5×1 → f1 = 2·1·0.5/(1+0.5) = 0.667
    _, jf1 = os_.hotpot_joint(1.0, 1.0, 0.5, one)
    assert jf1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))


def test_triviaqa_ground_truths_map_both_key_spellings() -> None:
    """★ 官方吃**原始 JSON** 的键（`NormalizedAliases` / `HumanAnswers`）,
    而 HF 给的是 `normalized_aliases` 且**没有** human answers。

    两个键名都要映射 —— 只认一个的话,另一侧的样本会**静默地少掉别名**,
    于是判分偏严,而总分看起来正常。
    """
    assert os_.triviaqa_get_ground_truths({"NormalizedAliases": ["a"], "HumanAnswers": ["B!"]}) \
        == ["a", _ref_normalize("B!")]
    assert os_.triviaqa_get_ground_truths({"normalized_aliases": ["a"]}) == ["a"]
    assert os_.triviaqa_get_ground_truths({}) == []


def test_triviaqa_official_metric_is_max_over_aliases() -> None:
    """★ 官方口径是 `max` over 金标集 —— **命中任意别名即算对**。

    不是 exact match on `value`（那是 ReWOO 的口径）。
    """
    answer = {"NormalizedAliases": ["david seville", "dave seville"]}
    assert os_.triviaqa_is_exact_match(answer, "Dave Seville") is True
    assert os_.triviaqa_is_exact_match(answer, "Ross Bagdasarian") is False
    assert os_.triviaqa_metric_max_over_ground_truths(
        os_.triviaqa_exact_match, "nope", []) == 0.0


def test_fever_strict_matches_the_official_one() -> None:
    """★ `strict` = 标签对 **且** 存在一整组金标证据 ⊆ 预测证据的前 5 个;NEI 免检。"""
    ev = [[["page", 0, "Page", 3], ["page", 0, "Page", 4]]]
    good = [["Page", 3], ["Page", 4]]
    half = [["Page", 3]]

    for label, pred_label, pred_ev, want in [
        ("SUPPORTS", "SUPPORTS", good, True),
        ("SUPPORTS", "SUPPORTS", half, False),          # 证据不全
        ("SUPPORTS", "REFUTES", good, False),           # 标签错
        ("NOT ENOUGH INFO", "NOT ENOUGH INFO", [["x", 1]], True),   # NEI 免检
    ]:
        mine = os_.fever_is_strictly_correct(label, pred_label, ev, pred_ev)
        ref = _ref_fever_is_strictly_correct(label, pred_label, ev, pred_ev)
        assert mine is ref is want, (label, pred_label, pred_ev)


def test_fever_max_evidence_truncates_at_five() -> None:
    """★ 官方默认 `max_evidence=5` —— 第 6 个证据**不算数**。

    这个截断在报告里几乎从不出现,但它会**静默地把长证据的回答判错**。
    """
    ev = [[["page", 0, "Page", 9]]]
    six = [["a", 1]] * 5 + [["Page", 9]]
    assert os_.fever_is_strictly_correct("SUPPORTS", "SUPPORTS", ev, six) is False
    assert os_.fever_is_strictly_correct("SUPPORTS", "SUPPORTS", ev, six,
                                         max_evidence=6) is True


# ═══════════════════════════════════════════════════════════
# ★★★ 二、主指标缺输入时**抛**,不冒充
# ═══════════════════════════════════════════════════════════


def test_the_headline_metrics_refuse_to_run_without_evidence() -> None:
    """★★★ HotpotQA 的 **joint** 和 FEVER 的 **strict** 都要模型输出证据,
    而我们的答案契约里只有最终答案。

    ⇒ 它们**抛 `MissingEvidence`**,而不是:
      - 拿 answer-only 的 em 冒充 joint（**表上看不出区别**）
      - 或者返回 0（**一个接口缺口看起来像一个算法读数**）

    ★ 要算它们得让答案契约包含证据 —— **那是接口改动,不是打分器的改动**。
    """
    with pytest.raises(os_.MissingEvidence, match="supporting facts"):
        os_.hotpot_update_sp([], [["A", 0]])
    with pytest.raises(os_.MissingEvidence, match="predicted_evidence"):
        os_.fever_is_strictly_correct("SUPPORTS", "SUPPORTS", [[]], [])


def test_the_error_says_what_is_missing_and_what_to_do() -> None:
    """★ 报错要说清**缺什么**和**该怎么办** —— 不然下一个人只会看到「不支持」。"""
    with pytest.raises(os_.MissingEvidence) as exc:
        os_.hotpot_update_sp([], [])
    text = str(exc.value)
    assert "supporting facts" in text and "接口改动" in text


def test_the_loaders_expose_the_gap_as_a_variant_that_raises() -> None:
    """★ 缺口要**在接口上看得见**:`score_variants()` 里有 `joint` / `strict` 这两个名字,
    调它们才抛。**藏起来的话,读表的人不知道官方头条是另一个数。**
    """
    from experiments.benchmark.fever import Fever
    from experiments.benchmark.hotpotqa import HotpotQa

    assert "joint" in HotpotQa().score_variants()
    assert "strict" in Fever().score_variants()

    from experiments.core.types import Task, Trajectory

    task = Task(task_id="t", prompt="p", gold="yes", oracle_context=None, meta={})
    traj = Trajectory(task_id="t", arm="x", steps=(), final_answer="yes",
                      decisions=(), usage=(), escalated=False, error=None)
    with pytest.raises(os_.MissingEvidence):
        HotpotQa().score_variants()["joint"](task, traj)
    with pytest.raises(os_.MissingEvidence):
        Fever().score_variants()["strict"](task, traj)


def test_answer_only_em_is_clearly_labelled_as_answer_only() -> None:
    """★★ `em` 的 `detail` 里**必须**写着它是 answer-only。

    排行榜报 joint、正文常引 answer-only,两者差很远 ——
    所以这两个字要跟着**每一个**读数走,不是只写在文档里。
    """
    from experiments.benchmark.hotpotqa import HotpotQa
    from experiments.core.types import Task, Trajectory

    task = Task(task_id="t", prompt="p", gold="Paris", oracle_context=None, meta={})
    traj = Trajectory(task_id="t", arm="x", steps=(), final_answer="Paris",
                      decisions=(), usage=(), escalated=False, error=None)
    j = HotpotQa().score_variants()["em"](task, traj)
    assert "answer-only" in j.detail, j.detail
    assert "非 joint" in j.detail


# ═══════════════════════════════════════════════════════════
# ★★★ 伪步骤不是函数调用
# ═══════════════════════════════════════════════════════════


def test_a_parse_error_step_is_not_a_tool_call() -> None:
    """★★★ 实测（2026-09-22,`bfcl-v3-multiple × react-typed`）:

    `run_loop` 解析不出动作时会往轨迹里塞一步 `action_tool("__parse_error__", {})`
    （好让下一轮 prompt 带着纠正提示）。它的 `kind` 是 `"tool"` ——
    因为 `Action.kind` **只有三种,不许加第四种**。

    ⇒ 任何「数一下调了几次工具」的地方都会把它算进去。BFCL 的判分器就是:

        called = [... if s.action.kind == "tool"]

    于是一次「**调对了 + 解析失败过一次**」被判成 `wrong_tool`。

    ★ 而它**不产生 `ToolCallEvent`**（伪步骤不过 executor）——
      所以**工具事件里看不见它,只在 `steps` 里**。这就是它藏了这么久的原因。
    """
    from experiments.core.agent import action_answer, action_ask, action_tool
    from experiments.core.types import is_tool_call

    assert is_tool_call(action_tool("lookup_capital", {"country": "Peru"})) is True
    assert is_tool_call(action_tool("__parse_error__", {})) is False
    assert is_tool_call(action_tool("__ask__", {})) is False, "弃答也不是调用"
    assert is_tool_call(action_answer("42")) is False
    assert is_tool_call(action_ask("不想猜")) is False


def test_bfcl_scorer_ignores_the_parse_error_step() -> None:
    """★★ 上一条的**后果版**:一次正确调用 + 一步解析失败,**必须判对**。

    ★ BFCL 的判据是 `sorted(called) == sorted(gold)`。多了个 `__parse_error__`
      就变成两个元素比一个 → 判错。而**模型那一步是调对了的**。
    """
    from experiments.benchmark.bfcl import Bfcl
    from experiments.core.agent import action_tool
    from experiments.core.types import Action, Step, Task, Trajectory

    bench = Bfcl(subset="v3-simple")
    task = Task(task_id="t", prompt="p", gold=["sculpture_price.calculate"],
                oracle_context=None, meta={})
    traj = Trajectory(
        task_id="t", arm="x", final_answer=None, decisions=(), usage=(),
        escalated=False, error=None,
        steps=(
            Step(index=0, action=Action(kind="tool", name="sculpture_price.calculate",
                                        arguments={}), observation="recorded"),
            # ★ 就是这一步曾经把它判错
            Step(index=1, action=action_tool("__parse_error__", {}),
                 observation="Could not parse an action."),
        ),
    )
    j = bench.score(task, traj)
    assert j.correct is True, f"调对了却被判错:{j.detail}"
    assert j.failure_class is None


def test_both_loaders_share_the_same_rule() -> None:
    """★★ **同一条规则不许在两处各写一遍。**

    我一开始在 `alfworld` 里手写了 `and not name.startswith("__")`,
    而 **BFCL 里没写** —— 于是同一个 bug 只在一个 loader 里被挡住。
    所以判据现在住在 `core/types.py`,两个 loader 都调它。
    """
    import inspect

    from experiments.benchmark.alfworld import alfworld as aw
    from experiments.benchmark.bfcl import bfcl as bf

    for mod, cls in ((bf, "Bfcl"), (aw, "AlfWorld")):
        src = inspect.getsource(getattr(mod, cls).score)
        # ★ 去掉注释行再查 —— 否则**解释这条规则的注释本身**会被当成违规
        #   （第一版就是这样误报的:我在注释里写了 `kind == "tool"` 这几个字）。
        code = "\n".join(line for line in src.splitlines()
                         if not line.strip().startswith("#"))
        assert 'kind == "tool"' not in code, f"{mod.__name__}.{cls}.score 自己判了 kind"
        assert "is_tool_call" in code, f"{mod.__name__}.{cls}.score 没用共用的判据"
