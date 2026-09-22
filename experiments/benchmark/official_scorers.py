"""**三个官方打分器的移植。** 一处放齐,因为判分口径就是可比性本身。

`core/bench.py` 的第一句:「尽量调**数据集官方的评分器**,不要自己重写。
**重写 = 换判分器,而换判分器之后的准确率不可比。**」

所以这里的每一段都是**逐字抄**的,并且在测试里对着原文做对拍
（`tests/test_official_scorers.py`）—— 和 `rewoo_port.py` 同一条纪律:
**「我照着抄了」这句话没有保障,一条对拍测试才有。**

| 数据集 | 出处 | 文件 |
|---|---|---|
| HotpotQA | `hotpotqa/hotpot` @ `36358534` | `hotpot_evaluate_v1.py` |
| FEVER | `sheffieldnlp/fever-scorer` @ `48016151` | `src/fever/scorer.py` |
| TriviaQA | `mandarjoshi90/triviaqa` @ `ca43b582` | `evaluation/triviaqa_evaluation.py` |

## ★★★ 三家的**主指标**都有一半我们算不了 —— 这件事必须说清

实测（2026-09-22,读原文）:

| 数据集 | 官方**主**指标 | 它还要什么 |
|---|---|---|
| HotpotQA | **`joint_em = em × sp_em`** | 模型要输出 **supporting facts**（`(title, sent_id)` 集合）|
| FEVER | **`strict_score`** | 模型要输出 **`predicted_evidence`**（`(page, line)` 列表,取前 5）|
| TriviaQA | 别名命中（`exact_match` over aliases）| ✅ 只要答案 —— **这个能算全** |

而我们的 `Trajectory` 只装**最终答案** —— 没有证据字段。
所以:

- **`em` / `f1` / 别名命中:能算,而且和官方逐字一致**
- **`sp_em` / `joint_em` / `strict`:算不了** —— 不是「暂时没实现」,
  是**输入里根本没有那个东西**。函数在那儿,但缺字段时它们会
  **明说缺什么**,而不是拿答案 em 冒充 joint。

★ 为什么值得单列一段:HotpotQA 的 leaderboard 报的是 **joint F1**,
  而很多论文正文引的是 **answer-only EM/F1** —— **两个数差很远**。
  我们报 answer-only 时必须写明它是 answer-only,
  否则读表的人会以为那是 joint,而**表上看不出这个区别**。

★ 要算全它们,得先让答案契约里包含证据（例如让模型输出
  `answer` + `supporting_facts` 两段）。**那是一个接口改动,不是打分器的改动** ——
  记在这里,等做到那一步再说,别用近似值糊过去。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

# ═══════════════════════════════════════════════════════════
# 一、HotpotQA —— `hotpot_evaluate_v1.py`，逐字
# ═══════════════════════════════════════════════════════════


def hotpot_normalize_answer(s: str) -> str:
    """`hotpot_evaluate_v1.py::normalize_answer`，逐字。

    ★ 顺序和 ReWOO 那份**一致**（先 lower 再去冠词）——
      所以 `rewoo_port.normalize_answer` 和它是同一个函数。
      这里仍然单独留一份,因为**官方的这一份才是权威**,
      而两者哪天分叉了要能看出来。
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


#: 官方的 `f1_score` 返回**三元组**（f1, precision, recall）——
#: 而 ReWOO 只取了第一个。这里保留官方的形状。
ZERO_METRIC = (0.0, 0.0, 0.0)


def hotpot_f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """`hotpot_evaluate_v1.py::f1_score`，逐字。返回 `(f1, precision, recall)`。"""
    p = hotpot_normalize_answer(prediction)
    g = hotpot_normalize_answer(ground_truth)

    if p in ("yes", "no", "noanswer") and p != g:
        return ZERO_METRIC
    if g in ("yes", "no", "noanswer") and p != g:
        return ZERO_METRIC

    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall), precision, recall


def hotpot_exact_match(prediction: str, ground_truth: str) -> float:
    """`hotpot_evaluate_v1.py::exact_match_score`，逐字。"""
    return float(hotpot_normalize_answer(prediction) == hotpot_normalize_answer(ground_truth))


class MissingEvidence(Exception):
    """**输入里没有这个字段,所以这个指标算不出来。**

    ★ 和「算出来是 0」是两件事。用一个近似值糊过去会让
      **一个算法问题的读数**和**一个接口缺口**在表里长得一样。
    """


@dataclass(frozen=True)
class SpScore:
    """supporting-facts 的四个数。

    ★★ **为什么用具名结构而不是元组:官方返回的是 `(em, prec, recall)` 三个值,
    而我第一版写成了 `(em, f1, prec)` —— 把 `f1` 放在了 `prec` 的位置上。**
    两种都是「三个 float 的元组」,所以类型系统拦不住,而调用方
    `sp_em, sp_prec, sp_recall = ...` 会**静默地把 f1 当成 prec 用下去**,
    一路进到 joint 里,而账面上每一个数都看着正常。

    ⇒ 顺序会错的接口,就不要用顺序表达。
    """

    em: float
    f1: float
    prec: float
    recall: float


def hotpot_update_sp(sp_prediction: Sequence[Sequence[Any]],
                     sp_gold: Sequence[Sequence[Any]]) -> SpScore:
    """`hotpot_evaluate_v1.py::update_sp` 的判据，逐字。

    supporting facts 是 `(title, sent_id)` 的集合:
    `em = 1.0 if fp + fn == 0 else 0.0` —— **要求完全一致,不是 F1**。

    ★ 官方那个函数返回 `(em, prec, recall)`（三个数,`f1` 只累加进 metrics）。
      这里多给一个 `f1`,因为 joint 那一层要用它 —— 但**用具名结构装着**,
      顺序就不会被记错。

    ⚠️ 需要模型输出它选了哪些证据。**我们的答案契约里没有这个字段** ——
       传空列表会得到一个**看起来正常的 0.0**,而那正是要避免的。
    """
    if not sp_prediction:
        raise MissingEvidence(
            "HotpotQA 的 sp_em 需要模型输出 supporting facts（(title, sent_id) 集合），"
            "而当前答案契约里只有最终答案。\n"
            "  → 这不是「没实现」,是**输入里没有那个东西**。"
            "要算它得先让答案契约包含证据（一个接口改动）"
        )
    cur = set(map(tuple, sp_prediction))
    gold = set(map(tuple, sp_gold))
    tp = sum(1 for e in cur if e in gold)
    fp = sum(1 for e in cur if e not in gold)
    fn = sum(1 for e in gold if e not in cur)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    return SpScore(em=1.0 if fp + fn == 0 else 0.0, f1=f1, prec=prec, recall=recall)


def hotpot_joint(em: float, prec: float, recall: float,
                 sp: "SpScore") -> tuple[float, float]:
    """`hotpot_evaluate_v1.py::eval` 里的 joint 那几行，逐字。

    `joint_em = em × sp_em`；`joint_prec = prec × sp_prec` 而
    `joint_recall = recall × sp_recall`。

    ★ **HotpotQA 排行榜报的是 joint** —— 而正文里常引 answer-only。
      两个数差很远,所以报的时候必须写明是哪个。
    """
    joint_prec = prec * sp.prec
    joint_recall = recall * sp.recall
    joint_f1 = (2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                if joint_prec + joint_recall > 0 else 0.0)
    return em * sp.em, joint_f1


# ═══════════════════════════════════════════════════════════
# 二、FEVER —— `src/fever/scorer.py`，逐字
# ═══════════════════════════════════════════════════════════


def fever_is_correct_label(label: str, predicted_label: str) -> bool:
    """`scorer.py::is_correct_label`，逐字（大写比较）。

    ★ 这就是 `acc_score` —— **普通的标签准确率**。
      而 FEVER 的**主**指标是 `strict_score`,它还要证据。
    """
    return str(label).upper() == str(predicted_label).upper()


def fever_is_strictly_correct(label: str, predicted_label: str,
                             evidence: Sequence[Any], predicted_evidence: Sequence[Any],
                             *, max_evidence: int = 5) -> bool:
    """`scorer.py::is_strictly_correct`，逐字。

    判据:标签对 **且** 存在**一整组**金标证据 ⊆ 预测证据的前 `max_evidence` 个。
    ★ **NEI 类免检证据** —— 那一支直接返回 True。

    ⚠️ 同样需要 `predicted_evidence`。缺了就抛 `MissingEvidence`。
    """
    if not predicted_evidence:
        raise MissingEvidence(
            "FEVER 的 strict 需要模型输出 predicted_evidence（(page, line) 列表），"
            "而当前答案契约里只有最终答案。\n"
            "  → 报 `acc` 是可以的,但要写明它是 acc,不是 strict"
        )
    if str(label).upper() == "NOT ENOUGH INFO":
        # ★ NEI 免检证据 —— 这一支不看不等于可以跳过前面那个检查
        return fever_is_correct_label(label, predicted_label)
    if not fever_is_correct_label(label, predicted_label):
        return False
    head = [list(e) for e in predicted_evidence[:max_evidence]]
    for group in evidence:
        actual = [[e[2], e[3]] for e in group]
        if all(sent in head for sent in actual):
            return True
    return False


# ═══════════════════════════════════════════════════════════
# 三、TriviaQA —— `triviaqa_evaluation.py`，逐字
# ═══════════════════════════════════════════════════════════


def triviaqa_normalize_answer(s: str) -> str:
    """`triviaqa_evaluation.py::normalize_answer`，逐字。

    ★ 和 HotpotQA 那一份**逐字相同**（同样先 lower）。
      分开留着是为了**出处清楚** —— 它们将来各自演化时能看出来。
    """
    return hotpot_normalize_answer(s)


def triviaqa_get_ground_truths(answer: dict[str, Any]) -> list[str]:
    """`triviaqa_evaluation.py::get_ground_truths`，逐字:

        answer['NormalizedAliases'] + [normalize_answer(a) for a in answer.get('HumanAnswers', [])]

    ★★ **两个键名都要映射。** 官方吃的是**原始 JSON** 的键
    （`NormalizedAliases` / `HumanAnswers`,首字母大写）,而 HF 版本给的是
    `normalized_aliases`,**并且没有 human answers**。

    ⚠️ 第一版我用了 `aliases`（**未归一化**的那一份）—— 那不是官方的金标集。
      官方的集合是**预先归一化好的**,拿原始别名再归一化一次,
      在带标点或冠词的别名上会得到不同的字符串。
    """
    truths = list(answer.get("NormalizedAliases")
                  or answer.get("normalized_aliases") or [])
    human = answer.get("HumanAnswers") or answer.get("human_answers") or []
    return truths + [triviaqa_normalize_answer(a) for a in human]


def triviaqa_exact_match(prediction: str, ground_truth: str) -> float:
    """`triviaqa_evaluation.py::exact_match_score`，逐字。"""
    return float(triviaqa_normalize_answer(prediction) == triviaqa_normalize_answer(ground_truth))


def triviaqa_metric_max_over_ground_truths(metric_fn, prediction: str,
                                           ground_truths: Sequence[str]) -> float:
    """`triviaqa_evaluation.py::metric_max_over_ground_truths`，逐字。

    ★ **官方口径就是这个 `max`** —— 命中任意一个别名即算对。
      拿它和 ReWOO 那个「只比 `value`」的口径比,是在比判分器。
    """
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def triviaqa_is_exact_match(answer: dict[str, Any], prediction: str) -> bool:
    """`triviaqa_evaluation.py::is_exact_match`，逐字。"""
    return any(triviaqa_exact_match(prediction, gt)
               for gt in triviaqa_get_ground_truths(answer))


__all__ = [
    "MissingEvidence",
    "hotpot_normalize_answer", "hotpot_f1_score", "hotpot_exact_match",
    "hotpot_update_sp", "SpScore", "hotpot_joint",
    "fever_is_correct_label", "fever_is_strictly_correct",
    "triviaqa_normalize_answer", "triviaqa_get_ground_truths",
    "triviaqa_exact_match", "triviaqa_metric_max_over_ground_truths",
    "triviaqa_is_exact_match",
]
