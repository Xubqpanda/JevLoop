"""三个 BigBench 任务 —— **改写自 ReWOO,所以测试要钉的是「改写得对不对」**。

★ 这个文件里最值钱的两条是:

1. `test_the_draw_matches_rewoo_...` —— 「我们跑的是哪 100 条」必须和 ReWOO 一样,
   否则两个数不可比,而**不可比在数字上看不出来**。
2. `test_our_scoring_matches_rewoo_verbatim` —— 判分函数逐字对齐。
   这里把 ReWOO 那两个函数**原文抄进测试**当参照,而不是靠人工比对。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.bigbench import bigbench as bb  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402


# ═══════════════════════════════════════════════════════════
# ★★★ 抽样：必须和 ReWOO 那一批逐位一致
# ═══════════════════════════════════════════════════════════


def test_the_draw_matches_pandas_sample_bit_for_bit() -> None:
    """★★★ ReWOO 用 `df.sample(k, random_state=seed)`，我们**不依赖 pandas** 复现它。

    `RandomState(seed).permutation(n)[:k]` 与 pandas 的 `sample` 逐位相同
    （pandas 内部就是 `choice(replace=False)`）。这条测试钉住那个等价关系 ——
    一旦不等价,「我们跑的是哪 100 条」就悄悄换了,而**分数还是照常出**。

    ⚠️ 这条测试用 pandas 当参照,所以**只在装了 pandas 时跑**。
       不装也应当绿（跳過),因为生产路径本来就不需要它。
    """
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": range(2290)})
    for seed, k in [(2024, 10), (2024, 100), (2023, 300), (0, 1)]:
        want = list(df.sample(k, random_state=seed).index)
        got = bb.rewoo_draw(2290, k, seed)
        assert got == want, f"seed={seed} k={k} 和 pandas 不一致"


def test_the_first_ten_of_rewoo_s_strategyqa_draw_are_pinned() -> None:
    """★★ **把那 10 个下标写死在测试里。**

    上一条测的是「我们的算法和 pandas 等价」,这一条测的是
    「**那一批具体是哪几条**」—— 它是和 ReWOO 数字可比的那个锚点,
    也是别人复现我们时的第一批题。

    `seed=2024` 是 `run_eval.py` 的 `--seed` 默认值（`DataLoader` 自己的默认
    2023 不作数 —— 它被 `run_eval` 覆盖了）。
    """
    assert bb.REWOO_SEED == 2024
    assert bb.rewoo_draw(2290, 10, 2024) == [994, 2185, 1311, 1000, 2005,
                                             66, 1035, 2010, 650, 1513]


def test_a_limit_larger_than_the_data_returns_everything_in_order() -> None:
    """★ 边界:要的比有的多就全给,而且**保持原顺序** —— 不要静默少给。"""
    assert bb.rewoo_draw(5, 100, 2024) == [0, 1, 2, 3, 4]
    assert bb.rewoo_draw(5, None, 2024) == [0, 1, 2, 3, 4]


def test_the_draw_is_not_the_rotation_gsm8k_uses() -> None:
    """★★ **两个 loader 的抽样方式不一样,这是有意的。**

    `gsm8k` 用轮转抽样,`bigbench` 用 ReWOO 的 `RandomState.permutation`。
    这里钉的是 bigbench 这一侧**没有**退化成轮转 ——
    轮转看起来也能给出「确定的一批题」,但它和 ReWOO 对不上。
    """
    assert bb.rewoo_draw(2290, 10, 2024) != list(range(10))


# ═══════════════════════════════════════════════════════════
# ★★★ 判分：和 ReWOO 逐字对齐（参照物原文抄在这里）
# ═══════════════════════════════════════════════════════════


def _rewoo_normalize_answer(s: str) -> str:
    """**ReWOO `utils/Evaluator.py` 的 `normalize_answer`，原文照抄。**"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _rewoo_f1_score(prediction: str, ground_truth: str) -> float:
    """**ReWOO `utils/Evaluator.py` 的 `f1_score`，原文照抄。**"""
    p = _rewoo_normalize_answer(prediction)
    g = _rewoo_normalize_answer(ground_truth)
    if p in ["yes", "no", "noanswer"] and p != g:
        return 0
    if g in ["yes", "no", "noanswer"] and p != g:
        return 0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pt)
    recall = 1.0 * num_same / len(gt)
    return (2 * precision * recall) / (precision + recall)


PARITY_CASES = [
    ("Yes", "Yes"), ("Yes", "No"), ("No", "No"), ("yes.", "Yes"),
    ("Yes, because it rains", "Yes"), ("The answer is Paris", "Paris"),
    ("Paris", "London"), ("a cat", "the cat"), ("", "Yes"),
    ("noanswer", "Yes"), ("7 km", "7187 m"),
]


def test_our_scoring_matches_rewoo_verbatim() -> None:
    """★★★ 两个函数**逐字对齐**,拿 ReWOO 的原文当参照跑一批用例。

    ★ 为什么把它们的实现抄进测试:判分器换了,准确率就不可比
      （`core/bench.py` 的头一句）。抄一份参照进来,以后谁改了我们的版本,
      这条测试会红 —— 而**只靠「我照着抄了」这句话是没有保障的**。
    """
    for pred, gold in PARITY_CASES:
        assert bb.normalize_answer(pred) == _rewoo_normalize_answer(pred), pred
        assert bb.token_f1(pred, gold) == pytest.approx(_rewoo_f1_score(pred, gold)), \
            f"f1 不一致:{pred!r} vs {gold!r}"


def test_the_yes_no_short_circuit_is_preserved() -> None:
    """★ 那条短路是**判分口径的一部分**:预测 `Yes` 而金标是别的时候直接 0,
    不让 `yes` 这个 token 蹭到部分分。去掉它,`f1` 就不再是 ReWOO 那个 f1。"""
    assert bb.token_f1("Yes", "Yes and no") == 0.0
    assert bb.token_f1("No", "No") == 1.0


# ═══════════════════════════════════════════════════════════
# ★★★ 物理题的容差 —— 金标自己就有离散度
# ═══════════════════════════════════════════════════════════


def test_the_gold_set_contains_variants_that_differ_by_percent() -> None:
    """★★★ `Can be one of ['7187 m', '7 km']` —— 这两个**不相等**,差 2.67%。

    它们是同一个量的两种舍入写法。**判分器的容差不能小于金标自己的离散度** ——
    否则一个完全命中金标的答案会被判错,而它在日志上和「算错了」一模一样。
    """
    acc = [q for q in (bb.parse_quantity(c) for c in bb._acceptable("Can be one of ['7187 m', '7 km']")) if q]
    assert len(acc) == 2
    tol = bb._tolerance(acc)
    assert tol > 0.0267, f"容差 {tol:.4f} 小于金标组内间距,会把命中金标的答案判错"
    # 而且不能大到把别的量也放过 —— 7 m 和 7 kg 之间必须是不可比
    assert bb._relative_error(bb.parse_quantity("7 m"), bb.parse_quantity("7 kg")) == float("inf")


def test_an_answer_matching_the_gold_variant_is_correct() -> None:
    """★ 上面那条的**后果版**:答 `7 km` 而金标写着 `7187 m` 时应当判对。

    只测容差的数值是不够的 —— 这一条测的是判分真的会走对那一支。
    """
    bench = bb.PhysicsQuestions()
    task = _task(gold="Can be one of ['7187 m', '7 km']",
                 options=["7187 m", "7 km"])
    assert bench.score(task, _traj("7 km")).correct, "金标自己列出的答案必须判对"
    assert bench.score(task, _traj("7187 m")).correct
    assert bench.score(task, _traj("7 m")).correct is False


def test_different_base_units_are_never_equal() -> None:
    """★ `7 m` 和 `7 kg` 数值相同而物理量不同 —— 必须判错。

    所以 `parse_quantity` 要连**基名**一起返回,不能只返回一个数。
    """
    assert bb.parse_quantity("7 m").unit == "m"
    assert bb.parse_quantity("7 km").unit == "m", "km 的基名是 m"
    assert bb.parse_quantity("7 kg").unit == "g"


def test_a_non_numeric_gold_is_a_scorer_error_not_a_wrong_answer() -> None:
    """★★ 金标解析不出来时报 `scorer_error`,**不是** `wrong_answer`。

    两者的行动完全不同:前者说明**判分器**坏了,后者说明模型答错了。
    混在一起会让一个坏掉的判分器看起来像模型不行 —— 而那种错会一路进表。
    """
    bench = bb.PhysicsQuestions()
    task = _task(gold="说不清", options=[])
    j = bench.score(task, _traj("12 m"))
    assert not j.correct and j.failure_class == "scorer_error"


def test_physics_has_two_variants_and_exact_is_the_bigbench_one() -> None:
    """★★ `numeric`（我们的）和 `exact`（BigBench 的字符串口径）**必须并存**。

    两个数大概率差很远 —— 而**那个差本身就是发现**:它说明 ReWOO 表里
    physics 的 `em` 那一列在量的是字符串,不是物理。
    """
    bench = bb.PhysicsQuestions()
    variants = bench.score_variants()
    assert set(variants) == {"numeric", "exact"}
    task = _task(gold="Can be one of ['7187 m', '7 km']", options=["7187 m", "7 km"])
    assert variants["numeric"](task, _traj("7 km")).correct is True
    assert variants["exact"](task, _traj("7 km")).correct is False, \
        "字符串口径下 `7 km` 对不上 `Can be one of [...]` —— 这是它本来的样子"


# ═══════════════════════════════════════════════════════════
# 输出契约：**任务文本里已经有了,就不许再加第二条**
# ═══════════════════════════════════════════════════════════


def test_two_of_them_do_not_add_a_second_output_contract() -> None:
    """★★ `strategy_qa` 的 input 是 `Answer with 'Yes' or 'No': ...`，
    `sports` 的 input 是 `Determine whether ... plausible or implausible: ...`
    —— **契约已经在里面了**。

    再加一句「answer with yes or no」就是**两条输出契约并存**,
    而两条冲突的指令从 prompt 上看不出来,只有输出能证明模型听了哪条
    （这一条我们踩过）。所以这两个 loader 必须**原样返回 input**。
    """
    for cls in (bb.StrategyQa, bb.SportsUnderstanding):
        bench = cls()
        rows = bench._rows()
        row = rows[0]
        assert bench._prompt(row, []) == row["input"], f"{bench.name} 动了 input"


def test_physics_does_add_one_because_its_input_has_none() -> None:
    """★ 反面:`physics_questions` 的 input **没有**契约,所以那一个要加。

    「不加第二条」的前提是**第一条存在** —— 不加区分地一律不加,
    就等于让自由文本任务没有输出契约。
    """
    bench = bb.PhysicsQuestions()
    row = bench._rows()[0]
    prompt = bench._prompt(row, [])
    assert prompt.startswith(row["input"]), "原题必须在"
    assert prompt != row["input"], "这一条必须补契约"
    assert "unit" in prompt.lower()


# ═══════════════════════════════════════════════════════════
# 声明与偏离
# ═══════════════════════════════════════════════════════════


def test_all_three_declare_themselves_and_state_the_divergence() -> None:
    """★★ 每条 loader 都要:①有自己的 `name`/`dataset_version`;
    ②**默认不给工具**,并把「ReWOO 给了五个工具」这条偏离写在文档里。

    ★ 工具那条不解决,我们的数就**不能和 ReWOO 表里的数并排** ——
      所以它必须是**代码里读得到的**,不是只写在文档里。
    """
    for cls, kind in ((bb.StrategyQa, "binary"),
                      (bb.SportsUnderstanding, "binary"),
                      (bb.PhysicsQuestions, "numeric")):
        bench = cls()
        assert bench.name and bench.dataset_version.startswith("bigbench/")
        assert bench.answer_kind == kind, f"{bench.name} 的答案空间没声明对"
        assert list(bench.tools()) == [], "默认不给工具（理由见模块头偏离 ①）"
        assert bench.tool_impls() == {}

    doc = bb.__doc__ or ""
    assert "工具集" in doc and "ReWOO" in doc, "偏离 ① 必须写在模块头"


def test_the_download_declaration_is_pinned_and_serves_all_three() -> None:
    """★★ 三条声明**共用同一个 tarball**,而且 `revision` 是 SHA 不是 `main`。

    ★ `pinned` 的判据是「不是 main/unpinned」—— `main` 等于「哪天拉的算哪天」,
      那样三个 loader 会在不同时间拿到不同版本的数据,而分数照常出。
    """
    specs = [cls().downloads()[0] for cls in
             (bb.StrategyQa, bb.SportsUnderstanding, bb.PhysicsQuestions)]
    assert {s.locator for s in specs} == {bb.REWOO_TARBALL}
    assert len({s.revision for s in specs}) == 1
    for s in specs:
        assert s.pinned, f"{s.dataset} 的 revision 没钉住"
        assert len(s.files) == 1 and s.files[0].endswith(".csv")
    assert {s.files[0] for s in specs} == {
        "strategy_qa.csv", "sports_understanding.csv", "physics_question.csv"}


def test_the_draw_uses_the_passed_seed_and_that_is_reported() -> None:
    """★ `seed` 从 CLI 来。**换 seed 就是换一批题**,所以它必须真的生效 ——
    一个被忽略的 seed 会让两次跑挂同一个名字却是不同的题。"""
    bench = bb.StrategyQa()
    a = [t.task_id for t in bench.tasks(split="test", limit=5, seed=2024)]
    b = [t.task_id for t in bench.tasks(split="test", limit=5, seed=7)]
    assert a != b, "seed 没生效"
    assert a == [t.task_id for t in bench.tasks(split="test", limit=5, seed=2024)], "不稳定"


# ── 夹具 ────────────────────────────────────────────────────


def _task(*, gold: str, options: list[str]):
    from experiments.core.types import Task

    return Task(task_id="t", prompt="p", gold=gold, oracle_context=None,
                meta={"options": options, "raw_target": gold})


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="test", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)
