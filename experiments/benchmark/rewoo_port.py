"""**ReWOO 移植过来的那一套:抽样 + 判分。**

## 为什么单独一个文件

出处是 `github.com/billxbf/ReWOO`（钉 `9cd0283043ff4be0c9d614fda2789d143ca6ffd1`）:

| 这里 | ReWOO 的哪一段 |
|---|---|
| `rewoo_draw` | `DataLoader` 里那句 `df.sample(k, random_state=seed)` |
| `normalize_answer` | `utils/Evaluator.py` 的 `normalize_answer` |
| `token_f1` | `utils/Evaluator.py` 的 `f1_score` |

★ **为什么不让它们住在某一个数据集的文件里**:用它们的**不止一个数据集** ——
BigBench 那三个、HotpotQA、TriviaQA 全是 ReWOO 那一套。
让 HotpotQA 去 `import bigbench` 拿 `normalize_answer` 是**错的名**,
下一个人会以为判分口径是 BigBench 的,然后去「修」一个不该动的东西。

★ **判分口径就是可比性本身**（`core/bench.py` 的第一句:换判分器,准确率不可比）。
所以这一份必须是**一处**,而且是**逐字**的那一份。
`tests/test_bigbench.py::test_our_scoring_matches_rewoo_verbatim` 把 ReWOO 的原文
抄进测试当参照跑 —— **「我照着抄了」这句话本身没有保障,一条对拍测试才有。**
（它已经赚回一次:`normalize_answer` 的**顺序**我一开始写反了。）
"""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

#: ReWOO 主干的 commit —— `DownloadSpec.revision` 用它,`main` 不算钉住。
REWOO_COMMIT = "9cd0283043ff4be0c9d614fda2789d143ca6ffd1"
REWOO_TARBALL = f"https://codeload.github.com/billxbf/ReWOO/tar.gz/{REWOO_COMMIT}"

#: ReWOO `run_eval.py` 的 `--seed` 默认值。**这就是它那一批题。**
#: （`DataLoader` 自己的默认是 2023,但它被 `run_eval` 覆盖了。）
REWOO_SEED = 2024


def rewoo_draw(total: int, limit: int | None, seed: int) -> list[int]:
    """复现 ReWOO 的 `df.sample(limit, random_state=seed)`。

    ★ 实测和 pandas 逐位相同（pandas 内部就是 `choice(replace=False)`,
      而它等于 `permutation(n)[:k]`）。**不依赖 pandas** —— 少一个重依赖,
      而且「我们跑的是哪几条」这句话不随 pandas 版本变化。
    """
    if limit is None or limit >= total:
        return list(range(total))
    return [int(i) for i in np.random.RandomState(seed).permutation(total)[:limit]]


# ── 判分：逐字抄 ReWOO 的 `Evaluator.py` ──────────────────────


def normalize_answer(s: str) -> str:
    """ReWOO `normalize_answer`，逐字。

    ★★★ **顺序是有意义的,而且第一版我写反了。**

    ReWOO 的顺序是 `white_space_fix(remove_articles(remove_punc(lower(s))))`
    —— **先 lower,再去冠词**。而 `re.sub(r"\\b(a|an|the)\\b", ...)` 没有
    `IGNORECASE`,所以去冠词必须在 lower **之后**做。

    我先去了冠词再 lower,于是 `"The answer is Paris"` 里的 `The` **去不掉**
    （大写 T 不匹配）,归一化结果是 `"the answer is paris"` 而不是
    `"answer is paris"` —— **两条不同的字符串,于是同一批轨迹的 em 会不一样。**

    ★ 抓到它的是 `test_our_scoring_matches_rewoo_verbatim`。
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    # ★ 顺序逐字照抄,不要「顺手优化」
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def token_f1(prediction: str, ground_truth: str) -> float:
    """ReWOO `f1_score`，逐字（含那条 yes/no/noanswer 短路）。

    ★ 那条短路很重要:预测是 `yes` 而金标不是时直接 0 分,
      不让「yes」这个 token 蹭到部分分。
    """
    p, g = normalize_answer(prediction), normalize_answer(ground_truth)
    if p in ("yes", "no", "noanswer") and p != g:
        return 0.0
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0

    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall)


def first_token(text: str) -> str:
    """闭集任务取第一个 token。**去标点** —— 模型爱写 `Yes.`。

    ★ 这不是放水:`_options()` 已经把可选值写进 prompt 了。
    """
    stripped = text.strip().strip("*`_ \t")
    token = re.split(r"[\s,.;:!?()\[\]{}]+", stripped, maxsplit=1)[0]
    return token or stripped


__all__ = [
    "REWOO_COMMIT", "REWOO_TARBALL", "REWOO_SEED",
    "rewoo_draw", "normalize_answer", "token_f1", "first_token",
]
