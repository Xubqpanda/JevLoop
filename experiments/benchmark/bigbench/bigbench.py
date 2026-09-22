"""BigBench 的三个任务 —— **改编自 ReWOO 的 loader,不是从零写的。**

## 出处（`docs/PLAN-*.md` §2 的硬要求:对着原论文仓库对齐并给出引用）

| | |
|---|---|
| **论文** | Xu, Peng, Islam, Lian, Xing. *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*. arXiv:2305.18323 |
| **仓库** | `github.com/billxbf/ReWOO`，钉 `9cd0283043ff4be0c9d614fda2789d143ca6ffd1` |
| **改编自** | `utils/DataLoader.py` 的 `load_strategy_qa` / `load_sports_understanding` / `load_physics_question` |
| **判分改编自** | `utils/Evaluator.py` 的 `normalize_answer` / `f1_score` / `get_metrics` |
| **原始出处** | BigBench `google/BIG-bench` 的 `benchmark_tasks/<task>/task.json` |

★ **为什么直接拿 ReWOO 的副本而不是回 BigBench 原站**:

1. **可比性。** 我们和 ReWOO 比的就是这三个数。回原站会引入版本差,
   而版本差**在数字上看不出来** —— 它只会让「我们比它好/差」多一个解释不了的原因。
2. **`strategy_qa` 在 BigBench 主干上已经没有了**（实测 `task.json` 返回 404）,
   后来被移出。官方另在 `eladsegal/strategyqa`。
   所以 ReWOO 那份 vendored CSV 是**现成可得的唯一一份**。

## ★★ 抽样:**能精确复现 ReWOO 那一批**

ReWOO 用 `df.sample(sample_size, random_state=seed)`，而 `run_eval.py` 的
`--seed` 默认是 **2024**（`DataLoader` 自己的默认是 2023，以 `run_eval` 为准）。

**实测**（pandas 2.2.2 / numpy 1.26.4）:

```python
df.sample(10, random_state=2024).index          == [994, 2185, 1311, ...]
np.random.RandomState(2024).permutation(2290)[:10] == [994, 2185, 1311, ...]
```

两者**逐位相同**，所以这里不依赖 pandas 也能复现那个抽样。
⇒ **`--seed 2024` 就是 ReWOO 那一批题。** 报数时必须写明 seed。

## ⚠️ 三处**必须记住**的偏离（不写明就会变成假的「可比」）

**① 工具集。** ReWOO 给这三个任务的是 `Google` / `Wikipedia` / `WolframAlpha` /
`Calculator` / `LLM` 五个工具（`run_eval.py` 的 `--toolset` 默认值），
而且 `strategy_qa` 走的是 `ReactExtraTool`，**确实带工具**。
我们这里**默认不给工具**，因为其中三个要付费/外部服务（SerpAPI、WolframAlpha），
给了也复现不了。⇒ **我们的数不能直接和它表里的数并排**,
要么补上可复现的等价工具，要么在表里标明工具集不同。**这一条没解决之前不许进对比表。**

**② 判分口径。** ReWOO 报三个数:`em` / `f1` / **`acc`**。
`acc` 是 **LLM 判官**（`langchain` 的 `QAEvalChain` + `OpenAI(temperature=0)`），
而那个判官**没有钉模型** —— `OpenAI()` 在老版 langchain 里落到
`text-davinci-003`，那个模型现在已经下线。**我们复现不了它，也不该假装复现了。**
所以我们默认报 `em`（可复现），`f1` 作为第二个口径；`acc` 不报,
或者报的时候必须写清用的是哪个判官。

**③ 崩溃不计入。** 它的 `Evaluator` 在异常时塞一个**全 `NaN`** 的响应,
而汇总用 `np.nanmean` —— 于是**跑挂的题被跳过,不是判错**。
那会把准确率抬高,而账面上看不出来。我们不这么做:
跑挂了就是 `failure_class` 里的一类,分母不变。

★ 顺带记一个它代码里的浪费:`_update_eval_dict` 对同一个例子调了 **三次**
`get_metrics()`，而 `get_metrics` 每次都调一次 LLM 判官 —— 于是**每个例子
要付 3 次判官的钱**，三次答案完全相同。（`em` / `f1` 本来不需要判官。）
"""

from __future__ import annotations

import csv
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# ★ 抽样与判分来自 ReWOO，**逐字**那一份住在 `benchmark/rewoo_port.py`
#   （不止 BigBench 用它 —— HotpotQA / TriviaQA 也是那一套）。
from experiments.benchmark.rewoo_port import (
    REWOO_COMMIT,
    REWOO_SEED,
    REWOO_TARBALL,
    first_token,
    normalize_answer,
    rewoo_draw,
    token_f1,
)
from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

# ── 判分：逐字抄 ReWOO 的 `Evaluator.py` ──────────────────────


@dataclass
class BigBenchTask:
    """三个任务的共同部分。**子类只声明 `name` / `csv_name` / 输出契约。**

    ★ 三个任务的 `target` 形状不同,而**判分口径必须各自说清楚**:
    `strategy_qa` 是 `Yes`/`No`，`sports_understanding` 是
    `plausible`/`implausible`（都是二选一 → 精确匹配有意义），
    而 `physics_questions` 的 target 形如 `"Can be one of ['7187 m', '7 km']"`
    —— **那不是答案,是 BigBench 的候选集**,精确匹配对它没有意义（见子类）。
    """

    #: 输出契约。★ 归 benchmark 不归 baseline（`core/bench.py` 的分工表）。
    #: 子类覆盖。
    answer_contract: str = ""

    limit: int | None = None
    seed: int = REWOO_SEED
    dataset_dir: Path = field(default_factory=lambda: DATASET_DIR)

    name: str = ""
    csv_name: str = ""
    #: 本题给出的工具。★ 默认空,理由见模块头「偏离 ①」。
    with_tools: bool = False

    def _csv_path(self) -> Path:
        return self.dataset_dir / self.name / self.csv_name

    def _rows(self) -> list[dict[str, str]]:
        path = self._csv_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: 缺 {path}\n"
                f"  跑 `python3 -m experiments.scripts.datasets --fetch {self.name}` 下载"
            )
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    # ── 协议 ────────────────────────────────────────────────

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """★ `split` 被忽略 —— 这三个是**单划分**数据集（BigBench 每个任务一份）。

        ★ **抽样用 ReWOO 那一次**:`seed=2024` + `RandomState(seed).permutation`。
          换 seed 就是换一批题,报数时必须写明。
        """
        if split not in ("test", "train", "all"):
            raise ValueError(f"{self.name} 只有一个划分，收到 {split!r}")
        rows = self._rows()
        for i in rewoo_draw(len(rows), limit if limit is not None else self.limit, seed):
            row = rows[i]
            options = self._options(row)
            yield Task(
                task_id=f"{self.name}/{i}",
                # ★ 任务陈述 + 输出契约都归 benchmark
                prompt=self._prompt(row, options),
                gold=self._gold(row, options),
                # 这三个任务没有可检索的外部证据 —— 所以 `direct + 金标证据`
                # 那一臂在这里退化成和 `direct` 一样。**是短板,记在这里别忘。**
                oracle_context=None,
                meta={"row": i, "raw_target": row.get("target", ""), "options": options},
            )

    # ── 子类填这几个 ────────────────────────────────────────

    def _prompt(self, row: dict[str, str], options: list[str]) -> str:
        raise NotImplementedError

    def _gold(self, row: dict[str, str], options: list[str]) -> str:
        raise NotImplementedError

    def _options(self, row: dict[str, str]) -> list[str]:
        """可选答案集合。空 = 自由文本。"""
        return []

    # ── 判分 ────────────────────────────────────────────────

    def score_variants(self) -> dict[str, object]:
        """★ **两种口径都报**（`em` 默认,`f1` 第二）—— 报的时候必须写明是哪个。

        ⚠️ ReWOO 的第三个口径 `acc`（LLM 判官）**不在里面**,理由见模块头「偏离 ②」。
        """
        return {
            "em": lambda t, tr: self._judge(t, tr, "em"),
            "f1": lambda t, tr: self._judge(t, tr, "f1"),
        }

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        return self._judge(task, trajectory, "em")

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            # ★ 跑挂/没答 **不跳过** —— 它进 failure_class,分母不变。
            #   ReWOO 那边是 `np.nanmean` 把 NaN 跳过的（模块头「偏离 ③」）。
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")

        want = str(task.gold)
        got = first_token(answer) if self._closed() else answer
        em = normalize_answer(got) == normalize_answer(want)
        f1 = token_f1(got, want)

        if mode == "em":
            return Judgment(correct=em, score=1.0 if em else 0.0,
                            detail=f"{got!r} vs {want!r}（em）" if not em
                            else f"{got!r} == {want!r}（em）",
                            failure_class=None if em else "wrong_answer")
        return Judgment(correct=f1 >= 0.5, score=f1,
                        detail=f"f1={f1:.3f}（{got!r} vs {want!r}）",
                        failure_class=None if f1 >= 0.5 else "wrong_answer")

    def _closed(self) -> bool:
        """答案是不是**闭集**里的一个。

        ★ 闭集任务要从输出里抠出第一个 token —— 模型常常写
        「Yes, because ...」,而金标只有 `Yes`。
          这不是放水:`_options()` 已经把可选值写进 prompt 了。
        """
        return bool(self._options({}) or self._static_options())

    def _static_options(self) -> list[str]:
        return []

    def check(self, task: Task, answer: str) -> bool:
        """给 Reflexion 的 Evaluator 用。**它吃到金标信号** —— 谁用了要写在名字里。"""
        return self.score(task, Trajectory(
            task_id=task.task_id, arm="check", steps=(), final_answer=answer,
            decisions=(), usage=(), escalated=False, error=None,
        )).correct

    # ── 工具 ────────────────────────────────────────────────

    def tools(self) -> Sequence[object]:
        return ()          # ★ 理由见模块头「偏离 ①」；这条不解决不许进对比表

    def tool_impls(self) -> dict[str, object]:
        return {}

    # ── 下载 ────────────────────────────────────────────────

    def downloads(self) -> Sequence[DownloadSpec]:
        """★ **一次 tarball 服务三个 loader** —— 所以三条声明长得一样,
        只有 `dataset` 和 `files` 不同。`dataset` 必须是各自的 `name`,
        它是「这份数据和谁有关」的那根线。"""
        return [DownloadSpec(
            dataset=self.name,
            kind="http",
            locator=REWOO_TARBALL,
            files=(self.csv_name,),
            revision=REWOO_COMMIT,
            size_hint="~6 MB（整个 ReWOO 仓库）",
            note=("ReWOO 仓库自带的 BigBench CSV。★ 一次下载抽三个文件。"
                  "原始出处是 `google/BIG-bench` 的 `benchmark_tasks/<task>/task.json`，"
                  "但**这里刻意用 ReWOO 那份** —— 我们和它比的就是这三个数，"
                  "回原站会引入一个在数字上看不出来的版本差"
                  "（而 `strategy_qa` 在 BigBench 主干上已被移除，实测 404）"),
        )]


# ═══════════════════════════════════════════════════════════
# 三个具体任务
#
# ★ 它们的 `input` **自带输出契约**:
#     strategy_qa  「Answer with 'Yes' or 'No': ...」
#     sports       「Determine whether ... plausible or implausible: ...」
#   ⇒ **一个字都不加。** 两条输出契约并存时,从 prompt 上看不出模型听了哪条
#     （这一条我们踩过:两边各写一句「答案格式」,只有输出能证明它听了哪个）。
#   而 physics 的 input **没有**契约,所以那一个要加。
# ═══════════════════════════════════════════════════════════


@dataclass
class StrategyQa(BigBenchTask):
    """**二选一**（`Yes` / `No`）—— 2,290 条。

    ★ 这个任务对我们是**最合适的一个**:它的答案空间就是 `noul`（P(真)）。
      常规 agent 让模型生成一句话来表达这个是非题,而它只需要一次前向。
    """

    name: str = "strategyqa"
    csv_name: str = "strategy_qa.csv"
    #: ★ 声明答案空间 —— 类型化的那些臂据此决定用 `noul` 还是 `choice`。
    answer_kind: str = "binary"
    dataset_version: str = f"bigbench/strategy_qa@rewoo-{REWOO_COMMIT[:8]}"

    def _prompt(self, row: dict[str, str], options: list[str]) -> str:
        return row["input"]                      # 契约已在里面

    def _gold(self, row: dict[str, str], options: list[str]) -> str:
        return row["target"]

    def _static_options(self) -> list[str]:
        return ["Yes", "No"]


@dataclass
class SportsUnderstanding(BigBenchTask):
    """**二选一**（`plausible` / `implausible`）—— 1,000 条。

    ★ 和 `strategy_qa` 一样是 `noul` 的天然形状。
    """

    name: str = "sportsunderstanding"
    csv_name: str = "sports_understanding.csv"
    answer_kind: str = "binary"
    dataset_version: str = f"bigbench/sports_understanding@rewoo-{REWOO_COMMIT[:8]}"

    def _prompt(self, row: dict[str, str], options: list[str]) -> str:
        return row["input"]                      # 契约已在里面

    def _gold(self, row: dict[str, str], options: list[str]) -> str:
        return row["target"]

    def _static_options(self) -> list[str]:
        return ["plausible", "implausible"]


@dataclass
class PhysicsQuestions(BigBenchTask):
    """**自由文本数值题** —— **只有 53 条**。

    ⚠️ 它的 `target` 不是答案,是**一组等价的写法**:

        Can be one of ['7187 m', '7 km']

    `7187 m` 和 `7 km` **不相等**（一个是 7.187 km）—— 它们是同一个量的
    舍入写法。所以:

    - **精确匹配对它没有意义**（拿 `normalize_answer` 比那串
      `can be one of 7187 m 7 km` 只会得到 0 分）—— 而 ReWOO 的
      `Evaluator` 正是这么算 `em` 的,所以**它报的 `em` 在这个数据集上没用**。
    - 我们实现的是**数值等价**:解析 `<数> <单位>`,换算到 SI 基本单位,
      按相对容差比。**这是我们的判据,不是 BigBench 的** —— 所以
      `score_variants()` 里同时给 `exact`（BigBench 的字符串口径）作对照。

    ★ 而且**题量只有 53**,95% 置信区间宽到没法比较两个方法。
    它的价值在**机制归因**（数值推理需要什么工具),不在准确率排名。
    """

    name: str = "physicsquestions"
    csv_name: str = "physics_question.csv"
    answer_kind: str = "numeric"
    dataset_version: str = f"bigbench/physics_questions@rewoo-{REWOO_COMMIT[:8]}"
    #: input 里**没有**契约,所以这一个要加。
    answer_contract: str = (
        "Give the numeric answer with its unit, for example `12.5 m`. "
        "A short expression is fine; do not explain."
    )

    def _prompt(self, row: dict[str, str], options: list[str]) -> str:
        return f"{row['input']}\n\n{self.answer_contract}"

    def _gold(self, row: dict[str, str], options: list[str]) -> str:
        return row["target"]

    def _options(self, row: dict[str, str]) -> list[str]:
        return _acceptable(row.get("target", ""))

    def _closed(self) -> bool:
        return False        # ★ 数值题不是闭集,不要走「取第一个 token」

    def score_variants(self) -> dict[str, object]:
        """两种口径:`numeric`（我们的,默认）和 `exact`（BigBench 的字符串口径）。

        ★ **两个数大概率差很远** —— 而那个差本身就是发现:
        它说明 ReWOO 表里 physics 的 `em` 那一列在量的是字符串,不是物理。
        """
        return {
            "numeric": lambda t, tr: self._judge(t, tr, "numeric"),
            "exact": lambda t, tr: self._judge(t, tr, "exact"),
        }

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")

        accepted = task.meta.get("options") or _acceptable(str(task.gold))
        if mode == "exact":
            hit = normalize_answer(answer) == normalize_answer(str(task.gold))
            return Judgment(correct=hit, score=1.0 if hit else 0.0,
                            detail=f"字符串口径（BigBench）：{answer[:40]!r}",
                            failure_class=None if hit else "wrong_answer")

        if not accepted:
            return Judgment(correct=False, score=0.0,
                            detail="金标里没有解析出可接受的答案",
                            failure_class="scorer_error")

        got = parse_quantity(answer)
        if got is None:
            # ★ 抠不出数 → 是**格式**失败,不是算错。分开报才看得出该怪谁。
            return Judgment(correct=False, score=0.0,
                            detail=f"抠不出数值：{answer[:60]!r}",
                            failure_class="no_numeric_answer")

        parsed = [q for q in (parse_quantity(c) for c in accepted) if q is not None]
        if not parsed:
            return Judgment(correct=False, score=0.0,
                            detail="金标里没有解析出可接受的数值",
                            failure_class="scorer_error")

        # ★ 容差由**金标自己**的离散度决定 —— 见 `_tolerance` 的说明。
        tol = _tolerance(parsed)
        best = None
        for candidate, want in zip(accepted, (parse_quantity(c) for c in accepted)):
            if want is None:
                continue
            rel = _relative_error(got, want)
            if best is None or rel < best[1]:
                best = (candidate, rel)
        assert best is not None

        candidate, rel = best
        ok = rel <= tol
        return Judgment(correct=ok, score=1.0 if ok else 0.0,
                        detail=f"{answer[:40]!r} vs {candidate!r}（相对误差 {rel:.4f}，容差 {tol:.4f}）",
                        failure_class=None if ok else "wrong_answer")


# ── 数值解析 ────────────────────────────────────────────────

#: **下限**相对容差。
#:
#: ★★ 实测（2026-09-22）:金标**自己那一组写法之间**就差了 **2.67%** ——
#: `7187 m` 和 `7 km` 是同一个量的两种舍入写法,而 7187 ≠ 7000。
#:
#: 我第一版定 1%,于是**一个完全命中金标的答案会被判错**,
#: 而它在日志上长得和「算错了」一模一样。**判分器的容差不能小于金标自己的离散度。**
#:
#: 所以真正的容差是**自适应**的:取 `max(这个下限, 金标组内最大间距 × 1.5)`。
#: 见 `_tolerance()`。
NUMERIC_TOL = 0.01

#: 金标组内间距 → 容差的放大系数。留一点余量:答案允许落在**任何**一个
#: 候选附近,而候选之间还有距离。
_TOL_MARGIN = 1.5


def _tolerance(accepted: list["Quantity"]) -> float:
    """由**金标自己**的离散度决定容差。

    ★ 这一条是实测逼出来的:金标写成 `['7187 m', '7 km']`,两者差 2.67%。
      固定 1% 的容差会把「答了 7 km」判错,而那是**金标自己列出的答案**。

    **一个判分器把金标里的答案判错,是判分器错了,不是答案错了。**
    """
    spread = 0.0
    for i, a in enumerate(accepted):
        for b in accepted[i + 1:]:
            err = _relative_error(a, b)
            if err != float("inf"):
                spread = max(spread, err)
    return max(NUMERIC_TOL, spread * _TOL_MARGIN)

#: SI 前缀。**只收常见的那几个** —— 收了不认识的宁可判「抠不出」,
#: 也不要猜一个换算率（猜错会静默算出一个看着正常的错答案）。
_PREFIX = {
    "": 1.0, "k": 1e3, "m": 1e-3, "c": 1e-2, "M": 1e6, "G": 1e9,
    "n": 1e-9, "u": 1e-6, "µ": 1e-6, "T": 1e12, "h": 1e2,
}
_CAN_BE_ONE_OF = re.compile(r"Can be one of \[(.*)\]", re.I)


def _acceptable(target: str) -> list[str]:
    """把 `Can be one of ['7187 m', '7 km']` 里的候选抠出来。

    ★ 认不出来就返回**空** —— 空会让上层报 `scorer_error`,而不是
      拿整串去比然后静默全判错。**一个静默的错分数比一个明说的问题糟。**
    """
    m = _CAN_BE_ONE_OF.search(target)
    if not m:
        return [target] if target.strip() else []
    inner = m.group(1)
    return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]


def parse_quantity(text: str) -> "Quantity | None":
    """`"7 km"` → `Quantity(7.0, 1000.0, "m")`。

    ★ 只认「一个数 + 一个可选的 SI 前缀 + 一个单位」。认不出来返回 `None`,
      **不猜。** 一个猜错的换算率会静默算出一个看着正常的错答案。

    ★ **单位名要一起带着走** —— 只返回数值是不够的:`7 m` 和 `7 kg`
      数值相同而物理量不同,而那种相等是必须被判成错的。
    """
    text = text.strip().strip("`*_$").rstrip(".")
    m = re.match(
        r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
        r"([A-Za-zµ°][-\^\*A-Za-z0-9/]*)?$",
        text,
    )
    if not m:
        # 没有单位也算（纯数字答案）—— 但那就是一个**无量纲**的量
        m2 = re.match(r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$", text)
        if not m2:
            return None
        return Quantity(value=float(m2.group(1)), scale=1.0, unit="")
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").strip()
    return Quantity(value=value, scale=_unit_scale(unit), unit=_base_unit(unit))


def _unit_scale(unit: str) -> float:
    """SI 前缀的换算率。不认识的单位返回 1.0（**但基名仍参与比较**）。"""
    if not unit:
        return 1.0
    for prefix in sorted(_PREFIX, key=len, reverse=True):
        if prefix and unit.startswith(prefix) and len(unit) > len(prefix):
            return _PREFIX[prefix]
    return 1.0


def _base_unit(unit: str) -> str:
    """剥掉 SI 前缀之后的单位基名。`km` → `m`，`kg` → `g`。

    ⚠️ `kg` 的基名是 `g` —— 这是 SI 的定义（千克是基本单位,但前缀是 k）。
    对比较而言无所谓:两边都会被同样地剥一次,`7 kg` 和 `7000 g` 仍然相等。
    """
    if not unit:
        return ""
    for prefix in sorted(_PREFIX, key=len, reverse=True):
        if prefix and unit.startswith(prefix) and len(unit) > len(prefix):
            return unit[len(prefix):]
    return unit


@dataclass(frozen=True)
class Quantity:
    """数值 + 到 SI 基本单位的换算率 + **基名**。三者缺一不可。"""

    value: float
    scale: float
    unit: str

    def si(self) -> float:
        return self.value * self.scale


def _relative_error(got: Quantity, want: Quantity) -> float:
    """两个量的相对误差。**基名不同 = 不可比,返回 `inf`。**"""
    if got.unit != want.unit:
        return float("inf")
    a, b = got.si(), want.si()
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return abs(a - b) / abs(b)


__all__ = [
    "BigBenchTask", "StrategyQa", "SportsUnderstanding", "PhysicsQuestions",
    "NUMERIC_TOL", "parse_quantity", "Quantity",
    # 转出去,免得用的人还要知道它们住在 `rewoo_port`
    "REWOO_COMMIT", "REWOO_SEED", "REWOO_TARBALL",
    "normalize_answer", "token_f1", "rewoo_draw",
]
