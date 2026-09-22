"""TriviaQA —— **改编自 ReWOO 的 `DataLoader.load_trivia_qa` + `Evaluator`。**

## 出处

| | |
|---|---|
| **论文** | Joshi, Choi, Weld, Zettlemoyer. *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. ACL 2017. arXiv:1705.03551 |
| **仓库** | `github.com/mandarjoshi/trivia_qa` |
| **ReWOO 的用法** | `utils/DataLoader.py::load_trivia_qa`、`utils/Evaluator.py` |
| **数据** | HF `trivia_qa`，config **`rc.nocontext`**（ReWOO 用的就是它）|

## ★★★ 这个数据集有一个**必须做选择**的判分口径问题

TriviaQA 的每条答案**不是一个字符串,是一组别名**:

```python
answer = {"value": "David Seville",
          "aliases": ["David Seville", "Dave Seville", "Ross Bagdasarian Sr."],
          "normalized_aliases": [...], ...}
```

而两种口径**都有出处**,而且**给出的数不一样**:

| 口径 | 判据 | 出处 |
|---|---|---|
| **ReWOO** | 只比 `answer["value"]` | `Evaluator` 里 `label = self.dataset["answer"][i]["value"]` |
| **官方** | 命中**任意一个别名**即算对 | TriviaQA 官方评测（`aliases` 就是为此存在的）|

★★ **和 GSM8K 那次是同一件事**（`strict` vs `flexible`:同一批轨迹,
`flexible` 12/12、`strict` 0/12）。所以这里也**两个口径都报**,
`score_variants()` 给 `rewoo` 和 `official` 两个,报数时必须写明是哪个。

⚠️ **默认口径是 `rewoo`** —— 因为我们要和 ReWOO 比。但**表里必须标出来**:
拿我们的 `rewoo` 口径去比别人的 `official` 口径,是在比判分器,不是比方法。

## ★ 两处必须记住的偏离

**① 工具集。** ReWOO 给五个工具（`Google` / `Wikipedia` / `WolframAlpha` /
`Calculator` / `LLM`），TriviaQA 走 `ReactBase`。同 HotpotQA:
**工具是这个数据集的一部分**（`nocontext` 就是不给上下文的那个 config,
agent 必须自己查）。默认不给工具 ⇒ **我们的数不能和 ReWOO 的并排**,
理由和处置同 `benchmark/hotpotqa/hotpotqa.py` 的「偏离 ①」。

**② 崩溃不计入。** 同 bigbench 模块头「偏离 ③」。

## `rc.nocontext` 这个名字的含义

`rc` = reading comprehension。`nocontext` 变体**不带检索文档**,
只给问题和答案 —— 所以它和 `hotpot_qa:fullwiki` 一样,
**没有工具就没法做**,而不是「靠参数化知识也能答」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.benchmark.official_scorers import (
    triviaqa_get_ground_truths,
    triviaqa_is_exact_match,
)
from experiments.benchmark.rewoo_port import (
    REWOO_SEED,
    normalize_answer,
    rewoo_draw,
)
from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

DATASET_ID = "trivia_qa"
CONFIG = "rc.nocontext"

#: ★★★ **不要照抄 ReWOO 的 `type="test"`。**
#:
#: ReWOO 的 `load_trivia_qa` 默认 `type="test"`，而 `run_eval.py` 从不覆盖它。
#: **实测（2026-09-22，HF 镜像）:`rc.nocontext` 的 test 划分 17,210 条,
#: `answer["value"]` 全是 `"<unk>"`、`aliases` 全是空表。**
#: 官方的 test 标注是 withhold 的。
#:
#: ⇒ 我们默认 `validation`（17,944 条,答案齐全）。
#: ⇒ `test` 仍可显式传,但 `tasks()` 会**当场炸**。
#:
#: ⚠️ 于是 ReWOO 里那一行 `label = self.dataset["answer"][i]["value"]`
#: 在 test 上**每个标签都是 `<unk>`** —— 它表里 TriviaQA 的数要么来自
#: `validation`,要么就是拿 `<unk>` 比出来的。**在弄清之前不能当对照。**
REWOO_SPLIT = "validation"


@dataclass
class TriviaQa:
    """TriviaQA(rc.nocontext)。"""

    name: str = "triviaqa"
    dataset_version: str = field(init=False, default="unpinned")
    hf_home: Path = field(default_factory=lambda: DATASET_DIR / "hf")
    #: ★ 默认**不给工具**,理由见模块头「偏离 ①」。
    with_tools: bool = False
    limit: int | None = None
    seed: int = REWOO_SEED
    #: ★ 默认 `rewoo` —— 我们要和它比。**表里必须标出来是哪个口径。**
    headline: str = "rewoo"

    def _rows(self, split: str) -> list[dict]:
        import os

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, CONFIG, split=split)
        self.dataset_version = f"{DATASET_ID}:{CONFIG}@{getattr(ds, '_fingerprint', 'unpinned')}"
        return [dict(row) for row in ds]

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        rows = self._rows(split or REWOO_SPLIT)
        # ★ 划分级检查:逐行那道守卫只看得见被抽到的那几行（见函数说明）
        _assert_split_has_gold(rows, who=self.name, split=split or REWOO_SPLIT,
                              pick=lambda r: (r.get("answer") or {}).get("value")
        if (r.get("answer") or {}).get("value") not in (None, "", "<unk>")
        and (r.get("answer") or {}).get("aliases") else None)
        for i in rewoo_draw(len(rows), limit if limit is not None else self.limit, seed):
            row = rows[i]
            answer = row["answer"] or {}
            # ★★ **`<unk>` 是一个字符串,不是一个缺失值** —— 它会被
            #   `str()` 老老实实地变成金标 `"<unk>"`,而分数照常出。
            #   所以这条守卫要在**出题时**挡住,不是等到看分数时才发现。
            if answer.get("value") in (None, "", "<unk>") or not answer.get("aliases"):
                raise ValueError(
                    f"{self.name}: {split or REWOO_SPLIT} 划分里第 {i} 条没有可用金标"
                    f"（value={answer.get('value')!r}, "
                    f"{len(answer.get('aliases') or [])} 个别名）。\n"
                    f"  → 官方的 test 标注是 withhold 的,用 `--split validation`"
                )
            yield Task(
                task_id=f"{self.name}/{row.get('question_id') or i}",
                prompt=self._prompt(row),
                # ★ `gold` 用 ReWOO 那个 `value` —— 它是默认口径的金标。
                #   别名集挂在 `meta` 上,官方口径从那里取。
                gold=str(answer.get("value", "")),
                # TriviaQA 没有可用的金标证据（`nocontext` 就是不给上下文）——
                # 所以「direct + 金标证据」那一臂在这里**退化成和 direct 一样**。
                # ★ 这是它的一个短板,记在这里别忘。
                oracle_context=None,
                meta={
                    "row": i,
                    # ★ **整份 answer 对象都留着** —— 官方打分器的入参就是它
                    #   （`get_ground_truths(answer)`）。
                    #   只留一个别名列表的话,官方的 `HumanAnswers` 那一支
                    #   就永远接不上,而那是**接口缺口**,不是数据缺口。
                    "answer_object": dict(answer),
                    "aliases": list(answer.get("aliases") or []),
                    "answer_kind": "free",
                },
            )

    def _prompt(self, row: dict) -> str:
        """★ 原题不带输出契约 —— 所以要给一条。

        TriviaQA 的答案常常是实体名,而模型爱写一整句。契约里明说「短答」,
        是为了让 `em` 有意义 —— **但这属于说明,不属于提示词工程**:
        所有 arm 拿到的是同一条（契约归 benchmark,见 `core/bench.py`）。
        """
        return (
            f"{row['question']}\n\n"
            "Answer with the shortest exact name or phrase. Do not explain."
        )

    # ── 判分 ────────────────────────────────────────────────

    def score_variants(self) -> dict[str, object]:
        """★★ 两个口径**都报**（同 GSM8K 的 strict/flexible）。

        - `rewoo`：只比 `answer["value"]`，ReWOO 的做法。
        - `official`：命中**任意别名**即算对，TriviaQA 官方的做法。
        """
        return {
            "rewoo": lambda t, tr: self._judge(t, tr, "rewoo"),
            "official": lambda t, tr: self._judge(t, tr, "official"),
        }

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        return self._judge(task, trajectory, self.headline)

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")

        # ★ 归一化**只做一次** —— 两个口径共用同一套 normalize_answer
        #   （它是 ReWOO 的逐字那一份）。口径的差别只在**比几个字符串**。
        got = normalize_answer(answer)
        want = normalize_answer(str(task.gold))

        if mode == "rewoo":
            hit = got == want
            return Judgment(correct=hit, score=1.0 if hit else 0.0,
                            detail=f"{mode}：{answer[:40]!r} vs {task.gold!r}",
                            failure_class=None if hit else "wrong_answer")

        # official：**命中任意别名即算对** —— 调官方函数,不自己写。
        #
        # ★★ 第一版我用的是 `aliases`（**未归一化**那一份），而官方吃的是
        #    `NormalizedAliases + [normalize(HumanAnswers)]`
        #    （`triviaqa_evaluation.py::get_ground_truths`）。
        #    拿原始别名再归一化一次,在带标点或冠词的别名上会得到不同的字符串 ——
        #    **一个只在部分样本上出现的偏差,而总分看起来正常。**
        answer_object = task.meta.get("answer_object") or {}
        truths = triviaqa_get_ground_truths(answer_object)
        if not truths:
            return Judgment(correct=False, score=0.0,
                            detail="官方口径：这条没有别名可用", failure_class="scorer_error")
        hit = triviaqa_is_exact_match(answer_object, answer)
        return Judgment(correct=hit, score=1.0 if hit else 0.0,
                        detail=(f"official（官方 max-over-aliases，{len(truths)} 个）:"
                                f"{answer[:40]!r} {'命中' if hit else '未命中'}"),
                        failure_class=None if hit else "wrong_answer")

    def check(self, task: Task, answer: str) -> bool:
        return self._judge(task, _traj(answer), self.headline).correct

    # ── 工具与下载 ──────────────────────────────────────────

    def tools(self) -> Sequence[object]:
        return ()          # ★ 见模块头「偏离 ①」

    def tool_impls(self) -> dict[str, object]:
        return {}

    def downloads(self) -> Sequence[DownloadSpec]:
        return [DownloadSpec(
            dataset=self.name,
            kind="hf-dataset",
            locator=f"{DATASET_ID}:{CONFIG}",
            revision="main",
            size_hint="~700 MB（138,384 + 17,944 + 17,210）",
            note=("★ config 必须是 `rc.nocontext` —— ReWOO 用的就是它。"
                  "★ 答案带**别名集**,官方口径与 ReWOO 口径不同,"
                  "见本文件头部。`revision` 未钉,已用 fingerprint 记进 dataset_version"),
        )]


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="check", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)


__all__ = ["TriviaQa", "DATASET_ID", "CONFIG"]

def _assert_split_has_gold(rows: list[dict], *, who: str, split: str, pick) -> None:
    """★★ **划分级**的金标检查 —— 逐行那道守卫不够。

    实测:逐行守卫只看得见**被抽到的那几行**,而 `rewoo_draw` 是抽样。
    一个「九成没有金标」的划分,抽 10 条很可能一条都没抽到坏行 ——
    于是**整个划分被当成好的一路跑完**,而分数看起来正常。

    ⚠️ 而 `test` 划分是 **100%** 没有金标（HotpotQA 的 `answer` 全 `None`,
    TriviaQA 的 `value` 全 `<unk>`）,不是「偶尔缺几条」——
    所以这一道检查的判据是「**有多少条能用**」,不是「有没有一条坏行」。
    """
    usable = sum(1 for r in rows if pick(r) is not None)
    if usable == 0:
        raise ValueError(
            f"{who}: {split} 划分里 **{len(rows)} 条没有一条有可用金标**。\n"
            f"  → 官方的 test 标注是 withhold 的（留给排行榜）,"
            f"用 `--split validation`"
        )
