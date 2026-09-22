"""FEVER —— **改编自 ReWOO 的 `DataLoader.load_fever`。**

## 出处

| | |
|---|---|
| **论文** | Thorne, Vlachos, Christodoulopoulos, Mittal. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL 2018. arXiv:1803.05355 |
| **仓库** | `github.com/facebookresearch/fever` ／ 评测基线 `sheffieldnlp/fever-baselines` |
| **ReWOO 的用法** | `utils/DataLoader.py::load_fever` — **它用的是 HF `copenlu/fever_gold_evidence`**，不是官方原始仓库 |
| **数据** | HF `copenlu/fever_gold_evidence`（train 228,277 / validation 15,935 / test 16,039）|

★ **这个数据集没有 HotpotQA / TriviaQA 那个坑**:实测三个划分**都带标签**
（后两个的 `test` 是 withhold 的）。所以 ReWOO 的 `type="test"` 默认值
在这里是**能跑的** —— 这也是为什么它只在这一个上没暴露出来。

## ★★ 它是**三选一**,而且是 `choice` 的天然形状

```
SUPPORTS / REFUTES / NOT ENOUGH INFO
```

实测 test 前 500 条分布:`NOT ENOUGH INFO 171 / REFUTES 170 / SUPPORTS 159`
—— **接近均匀**,所以「恒答一类」拿不到 1/3 以上的分,
而「答不上来」和「答反了」是**两种不同的错**（NEI vs REFUTES）。

★ 这正好是 `DECISION.md` 里 `choice` 那一格的用例:答案空间固定且小,
**一次前向就够,不需要逐 token 吐出来**。所以 `answer_kind = "choice"`、
`meta["options"]` 带上三个标签 —— 类型化的臂据此把它做成 `pickInput` 的候选。

⚠️ **但注意别把它和 `noul` 混了**:三选一**不是**「真/假」。
把它拆成两个 `noul`（「可验证吗」+「是支持还是反驳」）是一个**变体**,
它值不值得做要量了才知道 —— **别默认它更好**。

## 判分

官方口径是 **label accuracy**（三选一是否命中）。这里给两个口径:

- `label`（默认）:归一化后比标签。**这是 FEVER 官方报的那个数。**
- `contains`:预测里**包含**某个标签就算对。

⚠️ `contains` 是**宽松口径**,它会把「SUPPORTS 和 REFUTES 都有可能」这种
两头堵的回答判成对。**两个都报,但表里必须写明是哪个** ——
和 GSM8K 的 `strict`/`flexible`、TriviaQA 的 `rewoo`/`official` 是同一类选择。

## ★ 两处偏离

**① 工具集**:ReWOO 给五个工具，FEVER 走 `ReactExtraTool`（带工具）。
默认不给,理由同 `benchmark/hotpotqa/hotpotqa.py` 的「偏离 ①」。
★ 但 FEVER 比 HotpotQA **轻一些**:它的 claim 常可靠参数化知识判,
所以「没工具就没意义」这条对它**不成立得那么彻底** —— 记在这里,
别把两个数据集的处置混为一谈。

**② 崩溃不计入**:同 bigbench 模块头「偏离 ③」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.benchmark.official_scorers import (
    MissingEvidence,
    fever_is_correct_label,
)
from experiments.benchmark.rewoo_port import normalize_answer, rewoo_draw, REWOO_SEED
from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

DATASET_ID = "copenlu/fever_gold_evidence"

#: FEVER 的三个标签。**顺序有意义** —— 它进 prompt 时的排列会影响模型。
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

#: 实测:三个划分都带标签,所以 ReWOO 的 `test` 默认值在这里能用。
REWOO_SPLIT = "test"

_LABEL_RE = {label: re.compile(re.escape(label), re.I) for label in LABELS}


@dataclass
class Fever:
    """FEVER 事实核查（三选一）。"""

    name: str = "fever"
    dataset_version: str = field(init=False, default="unpinned")
    hf_home: Path = field(default_factory=lambda: DATASET_DIR / "hf")
    with_tools: bool = False
    limit: int | None = None
    seed: int = REWOO_SEED
    #: 默认口径。★ 官方报的是 `label`。
    headline: str = "label"

    def _rows(self, split: str) -> list[dict]:
        import os

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, split=split)
        self.dataset_version = f"{DATASET_ID}@{getattr(ds, '_fingerprint', 'unpinned')}"
        return [dict(row) for row in ds]

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        rows = self._rows(split or REWOO_SPLIT)
        usable = sum(1 for r in rows if str(r.get("label") or "") in LABELS)
        if usable == 0:
            # ★ 同一道守卫（HotpotQA / TriviaQA 都栽在这里）。这里它**不会触发**
            #   （实测三个划分都有标签）—— 留着是为了**划分被换掉**时能挡住。
            raise ValueError(f"{self.name}: {split or REWOO_SPLIT} 划分里没有一条带标签")

        for i in rewoo_draw(len(rows), limit if limit is not None else self.limit, seed):
            row = rows[i]
            label = str(row.get("label") or "")
            if label not in LABELS:
                raise ValueError(f"{self.name}: 第 {i} 条的标签认不出来:{label!r}")
            yield Task(
                task_id=f"{self.name}/{row.get('id') or i}",
                prompt=self._prompt(row),
                gold=label,
                # ★ 金标证据:ReWOO 把 `evidence` 整个丢掉了,我们留着。
                #   ⚠️ 但 FEVER 的 `evidence` 是 (page, sent_id) 元组,
                #      **不是句子本身** —— 要还原成句子得另下 Wikipedia dump
                #      （`docs/PLAN-*.md` 说的「最长前置」就是它）。
                #      所以这里先只记页名,`oracle_context` 留 `None`,
                #      **不假装有证据**。
                oracle_context=None,
                meta={
                    "row": i,
                    "options": list(LABELS),
                    "answer_kind": "choice",
                    "evidence_pages": self._evidence_pages(row),
                    "verifiable": row.get("verifiable"),
                },
            )

    def _prompt(self, row: dict) -> str:
        """★ 原数据只有 claim,没有输出契约 —— **所以要给一条**。

        契约里把三个标签**逐字**列出来（大写、原样）——
        FEVER 的标签是 `NOT ENOUGH INFO` 这种带空格的字符串,
        不写清楚的话模型会答 `NEI` 或 `Not enough information`,而判分会判错。
        """
        return (
            f"Claim: {row['claim']}\n\n"
            "Does the above claim follow from established facts? Answer with "
            "exactly one of these three labels and nothing else:\n"
            "  SUPPORTS\n  REFUTES\n  NOT ENOUGH INFO"
        )

    def _evidence_pages(self, row: dict) -> list[str]:
        """把 `evidence` 里的页名抠出来。**只留页名,不留句子** ——
        句子要从 Wikipedia dump 里取,那个前置我们还没做（见 `tasks()` 的说明）。"""
        pages: list[str] = []
        for item in row.get("evidence") or []:
            for ev in item if isinstance(item, (list, tuple)) else [item]:
                if isinstance(ev, (list, tuple)) and len(ev) >= 3 and ev[2]:
                    name = str(ev[2])
                    if name not in pages:
                        pages.append(name)
        return pages

    # ── 判分 ────────────────────────────────────────────────

    def score_variants(self) -> dict[str, object]:
        """★ 四个口径,而**官方的主指标是 `strict`,不是 `label`**:

        | 口径 | 是什么 | 能不能算 |
        |---|---|---|
        | `label` | `acc_score` —— 三选一的标签准确率 | ✅ |
        | `strict` | **FEVER score** —— 标签对 **且** 一整组金标证据在前 5 个预测里 | ❌ 缺证据字段 |
        | `contains` | 宽松:回答里出现某个标签就算 | ✅（**两头堵会判对**）|

        ⚠️ `strict` 抛 `MissingEvidence`。**NEI 类免检证据那一支不影响这件事** ——
          非 NEI 的那些仍然要预测证据,而我们的答案契约里没有。
        """
        return {
            "label": lambda t, tr: self._judge(t, tr, "label"),
            "strict": lambda t, tr: self._judge(t, tr, "strict"),
            "contains": lambda t, tr: self._judge(t, tr, "contains"),
        }

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        return self._judge(task, trajectory, self.headline)

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")
        want = str(task.gold)

        if mode == "strict":
            raise MissingEvidence(
                "FEVER 的 strict（FEVER score）需要模型输出 predicted_evidence —— "
                "当前答案契约里没有这个字段。\n"
                "  → 报 `label` 是可以的,但要写明它是 acc,不是 strict"
            )

        if mode == "label":
            # ★ 用**官方**那个函数（`scorer.py::is_correct_label`）,逐字那一份。
            hit = fever_is_correct_label(want, answer)
            return Judgment(correct=hit, score=1.0 if hit else 0.0,
                            detail=f"label（=官方 acc_score）：{answer[:40]!r} vs {want!r}",
                            failure_class=None if hit else "wrong_answer")

        found = {label for label, rx in _LABEL_RE.items() if rx.search(answer)}
        hit = want in found
        detail = f"contains：命中 {sorted(found) or '无'}"
        if len(found) > 1:
            detail += f"（⚠️ 答了 {len(found)} 个标签 —— 两头堵）"
        return Judgment(correct=hit, score=1.0 if hit else 0.0, detail=detail,
                        failure_class=None if hit else "wrong_answer")

    def check(self, task: Task, answer: str) -> bool:
        return self._judge(task, _traj(answer), self.headline).correct

    # ── 工具与下载 ──────────────────────────────────────────

    def tools(self) -> Sequence[object]:
        return ()

    def tool_impls(self) -> dict[str, object]:
        return {}

    def downloads(self) -> Sequence[DownloadSpec]:
        return [DownloadSpec(
            dataset=self.name,
            kind="hf-dataset",
            locator=DATASET_ID,
            revision="main",
            size_hint="~50 MB（228,277 + 15,935 + 16,039）",
            note=("★ ReWOO 用的就是这个 HF 源,**不是**官方原始仓库。"
                  "三个划分都带标签（实测）,没有 HotpotQA / TriviaQA 那个坑。"
                  "`revision` 未钉,已用 fingerprint 记进 dataset_version"),
        )]


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="check", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)


__all__ = ["Fever", "LABELS", "DATASET_ID"]
