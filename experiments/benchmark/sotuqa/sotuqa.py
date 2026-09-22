"""SOTU_QA —— **ReWOO 自己的 curated 数据集**（74 条）。

## 出处

| | |
|---|---|
| **论文** | Xu, Peng, Islam, Lian, Xing. *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*. arXiv:2305.18323 |
| **仓库** | `github.com/billxbf/ReWOO`，钉 `9cd0283043ff4be0c9d614fda2789d143ca6ffd1` |
| **数据** | 仓库里自带的 `data/SOTU/SOTU_QA.csv`（74 行，`question` / `answer`）|
| **原文** | `data/docs/state_of_the_union.txt`（42 KB，2023 年国情咨文全文）|

★ **这是 ReWOO 唯一一个自己造的数据集** —— README 里说的
「six public NLP benchmarks **and a curated dataset**」就是它。
所以它没有「别人的官方判分器」可调,**判分口径由它的 `Evaluator` 定义**。

## ★★★ 74 条,而且金标是一整句话 —— 这两件事都要说清

实测第一条:

```
Q: Based on State of the Union Address 2023: Is Speaker of the House
   Kevin McCarthy older than Nancy Pelosi?
A: No, Kevin is 58 while Nancy is 83
```

于是:

1. **`em` 和 `f1` 对它几乎没有意义。** 金标是一句解释,而模型不可能
   逐字复现 `"No, Kevin is 58 while Nancy is 83"`。★ 这和
   `physicsquestions` 是同一类问题（那里的金标是 `Can be one of [...]`）——
   **ReWOO 报的 `em` / `f1` 在这两个数据集上量的都是字符串,不是答案。**
   它真正有意义的那个数是 `acc`（LLM 判官）,而那个判官**没钉模型**
   （见 `benchmark/bigbench/bigbench.py` 模块头「偏离 ②」）。

2. **74 条太少,不能用来排名。** 准确率的 95% 置信区间在 n=74 时宽到
   ±11 个百分点 —— 比我们想量的差距还大。它的价值在**机制归因**
   （长文档 + 数值比较需要什么能力),不在「谁比谁高两个点」。

⇒ 所以这里**默认报 LLM 判官口径**,而不是 `em` —— 因为 `em` 在这上面
  必然接近 0,报它等于报了一个「所有方法都是 0」的列。
  ⚠️ 但判官必须**钉住模型并写进结果**,否则换一次判官数字就变。

## 判分口径

| 口径 | 判据 | 什么时候用 |
|---|---|---|
| `judge`（默认） | 一个**本地的**判定:金标里的关键事实是否出现在回答里 | 这个数据集唯一有意义的口径 |
| `em` / `f1` | ReWOO 的字符串口径 | **只用来对照**,说明它为什么没用 |

★ `judge` 的实现是**规则**（比对金标里的数字与 yes/no）,不是 LLM 调用 ——
因为判定模型的调用要走 `core/deciding.py` 那条有账目的路,
而判分器里偷偷发一次 LLM 请求会让账目对不上。
**规则判官是可复现的,LLM 判官要钉模型才有意义** —— 两者选前者。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.benchmark.rewoo_port import (
    REWOO_COMMIT,
    REWOO_SEED,
    REWOO_TARBALL,
    normalize_answer,
    rewoo_draw,
    token_f1,
)
from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

CSV_NAME = "SOTU_QA.csv"


@dataclass
class SotuQa:
    """ReWOO 的 curated 数据集（74 条）。"""

    name: str = "sotuqa"
    dataset_version: str = f"rewoo/SOTU_QA@rewoo-{REWOO_COMMIT[:8]}"
    dataset_dir: Path = field(default_factory=lambda: DATASET_DIR)
    with_tools: bool = False
    limit: int | None = None
    seed: int = REWOO_SEED
    
    #: ★ 默认 `recall` —— `em` 在这一列必然接近 0,
    #: 而 `judge` 只在 34/74 条上有可判要素（见 `_judge`）。
    headline: str = "recall"

    def _rows(self) -> list[dict[str, str]]:
        path = self.dataset_dir / self.name / CSV_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: 缺 {path}\n"
                f"  跑 `python3 -m experiments.scripts.datasets --fetch {self.name}`"
            )
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    # ── 协议 ────────────────────────────────────────────────

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        if split not in ("test", "train", "all"):
            raise ValueError(f"{self.name} 只有一个划分，收到 {split!r}")
        rows = self._rows()
        for i in rewoo_draw(len(rows), limit if limit is not None else self.limit, seed):
            row = rows[i]
            yield Task(
                task_id=f"{self.name}/{i}",
                prompt=self._prompt(row),
                gold=str(row["answer"]),
                # ★ 国情咨文全文**就是**这个数据集的证据,而它随数据一起在仓库里
                #   （`data/docs/state_of_the_union.txt`）——
                #   所以这里**能给真的 oracle**,不像 HotpotQA 需要检索工具。
                #   ⚠️ 但现在没把它接进来:那个文件在 tarball 里,
                #      `downloads()` 只抽了 CSV。要用得先把它加进 `files`。
                oracle_context=None,
                meta={"row": i, "answer_kind": "free",
                      # ★ 74 条这个事实要跟着结果走 —— 否则读表的人会以为
                      #   它和那些几千条的数据集一样能用来排名。
                      "n_total": len(rows), "small_n_warning": len(rows) < 100},
            )

    def _prompt(self, row: dict) -> str:
        """★ 原题没有输出契约（它是 `question` 一列裸文本）,所以给一条。

        ★ 第二条:**明说「答不完整没关系,给结论和依据」** —— 因为金标是
          一整句话,而如果契约要求「只给答案」,`judge` 口径就没东西可判。
        """
        return (
            f"{row['question']}\n\n"
            "Answer with the conclusion first, then the specific facts that "
            "support it. Keep it under three sentences."
        )

    # ── 判分 ────────────────────────────────────────────────

    def score_variants(self) -> dict[str, object]:
        """★ 四种口径并存 —— **报数时必须写明是哪个**。

        默认是 `recall`,不是 `judge`。理由见下面 `_judge` 的说明。

        `em` / `f1` 在这里是**对照**,不是成绩:金标是一整句话,
        它们必然接近 0。留着的目的是让「为什么不能用它们」看得见。
        """
        return {
            "recall": lambda t, tr: self._judge(t, tr, "recall"),
            "judge": lambda t, tr: self._judge(t, tr, "judge"),
            "em": lambda t, tr: self._judge(t, tr, "em"),
            "f1": lambda t, tr: self._judge(t, tr, "f1"),
        }

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        return self._judge(task, trajectory, self.headline)

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")
        want = str(task.gold)

        if mode == "em":
            hit = normalize_answer(answer) == normalize_answer(want)
            return Judgment(correct=hit, score=1.0 if hit else 0.0,
                            detail=f"em（对照口径）：{answer[:40]!r} vs {want[:40]!r}",
                            failure_class=None if hit else "wrong_answer")
        if mode == "f1":
            f1 = token_f1(answer, want)
            return Judgment(correct=f1 >= 0.5, score=f1,
                            detail=f"f1（对照口径）={f1:.3f}",
                            failure_class=None if f1 >= 0.5 else "wrong_answer")

        if mode == "recall":
            # ★★ **默认口径。** 判据是「金标里的实词有多少出现在回答里」。
            #
            #   为什么不用下面那个 `judge`:实测 **74 条里有 40 条（54%）
            #   既没有判定词也没有数字**（金标是
            #   `The new leader of the House Democrats is Hakeem Jeffries. He is known ...`
            #   这种整句话）。`judge` 对那些行**没有可判的要素**,
            #   于是无论回答什么都判对 —— 一个「任何答案都正确」的判分器。
            #
            #   实词回收率对全部 74 条都算得出,而且是**确定性的**（不调 LLM）。
            want_content = _content_tokens(want)
            if not want_content:
                return Judgment(correct=False, score=0.0,
                                detail="金标里没有实词，判不了", failure_class="scorer_error")
            got_content = _content_tokens(answer)
            hit_tokens = want_content & got_content
            recall = len(hit_tokens) / len(want_content)
            ok = recall >= RECALL_THRESHOLD
            missing = sorted(want_content - got_content)
            return Judgment(
                correct=ok, score=recall,
                detail=(f"recall={recall:.2f}（阈 {RECALL_THRESHOLD}）"
                        + (f"，缺 {missing[:4]}" if missing else "")),
                failure_class=None if ok else "wrong_answer",
            )

        # ── `judge`:严格规则口径 ─────────────────────────────
        # ★ 判据:**金标里的判定词和数字都要在回答里出现**。
        #   金标形如 `No, Kevin is 58 while Nancy is 83` ——
        #   判定词(Yes/No)+ 数字(58 / 83)就是它的全部信息量。
        #
        # ⚠️ **它只对一部分行有效。** 金标里没有判定词也没有数字时,
        #   这儿**没有可判的东西** —— 那时返回 `scorer_error`,
        #   **不是判对**。第一版就是判对,于是 54% 的行上任何回答都过。
        want_norm = normalize_answer(want)
        got_norm = normalize_answer(answer)

        verdict = _verdict(want_norm)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", want_norm))
        if verdict is None and not numbers:
            return Judgment(
                correct=False, score=0.0,
                detail=("judge：金标里既没有判定词也没有数字 —— "
                        "这个口径对这条没有可判的要素（用 `recall`）"),
                failure_class="scorer_error",
            )

        got_numbers = set(re.findall(r"\d+(?:\.\d+)?", got_norm))
        missing_numbers = sorted(numbers - got_numbers)
        verdict_ok = verdict is None or _verdict(got_norm) == verdict
        numbers_ok = not missing_numbers
        hit = verdict_ok and numbers_ok

        bits = []
        if verdict is not None:
            bits.append(f"判定词 {verdict} {'✓' if verdict_ok else '✗'}")
        if numbers:
            bits.append(f"数字 {len(numbers) - len(missing_numbers)}/{len(numbers)}"
                        + (f"（缺 {missing_numbers}）" if missing_numbers else ""))
        return Judgment(correct=hit, score=1.0 if hit else 0.0,
                        detail="judge：" + "; ".join(bits),
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
            kind="http",
            locator=REWOO_TARBALL,
            files=(CSV_NAME,),
            revision=REWOO_COMMIT,
            size_hint="~6 MB（整个 ReWOO 仓库；本数据集 22 KB）",
            note=("★ ReWOO 自带的 curated 数据集（74 条），全项目最小的一个。"
                  "★ 国情咨文全文 `data/docs/state_of_the_union.txt` 也在这个 tarball 里,"
                  "要接 oracle 就把它加进 `files`"),
        )]


#: 实词回收率的阈值。★ 0.6 是个**选择,不是真理** —— 所以它是个常量,
#: 换它要重跑一遍（`scripts/rescore.py` 可以对同一批轨迹换口径重判）。
RECALL_THRESHOLD = 0.6

#: 判实词时丢掉的词。**只丢功能词,不丢内容词** ——
#: 丢多了会把「答了一堆不相干的」也判过。
_STOP = frozenset("""a an the is are was were be been being of in on at to for from by with
and or but if then than that this these those it its he she they them his her their
as not no yes what which who whom whose when where why how all any both each few
more most other some such only own same so too very can will just should now""".split())


def _content_tokens(text: str) -> set[str]:
    """归一化之后的**实词**集合。数字也算实词。"""
    return {w for w in normalize_answer(text).split() if w not in _STOP and len(w) > 1}


def _verdict(text: str) -> str | None:
    """从归一化后的文本里取判定词。**取不到返回 `None`**（那就是没得判）。"""
    if re.search(r"\bno\b", text):
        return "no"
    if re.search(r"\byes\b", text):
        return "yes"
    return None


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="check", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)


__all__ = ["SotuQa", "CSV_NAME"]
