"""HotpotQA —— **改编自 ReWOO 的 `DataLoader.load_hotpot_qa` + `Evaluator`。**

## 出处

| | |
|---|---|
| **论文** | Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018. arXiv:1809.09600 |
| **仓库** | `github.com/hotpotqa/hotpot` |
| **ReWOO 的用法** | `utils/DataLoader.py::load_hotpot_qa`、`utils/Evaluator.py` |
| **数据** | HF `hotpot_qa`，config **`fullwiki`**（ReWOO 用的就是它）|

★ **ReWOO 的头条就是这个数据集**（README:「5× token efficiency and 4% accuracy
improvement on HotpotQA」）—— 所以这一格的判分口径**必须和它一致**,
否则那个「4%」没有对照物。

## ★ 比 ReWOO 多做的一件事:`oracle_context`

ReWOO 的 loader 只取 `["question", "answer"]`,把 `context` 和
`supporting_facts` **丢掉**了。我们留着,因为 `core/types.Task.oracle_context`
是「**direct + 金标证据**」那一臂的输入 —— 而那一臂是用来回答
「这个数据集难在生成还是难在找证据」的（`docs/PLAN-*.md` §1）。

`fullwiki` 的 `context` 是**检索回来的十段**,不是金标证据。真正的金标是
`supporting_facts` 指的那几句 —— 所以 `oracle_context` 从那里取,
**不是**把十段全塞进去（塞全文等于把「找证据」这件事也送掉了）。

⚠️ 一条要说清的:**ReWOO 没有给这个 oracle**。所以
「`direct-oracle`」那一行的数**没有 ReWOO 的对照物**,它只是我们的诊断臂。

## ★★ 两处必须记住的偏离

**① 工具集。** ReWOO 给的是 `Google` / `Wikipedia` / `WolframAlpha` /
`Calculator` / `LLM`（`run_eval.py` 的 `--toolset` 默认值），而 HotpotQA 走
`ReactBase`。★ 注意 `fullwiki` 这个名字的含义:它**不把 context 放进 prompt**,
agent 必须自己去查 —— 所以**工具是这个数据集的一部分,不是可选项**。
我们默认不给工具（那三个要付费服务),于是我们的 agent 只能靠参数化知识答,
**这会让 HotpotQA 上的比较失去意义**。⇒ 这一格在补上可复现的检索工具之前
**不许进对比表**,而要排在 `strategyqa` 那三个后面 —— 那三个至少不依赖检索。

**② 崩溃不计入。** 同 `benchmark/bigbench/bigbench.py` 模块头「偏离 ③」:
ReWOO 异常时塞全 NaN,汇总用 `np.nanmean` **跳过**。我们计入。

## 判分

`em` 与 `f1`，来自 `benchmark/rewoo_port.py`（那是 ReWOO 的逐字那一份）。
**`acc`（LLM 判官）不报** —— 它的判官没钉模型,理由同 bigbench 模块头「偏离 ②」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.benchmark.official_scorers import (
    MissingEvidence,
    hotpot_exact_match,
    hotpot_f1_score,
    hotpot_joint,
    hotpot_update_sp,
)
from experiments.benchmark.rewoo_port import REWOO_SEED, rewoo_draw
from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

#: HF 上的仓库 id 与 config。★ config 必须是 `fullwiki` —— ReWOO 用的就是它,
#: 换成 `distractor` 就是**另一个数据集**（它把十段里的两段换成金标段）。
DATASET_ID = "hotpot_qa"
CONFIG = "fullwiki"

#: ★★★ **不要照抄 ReWOO 的 `type="test"`。**
#:
#: ReWOO 的 `DataLoader.load_hotpot_qa` 默认 `type="test"`，而 `run_eval.py`
#: 从不覆盖它。**实测（2026-09-22，HF 镜像）:那个划分里 7,405 条
#: 一条答案都没有** —— `answer` 全是 `None`。官方的 test 标注是withhold 的,
#: 留给排行榜。
#:
#: 所以:
#:
#: - 我们**默认 `validation`**（7,405 条,答案齐全）—— 那也是文献里报数用的划分。
#: - `test` 仍可显式传,但 `_rows()` 会**当场炸**,不会拿 `None` 当金标打分。
#:
#: ⚠️ 这条同时是个**关于 ReWOO 的疑问**,记在这里别忘:它的默认值跑不起来
#: （`normalize_answer(None)` 会 `AttributeError`）。所以它表里 HotpotQA 那个数
#: 要么是拿 `validation` 跑的,要么它当年的 `datasets` 版本给了 test 标签。
#: **在弄清之前,那个「4%」不能当作同一批题上的对照。**
REWOO_SPLIT = "validation"


@dataclass
class HotpotQa:
    """HotpotQA(fullwiki)。"""

    name: str = "hotpotqa"
    dataset_version: str = field(init=False, default="unpinned")
    hf_home: Path = field(default_factory=lambda: DATASET_DIR / "hf")
    #: ★ 默认**不给工具**,理由见模块头「偏离 ①」。**这一条不解决不许进对比表。**
    with_tools: bool = False
    limit: int | None = None
    seed: int = REWOO_SEED

    def _rows(self, split: str) -> list[dict]:
        import os

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset  # 只在真要拿数据时 import

        ds = load_dataset(DATASET_ID, CONFIG, split=split)
        self.dataset_version = f"{DATASET_ID}:{CONFIG}@{getattr(ds, '_fingerprint', 'unpinned')}"
        return [dict(row) for row in ds]

    # ── 协议 ────────────────────────────────────────────────

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """★ 抽样用 ReWOO 那一次（`benchmark/rewoo_port.py::rewoo_draw`）。

        ★ `split` 默认走 ReWOO 的 `test`。**别的划分要显式传** ——
          `train` 有 90,447 条,误用会让一次筛选跑上一整天。
        """
        rows = self._rows(split or REWOO_SPLIT)
        # ★ 划分级检查:逐行那道守卫只看得见被抽到的那几行（见函数说明）
        _assert_split_has_gold(rows, who=self.name, split=split or REWOO_SPLIT,
                              pick=lambda r: r.get("answer") if r.get("answer") not in (None, "") else None)
        for i in rewoo_draw(len(rows), limit if limit is not None else self.limit, seed):
            row = rows[i]
            # ★★ **没有金标就不许出题。** `test` 划分的 `answer` 是 `None`,
            #   而 `str(None)` 是字符串 `"None"` —— 一个**看起来正常的金标**。
            #   照它打分会让整个数据集变成「模型有没有答出 `None` 这个词」。
            if row.get("answer") in (None, ""):
                raise ValueError(
                    f"{self.name}: {split or REWOO_SPLIT} 划分里第 {i} 条没有金标"
                    f"（answer={row.get('answer')!r}）。\n"
                    f"  → 官方的 test 标注是 withhold 的,用 `--split validation`"
                )
            yield Task(
                task_id=f"{self.name}/{row.get('id') or i}",
                prompt=self._prompt(row),
                gold=str(row["answer"]),
                # ★ 金标证据:从 `supporting_facts` 指的那几句取,不是十段全文。
                oracle_context=self._oracle(row),
                meta={
                    "row": i,
                    # ★ 这两条留着,因为「答得对不对」和「难在哪」是两件事 ——
                    #   `type` 是 HotpotQA 自己的题型标注(comparison / bridge),
                    #   而 comparison 那一类的答案常常就是 `yes` / `no`。
                    "type": row.get("type"),
                    "level": row.get("level"),
                    # ★ 答案可能是 `yes`/`no` —— 那种题是 `noul` 的天然形状,
                    #   类型化的臂据此决定用 `noul` 还是自由生成。
                    "answer_kind": ("binary" if str(row["answer"]).lower() in ("yes", "no")
                                    else "free"),
                },
            )

    def _prompt(self, row: dict) -> str:
        """HotpotQA 的原文只有问题,没有输出契约 —— **所以要给一条。**

        ★ 为什么不像 `strategy_qa` 那样「原样返回」:那个的 input 自带
          `Answer with 'Yes' or 'No':`,这个不自带。**「不加第二条」的前提是
          第一条存在** —— 不加区分地一律不加,等于让自由文本任务没有契约。
        """
        return (
            f"{row['question']}\n\n"
            "Give the answer as a short phrase. For yes/no questions answer "
            "`yes` or `no`. Do not explain."
        )

    def _oracle(self, row: dict) -> str | None:
        """金标证据 = `supporting_facts` 指向的那些句子。

        ★ 找不到就返回 `None`,**不退回全文** —— 把十段塞进去等于把「找证据」
          这件事也送掉了,而那正是这一臂要量的东西。
        """
        facts = row.get("supporting_facts") or {}
        titles = list(facts.get("title") or [])
        sent_ids = list(facts.get("sent_id") or [])
        if not titles:
            return None
        ctx = row.get("context") or {}
        ctx_titles = list(ctx.get("title") or [])
        ctx_sents = list(ctx.get("sentences") or [])

        picked: list[str] = []
        for title, sid in zip(titles, sent_ids):
            if title not in ctx_titles:
                continue
            sents = ctx_sents[ctx_titles.index(title)]
            if 0 <= int(sid) < len(sents):
                picked.append(f"[{title}] {sents[int(sid)]}")
        return "\n".join(picked) if picked else None

    # ── 判分 ────────────────────────────────────────────────

    def score_variants(self) -> dict[str, object]:
        """★ 三个口径,而且**它们不是一回事**:

        | 口径 | 是什么 | 能不能算 |
        |---|---|---|
        | `em` / `f1` | **answer-only** —— 正文里常引的那两个 | ✅ |
        | `joint` | **`joint_em = em × sp_em`** —— **排行榜报的是这个** | ❌ 缺证据字段 |

        ★★ 两者**差很远**,而表上看不出区别 —— 所以报 `em` 时必须写明是 answer-only。

        ⚠️ `joint` 会抛 `MissingEvidence`:它要模型输出 supporting facts,
          而我们的答案契约里只有最终答案。**这不能用近似值糊过去** ——
          那会让一个接口缺口看起来像一个算法读数。
        """
        return {
            "em": lambda t, tr: self._judge(t, tr, "em"),
            "f1": lambda t, tr: self._judge(t, tr, "f1"),
            "joint": lambda t, tr: self._judge(t, tr, "joint"),
        }

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        return self._judge(task, trajectory, "em")

    def _judge(self, task: Task, trajectory: Trajectory, mode: str) -> Judgment:
        answer = (trajectory.final_answer or "").strip()
        if not answer:
            return Judgment(correct=False, score=0.0,
                            detail="没有给出答案", failure_class="no_answer")
        want = str(task.gold)
        # ★ 金标是 yes/no 时取第一个 token —— 模型爱写「Yes, because ...」。
        got = answer.split()[0] if want.lower() in ("yes", "no") else answer

        if mode == "joint":
            raise MissingEvidence(
                "HotpotQA 的 joint_em 需要模型输出 supporting facts —— "
                "当前答案契约里没有这个字段。\n"
                "  → 要算它得先让答案包含证据（接口改动）,不是打分器的改动"
            )

        # ★ 用**官方**的两个函数（`hotpot_evaluate_v1.py`）,逐字那一份。
        em = hotpot_exact_match(got, want)
        f1, prec, recall = hotpot_f1_score(got, want)
        if mode == "em":
            return Judgment(correct=bool(em), score=em,
                            detail=f"em（**answer-only**,非 joint）：{got[:40]!r} vs {want[:40]!r}",
                            failure_class=None if em else "wrong_answer")
        return Judgment(correct=f1 >= 0.5, score=f1,
                        detail=f"f1（answer-only）={f1:.3f} prec={prec:.3f} rec={recall:.3f}",
                        failure_class=None if f1 >= 0.5 else "wrong_answer")

    def check(self, task: Task, answer: str) -> bool:
        """给 Reflexion 的 Evaluator 用。**它吃到金标信号。**"""
        return self._judge(task, _traj(answer), "em").correct

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
            size_hint="~1 GB（90,447 + 7,405 × 2）",
            note=("★ config 必须是 `fullwiki` —— ReWOO 用的就是它,"
                  "换成 `distractor` 就是另一个数据集。"
                  "`revision` 未钉,已用 fingerprint 记进 dataset_version"),
        )]


def _traj(answer: str) -> Trajectory:
    return Trajectory(task_id="t", arm="check", steps=(), final_answer=answer,
                      decisions=(), usage=(), escalated=False, error=None)


__all__ = ["HotpotQa", "DATASET_ID", "CONFIG"]

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
