"""★ **协议一:`Benchmark`。**

一个数据集要接进来,只需要回答三个问题:**有哪些题**、**给哪些工具**、**怎么判对**。

★ 工具归 benchmark,不归 baseline —— `docs/PLAN-*.md` §2 要求所有 arm
「同一批工具、同一份工具描述」。把定义权放在这一侧,这条要求就是**结构性**的。

★ 评分归 benchmark,不归 runner —— 每个数据集的官方判据都不一样
（状态检查 / 状态 diff / 逐动作校验 / 精确匹配）,**用统一评分器就是换判分器,
而换判分器 Acc 就不可比**。所以 `score()` 是 benchmark 的责任,而且
**实现时应当尽量调官方评分器**,不要重写。
"""

from __future__ import annotations

from typing import Callable, Iterator, Protocol, Sequence

from experiments.core.download import DownloadSpec
from experiments.core.types import Judgment, Task, Tool, Trajectory


class Benchmark(Protocol):
    """实现这四个成员即可接入。

    `name` 必须与 `docs/PLAN-*.md` §0.7 的清单一致 —— 它是 log 目录名和
    result 表里的主键,**不许自己起名**。
    """

    name: str
    """数据集标识。同时是 `log/<name>/` 和 `result/` 里的那一列。"""

    dataset_version: str
    """★ 版本必须逐字对上 `MANIFEST-datasets-*.md`。写 `unpinned` 也是信息,但别省。"""

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """产出题目。

        ★★ **`Task.prompt` 由 benchmark 决定,而且它是完整的任务陈述 + 输出契约。**

        分工是这样切的（这一条踩过才定下来）:

        | 层 | 归谁 | 例 |
        |---|---|---|
        | 任务陈述 + **输出契约** | **benchmark** | 「解这道题，最终答案给一个数」/ 「判 SUPPORTS / REFUTES / NOT ENOUGH INFO」|
        | **交互协议** | **baseline** | ReAct 的 `Thought/Action/Observation` 块、plan-then-execute 的计划要求 |

        **baseline 不许改任务陈述，只能加自己的协议块。**
        各臂自己写一句「答案格式」= 把**提示词工程**混进方法比较,
        而各臂该只差在**交互**上。

        - `limit` 是**筛选阶段**用的:先跑 300 条看分数,再决定要不要全量。
          **抽样必须用传进来的 `seed`**,否则「我们跑的是哪 300 条」说不清。
        - `Task.gold` 只给评分器看。**实现时注意别把它泄进 prompt。**
        """
        ...

    def tools(self) -> Sequence[Tool]:
        """这一批题可用的工具。**所有 baseline 拿到的是同一个列表。**

        ★ 如果**每题的工具不同**（BFCL 就是），把工具挂在 `Task.tools` 上，
        这里返回全集或空 —— runner 会优先用题级的那个。
        全局列表会让模型去调一个这道题根本没给它的函数。
        """
        ...

    def downloads(self) -> Sequence["DownloadSpec"]:
        """★ **这个 loader 需要的数据从哪来。**

        权威就住在这里 —— 不在文档里。`scripts/datasets.py --write` 会把所有
        loader 的声明汇总成 `dataset/DOWNLOADS.md`,**那份文件是生成的,不许手改**。
        两边各写一遍就会分叉,而分叉的后果是别人照文档下载、拿到的和我们对不上。

        `build` 那一类（Wikipedia 索引）**要人确认**,不会自动跑。
        """
        ...

    def tool_impls(self) -> dict[str, Callable[..., str]]:
        """`tools()` 里每个名字的实现。

        ★ 名字对不上会在 `ToolExecutor` 构造时就炸 —— **声明了却没实现**是个
        必须在开跑前发现的错误,不是运行时才撞的墙。
        """
        ...

    def check(self, task: Task, answer: str) -> bool:
        """**可选。** 只回答「这个答案对不对」,不产出完整 Judgment。

        为什么单独有这么一条:Reflexion 的 Evaluator 要的是**环境给的成败信号**
        （原文就是这么做的）。让 agent 去调完整的 `score()` 会把它和评分管线绑死,
        而它只需要一个布尔值。

        ★ **实现了它,就等于允许某些臂吃到金标信号。** 谁吃了,谁的名字里必须写清楚
        （见 `baseline/reflexion.py`）—— 一个吃了答案的臂不能和没吃的同列一张表。
        """
        ...

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        """判对错。

        尽量调**数据集官方的评分器**,不要自己重写一遍。
        **重写 = 换判分器,而换判分器之后的准确率不可比。**

        `Judgment.failure_class` 要按 `docs/PLAN-*.md` §3.8 那套词表填,
        **不许只报对错** —— 失败分布是结果的一部分。
        """
        ...


class Unscored(Benchmark):
    """**筛选阶段的 benchmark**:只出题,不评分。

    `direct` 那一轮只需要「模型答了什么」和「金标答案是什么」,官方评分器
    往往要装一大堆依赖。所以第一段用这个,分数由 `scripts/` 里的临时比对给,
    等确定要留下这个数据集,**再去接官方评分器**（`docs/PLAN-*.md` §0.9 的两段式）。
    """

    name = "unscored"
    dataset_version = "unpinned"

    def tools(self) -> Sequence[Tool]:
        return []

    def downloads(self) -> Sequence["DownloadSpec"]:
        """★ **这个 loader 需要的数据从哪来。**

        权威就住在这里 —— 不在文档里。`scripts/datasets.py --write` 会把所有
        loader 的声明汇总成 `dataset/DOWNLOADS.md`,**那份文件是生成的,不许手改**。
        两边各写一遍就会分叉,而分叉的后果是别人照文档下载、拿到的和我们对不上。

        `build` 那一类（Wikipedia 索引）**要人确认**,不会自动跑。
        """
        ...

    def tool_impls(self) -> dict[str, Callable[..., str]]:
        return {}

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        raise NotImplementedError(
            "筛选阶段不评分 —— 走 scripts/screen.py;要正式评分请接官方评分器"
        )
