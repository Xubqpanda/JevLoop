"""玩具数据集 —— **接口的自检**,不是实验的一部分。

它存在的唯一理由:**在没有 key、没有网、没有数据集**的机器上,
把 `Benchmark × Agent → log/ → result/` 整条链路跑通一遍。
新人可以照它写第一个真 loader。

它刻意覆盖了三种情况,因为接口最容易在这三处漏字段:
- 模型**答对**的题
- 模型**答错**的题（`failure_class` 那条路要走到）
- 带 `oracle_context` 的题（`direct + 金标证据` 那一臂要读到它）

★ 写真 loader 时**照抄这个形状,别照抄这里的 `CAPITALS`**。
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

from experiments.core.types import Judgment, Task, Tool, Trajectory

# 一半是常见国家（模型多半知道），一半是刻意的冷门 —— 让「对」和「错」都出现
CAPITALS: dict[str, str] = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Peru": "Lima",
    "Kyrgyzstan": "Bishkek",
    "Vanuatu": "Port Vila",
    "Suriname": "Paramaribo",
    "Bhutan": "Thimphu",
    "Eswatini": "Mbabane",
}

LOOKUP_TOOL = Tool(
    name="lookup_capital",
    description="查一个国家的首都。参数是国家名。",
    parameters={
        "type": "object",
        "properties": {"country": {"type": "string", "enum": sorted(CAPITALS)}},
        "required": ["country"],
    },
)


class ToyCapitals:
    """最小可跑的 benchmark。**示例,不要拿去发论文。**"""

    name = "toy"
    dataset_version = "toy-v1"

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        if split not in ("test", "train"):
            raise ValueError(f"toy 只有 test / train 两个划分，收到 {split!r}")
        # ★ 抽样必须用传进来的 seed —— 否则「我们跑的是哪几条」说不清。
        countries = sorted(CAPITALS)
        if split == "train":
            countries = countries[: max(1, len(countries) // 2)]
        if limit is not None:
            # 固定种子轮转抽样：同一个 seed 永远得到同一批题
            start = seed % len(countries)
            countries = [countries[(start + i) % len(countries)] for i in range(min(limit, len(countries)))]
        for country in countries:
            capital = CAPITALS[country]
            yield Task(
                task_id=f"toy/{country}",
                prompt=f"What is the capital of {country}? Answer with the city name only.",
                gold=capital,
                # 金标证据：`direct + 金标证据` 那一臂读它，`direct` 读不到
                oracle_context=f"Reference: the capital of {country} is {capital}.",
                meta={"country": country},
            )

    def tools(self) -> Sequence[Tool]:
        return [LOOKUP_TOOL]

    def tool_impls(self) -> dict[str, Callable[..., str]]:
        return {"lookup_capital": lambda country: CAPITALS.get(str(country), "(not found)")}

    def check(self, task: Task, answer: str) -> bool:
        """只回答对不对。Reflexion 的 Evaluator 要的就是这一个布尔值。"""
        return _normalize(answer) == _normalize(str(task.gold))

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        got = _normalize(trajectory.final_answer or "")
        want = _normalize(str(task.gold))
        if not got:
            return Judgment(
                correct=False, score=0.0,
                detail="没有给出答案", failure_class="no_answer",
            )
        if got == want:
            return Judgment(correct=True, score=1.0, detail=f"{got} == {want}")
        return Judgment(
            correct=False, score=0.0,
            detail=f"答了 {got!r}，期望 {want!r}",
            failure_class="wrong_answer",
        )


def _normalize(text: str) -> str:
    """大小写、标点、多余空白都不该影响判分。

    ⚠️ 真 benchmark **不要自己写这个** —— 用官方评分器。
    这里写是因为 toy 没有官方评分器,而且它要能在离线机器上跑。
    """
    keep = [c for c in text.strip().lower() if c.isalnum() or c.isspace()]
    return " ".join("".join(keep).split())
