"""名字 → 工厂。**一个普通字典,不做魔法。**

为什么不自动发现(扫目录、import 每个模块、抓子类):**一处坏的 import
会拖垮所有人的运行**。七个人并行时,`benchmark/` 下任何一个文件写上语法错,
全场的 `--benchmark <某个数据集>` 都会挂 —— 而那个人可能只是刚存了半行。

所以注册是**显式**的:谁要跑,谁 import。代价是每加一个数据集要动一行,
收益是**别人写坏的东西不会挡住你**。
"""

from __future__ import annotations

from typing import Callable

from experiments.core.agent import Agent
from experiments.core.bench import Benchmark

BenchmarkFactory = Callable[[], Benchmark]
AgentFactory = Callable[[], Agent]

BENCHMARKS: dict[str, BenchmarkFactory] = {}
AGENTS: dict[str, AgentFactory] = {}


def register_benchmark(name: str, factory: BenchmarkFactory) -> None:
    """★ `name` 必须和 `docs/PLAN-*.md` §0.7 的文件命名表一致 —— 别自己起名。

    重名直接炸:两个数据集叫同一个名字,会让 `log/<name>/` 混在一起,
    而那时候已经跑完一批了。
    """
    if name in BENCHMARKS:
        raise ValueError(f"benchmark 重名: {name}")
    BENCHMARKS[name] = factory


def register_agent(name: str, factory: AgentFactory) -> None:
    if name in AGENTS:
        raise ValueError(f"agent 重名: {name}")
    AGENTS[name] = factory


def known() -> str:
    """给 `--help` 和报错信息用。**空列表本身也是有用的信息。**"""
    return (
        f"benchmark: {sorted(BENCHMARKS) or '（还没接）'}\n"
        f"agent:     {sorted(AGENTS) or '（还没接）'}"
    )
