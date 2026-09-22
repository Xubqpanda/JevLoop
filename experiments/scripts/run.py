"""入口:**跑一个格子**（数据集 × arm × seed），或一整行。

用法（在 `JevLoop/` 下）::

    # 自检：离线,不用 key 不用网
    python3 -m experiments.scripts.run --benchmark toy --agent direct --seed 0 --model offline

    # 真跑 direct（筛选阶段要的那一个数）
    python3 -m experiments.scripts.run --benchmark <数据集名> --agent direct --seed 0 \\
        --model deepseek-chat --base-url https://api.deepseek.com/v1 --limit 300

**它只做分发**:把名字变成对象,交给 `runner`。不做任何领域判断。
"""

from __future__ import annotations

import argparse
import os
import sys

from experiments.core.agent import Agent
from experiments.core.models import CallableModel, Message, ModelClient, OpenAICompatModel
from experiments.core.registry import AGENTS, BENCHMARKS, known
from experiments.core.runner import Cell, run_cell
from experiments.core.types import Task, Tool

# 注册表是**显式**的：谁要跑，谁在这里 import。见 core/registry.py 的说明。
from experiments.benchmark import toy  # noqa: F401  —— 自检用
from experiments.baseline import act as act_baseline  # noqa: F401
from experiments.baseline import direct as direct_baseline  # noqa: F401
from experiments.baseline import react as react_baseline  # noqa: F401


def offline_demo_model() -> CallableModel:
    """自检用的假模型:**只知道 `CAPITALS` 里前一半的国家**,而且**会说 ReAct 格式**。

    它刻意答错一半,这样「判对」和「判错」两条路都会走到 ——
    一个全对的假模型验不出 `failure_class` 有没有被写进结果。

    ★ 它还会看 prompt 里要的是哪种格式:循环臂（act/react）要求 `Action:` 时,
    它就按 ReAct 的写法回 —— 这样离线自检才真的走到
    「解析 → 调工具 → 看观察 → finish」这条完整路径,
    而不是一步就弃答。（后者也测得到,但那是另一件事。）
    """
    known_half = set(sorted(toy.CAPITALS)[: len(toy.CAPITALS) // 2])

    def country_in(text: str) -> str | None:
        for country in toy.CAPITALS:
            if f"capital of {country}" in text:
                return country
        return None

    def responder(messages: list[Message]) -> str:
        text = messages[-1].content
        country = country_in(text)
        if country is None:
            return "I don't know."

        wants_react = "Action:" in text and "finish[" in text
        if not wants_react:
            # direct 那一臂:直接给答案
            return toy.CAPITALS[country] if country in known_half else "I don't know."

        # 循环臂:第一轮查工具,看到观察后收尾
        if "Observation:" not in text:
            return f"Thought: I should look up {country}.\nAction: lookup_capital[{country}]"
        return f"Thought: I have the answer.\nAction: finish[{toy.CAPITALS[country]}]"

    return CallableModel(responder, model_id="offline-demo")


def build_model(args: argparse.Namespace) -> ModelClient:
    if args.model == "offline":
        print("⚠️  --model offline 是**自检用**的假模型,跑出来的分数没有意义。", file=sys.stderr)
        return offline_demo_model()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "缺 API key：给 --api-key，或设 DEEPSEEK_API_KEY / OPENAI_API_KEY。\n"
            "（想离线验接口就用 --model offline）"
        )
    return OpenAICompatModel(
        base_url=args.base_url,
        api_key=api_key,
        model_id=args.model,
        model_version=args.model_version,
        provider=args.provider,
    )


# arm 名 → 构造器。**名字必须和 docs/PLAN-*.md 的 baseline 清单一致。**
_BUILTIN_ARMS = {
    "direct": direct_baseline.Direct,
    "direct-oracle": lambda: direct_baseline.Direct(with_evidence=True),
    "act": act_baseline.Act,
    "react": react_baseline.ReAct,
}


def resolve_agent(name: str) -> "type[Agent] | callable":
    if name in _BUILTIN_ARMS:
        return _BUILTIN_ARMS[name]
    factory = AGENTS.get(name)
    if factory is None:
        raise SystemExit(f"没有这个 agent: {name!r}\n{known()}")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑一个格子:benchmark × agent × seed")
    parser.add_argument("--benchmark", required=True, help=f"数据集名。已注册: {sorted(BENCHMARKS) or '无'}")
    parser.add_argument("--agent", required=True, help="arm 名，如 direct / react / jevloop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（筛选阶段用）")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--region", default=os.environ.get("JEV_REGION", "unknown"))
    parser.add_argument("--cold-start", action="store_true", help="这一轮包含冷连接（DNS/TCP/TLS）")
    # 模型
    parser.add_argument("--model", default="deepseek-chat", help="模型 id，或 offline（自检）")
    parser.add_argument("--model-version", default="unpinned")
    parser.add_argument("--provider", default="openai-compat")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    bench_factory = BENCHMARKS.get(args.benchmark)
    if bench_factory is None:
        # toy 也走注册表，免得「有的能跑有的不能」两套规则
        if args.benchmark == toy.ToyCapitals.name:
            bench_factory = toy.ToyCapitals
        else:
            raise SystemExit(f"没有这个 benchmark: {args.benchmark!r}\n{known()}")
    bench = bench_factory()

    model = build_model(args)
    agent_factory = resolve_agent(args.agent)

    def make_agent(task: Task, tools: list[Tool]) -> Agent:
        return agent_factory()

    results = run_cell(
        bench=bench,
        make_agent=make_agent,
        cell=Cell(dataset=bench.name, arm=args.agent, seed=args.seed),
        model=model,
        split=args.split,
        limit=args.limit,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        region=args.region,
        cold_start=args.cold_start,
        argv=sys.argv,
    )

    ok = sum(1 for r in results if r.correct)
    print(f"\n{bench.name} × {args.agent} × seed{args.seed}")
    print(f"  {ok}/{len(results)} 对   落盘: experiments/log/")
    if results:
        t = results[0].timing
        print(f"  墙钟 {t.wall_ms:.0f}ms · 框架自身 {t.framework_ms:.0f}ms · 往返 {t.round_trips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
