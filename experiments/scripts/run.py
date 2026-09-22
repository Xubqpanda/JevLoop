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
from pathlib import Path

from experiments.core.agent import Agent
from experiments.core.env import load_env
from experiments.core.deciding import (
    DEFAULT_JEV_URL,
    PINNED_JEV_MODEL,
    DecisionClient,
    FallbackClient,
    HttpJevClient,
    MockClient,
)
from experiments.core.models import CallableModel, Message, ModelClient, OpenAICompatModel
from experiments.core.registry import AGENTS, BENCHMARKS, known
from experiments.core.runner import Cell, run_cell
from experiments.core.types import Task, Tool

# 注册表是**显式**的：谁要跑，谁在这里 import。见 core/registry.py 的说明。
from experiments.benchmark import bfcl as bfcl_bench  # noqa: F401
# ★ 三个 BigBench 任务**共用一份 ReWOO tarball** —— 一次下载抽三个 CSV。
from experiments.benchmark import bigbench as bigbench_bench  # noqa: F401
from experiments.benchmark import alfworld as alfworld_bench  # noqa: F401
from experiments.benchmark import fever as fever_bench  # noqa: F401
from experiments.benchmark import gsm8k as gsm8k_bench  # noqa: F401
from experiments.benchmark import hotpotqa as hotpotqa_bench  # noqa: F401
from experiments.benchmark import sotuqa as sotuqa_bench  # noqa: F401
from experiments.benchmark import tau2bench as tau2_bench  # noqa: F401
from experiments.benchmark import triviaqa as triviaqa_bench  # noqa: F401
from experiments.benchmark import toy  # noqa: F401  —— 自检用
from experiments.baseline import act as act_baseline  # noqa: F401
from experiments.baseline import direct as direct_baseline  # noqa: F401
from experiments.baseline import plan_then_execute as pte_baseline  # noqa: F401
from experiments.baseline import react as react_baseline  # noqa: F401
from experiments.baseline import reflexion as reflexion_baseline  # noqa: F401
from experiments.baseline import rewoo as rewoo_baseline  # noqa: F401


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
            # ★ 顺序有意义:Solver 的 `# Evidence` 块里也含 `#E1 =`,
            #   先判蓝图会把 Solver 误认成 Planner —— 实测踩过。
            #
            # ReWOO 的 Solver:证据已在 prompt 里,直接给答案
            if "# Evidence" in text:
                return toy.CAPITALS[country]
            # ReWOO 的 Planner:要的是蓝图
            if "Devise a plan" in text:
                return (
                    f"Plan: Look up the capital of {country}.\n"
                    f"#E1 = lookup_capital[{country}]\n"
                    f"Plan: Read the capital off the evidence.\n"
                    f"#E2 = LLM[What is #E1?]"
                )
            # plan-then-execute 的 Planner:编号列表
            if "Break the task into" in text:
                return f"1. Look up the capital of {country}.\n2. Report it."
            # plan-then-execute 的执行段（它带 `# Plan`,且带 Action 格式——
            # 一般走不到这里,留着是为了完整）
            if "# Plan" in text:
                return toy.CAPITALS[country]
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


# ★ **benchmark 的显式注册表。** 不做自动发现（见 core/registry.py 的说明）:
#   七个人并行时,`benchmark/` 下任何一个文件写上语法错,自动发现会让全场一起挂。
#   加一个数据集就在这里加一行,顺便把它加进 docs/PLAN-*.md §0.7 的命名表。
#
# ★ 放在**模块级**而不是 main() 里 —— `scripts/datasets.py` 要读它来汇总
#   各 loader 的下载声明。藏在 main() 里就导不出来（早先就是这么写的）。
BENCHMARK_FACTORIES: dict[str, Callable[[], object]] = {
    toy.ToyCapitals.name: toy.ToyCapitals,
    gsm8k_bench.Gsm8k.name: gsm8k_bench.Gsm8k,
    # ReWOO 那一组（改编自它的 DataLoader,见 benchmark/bigbench/bigbench.py）
    bigbench_bench.StrategyQa.name: bigbench_bench.StrategyQa,
    bigbench_bench.SportsUnderstanding.name: bigbench_bench.SportsUnderstanding,
    bigbench_bench.PhysicsQuestions.name: bigbench_bench.PhysicsQuestions,
    # ReWOO 另外两个 —— 需要工具才有意义,见各自文件头的「偏离 ①」
    hotpotqa_bench.HotpotQa.name: hotpotqa_bench.HotpotQa,
    triviaqa_bench.TriviaQa.name: triviaqa_bench.TriviaQa,
    fever_bench.Fever.name: fever_bench.Fever,
    sotuqa_bench.SotuQa.name: sotuqa_bench.SotuQa,
    # ★ 交互式环境:需要 textworld,而且要用 `on_task` 绑定每题的环境
    alfworld_bench.AlfWorld.name: alfworld_bench.AlfWorld,
    # ★ 两方任务:还需要一个 LLM 用户模拟器（见它文件头的说明）
    tau2_bench.Tau2Bench.name: tau2_bench.Tau2Bench,
}
# BFCL 一个数据集两个子集,各自是独立的 `log/<name>/` 目录
for _sub in bfcl_bench.SUBSETS:
    BENCHMARK_FACTORIES[f"bfcl-{_sub}"] = (lambda s=_sub: bfcl_bench.Bfcl(subset=s))
del _sub


def build_decider(args: argparse.Namespace) -> DecisionClient:
    """**判定后端**要单独解析,不和生成模型合成一个（§8.9）。

    两者问的不是同一个问题:`resolve` 生成模型问「你想用哪个 LLM」,
    这里问「哪里有可用的判定模型」。合成一个「配置」反而说不清,
    而且判定后端**永远有兜底** —— 生成模型缺 key 就该报错,不猜。

    - `mock`（默认）:保守答案,**离线可跑**。§8.6 要它让 policy 自己走到 `escalate`。
    - `http`:`POST /v1/systemone`。挂了就降到 mock,**每一级降级都报一次**（§8.10）。
    """
    if args.decider == "mock":
        return MockClient()

    if args.decider == "http":
        # ★ 和 TS 侧 `backends.ts` 读同一个变量名 —— 两边读不同的名字,
        #   就是一个「配了一边另一边没生效」的坑。
        key = args.decider_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit(
                "缺判定后端的 key：给 --decider-key，或设 TYPESAFE_API_KEY。\n"
                "（想离线验接口就用 --decider mock —— 那是默认值）"
            )
        return FallbackClient([
            HttpJevClient(args.decider_url, key, model=args.decider_model),
            MockClient(),
        ])

    raise SystemExit(f"没有这个判定后端: {args.decider!r}（有 mock / http）")


# arm 名 → 构造器。**名字必须和 docs/PLAN-*.md 的 baseline 清单一致。**
_BUILTIN_ARMS = {
    "direct": direct_baseline.Direct,
    "direct-oracle": lambda: direct_baseline.Direct(with_evidence=True),
    "act": act_baseline.Act,
    "react": react_baseline.ReAct,
    "rewoo": rewoo_baseline.ReWOO,
    "plan-then-execute": pte_baseline.PlanThenExecute,
    "reflexion": reflexion_baseline.Reflexion,
    "reflexion-selfeval": lambda: reflexion_baseline.Reflexion(uses_success_signal=False),
}


def typed_arms(decider: DecisionClient) -> dict[str, Callable[[], Agent]]:
    """**换了决策者的那几格** —— `docs/PLAN-*.md` 表 2 的右列。

    命名跟着现成的 `<arm>-<变体>`（`reflexion-selfeval` 就是这个形状）,
    **不新造一套规则**。每一格和左列那个臂共用同一个 `run_loop`、
    同一个 `build_prompt`、同一批工具 —— **唯一不同的就是决策者**。

    ⚠️ 现在只有 `react` 和 `act` 两格。**不是漏了,是另外几个还不成立**:

    - `plan-then-execute × typed`:它的计划藏在 `view.prompt` 的 preamble 里,
      而类型化那一路不看 `prompt`（它用 `task_prompt` 自己拼）→ **计划会丢**。
      要么把计划放进 view,要么不列这一格。**列上去会跑出一个看起来能跑、
      其实没在做同一件事的臂** —— 那比缺一格糟。
    - `rewoo × typed`:Worker 根本没有决策可换（`NOTES-*.md` §2.10.1）。
    - `jevloop`:我们自己的完整臂,它有自己的循环,不是「react 换个控制器」。
    """
    from experiments.jloop.typed import TypedController

    return {
        "react-typed": lambda: react_baseline.ReAct(controller=TypedController(decider)),
        "act-typed": lambda: act_baseline.Act(controller=TypedController(decider)),
    }


def resolve_agent(name: str, decider: DecisionClient | None = None) -> "type[Agent] | callable":
    if name in _BUILTIN_ARMS:
        return _BUILTIN_ARMS[name]
    if decider is not None and name in typed_arms(decider):
        return typed_arms(decider)[name]
    factory = AGENTS.get(name)
    if factory is None:
        raise SystemExit(f"没有这个 agent: {name!r}\\n{known()}")
    return factory


def main(argv: list[str] | None = None) -> int:
    # ★★ **先读 `.env`,再解析参数。**
    #
    #   实测:Python 侧原来没有加载器,只读 `os.environ` —— 而 key 就躺在
    #   仓库根的 `.env` 里。于是 `--model deepseek-chat` 报「缺 API key」,
    #   把人指向「去设一个环境变量」,而真正该做的是**读仓库自己那份**。
    #
    #   ★ 顺序有要求:`--decider-url` 之类的默认值要从环境取,
    #     所以这一步必须在 `parse_args` **之前** —— TS 侧踩过同一个坑
    #     （`resolveProvider()` 在 `loadEnv()` 之前调用,读到的永远是空）。
    env = load_env(Path(__file__).resolve().parents[1])
    if env.skipped:
        # ★ 认不出来的行**要说出来**（§8.10）—— 静默跳过会让人以为 key 设上了
        print(f"⚠️  {env.path} 里有 {len(env.skipped)} 行认不出来，已跳过：",
              file=sys.stderr)
        for line in env.skipped:
            print(f"     {line}", file=sys.stderr)

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
    # ── 判定后端 —— **和生成模型分开**（§8.9）────────────────────
    parser.add_argument("--decider", default="mock", choices=("mock", "http"),
                        help="判定后端。mock = 保守答案、离线可跑（默认）")
    # ★ 默认值从环境取 —— 这就是 `load_env()` 必须在 `parse_args()` **之前**的原因。
    #   写死默认值的话,`.env` 里那份配置永远读不到（TS 侧踩过同一个坑:
    #   `resolveProvider()` 在 `loadEnv()` 之前调用,读到的永远是空）。
    parser.add_argument("--decider-url", default=os.environ.get("JEVOS_JEV_URL")
                        or DEFAULT_JEV_URL, help="--decider http 时的 base url")
    parser.add_argument("--decider-key", default=None,
                        help="判定后端的 key。缺省读 TYPESAFE_API_KEY（**不进仓库**）")
    parser.add_argument("--decider-model", default=PINNED_JEV_MODEL,
                        help=f"钉住版本,**不要用别名**（默认 {PINNED_JEV_MODEL}）")
    args = parser.parse_args(argv)


    bench_factory = BENCHMARKS.get(args.benchmark) or BENCHMARK_FACTORIES.get(args.benchmark)
    if bench_factory is None:
        raise SystemExit(f"没有这个 benchmark: {args.benchmark!r}\n{known()}")
    bench = bench_factory()

    model = build_model(args)
    # ★ 判定后端**只在真要用的时候才建** —— `--decider http` 缺 key 要报错,
    #   但一个跑 `direct` 的人不该被这个报错拦住。
    needs_decider = args.agent.endswith("-typed")
    decider = build_decider(args) if needs_decider else None
    agent_factory = resolve_agent(args.agent, decider)

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
