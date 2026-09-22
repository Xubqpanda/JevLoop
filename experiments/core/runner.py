"""把 `Benchmark` × `Agent` 跑成一个矩阵。**这里不做任何领域判断。**

它只做四件事:建 log 目录、逐题跑、收账、落盘。
评分在 benchmark 里,决策在 agent 里,记账在 session 里 —— runner 只负责**顺序和边界**。

★ `framework_ms` 在这里算:`墙钟 − (模型 + 判定 + 工具)`。
不留这一行,就永远不知道差距里有多少是自己的开销。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from experiments.core.agent import Agent, AgentOutcome, Session
from experiments.core.events import AnswerEvent
from experiments.core.bench import Benchmark
from experiments.core.models import ModelClient, describe
from experiments.core.runlog import EXPERIMENTS_DIR, RunLog, archive_hint, repo_commit
from experiments.core.spec import Cost, ModelRef, Result, RunMeta, Timing
from experiments.core.tools import ToolExecutor
from experiments.core.types import Judgment, Task, Trajectory

AgentFactory = Callable[[Task, list], Agent]
"""`(task, tools) -> Agent`。**每题一个新 agent** —— 免得上一题的状态漏进下一题。"""

NecessaryToolIds = set[str]
"""哪些工具调用算「必要」。不标就默认全部必要（`ToolExecutor.call` 的 `necessary`）。"""


@dataclass(frozen=True)
class Cell:
    """一个格子:`数据集 × arm × seed`。"""

    dataset: str
    arm: str
    seed: int


def aggregate_cost(session: Session) -> Cost:
    """从**事件流**汇总。**每个数都按 PLAN §3.2 的口径拆开。**

    ★ 输入是 `session.events` —— 一次调用一条记录。以前要 join 三个文件,
    现在读一处就够,而「token 和文本对不上」这类账目问题从结构上消失了。
    """
    llm = session.model_events()
    batches = session.decision_batches()
    necessary, exploratory = session.executor.counts()
    questions = sum(b.questions_in_batch for b in batches)

    return Cost(
        llm_calls=len(llm),
        decision_requests=len(batches),
        questions_per_request=(questions / len(batches)) if batches else 0.0,
        tool_calls=necessary + exploratory,
        tool_calls_necessary=necessary,
        tool_calls_exploratory=exploratory,
        input_tokens_cached=sum(e.usage.input_tokens_cache_read for e in llm),
        input_tokens_uncached=sum(e.usage.input_tokens_uncached for e in llm),
        output_tokens_reasoning=sum(e.usage.reasoning_tokens for e in llm),
        output_tokens_visible=sum(e.usage.output_tokens_visible for e in llm),
        usd=sum(e.usage.total_cost for e in llm),
    )


def aggregate_timing(session: Session, *, wall_ms: float) -> Timing:
    llm = session.model_events()
    model_ms = sum(e.timing.total_ms for e in llm)
    decision_ms = sum(b.latency_ms for b in session.decision_batches())
    tool_ms = session.executor.total_ms()
    retry_ms = session.retry_ms()

    # ★ 框架自身开销 = 墙钟 − 其余全部。**它可以是负的**（并发/计时误差）,
    #   负的时候如实报负 —— 夹到 0 就把「我们的开销」这件事藏起来了。
    framework_ms = wall_ms - (model_ms + decision_ms + tool_ms + retry_ms)

    return Timing(
        wall_ms=wall_ms,
        model_handshake_ms=sum(e.timing.handshake_ms for e in llm),
        model_ttft_ms=sum(e.timing.ttft_ms for e in llm),
        model_after_ttft_ms=sum(e.timing.after_ttft_ms for e in llm),
        decision_handshake_ms=0.0,  # 判定后端接进来时填（L2 接缝）
        decision_compute_ms=decision_ms,
        tool_ms=tool_ms,
        framework_ms=framework_ms,
        retry_ms=retry_ms,
        round_trips=len(llm) + len(session.decision_batches()),
    )


def run_cell(
    *,
    bench: Benchmark,
    make_agent: AgentFactory,
    cell: Cell,
    model: ModelClient,
    split: str = "test",
    limit: int | None = None,
    max_steps: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    region: str = "unknown",
    cold_start: bool = False,
    log_root: Path | None = None,
    argv: Sequence[str] = (),
) -> list[Result]:
    """跑一个格子。**一个格子一个 log 目录,永不覆盖。**"""
    tasks = list(bench.tasks(split=split, limit=limit, seed=cell.seed))
    commit, dirty = repo_commit(EXPERIMENTS_DIR.parent)

    # ★★★ **脏工作区要在开跑之前就说,不是在跑完之后记。**
    #
    #   实测（2026-09-22）:`dirty` 是写进 `meta.json` 的,而那份文件**跑完才写** ——
    #   于是「这一整批数字不可引用」这件事,代价**随运行时长增长**:
    #   一个 `--limit 300 × 6 臂` 的跑要 1.8 小时之后才告诉你它白跑了。
    #
    #   `repo_commit` 的文档早就写着「脏工作区跑出来的数字别人复现不了,
    #   连跑它的人自己都复现不了」——**项目知道这件事,但没有任何东西据此行动。**
    #   这和 §8.6 那条「要求写在文档里、没写在代码里」是同一个病。
    #
    #   ★ 同一个 commit 加 `dirty=True` 的两次跑**可以完全不同**:
    #   实测同一天两次 `bfcl × react-typed`,commit 都是 `5e9aee6b`、都 dirty,
    #   而一次 `framework_ms=-1670`、另一次 `=2` —— 中间的修复没提交。
    #   **从日志里分不出这两次。**
    if dirty:
        print(
            "\n" + "=" * 68 + "\n"
            f"⚠️  工作区是脏的（HEAD={commit[:8]}）—— 这一批**不可引用**\n"
            "    数字别人复现不了,连你自己也复现不了:同一个 commit 加 dirty,\n"
            "    两次跑可以完全不同。\n"
            "    → 提交之后再跑,或者明确接受这批只当**筛选**,不进表。\n"
            + "=" * 68 + "\n",
            file=sys.stderr,
        )

    with RunLog(cell.dataset, cell.arm, cell.seed, root=log_root) as log:
        log.write_cmd(argv or ("<in-process>",))
        results: list[Result] = []
        log.progress(0, len(tasks))

        # ★★ **跑到一半树变了要说出来。**
        #
        #   实测（2026-09-22）:一次 `--limit 300 × 7 臂` 的后台跑跨了 **5 个 commit** ——
        #   因为我在它跑的时候一直在提交。结果那一行的七条臂**各在一个代码版本上**,
        #   而每一格单独看都是 `commits=1`,**表上完全看不出来**。
        #
        #   「跑之前干净」已经有警告了（见上面那段）;这一段管的是**跑的中途变了** ——
        #   而那种情况更隐蔽:开始是干净的,所以第一道警告不会响。
        start_commit = commit
        mid_run_change: str | None = None

        for i, task in enumerate(tasks, start=1):
            # ★★ **在有状态的环境里,benchmark 必须知道「现在是哪个任务」。**
            #
            #   实测（2026-09-22,接 ALFWorld 时):那类数据集的每一题是一个
            #   **可交互环境**（TextWorld),`tool_impls()` 要绑定到「这一题的那个
            #   环境实例」上。而 `tool_impls()` 的签名里没有任务 ——
            #   于是只有两条路:靠**调用顺序**碰巧对上（`tasks()` 和
            #   `tool_impls()` 交替调用）,或者把这个钩子显式化。
            #
            #   顺序那条现在确实成立,但它是**隐式约定**:哪天有人把
            #   `bench.tools()` 提到循环外、或者并行跑两题,它就静默错了 ——
            #   而错法是「A 题的动作打在 B 题的环境上」,分数照出。
            #
            #   ★ 所以钩子是**可选**的（`getattr`),静态数据集一行都不用改。
            on_task = getattr(bench, "on_task", None)
            if on_task is not None:
                on_task(task)
            # ★ 题级工具集优先 —— 见 types.Task.tools 的说明
            tools = list(task.tools) if task.tools else list(bench.tools())
            executor = ToolExecutor(tools, bench.tool_impls())
            session = Session(
                run_id=log.run_id, task=task, arm=cell.arm, tools=tools, executor=executor,
                model=model, max_steps=max_steps, temperature=temperature, max_tokens=max_tokens,
            )

            # 每题查一次「树还是不是那个」。★ 用 `repo_commit` 而不是逐文件 stat ——
            #   它读的是 git 的状态,和我们开头那次判据**同一个**,不会两套口径。
            if mid_run_change is None and i % 10 == 1:
                now_commit, now_dirty = repo_commit(EXPERIMENTS_DIR.parent)
                if now_dirty or now_commit != start_commit:
                    mid_run_change = (
                        f"第 {i} 题时工作区已经不是开始时那个了"
                        f"（开始 {start_commit[:8]} 干净,现在 {now_commit[:8]}"
                        f"{' 脏' if now_dirty else ''}）"
                    )
                    print(
                        "\n" + "!" * 68 + "\n"
                        f"⚠️  {mid_run_change}\n"
                        "    → 这一批的**不同题跑在不同代码版本上**,不能当受控比较。\n"
                        "      跑完之前不要改仓库。\n" + "!" * 68 + "\n",
                        file=sys.stderr,
                    )

            agent = make_agent(task, tools)
            if getattr(agent, "needs_success_signal", False):
                # ★ 只有声明要的臂才接得上 —— 见 core/agent.py 的说明
                session.check_answer = _make_checker(bench, task)

            t0 = time.perf_counter()
            try:
                outcome: AgentOutcome = agent.solve(session)
            except Exception as exc:  # noqa: BLE001 —— 一题挂掉不许拖垮整轮
                # ★ 记下异常类型和原文。**不吞**:trace 里留着,失败率进表。
                outcome = AgentOutcome(error=f"{type(exc).__name__}: {exc}")
                log.stop(f"task {task.task_id}: {outcome.error}")
            wall_ms = (time.perf_counter() - t0) * 1000

            # ★ 先把答案落成事件 —— 没有它,日志不自足,重判无从下手
            session.events.append(AnswerEvent(
                run_id=log.run_id, task_id=task.task_id, step=len(outcome.steps),
                text=outcome.final_answer, escalated=outcome.escalated, error=outcome.error,
                steps=len(outcome.steps), working_start=session.clock.elapsed(),
            ))

            # ★★ **先封口,再建轨迹。** 这两行的顺序是个真 bug,修于 2026-09-22。
            #
            #   原来 `trajectory()` 在 `finish_events()` **之前** ——
            #   而 `trajectory()` 里的 `decision_records()` 是从**事件流**里读的,
            #   最后一批判定那时还躺在 `_batch_decisions` 里没进事件流。
            #
            #   所以**每一题的最后一批判定都缺席了**,而它恰恰是决策性的那批
            #   （收尾那一步:判「不用再调工具了」的那次）。
            #   `finish_events` 的文档自己就写着「不封口最后一批就会丢 ——
            #   而它常常正是决策性的那批」—— 写那句警告和写这个调用顺序的是同一份代码。
            #
            #   ★ 这和第 10 轮那些事故是**同一个形状**:东西是对的,
            #   但**没有任何东西检查「它有没有被送到该到的地方」**。
            session.finish_events()
            trajectory = session.trajectory(outcome)
            judgment = _safe_score(bench, task, trajectory, outcome)
            log.progress(i, len(tasks))

            # ★ **一条事件流,不是一个调用摊在三张表里。**
            #   照 Inspect 的 transcript 模型做的（见 core/events.py 头部）。
            for event in session.events:
                log.append_jsonl("events.jsonl", event)

            results.append(
                Result(
                    run_id=log.run_id,
                    meta=RunMeta(
                        dataset=cell.dataset,
                        dataset_version=bench.dataset_version,
                        split=split,
                        task_count=len(tasks),
                        arm=cell.arm,
                        generator=ModelRef(**describe(model)),
                        decider=ModelRef(id="none", version="n/a", provider="n/a"),
                        thinking_budget="n/a",
                        temperature=temperature,
                        top_p=1.0,
                        max_tokens=max_tokens,
                        seed=cell.seed,
                        prompt_hash=_prompt_hash(bench, cell.arm),
                        commit=commit,
                        dirty=dirty,
                        region=region,
                        cold_start=cold_start,
                        max_steps=max_steps,
                        timeout_s=0.0,
                    ),
                    task_id=task.task_id,
                    correct=judgment.correct,
                    score=judgment.score,
                    steps=len(trajectory.steps),
                    first_divergence_step=_first_divergence(bench, task, trajectory),
                    escalated=trajectory.escalated,
                    gate_false_reject=bool(judgment.failure_class == "gate_rejected_correct"),
                    gate_false_deny=bool(judgment.failure_class == "gate_passed_wrong"),
                    failure_class=judgment.failure_class or ("none" if judgment.correct else "unclassified"),
                    cost=aggregate_cost(session),
                    timing=aggregate_timing(session, wall_ms=wall_ms),
                    artifacts=("cmd.txt", "meta.json", "exit.json", "results.jsonl",
                               "events.jsonl"),
                )
            )

        for result in results:
            log.append_jsonl("results.jsonl", result)

        # ★ **完整的那套 RunMeta 要写进 `meta.json`**,不是一个薄壳。
        #
        #   之前这里只有 run_id / cell / tasks / commit / dirty / archive,
        #   而完整的 RunMeta（含 `split` / `task_count` / 温度 / 版本 / 地区）
        #   只重复躺在每一行 `results.jsonl` 里 —— 于是 `rescore.py` 想重建
        #   **同一批题**时拿不到 `split`,直接 KeyError（实测）。
        #
        #   这条也是 PROTOCOL §3.2 要求的:「`meta.json` 装 PROTOCOL §3.7 那一套,一个不少」。
        #   一行的身份不该靠「去第一行结果里翻」才能知道。
        first = results[0].meta if results else None
        log.write_meta(
            {
                "run_id": log.run_id,
                "cell": {"dataset": cell.dataset, "arm": cell.arm, "seed": cell.seed},
                "archive": archive_hint(log.dir),
                # 完整 RunMeta 摊平 —— 这就是「这次运行是什么」
                **({} if first is None else {
                    "dataset": first.dataset,
                    "dataset_version": first.dataset_version,
                    "split": first.split,
                    "task_count": first.task_count,
                    "arm": first.arm,
                    "generator": describe(model),
                    "temperature": first.temperature,
                    "max_tokens": first.max_tokens,
                    "seed": first.seed,
                    "prompt_hash": first.prompt_hash,
                    "commit": first.commit,
                    "dirty": first.dirty,
                    "region": first.region,
                    "cold_start": first.cold_start,
                    "max_steps": first.max_steps,
                }),
            }
        )
        # ★ 中途变过就记进 `exit.json` 的 `stop_reason` —— 归档之后那是唯一线索。
        if mid_run_change:
            log.stop(f"★ 这一批跨了多个代码版本：{mid_run_change}")
        return results


def _make_checker(bench: Benchmark, task: Task):
    """把 benchmark 的 `check()` 包成一个布尔判据。**没实现就返回 None。**"""
    hook = getattr(bench, "check", None)
    if hook is None:
        return None
    return lambda answer: bool(hook(task, answer))


def _safe_score(bench: Benchmark, task: Task, trajectory: Trajectory, outcome: AgentOutcome) -> Judgment:
    """评分失败不许把整轮带走 —— 但也**不许当成对**。

    当成对是最坏的一种处理:它把「判分器坏了」变成「我们赢了」。
    """
    if outcome.error:
        return Judgment(correct=False, score=0.0, detail=outcome.error, failure_class="agent_error")
    try:
        return bench.score(task, trajectory)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return Judgment(correct=False, score=0.0, detail=f"scorer: {exc}", failure_class="scorer_error")


def _first_divergence(bench: Benchmark, task: Task, trajectory: Trajectory) -> int:
    """第一次走偏在第几步。benchmark 没给 oracle 就返回 -1（**不是 0**）。"""
    hook = getattr(bench, "first_divergence_step", None)
    if hook is None:
        return -1
    return int(hook(task, trajectory))


def _prompt_hash(bench: Benchmark, arm: str) -> str:
    import hashlib

    blob = f"{bench.name}:{bench.dataset_version}:{arm}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_matrix(
    *,
    bench: Benchmark,
    arms: dict[str, AgentFactory],
    seeds: Iterable[int],
    model: ModelClient,
    **kwargs,
) -> list[Result]:
    """`benchmark × arm × seed`。**逐格串行** —— 并行留给调用方,免得两处都在管资源。"""
    out: list[Result] = []
    for arm_name, factory in arms.items():
        for seed in seeds:
            out.extend(
                run_cell(
                    bench=bench, make_agent=factory,
                    cell=Cell(dataset=bench.name, arm=arm_name, seed=seed),
                    model=model, **kwargs,
                )
            )
    return out
