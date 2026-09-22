"""把 `Benchmark` × `Agent` 跑成一个矩阵。**这里不做任何领域判断。**

它只做四件事:建 log 目录、逐题跑、收账、落盘。
评分在 benchmark 里,决策在 agent 里,记账在 session 里 —— runner 只负责**顺序和边界**。

★ `framework_ms` 在这里算:`墙钟 − (模型 + 判定 + 工具)`。
不留这一行,就永远不知道差距里有多少是自己的开销。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from experiments.core.agent import Agent, AgentOutcome, Session
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
    """从逐条记录汇总。**每个数都按 PLAN §3.2 的口径拆开。**"""
    llm = [r for r in session.model_calls]
    decisions = session.decisions
    necessary, exploratory = session.executor.counts()
    batches = {d.batch for d in decisions}

    return Cost(
        llm_calls=len(llm),
        decision_requests=len(batches),
        questions_per_request=(len(decisions) / len(batches)) if batches else 0.0,
        tool_calls=necessary + exploratory,
        tool_calls_necessary=necessary,
        tool_calls_exploratory=exploratory,
        input_tokens_cached=sum(r.input_tokens_cached for r in llm),
        input_tokens_uncached=sum(r.input_tokens_uncached for r in llm),
        output_tokens_reasoning=sum(r.output_tokens_reasoning for r in llm),
        output_tokens_visible=sum(r.output_tokens_visible for r in llm),
        usd=sum(r.usd for r in llm),
    )


def aggregate_timing(session: Session, *, wall_ms: float) -> Timing:
    llm = session.model_calls
    model_ms = sum(r.handshake_ms + r.ttft_ms + r.after_ttft_ms for r in llm)
    decision_ms = sum(d.latency_ms for d in session.decisions)
    tool_ms = session.executor.total_ms()
    retry_ms = session.retry_ms()

    # ★ 框架自身开销 = 墙钟 − 其余全部。**它可以是负的**（并发/计时误差）,
    #   负的时候如实报负 —— 夹到 0 就把「我们的开销」这件事藏起来了。
    framework_ms = wall_ms - (model_ms + decision_ms + tool_ms + retry_ms)

    return Timing(
        wall_ms=wall_ms,
        model_handshake_ms=sum(r.handshake_ms for r in llm),
        model_ttft_ms=sum(r.ttft_ms for r in llm),
        model_after_ttft_ms=sum(r.after_ttft_ms for r in llm),
        decision_handshake_ms=0.0,  # 判定后端接进来时填（L2 接缝）
        decision_compute_ms=decision_ms,
        tool_ms=tool_ms,
        framework_ms=framework_ms,
        retry_ms=retry_ms,
        round_trips=len(llm) + len({d.batch for d in session.decisions}),
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

    with RunLog(cell.dataset, cell.arm, cell.seed, root=log_root) as log:
        log.write_cmd(argv or ("<in-process>",))
        results: list[Result] = []
        log.progress(0, len(tasks))

        for i, task in enumerate(tasks, start=1):
            tools = list(bench.tools())
            executor = ToolExecutor(tools, bench.tool_impls())
            session = Session(
                run_id=log.run_id, task=task, arm=cell.arm, tools=tools, executor=executor,
                model=model, max_steps=max_steps, temperature=temperature, max_tokens=max_tokens,
            )

            t0 = time.perf_counter()
            try:
                outcome: AgentOutcome = make_agent(task, tools).solve(session)
            except Exception as exc:  # noqa: BLE001 —— 一题挂掉不许拖垮整轮
                # ★ 记下异常类型和原文。**不吞**:trace 里留着,失败率进表。
                outcome = AgentOutcome(error=f"{type(exc).__name__}: {exc}")
                log.stop(f"task {task.task_id}: {outcome.error}")
            wall_ms = (time.perf_counter() - t0) * 1000

            trajectory = session.trajectory(outcome)
            judgment = _safe_score(bench, task, trajectory, outcome)
            log.progress(i, len(tasks))

            for record in session.decisions:
                log.append_jsonl("trace.jsonl", record)
            for record in session.usage_records():
                log.append_jsonl("usage.jsonl", record)

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
                    artifacts=("cmd.txt", "meta.json", "exit.json"),
                )
            )

        for result in results:
            log.append_jsonl("results.jsonl", result)

        log.write_meta(
            {
                "run_id": log.run_id,
                "cell": {"dataset": cell.dataset, "arm": cell.arm, "seed": cell.seed},
                "tasks": len(tasks),
                "commit": commit,
                "dirty": dirty,
                "archive": archive_hint(log.dir),
            }
        )
        return results


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
