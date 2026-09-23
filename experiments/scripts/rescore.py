"""给**已有的轨迹**换判分口径重判 —— **不重跑。**

## 这是从哪来的

**移植自 Inspect 的 `inspect score`**（<https://inspect.aisi.org.uk/tasks.html.md>,
「Re-scoring existing logs」）。**不依赖它**,照形状自己写一份 ——
理由同 `AGENTS.md` §8.7「移植而不引用」。

## 为什么这条值得单独做一个脚本

我们反复写过一句:**「换判分器之后数字不可比」** —— 因为换判分器要**重跑**,
而**重跑出来的不是同一批轨迹**（模型有随机性、网络有抖动、服务端版本会变）。

它把这件事降成:**同一批轨迹,换把尺子**。而我们正要做三件这样的事:

| 要做的事 | 没有它会怎样 |
|---|---|
| `gsm8k` 报 `strict` 和 `flexible` 两个口径 | 跑两遍,而两遍的轨迹不同 → 两个数的差**分不清是口径还是随机性** |
| BFCL 从「函数名判分」换到官方 `bfcl-eval` | 重跑一遍才算得出来 |
| 交付闸门的假拒 / 假放 | 要改判据就得重跑 |

## 规矩

1. **永远写新目录,绝不覆盖。** 原始日志是证据。
2. **来源必须记下来。** 新 run 的 `meta.json` 里带
   `rescore.source_run` / `source_events` / `variant` —— 「同一批轨迹、不同尺子」
   这句话要能被核对,不然它就是一句口号。
3. **对不上的题要说出来。** 数据集换了版本、抽样参数变了,都会有题对不上 ——
   **静默跳过会让表看起来是全的**。

用法（在 `JevLoop/` 下）::

    # 看看有哪些口径
    python3 -m experiments.scripts.rescore --list-variants gsm8k

    # 给最近一次 gsm8k/direct 的运行换 strict 口径重判
    python3 -m experiments.scripts.rescore --dataset gsm8k --arm direct --variant strict

    # 也可以直接指一个 run 目录
    python3 -m experiments.scripts.rescore log/gsm8k/direct/2026-09-22T03-00-00Z-seed0 --variant strict
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from experiments.core.events import AnswerEvent, from_json
from experiments.core.runlog import LOG_DIR, RunLog
from experiments.core.runner import Cell, _safe_score  # noqa: PLC2701 —— 判分口径要完全一致
from experiments.core.agent import action_tool
from experiments.core.types import Step, Trajectory


def find_run(spec: str) -> Path:
    """`spec` 可以是一个 run 目录,也可以是 `dataset/arm`（取最近一次）。"""
    p = Path(spec)
    if p.is_dir() and (p / "events.jsonl").exists():
        return p
    hits = sorted(glob.glob(str(LOG_DIR / spec / "*")), key=lambda d: Path(d).stat().st_mtime)
    if not hits:
        raise SystemExit(f"找不到运行目录: {spec!r}（既不是目录，也没有 log/{spec}/*）")
    return Path(hits[-1])


def read_events(run_dir: Path) -> list:
    out = []
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(from_json(line))
    return out


def rebuild_trajectory(arm: str, task_id: str, events: list) -> Trajectory:
    """从事件流里把判分需要的那部分重建出来。

    ★ **只需要三样:最终答案、工具调用的轨迹、有没有弃答。**
    这三样在事件里都有,所以重判不用碰任何原始产物 —— 这就是「日志自足」的意思。
    （`AnswerEvent` 是为此加的:在此之前答案只活在最后一个 `ModelCallEvent` 的文本里,
    而「哪一句算答案」是各臂自己的事,重判无从下手。）
    """
    answers = [e for e in events if isinstance(e, AnswerEvent)]
    tools = [e for e in events if e.type == "tool"]
    if not answers:
        # ★ 不猜。没有 answer 事件 = 这一轮的日志是老格式,直说
        raise SystemExit(
            f"{task_id}: 日志里没有 `answer` 事件 —— 这是旧格式的 run,无法重判。\n"
            f"  （`answer` 事件是写重判时才发现缺的,见 events.py 的说明）"
        )
    a = answers[-1]

    # ★★★ **工具调用的轨迹必须重建出来,不能留空。**
    #
    #   实测（2026-09-22）:`steps=()` 让 `bfcl-v3-multiple × react-typed`
    #   重判时报出 **「翻转 85」** —— 而 85 恰好是原来判对的条数。
    #   原因:BFCL 的判分器读的是
    #
    #       called = [s.action.name for s in trajectory.steps if is_tool_call(...)]
    #
    #   而 `steps=()` ⇒ `called = []` ⇒ **凡是有金标的题全判 `needs_tool_missed`**。
    #
    #   ★ 它**静默**:报出来是「翻转 85」,看起来像一个巨大的口径差异,
    #     **而不像一个坏掉的工具**。差点让我得出「换了判分器,85 条变了」这个错结论。
    #
    #   ★ 而这份文档的上一段**本来就写着**「只需要三样:最终答案、
    #     **工具调用的轨迹**、有没有弃答」—— **代码做的是另一回事。**
    #     文档和代码说的不一致时,相信代码在做什么,别相信它说要做什么。
    #
    #   ⚠️ 重建不出来的一样东西:**`__parse_error__` 那种伪步骤**。
    #     它不过 executor,所以**不发 `ToolCallEvent`** —— 事件里没有它。
    #     ⇒ 重判出来的轨迹里只有**真实调用**,而判分器要的正是这个
    #       （见 `core/types.py::is_tool_call`）。**这个「丢」是对的。**
    steps = tuple(
        Step(index=int(getattr(e, "step", i) or i),
             action=action_tool(e.name, dict(e.arguments or {})),
             observation=getattr(e, "observation", "") or "")
        for i, e in enumerate(tools)
    )

    return Trajectory(
        task_id=task_id,
        arm=arm,
        final_answer=a.text,
        escalated=a.escalated,
        error=a.error,
        usage=(),
        steps=steps,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="给已有的轨迹换判分口径重判（不重跑）")
    ap.add_argument("run", nargs="?", help="run 目录，或 `dataset/arm`（取最近一次）")
    ap.add_argument("--dataset", help="按 dataset 找最近一次运行")
    ap.add_argument("--arm", help="配 --dataset 用")
    ap.add_argument("--variant", default="default", help="判分口径名（见 --list-variants）")
    ap.add_argument("--list-variants", metavar="DATASET", help="列出某个数据集支持的口径")
    ap.add_argument("--dry-run", action="store_true", help="只算不落盘")
    args = ap.parse_args(argv)

    from experiments.scripts.run import BENCHMARK_FACTORIES

    if args.list_variants:
        factory = BENCHMARK_FACTORIES.get(args.list_variants)
        if factory is None:
            raise SystemExit(f"没有这个 benchmark: {args.list_variants!r}\n有: {sorted(BENCHMARK_FACTORIES)}")
        bench = factory()
        print(f"{args.list_variants} 的判分口径:")
        for name in bench.score_variants():
            mark = " ← 默认头条" if name == "default" else ""
            print(f"  {name}{mark}")
        return 0

    spec = args.run or (f"{args.dataset}/{args.arm}" if args.dataset and args.arm else None)
    if not spec:
        raise SystemExit("要给一个 run 目录，或同时给 --dataset 和 --arm")

    run_dir = find_run(spec)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cell = meta["cell"]
    dataset, arm, seed = cell["dataset"], cell["arm"], cell["seed"]

    factory = BENCHMARK_FACTORIES.get(dataset)
    if factory is None:
        raise SystemExit(f"日志说这是 {dataset!r}，但现在没有这个 loader（改名了？）")
    bench = factory()
    variants = bench.score_variants()
    if args.variant not in variants:
        raise SystemExit(f"没有口径 {args.variant!r}；{dataset} 支持 {sorted(variants)}")
    judge = variants[args.variant]

    # ★ 数据集必须能重建出**同一批题** —— split/seed 从 meta 来,task_count 就是当时的 limit
    #   （抽样是确定的:同一个 limit + seed 永远给同一批）
    tasks = {t.task_id: t for t in bench.tasks(split=meta["split"], limit=meta["task_count"], seed=seed)}
    events = read_events(run_dir)
    by_task: dict[str, list] = {}
    for e in events:
        by_task.setdefault(getattr(e, "task_id", ""), []).append(e)

    before = {r["task_id"]: r for r in
              (json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    flipped, unmatched, rescored = 0, [], []
    for task_id, old in before.items():
        task = tasks.get(task_id)
        if task is None:
            unmatched.append(task_id)
            continue
        traj = rebuild_trajectory(arm, task_id, by_task.get(task_id, []))
        judgment = judge(task, traj)
        was = bool(old["correct"])
        if was != judgment.correct:
            flipped += 1
        rescored.append((task_id, old, judgment))

    print(f"来源: {run_dir.relative_to(LOG_DIR)}")
    print(f"口径: {bench.headline_extract if dataset == 'gsm8k' else 'default'} → {args.variant}")
    print(f"  {len(before)} 条 → 重判 {len(rescored)}，翻转 {flipped}")
    if unmatched:
        # ★ 静默跳过会让表看起来是全的
        print(f"  ★ {len(unmatched)} 条对不上当前数据集（换版本了？）: {unmatched[:5]}", file=sys.stderr)

    if args.dry_run:
        print("  （--dry-run，没落盘）")
        return 0

    # ★ 写**新目录**,绝不覆盖 —— 原始日志是证据
    with RunLog(dataset, f"{arm}-rescore-{args.variant}", seed) as log:
        log.write_cmd(sys.argv)
        log.write_meta({
            **meta,
            "rescore": {
                "source_run": str(run_dir.relative_to(LOG_DIR)),
                # ★ 「同一批轨迹、不同尺子」这句话要能被核对
                "source_events": str((run_dir / "events.jsonl").relative_to(LOG_DIR)),
                "variant": args.variant,
                "flipped": flipped,
                "unmatched": unmatched,
            },
        })
        for task_id, old, judgment in rescored:
            log.append_jsonl("results.jsonl", {**old, "correct": judgment.correct,
                                               "score": judgment.score,
                                               "failure_class": judgment.failure_class or "none",
                                               "detail": judgment.detail})
        log.progress(len(rescored), len(before))
        print(f"  新 run: log/{log.run_id}")
        print("  ⚠️ 两个 run 的数字**不可直接混比** —— 它们用的是不同口径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
