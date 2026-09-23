"""`pickTool` 的帧证据 A/B —— **只重放判定，不跑基准**。

跑法（在 `JevLoop/` 下，需要 `.env` 里有 `TYPESAFE_API_KEY`）::

    python3 -m experiments.scripts.frame_ab          # 默认 45 题,4 个变体
    LIMIT_TASKS=100 python3 -m experiments.scripts.frame_ab

★ 为什么要有这个工具:基准的**总分那一格噪声 ±9**（§8.17）,而帧改动
（`remaining` 这类）的效应是 **±3** —— 拿总分去比,n=1 根本读不出来。
这个台子量的是**判定离门限多远**,那是可以逐条配对的数:
同一个判定点、同一个后端、只换帧,差多少就是多少。

★★ 而且它顺带钉住了一个陷阱:`Request.digest()` 覆盖帧 + 问题 + **选项**。
   只比 `frame_digest` 会把「换了候选集」误读成「后端随机」—— 见 `frame.py::Request.digest`。


为什么不跑基准:总分那一格噪声 ±9（§8.17），而这里要量的是
**判定离门限多远**,那是个可以做差、可以逐条配对的数。

三个变体,同一批请求:
  A 现状      task + last_result
  B +历史     A + already_done(做过什么)
  C +剩余     A + actions_left(还剩哪些工具没调)

判据:**margin = top - 0.6**。证据变厚如果有效,分布应当整体离开门限。
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.core.env import load_env

load_env()

from experiments.benchmark.bfcl import Bfcl  # noqa: E402
from experiments.core.deciding import (DEFAULT_JEV_URL, PINNED_JEV_MODEL,  # noqa: E402
                                       DecideRequest, HttpJevClient)
from experiments.core.frame import (AgentCtx, FrameField, FrameSpec,  # noqa: E402
                                    Question, Request, StepRecord,
                                    compile_frame, frame_for)
from experiments.jloop.typed import DONE, PICK_TOOL_ASK, _wire  # noqa: E402

#: 从哪一批记录里重放判定点。**必须是干净跑出来的那一批**（`dirty=False`）。
RUN = os.environ.get("RUN", str(REPO / "experiments/log/bfcl-v3-multiple/"
                                      "react-typed/2026-09-23T01-39-57Z-seed0"))
LIMIT_TASKS = int(os.environ.get("LIMIT_TASKS", "45"))
THRESHOLD = 0.6

base = frame_for("pickTool")
SPECS = {
    "A 现状": base,
    "B +历史": FrameSpec(
        node="pickTool",
        fields=base.fields + (FrameField("history", 300, "already_done", clip="list"),),
        excluded=tuple(x for x in base.excluded if x[0] != "history"),
    ),
    "C +剩余": FrameSpec(
        node="pickTool",
        fields=base.fields + (FrameField("remaining", 200, "actions_left", clip="list"),),
        excluded=base.excluded,
    ),
    "D 历史+剩余": FrameSpec(
        node="pickTool",
        fields=base.fields + (FrameField("history", 300, "already_done", clip="list"),
                              FrameField("remaining", 200, "actions_left", clip="list")),
        excluded=tuple(x for x in base.excluded if x[0] != "history"),
    ),
}


def main() -> int:
    bench = Bfcl(subset="v3-multiple")
    tasks = {t.task_id: t for t in bench.tasks(split="test", limit=None, seed=0)}
    impls = bench.tool_impls()
    stub = impls[next(iter(impls))]()          # BFCL 的桩文案,所有工具一样
    desc = {tid: {t.name: t.description for t in tk.tools} for tid, tk in tasks.items()}

    # 从记录里重放每道题**实际调过哪些工具**,决定每一步的候选
    called: dict[str, list[str]] = collections.defaultdict(list)
    for line in open(f"{RUN}/events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") == "tool":
            called[e["task_id"]].append(e["name"])

    client = HttpJevClient(DEFAULT_JEV_URL, os.environ["TYPESAFE_API_KEY"],
                           model=PINNED_JEV_MODEL)

    # 收集要重放的判定点:候选 ≥ 2 才是真的问过 pickTool
    points = []
    for tid in sorted(called)[:LIMIT_TASKS]:
        tools = [t.name for t in tasks[tid].tools]
        for k in range(len(called[tid])):
            done = called[tid][:k]
            left = [t for t in tools if t not in done]
            if len(left) < 2:
                continue
            points.append((tid, k, done, left))
    print(f"重放 {len(points)} 个 pickTool 判定点 × {len(SPECS)} 个变体 "
          f"= {len(points) * len(SPECS)} 次请求\n")

    rows = []
    for i, (tid, k, done, left) in enumerate(points, 1):
        task = tasks[tid]
        ctx = AgentCtx(task=task.prompt, last_result=stub)
        ctx.history = tuple(StepRecord(step=j, tool=n, input="", result=stub)
                            for j, n in enumerate(done))
        ctx.last_tool = done[-1] if done else ""
        ctx.remaining = tuple(left)
        opts = left + [DONE]
        crit = {o: desc[tid].get(o, "") for o in left}
        crit[DONE] = "已有足够证据回答任务,工具循环可以结束了"

        for name, spec in SPECS.items():
            frame = compile_frame(spec, ctx)
            q = Question(node="pickTool", kind="choice", ask=PICK_TOOL_ASK,
                         options=tuple(opts), threshold=THRESHOLD, criteria=crit)
            a = client.decide(DecideRequest(
                state={"frame": frame.render()},
                questions={"pickTool": _wire(q)})).answers["pickTool"]
            rows.append({"task": tid.split("/")[-1], "step": k, "variant": name,
                         "choice": a.choice, "top": round(a.top(), 4),
                         "req": Request(frame=frame, question=q).digest()})
        if i % 10 == 0:
            print(f"  ...{i}/{len(points)}")

    with open(os.environ.get("OUT", "/tmp/frame_ab_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    # ── 报告 ────────────────────────────────────────────────
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)

    print(f"\n{'变体':10} {'n':>4} {'答案=出口':>9} {'离门限中位':>10} "
          f"{'<0.15 占比':>10} {'top 中位':>9}")
    for name in SPECS:
        rs = by[name]
        margins = sorted(abs(r["top"] - THRESHOLD) for r in rs)
        near = sum(1 for m in margins if m < 0.15) / len(margins)
        tops = sorted(r["top"] for r in rs)
        stop = sum(1 for r in rs if r["choice"] == DONE)
        print(f"{name:10} {len(rs):>4} {stop:>9} {margins[len(margins)//2]:>10.3f} "
              f"{near:>9.0%} {tops[len(tops)//2]:>9.3f}")

    # 逐条配对:同一个判定点,变体之间答案变了吗
    print("\n逐条配对(与 A 比):")
    for name in SPECS:
        if name == "A 现状":
            continue
        a = {(r["task"], r["step"]): r for r in by["A 现状"]}
        b = {(r["task"], r["step"]): r for r in by[name]}
        flip = [k for k in a if k in b and a[k]["choice"] != b[k]["choice"]]
        noop = sum(1 for k in a if k in b and a[k]["req"] == b[k]["req"])
        print(f"  {name}: 答案变了 {len(flip)}/{len(a)} 条; "
              f"其中请求指纹完全相同的有 {noop} 条(应为 0)")
        for k in sorted(flip)[:6]:
            print(f"      {k[0]:14} step{k[1]}  {a[k]['choice']}({a[k]['top']}) "
                  f"-> {b[k]['choice']}({b[k]['top']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
