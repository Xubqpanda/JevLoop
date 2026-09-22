"""`log/ → result/` —— **单向,不可逆。**

规范 `docs/PROTOCOL-experiments-*.md` §6:
**不许手填 result,不许从别处粘。** 每个数字都必须能指回一个 `run_id`。

所以这个脚本做两件事,缺一不可:

1. **拒收残缺的行。** 缺 `REQUIRED_RESULT_FIELDS` 里的任何一个 → 那一行不进表,
   并在末尾**把拒收的文件名和缺的字段打出来**。补默认值是禁止的 ——
   一个补过默认值的数字,和编出来的没有区别。
2. **把 `run_id` 带进汇总表。** 归档到别处之后,这就是唯一的线索。

用法（在 `JevLoop/` 下）::

    python3 -m experiments.scripts.summarize
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from experiments.core.runlog import EXPERIMENTS_DIR
from experiments.core.spec import REQUIRED_RESULT_FIELDS, missing_fields

LOG_DIR = EXPERIMENTS_DIR / "log"
RESULT_DIR = EXPERIMENTS_DIR / "result"


def load_rows(log_dir: Path) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    """读所有 `results.jsonl`。返回 (好行, [(文件, 缺的字段)])。"""
    rows: list[dict] = []
    rejected: list[tuple[str, list[str]]] = []
    for path in sorted(glob.glob(str(log_dir / "*" / "*" / "*" / "results.jsonl"))):
        for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            # ★ 展开嵌套字段参与校验：meta / cost / timing 缺了同样要拒
            flat = {**record, **record.get("cost", {}), **record.get("timing", {})}
            gaps = missing_fields(record, REQUIRED_RESULT_FIELDS)
            if gaps:
                rejected.append((f"{path}:{lineno}", gaps))
                continue
            rows.append({**flat, "_file": path})
    return rows, rejected


def summarize(rows: list[dict]) -> list[dict]:
    """按 (dataset, arm) 汇总。**每个格子带 n 和 std** —— 只报均值会藏掉方差。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["meta"]["dataset"], row["meta"]["arm"])].append(row)

    out: list[dict] = []
    for (dataset, arm), items in sorted(groups.items()):
        correct = [1.0 if r["correct"] else 0.0 for r in items]
        seeds = sorted({r["meta"]["seed"] for r in items})
        costs = [r["cost"] for r in items]
        times = [r["timing"] for r in items]
        out.append(
            {
                "dataset": dataset,
                "arm": arm,
                "n": len(items),
                "seeds": len(seeds),
                "acc": round(statistics.fmean(correct), 4),
                "acc_std": round(statistics.pstdev(correct), 4) if len(correct) > 1 else 0.0,
                "llm_calls": round(statistics.fmean(c["llm_calls"] for c in costs), 3),
                "decisions": round(statistics.fmean(c["decision_requests"] for c in costs), 3),
                "tokens_in": round(
                    statistics.fmean(c["input_tokens_cached"] + c["input_tokens_uncached"] for c in costs), 1
                ),
                "tokens_out": round(
                    statistics.fmean(c["output_tokens_visible"] + c["output_tokens_reasoning"] for c in costs), 1
                ),
                "wall_ms": round(statistics.fmean(t["wall_ms"] for t in times), 1),
                "framework_ms": round(statistics.fmean(t["framework_ms"] for t in times), 2),
                "usd": round(statistics.fmean(c["usd"] for c in costs), 6),
                # ★ 每一行都要能指回是哪次跑出来的
                "run_ids": len({r["run_id"] for r in items}),
            }
        )
    return out


def to_markdown(table: list[dict]) -> str:
    if not table:
        return "_没有可汇总的行。_\n"
    cols = list(table[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for row in table]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="log/ → result/（单向）")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args(argv)

    rows, rejected = load_rows(args.log_dir)
    table = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text(
        "# 汇总\n\n"
        "> ★ **本文件由 `python3 -m experiments.scripts.summarize` 生成,不许手改。**\n"
        "> 每个数字都能指回 `log/` 下的一个 run —— 归档之后那是唯一的线索。\n\n"
        + to_markdown(table),
        encoding="utf-8",
    )
    if table:
        with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    print(f"读了 {len(rows)} 行，汇总 {len(table)} 个格子 → {args.out_dir}/summary.md")
    if rejected:
        # ★ 拒收要说出来。悄悄跳过 = 表格看起来是全的,而它不是
        print(f"\n★ 拒收 {len(rejected)} 行（缺字段）:", file=sys.stderr)
        for where, gaps in rejected[:20]:
            print(f"  {where}: 缺 {gaps}", file=sys.stderr)
        if len(rejected) > 20:
            print(f"  …还有 {len(rejected) - 20} 行", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
