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


def load_rows(log_dir: Path, *, allow_dirty: bool = False
              ) -> tuple[list[dict], list[tuple[str, list[str]]], list[str]]:
    """读所有 `results.jsonl`。返回 `(好行, 拒收的, 因脏被排除的 run)`。

    ★★ **脏工作区跑出来的行不进表。**

    `repo_commit` 的文档写着「脏工作区跑出来的数字别人复现不了,连跑它的人
    自己都复现不了」—— 而实测同一天两次 `bfcl × react-typed`,
    `commit` 都是 `5e9aee6b`、都是 `dirty=True`,一次 `framework_ms=-1670`、
    另一次 `=2`,**从日志里完全分不出这两次**。

    所以 `dirty` 不是一条备注,是**引用资格**:一个从 commit 还原不出来的代码
    状态,产出的数不能和别的数放在一张表里平均。
    """
    rows: list[dict] = []
    rejected: list[tuple[str, list[str]]] = []
    dirty_runs: list[str] = []
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
            if record.get("meta", {}).get("dirty"):
                dirty_runs.append(record["run_id"])
                if not allow_dirty:
                    continue
            rows.append({**flat, "_file": path})
    return rows, rejected, sorted(set(dirty_runs))


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
                # ★★ **列出 id,不是数个数。**
                #   第一版这里写的是 `len({...})`,于是这一列的值是 `2` ——
                #   读表的人拿不到任何线索,而**这一列存在的全部理由**
                #   就是「归档之后那是唯一的线索」（本文件的头一句）。
                #   一个数个数把它变成了一个看起来正常的空壳。
                "runs": " ".join(sorted({r["run_id"].split("/")[-1] for r in items})),
                # ★ 一个格子跨了几个 commit。>1 就是在**把不同代码版本平均**
                "commits": len({r["meta"].get("commit") for r in items}),
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
    parser.add_argument("--allow-dirty", action="store_true",
                        help="把脏工作区跑出来的行也算进来（**开发时用,别用来出表**）")
    args = parser.parse_args(argv)

    rows, rejected, dirty_runs = load_rows(args.log_dir, allow_dirty=args.allow_dirty)
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
    if dirty_runs and not args.allow_dirty:
        # ★ 静默排除 = 表格看起来是全的,而它不是
        print(f"\n★ 排除 {len(dirty_runs)} 个**脏工作区**的 run（不可引用）:", file=sys.stderr)
        for rid in dirty_runs[:10]:
            print(f"    {rid}", file=sys.stderr)
        print("    要包括它们就加 --allow-dirty —— 但那一批的数字不该进表。", file=sys.stderr)

    mixed = [t for t in table if t.get("commits", 1) > 1]
    if mixed:
        print(f"\n★ {len(mixed)} 个格子跨了多个 commit —— 那是**把不同代码版本平均**:",
              file=sys.stderr)
        for t in mixed[:10]:
            print(f"    {t['dataset']} × {t['arm']}: {t['commits']} 个 commit", file=sys.stderr)
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
