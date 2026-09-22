"""数据下载入口。**「怎么下载」的唯一权威在 loader 里,这个脚本只做展示和触发。**

用法（在 `JevLoop/` 下）::

    python3 -m experiments.scripts.datasets --list          # 看所有 loader 要什么
    python3 -m experiments.scripts.datasets --check         # 哪些还没下载
    python3 -m experiments.scripts.datasets --write         # 刷新 dataset/DOWNLOADS.md
    python3 -m experiments.scripts.datasets --fetch all     # 全部下载
    python3 -m experiments.scripts.datasets --fetch gsm8k

★ 索引是**生成的**。`dataset/DOWNLOADS.md` 不要手改 —— 权威是各 loader 的
`downloads()`,理由见 `core/download.py`。
"""

from __future__ import annotations

import argparse
import sys

from experiments.core.bench import Benchmark
from experiments.core.download import (
    DATASET_DIR,
    INDEX_NAME,
    DownloadSpec,
    fetch,
    is_present,
    render_index,
    use_local_data_dir,
)


def all_specs() -> list[DownloadSpec]:
    """把所有注册过的 benchmark 的下载声明收上来。

    ★ **只列有 loader 的数据集。** 没有 loader 的东西不进这张表 ——
    它还不参与实验,列上去只会让索引变成愿望清单。
    """
    # 这里的 import 就是注册动作本身（见 core/registry.py 为什么不做自动发现）
    from experiments.scripts.run import BENCHMARK_FACTORIES

    specs: list[DownloadSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for name, factory in sorted(BENCHMARK_FACTORIES.items()):
        bench: Benchmark = factory()
        declare = getattr(bench, "downloads", None)
        if declare is None:
            continue
        for spec in declare():
            key = (spec.dataset, spec.kind, spec.locator)
            if key in seen:
                continue
            seen.add(key)
            specs.append(spec)
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据下载索引与触发")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="打印所有下载声明")
    group.add_argument("--check", action="store_true", help="检查哪些还没下载")
    group.add_argument("--write", action="store_true", help=f"刷新 {DATASET_DIR.name}/{INDEX_NAME}")
    group.add_argument("--fetch", metavar="NAME", help="下载（数据集名，或 all）")
    args = parser.parse_args(argv)

    use_local_data_dir()
    specs = all_specs()

    if args.write:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        target = DATASET_DIR / INDEX_NAME
        target.write_text(render_index(specs), encoding="utf-8")
        print(f"写了 {target}（{len(specs)} 条声明）")
        return 0

    if args.list:
        if not specs:
            print("还没有任何 loader 声明下载来源。")
            return 0
        for s in specs:
            mark = "✓" if is_present(s) else "·"
            pin = s.revision if s.pinned else f"{s.revision}(未钉)"
            print(f"  {mark} {s.dataset:<18} {s.kind:<11} {s.locator:<52} {pin}")
        return 0

    if args.check:
        missing = [s for s in specs if not is_present(s)]
        for s in specs:
            state = "已有" if is_present(s) else ("**要现场建**" if s.kind == "build" else "缺")
            print(f"  {state:<10} {s.dataset:<18} {s.locator}")
        if missing:
            print(f"\n{len(missing)} 份没就位。跑 `--fetch all`（`build` 那类要手动）。")
        return 0

    target = args.fetch
    todo = specs if target == "all" else [s for s in specs if s.dataset == target]
    if not todo:
        print(f"没有这个数据集: {target!r}\n有的: {sorted({s.dataset for s in specs})}", file=sys.stderr)
        return 1
    for s in todo:
        if is_present(s):
            print(f"  已有 {s.dataset} ({s.locator})")
            continue
        print(f"  下载 {s.dataset} ({s.kind} {s.locator} …)")
        try:
            fetch(s)
            print("    完成")
        except Exception as exc:  # noqa: BLE001 —— 一份失败不该挡住其余的
            # ★ 不吞:说清是哪一份、什么原因。`build` 那类本来就会抛。
            print(f"    ★ 失败: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
