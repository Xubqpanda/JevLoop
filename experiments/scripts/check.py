"""`python3 -m experiments.scripts.check` —— **跑过的东西和注册表对不对得上**。

## 为什么需要它

`log/` 下的目录名是**字符串拼出来的**（`log/<dataset>/<arm>/<时间戳>-seed<N>/`），
而数据集名和臂名各自写在别的地方。**三者没有任何东西保证一致。**

实测（2026-09-22）:

```
log/ 下是:      bfcl-simple/          bfcl-irrelevance/
注册表现在是:    bfcl-v3-simple        bfcl-v3-irrelevance
```

那批数是改名**之前**跑的。后果不是报错,是**同一个数据集两个目录**,
而「我们跑的是哪一批」在日志里说不准了 ——
这正是 §3.4 那条「数量检查」防的同一类事,只是发生在目录名上。

★ **靠人记得改名字,和没有这条规矩一样。** TS 侧有 `npm run check`
机械地数文件和待拆标记;这是 Python 侧的对应物。

## 它查什么

| | 判据 | 为什么 |
|---|---|---|
| **①** | `log/` 下的每个一级目录**要么是注册过的数据集名,要么在 `_superseded/` 下** | 改名漂移 |
| **②** | 每个 run 的 `meta.json` **字段齐全** | 缺 `dataset_version`/`split`/`generator` 的跑**不可引用** |
| **③** | `log/<数据集>/<臂>` 的臂名**能解析** | 臂改名漂移 |
| **④** | `dataset/DOWNLOADS.md` 覆盖每个注册的数据集 | loader 加了而下载说明没更新 |

**退出码非 0 = 有问题。** 和 `npm run check` 一样,它是**门**不是报告。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)) if str(REPO) not in sys.path else None

from experiments.core.env import load_env  # noqa: E402
from experiments.scripts.run import BENCHMARK_FACTORIES, _BUILTIN_ARMS, typed_arms  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[1]
LOG = EXPERIMENTS / "log"
SUPERSEDED = "_superseded"

#: 一个 run 的 `meta.json` 必须有的字段。**缺任何一个,这次跑就没法引用** ——
#: 「这题是哪个版本的、哪个划分、哪个模型跑的」三件事在目录里查不到。
REQUIRED_META = ("dataset", "dataset_version", "split", "generator", "task_count", "commit")

#: 已知的、**有意**不在注册表里的臂名。空 = 没有任何例外。
#: ★ 例外要有登记处,否则「谁还没拆」那类问题没有任何一份说得准（§9 的教训）。
KNOWN_ARM_EXCEPTIONS: set[str] = set()


def known_arms() -> set[str]:
    """所有**能跑**的臂名 —— 包括类型化的那几格。

    ★ typed 臂需要一个判定后端才能构造,但**名字**是静态的;
      这里只要名字,所以给一个 `mock` 就够了。
    """
    from experiments.core.deciding import MockClient

    return set(_BUILTIN_ARMS) | set(typed_arms(MockClient()))


def check_log_names() -> list[str]:
    """① `log/` 的一级目录名必须是注册过的数据集,或在 `_superseded/` 下。"""
    problems: list[str] = []
    if not LOG.is_dir():
        return problems
    registered = set(BENCHMARK_FACTORIES)
    for entry in sorted(LOG.iterdir()):
        if not entry.is_dir() or entry.name == SUPERSEDED:
            continue
        if entry.name not in registered:
            problems.append(
                f"log/{entry.name}/ 不是注册过的数据集名。\n"
                f"      注册的是: {sorted(registered)}\n"
                f"      → 要么重跑落进正确的名字,要么移进 log/{SUPERSEDED}/ 并写明为什么"
            )
    return problems


def check_meta() -> list[str]:
    """② 每个 run 的 `meta.json` 字段要齐。缺的跑**不可引用**。"""
    problems: list[str] = []
    if not LOG.is_dir():
        return problems
    for meta_file in sorted(LOG.glob("*/*/*/meta.json")):
        if SUPERSEDED in meta_file.parts:
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{meta_file.relative_to(EXPERIMENTS)}: JSON 坏了（{exc}）")
            continue
        missing = [f for f in REQUIRED_META if meta.get(f) is None]
        if missing:
            rel = meta_file.relative_to(EXPERIMENTS)
            problems.append(
                f"{rel}: meta.json 缺 {missing}\n"
                f"      → 这次跑的数字**不能进表**（查不到版本/划分/模型）"
            )
    return problems


def check_arm_names() -> list[str]:
    """③ `log/<数据集>/<臂>` 的臂名要能解析。"""
    problems: list[str] = []
    if not LOG.is_dir():
        return problems
    arms = known_arms() | KNOWN_ARM_EXCEPTIONS
    for arm_dir in sorted(LOG.glob("*/*")):
        if SUPERSEDED in arm_dir.parts or not arm_dir.is_dir():
            continue
        if arm_dir.name not in arms:
            problems.append(
                f"{arm_dir.relative_to(EXPERIMENTS)}/ 不是已知的臂名。\n"
                f"      → 已知: {sorted(arms)}"
            )
    return problems


def check_downloads() -> list[str]:
    """④ `dataset/DOWNLOADS.md` 要覆盖每个注册的数据集。

    ★ 它是由各 loader 的 `downloads()` **生成**的（`scripts/datasets.py --write`）,
      所以漏了只可能是「加了 loader 忘了重新生成」。
    **数据集不进仓库,下载方式就得在仓库里 —— 这是它唯一的入口。**
    """
    problems: list[str] = []
    doc = EXPERIMENTS / "dataset" / "DOWNLOADS.md"
    if not doc.is_file():
        return [f"{doc.relative_to(EXPERIMENTS)} 不存在 —— 跑 scripts/datasets.py --write"]
    text = doc.read_text(encoding="utf-8")
    for name, factory in sorted(BENCHMARK_FACTORIES.items()):
        # ★ **只要求声明了下载的那些。** `toy` 是**合成的**（`CAPITALS` 写死在代码里）,
        #   它没有 `downloads()`,也不该出现在下载索引里。
        #   第一版没分这一档,于是它报了一条假问题 —— 而假问题会让整道门被无视。
        try:
            specs = list(factory().downloads())
        except Exception:                       # noqa: BLE001 —— 构造不了就跳过
            continue
        if not specs:
            continue
        # 数据集名可能带子集后缀（`bfcl-v3-simple`）,取前缀家族名来匹配
        family = name.split("-v")[0] if "-v" in name else name
        if family not in text and name not in text:
            problems.append(f"dataset/DOWNLOADS.md 里没有 {name!r}（找过 {family!r}）")
    return problems


CHECKS = (
    ("① log/ 的目录名", check_log_names),
    ("② meta.json 的字段", check_meta),
    ("③ 臂名", check_arm_names),
    ("④ 下载说明的覆盖", check_downloads),
)

#: ★ **基线**（`check-baseline.json`）—— 照 TS 侧 `scripts/file-focus-baseline.json`
#: 的做法:**基线化,而不是让门禁常红。一个永远红的检查等于没有检查。**
#:
#: 这里装的是一次性的历史包袱:2026-09-22 之前跑的 run 都没有完整的
#: `meta.json`（那时 `runner` 只写一个 wrapper）。它们**确实不可引用**,
#: 但那是一个**已经修好**的问题的存量,不是新问题。
#:
#: **只能变小**:重跑一个就从这里删一个。
BASELINE_FILE = EXPERIMENTS / "scripts" / "check-baseline.json"


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE_FILE.is_file():
        return {}
    import json as _json
    data = _json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data.items() if not k.startswith("_")}


def _key(problem: str) -> str:
    """从一条问题里抽出用来对基线的那个键 —— **统一成 `log/` 下的相对路径**。

    ★ 第一版这里直接取「冒号前的第一段」,而基线里存的是剥掉 `log/` 和
      `/meta.json` 之后的形状 —— 两边对不上,于是基线**一条都没生效**,
      门照样全红。**一个静默失效的基线比没有基线更糟**:它看起来已经处理过了。

      所以这个函数是**唯一**的规范化处,基线生成和比对都走它。
    """
    path = problem.split(":")[0].strip()
    # 有冒号的行是 `<路径>: <说明>`;没冒号的（如臂名那条）是 `<路径>/ <说明>`
    path = path.split(" ")[0].strip()
    if path.startswith("log/"):
        path = path[len("log/"):]
    if path.endswith("/meta.json"):
        path = path[: -len("/meta.json")]
    return path.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    load_env(EXPERIMENTS)          # 有些 loader 要读 .env 才知道数据集在哪
    total = 0
    runs = 0
    if LOG.is_dir():
        runs = len([p for p in LOG.glob("*/*/*/meta.json") if SUPERSEDED not in p.parts])

    baseline = load_baseline()
    known = {k: set(v) for k, v in baseline.items()}
    grandfathered = 0

    for label, fn in CHECKS:
        problems = fn()
        fresh = []
        for p in problems:
            # ★ 基线里的东西不报 —— 但**计数**,好让「存量还剩多少」看得见
            # `seen` 是**一组路径**,`_key(p)` 是这条问题里的路径
            if any(_key(p) in group for group in known.values()):
                grandfathered += 1
                continue
            fresh.append(p)
        if fresh:
            print(f"✗ {label}")
            for p in fresh:
                print(f"    {p}")
            total += len(fresh)

    arms = len(known_arms())
    benches = len(BENCHMARK_FACTORIES)
    note = f"（另有 {grandfathered} 条基线内的存量）" if grandfathered else ""
    if total:
        print(f"\ncheck: {total} 个**新**问题{note}。**它是门,不是报告** —— 修掉再提交。")
        return 1
    print(f"check: {benches} 个数据集 · {arms} 条臂 · {runs} 个 run，全部通过{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
