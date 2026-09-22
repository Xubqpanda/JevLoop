"""`scripts/check.py` —— **机械检查本身也要有测试**。

一个「永远说通过」的检查和一个坏掉的检查在效果上一样,
而**没有任何东西会告诉你它是哪一种** —— 除非有一条测试专门造一个违规出来,
看它红不红。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.scripts import check  # noqa: E402


# ═══════════════════════════════════════════════════════════
# ★★ 基线键的规范化 —— 它曾经静默失效过
# ═══════════════════════════════════════════════════════════


def test_the_baseline_key_strips_log_and_meta_json() -> None:
    """★★★ 第一版 `_key()` 取的是「冒号前的第一段」,而基线里存的是剥掉
    `log/` 和 `/meta.json` 之后的形状 —— **两边对不上,基线一条都没生效**,
    门照样全红。

    **一个静默失效的基线比没有基线更糟**:它看起来已经处理过了。

    所以 `_key()` 是**唯一**的规范化处,基线的生成和比对都走它。
    """
    assert check._key("log/gsm8k/direct/2026-09-22T02-58-49Z-seed0/meta.json: 缺字段") \
        == "gsm8k/direct/2026-09-22T02-58-49Z-seed0"
    # 臂名那条没有冒号,靠空格切
    assert check._key("log/gsm8k/direct-rescore-strict/ 不是已知的臂名。") \
        == "gsm8k/direct-rescore-strict"


def test_the_shipped_baseline_matches_what_the_checks_actually_emit() -> None:
    """★★ 基线和检查**必须互相认得** —— 上面那条 bug 就是它们对不上。

    这条测试是那个 bug 的**结构性**防法:不测某个键,而是测
    「基线里存的每一个键,都能被 `_key()` 从某条真实问题里还原出来」。
    """
    baseline = check.load_baseline()
    assert baseline, "基线文件不见了?"

    emitted = set()
    for _, fn in check.CHECKS:
        for problem in fn():
            emitted.add(check._key(problem))

    stale = [k for group in baseline.values() for k in group if k not in emitted]
    assert not stale, (
        f"基线里有 {len(stale)} 条**已经不存在**的键 —— 基线只能变小,"
        f"这些要删掉:{stale[:5]}"
    )


def test_a_fresh_violation_is_not_swallowed_by_the_baseline() -> None:
    """★★★ **门要抓得住新问题。** 造一个不在注册表里的数据集 + 没听过的臂。

    这是「基线化」最危险的一面:基线一上线,很容易变成
    「反正它说通过」。所以专门造一条违规,确认它**不在**基线里、
    而且真的让 `main()` 返回非 0。
    """
    fake = check.LOG / "绝对没注册过的数据集" / "some-arm" / "2026-01-01T00-00-00Z-seed0"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "meta.json").write_text("{}", encoding="utf-8")
    try:
        known = {k for group in check.load_baseline().values() for k in group}
        assert "绝对没注册过的数据集" not in " ".join(known)

        assert check.check_log_names(), "① 没抓到没注册过的数据集名"
        assert check.check_meta(), "② 没抓到空 meta.json"
        assert check.check_arm_names(), "③ 没抓到没听过的臂名"
        assert check.main([]) == 1, "有真问题时 main() 必须返回非 0"
    finally:
        import shutil
        shutil.rmtree(check.LOG / "绝对没注册过的数据集", ignore_errors=True)


def test_the_check_passes_on_the_current_tree() -> None:
    """★ 当前这棵树上它应当是绿的。红了就说明真有问题,不要动基线 —— 去修问题。"""
    assert check.main([]) == 0


def test_the_unusable_runs_are_explained_in_a_tracked_file() -> None:
    """★★ `log/_superseded/` 下的东西**不参与检查** —— 但那份解释必须**进 git**。

    ★ 这里踩过一个坑:`log/*` 整个被 gitignore,于是我把说明写在
      `_superseded/README.md` 里 —— **它进不了库**。别人 clone 下来
      看到的是一份没有解释的规则,而那份说明只存在于我这台机器上。

      修法照 `dataset/` 的现成做法:给 `log/README.md` 开同一个例外。
      **「这东西不能用」本身必须写在能被找到的地方** ——
      否则下一个人只会看到一堆旧目录,然后挑一个引用。
    """
    doc = check.LOG / "README.md"
    assert doc.is_file(), (
        "log/README.md 不在 —— 它应当是**进库**的那一份说明"
        "（.gitignore 里要有 `!experiments/log/README.md`）"
    )
    text = doc.read_text(encoding="utf-8")
    assert "不可引用" in text and "meta.json" in text and "dataset_version" in text


def test_superseded_runs_do_not_trip_the_meta_check() -> None:
    """★ `_superseded/` 里的跑**确实缺字段**,但它们不该被 ② 报出来 ——
    那正是把它们移进去的目的。"""
    for problem in check.check_meta() + check.check_arm_names():
        assert check.SUPERSEDED not in problem, f"_superseded 还是被检查了:{problem}"
