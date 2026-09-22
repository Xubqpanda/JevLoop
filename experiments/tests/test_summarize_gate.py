"""`log/ → result/` 的**引用资格** —— 什么数字能进表。

★ 这一层以前只拒「缺字段」的行。而实测发现另一半更隐蔽:
**同一个 commit 加 `dirty=True` 的两次跑可以完全不同** ——
`bfcl × react-typed` 那两次,commit 都是 `5e9aee6b`、都 dirty,
一次 `framework_ms=-1670`、另一次 `=2`,中间那次修复**没提交**。
**从日志里分不出这两次。**

所以 `dirty` 不是备注,是引用资格。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.scripts import summarize  # noqa: E402


def _write_run(root: Path, *, run_id: str, dirty: bool, commit: str, framework: float) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    # ★ 必填字段一个都不能少 —— 少了会被 `missing_fields` 拒掉,
    #   而那会把这条测试变成「因为别的理由被拒」,看起来还是绿的。
    #   （第一版就是这样:0 行,而我以为它测的是 dirty。）
    row = {
        "run_id": run_id,
        "meta": {"dataset": "d", "arm": "a", "seed": 0, "dirty": dirty, "commit": commit},
        "task_id": run_id,
        "correct": True, "score": 1.0, "steps": 1, "first_divergence_step": None,
        "escalated": False, "gate_false_reject": False, "gate_false_deny": False,
        "failure_class": None, "artifacts": {},
        "cost": {"llm_calls": 1, "decision_requests": 0, "input_tokens_cached": 0,
                 "input_tokens_uncached": 10, "output_tokens_visible": 1,
                 "output_tokens_reasoning": 0, "usd": 0.0},
        "timing": {"wall_ms": 100.0, "framework_ms": framework},
    }
    (d / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_a_dirty_run_does_not_reach_the_table(tmp_path: Path) -> None:
    """★★★ 脏工作区跑出来的行**默认不进表**。

    理由不是洁癖:`repo_commit` 的文档自己写着「脏工作区跑出来的数字
    别人复现不了,连跑它的人自己都复现不了」。
    """
    _write_run(tmp_path, run_id="d/a/clean", dirty=False, commit="c1", framework=2.0)
    _write_run(tmp_path, run_id="d/a/dirty", dirty=True, commit="c1", framework=-1670.0)

    rows, _, dirty_runs = summarize.load_rows(tmp_path)
    assert len(rows) == 1, "脏的那次不该进来"
    assert dirty_runs == ["d/a/dirty"], "但**要说出来**排除了谁"


def test_allow_dirty_is_available_for_development(tmp_path: Path) -> None:
    """★ 开发时要能把它算进来 —— 否则「先看看跑通没有」这一步没法做。"""
    _write_run(tmp_path, run_id="d/a/clean", dirty=False, commit="c1", framework=2.0)
    _write_run(tmp_path, run_id="d/a/dirty", dirty=True, commit="c1", framework=-1670.0)

    rows, _, _ = summarize.load_rows(tmp_path, allow_dirty=True)
    assert len(rows) == 2


def test_a_cell_spanning_commits_is_flagged(tmp_path: Path) -> None:
    """★★ 一个格子跨多个 commit = **把不同代码版本平均**,报表时要报出来。

    `run_ids` 那一列以前写的是 `len(...)`,于是它的值是 `2` ——
    读表的人拿不到任何线索,而那一列存在的全部理由就是「归档之后那是唯一的线索」。
    现在它列出 id,并多一列 `commits`。
    """
    _write_run(tmp_path, run_id="d/a/one", dirty=False, commit="c1", framework=1.0)
    _write_run(tmp_path, run_id="d/a/two", dirty=False, commit="c2", framework=2.0)

    rows, _, _ = summarize.load_rows(tmp_path)
    table = summarize.summarize(rows)
    assert len(table) == 1
    assert table[0]["commits"] == 2, "跨了 commit 就要显示 2"
    assert table[0]["runs"] == "one two", f"要列出 id,不是个数:{table[0]['runs']!r}"
    assert table[0]["n"] == 2


def test_the_shipped_summary_only_contains_citable_runs() -> None:
    """★ 当前仓里那份 `summary.md` **不许含脏跑的格子**。

    它从 74/80 掉到 3 个格子,那是对的 —— **这个项目现在可引用的数字很少**
    是事实,不是回归。这条测试防的是「哪天有人为了表好看把 --allow-dirty 加上」。
    """
    rows, _, dirty_runs = summarize.load_rows(summarize.LOG_DIR)
    assert all(not r["meta"].get("dirty") for r in rows)
    # 而且被排除的要能数得出来 —— 静默排除等于伪装成「表是全的」
    assert isinstance(dirty_runs, list)
