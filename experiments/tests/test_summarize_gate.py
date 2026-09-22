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


# ═══════════════════════════════════════════════════════════
# ★★★ 行级检查：**几条臂之间**跨了几个 commit
# ═══════════════════════════════════════════════════════════


def test_a_row_whose_arms_span_commits_is_reported(tmp_path, capsys) -> None:
    """★★★ 实测（2026-09-22）:一次 `--limit 300 × 7 臂` 的后台跑跨了 **5 个 commit** ——
    因为跑的时候一直在提交。**七条臂各在一个代码版本上。**

    而每一格单独看都是 `commits=1` —— 按格分组的检查**一条都不会报**。

    ★ 比较的意义在于「只差被测变量」。几条臂跑在不同代码版本上,
      差的就是不止一个变量了,而**这一行看起来和受控比较一模一样**。

    ★★ 这条测试的判据是「**臂与臂之间**一共几个 commit」。
      第一版写成了「某一条臂内部几个」,于是**一条都没报** ——
      而**一个把判据写反的检查,和没有这个检查在输出上完全一样**。
    """
    for arm, commit in (("direct", "c1"), ("react", "c2"), ("act", "c3")):
        _write_run(tmp_path, run_id=f"d/{arm}/run", dirty=False, commit=commit, framework=1.0)
        # 让每一行属于不同的臂
        p = tmp_path / f"d/{arm}/run" / "results.jsonl"
        import json as _json

        row = _json.loads(p.read_text(encoding="utf-8"))
        row["meta"]["arm"] = arm
        p.write_text(_json.dumps(row) + "\n", encoding="utf-8")

    rows, _, _ = summarize.load_rows(tmp_path)
    table = summarize.summarize(rows)
    assert len(table) == 3, "三条臂三个格子"

    assert summarize.main(["--log-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 0
    err = capsys.readouterr().err
    assert "几条臂跑在不同 commit 上" in err
    assert "不是受控比较" in err
    for commit in ("c1", "c2", "c3"):
        assert commit in err, f"要点名每条臂在哪个 commit:{commit}"


def test_a_row_whose_arms_share_one_commit_is_silent(tmp_path, capsys) -> None:
    """★ 反面:几条臂都在同一个 commit 上时**不许出声**。

    ★ 假警告和真警告一样有害 —— 它会让这一行被无视。
    """
    for arm in ("direct", "react"):
        _write_run(tmp_path, run_id=f"d/{arm}/run", dirty=False, commit="same", framework=1.0)
        import json as _json

        p = tmp_path / f"d/{arm}/run" / "results.jsonl"
        row = _json.loads(p.read_text(encoding="utf-8"))
        row["meta"]["arm"] = arm
        p.write_text(_json.dumps(row) + "\n", encoding="utf-8")

    assert summarize.main(["--log-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 0
    assert "不是受控比较" not in capsys.readouterr().err


def test_internal_fields_do_not_leak_into_the_table(tmp_path) -> None:
    """★ 下划线开头的是**内部字段**（行级检查的中间量),不许进 markdown / csv。"""
    _write_run(tmp_path, run_id="d/a/run", dirty=False, commit="c1", framework=1.0)
    rows, _, _ = summarize.load_rows(tmp_path)
    table = summarize.summarize(rows)
    assert any(k.startswith("_") for k in table[0]), "内部字段要存在（检查要用）"
    md = summarize.to_markdown(table)
    assert "_arm_commits" not in md
