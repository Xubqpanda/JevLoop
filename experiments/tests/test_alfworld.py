"""ALFWorld —— 钉住**过滤器**和**缺依赖时的行为**。

★ 这个文件最值钱的一条是 `test_the_filter_reproduces_the_published_counts`:
评估集不是「原始 zip 里的全部」,而是 `collect_game_files()` 从 255 条里
**筛出来的 134 条**。四条判据少任何一条,数目都会变大 ——
而**变大不会报错**,它只会让我们跑的是另一批题,然后和文献的数字比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.alfworld import alfworld as aw  # noqa: E402
from experiments.core.types import Trajectory  # noqa: E402

_HAS_DATA = (aw.DATASET_DIR / "alfworld" / "json_2.1.1").is_dir()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="ALFWorld 数据没下载（跑 datasets.py --fetch alfworld）")


# ═══════════════════════════════════════════════════════════
# ★★★ 过滤器：必须筛出和论文一样的条数
# ═══════════════════════════════════════════════════════════


@needs_data
def test_the_filter_reproduces_the_published_counts() -> None:
    """★★★ 实测 140 / 134 / 3,553 —— 和 ALFWorld 论文一致。

    原始 zip 里 `valid_unseen` 有 **255** 个 trial,而评估集是 **134**。
    差额是 `collect_game_files()` 的四条判据筛掉的:

    ① 路径含 `movable` / `Sliced`（THOR 动作,文本环境不支持）
    ② `task_type` 不在六个里
    ③ 没有 `game.tw-pddl`
    ④ `gamedata['solvable']` 缺失或 `False`

    ★ **少任何一条,数目都会变大,而且不会报错** —— 它只会让
      「我们跑的是哪 134 条」变成另一批题。这就是这条测试存在的理由。
    """
    bench = aw.AlfWorld()
    for split, expected in aw.SPLIT_SIZES.items():
        assert len(bench.game_dirs(split)) == expected, f"{split} 对不上论文"


@needs_data
def test_missing_data_says_how_to_get_it() -> None:
    """★ 数据不在时报错要**说清怎么拿** —— 数据不进仓库,下载方式就是唯一的入口。"""
    bench = aw.AlfWorld()
    bench.data_dir = Path("/nonexistent")
    with pytest.raises(FileNotFoundError, match=r"scripts\.datasets --fetch alfworld"):
        bench.game_dirs("valid_unseen")


@needs_data
def test_a_split_with_the_wrong_size_is_rejected(monkeypatch) -> None:
    """★ 上一条的**接口版**:`tasks()` 里那道数目检查。

    把 `SPLIT_SIZES` 改一个数就应当炸 —— 这模拟「过滤器漏了一条」。
    """
    bench = aw.AlfWorld()
    monkeypatch.setitem(aw.SPLIT_SIZES, "valid_unseen", 999)
    with pytest.raises(ValueError, match="论文是 999 条"):
        list(bench.tasks(split="valid_unseen", limit=1, seed=0))


@needs_data
def test_the_default_split_is_the_out_of_distribution_one() -> None:
    """★★ 官方 `base_config.yaml` 的默认评估划分是 **`valid_seen`**,
    而文献里报的 OOD 数是 **`valid_unseen`**。

    ⇒ 默认值会让一次「跑 ALFWorld」拿到另一个划分的数,
      **而目录名不会提醒你**。所以我们默认 unseen,并且划分名进 `meta`。
    """
    assert aw.DEFAULT_SPLIT == "valid_unseen"
    bench = aw.AlfWorld()
    task = next(iter(bench.tasks(split="", limit=1, seed=0)))
    assert task.meta["split"] == "valid_unseen"


@needs_data
def test_each_task_carries_its_goal_and_its_split() -> None:
    """★ 目标是**人工标注**那一句（`turk_annotations.anns[0].task_desc`）——
    文献里 LLM agent 看到的就是它,形如 `'Place a microwaved apple in the fridge.'`。

    ⚠️ 官方 `goal_desc_human_anns_prob` 默认是 **0.0**（用模板生成的 goal）。
      两种措辞不同,而 prompt 措辞会改分数 —— 所以模板那一份也一起留在
      `meta['synthetic_goal']` 里,好在需要时对照。
    """
    task = next(iter(aw.AlfWorld().tasks(split="valid_unseen", limit=1, seed=0)))
    assert task.prompt and task.prompt.endswith(".")
    assert " " in task.prompt, "应当是一句自然语言目标"
    assert task.meta["task_type"] in aw.TASK_TYPES.values()
    assert task.meta["synthetic_goal"], "模板口径也要留着"
    assert task.meta["answer_kind"] == "actions"


@needs_data
def test_the_same_seed_gives_the_same_tasks() -> None:
    """★ 抽样可复现 —— 用全项目同一个做法（`rewoo_port.rewoo_draw`）。"""
    a = [t.task_id for t in aw.AlfWorld().tasks(split="valid_unseen", limit=5, seed=0)]
    b = [t.task_id for t in aw.AlfWorld().tasks(split="valid_unseen", limit=5, seed=0)]
    assert a == b
    c = [t.task_id for t in aw.AlfWorld().tasks(split="valid_unseen", limit=5, seed=1)]
    assert a != c, "换 seed 要换一批题"


# ═══════════════════════════════════════════════════════════
# ★★★ 缺依赖时**明说缺什么**,不给一个 0
# ═══════════════════════════════════════════════════════════


def test_running_the_env_without_textworld_says_what_is_missing() -> None:
    """★★★ 环境那一半还没接线（或者 textworld 没装）时,必须**抛**。

    ★ **一个缺依赖看起来像一个算法读数** —— 报 0 分的 ALFWorld 和
      「所有方法都考零分」在表上长得一模一样。所以这里要 `NeedsTextWorld`,
      而且报错里要写清**缺什么**和**怎么装**。

    ★ 同时要说清**哪一半是好的**:`tasks()` 不依赖 textworld,
      而那正是「我们跑的是哪 134 条」这个可比性锚点。
    """
    bench = aw.AlfWorld()
    from experiments.core.types import Task

    bench.on_task(Task(task_id="t", prompt="p", gold="g", oracle_context=None, meta={}))
    with pytest.raises((aw.NeedsTextWorld, NotImplementedError)) as exc:
        bench.tools()
    text = str(exc.value)
    assert "textworld" in text or "还没写" in text
    assert "0" not in text.split("**")[0], "不要在报错里暗示一个分数"


def test_scoring_without_the_env_does_not_return_zero() -> None:
    """★ `score()` 在环境没接上时**抛**,绝不返回 `Judgment(correct=False)`。

    ★ 返回 `False` 会让「环境没跑起来」和「agent 失败了」在结果里同形 ——
      而那正是这个项目反复记过的那一类错。
    """
    bench = aw.AlfWorld()
    from experiments.core.types import Task

    bench.on_task(Task(task_id="t", prompt="p", gold="g", oracle_context=None, meta={}))
    traj = Trajectory(task_id="t", arm="x", steps=(), final_answer=None,
                      decisions=(), usage=(), escalated=False, error=None)
    with pytest.raises((aw.NeedsTextWorld, NotImplementedError)):
        bench.score(Task(task_id="t", prompt="p", gold="g", oracle_context=None, meta={}), traj)


def test_the_download_declares_all_three_archives_and_says_why() -> None:
    """★★ **三个 zip 都要**,而且标签不统一（json/pddl 在 0.2.2、tw-pddl 在 0.4.2）。

    ★ `solvable` 那个键在 `game.tw-pddl` 里 —— 少了 0.4.2 那个包,
      过滤器第 ④ 条会把**全部**题目筛掉,于是得到 **0 条**而不是 134 条。
    """
    specs = aw.AlfWorld().downloads()
    assert len(specs) == 3
    names = {Path(s.files[0]).name for s in specs}
    assert names == {"json_2.1.1_json.zip", "json_2.1.1_pddl.zip", "json_2.1.3_tw-pddl.zip"}
    assert "0.4.2/json_2.1.3_tw-pddl.zip" in " ".join(s.locator for s in specs), "标签不统一"
    assert all(s.pinned for s in specs), "release 也要钉 —— main 等于哪天下的算哪天"
    assert all("三个 zip 都要" in s.note for s in specs), "理由要跟着声明走"
