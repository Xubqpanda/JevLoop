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


def test_the_missing_dependency_path_still_says_what_is_missing(monkeypatch) -> None:
    """★★★ 环境**已经接上了**,但「缺依赖时怎么说」这条路径要留着。

    ★ 理由不是洁癖:**一个缺依赖看起来像一个算法读数** ——
      报 0 分的 ALFWorld 和「所有方法都考零分」在表上长得一模一样。
      所以这里用 monkeypatch 把 import 弄坏,确认它**抛**而不是给 0。

    （这条现在是注入的,不是环境真的缺 —— 环境已经能跑了,见下面那条。）
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("textworld", "alfworld"):
            raise ImportError(f"injected: {name} unavailable")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    bench = aw.AlfWorld()
    from experiments.core.types import Task

    bench.on_task(Task(task_id="t", prompt="p", gold="g", oracle_context=None,
                       meta={"game_dir": "x", "split": "valid_unseen"}))
    with pytest.raises(aw.NeedsTextWorld) as exc:
        bench.tools()
    assert "pip install alfworld" in str(exc.value), "报错要说清怎么装"
    assert "任务枚举不需要它们" in str(exc.value), "要说清哪一半是好的"


@needs_data
def test_the_environment_actually_opens_and_gives_admissible_commands() -> None:
    """★★★ 环境真的能开 —— 而且**候选是环境给的**,不是我们编的。

    实测:一题开局有 **21 条**候选（`go to bed 1` / `go to desk 1` / …）。

    ★★ 这正是「候选每步重建」（§8.4）在 ALFWorld 上**不是优化而是必需**的原因:
      上一步的动作改变了可选项,固定的候选列表会立刻失效。

    ★ 而它也是「判定模型只能从枚举里挑」那条硬约束**天然满足**的地方 ——
      环境直接把候选给你,不用猜。
    """
    if not pytest.importorskip("textworld", reason="没装 textworld"):
        pytest.skip("no textworld")
    import importlib.util

    if not importlib.util.find_spec("alfworld"):
        pytest.skip("no alfworld")

    bench = aw.AlfWorld()
    task = next(iter(bench.tasks(split="valid_unseen", limit=1, seed=0)))
    bench.on_task(task)

    obs = bench.reset_episode()
    assert isinstance(obs, str) and obs.strip(), "环境要给一段观察"
    candidates = bench.admissible()
    assert len(candidates) >= 5, f"候选太少:{candidates[:3]}"
    assert all(isinstance(c, str) and c.strip() for c in candidates)
    # ★ 命令的语法是 TextWorld 给的（`go to X` / `take X from Y` …）——
    #   我们自己拼一套语法的话,只要有一处对不上命令就会被拒,
    #   而错误看起来像「agent 选错了」。
    assert any(c.startswith("go to ") for c in candidates)


@needs_data
def test_the_env_reports_won_as_a_bool_and_a_wrong_action_is_not_a_crash() -> None:
    """★ 判分是**二值**的 `won`。乱走一步应当返回 `won=False`,而不是抛异常。

    ★ 而 `steps` 的返回值形状也要对 —— 报错「工具坏了」比「agent 选了错动作」
      严重得多,两者必须分得开。
    """
    if not pytest.importorskip("textworld", reason="没装 textworld"):
        pytest.skip("no textworld")
    import importlib.util

    if not importlib.util.find_spec("alfworld"):
        pytest.skip("no alfworld")

    bench = aw.AlfWorld()
    task = next(iter(bench.tasks(split="valid_unseen", limit=1, seed=0)))
    bench.on_task(task)
    bench.reset_episode()
    obs, won, done = bench.step("look")
    assert isinstance(obs, str) and obs.strip()
    assert isinstance(won, bool) and isinstance(done, bool)
    assert won is False, "看一眼不该赢"


@needs_data
def test_scoring_replays_the_trajectory_rather_than_trusting_a_live_env() -> None:
    """★★★ `score()` **回放轨迹**,不依赖「跑的时候那个环境实例」。

    ★ 轨迹是**唯一的事实**:回放让评分可复现,也让「同一批轨迹换判据重判」
      成为可能（`scripts/rescore.py` 要的正是这个）。

    ★ 而空轨迹要给 `no_answer`,**不是**一个 `won=False` 的普通失败 ——
      「一条动作都没有」和「走了几步没做成」是两种不同的失败。
    """
    from experiments.core.types import Action, Step, Task

    bench = aw.AlfWorld()
    task = Task(task_id="t", prompt="p", gold="g", oracle_context=None,
                meta={"game_dir": next(iter(aw.AlfWorld().tasks(
                    split="valid_unseen", limit=1, seed=0))).meta["game_dir"],
                    "split": "valid_unseen"})

    empty = Trajectory(task_id="t", arm="x", steps=(), final_answer=None,
                       decisions=(), usage=(), escalated=False, error=None)
    j = bench.score(task, empty)
    assert j.correct is False and j.failure_class == "no_answer"

    if not pytest.importorskip("textworld", reason="没装 textworld"):
        pytest.skip("no textworld")
    import importlib.util

    if not importlib.util.find_spec("alfworld"):
        pytest.skip("no alfworld")

    # 走一步 `look`（几乎不可能完成任务）→ 回放后应当是「做了但没成」
    traj = Trajectory(
        task_id="t", arm="x", final_answer=None, decisions=(), usage=(),
        escalated=False, error=None,
        steps=(Step(index=0, action=Action(kind="tool", name="act",
                                           arguments={"command": "look"}),
                    observation="…"),),
    )
    j = bench.score(task, traj)
    assert j.correct is False
    assert j.failure_class == "task_not_completed", "有动作但没成 —— 和「没动作」要分开"
    assert "回放 1 步" in j.detail


def test_the_download_declares_all_three_archives_and_says_why() -> None:
    """★★ **三个 zip 都要**,而且标签不统一（json/pddl 在 0.2.2、tw-pddl 在 0.4.2）。

    ★ `solvable` 那个键在 `game.tw-pddl` 里 —— 少了 0.4.2 那个包,
      过滤器第 ④ 条会把**全部**题目筛掉,于是得到 **0 条**而不是 134 条。
    """
    specs = aw.AlfWorld().downloads()
    assert len(specs) == 4, "三个数据包 + 一份仓库（logic 文件）"

    zips = [s for s in specs if s.files[0].endswith(".zip")]
    assert {Path(s.files[0]).name for s in zips} == {
        "json_2.1.1_json.zip", "json_2.1.1_pddl.zip", "json_2.1.3_tw-pddl.zip"}
    assert "0.4.2/json_2.1.3_tw-pddl.zip" in " ".join(s.locator for s in zips), "标签不统一"

    # ★★ 第四份:logic 那两个文件**不在数据包里**,在仓库里。
    #   少了它们环境起不来,而报错说的是「Unsupported game format」——
    #   不指向真正的原因。
    logic = [s for s in specs if not s.files[0].endswith(".zip")][0]
    assert set(logic.files) == {"alfred.pddl", "alfred.twl2"}
    assert logic.locator.endswith(aw.ALFWORLD_COMMIT), "仓库也要钉 commit"
    assert "不在数据包里" in logic.note

    assert all(s.pinned for s in specs), "release 和 tarball 都要钉 —— main 等于哪天下的算哪天"
