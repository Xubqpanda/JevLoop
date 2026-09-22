"""ALFWorld —— **改编自 `alfworld/alfworld` 的环境与过滤器。**

## 出处

| | |
|---|---|
| **论文** | Shridhar, Yuan, Côté, Bisk, Saveliev, Hausknecht. *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*. ICLR 2021. arXiv:2010.03768 |
| **仓库** | `github.com/alfworld/alfworld` @ master `aaba6870f86c5be6a08a491f32a50b906227bc3e` |
| **改编自** | `alfworld/agents/environment/alfred_tw_env.py::collect_game_files`（**过滤器**）、`alfworld/agents/eval/evaluate_dagger.py`（判分）|
| **数据** | GitHub **release assets**（不在仓库里,也不在 HF 上）|

## ★★★ 这个数据集是**交互式环境**,形状和前面全部不一样

前面那些（GSM8K / BFCL / BigBench / HotpotQA …）都是**静态问答**:
`tasks()` 出一批题,`score()` 拿轨迹判分。ALFWorld 的每一题是一个
**TextWorld 环境** —— 动作会改变它的状态,而「对不对」只有环境知道。

于是接口上多了两件事:

1. **`on_task(task)`**（`core/bench.py` 里那个可选钩子）——
   每题开跑前 runner 调一次,benchmark 据此绑定**这一题的环境实例**。
   ★ 没有它就只能靠「`tasks()` 和 `tool_impls()` 交替调用」这个**隐式顺序**去猜,
     而猜错是「A 题的动作打在 B 题的环境上」,分数照出。
2. **判分是回放**:`score()` 拿轨迹里的动作在一个**新环境**里重放,
   再看 `won`。所以它不依赖「跑的时候那个环境实例还在」。

## ★★★ 评估集的**过滤器**必须逐条照抄

原始 zip 里 `valid_unseen` 有 255 个 trial,而评估集是 **134** ——
差额是 **`collect_game_files()` 筛掉的**。四条,缺一条数目就对不上:

1. 路径里有 `movable` 或 `Sliced` → 跳过（THOR 的动作,文本环境不支持）
2. `traj_data['task_type']` 不在六个里 → 跳过
3. 没有 `game.tw-pddl` → 跳过
4. `gamedata['solvable']` 缺失或为 `False` → 跳过

**实测（2026-09-22）:筛出 train 3,553 / valid_seen 140 / valid_unseen 134**
—— 和论文一致。所以「我们跑的是哪 134 条」和文献里是同一批。

## ⚠️ 四条必须记住的坑（读原仓库读出来的）

**① 文本模式下 GCS 恒为 0。** `goal_condition_success_rate` 只在 **THOR** 上有;
TextWorld 1.6.2 不提供它。所以**不能报 GCS** —— 报出来是一列 0,
而它看起来像一个「所有方法都很差」的结果。

**② 判分是二值的 `won`。** `evaluate_dagger.py` 用 `infos["won"]`,
每题取 `np.max`（多次试跑取最好）再 `np.mean`。★ `np.max` 那一层是
**多次试跑**才有的 —— 单次跑就是那个二值本身。

**③ 默认配置跑的是 `valid_seen`,不是 unseen。** 而文献里报的 OOD 数是
**`valid_unseen`**。这个默认值会让一次「跑 ALFWorld」拿到另一个划分的数,
而目录名不会提醒你。⇒ **我们默认 `valid_unseen`**,并在 `meta` 里带上划分名。

**④ 没有排行榜,答案也没被 withhold。** 每个 unseen 任务的 walkthrough
都随数据发出来 —— 所以「偷看」在这里**技术上可行**。
`benchmark` 不把它放进 prompt,但**这一点是约定,不是结构**:
它和 `DecisionView` 只带 `task_id`（拿不到 gold）不是一回事。

## ★ 现在**没有** TextWorld 也能做的那一半

`tasks()` / `downloads()` **不依赖 TextWorld** —— 它们只读数据文件。
而 `tools()` / `score()` 需要它,缺了就**明说缺什么**,不静默给一个 0。

这不是权宜:枚举出来的那 134 条**正是可比性的锚点**,
而它现在就有测试钉着。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Tool, Trajectory

#: ALFWorld 主干的 commit（树会动,这条是**我们读的那一份**）。
ALFWORLD_COMMIT = "aaba6870f86c5be6a08a491f32a50b906227bc3e"

#: 数据在 **release assets** 里,而且**标签不统一** —— json/pddl 在 0.2.2,
#: tw-pddl 在 0.4.2。照抄官方 `scripts/alfworld-download` 的那三行。
RELEASES = (
    ("0.2.2", "json_2.1.1_json.zip"),
    ("0.2.2", "json_2.1.1_pddl.zip"),
    ("0.4.2", "json_2.1.3_tw-pddl.zip"),
)

#: 六个任务类型。★ 顺序和 `alfred_thor_env.py::TASK_TYPES` 一致。
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
_TASK_TYPE_NAMES = frozenset(TASK_TYPES.values())

#: 划分 → 论文里的条数。★ 写成常量是为了**能当场核对**:
#: 筛出来的数目对不上就说明过滤器漏了一条,而不是「数据换版本了」。
SPLIT_SIZES = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}

#: ★ **默认 OOD** —— 文献报的是它（见模块头「坑 ③」）。
DEFAULT_SPLIT = "valid_unseen"


class NeedsTextWorld(RuntimeError):
    """要跑环境,但 `textworld` / `alfworld` 没装。

    ★ 和「跑出来是 0」是两件事。**一个缺依赖看起来像一个算法读数**,
      是这类工作里最难发现的一种错。
    """


@dataclass
class AlfWorld:
    """ALFWorld 文本环境。"""

    name: str = "alfworld"
    dataset_version: str = f"alfworld@{ALFWORLD_COMMIT[:8]}+data-2.1.1/2.1.3"
    data_dir: Path = field(default_factory=lambda: DATASET_DIR / "alfworld")
    split: str = DEFAULT_SPLIT
    limit: int | None = None
    seed: int = 0
    #: 环境在「不可解」时自己放弃的步数上限。★ 官方 `expert_timeout_steps` 是 150,
    #: 那是 **expert** 的上限,不是 agent 的 —— 后者由 runner 的 `max_steps` 定。
    max_steps: int = 50
    #: 当前题的环境（由 `on_task` 绑定）。**评分的回放不用它** —— 见 `score()`。
    _env: object = field(default=None, init=False, repr=False)
    _current: Task | None = field(default=None, init=False, repr=False)

    # ── 数据在哪 ────────────────────────────────────────────

    def _json_root(self) -> Path:
        root = self.data_dir / "json_2.1.1"
        if not root.is_dir():
            raise FileNotFoundError(
                f"{self.name}: 缺 {root}\n"
                f"  跑 `python3 -m experiments.scripts.datasets --fetch {self.name}`"
            )
        return root

    def game_dirs(self, split: str) -> list[Path]:
        """**评估集的过滤器** —— 逐条照抄 `collect_game_files()`（模块头有说明）。

        ★ 四条缺一不可。实测这四条筛出 140 / 134,和论文一致;
          少任何一条,数目都会变大,而**变大不会报错** ——
          它只会让我们的 134 条变成别人的另一批题。
        """
        base = self._json_root() / split
        if not base.is_dir():
            raise FileNotFoundError(f"{self.name}: 没有 {split} 划分（{base}）")

        kept: list[Path] = []
        # ★ `topdown=False` 照抄官方 —— 遍历顺序影响 `game_files` 的顺序,
        #   而顺序影响「第 i 条是哪一题」。这里**额外排一次序**,
        #   因为 `os.walk` 的顺序是文件系统相关的,不排序就没法复现。
        for root, _dirs, files in os.walk(base, topdown=False):
            if "traj_data.json" not in files:
                continue
            if "movable" in root or "Sliced" in root:      # ①
                continue
            json_path = Path(root) / "traj_data.json"
            try:
                traj = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if traj.get("task_type") not in _TASK_TYPE_NAMES:   # ②
                continue
            game_path = Path(root) / "game.tw-pddl"
            if not game_path.exists():                          # ③
                continue
            try:
                game = json.loads(game_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not game.get("solvable"):                        # ④
                continue
            kept.append(Path(root))
        return sorted(kept)

    def _goal_text(self, game_dir: Path) -> str:
        """给 agent 看的那句话。

        ★ 用**人工标注**的 `task_desc`（`turk_annotations.anns[0]`）——
          实测形如 `'Look at a mug in lamp light.'`。这也是文献里
          LLM agent 看到的那个文本。

        ⚠️ 官方的 `goal_desc_human_anns_prob` 默认是 **0.0**,也就是用
          **模板生成**的 goal。两者的措辞不同,而 prompt 措辞会改分数。
          这里选人工标注那一份,理由:**它是文献里报数的那个设置**;
          要复现官方默认值就走 `meta['synthetic_goal']`。
        """
        traj = json.loads((game_dir / "traj_data.json").read_text(encoding="utf-8"))
        anns = ((traj.get("turk_annotations") or {}).get("anns") or [])
        for ann in anns:
            desc = (ann or {}).get("task_desc")
            if desc:
                return str(desc).strip()
        raise ValueError(f"{self.name}: {game_dir} 里没有 task_desc")

    def _synthetic_goal(self, game_dir: Path) -> str:
        """模板生成的 goal（官方默认设置）。留着做对照,不作默认。"""
        traj = json.loads((game_dir / "traj_data.json").read_text(encoding="utf-8"))
        params = traj.get("pddl_params") or {}
        ttype = traj.get("task_type", "")
        bits = [f"{k}={v}" for k, v in sorted(params.items()) if v not in ("", None, False)]
        return f"{ttype}（{', '.join(bits)}）"

    # ── 协议：任务 ──────────────────────────────────────────

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """★ `split` 默认取 `DEFAULT_SPLIT`（`valid_unseen`）—— 理由见模块头「坑 ③」。

        ★ **抽样用和 ReWOO 那一套相同的 `rewoo_draw`**（`RandomState(seed).permutation`）
          —— 不因为它是 ReWOO 的就只能用在那几个上;它只是一个
          「用 seed 定一批题」的可复现做法,而全项目用同一种做法比各写各的好。
        """
        from experiments.benchmark.rewoo_port import rewoo_draw

        chosen = split or self.split
        dirs = self.game_dirs(chosen)
        expected = SPLIT_SIZES.get(chosen)
        if expected is not None and len(dirs) != expected:
            # ★ 数目不对就**当场炸** —— 这说明过滤器漏了一条,
            #   而不是「数据换版本了」。静默继续会让我们跑的是另一批题。
            raise ValueError(
                f"{self.name}: {chosen} 筛出 {len(dirs)} 条，而论文是 {expected} 条。\n"
                f"  → 过滤器漏了一条？见 `game_dirs()` 的四条判据（模块头也有一份）"
            )
        for i in rewoo_draw(len(dirs), limit if limit is not None else self.limit, seed):
            game_dir = dirs[i]
            yield Task(
                task_id=f"{self.name}/{chosen}/{game_dir.parent.name}",
                prompt=self._goal_text(game_dir),
                # ★ ALFWorld **没有「金标答案」这种字符串** —— 判据是环境里的
                #   目标条件。`gold` 在这里只是给日志读的,
                #   **真正的判定在 `score()` 里跑环境**。
                gold="(由环境判定)",
                # 金标证据:每个 unseen 任务的 walkthrough 都随数据发出来。
                # ★ 我们**不接**它 —— 接了就等于把答案送进 prompt（模块头「坑 ④」）。
                oracle_context=None,
                meta={
                    "game_dir": str(game_dir.relative_to(self.data_dir)),
                    "split": chosen,
                    "task_type": json.loads(
                        (game_dir / "traj_data.json").read_text(encoding="utf-8")
                    ).get("task_type"),
                    "synthetic_goal": self._synthetic_goal(game_dir),
                    "answer_kind": "actions",
                },
            )

    # ── 协议：工具与环境 ────────────────────────────────────

    def on_task(self, task: Task) -> None:
        """runner 每题调一次 —— **在这里绑定这一题的环境**。

        ★ 这是 `core/bench.py::on_task` 存在的**唯一理由**（见模块头）。
          没有它,`tool_impls()` 只能靠调用顺序猜自己该绑定谁。
        """
        self._current = task
        self._env = None          # 懒建:没装 textworld 时不该在出题阶段就炸

    def _ensure_env(self):
        """建出这一题的环境。**缺依赖时明说缺什么。**"""
        if self._env is not None:
            return self._env
        if self._current is None:
            raise NeedsTextWorld(
                f"{self.name}: 没有绑定任务。`tool_impls()` 需要 runner 先调 `on_task(task)`"
            )
        try:
            import textworld  # noqa: F401
            import alfworld  # noqa: F401
        except ImportError as exc:
            raise NeedsTextWorld(
                f"{self.name}: 跑环境需要 `textworld` 和 `alfworld` —— 没装（{exc}）。\n"
                f"  装法: `python3 -m pip install alfworld`（基础依赖只有 textworld[pddl]）\n"
                f"  ★ **任务枚举不需要它们**（`tasks()` 只读数据文件）——"
                f"   缺的只是环境和判分,别把这两件事混成一件"
            ) from exc
        raise NotImplementedError(
            f"{self.name}: TextWorld 环境接线**还没写**。\n"
            f"  已经做完并验证的:数据下载、评估集过滤器（140/134 与论文一致）、任务枚举。\n"
            f"  还没做:环境步进、`admissible_commands` 作为候选来源、回放判分。\n"
            f"  ⚠️ 现在报 `NotImplementedError`,**不是返回一个 0 分** ——"
            f"    缺一块和考零分在表上长得一样,而那是要避免的。"
        )

    def tools(self) -> Sequence[Tool]:
        """ALFWorld 的动作空间。

        ★★ **它是逐步变化的,而且由环境给出** —— TextWorld 的
        `admissible_commands` 就是当前可执行的命令列表（通常 10–40 条）。
        所以这个数据集上「候选每步重建」（§8.4）**不是优化,是必需**:
        固定的候选列表在这里会立刻失效,因为上一步的动作改变了可选项。

        ⚠️ 现在还没接（见 `_ensure_env`）。接上之后,
        `pickInput` 的候选来源就是这个列表 —— 那正好是判定模型
        「只能从枚举里挑」那条硬约束的**天然满足**的一个场景。
        """
        self._ensure_env()
        raise AssertionError("unreachable")

    def tool_impls(self) -> dict[str, object]:
        self._ensure_env()
        raise AssertionError("unreachable")

    # ── 协议：判分 ──────────────────────────────────────────

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        """**回放轨迹里的动作,再看 `won`。**

        ★ 为什么不依赖「跑的时候那个环境实例」:轨迹是**唯一的事实**,
          而评分要在轨迹上可复现 —— 这也让**同一批轨迹换判据重判**
          （`scripts/rescore.py`）成为可能。

        ⚠️ 判分是**二值**的（`infos["won"]`）。**不要报 GCS** ——
          TextWorld 不提供 `goal_condition_success_rate`,报出来恒为 0
          （模块头「坑 ①」）。
        """
        self._ensure_env()
        raise AssertionError("unreachable")

    def check(self, task: Task, answer: str) -> bool:
        """Reflexion 的 Evaluator 要的成败信号。ALFWorld 里它就是 `won`。"""
        raise NeedsTextWorld(
            f"{self.name}: `check()` 要跑环境（`won`）—— 还没接,见 `score()`"
        )

    # ── 下载 ────────────────────────────────────────────────

    def downloads(self) -> Sequence[DownloadSpec]:
        """★ 三份 release asset,而且**标签不统一**（json/pddl 在 0.2.2、tw-pddl 在 0.4.2）。

        ⚠️ 三个 zip **都要**:`solvable` 那个键在 `game.tw-pddl` 里
          （来自 0.4.2 那个包）,而 `traj_data.json` 在 0.2.2 那个包里。
          只下前两个的话,过滤器第 ④ 条会把**全部**题目筛掉 ——
          于是得到 0 条,而不是 134 条。
        """
        return [
            DownloadSpec(
                dataset=self.name,
                kind="http",
                locator=(f"https://github.com/alfworld/alfworld/releases/download/{tag}/{fname}"),
                files=(fname,),
                revision=ALFWORLD_COMMIT,
                size_hint={"json_2.1.1_json.zip": "69 MB",
                           "json_2.1.1_pddl.zip": "34 MB",
                           "json_2.1.3_tw-pddl.zip": "35 MB"}[fname],
                note=("★ 数据在 **release assets** 里,既不在仓库也不在 HF。"
                      "★ **三个 zip 都要**:`solvable` 键在 `game.tw-pddl`（0.4.2 那个包）里,"
                      "少了它过滤器会把全部题目筛掉。"
                      "解压到 `dataset/alfworld/json_2.1.1/<split>/`"),
            )
            for tag, fname in RELEASES
        ]


__all__ = ["AlfWorld", "NeedsTextWorld", "TASK_TYPES", "SPLIT_SIZES",
           "DEFAULT_SPLIT", "ALFWORLD_COMMIT", "RELEASES"]
