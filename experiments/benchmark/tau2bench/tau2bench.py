"""τ²-bench —— **改编自 `sierra-research/tau2-bench` 的 v0.1.0（钉住）。**

## 出处

| | |
|---|---|
| **论文** | Barres, Dong, Ray, Si, Narasimhan, Wang, Wang, Wang, Yang, Zhou, Zhu, Klein, Lu, Sheng, Wang, Zhang, Zheng, Zhou, Yao. *τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment*. arXiv:2506.07982 |
| **仓库** | `github.com/sierra-research/tau2-bench`，**钉 tag `v0.1.0`**（`37199f36924c`）|
| **改编自** | `data/tau2/domains/<域>/tasks.json`、`policy.md`、`db.json`；判分读 `src/tau2/evaluator/evaluator_env.py` |

## ★★★ 为什么必须钉 `v0.1.0` —— 仓库 HEAD 已经是**另一个 benchmark**

**实测（2026-09-22）:仓库 main 那棵树现在是 `τ³-bench v1.0.1`。**
而它自己的 README 写着:**`<1.0.1` 的结果不能和 `>=1.0.1` 比**。

⇒ 拿 HEAD 跑出来的数**不能和论文比**,而**目录名不会提醒你**。
   所以这里钉 tag,并且 `dataset_version` 里带着它。

★ 另一个只有钉 v0.1.0 才成立的事实:**那个版本的 `tasks.json` 就是 base 划分**
（airline 50 / retail 114 / telecom 114 = **278**,正是论文那个数）。
HEAD 上多了 `split_tasks.json`,base 要从里面挑 —— 换版本连「哪 278 条」都会变。

## ★★★ 它是**两方**任务,而 agent 看不到用户那一半

和前面所有数据集都不同:这里有一个 **LLM 用户模拟器**扮演顾客。

| 谁 | 看到什么 |
|---|---|
| **agent** | 域的 **`policy.md`**（业务规则）+ 工具 |
| **用户模拟器** | `user_scenario`（persona + `task_instructions`）|

⇒ **`Task.prompt` 是 policy,不是 user_scenario。**
   把用户意图写进 agent 的 prompt 等于**把要通过对话发现的东西直接告诉它** ——
   而那样跑出来的分**看起来会更好**,且看不出为什么。

★ 所以 `user_scenario` / `description` / `evaluation_criteria` 全在 `meta` 里,
  **不进 prompt** —— 它们分别是用户模拟器的输入、出题人的备注、和判分依据。

## ★★★ 判分是**整个数据库的状态哈希相等**

`evaluator_env.py::calculate_reward` 的做法:

1. 用 agent 的动作**重放**一遍 → `agent_db_hash`
2. 用**金标动作**（`evaluation_criteria.actions`）重放一遍 → `predicted_agent_db_hash`
3. `agent_db_match = agent_db_hash == predicted_agent_db_hash`
4. `reward *= db_reward`（当 `DB` 在 `reward_basis` 里）
5. 再乘上 `ENV_ASSERTION` 那一项,由 `task.reward_basis` 决定乘哪些

★ **这是全项目最严的判分**:不是「答对了没有」,是**整个数据库一个字节都不能差**。

## 实测:278 条的判分基础分布

| 域 | 条数 | `evaluation_criteria` 里有什么 |
|---|---|---|
| airline | 50 | 7 条只有 `nl_assertions`；37 条 `actions`+`nl`；6 条三者都有 |
| retail | 114 | 68 条只有 `actions`；36 条 `actions`+`communicate`；其余零星 |
| telecom | 114 | **全部只有 `actions`** |

⚠️ `reward_basis` **不在数据里** —— 它是 `EvaluationCriteria` 的 `default_factory`
（`[DB, COMMUNICATE]`）,而且**不由那三个字段推出来**。
⇒ 要复现官方的乘算,得读它的代码,不能从 `tasks.json` 猜。

## ★ 现在做到了哪一步

| | |
|---|---|
| ✅ `tasks()` | 278 条,按时域分,policy 进 prompt、用户那一半不进 |
| ✅ `downloads()` | 钉 tag 的 tarball,抽出三个域的 `tasks.json` / `policy.md` / `db.*` |
| ❌ 工具 | 每个域一套（airline / retail / telecom 各不同),还没接 |
| ❌ 判分 | 要重放 + 数据库哈希,还没接 |
| ❌ 用户模拟器 | **全项目第一个两方任务** —— 它不是 loader 的工作量,是**新的一层** |

★ 后面三样都**明说没做**,不返回 0 —— 理由同 ALFWorld:
**一个缺一块看起来像一个考零分**,而那是这类工作里最难发现的一种错。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Trajectory

#: ★ **钉 tag**,不是 `main` —— 理由见模块头（HEAD 已经是 τ³-bench v1.0.1）。
TAU2_TAG = "v0.1.0"
TAU2_COMMIT = "37199f36924c"

#: 三个域,以及论文 base 划分的条数。**实测 = 论文的 278。**
DOMAINS: dict[str, int] = {"airline": 50, "retail": 114, "telecom": 114}
TOTAL_TASKS = sum(DOMAINS.values())

#: 每个域要抽的文件。`db.*` 的后缀两个域不同（json / toml）——
#: ★ 写死一个后缀会让 telecom 缺库,而缺库的报错不会说「是后缀写错了」。
DOMAIN_FILES: dict[str, tuple[str, ...]] = {
    "airline": ("tasks.json", "policy.md", "db.json"),
    "retail": ("tasks.json", "policy.md", "db.json"),
    "telecom": ("tasks.json", "main_policy.md", "db.toml"),
}


class NeedsSimulator(RuntimeError):
    """要跑两方对话,但用户模拟器还没接。

    ★ 和「考了 0 分」是两件事。**一个缺一层看起来像一个算法读数**,
      是这类工作里最难发现的一种错 —— 所以这里抛,不返回 0。
    """


@dataclass
class Tau2Bench:
    """τ²-bench 的一个域（或全部）。"""

    name: str = "tau2-bench"
    dataset_version: str = f"tau2-bench@{TAU2_TAG}"
    data_dir: Path = field(default_factory=lambda: DATASET_DIR / "tau2-bench")
    #: 空 = 三个域全跑（278 条）。
    domain: str = ""
    limit: int | None = None
    seed: int = 0

    def _domains(self) -> list[str]:
        if self.domain:
            if self.domain not in DOMAINS:
                raise ValueError(f"{self.name}: 没有这个域 {self.domain!r}；有 {sorted(DOMAINS)}")
            return [self.domain]
        return list(DOMAINS)

    def _read(self, domain: str, fname: str):
        path = self.data_dir / domain / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: 缺 {path}\n"
                f"  跑 `python3 -m experiments.scripts.datasets --fetch {self.name}`"
            )
        return path.read_text(encoding="utf-8")

    # ── 协议：任务 ──────────────────────────────────────────

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """★ **`Task.prompt` 是域 policy,不是 user_scenario** —— 理由见模块头。

        ★ 抽样用全项目同一个做法（`rewoo_port.rewoo_draw`）——
          「用 seed 定一批题」应当只有一种写法。
        """
        from experiments.benchmark.rewoo_port import rewoo_draw

        if split not in ("test", "base", "all"):
            raise ValueError(f"{self.name}: base 划分之外没有别的，收到 {split!r}")

        for domain in self._domains():
            rows = json.loads(self._read(domain, "tasks.json"))
            expected = DOMAINS[domain]
            if len(rows) != expected:
                # ★ 条数对不上就炸 —— 见模块头:换版本连「哪 278 条」都会变。
                raise ValueError(
                    f"{self.name}: {domain} 有 {len(rows)} 条，而 v0.1.0 的 base 是 {expected} 条。\n"
                    f"  → 版本没钉住？HEAD 已经是 τ³-bench,base 划分在另一个文件里"
                )
            policy_file = "main_policy.md" if domain == "telecom" else "policy.md"
            policy = self._read(domain, policy_file)
            for i in rewoo_draw(len(rows), limit, seed):
                row = rows[i]
                yield Task(
                    task_id=f"{self.name}/{domain}/{row.get('id', i)}",
                    # ★ agent 看到的是**业务规则**,不是用户想干什么。
                    prompt=f"{policy}\n\n---\n\nA customer is contacting you. Help them.",
                    gold="(由数据库状态判定)",
                    oracle_context=None,
                    meta={
                        "domain": domain,
                        "row": i,
                        # ↓↓↓ **这三个都不进 prompt** —— 它们是别人的输入 ↓↓↓
                        # 用户模拟器的 persona + 指令
                        "user_scenario": row.get("user_scenario"),
                        # 出题人的备注（说明这道题在考什么）
                        "description": row.get("description"),
                        # 判分依据（金标动作 / 该说的话 / 自然语言断言）
                        "evaluation_criteria": row.get("evaluation_criteria"),
                        "answer_kind": "dialogue",
                    },
                )

    # ── 还没接的三样：明说 ──────────────────────────────────

    def tools(self) -> Sequence[object]:
        self._not_yet("工具")

    def tool_impls(self) -> dict[str, object]:
        self._not_yet("工具")

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        self._not_yet("判分")

    def check(self, task: Task, answer: str) -> bool:
        self._not_yet("判分")

    def _not_yet(self, what: str) -> None:
        raise NeedsSimulator(
            f"{self.name}: {what}还没接。已经做完的是**任务枚举**"
            f"（{TOTAL_TASKS} 条 = 50+114+114,和论文一致）与下载声明。\n"
            f"  ★ 还没接的原因不是工作量,是**它是两方的**:\n"
            f"    agent 之外还要一个 **LLM 用户模拟器**扮演顾客 —— "
            f"而前面所有数据集都是一个 agent 对着静态题目。\n"
            f"  ★ 判分也不是「答对了没有」:是**整个数据库的状态哈希相等**"
            f"（`evaluator_env.py`），要重放两遍（agent 的动作 / 金标动作）。\n"
            f"  ⚠️ 现在抛,**不是返回 0 分** —— 缺一层和考零分在表上长得一样"
        )

    # ── 下载 ────────────────────────────────────────────────

    def downloads(self) -> Sequence[DownloadSpec]:
        files: list[str] = []
        for domain, names in DOMAIN_FILES.items():
            files.extend(f"{domain}/{n}" for n in names)
        return [DownloadSpec(
            dataset=self.name,
            kind="http",
            locator=f"https://codeload.github.com/sierra-research/tau2-bench/tar.gz/refs/tags/{TAU2_TAG}",
            files=tuple(files),
            revision=TAU2_COMMIT,
            size_hint="~56 MB（整个仓库；本数据集约 25 MB）",
            note=(f"★ **必须钉 {TAU2_TAG}** —— 仓库 HEAD 已经是 τ³-bench v1.0.1,"
                  f"而它自己的 README 写着 `<1.0.1` 的结果不能和 `>=1.0.1` 比。\n"
                  f"  ★ 而 v0.1.0 的 `tasks.json` **就是 base 划分**（278 条）;"
                  f"HEAD 上多了 `split_tasks.json`,base 要从里面挑 —— 换版本连哪 278 条都会变。\n"
                  f"  ★ `test` 那一列在排行榜口径下要跑 **pass^k = 4 次/任务**"
                  f"（`comb(success,k)/comb(trials,k)`）"),
        )]


__all__ = ["Tau2Bench", "NeedsSimulator", "DOMAINS", "DOMAIN_FILES",
           "TOTAL_TASKS", "TAU2_TAG", "TAU2_COMMIT"]
