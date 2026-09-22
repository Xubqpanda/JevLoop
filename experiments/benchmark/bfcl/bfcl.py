"""`bfcl` —— 函数调用。**第一个带 tool definition 的 loader,后面照着它写。**

**出处**:Berkeley Function Calling Leaderboard（BFCL）。
论文:Patil, Mao, Ji, Yan, Suresh, Stoica, Gonzalez. *The Berkeley Function Calling
Leaderboard (BFCL) and the Gorilla LLM*（ICML 2025）。数据:
HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard`（**JSONL，不是 JSON 数组**）。
官方仓库与评分器:`ShishirPatil/gorilla` → `berkeley-function-call-leaderboard`。

## ★ 为什么先做它:**工具定义就在数据里**

每一条自带 `function`,内容是标准的 JSON Schema。所以它**不需要**索引、不需要模拟器、
不需要 Docker、不需要真环境 —— **工具定义这条路径可以在完全没有基础设施的情况下验完**。
`gsm8k` 验不到的那一半（`tools()` / `tool_impls()`）由它补上。

## 取哪些子集

| 子集 | 条数 | 金标 | 判什么 |
|---|---|---|---|
| **`simple`** | 400 | 在 `possible_answer/BFCL_v3_simple.json` | 该调哪个函数（**只判函数名**）|
| **`irrelevance`** | 240 | **没有金标文件** —— 正确答案就是「不调」 | 该不该调 |

★ `irrelevance` 自足这一点很重要:它不需要额外的金标文件,是**最便宜的逐判定标签**。
而它判的正是 `needsTool`（`docs/PLAN-*.md` §0.7 把这一类记成我们的节点之一）。

## ⚠️ 只判函数名,**不报官方分** —— 以及为什么

官方 BFCL 分数要**完整调用**,含参数值。而参数在数据里长这样::

    {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}

**是「可接受值列表」,形状自由** —— 判定模型只能从枚举里挑,做不到这个。
所以这里判两件事:**函数名选对没有** + **该不该调判对没有**,并在报告里写明这不是官方分。

★ 顺带印证了 `Tool.parameters` 那条约束:**能让判定模型枚举的参数必须来自环境**,
而 BFCL 的参数是自然语言里抽出来的,没有候选来源。

## 两个数据上的坑

1. **`parameters.type` 是 `"dict"` 不是 `"object"`** —— BFCL 的怪癖。`render_tools` 只读
   `properties` / `required`,不受影响,但这里仍然归一化,免得以后有人直接拿它当标准 JSON Schema 用。
2. **`question` 是 `[[{"role": ..., "content": ...}]]`** —— 两层的对话结构。v3 的
   `simple` / `irrelevance` 都是一轮一条,但不要假设它永远是一层。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from experiments.core.download import DownloadSpec
from experiments.core.types import Judgment, Task, Tool, Trajectory

DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]  # experiments/
DEFAULT_HF_HOME = EXPERIMENTS_DIR / "dataset" / "hf"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 子集名 → (题目文件, 金标文件或 None)
# ★★ **这里是 v3，只能是 v3 —— 因为 HF 上只有 v3。**
#
#   `gorilla-llm/Berkeley-Function-Calling-Leaderboard` 里的文件全叫 `BFCL_v3_*`。
#   **v4 的 agentic 部分（web search 200 / memory 465 / format sensitivity 5,200）
#   只在 GitHub 仓库里，不在 HF 上。** 所以：
#
#     v3  →  HF 的散文件（这个 loader 现在走的路）
#     v4  →  **必须 clone 官方仓库** `ShishirPatil/gorilla`，取
#            `berkeley-function-call-leaderboard/bfcl_eval/data/`
#
#   ★ 两件事因此一起解决：**v4 的数据 + 官方评分器（`bfcl-eval` 包）**。
#     自己按函数名判分是权宜之计（见模块头「不报官方分」那一节），
#     一旦接了官方评分器就该换过去 —— 而且**换判分器之后数字不可比**，
#     所以新旧数字要分开报。
#
#   子集名带版本号（`v3-simple`），因为 `bfcl-simple` 这个词在 v4 下含义不同：
#   v4 把 v3 四个子集并进 30% 的 Multi-Turn 权重里，另加了 Memory 和 Web Search。
VERSION = "v3"

SUBSETS: dict[str, tuple[str, str | None]] = {
    "v3-simple": ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json"),
    "v3-irrelevance": ("BFCL_v3_irrelevance.json", None),
    "v3-multiple": ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"),
    "v3-live-simple": ("BFCL_v3_live_simple.json", "possible_answer/BFCL_v3_live_simple.json"),
    # ⚠️ v4 的子集不在这里 —— 它们要 clone 官方仓库才有数据,见上面的说明。
}

# ★ 输出契约归 benchmark（见 core/bench.py 的分工表）。
#   它说的是「怎么算回答了这个任务」,**不说要不要推理** —— 所以同一句
#   在 `simple` 和 `irrelevance` 上都成立,两个子集的差别全在数据里。
ANSWER_CONTRACT = (
    "Decide whether one of the available functions answers the request.\n"
    "- If one does, call it.\n"
    "- If none of them applies, say that no function is needed."
)


@dataclass
class Bfcl:
    """BFCL 的一个子集。`subset` 见 `SUBSETS`。"""

    subset: str = "v3-simple"
    rows: dict[str, list[dict[str, Any]]] | None = None  # 测试直接喂，不联网
    hf_home: Path = field(default_factory=lambda: DEFAULT_HF_HOME)

    name: str = field(init=False, default="bfcl")
    dataset_version: str = field(init=False, default="unpinned")
    _questions: list[dict[str, Any]] = field(init=False, default_factory=list)
    _answers: dict[str, list[dict[str, Any]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.subset not in SUBSETS:
            raise ValueError(f"未知子集 {self.subset!r}，可选 {sorted(SUBSETS)}")
        # ★ 臂名和数据集名要能分开 —— `bfcl/simple` 和 `bfcl/irrelevance`
        #   是两张不同的表,`log/<name>/` 也必须是两个目录。
        self.name = f"bfcl-{self.subset}"  # 已经是 `bfcl-v3-simple` 这种带版本的形状
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # ── 数据 ────────────────────────────────────────────────

    def _load(self, split: str) -> list[dict[str, Any]]:
        subject, answer_file = SUBSETS[self.subset]
        if self.rows is not None:
            if subject not in self.rows:
                raise ValueError(f"喂进来的 rows 里没有 {subject!r}")
            self._questions = self.rows[subject]
            self._answers = {r["id"]: r["ground_truth"] for r in self.rows.get(answer_file or "", [])}
            self.dataset_version = f"{DATASET_ID}#{VERSION}/{self.subset}@fixture"
            return self._questions

        from huggingface_hub import hf_hub_download

        q_path = hf_hub_download(DATASET_ID, subject, repo_type="dataset")
        self._questions = _read_jsonl(q_path)
        if answer_file:
            a_path = hf_hub_download(DATASET_ID, answer_file, repo_type="dataset")
            self._answers = {r["id"]: r["ground_truth"] for r in _read_jsonl(a_path)}
        self.dataset_version = f"{DATASET_ID}#{VERSION}/{self.subset}"
        return self._questions

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        rows = self._load(split)
        picked = sample(rows, limit, seed)
        names = {t.name for t in self.tools()}

        for i, row in enumerate(picked):
            qid = str(row["id"])
            functions = list(row.get("function") or [])
            # ★ **这一条题自己的工具集。** 不同的题给不同的函数 —— 候选集必须
            #   跟着题走,否则模型会去调一个这道题根本没给它的函数。
            gold_names = _gold_names(self._answers.get(qid))
            yield Task(
                task_id=f"{self.name}/{qid}",
                prompt=f"{first_user_message(row['question'])}\n\n{ANSWER_CONTRACT}",
                # 金标 = 该调的函数名列表；`irrelevance` 下是**空列表**（不该调）。
                # 空列表和「没有金标」是两件事,所以用 None 表示后者。
                gold=gold_names,
                oracle_context=None,
                meta={
                    "subset": self.subset,
                    "functions": [f.get("name") for f in functions],
                    "unknown_tools": [f.get("name") for f in functions if not _tool_name(f) in names]
                    if names else [],
                    "gold_calls": gold_names,
                },
                # 工具集随题走 → 挂在题目上,`tools()` 只做兜底
                tools=tuple(_to_tool(f) for f in functions),
            )

    def downloads(self) -> Sequence[DownloadSpec]:
        """★ 一个数据集两个子集 → **两条声明**。

        `dataset` 写的是各自 loader 的 `name`（`bfcl-simple` / `bfcl-irrelevance`）,
        因为那才是「谁需要这份数据」的那根线。
        """
        subject, answer_file = SUBSETS[self.subset]
        files = (subject,) if answer_file is None else (subject, answer_file)
        return [DownloadSpec(
            dataset=self.name,
            kind="hf-file",
            locator=DATASET_ID,   # ★ HF 上只有 v3；v4 要 clone 官方仓库（见 SUBSETS 的说明）
            files=files,
            revision="main",
            size_hint="~1 MB（JSONL 散文件）",
            note="**JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下",
        )]

    # ── 工具 ────────────────────────────────────────────────

    def tools(self) -> Sequence[Tool]:
        """★ 兜底用的全集。

        BFCL **每题给的函数不同**,所以真正该用的是 `Task.tools`。
        这里返回所有题目的函数并集,只为满足协议 —— 谁要用它,自己知道那是什么。
        """
        seen: dict[str, Tool] = {}
        for row in (self._questions or []):
            for f in row.get("function") or []:
                tool = _to_tool(f)
                seen.setdefault(tool.name, tool)
        return tuple(seen.values())

    def tool_impls(self) -> dict[str, Callable[..., str]]:
        """★ **每个函数都给一个存根,而且存根会如实说「没执行」。**

        BFCL 判的是「该不该调、调哪个」,**它不执行任何函数** —— 没有环境。
        但 `ToolExecutor` 有一条硬规矩:**声明了就必须有实现**,否则开跑前抛。
        那条规矩是好的（它挡住「声明了工具却忘了接」),所以这里不去绕过它,
        而是给一个**说真话的存根**:观察里写明这个函数没有被执行。

        为什么不干脆返回空字典:那样任何一条 BFCL 的题都跑不起来,
        而错误信息会指向 `ToolExecutor`,看起来像配置问题 —— 其实是这个 loader 的语义。
        """
        return {name: _not_executed for name in self._all_tool_names()}

    def _all_tool_names(self) -> list[str]:
        names: list[str] = []
        for row in self._questions or []:
            for f in row.get("function") or []:
                name = _tool_name(f)
                if name and name not in names:
                    names.append(name)
        return names

    # ── 判分 ────────────────────────────────────────────────

    # ★ **故意不实现 `check()`。**
    #
    #   Reflexion 的 Evaluator 要的是「这个答案对不对」,而 BFCL 判的是
    #   **轨迹里调用了什么**,光看答案文本判不出来。给一个「看起来能用」的
    #   文本判据比不给更糟 —— 它会让 Reflexion 臂的读数悄悄失真。
    #
    #   不实现它,`reflexion` 臂会**明确报错**并告诉你要么实现 `check()`,
    #   要么改用 `reflexion-selfeval`（见 core/agent.py 的说明）。

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        gold = task.gold if isinstance(task.gold, list) else []
        called = [s.action.name for s in trajectory.steps if s.action.kind == "tool"]

        if not called:
            if not gold:
                return Judgment(correct=True, score=1.0, detail="正确地没有调用任何函数")
            return Judgment(correct=False, score=0.0,
                            detail=f"一个函数都没调，期望 {gold}",
                            failure_class="needs_tool_missed")
        if not gold:
            return Judgment(correct=False, score=0.0,
                            detail=f"不该调却调了 {called}",
                            failure_class="unnecessary_tool_call")
        if sorted(called) == sorted(gold):
            return Judgment(correct=True, score=1.0, detail=f"调对了 {called}（**只判函数名**）")
        return Judgment(correct=False, score=0.0,
                        detail=f"调了 {called}，期望 {gold}",
                        failure_class="wrong_tool")


# ═══════════════════════════════════════════════════════════
# 纯函数 —— 数据形状全在这里，所以它可测
# ═══════════════════════════════════════════════════════════


def _not_executed(*args: Any, **kwargs: Any) -> str:
    """BFCL 的存根工具。**它如实说自己没被执行** —— 不假装成功。"""
    return "(BFCL does not execute functions; the call has been recorded and is not run.)"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """★ BFCL 的文件是 **JSONL**（一行一个对象），不是 JSON 数组。

    直接 `json.load` 会报 `Extra data: line 2 column 1` —— 这个错看起来像文件坏了,
    其实是格式不同。第一次读就是这么踩的。
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """等距抽样，位置由 `seed` 决定。

    ★ 等距而不是 `random.shuffle`:同样是确定的,但不依赖 Python 随机数的实现
    （跨版本会变),而且样本在全集上铺得开。
    """
    if limit is None or limit >= len(rows):
        return rows
    step = len(rows) / limit
    offset = seed % len(rows)
    return [rows[(offset + int(i * step)) % len(rows)] for i in range(limit)]


def first_user_message(question: Any) -> str:
    """从 `[[{role, content}]]` 里取第一条 user 消息。

    ★ 不要假设它是一层 —— v3 的 `simple` 是一层,但结构上允许两层;
    真遇到两层时**把整个对话拼起来**,而不是只取第一句。
    """
    turns = question if isinstance(question, list) else [question]
    parts: list[str] = []
    for turn in turns:
        messages = turn if isinstance(turn, list) else [turn]
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                parts.append(str(msg.get("content", "")))
    return "\n".join(parts).strip()


def _to_tool(function: dict[str, Any]) -> Tool:
    """BFCL 的函数定义 → 我们的 `Tool`。"""
    params = dict(function.get("parameters") or {})
    # ★ BFCL 用 `"dict"` 当顶层类型（标准 JSON Schema 是 `"object"`）。
    #   `render_tools` 只读 properties/required，本来不受影响，
    #   但归一化一次，免得以后有人直接把它当标准 schema 用。
    if params.get("type") == "dict":
        params["type"] = "object"
    return Tool(
        name=_tool_name(function),
        description=str(function.get("description", "")),
        parameters=params,
    )


def _tool_name(function: dict[str, Any]) -> str:
    """函数名。BFCL 里有 `triangle_properties.get` 这种带点的名字 —— **原样保留**,
    不要替换成下划线:那是它真实的名字,改了就和官方对不上。"""
    return str(function.get("name", ""))


def _gold_names(ground_truth: Any) -> list[str]:
    """从 `[{"fn": {"arg": [v, ...]}}]` 里取出函数名。

    ★ **只取名字。** 参数值是「可接受值列表」,形状自由,判定模型枚举不了 ——
    理由见模块头。所以这个 loader **不报官方 BFCL 分数**,只报函数名。
    """
    if not isinstance(ground_truth, list):
        return []
    names: list[str] = []
    for call in ground_truth:
        if isinstance(call, dict):
            names.extend(str(k) for k in call)
    return names
