"""`gsm8k` —— 小学数学应用题。**第一个真 loader,后面照着它写。**

**出处**:Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton,
Nakano, Hesse, Schulman. *Training Verifiers to Solve Math Word Problems*.
arXiv:2110.14168。数据:HF `openai/gsm8k`,配置 `main`。
训练 7,473 / 测试 1,319 条;答案字段末尾用 `#### <数>` 给出最终数值。

## ⚠️ 它**验不到工具那条路** —— 第二个 loader 必须验

`tools()` 返回**空列表**。所以这个文件能给你看的样板是:
**取数据 / 按 seed 抽样 / 从答案字段抠金标 / 对齐官方判分口径 / 报失败分类**。
**工具定义(`tools()` + `tool_impls()`)这一块它一行都没碰** ——
而那恰恰是判定模型能不能枚举候选的前提。别照着这个文件以为工具不用管。

（文献里 GSM8K 是带工具的:ReWOO 的 baseline 给了 `LLM` / `WolframAlpha` / `Calculator` 三个。
我们默认不给,理由见 `docs/PLAN-*.md` §1:**那三个工具没有带标注的「该选哪个」岔路**,
`needsTool` 会恒为否、零方差。`with_calculator=True` 留着,给要用它对齐文献的人。）

## ★ 判分口径是这里唯一容易出错的地方

GSM8K 的分数**对「怎么从模型输出里抠出那个数」极其敏感**,而两种抠法都有出处:

| 口径 | 规则 | 谁在用 |
|---|---|---|
| `strict` | 模型必须写出 `#### <数>` | lm-evaluation-harness 的 `exact_match,strict-match` |
| `flexible` | 取输出里**最后一个数** | lm-evaluation-harness 的 `exact_match,flexible-extract` |

**两个数都会报,而且必须写明报的是哪个** —— 只报一个还不说口径,别人复现不出来。
默认头条口径是 `flexible`(文献里常见的那个)。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from experiments.core.download import DownloadSpec
from experiments.core.types import Judgment, Task, Tool, Trajectory

DATASET_ID = "openai/gsm8k"
CONFIG = "main"
# ★ 三层 parent：这个文件在 `experiments/benchmark/gsm8k/gsm8k.py`。
#   早先只写了两层,于是数据落到了 `experiments/benchmark/dataset/` ——
#   **那个路径不在 .gitignore 里**,13 个数据文件一度进了暂存区（实测）。
#   路径推导这种东西要么写死层级并配一个测试,要么就别推。
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]

# 数据落 `experiments/dataset/` —— 和 `dataset/` 的约定一致（那份目录不进 git）。
DEFAULT_HF_HOME = EXPERIMENTS_DIR / "dataset" / "hf"

# ★ 国内直连 huggingface.co 不通（实测 HTTP 000），hf-mirror.com 通。
#   已设 `HF_ENDPOINT` 就尊重它 —— 别替用户改主意。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

ANSWER_MARK = "####"

# ★ 输出契约：**归 benchmark**，因为它说的是「答案长什么样」，是任务的一部分。
#   它**不说**「要不要推理」—— 那是 arm 的事，写了就等于替所有臂选了范式。
ANSWER_CONTRACT = "Give the final answer as a single number on its own line."
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
CALCULATOR = Tool(
    name="calculate",
    description="Evaluate a pure arithmetic expression, e.g. '16 - 3 - 4'.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)


@dataclass
class Gsm8k:
    """GSM8K。`split="test"` 是评测用的那个。"""

    # `strict` | `flexible` —— 见模块头。**两个都算得出来,报的时候要写明。**
    headline_extract: str = "flexible"
    with_calculator: bool = False
    # 测试直接喂 rows,不联网（见 tests/）
    rows: dict[str, list[dict[str, Any]]] | None = None
    hf_home: Path = field(default_factory=lambda: DEFAULT_HF_HOME)

    name: str = "gsm8k"
    dataset_version: str = field(init=False, default="unpinned")
    _loaded: dict[str, list[dict[str, Any]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if self.headline_extract not in ("strict", "flexible"):
            raise ValueError(f"headline_extract 只能是 strict / flexible，收到 {self.headline_extract!r}")

    # ── 数据 ────────────────────────────────────────────────

    def _rows(self, split: str) -> list[dict[str, Any]]:
        if self.rows is not None:
            if split not in self.rows:
                raise ValueError(f"喂进来的 rows 里没有 {split!r}（只有 {sorted(self.rows)}）")
            return self.rows[split]
        if split in self._loaded:
            return self._loaded[split]

        from datasets import load_dataset  # 只在真要拿数据时 import

        ds = load_dataset(DATASET_ID, CONFIG, split=split)
        self._loaded[split] = list(ds)
        # ★ 指纹进版本号 —— 「哪个版本的数据」这句话必须能被复现,
        #   而 HF 的 fingerprint 是唯一能自动拿到的那个标识。
        fingerprint = getattr(ds, "_fingerprint", None)
        if fingerprint:
            self.dataset_version = f"{DATASET_ID}@{fingerprint}"
        return self._loaded[split]

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        rows = self._rows(split)
        if limit is not None and limit < len(rows):
            # ★ 抽样必须由传进来的 seed 决定 —— 否则「我们跑的是哪 300 条」说不清。
            #   用等距步长而不是 random.shuffle：同样是确定的,但不依赖 Python 的
            #   随机数实现（跨版本会变）,而且样本在全集上铺得开。
            step = len(rows) / limit
            offset = seed % len(rows)
            picked = [rows[(offset + int(i * step)) % len(rows)] for i in range(limit)]
        else:
            picked = rows

        for i, row in enumerate(picked):
            gold = extract_gold(str(row["answer"]))
            yield Task(
                task_id=f"gsm8k/{split}/{i}",
                # ★ **输出契约归 benchmark**（见 core/bench.py）：它说「答案长什么样」，
                #   但**不说「要不要推理」** —— 那是 arm 的交互方式。
                #   早先这里写的是「Show your reasoning, then give the final number」,
                #   而 direct 臂又追加「Answer with the final answer only」——
                #   两条指令打架,模型听最后一条,只输出 1 个 token（实测 out=1）。
                prompt=f"{str(row['question']).strip()}\n\n{ANSWER_CONTRACT}",
                gold=gold,
                # GSM8K 没有可检索的外部证据 —— 所以「direct + 金标证据」那一臂
                # 在这里退化成和 direct 一样。**这是它的一个短板,记在这里别忘。**
                oracle_context=None,
                meta={"gold_raw": str(row["answer"])[-40:]},
            )

    def downloads(self) -> Sequence[DownloadSpec]:
        return [DownloadSpec(
            dataset=self.name,
            kind="hf-dataset",
            locator=f"{DATASET_ID}:{CONFIG}",
            revision="main",
            size_hint="~2 MB（7,473 + 1,319 条）",
            note="纯文本，无前置；`revision` 未钉 —— 已用 fingerprint 记进 dataset_version",
        )]

    # ── 工具 ────────────────────────────────────────────────

    def tools(self) -> Sequence[Tool]:
        return (CALCULATOR,) if self.with_calculator else ()

    def tool_impls(self) -> dict[str, Callable[..., str]]:
        return {"calculate": _calculate} if self.with_calculator else {}

    # ── 判分 ────────────────────────────────────────────────

    def check(self, task: Task, answer: str) -> bool:
        """**可选方法**,给 Reflexion 的 Evaluator 用。这里就是「对不对」。"""
        return numeric_equal(extract_pred(answer, self.headline_extract), str(task.gold))

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        answer = trajectory.final_answer or ""
        if not answer.strip():
            return Judgment(correct=False, score=0.0, detail="没有给出答案", failure_class="no_answer")

        got_strict = extract_pred(answer, "strict")
        got_flexible = extract_pred(answer, "flexible")
        got = got_strict if self.headline_extract == "strict" else got_flexible
        want = str(task.gold)

        if numeric_equal(got, want):
            return Judgment(correct=True, score=1.0,
                            detail=f"{got} == {want}（口径 {self.headline_extract}）")
        if got == "":
            # 抠不出数 → 是**格式**失败,不是算错。分开报才能看出该怪谁。
            return Judgment(correct=False, score=0.0,
                            detail=f"抠不出数值（{self.headline_extract}）",
                            failure_class="no_numeric_answer")
        return Judgment(correct=False, score=0.0,
                        detail=f"答 {got!r}，期望 {want!r}（口径 {self.headline_extract}）",
                        failure_class="wrong_answer")


# ═══════════════════════════════════════════════════════════
# 纯函数 —— 判分逻辑全在这里，所以它可测
# ═══════════════════════════════════════════════════════════


def extract_gold(answer_field: str) -> str:
    """金标:答案字段里 `####` 后面那个数。

    抠不到就返回**整段**,并让上层看得见 —— GSM8K 的金标字段偶尔不规范,
    静默返回空串会把「数据有问题」变成「模型答错了」。
    """
    if ANSWER_MARK not in answer_field:
        return answer_field.strip()
    return answer_field.split(ANSWER_MARK)[-1].strip()


def extract_pred(text: str, mode: str) -> str:
    """从模型输出里抠出那个数。

    - `strict`:必须出现 `####`;没写就是**没答**(返回空串),不是「取最后一个数」
    - `flexible`:取整段里最后一个数

    ★ **两种都不会「猜」**:抠不到就返回空串,由 `score` 归到 `no_numeric_answer`。
    """
    if mode == "strict":
        if ANSWER_MARK not in text:
            return ""
        text = text.split(ANSWER_MARK)[-1]
    found = _NUMBER.findall(text)
    if not found:
        return ""
    return found[-1].replace(",", "").replace("$", "")


def numeric_equal(a: str, b: str) -> bool:
    """数值比较。抠不出数是 `False`,**不是异常** —— 判分器不许把整轮带走。"""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _calculate(expression: str) -> str:
    """一个**只做四则运算**的计算器。

    ★ 用 `ast` 白名单而不是 `eval` —— 这条路径的输入来自模型输出,
    直接 `eval` 等于把任意代码执行权交出去。`eval` 在这里不是「方便」,
    是一个必须挡住的洞。
    """
    import ast

    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
               ast.FloorDiv, ast.Mod)
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return f"Error: 表达式里有不允许的语法 {type(node).__name__}"
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # noqa: BLE001 —— 表达式是模型给的,什么都可能
        return f"Error: {type(exc).__name__}"
    # 整数就出整数,免得 18.0 和 18 在字符串上不等（数值比较本来不怕,但看着别扭）
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
