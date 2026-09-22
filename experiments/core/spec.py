"""★ 结果 schema —— **缺字段的结果不收**。

这是整个 pipeline 里最重要的一份文件,理由在 `docs/PROTOCOL-experiments-*.md`:
七个人各写各的,最后合不起来**不报错**,只是第 4 周发现七套 CSV 拼不到一起。

**怎么做到「不收」**:必填字段在 dataclass 里**没有默认值**。
于是少一个字段 = 构造时 `TypeError`,**在写盘之前就炸** ——
比事后校验强,因为它不给「先写下来回头补」留余地。

字段来源:`docs/PLAN-confidence-*.md` §3(成本 / 时间 / 置信度三节)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

# ═══════════════════════════════════════════════════════════
# 元数据 —— PLAN §3.7「每一行都要带」
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelRef:
    """模型身份。**版本必须记** —— 「刷爆」是相对模型和时间说的。"""

    id: str
    version: str
    provider: str


@dataclass(frozen=True)
class RunMeta:
    """这一行是**怎么跑出来的**。缺任何一项,这个数字别人就复现不了。"""

    dataset: str
    dataset_version: str
    split: str
    task_count: int
    arm: str
    generator: ModelRef
    decider: ModelRef
    thinking_budget: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    prompt_hash: str
    commit: str
    # ★ 脏工作区跑出来的数字,连跑它的人都复现不了 —— 所以必须盖章
    dirty: bool
    # ★ 同一件事我们量到 254ms vs 65–90ms,取决于对端
    region: str
    cold_start: bool
    max_steps: int
    timeout_s: float


# ═══════════════════════════════════════════════════════════
# 成本 —— PLAN §3.2
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Cost:
    """★ 几个必须**拆开**的数,理由都是踩过的:

    - `input_tokens_cached` / `uncached`:不分列,谁重复上下文谁的数字就失真
    - `output_tokens_reasoning`:分出来才能和「思考程度」那个轴对齐
    - `tool_calls_necessary` / `exploratory`:有的评分器**明确不惩罚探索性调用**,
      不拆的话,我们的调用数和惩罚探索的评分器根本不可比
    """

    llm_calls: int
    decision_requests: int
    questions_per_request: float
    tool_calls: int
    tool_calls_necessary: int
    tool_calls_exploratory: int
    input_tokens_cached: int
    input_tokens_uncached: int
    output_tokens_reasoning: int
    output_tokens_visible: int
    usd: float


# ═══════════════════════════════════════════════════════════
# 时间 —— PLAN §3.3「拆到和我们一样细」
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Timing:
    """★ `framework_ms` 和 `retry_ms` **必须有**。

    不留这两行,就永远分不清差距里有多少是自己的开销、多少是对面在退避。
    我们踩过:`generate` 那步墙钟 21.8s,meter 只记 11.0s,
    差掉的一半被误读成「JevLoop 慢 10 倍」。
    """

    wall_ms: float
    model_handshake_ms: float
    model_ttft_ms: float
    model_after_ttft_ms: float
    decision_handshake_ms: float
    decision_compute_ms: float
    tool_ms: float
    framework_ms: float
    retry_ms: float
    round_trips: int


# ═══════════════════════════════════════════════════════════
# 一次运行的结果
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Result:
    """一道题 × 一个 arm × 一个 seed 的结果。

    ⚠️ **所有字段都没有默认值** —— 这是故意的,见模块头。
    """

    # —— 身份
    run_id: str
    meta: RunMeta
    task_id: str
    # —— 效果（PLAN §3.1）
    correct: bool
    score: float
    steps: int
    first_divergence_step: int
    escalated: bool
    gate_false_reject: bool
    gate_false_deny: bool
    failure_class: str
    # —— 成本与时间
    cost: Cost
    timing: Timing
    # —— 这次运行产出的文件（相对 run 目录）
    artifacts: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


# ═══════════════════════════════════════════════════════════
# 逐条记录 —— 写进 trace.jsonl / usage.jsonl
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DecisionRecord:
    """★ **一行一条判定**,不是一行一个任务。

    `confidence` 和 `correct` 必须在同一行 —— 分桶 / ECE / Brier / AUC /
    分离度 / 风险–覆盖 **全部**从这一对算出来。少了任何一个,RQ2 就是空的。

    `batch` 也要有:一次请求判多路时**按批计时**,按记录求和会多算几倍
    （我们踩过,`decisionMs` 一度比整轮墙钟还大）。
    """

    run_id: str
    task_id: str
    step: int
    node: str
    answer: str
    confidence: float
    correct: bool
    batch: int
    latency_ms: float


@dataclass(frozen=True)
class UsageRecord:
    """逐调用一行。字段对应 `Cost` 里那些**必须拆开**的列。"""

    run_id: str
    task_id: str
    kind: str  # "llm" | "decision" | "tool"
    name: str
    input_tokens_cached: int = 0
    input_tokens_uncached: int = 0
    output_tokens_reasoning: int = 0
    output_tokens_visible: int = 0
    usd: float = 0.0
    handshake_ms: float = 0.0
    ttft_ms: float = 0.0
    after_ttft_ms: float = 0.0
    compute_ms: float = 0.0
    questions_in_batch: int = 0


# ═══════════════════════════════════════════════════════════
# 校验 —— 给「从盘上读回来」用
# ═══════════════════════════════════════════════════════════

REQUIRED_RESULT_FIELDS: tuple[str, ...] = (
    "run_id",
    "meta",
    "task_id",
    "correct",
    "score",
    "steps",
    "first_divergence_step",
    "escalated",
    "gate_false_reject",
    "gate_false_deny",
    "failure_class",
    "cost",
    "timing",
    "artifacts",
)

REQUIRED_TRACE_FIELDS: tuple[str, ...] = (
    "run_id",
    "task_id",
    "step",
    "node",
    "answer",
    "confidence",
    "correct",
    "batch",
    "latency_ms",
)


def missing_fields(record: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """缺哪些字段。返回空列表 = 完整。

    `summarize` 用它拒收残缺的行 —— **不许猜、不许补默认值**。
    一个补过默认值的数字,和编出来的没有区别。
    """
    return [f for f in required if f not in record]
