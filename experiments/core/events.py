"""事件流 —— 一次调用（或一段工作）**一条记录,自己装齐**。

## 这是从哪来的

**移植自 Inspect（UK AISI + Meridian Labs）的 transcript 事件模型** ——
<https://inspect.aisi.org.uk/analysis.html.md> 与 `inspect_ai.event`。
**我们没有依赖它,是照着它的形状自己写了一份**:理由和 `AGENTS.md` §8.7
「移植而不引用」一样 —— 这个仓库要能**单独 clone 就跑**,而多一个重依赖
（pydantic + 70 个包）对「clone 下来跑不起来」这类伤害,比多写 200 行更大。

**核对依据**:`docs/PROBE-inspect-2026-09-22.md` 是对着真包实测出来的,不是读文档猜的。

## 搬过来的那条,具体是什么

**在它之前,我们是三个文件靠 `(run_id, task_id, call)` 三个键 join**:

```
prompts.jsonl      发出去什么
completions.jsonl  回来什么
usage.jsonl        token 和时间
```

**它是一条记录装齐**（`ModelEvent`）:

```
call.request · call.response · output.usage · timestamp · completed · working_time · retries · error
```

**改成一体的好处不是省文件,是「一次调用」这个对象**在盘上只有一个权威表示 ——
三个文件 join 的地方,任何一处漏写都会让那次调用**看起来像没发生过**,
或者更糟:让 token 数和文本对不上,而账面看不出来。

## 时间字段:三个,别混

| 字段 | 意思 |
|---|---|
| `timestamp` / `completed` | 墙钟起点与终点（ISO8601,UTC） |
| `working_start` | 从**这次 run 开始**算起的秒 —— 用来把事件按真实顺序排,不受时钟回拨影响 |
| `working_time` | **真正干活的时间** |

★ 最后一个是 Inspect 原生就区分的那条:**样本上有 `total_time` 和 `working_time` 两个数**。
我们自己的 `wall_ms` / `framework_ms` 是同一个区分的另一种说法,这里保留两边都能对上。

## 事件类型

不照抄它的全部（它有 20 多种）,**只留我们真正会读的四种**,理由是
`AGENTS.md` §8.7 那句「精简是**有意的**,不是复制粘贴」:

| type | 什么时候发 | 装什么 |
|---|---|---|
| `model` | 每次生成模型调用 | 请求 + 响应 + usage + 重试 + 错误 |
| `tool` | 每次工具调用 | 名字 + 参数 + 观察 + 错误 |
| `decision_batch` | 每一**批**类型化判定 | 这一批有几个问题 + 逐条判定 |
| `span` | 一段工作的开头或结尾 | 名字 —— **用来给一步单独计时** |

★ 第五种「span」是探针里发现的:`SpanBeginEvent` / `SpanEndEvent` 成对出现,
**每一步的耗时不用手写计时**。我们照这个形状做。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid as uuid_mod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ═══════════════════════════════════════════════════════════
# 词汇
# ═══════════════════════════════════════════════════════════

EVENT_TYPES = ("model", "tool", "decision_batch", "span", "answer")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return uuid_mod.uuid4().hex[:22]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Usage:
    """一次调用的用量。

    ★ **四样必须分开**（`docs/PLAN-*.md` §3.2）—— 而 Inspect 的 `ModelUsage`
    原生就是这四样:`input_tokens_cache_read` / `input_tokens_cache_write` /
    `reasoning_tokens` / `output_tokens`。名字我们照它取,免得以后对不上:

    - 缓存命中 / 未命中分开:不分的话,谁重复上下文谁的数字就失真
    - 推理 / 可见分开:分出来才能和「思考程度」那个轴对齐
    - `total_cost`:Inspect 有,**我们之前没算过** —— 加上,白拿的一列
    """

    input_tokens_cache_read: int = 0
    input_tokens_cache_write: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_cost: float = 0.0

    @property
    def input_tokens_uncached(self) -> int:
        return max(0, self.input_tokens - self.input_tokens_cache_read)

    @property
    def output_tokens_visible(self) -> int:
        return max(0, self.output_tokens - self.reasoning_tokens)


@dataclass(frozen=True)
class Timing:
    """一次调用的时间。**三层都留着** —— 合成一个 latency 就归因不了。

    `handshake` 是我们独有的仪器（实测 254ms,而计算只有 78ms）;
    Inspect **没有**这一段,它只给整体 `working_time`。所以这里两边都留:
    能对上它的口径,也能拆出我们那条结论。
    """

    handshake_ms: float = 0.0
    ttft_ms: float = 0.0
    after_ttft_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.handshake_ms + self.ttft_ms + self.after_ttft_ms


@dataclass
class Event:
    """所有事件的公共部分。字段名字**对齐 Inspect**,方便交叉核对。"""

    type: str
    uuid: str = field(default_factory=new_uuid)
    timestamp: str = field(default_factory=now_iso)
    completed: str | None = None
    # 从这次 run 开始算起的秒 —— 排序用它,不用墙钟
    working_start: float = 0.0
    working_time: float = 0.0
    run_id: str = ""
    task_id: str = ""
    step: int = 0
    # 所属 span 的 uuid（`span` 事件用它表达嵌套）
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class ModelCallEvent(Event):
    """一次生成模型调用。**这是 `ModelEvent` 的对应物。**

    ★ `request` 和 `response` 装在同一张记录里 —— 这是从 Inspect 搬来的那条。
    以前它们分在 `prompts.jsonl` 和 `completions.jsonl` 两个文件,
    靠 `call` 序号 join;任何一处漏写,那次调用就**看起来像没发生过**。
    """

    type: str = "model"
    model: str = ""
    role: str = "generator"
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    timing: Timing = field(default_factory=Timing)
    retries: int = 0
    error: str | None = None


@dataclass
class ToolCallEvent(Event):
    type: str = "tool"
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    # ★ 必要 / 探索性分开 —— 有的评分器**明确不惩罚探索性调用**,
    #   不拆的话调用数和它不可比
    necessary: bool = True
    error: str | None = None


@dataclass
class DecisionBatchEvent(Event):
    """一批类型化判定。**按批发,不按条** —— 一次请求判多路时按条计时会多算几倍。"""

    type: str = "decision_batch"
    questions_in_batch: int = 0
    #: ★ **一次 HTTP 往返算一个。** 和 `questions_in_batch` 分开记 ——
    #: 两者的比值就是「判定有没有被合并」:相等 = 一步一请求（PLAN 说那是
    #: 全项目最大的已知浪费）,远小于 = 合并得对。**合成一个数就分不出来。**
    requests_in_batch: int = 0
    latency_ms: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnswerEvent(Event):
    """这次运行**最终交出来的东西**。对应 Inspect 的 `Sample.output`。

    ★ **有了它,日志才是自足的。** 之前答案只存在于最后一个 `ModelCallEvent`
    的 `response.text` 里,而「哪一句算答案」是各臂自己的事 ——
    于是**重判（`scripts/rescore.py`）根本无从下手**。
    这是写重判时发现的缺口,不是设计时就想到的。
    """

    type: str = "answer"
    text: str | None = None
    escalated: bool = False
    error: str | None = None
    steps: int = 0


@dataclass
class SpanEvent(Event):
    """一段工作。`begin` / `end` 成对 —— **给一步单独计时,不用手写。**

    移植自 `SpanBeginEvent` / `SpanEndEvent`。嵌套用 `parent` 表达。
    """

    type: str = "span"
    name: str = ""
    phase: str = "begin"  # begin | end


EVENT_CLASSES: dict[str, type[Event]] = {
    "model": ModelCallEvent,
    "tool": ToolCallEvent,
    "decision_batch": DecisionBatchEvent,
    "span": SpanEvent,
    "answer": AnswerEvent,
}


def from_json(line: str) -> Event:
    """从盘上读回来。**认不出来的类型不许静默丢** —— 那会让事件流看起来是完整的。"""
    raw = json.loads(line)
    cls = EVENT_CLASSES.get(raw.get("type", ""))
    if cls is None:
        raise ValueError(f"认不出的事件类型 {raw.get('type')!r}；已知 {sorted(EVENT_CLASSES)}")
    if cls is ModelCallEvent:
        raw["usage"] = Usage(**raw.get("usage", {}))
        raw["timing"] = Timing(**raw.get("timing", {}))
    return cls(**raw)


# ═══════════════════════════════════════════════════════════
# 计时器 —— 一次 run 一个,给 `working_start` 定基准
# ═══════════════════════════════════════════════════════════


class Clock:
    """这次 run 的时钟。`working_start` 是**相对它**的秒。

    为什么不用墙钟排序:墙钟会回拨（NTP 校正、虚拟机挂起),
而事件顺序是我们要拿来算时间账的东西。Inspect 的 `working_start`
就是同一件事 —— 它是**单调**的。
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def now_iso(self) -> str:
        return now_iso()
