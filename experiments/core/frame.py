"""★ **决策帧 —— 判定模型看得见的那一份有界状态。**

## 为什么这个文件值得单独存在

`AGENTS.md` §8.2:**帧里没有的,它判不出来** —— 不是判错,是压根看不见。
而我们的三次事故**全部出在帧上**,没有一次是判定模型的问题:

| 事故 | 帧错在哪 | 症状 |
|---|---|---|
| `canDeliver` 把工具结果 clip 到 **100** 字符 | 帧喂少了 | 判定**正确地**判出 `unsupported=0.67`;改成 600 立刻通过 |
| `stepOk` 的帧里**带着 `task`** | 帧喂多了 | 被判到任务级,`ok=0.470` 判否 → `stop` → **任何多步任务都跑不完** |
| `needsTool` 给的是 `steps_done: 2`（**计数**）| 形状不对 | 分不出「读过了」和「写过了」→ 写任务里文件从没被写出来 |

**三次都改对了判定,但看起来像判定错了。** 所以帧不能是「各调用点手拼的一个 dict」——
那样上面三处**没有任何办法被复查**。

## 所以:帧是**声明**出来的,不是拼出来的

```python
FrameSpec(
    node="step_ok",
    fields=(FrameField("last_tool", 60, "tool"),
            FrameField("last_input", 200, "input"),
            FrameField("last_result", 500, "output")),
    # ★ 显式声明「本节点故意不看什么」,以及为什么
    excluded=(("task", "判的是这一步,不是这个任务 —— 带上它会被拉到任务级（实测）"),),
)
```

**`excluded` 是这份设计里最值钱的一栏。** 三次事故里有两次是「不该看的看了」,
而删掉一个字段之后**没有任何东西记得它曾经在过** —— 声明出来,下一个人才知道为什么不能加回去。

## 和 TS 那份的关系（★ 必须对齐,`AGENTS.md` §8.7.2）

`JevLoop/src/frame.ts` 是**权威**。Python 这份是它的**推广**,差别只有一处:

| TS 的 `AgentCtx` | 为什么它不通用 | 这里 |
|---|---|---|
| `files` / `readFiles` | 那是**文件工具**特有的 | 改成「候选由每个工具的 `candidates(ctx)` 算」—— 换成检索 / API / 环境动作都成立 |
| `canWrite` | 同上（`write_file` 有没有内容来源）| 同上,归候选提供方 |

其余字段（`task` / `history` / `earlier` / `lastTool` / `lastResult` / `draft`）**逐字保留**。

★ `AGENTS.md` §8.7.2 重写 `verify:port` 时,**这里就是行为对照的主要对象**:
桩掉判定模型,两份实现跑同一批输入,**每一步的帧必须逐字相同**。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from experiments.core.types import Step, Tool

# ═══════════════════════════════════════════════════════════
# 词汇
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StepRecord:
    """历史里的一步。**和 TS 的 `StepRecord` 逐字对齐**（`step` / `tool` / `input` / `result`）。"""

    step: int
    tool: str
    input: str
    result: str


@dataclass
class AgentCtx:
    """agent 的**可观测状态** —— 帧的原料池。

    ★ 这里**没有 `gold`**、没有 reward、没有数据集内部字段。
    和 `DecisionView` 只带 `task_id` 是同一条理由:**接口上拿不到,比约定不许强**。
    """

    task: str = ""
    history: tuple[StepRecord, ...] = ()
    # 之前几轮**问过什么**,压成一句话。判定要的是**指代关系**
    # （「再读一遍那个文件」里的「那个」）,不是上一轮的完整经过。
    earlier: str = ""
    last_tool: str = ""
    last_input: str = ""
    last_result: str = ""
    # 生成出来的回答 —— `canDeliver` 用它
    draft: str = ""
    # 每个工具的候选参数（`None` = 这个工具不用挑参数）。
    # ★ **每步重建** —— 固定的候选会让模型去选一个已经不适用的动作（§8.4）。
    candidates: dict[str, list[str]] = field(default_factory=dict)
    # 候选被窗口截断时**必须明说**（TS 的 `MAX_FILE_OPTIONS = 20` 那条）
    candidate_notes: dict[str, str] = field(default_factory=dict)

    def records(self) -> list[StepRecord]:
        return list(self.history)


def ctx_from_steps(task: str, steps: Sequence[Step], *, draft: str = "",
                   earlier: str = "") -> AgentCtx:
    """从轨迹拼出一个 `AgentCtx`。

    ★ **`result` 用观察原文,不预截** —— 截多少是**帧的预算**说了算
    （`FrameField.budget`）,不是这里说了算。在两处截会得到两套口径,
    而 `canDeliver` 那次事故就是「截了但没人知道截了多少」。
    """
    history = tuple(
        StepRecord(step=s.index, tool=s.action.name or s.action.kind,
                   input=str(s.action.arguments or s.action.content), result=s.observation)
        for s in steps
    )
    last = history[-1] if history else None
    return AgentCtx(
        task=task, history=history, earlier=earlier, draft=draft,
        last_tool=last.tool if last else "",
        last_input=last.input if last else "",
        last_result=last.result if last else "",
    )


@dataclass(frozen=True)
class FrameField:
    """帧里的一个字段。

    `budget` 是**字符**上限 —— 判定模型的上下文只有 512/1024 token。
    `clip` 决定超预算怎么截,而**截了多少必须报出来**（见 `Frame.truncations`）。
    """

    name: str          # `AgentCtx` 上的字段名
    budget: int        # 字符上限
    label: str         # 进给判定模型时那行字
    # tail: 留尾巴（证据通常是**后面**才是结论）；head: 留开头；list: 只留前 N 条
    clip: str = "tail"


@dataclass(frozen=True)
class FrameSpec:
    """**一个节点的帧声明。** 这是 §8.2 从「注释」变成「可复查的产物」的地方。"""

    node: str
    fields: tuple[FrameField, ...]
    # ★ 「本节点故意不看什么」+ 为什么。**删掉一个字段之后没有东西记得它来过**,
    #   而三次事故里有两次就是「不该看的看了」。
    excluded: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Frame:
    """编好的帧。**它自己记得被截断和被省略的东西。**"""

    node: str
    lines: tuple[str, ...]
    # 字段名 → 原文长度 vs 实际放入的长度
    truncations: dict[str, tuple[int, int]]
    # 声明了但**这次没拿到**的字段（`AgentCtx` 上是空的）
    missing: tuple[str, ...]
    # ★ 声明里明说「不看」的字段 —— 一起进日志,这样「这一格当时看到什么」可核对
    excluded: tuple[tuple[str, str], ...]

    def render(self) -> str:
        """进给判定模型的正文。

        ★ **被截断的字段要在正文里留下痕迹。** 判定模型看不到「这里少了 900 字」,
        它会当成「证据就这么多」—— 而 `canDeliver` 那次事故正是这个形状。
        """
        out: list[str] = []
        for line in self.lines:
            out.append(line)
        if self.truncations:
            parts = [f"{k}: {was}→{now}" for k, (was, now) in sorted(self.truncations.items())]
            out.append(f"(truncated — {'; '.join(parts)})")
        if self.missing:
            out.append(f"(absent — {', '.join(self.missing)})")
        return "\n".join(out)

    def digest(self) -> str:
        """帧的指纹,**进日志**。

        两次运行如果帧不一样而没人发现,那「同一个方法」这句话就不成立。
        """
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════
# 编译 —— 唯一的入口
# ═══════════════════════════════════════════════════════════

# 候选列表超过这个数就不再列全 —— **并且必须明说**（TS 的 `MAX_FILE_OPTIONS`）。
MAX_LISTED_CANDIDATES = 20


def _clip(text: str, budget: int, mode: str) -> tuple[str, int]:
    """返回 (裁过的文本, 原长)。**不在这里决定要不要报** —— 报是 `compile_frame` 的事。"""
    original = len(text)
    if original <= budget:
        return text, original
    if mode == "head":
        return text[:budget], original
    return text[-budget:], original


def compile_frame(spec: FrameSpec, ctx: AgentCtx) -> Frame:
    """把状态投影成**有界**的帧。

    ★ 三条不变量,每一条都对应一次事故:

    1. **有界** —— 每个字段按 `budget` 截,截了就报（`canDeliver` 那条）
    2. **只给声明的字段** —— 没在 `fields` 里的**一个都不进**（`stepOk` 那条）
    3. **缺的要说** —— 声明了但 `ctx` 上是空的,记进 `missing`,**不静默留白**
       （留白会让判定模型以为「证据就这么多」）
    """
    lines: list[str] = []
    truncations: dict[str, tuple[int, int]] = {}
    missing: list[str] = []

    for spec_field in spec.fields:
        raw = getattr(ctx, spec_field.name, None)

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            missing.append(spec_field.name)
            continue

        if isinstance(raw, (list, tuple)):
            items = list(raw)
            if len(items) > MAX_LISTED_CANDIDATES:
                # ★ 截断候选**必须明说** —— 模型会以为「就这些」
                truncations[spec_field.name] = (len(items), MAX_LISTED_CANDIDATES)
                items = items[:MAX_LISTED_CANDIDATES]
            body = ", ".join(str(x) for x in items)
        else:
            body = str(raw)

        body, original = _clip(body, spec_field.budget, spec_field.clip)
        if original > len(body):
            truncations[spec_field.name] = (original, len(body))
        lines.append(f"{spec_field.label}: {body}")

    return Frame(node=spec.node, lines=tuple(lines), truncations=truncations,
                 missing=tuple(missing), excluded=spec.excluded)


# ═══════════════════════════════════════════════════════════
# 七个节点的帧声明
#
# ★ 预算逐字抄自 TS 侧（`docs/PLAN-*.md` 记的那张表）。
#   **改任何一个数都要两边一起改,并说明为什么** —— 否则 §8.7.2 的行为对照会红。
# ═══════════════════════════════════════════════════════════

#: 证据在帧里最多占多少字符。★ 100 那次事故之后定的。
EVIDENCE_INPUT_CHARS = 120
EVIDENCE_WRITE_INPUT_CHARS = 600

NODE_FRAMES: dict[str, FrameSpec] = {
    "needsTool": FrameSpec(
        node="needsTool",
        fields=(
            FrameField("task", 400, "task"),
            FrameField("last_tool", 60, "last_tool"),
            FrameField("last_result", 300, "last_result"),
            # ★ 是一份**清单**,不是一个计数 —— `steps_done: 2` 那种写法分不出
            #   「读过了」和「写过了」（实测：写任务里文件从没被写出来）
            FrameField("history", 300, "already_done", clip="list"),
        ),
        excluded=(("draft", "还没生成,这时没有 draft"),),
    ),
    "pickTool": FrameSpec(
        node="pickTool",
        fields=(
            FrameField("task", 400, "task"),
            FrameField("last_result", 300, "last_result"),
        ),
        # 候选在问题那一侧（`choice` 的选项），不占帧的字段 —— 见 §8.4「每步重建」
        excluded=(
            ("draft", "还没到生成那一步"),
            ("history", "候选本身已经排除了做过的动作;再给历史会把候选挤掉"),
        ),
    ),
    "pickInput": FrameSpec(
        node="pickInput",
        fields=(FrameField("task", 400, "task"),),
        excluded=(
            ("last_result", "判的是**这一调**的目标,不是上一步返回了什么"),
            ("history", "候选由 `candidates` 每步算出来,历史是另外的事"),
        ),
    ),
    "gradeRisk": FrameSpec(
        node="gradeRisk",
        fields=(
            FrameField("last_input", 200, "target"),
            FrameField("task", 300, "task"),
        ),
        excluded=(("last_result", "判的是**调用之前**的风险,这时还没有结果"),),
    ),
    "stepOk": FrameSpec(
        node="stepOk",
        fields=(
            FrameField("last_tool", 60, "tool"),
            FrameField("last_input", EVIDENCE_INPUT_CHARS, "input"),
            FrameField("last_result", 500, "output"),
        ),
        # ★★★ **故意没有 `task`** —— 这是修出来的:
        #   以前帧带着 `task`、问题写着 "for the task"、判据写着 "what the task needed",
        #   三处一起把它拉到了任务级。实测第一步 `list_dir` 确实成功了,
        #   但没回答任务,于是 `ok=0.470` 判否 → `stop` → **整个循环结束**,
        #   任何需要多于一个工具的任务都跑不完。
        excluded=(
            ("task", "判的是**这一步**,不是这个任务 —— 带上它会被拉到任务级（实测）"),
            ("history", "同上,历史会把它拉到任务级"),
            ("draft", "还没生成"),
        ),
    ),
    "isDone": FrameSpec(
        node="isDone",
        fields=(
            FrameField("task", 400, "task"),
            # ★ 是清单不是计数 —— 和 `needsTool` 同一条理由
            FrameField("history", 300, "already_done", clip="list"),
        ),
        excluded=(
            ("last_result", "判的是「任务做完没有」,不是「上一步返回了什么」"),
            # ★ 旧措辞里有半句「any further tool call would not add information」——
            #   它是个陷阱:**读任何一个还没读过的文件都会「增加信息」**,
            #   哪怕那个文件与任务无关。实测因此去读了任务不需要的 notes.md 还不肯停。
            ("_note", "判据是「任务要求的都做了」,不是「再多调一次会不会增加信息」"),
        ),
    ),
    "canDeliver": FrameSpec(
        node="canDeliver",
        fields=(
            FrameField("task", 400, "task"),
            # ★ 900,不是 100 —— 100 那次判定**正确地**判出 `unsupported=0.67`,
            #   是帧喂少了。这一格的预算直接决定闸门准不准。
            FrameField("draft", 900, "answer"),
        ),
        excluded=(("history", "判的是**回答**与任务,历史会把回答挤掉"),),
    ),
}


def frame_for(node: str) -> FrameSpec:
    """取一个节点的帧声明。**没有这个节点就炸** —— 打错名字不该静默给个空帧。"""
    if node not in NODE_FRAMES:
        raise KeyError(f"没有 {node!r} 的帧声明；有 {sorted(NODE_FRAMES)}")
    return NODE_FRAMES[node]


def candidate_provider(tool: Tool) -> Callable[[AgentCtx], list[str]] | None:
    """工具的候选从哪来。**没有候选来源的工具不能被 `pickInput` 选中。**

    ★ 这是 `AGENTS.md` §8.2 那条硬约束在帧这一层的落点:
    判定模型只能从枚举里挑,所以**一个参数没有候选来源的工具,把它的参数做成
    `choice` 是在骗自己** —— 它只能靠猜,而我们量到 77 个候选时选中概率已经掉到 0.425。

    返回 `None` = 这个工具的参数**不可枚举**（例如自由文本 query）。
    那种情况必须走「生成提候选、判定排序」,或者干脆不放进候选集。
    """
    enum = tool.parameters.get("properties", {})
    for key, schema in (enum or {}).items():
        if isinstance(schema, dict) and schema.get("enum"):
            return lambda ctx, k=key: [str(v) for v in schema["enum"]]
    return None


__all__ = [
    "AgentCtx", "StepRecord", "Frame", "FrameField", "FrameSpec",
    "NODE_FRAMES", "MAX_LISTED_CANDIDATES", "EVIDENCE_INPUT_CHARS",
    "EVIDENCE_WRITE_INPUT_CHARS", "compile_frame", "ctx_from_steps", "frame_for",
    "candidate_provider",
]
