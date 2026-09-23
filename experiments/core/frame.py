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

    def __str__(self) -> str:
        """★★★ **帧里的人话版** —— 不这样,模型看到的是 dataclass 的 repr。

        实测（2026-09-23,`multiple_68`）:`clip="list"` 走
        `", ".join(str(x) for x in items)`,于是 `already_done` 那一行是::

            already_done: StepRecord(step=0, tool='library.search_books', input='',
                          result='(BFCL scores the call itself rather than...')

        **204 个字符,主体是 Python 语法,而且把 `last_result` 逐字重复了一遍。**
        可帧要回答的是「**任务要求的事做完没有**」—— 交给模型的却是对象 dump。

        ★ 分工:**这里给形状**(调过什么、按什么顺序),**`last_result` 给细节**
          (最近一次到底返回了什么)。所以每条的结果在这里截短 ——
          两份都全给,等于同一段话在帧里出现两遍,白占预算。
        """
        arg = f"({self.input})" if self.input else "()"
        res = " ".join(self.result.split())
        if len(res) > 60:
            res = res[:57] + "..."
        return f"{self.tool}{arg} -> {res}" if res else f"{self.tool}{arg}"


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
    #: ★★★ **还剩哪些动作没做** —— 工具名的列表，由调用方每步重算。
    #:
    #: `needsTool` 问的是「这个任务还有没有没做的动作?」，
    #: 而它的帧原来只有 `already_done`（**做过什么**）——
    #: **问「还有没有」却不给「有哪些」**，模型只能从「做过的」反推，那是猜。
    #:
    #: 实测（2026-09-22,`bfcl-v3-multiple × react-typed`）：调对了唯一合适的工具之后，
    #: `needsTool` 仍判「还要动作」（0.73），于是又调了一个语义邻居 ——
    #: 10 条 `wrong_tool` 全是这个形状。
    remaining: tuple[str, ...] = ()

    #: ★★ **调过的工具各自是干什么的** —— 已经渲染成文本,直接进帧。
    #:
    #: 为什么需要它:`needsTool` 要判「任务要求的事做完没有」,
    #: 而它原来只拿到工具**名字**::
    #:
    #:     already_done: library.search_books({'location': 'New York public library'})
    #:
    #: `library.search_books` 是一个光秃秃的标识符 —— **它不知道这个工具能做什么**,
    #: 只能从名字猜「search_books 够不够回答『找一本历史小说』」。
    #: 而 `pickTool` 那边**有**描述（`criteria` 的值）。
    #:
    #: ★ 这是**通用**的,不是 BFCL 特有的:任何数据集上,判「覆盖了没有」
    #:   都需要知道那个工具**做什么**,而不只是它**叫什么**。
    done_tools: str = ""

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
    #: ★★ 这个字段是**判定的依据**吗?
    #:
    #: `True` = 节点判的就是它,缺了**没有对象可判**（`stepOk` 没有 `last_result`、
    #: `canDeliver` 没有 `draft`）→ 缺席要让请求**发不出去**。
    #:
    #: `False`（默认）= 它只是**上下文**,缺席是**正常状态**:第 0 步本来就没有
    #: `last_result`,`draft` 在生成之前本来就不存在。
    #:
    #: **为什么必须分开**:两类缺席混在一起报,结果是这条不变量**每一步都在响**,
    #: 而那唯一一次真的缺了依据（`canDeliver` 的证据被 clip 到 100 字符,
    #: 闸门**正确地**判出 `unsupported=0.67`）就淹没在噪声里。
    #: 「天天误报」和「没有这条检查」在效果上没有区别 —— 两种都不再有人看。
    required: bool = False


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
    # ★ 上面那些里,**判定真的靠它**的那些（`FrameField.required`）。
    #   渲染时两者一视同仁（缺了就得说）,但**只有这一份**拦请求。
    missing_required: tuple[str, ...]
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

    ★ 第 3 条要分两档（`FrameField.required`）:**判定的依据**缺了,
    这次判定没有对象;只是**上下文**的字段缺了,那本来就可能是正常状态。
    两档都渲染（缺了就得说）,但只有前一档进 `missing_required`。
    """
    lines: list[str] = []
    truncations: dict[str, tuple[int, int]] = {}
    missing: list[str] = []
    missing_required: list[str] = []

    for spec_field in spec.fields:
        raw = getattr(ctx, spec_field.name, None)

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            missing.append(spec_field.name)
            if spec_field.required:
                missing_required.append(spec_field.name)
            continue

        if isinstance(raw, (list, tuple)):
            items = list(raw)
            if len(items) > MAX_LISTED_CANDIDATES:
                # ★ 截断候选**必须明说** —— 模型会以为「就这些」
                truncations[spec_field.name] = (len(items), MAX_LISTED_CANDIDATES)
                items = items[:MAX_LISTED_CANDIDATES]
            # ★ **空的列表是一个事实,不是一处空白。**
            #   `actions_left:` 后面什么都没有,和「这个字段没喂上」在渲染上
            #   长得一模一样 —— 而判定模型分不出,它会当成「这里没东西」。
            #   这正是 §8.15 那条 `Frame.missing` 的形状:报不出「没有」和
            #   「忘了喂」的检查,等于没有检查。
            body = ", ".join(str(x) for x in items) if items else "(none)"
        else:
            body = str(raw)

        body, original = _clip(body, spec_field.budget, spec_field.clip)
        if original > len(body):
            truncations[spec_field.name] = (original, len(body))
        lines.append(f"{spec_field.label}: {body}")

    return Frame(node=spec.node, lines=tuple(lines), truncations=truncations,
                 missing=tuple(missing), missing_required=tuple(missing_required),
                 excluded=spec.excluded)


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
            # 判的就是「这个任务还有没有没做的动作」—— 没有 task 就没有对象可判
            FrameField("task", 400, "task", required=True),
            FrameField("last_tool", 60, "last_tool"),
            FrameField("last_result", 300, "last_result"),
            # ★ 是一份**清单**,不是一个计数 —— `steps_done: 2` 那种写法分不出
            #   「读过了」和「写过了」（实测：写任务里文件从没被写出来）
            FrameField("history", 300, "already_done", clip="list"),
            # ★ 「调过的工具能做什么」—— 判「任务被满足了吗」的依据。
            #   名字 + 历史只说明**调过什么**,说明不了**覆没覆盖**。
            FrameField("done_tools", 400, "done_tools"),
        ),
        excluded=(("draft", "还没生成,这时没有 draft"),),
    ),
    "pickTool": FrameSpec(
        node="pickTool",
        fields=(
            FrameField("task", 400, "task", required=True),
            # ⚠️ **不是** required:第 0 步本来就没有上一步的结果。
            #    标成 required 会让这条不变量每一步都响,然后被无视。
            FrameField("last_result", 300, "last_result"),
            # ★★★ **「还剩哪些动作」** —— 实测（2026-09-23,101 个 pickTool 判定点,
            #   逐条重放同一批请求、只换帧）:
            #
            #   | 帧 | 离 0.6 门限中位 | 贴门限(<0.15) | 答案变了 |
            #   |---|---|---|---|
            #   | `task`+`last_result`（原来）| 0.280 | **12%** | — |
            #   | + `remaining`（这一条）     | 0.350 | **2%**  | **0/101** |
            #   | + `history`                 | 0.330 | 4%    | 1/101 |
            #   | + 两个都加                   | 0.370 | 3%    | 1/101 |
            #
            #   ⇒ **把证据喂厚,判定就离开门限**:贴门限的从 12% 掉到 2%,
            #     而**答案一条没变** —— 变的不是「判什么」,是「判得多稳」。
            #     §8.17 那条「判定贴在门限边上」说的是同一个东西。
            #
            #   ★ 为什么选它而不是 `history`:`history` 的边际也有效（0.280→0.330）,
            #     但它会**翻掉一条答案**（`multiple_18 step1`:`war_details(0.71)`
            #     → `__done__(0.59)`——方向是对的,可是 0.59 低于 0.6,策略会
            #     `escalate` 而不是收工),而「两个都加」翻的那条是**坏的**
            #     （`modify_columns` → `create_backup`）。
            #     **边际大 + 零答案变化**才是能安全落地的那个。
            FrameField("remaining", 200, "actions_left", clip="list"),
        ),
        # 候选在问题那一侧（`choice` 的选项），不占帧的字段 —— 见 §8.4「每步重建」
        excluded=(
            ("draft", "还没到生成那一步"),
            # ⚠️ 这条**是量过之后才留着的**,不是想当然:见上面 `remaining` 那张表 ——
            #    加 `history` 边际也涨,但会翻掉一条答案（方向对、幅度不够）。
            #    留 `remaining` 是**边际最大且零翻转**的那个组合。
            ("history", "候选本身已经排除了做过的动作;再给历史会翻掉答案且边际更小(实测见上)"),
        ),
    ),
    "pickInput": FrameSpec(
        node="pickInput",
        fields=(FrameField("task", 400, "task", required=True),),
        excluded=(
            ("last_result", "判的是**这一调**的目标,不是上一步返回了什么"),
            ("history", "候选由 `candidates` 每步算出来,历史是另外的事"),
        ),
    ),
    "gradeRisk": FrameSpec(
        node="gradeRisk",
        fields=(
            # 判的是**这一调**的风险 —— 没有 target 就没有可判的东西
            FrameField("last_input", 200, "target", required=True),
            FrameField("task", 300, "task"),
        ),
        excluded=(("last_result", "判的是**调用之前**的风险,这时还没有结果"),),
    ),
    "stepOk": FrameSpec(
        node="stepOk",
        fields=(
            FrameField("last_tool", 60, "tool"),
            FrameField("last_input", EVIDENCE_INPUT_CHARS, "input"),
            # ★ 判的就是「这一步成没成」—— 唯一没有它就没法判的字段
            FrameField("last_result", 500, "output", required=True),
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
            FrameField("task", 400, "task", required=True),
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
            FrameField("task", 400, "task", required=True),
            # ★ 900,不是 100 —— 100 那次判定**正确地**判出 `unsupported=0.67`,
            #   是帧喂少了。这一格的预算直接决定闸门准不准。
            #   ★ 而且它是 required:**没有 draft 就没有可交付的东西**。
            #   这次事故的形状正是「有 draft 但被截短了」—— 那走 `truncations`,
            #   不在这里;这里是「压根没有」。
            FrameField("draft", 900, "answer", required=True),
        ),
        excluded=(("history", "判的是**回答**与任务,历史会把回答挤掉"),),
    ),
}



# ═══════════════════════════════════════════════════════════
# ★★ 请求 = 帧 + 问题 —— **预算要一起算,而且超了必须有人接**
#
# 这一段是补第十轮 R2/R5 的。那一轮的两条住在帧编译里：
#
#   R5  `fileOptions` 的**选项数没有上界**,而同一个帧里的字段都老老实实截断了
#       → **帧有界、选项无界**,恰好是 §8.2 说要避免的形态
#   R2  `write_file` **永不从候选中移除**,而同文件 185 行把不变量写成了通则
#       → 不变量写成通则、实现三个工具三种待遇
#
# 两条合起来是一条五步失效链（审计原文）：
#
#   选项超限 → 校验发现 → **无人接收** → 请求照发 → **判定静默掉点**
#
# ★ 我第一版的帧模块**三处全中**：候选被推到「问题那一侧」所以这里不管它（R5）、
#   候选由调用方给所以没人检查它删没删做过的动作（R2）、
#   `truncations` 只是渲染成正文里一句注释,**没有东西失败也没有东西降级**（无人接收）。
#
# 所以这一段做三件事：**把帧和选项合起来算一次预算**、**把不变量变成可检查的**、
# **把违规变成调用方必须处理的东西,而不是一句注释**。
# ═══════════════════════════════════════════════════════════

#: 判定模型一次请求的选项上限。★ 实测 77 个候选时选中概率掉到 **0.425**（§8.2）。
#: 20 是 TS 侧 `LIMITS[*].maxOptions` 的值,这里逐字对齐。
MAX_OPTIONS = 20

#: 一次请求（帧 + 问题 + 选项）的字符上限。判定模型上下文只有 512/1024 token。
#: ★ **这是整个请求的预算,不是帧的预算** —— R5 的病就是只算了帧那一半。
MAX_REQUEST_CHARS = 4000


@dataclass(frozen=True)
class Question:
    """一个类型化问题。

    `kind` 决定 `policy` 怎么读答案:

    - `noul`：P(true)
    - `choice`：选中项的**概率**（卡阈值用它,**不是 `confidence`** —— §8.3）
    - `score`：序数

    ★ `options` 的**数量要受 `MAX_OPTIONS` 管** —— 它和帧的字段抢同一段上下文。
    """

    node: str
    kind: str
    ask: str
    options: tuple[str, ...] = ()
    #: ★ **判据**（`vocab.ts` 的 `criteria`）。
    #:
    #: - `noul` → `{"true": "什么条件下算 true", "false": "什么条件下算 false"}`
    #: - `choice` → 由 `options` 生成（键是选项本身,值是「什么条件下该选它」）
    #:
    #: ★ `vocab.ts` 的原话:「**问题 ID 不会到达模型**」—— 所以措辞和判据
    #: 是模型能看到的全部,写错了没有别的东西兜得住。
    #: 而且 `noul` 的判据「显著提升判定质量」,不给是白丢的。
    criteria: dict[str, str] = field(default_factory=dict)
    #: **答得果不果断**的门限 —— 逐字对应 TS 的 `topGte`。
    #:
    #: ⚠️ `noul` 上判的是 `max(p, 1-p)`,**不是** `p`:一个果断的「否」也算果断。
    #: 要「答是的概率」那个门限是另一件事（TS 的 `probGte`）——
    #: `TypedController.yes` 就是它。**一道题有两个数、两个门限,别混。**
    #:
    #: ★ **每个节点可以不同** —— `DECISION.md` 里 `pick_tool` 是 0.6、
    #: `pick_input` 是 0.5。用一个全局常量会把其中一个改错。
    threshold: float = 0.5

    def option_labels(self) -> tuple[str, ...]:
        return self.options


@dataclass(frozen=True)
class Violation:
    """**一条必须有人接的违规。**

    ★ 它不是注释、不是日志 —— `Request.check()` 返回它,
    而 `Request.fatal` 为真时**调用方不许把请求发出去**
    （要么修帧、要么标 degraded、要么弃答）。

    「校验发现 → 无人接收 → 请求照发」正是 R2/R5 那条失效链的中间三步。
    """

    code: str          # options_over_budget | request_over_budget | candidate_already_done | field_missing
    detail: str
    fatal: bool = True


@dataclass(frozen=True)
class Request:
    """**一次判定请求的完整形态**:帧（状态）+ 问题（问什么、选项是什么）。

    ★ 预算必须在这个粒度上算 —— 分开算就是 R5。
    """

    frame: Frame
    question: Question

    def render(self) -> str:
        """判定的正文:帧 + 问题 + 选项编号。

        ★ 选项**带序号**列出来 —— 判定模型返回的是序号或标签,
        而「第几个」在选项被截断时仍然要指向同一个东西。
        """
        parts = [self.frame.render(), "", f"# Question", self.question.ask]
        if self.question.options:
            parts.append("# Options")
            parts.extend(f"{i}. {o}" for i, o in enumerate(self.question.options, start=1))
        return "\n".join(parts)

    def digest(self) -> str:
        """★★★ **整个请求的指纹** —— 帧 + 问题 + **选项**。

        ⚠️ 这里踩过一次,值得记:**`frame_digest` 只覆盖帧,而帧不等于请求。**

        `pickTool` 的帧只有 `task` + `last_result`（候选**故意不占帧的字段**,
        见 §8.4「每步重建」）—— 于是**换了候选集而帧指纹一动不动**。
        实测（2026-09-23,同一帧指纹 `bb30c546d43b63cd`）:

        ====================== ==================== ========
        候选                    判定                  top
        ====================== ==================== ========
        `[war, leader]`         `war_details`        **1.00**
        `[leader, war]`         `war_details`        1.00
        `[war, leader, battle]` **`battle_details`** 0.99
        `[war, leader, done]`   `war_details`        **0.71**
        ====================== ==================== ========

        ★ 我拿「帧指纹相同」当成了「请求相同」,于是把第 1 行和第 4 行的差别
        （1.00 vs 0.71）读成了**判定后端随机**,还写进了文档。
        **后端是确定的**:同一个请求原样发 12 次,12 次都是 `top=1.0000`。

        ⇒ **比「两次跑的是不是同一个判定」,要比这个,不是比 `frame_digest`。**
          帧的指纹回答的是「它看到了什么」;请求的指纹才回答「它被问了什么」。
        """
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:16]

    def chars(self) -> int:
        return len(self.render())

    def check(self, ctx: AgentCtx | None = None) -> list[Violation]:
        """**发请求之前**跑（§8.2 原话:「发请求之前就要校验」）。

        返回空列表 = 可以发。否则**调用方必须处理**,不许照发。
        """
        out: list[Violation] = []

        # R5:选项数 —— 它和帧抢同一段上下文,所以**在这里一起算**
        if len(self.question.options) > MAX_OPTIONS:
            out.append(Violation(
                code="options_over_budget",
                detail=(f"{self.question.node}: {len(self.question.options)} 个选项 > {MAX_OPTIONS}。"
                        f"实测 77 个候选时选中概率掉到 0.425 —— "
                        f"**必须裁到候选提供方那一侧,不是在这里切**"),
            ))

        # R5 的另一半:整个请求的预算,不只是帧那一半
        total = self.chars()
        if total > MAX_REQUEST_CHARS:
            out.append(Violation(
                code="request_over_budget",
                detail=f"请求 {total} 字符 > {MAX_REQUEST_CHARS}（帧 {len(self.frame.render())} + 选项）",
            ))

        # R2:候选里有没有**已经做过的动作**
        if ctx is not None:
            done = {r.tool for r in ctx.records()}
            repeated = sorted(done & set(self.question.options))
            if repeated:
                out.append(Violation(
                    code="candidate_already_done",
                    detail=(f"{self.question.node}: 候选里有做过的动作 {repeated}。"
                            f"§8.4 要求**每步重建**、把做过的删掉 —— "
                            f"固定候选会让模型去选一个已经不适用的动作（实测：写完文件后 "
                            f"`write_file` 还在候选里,模型会再选它）"),
                    # ⚠️ 不是 fatal:重读一个文件有时是合理的。
                    #    **但必须报出来** —— R2 的病正是「没人看这个不变量」。
                    fatal=False,
                ))

        # 帧里缺字段。**分两档** —— 这是把 §8.2 那条不变量从噪声里救出来的关键:
        #
        #   required 缺 → **判定的依据不在**,这次判定没有对象。发出去只会得到一个
        #                 凭空生成的答案,而日志上它和一次正常判定长得一样（§8.10）。
        #                 → fatal,调用方必须修帧 / 标 degraded / 弃答。
        #   只是上下文缺 → 第 0 步没有 `last_result`、生成之前没有 `draft` ——
        #                 **正常状态**,不进这个列表。
        #
        # ★ **不进列表不等于不说。** 它照旧渲染成 `(absent — last_result)`、
        #   照旧记在 `Frame.missing` 上进日志 —— 信息一个字节没少,
        #   少的是「每一步都喊一次」。
        #   `Violation` 的含义是「必须处理」,把正常状态放进去就等于教会读的人跳过它。
        if self.frame.missing_required:
            out.append(Violation(
                code="field_missing",
                detail=(f"{self.question.node}: 帧里缺**判定依据** "
                        f"{list(self.frame.missing_required)} —— 这一次判定没有对象可判。"
                        f"（另外缺的上下文：{sorted(set(self.frame.missing) - set(self.frame.missing_required))}）"),
                fatal=True,
            ))
        return out

    @property
    def fatal(self) -> list[Violation]:
        """**必须处理的那些。** 调用方拿到非空就该：修帧 / 标 degraded / 弃答 —— 三选一。"""
        return [v for v in self.check() if v.fatal]

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
    "Question", "Request", "Violation", "MAX_OPTIONS", "MAX_REQUEST_CHARS",
    "NODE_FRAMES", "MAX_LISTED_CANDIDATES", "EVIDENCE_INPUT_CHARS",
    "EVIDENCE_WRITE_INPUT_CHARS", "compile_frame", "ctx_from_steps", "frame_for",
    "candidate_provider",
]
