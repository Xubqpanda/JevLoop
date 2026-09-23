"""我们把 JevLoop 这条臂。**一个 `decide()` = 一次节点序列。**

## 这个文件是论文那张表的第一个格子

`docs/PLAN-*.md` 的 Table 2 按两根轴切臂:**（一）谁来决定**、**（二）历史怎么到达下一步**。
这个文件动的是第一根轴 —— 循环还是 ReAct 那个循环（`common.run_loop`）,
**换掉的只有控制器**。

所以 `react × typed` 和 `react × llm` 的差**只可能来自决策者**。这不是声明,是构造:
两个臂共用同一个 `run_loop`、同一个 `build_prompt`、同一批工具。

## 逐步是什么

```
decide()
 ├─ needsTool  noul    「这个任务还有没有没做的动作?」
 │    否 ─────────────────────────────► 生成答案（LLM）→ Decision(answer)
 │    是
 ├─ pickTool   choice  在**这一步重建的**候选里挑一个
 └─ pickInput  choice  该工具的参数候选（枚举得出时）
       └──────────────────────────────► Decision(tool)
```

★ **七个节点里这条臂只驱动三个**（`needsTool` / `pickTool` / `pickInput`）。
另外四个（`gradeRisk` / `stepOk` / `isDone` / `canDeliver`）属于**完整 JevLoop**
那条臂 —— 它有自己的循环和授权闸门。这里不假装驱动了它们:
拿一个用不上的节点去说「我们也解耦了」,是**用帧的数量冒充解耦的深度**。

## 三分法的落点（§8.1）

| 这一步是 | 交给谁 | 在这里 |
|---|---|---|
| 生成文本 | LLM | `_generate()` —— 只有这一处 |
| 挑选 / 打分 / 是否 | Jev | `_ask()` —— 一次请求判一批 |
| 遵循精确规则 | 代码 | 候选重建、预算校验、`Request.check()` |

## ★ 两个从事故里长出来的约束,在这里**是构造性的**

**候选每步重建**（§8.4）:`_candidates()` 每次都从 `view.tools` 减去 `ctx.records()`
里做过的动作。固定候选会让模型去选一个已经不适用的动作 —— 实测写完文件后
`write_file` 还在候选里,模型会**再选它**。

**违规必须有人接**（第 10 轮 R2/R5）:`_ask()` 里的 `Request.check()` 不是装饰,
`fatal` 会让这一步**发不出去**,并退回一个 `unparsed` + 原因。
那条失效链是「选项超限 → 校验发现 → **无人接收** → 请求照发 → 判定静默掉点」——
断在第三步还是第四步,区别就是这里有没有人接。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from experiments.core.agent import Session
from experiments.core.controller import Decision, DecisionView
from experiments.core.deciding import Answer, DecideRequest, DecisionClient
from experiments.core.frame import (
    AgentCtx,
    Question,
    Request,
    candidate_provider,
    compile_frame,
    ctx_from_steps,
    frame_for,
)
from experiments.core.models import Message

# ★ `DEFAULT_YES` / `DEFAULT_TOP` 两个全局常量**已删** —— 见下面的
#   `NODE_THRESHOLDS`:`DECISION.md` 里 `pick_tool` 是 0.6 而 `pick_input` 是 0.5,
#   用一个全局值会把其中一个改错,而**两边都还是「看起来在卡门限」**。

# ★★★ `ANSWER_FORMAT` **已删** —— 它违了我们自己定的分工。
#
#   原来 `_answer_text` 拼的是:
#
#       Reply with the final answer only. Do not emit an action.   ← 我们加的
#       # Task
#       <题目>
#       Give the final answer as a single number on its own line.  ← benchmark 的契约
#
#   ⇒ **两条输出契约并存**,而 `core/bench.py` 的分工表写着:
#     「任务陈述 + **输出契约** → **benchmark**;交互协议 → baseline。
#      **baseline 不许改任务陈述,只能加自己的协议块。**」
#     我们加的那条**不是交互协议,是第二条输出契约**。
#
#   ★ 实测代价(GSM8K × 100,单 commit)`react-typed` **63%**、平均 **9.8** 个输出
#     token;而同一个模型的 `direct` 是 **97%**、**131.7** 个 token。
#     **第一条契约把推理一起禁掉了** —— 模型老老实实只吐一个数。
#
#   ★ 这正是已经记过的那条教训:「两条输出契约并存时,从 prompt 上看不出
#     模型听了哪条」。教训记过了,我又犯了一次 —— 所以这次留注释在代码里。
#
#   ⇒ 生成那一步**就是生成**,用 benchmark 给的 `task_prompt`,一个字都不加。


# ═══════════════════════════════════════════════════════════
# ★★★ 问题和门限 —— **逐字抄 `DECISION.md`,不许自己写**
#
# 为什么这么重:第一版我按自己的理解给这三个节点写了措辞,结果
# **`needsTool` 那一句正好是 `DECISION.md` 里记着的那次事故的写法**。
#
# `DECISION.md` 的原文（`## needs_tool`）:
#
#   ★ 判据是「任务还有没有没做的动作」,不是「还有没有没拿到的信息」。
#   这两个在只读任务上恰好一致,而在**写任务**上分道扬镳 ——
#   实测:任务「把 alpha.ts 里的 totalOf 抄到一个新文件 summary.ts 里」,
#   读完 alpha.ts 之后这个节点判了 `answer`(0.36),于是 loop 直接去生成回答,
#   **文件从没被写出来**。
#
# 我写的是 `"Does this task still require a tool call before it can be answered?"`
# —— 问的是**工具调用**,而事故的教训正是「要问**任务要求的动作**」。
# 「写一个文件」在这个问法下又一次落在问题之外。
#
# ★ 而且 `vocab.ts` 说「**问题 ID 不会到达模型**」—— 所以措辞是模型能看到的全部,
#   写错了没有任何别的东西兜得住。**这是照抄比回忆准的第三个例子**（前两个:
#   `top()` 的语义、`.env` 的变量名）。
#
# 门限也一起抄。★ `pickTool` 是 **0.6**、`pickInput` 是 **0.5** —— 两个不一样,
# 而我第一版只有一个全局 `top`。
# ═══════════════════════════════════════════════════════════

#: `needs_tool`:`prob:needs_tool >= 0.5 → use_tool`（`DECISION.md`）
NEEDS_TOOL_ASK = (
    "The agent still has work to do before it can answer the task — "
    "an action the task requires that has not been taken yet"
)
#: ★ 判据**必须给**:`vocab.ts` 说 noul 的 `criteria` 「显著提升判定质量」,
#: 而且它正是上面那次事故的修法 —— 把「写」明确写进 true 那一侧。
NEEDS_TOOL_CRITERIA = {
    "true": ("the task still requires an action that has not happened: "
             "reading something, listing something, writing something, running something"),
    "false": ("every action the task asks for has already been taken, "
              "and there is enough information to answer"),
}

PICK_TOOL_ASK = "Which tool should the agent call next?"
PICK_INPUT_ASK = "Which file should this tool call target?"

#: 每个节点的门限,**逐条注明出处**。
#:
#: ⚠️ 两个 `choice` 的门限**不一样**（0.6 vs 0.5）—— 这不是笔误,
#: 是 `DECISION.md` 里就写着两个数。用一个全局常量会把其中一个改错,
#: 而**两边都还是「看起来在卡门限」**。
NODE_THRESHOLDS: dict[str, float] = {
    # prob:needs_tool >= 0.5 → use_tool   （probGte 语义:P(true)）
    "needsTool": 0.5,
    # top >= 0.6 → call / else → escalate （topGte 语义:果断程度）
    "pickTool": 0.6,
    # top >= 0.5 → use  / else → escalate （topGte 语义:果断程度）
    "pickInput": 0.5,
}


def _noul_question(node: str, ask: str, *, threshold: float,
                   criteria: dict[str, str] | None = None) -> Question:
    return Question(node=node, kind="noul", ask=ask, threshold=threshold,
                    criteria=criteria or {})


def _choice_question(node: str, ask: str, options: list[str],
                     *, threshold: float,
                     criteria: dict[str, str] | None = None) -> Question:
    """★★★ **候选必须带说明。**

    实测（2026-09-22,`bfcl-v3-multiple × react-typed`）:候选只传了**工具名**,
    描述是空字符串 —— 于是判定模型只能**光看名字猜**。
    结果 **82/100 是 `wrong_tool`**,而置信度经常是 **1.000**(判得很果断,只是错了)。

    ★ 这正是 §8.2 那个形状:**帧里没有的,它判不出来**。
      而且它和 `canDeliver` 那次一样 —— **看起来像判定错,其实是喂少了**。

    `vocab.ts` 对 `choice` 的 `criteria` 的定义就是「键是选项本身,
    值是**什么条件下该选它**」—— 而 `DECISION.md` 的原文里,
    每个选项后面都跟着一句说明（`read_file — 需要文件内容才能继续…`）。
    我们只传了键,没传值。
    """
    return Question(node=node, kind="choice", ask=ask, options=tuple(options),
                    threshold=threshold, criteria=dict(criteria or {}))


#: ★★★ `pickTool` 候选里的**终止项** —— `DECISION.md` 的 `pick_tool` 原文里就有它:
#:
#:     - done — 已有足够证据回答任务,工具循环可以结束了
#:
#: **实测没实现它的代价**(2026-09-22,`bfcl-v3-multiple × react-typed`):
#: 判定**选对了工具**(82 → 25 条 `wrong_tool` 里,剩下的都是这一类),
#: 但调完之后下一步 `needsTool` 仍说「还要动作」——
#: 而候选里**已经把做过的删掉了**(§8.4),于是**只剩错的工具可选**。
#: 结果 `sorted(called) != sorted(gold)` → 判 `wrong_tool`。
#:
#: ⇒ **把做过的删掉,就必须同时给一个「不做了」的出口。**
#:   否则候选集在第一次调对之后**只剩错的选项** —— 这不是模型选错,是我们没给对的选项。
#:   这是我实现 §8.4 时漏掉的另一半。
DONE = "__done__"


def candidates(session: Session, ctx: AgentCtx) -> list[str]:
    """**这一步**能选的工具。§8.4 的落点。

    ★ 做过的动作**在这里删掉,不是在帧里提示一句** —— 提示是可以被无视的,
    而候选列表是模型唯一能选的东西。

    ★★ **删掉之后必须补一个 `DONE`** —— 见上面那段。少了它,
      第一次调对之后候选里**只剩错的选项**,而模型没有别的可挑。

    ⚠️ 注意这只是**工具级**的去重。同一个工具做两次常常是合理的
    （读两个不同的文件）,所以删的是**已经做过的那一次动作**,
    而 `pickInput` 会在参数那一层再算一次候选。
    """
    done = {r.tool for r in ctx.records()}
    left = [t.name for t in session.tools if t.name not in done]
    # ★ 只要**已经做过什么**,就给出口。一次都没做时不给 ——
    #   那时「不做了」应当由 `needsTool` 回答,而它的帧正是为那个问题准备的。
    return (left + [DONE]) if done else left


@dataclass
class TypedController:
    """判定与生成分开的控制器。

    `client` 是判定后端（`core/deciding.py`）;生成仍然走 `session.call_model`。
    **一次 `decide()` 里的所有判定走同一批** —— `session.next_batch()` 开批,
    按批计时,不按条（同一次请求判多路时按条求和会多算几倍,实测过）。
    """

    client: DecisionClient
    #: 「还要不要动作」的门限（`probGte` 语义）。
    #: ★ 默认值取自 `NODE_THRESHOLDS`,**不是另一个常量** ——
    #:   两处各写一个数,改了其中一个就是「配了没生效」那个坑。
    yes: float = NODE_THRESHOLDS["needsTool"]
    name: str = "typed"
    #: 每一步的判定轨迹（节点 / 答案 / 置信度）。**进日志用,不是给 policy 读的。**
    trace: list[dict] = field(default_factory=list)

    # —— 对外 ────────────────────────────────────────────────

    def decide(self, session: Session, view: DecisionView) -> Decision:
        ctx = ctx_from_steps(view.task_prompt, list(view.history))
        # ★★★ **每步重算「还剩哪些动作」并放进帧。**
        #   `needsTool` 问「还有没有没做的动作」—— 而它的帧必须**装着动作**,
        #   否则模型只能从「做过什么」反推（§8.2:帧里没有的,它判不出来）。
        #   和 `candidates()` 同一个来源,所以两处不会分叉。
        #   ⚠️ **`DONE` 不算动作** —— 它是「没有动作了」的出口,不是一件事。
        #   `pickTool` 需要它当**选项**（选项键),`needsTool` 需要的是
        #   「还剩哪些**事**」,两者差这一个哨兵。混进来会让一个已经做完的
        #   任务在帧里显示成「还剩 `__done__` 可做」。
        ctx.remaining = tuple(a for a in candidates(session, ctx) if a != DONE)
        self.trace = []

        if not view.tools:
            # 没有工具的任务（GSM8K 那种）不该问「要不要用工具」——
            # 那个问题的答案恒为否,问它只是白花一次判定。
            return self._generate(session, view, ctx, why="这个任务没有工具")

        # ① 还要不要动作
        batch = session.next_batch()
        need = self._ask(session, batch, ctx, view.step, "needsTool",
                         _noul_question("needsTool", NEEDS_TOOL_ASK,
                                        criteria=NEEDS_TOOL_CRITERIA,
                                        threshold=NODE_THRESHOLDS["needsTool"]))
        if need is None:
            return self._blocked(session, view, "needsTool")

        if need.prob_true() < self.yes:  # noqa: SIM201 —— 刻意留成可覆盖的
            # ★ 用 `prob_true()`（「是」的概率）,不是 `top()`（果断程度）——
            #   这里问的是「还要不要动作」,答案本身是「是/否」。
            #   一个**果断的「否」**（noul=0.05）在 `top()` 上是 0.95,
            #   拿它来比就会得出「很确定还要工具」—— 正好反了。
            return self._generate(session, view, ctx, why=f"needsTool={need.noul:.3f}",
                                  batch=batch)

        # ② 挑哪个工具 —— **候选在这一步重建**
        options = candidates(session, ctx)
        if not options:
            # 候选空了:做过的都做过了,那就是可以答了。
            # ★ 但 `needsTool` 刚说「还要一个动作」—— 这两句是**矛盾的**,
            #   矛盾本身要记下来,否则它看起来像一次顺利的收尾。
            self.trace.append({"step": view.step, "node": "pickTool",
                               "violation": "no_candidate_left", "fatal": False,
                               "detail": (f"needsTool={need.noul:.3f} 说还要动作,"
                                          f"但候选已空（做过的都做过了）")})
            # ★ 收尾不是「模型判的」,是**代码按精确规则判的**（§8.1 第三行）——
            #   所以要说明来源,不能让日志看起来像一次判定。
            return self._generate(session, view, ctx, why="候选已空（做过的都做过了）",
                                  batch=batch)

        # ★★★ **`DONE` 必须在「唯一候选」捷径之前处理。**
        #
        #   实测（2026-09-22）:加 `DONE` 出口之后 `bfcl-v3-simple × react-typed`
        #   **从 97/100 掉到 0/100** —— 因为那个子集只有 **1 个工具**,
        #   工具做完之后 `candidates()` 返回 `[DONE]`,而捷径把 `DONE`
        #   当成了**工具名**去 `view.tools` 里找。
        #
        #   ★ 这是**加出口时引入的回归**,而且只打在「工具数 = 1」的子集上 ——
        #     另一个子集（2–4 个工具）走的是 `else` 分支,完全没受影响。
        #     **同一处改动在两个子集上一好一坏,是「只在有工具的数据集上测」
        #     这条纪律的又一个例子 —— 但还得再加一条:两个子集都要测。**
        if options == [DONE]:
            self.trace.append({"step": view.step, "node": "pickTool",
                               "answer": DONE, "top": 1.0, "provider": "typed"})
            return self._generate(session, view, ctx,
                                  why="候选只剩 done（工具都做过了）", batch=batch)

        if len(options) == 1:
            # ★ 只有一个候选就**不问** —— 「要不要用工具」刚由 `needsTool` 判过,
            #   再问「要哪一个（而只有一个）」是白花一次判定。
            #   这正是 §8.1 第三行:能由精确规则定的,不交给判定模型。
            picked_choice = options[0]
            tool = next(t for t in view.tools if t.name == picked_choice)
        else:
            # ★★ 候选**带描述** —— 光给名字等于让判定模型猜（见 `_choice_question`）。
            #   `Task.tools` 里就有 description,而它此前**根本没进过问题**。
            by_name = {t.name: t for t in view.tools}
            picked = self._ask(session, batch, ctx, view.step, "pickTool",
                               _choice_question(
                                   "pickTool", PICK_TOOL_ASK, options,
                                   threshold=NODE_THRESHOLDS["pickTool"],
                                   criteria={
                                       o: (by_name[o].description if o in by_name
                                           else "there is enough evidence to answer; stop calling tools")
                                       for o in options}))
            if picked is None:
                return self._blocked(session, view, "pickTool")
            picked_choice = picked.choice
            if picked_choice == DONE:
                # ★ 判定模型说「够了」—— 去生成答案,不再调工具。
                #   这条出口是 `DECISION.md` 里就有的（`done`）,不是我加的。
                self.trace.append({"step": view.step, "node": "pickTool",
                                   "answer": DONE, "top": picked.top(),
                                   "provider": "typed"})
                return self._generate(session, view, ctx,
                                      why="pickTool 判了 done（证据够了）", batch=batch)
            tool = next((t for t in view.tools if t.name == picked_choice), None)
            if tool is None:
                # 判定模型选了一个**不在候选里**的工具 —— 那是它的错,
                # 但必须能被看见,不能悄悄退回一个默认值。
                return self._blocked(session, view,
                                     f"pickTool 选了候选外的 {picked_choice!r}",
                                     answer=picked)

        # ③ 参数
        arguments, err = self._arguments(session, batch, ctx, view, tool)
        if err is not None:
            return self._blocked(session, view, err)

        return Decision(kind="tool", tool=tool.name, arguments=arguments,
                        syntax="typed", raw=f"picked={picked_choice}")

    # —— 节点 ────────────────────────────────────────────────

    def _arguments(self, session: Session, batch: int, ctx: AgentCtx,
                   view: DecisionView, tool) -> tuple[dict, str | None]:
        """参数从哪来。**能枚举的就判,不能枚举的只能生成** —— 而且要说明是哪一种。

        ★ 这是 §8.2 那条硬约束的落点:判定模型只能从枚举里挑
        （实测 77 个候选时选中概率掉到 0.425）。所以一个参数**自由文本**的工具
        （`search[query]` 那种）**不能被 `pickInput` 选中** ——
        把它做成 `choice` 是在骗自己,它只能靠猜。

        那种情况走「生成提候选、判定排序」里的**前半段**:让 LLM 生成。
        ★ 于是**哪一步是判的、哪一步是生成的**必须记清楚 ——
        那正是论文第一根轴要量的东西,含糊过去这个臂就没有意义了。
        """
        provider = candidate_provider(tool)
        params = list((tool.parameters.get("properties") or {}).keys())

        if not params:
            return {}, None

        first = params[0]
        if provider is None:
            # 不可枚举 → 生成。**记下来 —— 这一步不是判定。**
            session.record_decision(step=view.step, node=f"pickInput:{first}",
                                    answer="(generated)", confidence=0.0,
                                    correct=True, latency_ms=0.0, batch=batch)
            return {first: self._generate_argument(session, view, tool, first)}, None

        options = [o for o in provider(ctx)]
        if not options:
            return {}, None
        if len(options) == 1:
            # 只有一个候选 —— 问它是在浪费一次判定,而且答案必然是它。
            return {first: options[0]}, None

        picked = self._ask(session, batch, ctx, view.step, "pickInput",
                           _choice_question("pickInput", PICK_INPUT_ASK, options,
                                            threshold=NODE_THRESHOLDS["pickInput"]))
        if picked is None:
            return {}, "pickInput"
        if picked.choice not in options:
            return {}, f"pickInput 选了候选外的 {picked.choice!r}"
        return {first: picked.choice}, None

    def _generate(self, session: Session, view: DecisionView, ctx: AgentCtx,
                  *, why: str, batch: int = 0) -> Decision:
        """生成答案。**这条臂唯一一次让 LLM 生成的地方**（另一个是自由文本参数）。

        ★ 用的是 `view.task_prompt` + 自己的格式要求,**不是** `view.prompt` ——
        后者是 ReAct 的提示词,它要求模型吐一个 `Action: <tool>[<arg>]` 行。
        拿它去生成答案,模型会照着格式吐一个动作,而我们把它当答案收下。

        ★★ **这里不写 `record_decision`。** 第一版写了,记成
        `node="needsTool", confidence=1.0, correct=True` —— 那是**一条假判定**:
        同一个节点、同一步,前面刚记过一条 `0.1 / False` 的真判定,
        读日志的人会看到两个相反的结论。

        更糟的是 `correct` 那一列**正是 RQ2 要量的东西**
        （「置信度说 0.8 的那批判定,到底对了多少」）——
        往里塞一个恒真的值,那条曲线就永远是 45°。

        「这一步走了生成」这件事由轨迹里的 `Action(kind="answer")` 表达,
        不属于判定账。**§8.1:代码按精确规则做的分支不是判定。**
        """
        del batch  # 只为了和 `_ask` 的调用形状一致;这里不记账
        return Decision(kind="answer",
                        answer=self._answer_text(session, view),
                        thought=f"生成答案（{why}）", syntax="generated")

    def _answer_text(self, session: Session, view: DecisionView) -> str:
        """生成最终答案。**一个字都不加** —— 见上面 `ANSWER_FORMAT` 那段。

        ★ `view.task_prompt` **已经包含 benchmark 的输出契约**
          （GSM8K 的 `ANSWER_CONTRACT` 就拼在题目后面）。我们再补一条
          就是第二条契约,而两条并存时模型听哪条从 prompt 上看不出来。

        ★ 「这做过什么」那一行留着:它是**上下文**,不是契约 ——
          多步任务里模型需要知道已经查过什么。
        """
        parts = [view.task_prompt]
        done = "; ".join(f"{s.action.name or s.action.kind}" for s in view.history)
        if done:
            parts += ["", "# What was done", done]
        return session.call_model([Message(role="user", content="\n".join(parts))]).text

    def _generate_argument(self, session: Session, view: DecisionView, tool, name: str) -> str:
        prompt = "\n".join([
            f"Give only the value for `{name}` — nothing else, no quotes, no label.",
            "",
            "# Task",
            view.task_prompt,
            f"# Tool\n{tool.name}: {tool.description}",
        ])
        return session.call_model([Message(role="user", content=prompt)]).text.strip()

    # —— 判定本身 ────────────────────────────────────────────

    def _ask(self, session: Session, batch: int, ctx: AgentCtx, step: int,
             node: str, question: Question) -> Answer | None:
        """发一次判定请求。**返回 `None` = 这一步发不出去。**

        ★★ 这里是「有人接收」那一步（第 10 轮 R2/R5）。

        那条失效链是:**选项超限 → 校验发现 → 无人接收 → 请求照发 → 判定静默掉点**。
        断在第三步和断在第四步,对这个函数来说就是 `return None` 和 `continue` 的区别。

        两种违规分开处理:
        - **fatal**（缺判定依据 / 超预算）→ **不发**。发出去只会得到一个
          凭空生成的答案,而日志上它和一次正常判定长得一样（§8.10）。
        - **非 fatal**（候选里有做过的动作）→ 发,但记下来 ——
          §8.4 那条不变量要有人**看见**,不是有人拦住。
        """
        req = Request(frame=compile_frame(frame_for(node), ctx), question=question)
        t0 = time.perf_counter()

        for v in req.check(ctx):
            self.trace.append({"step": step, "node": node, "violation": v.code,
                               "fatal": v.fatal, "detail": v.detail})
            # ★★ **违规必须落到事件流里,否则不算有人接收。**
            #
            #   第 10 轮那条失效链是「校验发现 → **无人接收** → 请求照发 → 静默掉点」。
            #   写进内存里一个 list 仍然不是接收 —— 跑完之后没人读得到它,
            #   于是下一个人重建现场时,看到的是「这里怎么少了一次判定」。
            #   所以按「一次没成立的判定」记账:`correct=False` + 说明原因。
            session.record_decision(
                step=step, node=node, answer=f"({v.code})", confidence=0.0,
                correct=False, latency_ms=0.0, batch=batch,
                frame_digest=req.frame.digest(),
                request_digest=req.digest(), request_text=req.render(),
                note=v.detail,
            )
            if not v.fatal:
                continue
            # ★ 不发。为什么发不出去已经记在账上了 —— 否则调用方只知道「没判定」。
            return None

        t_req = time.perf_counter()
        resp = self.client.decide(DecideRequest(
            state={"frame": req.frame.render()}, questions={node: _wire(question)},
        ))
        # ★ 把**这一次请求**的墙钟报给 session —— 批次耗时按请求累加。
        #   不报的话 `decision_ms` 会退化成「两次开批之间」,把那之间的
        #   生成调用和工具执行一起算进来（实测把框架时间减成了负数）。
        session.note_decision_request((time.perf_counter() - t_req) * 1000)
        answer = resp.answers.get(node)
        if answer is None:
            # 后端没给 / 给了畸形的 —— §8.10:报出来,不假装
            detail = f"dropped={resp.dropped} missing={resp.missing} {resp.warnings}"
            self.trace.append({"step": step, "node": node, "violation": "answer_unusable",
                               "fatal": True, "detail": detail})
            session.record_decision(
                step=step, node=node, answer="(answer_unusable)", confidence=0.0,
                correct=False, latency_ms=(time.perf_counter() - t0) * 1000, batch=batch,
                frame_digest=req.frame.digest(), request_digest=req.digest(),
                request_text=req.render(), note=detail,
            )
            return None

        self.trace.append({
            "step": step, "node": node, "answer": answer.choice or answer.noul,
            "top": answer.top(), "provider": resp.provider,
            "frame_digest": req.frame.digest(),
            "truncations": dict(req.frame.truncations),
            "missing": list(req.frame.missing),
        })

        # ★★ **果断程度不够就不许往下走。**
        #
        #   这是一个**被迫的选择**:选项是穷尽的,判定模型必须挑一个 ——
        #   所以「挑了一个」不等于「挑得对」。`top()` 判的正是这件事
        #   （`choice` 看被选中那项的概率;`noul` 看 `max(p,1-p)`）。
        #
        #   §8.6 要求 Mock 必须让 policy **自己走到 `escalate`** ——
        #   而 Mock 给的是平均分布（8 个选项时每项 0.125）。
        #   没有这道门,`_arguments` 会照拿第一个候选,
        #   于是「不确定就别猜」在代码里**从来没有被走到过**。
        decided = answer.top() >= question.threshold
        session.record_decision(
            step=step, node=node, answer=str(answer.choice or f"{answer.noul:.3f}"),
            confidence=answer.top(),
            # ★ 帧的指纹进日志（§8.14）—— 两次运行帧不一样而没人发现,
            #   「同一个方法」这句话就不成立。
            frame_digest=req.frame.digest(),
            # ★★★ **请求的指纹也进**（§8.17）—— 判「两次跑的是不是同一个判定」
            #   要比的是这个,不是 `frame_digest`:那个只覆盖帧,
            #   而 `choice` 的**选项不在帧里**,换了候选帧指纹一动不动。
            request_digest=req.digest(),
            # ★ `correct` 是「这次判定**成立**吗」,不是「护身符」。
            #   全填 True 的话,`correct` 那一列恒为真,
            #   于是「置信度和正确性同现」这句话**在账面上永远成立** —— 而它本该被检验。
            correct=decided,
            # ★★ **一条判定只记一行。**
            #
            #   第一版这里在「不果断」时**又记了一条** —— 于是同一次判定在账上出现两遍,
            #   而**判定次数正是论文的头条指标之一**。这和 §8.10「按批计时、按记录求和
            #   会多算几倍」是同一类错:账目上的一个数被数了两遍,而账面上看不出来。
            #
            #   所以原因写在**同一行的 `note` 里**,不是另起一行。
            note="" if decided else f"top={answer.top():.3f} < {question.threshold}",
            latency_ms=(time.perf_counter() - t0) * 1000, batch=batch,
            # ★★ **正文落在这里,不是 `trace` 里** —— `trace` 是内存里的调试列表,
            #   而落盘在 `record_decision`。第一次接的时候接到了 `trace` 上,
            #   于是 `--dump-requests` 静默地一个文件都不写（跑完才发现目录不存在）。
            #   和 §8.15 那条「要求写在文档里、没写在代码里」是同一个病。
            request_text=req.render(),
        )
        if not decided:
            self.trace.append({
                "step": step, "node": node, "violation": "below_threshold", "fatal": True,
                "detail": (f"{node}: top={answer.top():.3f} < {question.threshold}"
                           f"（答得不果断,不许拿它往下走）"),
            })
            return None
        return answer

    def _blocked(self, session: Session, view: DecisionView, why: str,
                 *, answer: Answer | None = None) -> Decision:
        """这一步发不出去 / 判定不成立。

        ★ 返回 `unparsed` 而不是硬凑一个动作 —— `unparsed` 的措辞是
        **「没答出来」不是「答错了」**（`core/controller.py`）。循环会重试,
        重试用完就 `escalate`,而 `escalate` 是一种**能被看见**的结果。

        ⚠️ 弃答的**原因**放进 `raw`,而且开头就写明是 `typed:` ——
        不写的话,日志上它和「模型格式错了」长得一样,而那是完全不同的事。
        """
        detail = ""
        if answer is None and self.trace:
            detail = self.trace[-1].get("detail", "")
        return Decision(kind="unparsed", syntax="typed",
                        raw=f"typed: {why}{' — ' + detail if detail else ''}")


def _wire(question: Question) -> dict:
    """`Question` → 线协议的问题形状。**逐字抄 `src/vocab.ts`。**

    ★★ 第一版这三个字段名**全是我猜的**（`kind` / `ask` / `options`）,
    而正本是 **`type` / `instructions` / `criteria`**。这和 `top()` 那次是同一个毛病:
    **猜一份已经写在隔壁的协议,而不去读它。**

    形状（`vocab.ts`）::

        noul   {type, instructions, criteria?: {true, false}}
        choice {type, instructions, criteria: {选项: "什么条件下该选它"}}
        score  {type, instructions, criteria: [档位, 从低到高]}

    ★ `choice` 的 `criteria` 键**就是选项本身**（会原样回到 `answers[id].choice`）,
    不是选项列表 —— 我之前写成 `options` 一个 map。
    """
    if question.kind == "choice":
        return {"type": "choice", "instructions": question.ask,
                "criteria": {o: question.criteria.get(o, "") for o in question.options}}
    if question.kind == "score":
        return {"type": "score", "instructions": question.ask,
                "criteria": list(question.criteria.values())}
    wire = {"type": "noul", "instructions": question.ask}
    if question.criteria:
        wire["criteria"] = dict(question.criteria)
    return wire


__all__ = ["TypedController", "candidates", "DONE", "NODE_THRESHOLDS",
           "NEEDS_TOOL_ASK", "NEEDS_TOOL_CRITERIA", "PICK_TOOL_ASK",
           "PICK_INPUT_ASK"]
