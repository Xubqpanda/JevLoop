"""`react × typed` —— 论文 Table 2 的第一个格子。

这个文件测的不是「准确率」,是**两条臂只差一个控制器**这句话在代码里成不成立。

★ 最值钱的一条是 `test_the_typed_arm_does_not_reuse_the_react_prompt` ——
它钉住的是**加第二个控制器实现时撞出来的接口缺口**（`DecisionView` 只有
渲染好的 ReAct 提示词,没有任务原文）。那一条在只有一种实现时**永远看不到**。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.baseline.common import LoopConfig, run_loop  # noqa: E402
from experiments.baseline.react import ReAct  # noqa: E402
from experiments.benchmark.toy import CAPITALS, ToyCapitals  # noqa: E402
from experiments.core.agent import Session  # noqa: E402
from experiments.core.controller import DecisionView  # noqa: E402
from experiments.core.deciding import DecideResponse, parse_answers  # noqa: E402
from experiments.core.models import CallableModel, Message  # noqa: E402
from experiments.core.tools import ToolExecutor  # noqa: E402
from experiments.jloop.typed import TypedController, candidates  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 夹具
# ═══════════════════════════════════════════════════════════


class ScriptedClient:
    """按脚本回答的判定后端。**它把每次请求都记下来** —— 「有没有发出去」要用。"""

    name = "scripted"

    def __init__(self, *, noul: float | list[float] = 0.9,
                 pick: dict[str, str] | None = None,
                 omit: tuple[str, ...] = (),
                 spread: float = 0.9) -> None:
        self.noul = noul if isinstance(noul, list) else [noul]
        self.pick = pick or {}
        #: 选中项拿多少概率。**调低它就是「答得不果断」** ——
        #: §8.6 要 Mock 走到的那条路,这里可以精确地只让它发生在一道题上。
        self.spread = spread
        self.omit = set(omit)
        self.requests: list = []
        self._call = 0

    def decide(self, request):
        self.requests.append(request)
        noul = self.noul[min(self._call, len(self.noul) - 1)]
        self._call += 1

        raw: dict = {}
        for qid, q in request.questions.items():
            if qid in self.omit:
                continue
            # ★ 读**线协议**的字段名（`type` / `criteria`）—— 这就是
            #   `vocab.ts` 里那个形状。读错了这套测试就白测。
            if q["type"] == "choice":
                opts = list(q["criteria"])
                chosen = self.pick.get(qid, opts[0])
                rest = max(1, len(opts) - 1)
                leftover = max(0.0, 1.0 - self.spread)
                raw[qid] = {"type": "choice", "choice": chosen,
                            "probabilities": {o: (self.spread if o == chosen
                                                  else leftover / rest)
                                              for o in opts}}
            else:
                raw[qid] = {"noul": noul}

        # ★ 走真的归一化,不绕过它 —— 否则这些测试和 `parse_answers` 是两套口径
        answers, dropped, missing = parse_answers(raw, list(request.questions))
        return DecideResponse(answers=answers, provider=self.name,
                              dropped=dropped, missing=missing,
                              degraded=bool(dropped or missing))

    def asked(self, node: str) -> int:
        return sum(1 for r in self.requests if node in r.questions)


def capital_model(seen: list | None = None) -> CallableModel:
    """假模型:任务里问哪个国家就答哪个首都。**记下它看到的每条 prompt。**"""

    def responder(messages: list[Message]) -> str:
        if seen is not None:
            seen.append(messages[-1].content)
        text = messages[-1].content
        for country, capital in CAPITALS.items():
            if f"capital of {country}" in text:
                return capital
        return "I don't know."

    return CallableModel(responder, model_id="fake")


def make_session(arm: str = "react", *, country: str = "France") -> Session:
    bench = ToyCapitals()
    tools = list(bench.tools())
    task = next(t for t in bench.tasks(split="test", limit=None, seed=0)
                if t.task_id == f"toy/{country}")
    return Session(
        run_id=f"test/{arm}", task=task, arm=arm, tools=tools,
        executor=ToolExecutor(tools, bench.tool_impls()),
        model=capital_model(), max_steps=8, temperature=0.0, max_tokens=128,
    )


# ═══════════════════════════════════════════════════════════
# ★★ 两根轴是**组合**,不是两套代码
# ═══════════════════════════════════════════════════════════


def test_both_controllers_drive_the_very_same_loop() -> None:
    """★★ `react × llm` 和 `react × typed` 的差**只可能来自决策者**。

    这不是声明,是构造:两条臂共用同一个 `run_loop`、同一个 `build_prompt`、
    同一批工具、同一个 `max_steps`。**唯一不同的那个字段就是 `cfg.controller`。**
    """
    llm = ReAct()
    typed = ReAct(controller=TypedController(ScriptedClient()))

    assert llm.config.controller is None, "llm 那格用默认控制器"
    assert isinstance(typed.config.controller, TypedController)

    differing = {f for f in LoopConfig.__dataclass_fields__
                 if getattr(llm.config, f) != getattr(typed.config, f)}
    assert differing == {"controller"}, f"两条臂还差了别的:{differing}"


def test_the_arm_name_stays_the_same_so_logs_do_not_get_two_spellings() -> None:
    """日志里 `arm` 决定 `log/<dataset>/<arm>/` 的路径。

    ★ 两条臂靠**目录名后缀**区分,不靠 `cfg.name` —— 否则同一个循环会有两个名字,
    而「我们跑的是哪一条臂」在日志里就说不准了（§3.4 那条计数检查防的是同一类事）。
    """
    assert ReAct().config.name == ReAct(controller=TypedController(ScriptedClient())).config.name


def test_a_run_with_the_typed_controller_produces_a_correct_answer() -> None:
    """端到端:`react × typed` 在 toy 上真的走得通,并交出正确答案。"""
    session = make_session()
    outcome = ReAct(controller=TypedController(ScriptedClient(
        noul=[0.9, 0.1],                      # 第一步要工具,第二步收尾
        pick={"pickInput": "France"},
    ))).solve(session)

    assert outcome.final_answer == "Paris"
    assert not outcome.escalated
    tool_steps = [s for s in outcome.steps if s.action.kind == "tool"]
    assert [s.action.name for s in tool_steps] == ["lookup_capital"]
    assert tool_steps[0].action.arguments == {"country": "France"}


# ═══════════════════════════════════════════════════════════
# ★★★ 加第二个实现才撞出来的接口缺口
# ═══════════════════════════════════════════════════════════


def test_the_typed_arm_does_not_reuse_the_react_prompt() -> None:
    """★★★ **`view.prompt` 是渲染产物,不是材料。**

    它是 `build_prompt` 拼好的 **ReAct 提示词** —— 里面写着
    `Action: <tool>[<argument>]`。拿它去让 LLM 生成最终答案,模型会照着格式
    吐一个动作行,而我们把那一行当答案收下。

    加 `TypedController` 时撞上的:想自己拼提示词,却发现**任务原文不在 view 里**,
    它只存在于 `prompt` 字符串的中间某处。于是第二个实现只有两条路:
    去 `prompt` 里做字符串手术,或者照抄 ReAct 的格式。**两条都是错的。**

    修法是把材料补进 view（`task_prompt`）。这条测试钉住的是:**生成答案用的
    提示词里不许出现 ReAct 的动作格式** —— 加了 `task_prompt` 之后仍然可能被改回去。
    """
    seen: list[str] = []
    session = make_session()
    session.model = capital_model(seen)

    ReAct(controller=TypedController(ScriptedClient(
        noul=[0.9, 0.1], pick={"pickInput": "France"},
    ))).solve(session)

    assert seen, "模型一次都没被调用?"

    # ★ 认「生成答案那一次」的判据:**它带着任务原文**（ReAct 的提示词也带着,
    #   但那条是 `build_prompt` 拼的,里面一定有 `Action:`）。
    #   ⚠️ 原来这里用「含 final answer」来认,而后来我们把那句多余的契约删了 ——
    #     **一条依赖措辞的测试,会因为我们改措辞而失效,而不是因为被测的东西坏了。**
    answer_prompts = [p for p in seen if "What is the capital of" in p]
    assert answer_prompts, "没找到生成答案的那次调用"
    for prompt in answer_prompts:
        assert "Action:" not in prompt, (
            "生成答案的提示词里带着 ReAct 的动作格式 —— "
            "模型会照着吐一个动作,而我们把它当答案收下"
        )


def test_the_typed_arm_does_not_add_a_second_output_contract() -> None:
    """★★★ **一条输出契约就够了** —— 而且它归 benchmark。

    `core/bench.py` 的分工表:「任务陈述 + **输出契约** → **benchmark**;
    交互协议 → baseline。**baseline 不许改任务陈述,只能加自己的协议块。**」

    实测代价(GSM8K × 100,单 commit):`react-typed` **63%**、9.8 个输出 token;
    而 `direct` **97%**、131.7 个 token —— **同一个模型**。
    我们那条 `ANSWER_FORMAT`(「只给最终答案」)**把推理一起禁掉了**。

    ★ 这正是已经记过的教训(两条契约并存时看不出模型听了哪条)——
      记过了还是又犯了一次,所以这条测试要**机械地**钉住。
    """
    session = make_session()
    seen: list[str] = []
    session.model = capital_model(seen)

    ReAct(controller=TypedController(ScriptedClient(
        noul=[0.9, 0.1], pick={"pickInput": "France"},
    ))).solve(session)

    answer_prompts = [p for p in seen if "What is the capital of" in p]
    assert answer_prompts
    for prompt in answer_prompts:
        # 契约只该出现一次,而且只该是 benchmark 写的那一条
        assert prompt.count("Answer with the city name only") == 1
        assert "final answer only" not in prompt.lower(), "我们又加了一条输出契约"
        assert "Do not emit an action" not in prompt, "同上"


def test_the_view_carries_the_task_text_not_only_a_rendered_prompt() -> None:
    """★ 上一条的根:view 里得有**任务原文**,不能只有渲染好的提示词。

    `task_prompt` 是后加的。它缺席时,任何不按 ReAct 格式生成的控制器
    都只能去 `prompt` 里做字符串手术 —— 而那是**靠字符串巧合**在工作。
    """
    import dataclasses

    assert "task_prompt" in {f.name for f in dataclasses.fields(DecisionView)}
    view = DecisionView(prompt="<渲染好的 ReAct 提示词>", tools=(), task_prompt="问题原文")
    assert view.task_prompt == "问题原文"


def test_run_loop_actually_fills_the_task_prompt() -> None:
    """★ 光有字段不算 —— `run_loop` 得真的把它填上,否则它是**永远为空的摆设**。"""
    captured: list[DecisionView] = []

    class Spy:
        name = "spy"

        def decide(self, session, view):
            captured.append(view)
            from experiments.core.controller import Decision
            return Decision(kind="answer", answer="Paris")

    session = make_session()
    session.model = capital_model()
    run_loop(session, LoopConfig(name="spy", instruction="x", controller=Spy()))

    assert captured and captured[0].task_prompt == session.task.prompt
    assert "capital of France" in captured[0].task_prompt


# ═══════════════════════════════════════════════════════════
# §8.4 候选每步重建 —— 在这里是**构造性的**,不是提示
# ═══════════════════════════════════════════════════════════


def test_candidates_drop_what_was_already_done() -> None:
    """★★ §8.4:**做过的动作必须从候选里消失。**

    ★ 这里是**删掉**,不是在帧里提示一句 —— 提示可以被无视,
      而候选列表是模型唯一能选的东西。

    ★★ **删完就是空的,不补「出口」。** 我一度补过一个 `DONE` 选项,
      理由听起来对(「删了就得给条退路」),但实测它**有害**:
      `DONE` 和 `needsTool` 是**同一个问题问了两次**,而两者会矛盾 ——
      `needsTool` 刚说「还有动作」,`pickTool` 就问「哪个工具**或者不做**」。
      模型于是相信前一个,挑一个**语义邻居**再调一次
      (`battle_details`→`war_details`、`currency_conversion`→`unit_conversion`),
      `bfcl-v3-multiple` 上 10 条 `wrong_tool` 全是这个形状。

      ⇒ **「停不停」只归 `needsTool`。** 候选删空了就走到「候选已空 → 生成」,
        而那是**代码按精确规则判的**,不该占一次判定。
    """
    session = make_session()
    seen: list[list[str]] = []

    class Spy:
        name = "spy"

        def decide(self, session_, view):
            from experiments.core.controller import Decision
            from experiments.core.frame import ctx_from_steps
            ctx = ctx_from_steps(view.task_prompt, list(view.history))
            seen.append(candidates(session_, ctx))
            if view.history:
                return Decision(kind="answer", answer="Paris")
            return Decision(kind="tool", tool="lookup_capital",
                            arguments={"country": "France"})

    run_loop(session, LoopConfig(name="spy", instruction="x", controller=Spy()))
    assert seen[0] == ["lookup_capital"]
    assert seen[1] == [], "做过的动作必须从候选里消失;空候选 = 该收尾了"


def test_there_is_no_done_option_in_the_candidates() -> None:
    """★★★ **候选里不许出现「停」这个选项** —— 那是 `needsTool` 的问题。

    实测(2026-09-22):加了 `DONE` 之后 `bfcl-v3-multiple × react-typed`
    的 10 条失败全是「调对了又调一个语义邻居」,而 `DONE` 就在候选里、没被选中。

    ★ 判据不是「它有没有被选中」,而是**同一个问题只该有一个归属**。
    """
    from experiments.core.frame import ctx_from_steps
    from experiments.core.types import Action, Step

    session = make_session()
    assert candidates(session, ctx_from_steps("t", [])) == ["lookup_capital"]
    step = Step(index=0, action=Action(kind="tool", name="lookup_capital",
                                       arguments={"country": "France"}), observation="Paris")
    after = candidates(session, ctx_from_steps("t", [step]))
    assert after == [], f"做完之后应当是空候选,而不是带一个出口:{after}"
    assert all(not c.startswith("__") for c in after), "伪名字不许进候选"

def test_the_redo_violation_would_be_caught_if_a_caller_passed_a_stale_list() -> None:
    """★ 上一条是**构造上不可能**发生;这一条证明**万一发生会被抓住**。

    两道防线都要在:「构造上不会」防的是我们自己写错,
    「会被抓住」防的是以后有人从别的路径塞候选进来。
    """
    from experiments.core.frame import AgentCtx, Question, Request, compile_frame, frame_for
    from experiments.core.types import Action, Step

    steps = [Step(index=0, action=Action(kind="tool", name="lookup_capital",
                                         arguments={"country": "France"}),
                  observation="Paris")]
    ctx = AgentCtx(task="t", history=(), last_tool="", last_result="")
    from experiments.core.frame import ctx_from_steps
    stale = ctx_from_steps("t", steps)

    req = Request(
        frame=compile_frame(frame_for("pickTool"), ctx),
        question=Question(node="pickTool", kind="choice", ask="?",
                          options=("lookup_capital",)),
    )
    codes = {v.code for v in req.check(stale)}
    assert "candidate_already_done" in codes


# ═══════════════════════════════════════════════════════════
# ★★ 有人接收 —— 第 10 轮 R2/R5 那条失效链断在这里
# ═══════════════════════════════════════════════════════════


def test_a_fatal_violation_stops_the_request_from_being_sent() -> None:
    """★★★ **校验发现 → 无人接收 → 请求照发 → 判定静默掉点。**

    断在第三步和断在第四步的区别就是这里:帧缺了判定依据时
    `_ask()` 返回 `None`,而 `client.requests` 里**一条都不会多**。

    发出去会怎样?判定模型答一个没有依据的答案,而日志上它和正常判定一样。
    """
    client = ScriptedClient()
    ctrl = TypedController(client)

    session = make_session()
    # 构造一个「判定依据不在」的帧:空 ctx 让 needsTool 的 task 缺失
    from experiments.core.frame import AgentCtx

    from experiments.jloop.typed import NODE_THRESHOLDS, _noul_question

    blocked = ctrl._ask(session, 0, AgentCtx(), 0, "needsTool",
                        _noul_question("needsTool", "?",
                                       threshold=NODE_THRESHOLDS["needsTool"]))
    assert blocked is None, "缺判定依据时必须返回 None"
    assert client.requests == [], "★ 请求不许发出去"
    assert ctrl.trace[-1]["violation"] == "field_missing" and ctrl.trace[-1]["fatal"]


def test_a_blocked_step_becomes_unparsed_not_a_made_up_action() -> None:
    """★ 发不出去时返回 `unparsed`,**不硬凑一个动作**。

    `unparsed` 的措辞是「**没答出来**」不是「答错了」（`core/controller.py`）。
    循环会重试,重试用完就 `escalate` —— 而 `escalate` 是一种**能被看见**的结果。

    ⚠️ 而且原因必须写明是 `typed:` 开头 —— 不写的话,日志上它和
    「模型格式错了」长得一样,而那是完全不同的事。
    """
    ctrl = TypedController(ScriptedClient(omit=("needsTool",)))
    session = make_session()
    outcome = ReAct(controller=ctrl).solve(session)

    assert outcome.final_answer is None and outcome.escalated
    assert "typed:" in (outcome.error or ""), f"弃答原因没说清是我们这一侧:{outcome.error!r}"


def test_an_unusable_answer_from_the_backend_is_reported_not_defaulted() -> None:
    """★★ §8.10:后端**少给**或给了**畸形**答案 → 报出来,**不许退回默认值**。

    返回 `0.5` 看着更「健壮」,但那会让「后端挂了」和「判定完成了」
    在调用方看来一样 —— 而这两件事要采取的行动完全不同。
    """
    client = ScriptedClient(omit=("needsTool",))
    ctrl = TypedController(client)
    session = make_session()

    assert ctrl.decide(session, _view(session)) is not None  # 先跑一次,占位
    last = ctrl.trace[-1]
    assert last["violation"] == "answer_unusable" and last["fatal"]


def _view(session: Session, *, history: tuple = ()) -> DecisionView:
    return DecisionView(prompt="", tools=tuple(session.tools),
                        task_prompt=session.task.prompt, task_id=session.task.task_id,
                        step=len(history), history=history)


# ═══════════════════════════════════════════════════════════
# 判的 vs 生成的 —— 第一根轴要量的东西
# ═══════════════════════════════════════════════════════════


def test_a_free_text_argument_is_generated_and_marked_as_such() -> None:
    """★★ **哪一步是判的、哪一步是生成的,必须记清楚。**

    §8.2 那条硬约束:判定模型只能从枚举里挑（77 个候选时掉到 0.425）。
    所以一个参数**自由文本**的工具**不能被 `pickInput` 选中** ——
    把它做成 `choice` 是在骗自己,它只能靠猜。

    那种情况只能生成。★ 而含糊过去这个臂就没有意义了:
    论文第一根轴量的就是「多少决定真的被解耦了」。
    """
    from experiments.core.types import Tool
    from experiments.jloop.typed import _wire  # noqa: F401  (确认它在)

    session = make_session()
    session.tools = [Tool(name="search", description="自由文本",
                          parameters={"type": "object",
                                      "properties": {"query": {"type": "string"}}})]
    session.executor = ToolExecutor(session.tools, {"search": lambda query: "ok"})

    client = ScriptedClient(noul=0.9)
    ctrl = TypedController(client)
    decision = ctrl.decide(session, _view(session))
    session.finish_events()          # ★ 批次要封口才进事件流（见 runner 里那两行的顺序）

    assert decision.kind == "tool" and decision.tool == "search"
    assert decision.arguments.get("query"), "自由文本参数得生成出来"
    assert client.asked("pickInput") == 0, "不可枚举的参数不该被拿去当 choice 判"
    # ★ 它记进账了,而且节点名说明了它不是判定
    nodes = [r.node for r in session.decision_records()]
    assert any(n.startswith("pickInput:") for n in nodes), nodes


def test_an_enumerable_argument_is_scored_not_generated() -> None:
    """反面:`country` 有 enum,所以它**应该**被判 —— 而且不该走生成。"""
    session = make_session()
    client = ScriptedClient(noul=0.9, pick={"pickInput": "Peru"})
    ctrl = TypedController(client)
    decision = ctrl.decide(session, _view(session))

    session.finish_events()
    assert decision.arguments == {"country": "Peru"}
    assert client.asked("pickInput") == 1, "有候选来源就该判"
    assert not any("(generated)" in r.answer for r in session.decision_records())


def test_a_single_candidate_is_not_worth_a_decision_request() -> None:
    """★ 只有一个候选还去问「要哪一个」是白花一次判定。

    `needsTool` 刚判过「要不要用工具」,所以工具数=1 时这个问题**没有信息量**。
    §8.1 第三行:能由精确规则定的,不交给判定模型。
    """
    session = make_session()          # toy 只有一个工具
    client = ScriptedClient(noul=0.9)
    decision = TypedController(client).decide(session, _view(session))

    assert decision.tool == "lookup_capital"
    assert client.asked("pickTool") == 0, "唯一候选不该发判定请求"


def test_a_task_with_no_tools_never_asks_whether_it_needs_one() -> None:
    """★ 没有工具的任务（GSM8K 那种）不该问「要不要用工具」——
    那个问题的答案恒为否,问它只是白花一次判定。"""
    session = make_session()
    session.tools = []
    session.executor = ToolExecutor([], {})
    client = ScriptedClient()
    decision = TypedController(client).decide(session, _view(session))

    assert decision.kind == "answer"
    assert client.requests == [], "没有工具时一次判定都不该发"


# ═══════════════════════════════════════════════════════════
# 记账:按批,不按条
# ═══════════════════════════════════════════════════════════


def test_decisions_in_one_step_share_a_batch() -> None:
    """★★ **按批计时,不按记录。**

    一次请求判多路时,每一路都记同一份 latency 再求和会**多算几倍** ——
    实测 `decisionMs` 一度比整轮墙钟还大（3.31s vs 3.29s）。
    """
    session = make_session()
    # 两个工具 ⇒ pickTool 真的要判一次,加上 needsTool,共两次判定、同一批
    from experiments.core.types import Tool
    session.tools = list(session.tools) + [
        Tool(name="other", description="另一个", parameters={"type": "object", "properties": {}})]
    session.executor = ToolExecutor(session.tools, {"lookup_capital": lambda country: "Paris",
                                                    "other": lambda: "ok"})

    TypedController(ScriptedClient(noul=0.9, pick={"pickTool": "lookup_capital"})).decide(
        session, _view(session))
    session.finish_events()

    records = session.decision_records()
    assert len(records) >= 2, f"这一步该有多次判定:{[r.node for r in records]}"
    assert len({r.batch for r in records}) == 1, "同一步的判定要算作同一批"


def test_confidence_and_correctness_land_on_the_same_row() -> None:
    """★ RQ2 全靠这一对:**判定说它有多确定**,和**它到底对不对**,必须在同一行。

    分开记的话,「置信度 0.8 的那批判定对了多少」这个问题就没法回答 ——
    而它是「能不能拿置信度当门限」的唯一依据。
    """
    session = make_session()
    TypedController(ScriptedClient(noul=0.9)).decide(session, _view(session))
    session.finish_events()

    records = session.decision_records()
    assert records
    for r in records:
        assert 0.0 <= r.confidence <= 1.0, f"{r.node} 的置信度不在 [0,1]:{r.confidence}"
        assert isinstance(r.correct, bool)


def test_a_confident_no_is_decisive_even_though_it_is_a_no() -> None:
    """★★★ **两个数是两件事,`top()` 和 `prob_true()` 不是同一个。**

    这是照 TS 的 `topGte` / `probGte` 逐字对齐时发现的 —— 我第一版写反了。

    | | 算什么 | TS |
    |---|---|---|
    | `top()` | **答得果不果断** —— `noul` 上是 `max(p, 1-p)` | `topGte` |
    | `prob_true()` | **「是」的概率** —— `noul` 上就是 `p` | `probGte` |

    所以 `noul=0.1` 是**一个果断的「否」**:`top()` 是 **0.9**,过得了果断度门限;
    而 `prob_true()` 是 **0.1**,过不了「是」的门限。
    同一个答案,两个门限,两个结论 —— 而它们**都对**。

    ⚠️ 第一版我把 `top()` 写成返回 `noul`,于是「果断的否」被读成「不确定」,
    而「要不要工具」那一支反而拿它当「很确定要工具」。**正好反了。**
    """
    from experiments.core.deciding import Answer

    no = Answer(kind="noul", noul=0.1, confidence=0.1)
    assert no.top() == pytest.approx(0.9), "果断的否也是果断"
    assert no.prob_true() == pytest.approx(0.1), "但它不是「是」"

    yes = Answer(kind="noul", noul=0.9, confidence=0.9)
    assert yes.top() == pytest.approx(0.9) and yes.prob_true() == pytest.approx(0.9)

    unsure = Answer(kind="noul", noul=0.5, confidence=0.5)
    assert unsure.top() == pytest.approx(0.5), "0.5 才是不果断"


def test_the_choice_top_is_the_selected_option_not_the_maximum() -> None:
    """★ TS 取的是**被选中的那一项**的概率,不是概率表的最大值。

    后端理论上可以给一个不是最大值的选项（概率表只是个分布,
    而 `choice` 是它自己声明的选中项）。两边取的不是同一个东西时,
    「同一个方法」这句话就有漏洞 —— 照抄 TS。"""
    from experiments.core.deciding import Answer

    a = Answer(kind="choice", choice="b",
               probabilities={"a": 0.9, "b": 0.4, "c": 0.1})
    assert a.top() == pytest.approx(0.4), "取选中项,不取最大值"


def test_an_undecided_choice_is_recorded_as_not_correct() -> None:
    """★★ **别拿 `correct=True` 当护身符。**

    全填 True 的话 `correct` 那一列恒为真,于是
    「置信度和正确性同现」这句话**在账面上永远成立** —— 而它本该被检验。
    """
    from experiments.jloop.typed import NODE_THRESHOLDS

    session = make_session()
    # ★ 只让 **pickInput** 不果断（选中项 0.2,其余 7 个分掉 0.8）——
    #   抬高全局门限会把前面的 needsTool 也一起拦掉,那是另一条测试的事。
    ctrl = TypedController(ScriptedClient(noul=0.9, pick={"pickInput": "France"},
                                          spread=0.2))
    ctrl.decide(session, _view(session))
    session.finish_events()

    rows = [r for r in session.decision_records() if r.node == "pickInput"]
    assert rows, [r.node for r in session.decision_records()]
    assert rows[-1].confidence == pytest.approx(0.2)
    assert rows[-1].confidence < NODE_THRESHOLDS["pickInput"]
    assert rows[-1].correct is False, "没过果断度门限就该记 False"


def test_an_undecided_choice_blocks_the_step_instead_of_guessing() -> None:
    """★★★ **被迫的选择 ≠ 挑得对。**

    选项是穷尽的,判定模型必须挑一个 —— 所以「挑了一个」本身不是证据。
    `top()` 判的正是这件事,不过门就**不许拿它往下走**。

    ★ 没有这道门的时候,`_arguments` 会照拿第一个候选,
    于是「不确定就别猜」在代码里**从来没有被走到过** —— 而 §8.6 要求
    Mock 必须让 policy 自己走到 `escalate`。
    """
    session = make_session()
    ctrl = TypedController(ScriptedClient(noul=0.9, pick={"pickInput": "France"},
                                          spread=0.2))
    decision = ctrl.decide(session, _view(session))

    assert decision.kind == "unparsed", f"不该硬猜:{decision}"
    assert "below_threshold" in {t.get("violation") for t in ctrl.trace}
    assert "typed:" in decision.raw


def test_mock_actually_walks_to_escalate() -> None:
    """★★★ §8.6 的原话:**Mock 一律保守,让 policy 的置信度门限自己走到 `escalate`。**

    这条要求以前**没有被满足** —— Mock 把概率平均分给 8 个国家（每项 0.125）,
    而 `_arguments` 根本不看门限,照拿第一个。于是
    「不确定就别猜」这句话写在了文档里,在代码里走不到。

    现在它走到了:整条臂对着 Mock 跑完,必须**弃答**而不是猜一个。
    """
    from experiments.core.deciding import MockClient

    session = make_session()
    outcome = ReAct(controller=TypedController(MockClient())).solve(session)

    assert outcome.final_answer is None
    assert outcome.escalated, "Mock 下必须弃答 —— 否则「不确定就别猜」没被走到"


def test_one_decision_is_recorded_exactly_once() -> None:
    """★★★ **判定次数是论文的头条指标之一 —— 一次调用绝不能记两行。**

    第一版在「答得不果断」时**又记了一条**:一条 `note=""`、一条 `note="top<0.5"`,
    两次都 `correct=False`。于是同一次判定在账上出现两遍 ——
    而账面上**看不出来**,因为两行各自都长得合理。

    这和 §8.10 那条「按批计时、按记录求和会多算几倍」是**同一类错**:
    一个数被数了两遍,而没有任何东西会因此报错。

    所以「为什么没成立」写在**同一行的 `note` 里**,不是另起一行。
    """
    session = make_session()
    ctrl = TypedController(ScriptedClient(noul=0.9, pick={"pickInput": "France"},
                                          spread=0.2))
    ctrl.decide(session, _view(session))
    session.finish_events()

    picks = [r for r in session.decision_records() if r.node == "pickInput"]
    assert len(picks) == 1, f"同一次判定记了 {len(picks)} 行:{[(r.answer, r.note) for r in picks]}"
    assert picks[0].note, "不成立的原因要在这一行的 note 里"
    assert picks[0].correct is False


def test_a_violation_reaches_the_event_stream_not_just_memory() -> None:
    """★★★ **写进内存里一个 list 不算「有人接收」。**

    第 10 轮那条失效链的中间三步是「校验发现 → **无人接收** → 请求照发」。
    把违规放进 controller 的一个属性上,跑完之后**没人读得到** ——
    下一个人重建现场时看到的只是「这里怎么少了一次判定」。

    所以违规要按「一次没成立的判定」记账,并落进事件流（§8.14 的帧指纹同理）。
    """
    from experiments.core.frame import AgentCtx
    from experiments.jloop.typed import NODE_THRESHOLDS, _noul_question

    session = make_session()
    ctrl = TypedController(ScriptedClient())
    # 空 ctx ⇒ needsTool 缺「判定依据」⇒ fatal ⇒ 请求不发
    assert ctrl._ask(session, session.next_batch(), AgentCtx(), 0, "needsTool",
                     _noul_question("needsTool", "?",
                                    threshold=NODE_THRESHOLDS["needsTool"])) is None
    session.finish_events()

    rows = session.decision_records()
    assert rows, "违规没进事件流 —— 那就还是「无人接收」"
    assert rows[0].node == "needsTool" and rows[0].correct is False
    assert "field_missing" in rows[0].answer
    assert "判定依据" in rows[0].note, f"原因要可读:{rows[0].note!r}"
    assert rows[0].frame_digest, "帧的指纹也要在（§8.14）"


def test_the_frame_digest_separates_nodes_that_see_different_frames() -> None:
    """★★ §8.14:`Frame.digest()` 每次都记。

    这条同时验证两件事,少一件这个串就没有意义:

    ① 同一个节点在**历史不同**时指纹要变 —— 否则「两次运行帧不一样」
       永远发现不了,而「同一个方法」这句话就没有依据;
    ② 只看 `task` 的那一格指纹**不该变** —— 它确实什么都没多看。

    ② 是这条测试的另一半,而且它防的是反过来的错:
    一个**恒变**的指纹(比如掺了时间戳）看起来也在「记录帧」,其实什么都说明不了。
    """
    from experiments.core.frame import AgentCtx, StepRecord, compile_frame, frame_for

    empty = AgentCtx(task="把 a 抄到 b")
    did = AgentCtx(task="把 a 抄到 b",
                   history=(StepRecord(step=0, tool="read_file", input="a", result="..."),),
                   last_tool="read_file", last_input="a", last_result="...")

    # ① 历史进了 `needsTool` 的帧 → 指纹必须变
    needs = frame_for("needsTool")
    assert compile_frame(needs, empty).digest() != compile_frame(needs, did).digest()

    # ② `pickInput` 只看 task（它的 excluded 里明说了不看 last_result / history）
    pick = frame_for("pickInput")
    assert compile_frame(pick, empty).digest() == compile_frame(pick, did).digest()
    # 而 task 一变它就得变 —— 否则它连「看 task」这件事都没做到
    assert compile_frame(pick, empty).digest() != compile_frame(
        pick, AgentCtx(task="换个任务")).digest()


# ═══════════════════════════════════════════════════════════
# 注册 —— 「能跑」和「表里有数」之间差着这一步
# ═══════════════════════════════════════════════════════════


def test_the_typed_cells_are_reachable_by_name() -> None:
    """★ 一条臂写在文件里不等于**能被跑起来**。

    `scripts/run.py` 的名字解析是**显式**的（见 `core/registry.py` 的说明）——
    没登记的名字就是「没有这个 agent」,哪怕实现就在旁边。
    """
    from experiments.scripts.run import resolve_agent, typed_arms

    decider = ScriptedClient()
    assert set(typed_arms(decider)) == {"react-typed", "act-typed"}
    for name in ("react-typed", "act-typed"):
        agent = resolve_agent(name, decider)()
        assert agent.config.controller is not None, f"{name} 没接上控制器"


def test_the_honest_gaps_are_gaps_not_silent_wrong_arms() -> None:
    """★★ **`plan-then-execute × typed` 故意不在表里。**

    它的计划藏在 `view.prompt` 的 preamble 里,而类型化那一路**不看 `prompt`**
    （它用 `task_prompt` 自己拼）—— 接上去的话计划会丢。

    列上去会跑出一个**看起来能跑、其实没在做同一件事**的臂 —— 那比缺一格糟得多:
    缺一格是空的,错的格子是**一个数**,而那个数会被当成「换了决策者」的对照。
    """
    from experiments.scripts.run import resolve_agent, typed_arms

    decider = ScriptedClient()
    assert "plan-then-execute-typed" not in typed_arms(decider)
    assert "rewoo-typed" not in typed_arms(decider), "ReWOO 的 Worker 没有决策可换"


def test_a_typed_arm_needs_a_decider_and_a_plain_arm_does_not() -> None:
    """★ 判定后端**只在真要用的时候才建** —— 一个跑 `direct` 的人
    不该被判定后端的配置拦住。两者问的是不同的问题（§8.9）。

    而且 `--decider http` 缺 key 时**要报错,不许静默退回 mock**:
    静默退回会让「判定后端是官方 Jev」和「判定后端是 mock」在日志上长得一样 ——
    而两者跑出来的数不能放在同一张表里。
    """
    import argparse

    from experiments.core.deciding import MockClient
    from experiments.scripts.run import build_decider

    def ns(**kw):
        return argparse.Namespace(**{"decider": "mock", "decider_url": "http://x/v1",
                                     "decider_key": None, "decider_model": "m", **kw})

    assert isinstance(build_decider(ns()), MockClient), "默认是 mock（离线可跑）"

    # 没有 key ⇒ 报错退出，不是静默降级
    # ★ 变量名和 TS 侧 `backends.ts` 一致 —— 两边读不同的名字就是一个坑
    old = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
            build_decider(ns(decider="http"))
    finally:
        if old is not None:
            os.environ["TYPESAFE_API_KEY"] = old

    # 有 key ⇒ 真的建出 HTTP 客户端，而且**挂着 mock 兜底**（§8.10 每级都报）
    decider = build_decider(ns(decider="http", decider_key="k"))
    assert decider.name == "http:m→mock", decider.name


# ═══════════════════════════════════════════════════════════
# ★★★ 线协议和措辞 —— 两处「照着隔壁抄」而不是「照着自己想」
# ═══════════════════════════════════════════════════════════


def test_the_wire_shape_matches_vocab_ts_exactly() -> None:
    """★★★ 第一版这三个字段名**全是我猜的**（`kind`/`ask`/`options`）,
    而正本是 `src/vocab.ts` 的 **`type`/`instructions`/`criteria`**。

    这和 `top()` 那次是同一个毛病:**猜一份已经写在隔壁的协议,而不去读它。**
    猜错的下场不是报错,是**打到一个真端点上才发现** —— 而那时已经跑了一批。

    ★ 而且 `choice` 的 `criteria` **键就是选项本身**
    （会原样回到 `answers[id].choice`）,不是另开一个 `options` 列表。
    """
    from experiments.core.frame import Question
    from experiments.jloop.typed import _wire

    n = _wire(Question(node="q", kind="noul", ask="A?", criteria={"true": "t", "false": "f"}))
    assert set(n) == {"type", "instructions", "criteria"}
    assert n == {"type": "noul", "instructions": "A?", "criteria": {"true": "t", "false": "f"}}

    c = _wire(Question(node="q", kind="choice", ask="B?", options=("x", "y")))
    assert c["type"] == "choice" and c["instructions"] == "B?"
    assert list(c["criteria"]) == ["x", "y"], "选项就是 criteria 的键"

    s = _wire(Question(node="q", kind="score", ask="C?"))
    assert s["type"] == "score" and "criteria" in s


def test_an_absent_noul_criteria_is_omitted_not_sent_empty() -> None:
    """★ `vocab.ts` 里 `criteria` 对 `noul` 是**可选**的:
    `criteria ? {type,instructions,criteria} : {type,instructions}`。

    送一个空 `{}` 和「不送」不是一回事 —— 后者才是那个类型声明说的形状。"""
    from experiments.core.frame import Question
    from experiments.jloop.typed import _wire

    assert "criteria" not in _wire(Question(node="q", kind="noul", ask="A?"))


def test_needs_tool_asks_about_actions_not_about_tool_calls() -> None:
    """★★★ **这一条钉的是三次事故里的那一次。**

    `DECISION.md` 的 `## needs_tool` 原文:

    > ★ 判据是「任务还有没有没做的动作」,不是「还有没有没拿到的信息」。
    > 实测:任务「把 alpha.ts 里的 totalOf 抄到新文件 summary.ts 里」,
    > 读完 alpha.ts 之后这个节点判了 `answer`(0.36) → loop 直接去生成回答,
    > **文件从没被写出来**。

    我第一版写的措辞是
    `"Does this task still require a tool call before it can be answered?"`
    —— 问的是**工具调用**,而事故的教训正是「要问**任务要求的动作**」。
    「写一个文件」在这个问法下又一次落在问题之外。

    ★ 而且 `vocab.ts` 说「**问题 ID 不会到达模型**」—— 措辞是模型能看到的全部。
    """
    from experiments.jloop.typed import NEEDS_TOOL_ASK, NEEDS_TOOL_CRITERIA

    assert "action" in NEEDS_TOOL_ASK.lower(), "问的是动作,不是工具调用"
    # ★ 判据必须把「写」明说 —— 那正是事故的修法
    assert "writing" in NEEDS_TOOL_CRITERIA["true"]
    assert set(NEEDS_TOOL_CRITERIA) == {"true", "false"}


def test_the_two_choice_nodes_do_not_share_a_threshold() -> None:
    """★★ `DECISION.md` 里 `pick_tool` 是 **0.6**、`pick_input` 是 **0.5**。

    我第一版只有一个全局 `top`。用一个常量卡两个节点,
    **必然改错其中一个,而两边都还是「看起来在卡门限」** ——
    和那个「死配置」的 bug 是同一个形状。

    ★ 顺带钉住:`needsTool` 的 0.5 是 **`probGte`** 语义（P(true)）,
    两个 `choice` 是 **`topGte`** 语义（果断程度）。同一批数字,两种含义。
    """
    from experiments.jloop.typed import NODE_THRESHOLDS

    assert NODE_THRESHOLDS == {"needsTool": 0.5, "pickTool": 0.6, "pickInput": 0.5}
    assert NODE_THRESHOLDS["pickTool"] != NODE_THRESHOLDS["pickInput"]


# ═══════════════════════════════════════════════════════════
# ★★★ 批次耗时量的必须是「判定请求」,不是「两次开批之间」
# ═══════════════════════════════════════════════════════════


def test_a_batch_times_its_own_requests_not_the_gap_between_batches() -> None:
    """★★★ 实测（`bfcl-v3-simple × react-typed`,2026-09-22）:`decision_ms` 记成了**整个任务时长**。

    因为 `next_batch()` 在 `decide()` 开头关上一批,而那一批的开批时间是
    **上一步 decide 开始时** —— 于是中间的**生成调用**和**工具执行**全落在区间里。
    后果:`decison_ms ≈ wall`,和 `tool_ms` **双重计算**,
    `framework_ms = wall − model − decision − tool` 被减成 **−1670ms / −11335ms**。

    ★ **负的框架时间是症状,不是病。** 病是那个数**量的不是它名字说的东西** ——
    和今天修的其他几个是同一个形状。

    所以批次耗时改成**该批内判定请求自己的墙钟之和**。
    """
    session = make_session()
    ctrl = TypedController(ScriptedClient(noul=0.9, pick={"pickInput": "France"}))

    # 开批 → 判一次 → 中间**假装去生成/调工具**(纯 sleep,不属于判定)
    batch = session.next_batch()
    ctrl.decide(session, _view(session))
    time.sleep(0.05)                     # ← 这 50ms 绝不该进 decision
    session.finish_events()

    batches = session.decision_batches()
    assert batches, "一批都没有"
    for b in batches:
        if b.requests_in_batch:
            assert b.latency_ms < 50.0, (
                f"批次耗时 {b.latency_ms:.1f}ms 把开批之后那 50ms 也算进去了 —— "
                f"它量的应该是判定请求自己"
            )


def test_requests_and_questions_are_counted_separately() -> None:
    """★★ **两个数分开记,而它们的比值就是论文要量的东西。**

    - **请求数** = HTTP 往返次数。PLAN 表 2:「一步 4–6 次请求,
      这是全项目最大的已知浪费」—— 那个「4–6」就是这个计数器。
    - **题数** = 几道判定题。一次请求可以判多路（Jev 一次前向并行打分）。

    相等 = 一步一请求;请求数远小于题数 = 合并得对。
    **合成一个数就永远分不出这两种情况** —— 而这两种情况的成本差着几倍。
    """
    session = make_session()
    # 两个工具 ⇒ pickTool 真要判一次;加上 needsTool、pickInput,一步三次请求
    from experiments.core.types import Tool
    session.tools = list(session.tools) + [
        Tool(name="other", description="另一个", parameters={"type": "object", "properties": {}})]
    session.executor = ToolExecutor(session.tools, {"lookup_capital": lambda country: "Paris",
                                                    "other": lambda: "ok"})

    TypedController(ScriptedClient(noul=0.9, pick={"pickTool": "lookup_capital"})).decide(
        session, _view(session))
    session.finish_events()

    b = session.decision_batches()[0]
    assert b.requests_in_batch > 0
    assert b.questions_in_batch == b.requests_in_batch, (
        "现在每道题各发一次请求 —— 相等是对的,而**它本身就是要量的那个浪费**"
    )


def test_a_single_tool_dataset_still_terminates_after_doing_it() -> None:
    """★★★ 回归:`bfcl-v3-simple × react-typed` 曾经从 **97/100 掉到 0/100**。

    那个子集只有 **1 个工具**。做完之后 `candidates()` 是**空**,
    而「唯一候选」那条捷径对空列表不成立 —— 应当走**「候选已空 → 生成」**。

    ★ 加 `DONE` 的那一版把空候选变成了 `[DONE]`,于是捷径把 `DONE` 当成工具名
      去 `view.tools` 里找,`next(...)` 直接 `StopIteration` → 整题 `agent_error`。
      **现在 `DONE` 删了,那条路不存在了** —— 而这条测试留着防它回来。
    """
    from experiments.core.controller import Decision
    from experiments.core.types import Action, Step
    from experiments.jloop.typed import TypedController

    session = make_session()           # toy 只有一个工具
    client = ScriptedClient(noul=0.9)  # needsTool 恒为「还要」

    class Spy(TypedController):
        def _generate(self, session, view, ctx, *, why, batch=0):
            return Decision(kind="answer", answer="PARIS", syntax="generated")

    ctrl = Spy(client)
    d1 = ctrl.decide(session, _view(session))
    assert d1.kind == "tool" and d1.tool == "lookup_capital"

    step = Step(index=0, action=Action(kind="tool", name="lookup_capital",
                                       arguments={"country": "France"}),
                observation="Paris")
    session.next_batch()
    d2 = ctrl.decide(session, _view(session, history=(step,)))
    assert d2.kind == "answer", f"候选空了必须能收尾,而不是炸:{d2}"
