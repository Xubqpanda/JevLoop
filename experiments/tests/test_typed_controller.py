"""`react × typed` —— 论文 Table 2 的第一个格子。

这个文件测的不是「准确率」,是**两条臂只差一个控制器**这句话在代码里成不成立。

★ 最值钱的一条是 `test_the_typed_arm_does_not_reuse_the_react_prompt` ——
它钉住的是**加第二个控制器实现时撞出来的接口缺口**（`DecisionView` 只有
渲染好的 ReAct 提示词,没有任务原文）。那一条在只有一种实现时**永远看不到**。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import sys
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
                 omit: tuple[str, ...] = ()) -> None:
        self.noul = noul if isinstance(noul, list) else [noul]
        self.pick = pick or {}
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
            if q["kind"] == "choice":
                opts = list(q["options"])
                chosen = self.pick.get(qid, opts[0])
                rest = max(1, len(opts) - 1)
                raw[qid] = {"type": "choice", "choice": chosen,
                            "probabilities": {o: (0.9 if o == chosen else 0.1 / rest)
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
    answer_prompts = [p for p in seen if "final answer" in p.lower()]
    assert answer_prompts, "没找到生成答案的那次调用"
    for prompt in answer_prompts:
        assert "Action:" not in prompt, (
            "生成答案的提示词里带着 ReAct 的动作格式 —— "
            "模型会照着吐一个动作,而我们把它当答案收下"
        )


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
    """★★ §8.4:**固定的候选会让模型去选一个已经不适用的动作。**

    实测:写完文件后 `write_file` 还在候选里,模型**会再选它**。

    ★ 这里是**删掉**,不是在帧里提示一句 —— 提示可以被无视,而候选列表
    是模型唯一能选的东西。第 10 轮 R2 的病正是「候选由调用方给,没人检查删没删」。
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
    assert seen[1] == [], "做过的动作必须从候选里消失"


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

    blocked = ctrl._ask(session, 0, AgentCtx(), 0, "needsTool",
                        __import__("experiments.jloop.typed", fromlist=["x"])
                        ._noul_question("needsTool", "?"))
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


def test_a_low_confidence_decision_is_recorded_as_not_correct() -> None:
    """★★ **别拿 `correct=True` 当护身符。**

    门限没过就是没过 —— 把所有判定都记成 correct,`correct` 这一列就恒为真,
    于是「置信度和正确性同现」这句话**在账面上永远成立**,而它本该是被检验的。
    """
    session = make_session()
    # noul=0.1 < DEFAULT_YES ⇒ needsTool 判否
    ctrl = TypedController(ScriptedClient(noul=0.1))
    decision = ctrl.decide(session, _view(session))
    session.finish_events()

    assert decision.kind == "answer"
    rows = [r for r in session.decision_records() if r.node == "needsTool"]
    assert rows, "needsTool 这一次判定要记账"
    assert rows[-1].confidence == pytest.approx(0.1)
    assert rows[-1].correct is False, "没过门限就该记 False"


# ═══════════════════════════════════════════════════════════
# 帧的指纹进轨迹 —— 「同一个方法」这句话要有依据
# ═══════════════════════════════════════════════════════════


def test_the_trace_carries_the_frame_digest_and_what_was_hidden() -> None:
    """★★ `Frame.digest()` 每次记（§8.14）。

    **两次运行帧不一样而没人发现,「同一个方法」这句话就不成立。**
    而且截断和缺失要一起进轨迹 —— 判定模型看不到「这里少了 900 字」,
    它会当成「证据就这么多」,然后**正确地**判出一个错的结论。
    """
    session = make_session()
    ctrl = TypedController(ScriptedClient(noul=0.9))
    ctrl.decide(session, _view(session))

    judged = [t for t in ctrl.trace if "frame_digest" in t]
    assert judged, ctrl.trace
    t = judged[0]
    assert t["frame_digest"] and len(t["frame_digest"]) >= 8
    assert "truncations" in t and "missing" in t
    assert t["provider"] == "scripted"


# ═══════════════════════════════════════════════════════════
# ★★★ runner 里那两行的顺序 —— 每一题的最后一批判定曾经缺席
# ═══════════════════════════════════════════════════════════


def test_the_last_batch_is_missing_until_it_is_closed() -> None:
    """★★★ 修于 2026-09-22:`runner.py` 里 `trajectory()` 曾经在 `finish_events()` **之前**。

    而 `trajectory()` 的 `decision_records()` 是从**事件流**里读的 ——
    最后一批判定那时还躺在 `_batch_decisions` 里没进事件流。于是
    **每一题的最后一批判定都缺席了**,而它恰恰是决策性的那批
    （收尾那一步:判「不用再调工具了」的那次）。

    ★ 这条测试把「顺序有要求」这件事变成**看得见的**:
    先建轨迹拿不到,封口之后再建才拿得到。少了它,下一个人把两行换回去
    不会有任何东西变红 —— 而这次的 bug 就是这么进来的。
    """
    session = make_session()
    ctrl = TypedController(ScriptedClient(noul=0.9))
    ctrl.decide(session, _view(session))

    from experiments.core.agent import AgentOutcome

    before = session.trajectory(AgentOutcome(steps=[]))
    session.finish_events()
    after = session.trajectory(AgentOutcome(steps=[]))

    assert len(after.decisions) > len(before.decisions), (
        "封口之后轨迹里的判定应该变多 —— 不变的话说明这一批本来就没丢,"
        "那这条测试就测不到东西了"
    )
    assert before.decisions == (), "没封口时最后一批确实不在轨迹里"


def test_the_runner_closes_the_batch_before_building_the_trajectory(tmp_path) -> None:
    """★ 上一条的**整格版**:真的跑一次 `run_cell`,看**交给评测的那个 trajectory**。

    ⚠️ 第一版这条测试数的是 `events.jsonl` 里的判定批 —— **它抓不住这个 bug**。
    把 runner 改回错的顺序,它照样是绿的:因为 `finish_events()` 虽然在
    `trajectory()` 之后,却在**写事件之前**,所以事件流是全的。
    丢的只有**传给 `score` 的那个 trajectory**。

    「测试通过」和「测试抓得住」是两件事 —— 这条测试是被这个区别教出来的:
    写完先跑一遍是绿的,把 bug 改回去**还是绿的**,才知道测错了地方。
    """
    from experiments.core.runner import Cell, run_cell
    from experiments.core.types import Judgment

    seen: list = []

    class Recording(ToyCapitals):
        def score(self, task, trajectory) -> Judgment:
            seen.append(trajectory)
            return super().score(task, trajectory)

    run_cell(
        bench=Recording(),
        make_agent=lambda task, tools: ReAct(controller=TypedController(
            ScriptedClient(noul=[0.9, 0.1], pick={"pickInput": "France"}))),
        model=capital_model(),
        cell=Cell(dataset="toy", arm="typed-probe", seed=0), log_root=tmp_path,
    )

    assert seen, "评测一次都没被调用?"
    steps = {d["step"] for t in seen for d in t.decisions}
    assert steps, f"轨迹里一条判定都没有:{[t.decisions for t in seen]}"
    # ★ **收尾那一步的判定必须在里面** —— 那正是曾经丢掉的那一批
    assert max(steps) >= 1, f"判定只记到第 {max(steps)} 步,收尾那批缺席了"
