"""决策帧 —— 三条不变量 + **三次实测事故的回归**。

帧这一层的事故**全部看起来像「判定模型判错了」,而实际上是我们喂错了**。
所以这里每一条测试都对着一次真实事故写,不测「函数返回什么」。

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

from experiments.core.frame import (  # noqa: E402
    MAX_LISTED_CANDIDATES,
    MAX_OPTIONS,
    AgentCtx,
    FrameField,
    FrameSpec,
    NODE_FRAMES,
    candidate_provider,
    compile_frame,
    ctx_from_steps,
    frame_for,
)
from experiments.core.types import Action, Step, Tool  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 不变量一:有界,而且**截了要报**
# ═══════════════════════════════════════════════════════════


def test_clipping_is_reported_not_silent() -> None:
    """★★ 事故一:`canDeliver` 曾把工具结果 clip 到 **100** 字符。

    交付闸门拿被截断的证据去核对回答,**正确地**判出 `unsupported=0.67`。
    **判定没错,是帧喂少了。**

    教训不是「调大预算」,是**截断必须报出来** —— 判定模型看不到「这里少了 900 字」,
    它会当成「证据就这么多」。所以下面这个断言是两条:
    正文里有痕迹,而且 `truncations` 里记得原长。
    """
    spec = FrameSpec("t", (FrameField("last_result", 100, "output"),))
    frame = compile_frame(spec, AgentCtx(last_result="x" * 500))
    assert frame.truncations["last_result"] == (500, 100)
    assert "truncated" in frame.render() and "500→100" in frame.render()


def test_no_truncation_no_note() -> None:
    """没截就不该冒出「截断」的提示 —— 否则那句话会变成噪声,没人再看。"""
    spec = FrameSpec("t", (FrameField("last_result", 500, "output"),))
    frame = compile_frame(spec, AgentCtx(last_result="short"))
    assert frame.truncations == {}
    assert "truncated" not in frame.render()


def test_candidate_list_cap_is_reported() -> None:
    """候选被窗口截断时**必须明说**（TS 的 `MAX_FILE_OPTIONS = 20` 那条）。

    不说的话,判定模型会以为「就这些候选」—— 而它本来可能选的是第 21 个。
    """
    spec = FrameSpec("t", (FrameField("candidates_list", 4000, "options", clip="list"),))
    ctx = AgentCtx()
    setattr(ctx, "candidates_list", [f"f{i}" for i in range(50)])
    frame = compile_frame(spec, ctx)
    assert frame.truncations["candidates_list"] == (50, MAX_LISTED_CANDIDATES)
    assert "truncated" in frame.render()


# ═══════════════════════════════════════════════════════════
# 不变量二:**只给声明的字段**
# ═══════════════════════════════════════════════════════════


def test_step_ok_frame_deliberately_has_no_task() -> None:
    """★★★ 事故二:`stepOk` 的帧里**带着 `task`**。

    以前帧带着 `task`、问题写着 "for the task"、判据写着 "what the task needed",
    三处一起把它拉到了任务级。实测任务「读一下 invoice.ts」时第一步 `list_dir`
    确实**成功**了,但没回答那个文件定义了哪些函数,于是 `ok=0.470` 判否 →
    `stop` → **整个循环结束,任何多于一个工具的任务都跑不完。**

    所以这里测两件事:**帧里没有 `task`**,而且**声明里写清了为什么**。
    """
    spec = frame_for("stepOk")
    assert "task" not in {f.name for f in spec.fields}
    ctx = AgentCtx(task="读一下 invoice.ts", last_tool="list_dir",
                   last_input=".", last_result="invoice.ts")
    rendered = compile_frame(spec, ctx).render()
    assert "读一下 invoice.ts" not in rendered, "task 漏进 stepOk 的帧了"
    # 声明里必须留下「为什么不能加回去」—— 删掉一个字段之后没有东西记得它来过
    reasons = dict(spec.excluded)
    assert "task" in reasons and "任务级" in reasons["task"]


def test_every_node_declares_what_it_deliberately_excludes() -> None:
    """**每个节点都要说清「故意不看什么」。**

    三次事故里有两次是「不该看的看了」。没有这一栏,下一个人只会看到
    「这里少了个字段」,然后好心地加回去。
    """
    for node, spec in NODE_FRAMES.items():
        assert spec.excluded, f"{node} 没声明任何排除项 —— 那这一栏就白设了"
        for name, why in spec.excluded:
            if name == "_note":
                continue
            # 排除项和字段集**不能同时包含同一个名字**
            assert name not in {f.name for f in spec.fields}, f"{node}: {name} 既声明又不声明"
            assert why.strip(), f"{node}.{name} 没说为什么"


def test_undeclared_field_never_enters_the_frame() -> None:
    """没声明的字段**一个都不进** —— 这是「帧是声明出来的」那句话的判据。"""
    spec = FrameSpec("t", (FrameField("task", 100, "task"),))
    ctx = AgentCtx(task="t", draft="SECRET-DRAFT", last_result="SECRET-RESULT")
    rendered = compile_frame(spec, ctx).render()
    assert "SECRET" not in rendered


# ═══════════════════════════════════════════════════════════
# 不变量三:**缺的要说**,不静默留白
# ═══════════════════════════════════════════════════════════


def test_missing_field_is_reported_not_left_blank() -> None:
    """留白会让判定模型以为「证据就这么多」。

    **「拿不到」和「本来就没有」是两件事**,而它们在正文里长得一样 —— 除非写出来。
    """
    spec = frame_for("canDeliver")
    frame = compile_frame(spec, AgentCtx(task="t"))  # 没有 draft
    assert "draft" in frame.missing
    assert "absent" in frame.render() and "draft" in frame.render()


# ═══════════════════════════════════════════════════════════
# 事故三:**清单不是计数**
# ═══════════════════════════════════════════════════════════


def test_already_done_is_a_list_not_a_count() -> None:
    """★★★ 事故三:`needsTool` 拿到的曾经是 `steps_done: 2`（一个计数）,而它需要一份清单。

    计数分不出「读过了」和「写过了」。实测任务「把 alpha.ts 里的 totalOf 抄到
    summary.ts」,读完 alpha.ts 之后 `needsTool` 判了 `answer`（0.36）——
    信息确实齐了 —— 于是 loop 直接去生成回答,**文件从没被写出来**。

    所以帧里给的必须是**清单**,而且每一项要能看出「做过什么」。
    """
    steps = [
        Step(index=0, action=Action(kind="tool", name="read_file", arguments={"path": "alpha.ts"}),
             observation="export function totalOf(...)"),
        Step(index=1, action=Action(kind="tool", name="write_file", arguments={"path": "summary.ts"}),
             observation="wrote 120 bytes"),
    ]
    ctx = ctx_from_steps("把 alpha.ts 里的 totalOf 抄到 summary.ts", steps)
    frame = compile_frame(frame_for("needsTool"), ctx)
    rendered = frame.render()
    assert "read_file" in rendered and "write_file" in rendered, "清单里要看得出做过什么"
    assert "alpha.ts" in rendered and "summary.ts" in rendered, "参数也要在,否则分不出抄的是哪个"


def test_needs_tool_frame_carries_the_task_and_the_history() -> None:
    """`needsTool` 判的是「任务还有没有没做的动作」—— 所以它**两面都需要**。"""
    names = {f.name for f in frame_for("needsTool").fields}
    assert {"task", "history"} <= names


# ═══════════════════════════════════════════════════════════
# 指纹与查找
# ═══════════════════════════════════════════════════════════


def test_frame_digest_changes_when_the_frame_changes() -> None:
    """帧的指纹进日志 —— 两次运行帧不一样而没人发现,「同一个方法」就不成立。"""
    spec = frame_for("canDeliver")
    a = compile_frame(spec, AgentCtx(task="t", draft="答 A"))
    b = compile_frame(spec, AgentCtx(task="t", draft="答 B"))
    assert a.digest() != b.digest()
    assert a.digest() == compile_frame(spec, AgentCtx(task="t", draft="答 A")).digest()


def test_unknown_node_raises_rather_than_returning_an_empty_frame() -> None:
    """打错节点名不该静默给一个空帧 —— 空帧会让判定模型答一个没有依据的答案。"""
    with pytest.raises(KeyError, match="没有"):
        frame_for("no_such_node")


# ═══════════════════════════════════════════════════════════
# 候选来源 —— §8.2 那条硬约束在帧这一层的落点
# ═══════════════════════════════════════════════════════════


def test_a_tool_without_an_enum_has_no_candidate_provider() -> None:
    """★ **没有候选来源的参数不能被 `pickInput` 选中。**

    判定模型只能从枚举里挑（实测 77 个候选时选中概率掉到 0.425）。
    所以自由文本参数（`search[entity]` 那种）**不能**做成 `choice` ——
    那是在骗自己,它只能靠猜。那种情况要走「生成提候选、判定排序」。
    """
    free = Tool(name="search", description="d",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}})
    assert candidate_provider(free) is None


def test_a_tool_with_an_enum_gets_a_provider() -> None:
    closed = Tool(name="open", description="d",
                  parameters={"type": "object",
                              "properties": {"path": {"type": "string", "enum": ["a.ts", "b.ts"]}}})
    provider = candidate_provider(closed)
    assert provider is not None
    assert provider(AgentCtx()) == ["a.ts", "b.ts"]


# ═══════════════════════════════════════════════════════════
# ★★ 第 10 轮 R2/R5 —— 那条失效链的回归
# ═══════════════════════════════════════════════════════════


def _req(node: str = "pickTool", n_options: int = 3, ctx: AgentCtx | None = None):
    from experiments.core.frame import Question, Request, frame_for as _ff

    opts = tuple(f"tool_{i}" for i in range(n_options))
    q = Question(node=node, kind="choice", ask="选哪个工具？", options=opts)
    return Request(frame=compile_frame(_ff(node), ctx or AgentCtx(task="t")), question=q)


def test_a_request_over_the_option_budget_is_fatal() -> None:
    """★★★ R5 前半:选项被推到问题那一侧 ⇒ **帧有界,选项无界**。

    以前 `FrameSpec` 只声明字段预算,选项数**没有任何人管** —— 而两者抢的是
    同一段上下文（实测 77 个候选时选中概率掉到 **0.425**）。

    所以预算必须在 `Request` 这个粒度上算。30 > 20 时**必须**报出来。
    """
    req = _req(n_options=30)
    codes = {v.code for v in req.check()}
    assert "options_over_budget" in codes
    assert [v.code for v in req.fatal] == ["options_over_budget"], "超预算是 fatal —— 不许照发"


def test_a_request_at_the_option_budget_is_clean() -> None:
    """边界:`MAX_OPTIONS` 本身**是允许的** —— 否则这条不变量会因为差一而天天误报,然后被无视。"""
    assert _req(n_options=MAX_OPTIONS).check() == []


def test_the_request_budget_covers_the_options_not_just_the_frame() -> None:
    """★★ R5 后半:以前只算帧的字符数,选项那几百字符**不在账上**。

    帧很短、选项很长 —— 这正是「帧有界、请求无界」的样子。
    """
    req = _req(n_options=3)
    assert len(req.frame.render()) < 400, "帧确实很短"
    assert req.chars() > len(req.frame.render()), "请求比帧长——选项要算进去"
    for opt in req.question.options:
        assert opt in req.render()


def test_a_candidate_that_was_already_done_is_reported_but_not_fatal() -> None:
    """★★★ R2:候选里混进了**已经做过的动作**,而没有任何东西检查过。

    以前 `candidates` 是调用方塞进来的,帧编译器只管渲染 —— 于是
    「做过的动作还在候选里」这件事**没有任何接收方**。

    实测后果:写完文件后 `write_file` 还在候选里,模型**会再选它**（§8.4）。

    ⚠️ **但不是 fatal**:重读一个文件有时是合理的。R2 的病不是「发生了重复」,
    是「**没人看这个不变量**」—— 所以断言的是「报出来了」,不是「拦住」。
    """
    steps = [Step(index=0, action=Action(kind="tool", name="tool_1", arguments={}), observation="ok")]
    ctx = ctx_from_steps("t", steps)
    req = _req(node="pickTool", n_options=3, ctx=ctx)  # 候选里有 tool_1,而 tool_1 做过了

    v = [x for x in req.check(ctx) if x.code == "candidate_already_done"]
    assert len(v) == 1 and "tool_1" in v[0].detail
    assert not v[0].fatal, "重复候选要报,但不能拦住整个请求"
    assert req.fatal == [], "所以它对 fatal 没有贡献"


def test_check_without_ctx_cannot_see_done_actions_and_that_is_the_api_saying_so() -> None:
    """★ 不传 `ctx` 就查不了 R2 —— 但那是**签名上看得见的**,不是静默通过。

    判定「候选是不是做过了」需要历史。没有历史时 `check()` 只能跳过这一条,
    而调用方从「要传 ctx」这件事就知道自己放弃了什么。
    """
    req = _req(n_options=3)
    assert req.check() == []
    assert req.check(AgentCtx(task="t")) == [], "空历史里没有做过的动作"


# ═══════════════════════════════════════════════════════════
# ★★ 缺字段要分两档 —— 否则这条不变量会被自己的噪声淹掉
# ═══════════════════════════════════════════════════════════


def test_a_field_that_does_not_exist_yet_does_not_raise_a_violation() -> None:
    """★★★ **第 0 步没有 `last_result` 不是缺陷,是正常状态。**

    这一条是修出来的。第一版把「声明了但这次是空的」一律当缺字段报,
    于是 `pickTool` 在**每一步**都报一次 —— 而它唯一真正重要的那次
    （`canDeliver` 的证据被 clip 到 100 字符,闸门**正确地**判出
    `unsupported=0.67`）就淹没在里面了。

    **「天天误报」和「没有这条检查」在效果上没有区别** —— 两种都不再有人看。
    """
    req = _req(node="pickTool", ctx=AgentCtx(task="把 a 抄到 b"))  # 第一步,没有 last_result
    assert req.check() == [], "第一步就报缺字段 = 噪声"
    # 但它**仍然**渲染出来（缺了就得说,只是不拦）
    assert "last_result" in req.frame.missing
    assert req.frame.missing_required == ()


def test_a_missing_judgement_basis_blocks_the_request() -> None:
    """★★★ 事故一的形状,但走的是**另一条路**:`canDeliver` 没有 `draft`。

    判定模型不会说「我看不到」,它会当成「证据就这么多」——
    于是**正确地**判出一个没有依据的结论。所以依据字段缺席必须**拦住请求**:
    发出去只会得到一个凭空生成的答案,而日志上它和一次正常判定长得一样（§8.10）。

    ⚠️ 注意区分:事故一那次是 draft **被截短了**,那是 `truncations` 的事
    （有依据,只是不全）;这里是 draft **压根没有**（没有可判的对象）。
    """
    ctx = AgentCtx(task="把 a 抄到 b")  # 有 task,没有 draft
    req = _req(node="canDeliver", ctx=ctx)
    v = [x for x in req.check() if x.code == "field_missing"]
    assert len(v) == 1 and v[0].fatal
    assert "draft" in v[0].detail
    assert [x.code for x in req.fatal] == ["field_missing"], "不许发"


def test_truncation_is_not_an_absence() -> None:
    """★ 有依据、只是不全 → 报截断,**不拦**。两件事别混。

    截断的正文里留着 `原文→实际` 的痕迹,判定模型至少知道「这里少了」;
    而缺席是「压根没有」。前者可以判,后者不能。
    """
    long = "证据" * 1000
    req = _req(node="canDeliver", ctx=AgentCtx(task="t", draft=long))
    assert req.frame.missing_required == ()
    assert req.frame.truncations["draft"][0] == len(long)
    assert req.fatal == [], "截断是报出来,不是拦住"


def test_the_seven_nodes_all_declare_a_required_field() -> None:
    """★ 每个节点都得至少有一个**判定依据** —— 否则它就是在没有对象的情况下判。

    反过来也成立:九个 `required` 之外的多余标记会让噪声回来。所以这条测试
    同时钉住「至少一个」和「清单就是这九个」。
    """
    from experiments.core.frame import NODE_FRAMES

    got = {n: sorted(f.name for f in s.fields if f.required) for n, s in NODE_FRAMES.items()}
    assert all(v for v in got.values()), f"有节点没有声明判定依据:{got}"
    assert got == {
        "needsTool": ["task"],
        "pickTool": ["task"],
        "pickInput": ["task"],
        "gradeRisk": ["last_input"],
        "stepOk": ["last_result"],
        "isDone": ["task"],
        "canDeliver": ["draft", "task"],
    }
