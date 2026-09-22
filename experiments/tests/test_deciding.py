"""判定客户端 —— 线协议、畸形响应、降级链。

★ 这个文件存在的理由**一半是为了 `parse_answers` 是纯函数**。
「后端返回畸形答案」这件事隔着网络几乎测不到（要造一个假 HTTP 服务,
而归一化的那条规则本身是纯逻辑）。把它抽出来的唯一动机就是**能在这里钉住**。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.core.deciding import (  # noqa: E402
    Answer,
    DecideRequest,
    DecideResponse,
    FallbackClient,
    HttpJevClient,
    MockClient,
    parse_answers,
)


# ═══════════════════════════════════════════════════════════
# ★★ 认值,不认标签 —— 这条是从 TS 侧搬过来的教训
# ═══════════════════════════════════════════════════════════


def test_a_label_without_a_value_is_dropped_not_coerced_to_zero() -> None:
    """★★★ **一个只有标签、没有值的畸形答案必须被丢掉。**

    以前的写法是 `a.type === 'noul' || typeof a.noul === 'number'`,
    于是 `{"type":"noul"}` 走到 `Number(undefined) || 0`,被**补成 `noul: 0`**。

    它既不进 `dropped` 也不进 `missing`,`degraded` 保持 `false` ——
    **在日志上和一次正常判定完全一样**。而 `0` 在策略里是**明确的否定**
    （「不需要工具」「没成功」）,**比整个后端挂掉危险得多**:
    挂掉会被降级链抓到,伪造的 `0` 不会。
    """
    answers, dropped, missing = parse_answers(
        {"needs_tool": {"type": "noul"}}, ["needs_tool"]
    )
    assert answers == {}, "伪造的 0 绝不能变成答案"
    assert dropped == ["needs_tool"]
    assert missing == ["needs_tool"], "丢了也就算缺 —— 两条都要报"


def test_a_value_without_a_label_is_accepted() -> None:
    """★ 容忍的是**缺标签**,不是缺值。`{noul: 0.7}` 语义完整,照收。

    这条和上一条是一对:**判据是值在不在,不是标签在不在**。
    只看标签会把这条正常的答案拒掉;只看值会把上面那条畸形的收下。
    """
    answers, dropped, _ = parse_answers({"needs_tool": {"noul": 0.7}}, ["needs_tool"])
    assert dropped == []
    assert answers["needs_tool"].kind == "noul"
    assert answers["needs_tool"].noul == pytest.approx(0.7)


def test_an_empty_choice_is_not_an_option() -> None:
    """★ 空字符串放过去会让 `pickInput` 返回 `''`,上游拿它去拼路径,
    错误要隔好几层才暴露出来 —— 那时已经看不出是这里来的。"""
    answers, dropped, _ = parse_answers({"which": {"choice": ""}}, ["which"])
    assert answers == {} and dropped == ["which"]

    answers, dropped, _ = parse_answers({"which": {"choice": "b.ts"}}, ["which"])
    assert dropped == [] and answers["which"].choice == "b.ts"


def test_a_non_numeric_answer_is_dropped() -> None:
    """值在,但**不是数**（字符串 `"0.9"` / 布尔 / 嵌套对象）—— 一样丢。

    ⚠️ 布尔要单独挡:`isinstance(True, int)` 在 Python 里是 `True`,
    不挡的话 `{"noul": true}` 会变成 `1.0` —— 又是伪造的确定答案。
    """
    for bad in ("0.9", True, False, {"v": 1}, [0.9]):
        answers, dropped, _ = parse_answers({"q": {"noul": bad}}, ["q"])
        assert answers == {}, f"{bad!r} 不该被当成数"
        assert dropped == ["q"]


def test_probabilities_survive_and_top_is_the_selected_option_probability() -> None:
    """★★ §8.3:卡阈值要用**选中项的概率**,不是那个叫 `confidence` 的东西。

    Laya 的 `confidence` 是归一化香农熵 —— `p=[0.8,0.2]` 时它是 **0.269**。
    所以 `Answer.top()` 必须能从概率表里算出来,哪怕后端没给 `confidence`。
    """
    answers, _, _ = parse_answers(
        {"pick": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}},
        ["pick"],
    )
    a = answers["pick"]
    assert a.top() == pytest.approx(0.8), "选中项概率"
    assert a.confidence == pytest.approx(0.8), "后端没给 confidence 时用 top 兜底"

    # 后端给了 confidence（熵）时**不要用它盖掉 top** —— 两个是不同的数
    answers, _, _ = parse_answers(
        {"pick": {"type": "choice", "choice": "a",
                  "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.269}},
        ["pick"],
    )
    assert answers["pick"].confidence == pytest.approx(0.269)
    assert answers["pick"].top() == pytest.approx(0.8), "卡阈值必须用这个"


def test_a_score_answer_carries_its_legend() -> None:
    """序数要带图例 —— 否则 `score: 2` 对读日志的人来说没有含义。"""
    answers, _, _ = parse_answers(
        {"risk": {"type": "score", "score": 2, "legend": {"2": "会改到别人的文件"},
                  "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}},
        ["risk"],
    )
    a = answers["risk"]
    assert (a.kind, a.score, a.top()) == ("score", 2, pytest.approx(0.7))
    assert a.legend == {"2": "会改到别人的文件"}


def test_an_answer_for_a_question_nobody_asked_is_ignored_not_fatal() -> None:
    """后端多答一个不是我们要问的 —— 不当错误。**但也不进答案集**,
    否则 policy 会读到一个它没有声明过的判定。"""
    answers, dropped, missing = parse_answers(
        {"asked": {"noul": 0.9}, "extra": {"noul": 0.1}}, ["asked"]
    )
    assert set(answers) == {"asked"}
    assert dropped == [] and missing == []


def test_answers_that_is_not_a_dict_at_all_yields_everything_missing() -> None:
    """后端返回 `answers: null` / 一个数组 / 一个字符串 —— 不能炸,要报成全缺。"""
    for bad in (None, [], "oops", 3):
        answers, _, missing = parse_answers(bad, ["a", "b"])
        assert answers == {} and missing == ["a", "b"]
        assert missing, "全缺也是一种要报出来的状态"


# ═══════════════════════════════════════════════════════════
# §8.10 不假装成功 —— 缺答案要在响应上看得见
# ═══════════════════════════════════════════════════════════


def _serve(payload: dict) -> tuple[str, HTTPServer]:
    """起一个假后端。**这样测的是真的 HTTP 路径,不是被 mock 掉的那个。**"""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:  # 别往测试输出里写字
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv


def _req(questions: dict) -> DecideRequest:
    return DecideRequest(state={"task": "t"}, questions=questions)


def test_a_backend_that_returns_fewer_answers_is_marked_degraded() -> None:
    """★ §8.10:**后端少返回答案要标 `degraded` 并写明缺哪些,不静默丢。**

    静默丢的后果不是「这次判定没了」,是**这次判定看起来还在** ——
    日志上 `degraded: false`、`warnings: []`,和一次正常判定完全一样。
    """
    url, srv = _serve({"answers": {"a": {"noul": 0.9}}})
    try:
        resp = HttpJevClient(url, "k", timeout_s=5).decide(
            _req({"a": {"kind": "noul"}, "b": {"kind": "noul"}})
        )
    finally:
        srv.shutdown()

    assert set(resp.answers) == {"a"}
    assert resp.degraded is True
    assert resp.missing == ["b"]
    assert any("b" in w for w in resp.warnings), "缺了谁要说出来"


def test_a_malformed_answer_degrades_over_real_http() -> None:
    """★★ 上一条的端到端版:**只有标签没有值**走过真的 HTTP 之后,
    响应上必须是 `degraded` + `dropped` —— 而不是一个伪造的 `noul: 0`。"""
    url, srv = _serve({"answers": {"a": {"type": "noul"}}})
    try:
        resp = HttpJevClient(url, "k", timeout_s=5).decide(_req({"a": {"kind": "noul"}}))
    finally:
        srv.shutdown()

    assert resp.answers == {}
    assert resp.degraded is True and resp.dropped == ["a"] and resp.missing == ["a"]


def test_a_healthy_backend_is_not_marked_degraded() -> None:
    """反向断言 —— 全须全尾的响应不能被标成降级,否则 `degraded` 这个词会失去含义。"""
    url, srv = _serve({"answers": {"a": {"noul": 0.9}}, "model": "jev-test",
                       "usage": {"input_tokens": 12, "output_tokens": 3}})
    try:
        resp = HttpJevClient(url, "k", timeout_s=5).decide(_req({"a": {"kind": "noul"}}))
    finally:
        srv.shutdown()

    assert resp.degraded is False and resp.warnings == []
    assert (resp.input_tokens, resp.output_tokens) == (12, 3)
    assert resp.model == "jev-test"


def test_the_two_timing_halves_are_recorded_separately() -> None:
    """★★ 实测同一件事 **handshake 254ms vs compute 78ms**。

    合成一个 `latency` 就归因不了 —— 而「墙钟为什么慢」那个结论
    整个建立在它们分不分得开上（§8.11:2 倍的算力优势被 12 次往返吃掉）。
    """
    url, srv = _serve({"answers": {"a": {"noul": 0.9}}})
    try:
        resp = HttpJevClient(url, "k", timeout_s=5).decide(_req({"a": {"kind": "noul"}}))
    finally:
        srv.shutdown()

    assert resp.handshake_ms > 0 and resp.compute_ms >= 0
    assert resp.latency_ms >= resp.handshake_ms + resp.compute_ms - 1.0, "两段加起来不该超过总时长"


# ═══════════════════════════════════════════════════════════
# §8.6 Mock 只给保守答案 / §8.10 降级链每级都报
# ═══════════════════════════════════════════════════════════


def test_mock_is_conservative_and_says_so() -> None:
    """★ §8.6:**猜得越像,越容易让人误以为判定是对的。**

    一律 0.5 —— 让 policy 的置信度门限**自己**走到 `escalate`。
    这恰好演示了「不确定就别猜」。
    """
    resp = MockClient().decide(_req({
        "a": {"kind": "noul"},
        "b": {"kind": "choice", "options": {"x": 0.5, "y": 0.5}},
        "c": {"kind": "score"},
    }))
    assert resp.answers["a"].noul == 0.5
    assert resp.answers["b"].top() == pytest.approx(0.5), "4 个选项时也是均分"
    assert resp.answers["c"].score == 0
    assert resp.degraded and resp.warnings, "mock 的答案不能看起来像真判定"


def test_mock_splits_probability_across_all_options() -> None:
    """★ 选项越多,选中项概率越低 —— 这正是 `topGte` 与选项个数无关的体现（§8.3）。"""
    resp = MockClient().decide(_req({"q": {"kind": "choice", "options": {c: 0.25 for c in "abcd"}}}))
    a = resp.answers["q"]
    assert len(a.probabilities) == 4
    assert a.top() == pytest.approx(0.25)
    assert sum(a.probabilities.values()) == pytest.approx(1.0)


def test_fallback_reports_the_degradation_and_keeps_the_planned_provider_name() -> None:
    """★★ §8.10:**降级发生了要报,而且 provider 记的是「原计划用的那个」。**

    覆盖成实际用的那个,日志上就看不出发生过降级 —— 那正是「不假装成功」要防的。
    """

    class Broken:
        name = "broken"

        def decide(self, request: DecideRequest) -> DecideResponse:
            raise ConnectionError("connection refused")

    fb = FallbackClient([Broken(), MockClient()])
    resp = fb.decide(_req({"a": {"kind": "noul"}}))

    assert resp.provider == "broken", "记原计划的,不覆盖"
    assert any("降级" in w for w in resp.warnings)
    assert fb.degradations and "ConnectionError" in fb.degradations[0]["error"]


def test_fallback_with_every_backend_down_raises_rather_than_inventing_an_answer() -> None:
    """★ 全挂时**抛**,不返回一个编出来的答案。

    返回保守答案看着更「健壮」,但那会让「后端全挂了」和「判定完成了」
    在调用方看来一样 —— 而这两件事要采取的行动完全不同。
    """

    class Broken:
        def __init__(self, name: str) -> None:
            self.name = name

        def decide(self, request: DecideRequest) -> DecideResponse:
            raise TimeoutError("timed out")

    with pytest.raises(RuntimeError, match="全部不可用"):
        FallbackClient([Broken("a"), Broken("b")]).decide(_req({"q": {"kind": "noul"}}))


def test_a_fallback_chain_cannot_be_empty() -> None:
    """空链是个配置错误 —— 让它当场炸,别等到第一次判定。"""
    with pytest.raises(ValueError, match="不能是空"):
        FallbackClient([])


# ═══════════════════════════════════════════════════════════
# ★★★ `top()` 对 `noul` 恒为 0 —— 又一个「伪造的 0」
# ═══════════════════════════════════════════════════════════


def test_top_of_a_noul_answer_is_the_probability_not_zero() -> None:
    """★★★ `noul` 答案**没有概率表** —— 而 `top()` 第一版只会从概率表里取。

    于是它 **恒为 0.0**,后果不是「少了一个数」,是**所有 `noul` 门限永远判否**:
    「不需要工具」那一支永远走不到,而日志上每一行都长得像一次正常判定。

    ★ 这和本文件开头警告过的「伪造的 `0`」是**同一个东西,从另一扇门进来**:
    那边是**值缺失**被补成 0,这边是**值在,但没人读它**。
    """
    answers, _, _ = parse_answers({"q": {"noul": 0.87}}, ["q"])
    assert answers["q"].top() == pytest.approx(0.87)
    assert answers["q"].confidence == pytest.approx(0.87), "confidence 留 0 也是假的"


def test_top_still_prefers_the_probability_table_for_choice() -> None:
    """★ 修 `noul` 不能把 `choice` 弄坏 —— 那边 `top()` 仍然是**概率表的最大值**,
    而不是 `confidence`（§8.3:后者是归一化香农熵）。"""
    answers, _, _ = parse_answers(
        {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2},
               "confidence": 0.269}}, ["q"])
    assert answers["q"].top() == pytest.approx(0.8)


def test_a_noul_threshold_actually_fires_end_to_end() -> None:
    """★ 上一条的**后果**版:0.87 要能过 0.5 的门限。

    只测 `top()` 得到 0.87 是不够的 —— 这一条测的是「策略真的会走那一支」。
    """
    from experiments.core.deciding import DecideRequest

    url, srv = _serve({"answers": {"needs": {"noul": 0.87}}})
    try:
        resp = HttpJevClient(url, "k", timeout_s=5).decide(
            _req({"needs": {"kind": "noul"}}))
    finally:
        srv.shutdown()

    assert resp.answers["needs"].top() >= 0.5, "过不了门限的话这条路永远是死的"
