"""判定客户端 —— 把问题发给判定模型,把答案收回来。

## 线协议（照抄 TS 侧 `src/provider-http.ts`）

```
POST <base_url>/v1/systemone
  请求: {model, state, questions}
  响应: {answers, model, usage?, warnings?, dropped?}
```

**一次请求拿到全部问题的答案** —— 这是判定模型的性质（一次前向对所有问题并行打分）,
不是优化。所以「一次请求判多路」在这里是默认形态,而**按批计时**（`core/events.py`
的 `DecisionBatchEvent`）就是为它准备的。

## ★★ 从 TS 侧搬过来的一条教训:**认值,不认标签**

以前三个分支的写法是 `a.type === 'noul' || typeof a.noul === 'number'` ——
一个**只有标签、没有值**的畸形答案（`{"type":"noul"}`）会走到
`Number(undefined) || 0`,被**补成 `noul: 0`**。

它既不进 `dropped` 也不进 `missing`,于是 `degraded` 保持 `false` ——
**在日志上和一次正常判定完全一样**。

而 `0` 在策略里是一个**明确的否定**（「不需要工具」「没成功」）,
**比整个后端挂掉危险得多**:挂掉会被降级链抓到,伪造的 `0` 不会。

**容忍的是缺标签（`{noul: 0.7}` 照收）,不是缺值。**

## 分层

`Answer` / `parse_answers` 是纯函数（**抽出来是为了能测** —— 隔着网络的畸形响应测不到）;
三个客户端实现同一个 `DecisionClient` 协议,由 `FallbackClient` 串成降级链,
**每降一级都报一次**（§8.10 不假装成功）。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# ═══════════════════════════════════════════════════════════
# 词汇
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Answer:
    """一个判定答案。三种形状,和 `DECISION.md` 的 `kind` 一一对应。

    ★ `probabilities` 是**每个选项的概率**;`confidence` 对 `choice` 而言
    是**选中项的概率**（不是熵！见 §8.3 —— 卡阈值要用这个,不是 `confidence` 那个名字
    在原始 API 里指的东西）。
    """

    kind: str                      # noul | choice | score
    noul: float = 0.0
    choice: str = ""
    score: int = 0
    probabilities: dict[str, float] = field(default_factory=dict)
    legend: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0

    def top(self) -> float:
        """**卡阈值要用的那个数**（§8.3）。

        三种形状各有各的「选中项概率」:

        - `noul` → **`noul` 自己**（P(true)）
        - `choice` / `score` → **概率表里的最大值**

        ★★ 第一版这里只写了「概率表为空就退回 `confidence`」——
        而 `noul` 答案**没有概率表**,`confidence` 也没设,于是 `top()` **恒为 0.0**。

        后果不是「少了一个数」,是**所有 `noul` 门限永远判否**:
        「不需要工具」这一支永远走不到,而日志上每一行都长得像一次正常判定。
        这和在 `parse_answers` 里警告过的「伪造的 `0`」是**同一个东西**,
        只是从另一扇门进来的 —— 那边是值缺失被补成 0,这边是值在、但没人读它。
        """
        if self.kind == "noul":
            return self.noul
        if self.probabilities:
            return max(self.probabilities.values())
        return self.confidence


@dataclass
class DecideRequest:
    state: dict[str, Any]
    questions: dict[str, Any]
    model: str | None = None


@dataclass
class DecideResponse:
    answers: dict[str, Answer]
    provider: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    handshake_ms: float = 0.0
    compute_ms: float = 0.0
    # ★ **缺答案不是异常,是一种要报出来的状态** —— 见模块头那条教训
    degraded: bool = False
    dropped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DecisionClient(Protocol):
    name: str

    def decide(self, request: DecideRequest) -> DecideResponse:
        ...


# ═══════════════════════════════════════════════════════════
# 解析 —— 纯函数,所以它可测
# ═══════════════════════════════════════════════════════════


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_map(v: Any) -> dict[str, float]:
    if not isinstance(v, dict):
        return {}
    return {str(k): float(x) for k, x in v.items() if _is_num(x)}


def parse_answers(raw: Any, asked: Sequence[str]) -> tuple[dict[str, Answer], list[str], list[str]]:
    """把后端的 `answers` 解析成 `Answer`。返回 `(答案, 丢掉的, 缺的)`。

    ★★ **认值不认标签。** 只有标签没有值的畸形答案必须**丢掉**,
    不能被补成 `0` —— 那是 §8.10「不假装成功」在解析层的漏洞,而且
    `0` 在策略里是明确的否定,比后端挂掉危险得多（挂掉会被降级链抓到,伪造的 0 不会）。
    """
    answers: dict[str, Answer] = {}
    dropped: list[str] = []
    if not isinstance(raw, dict):
        # ★ **一个字节都没回来** → 全缺。不是 `dropped`
        #   （那个词的含义是「后端为这个问题返回了畸形的东西」,
        #   两者要采取的行动不同:一个是降级,一个是查协议)。
        return answers, [], list(asked)

    asked_set = set(asked)
    for qid, a in raw.items():
        if str(qid) not in asked_set:
            # ★ **没问过的答案不进答案集。** 后端多答一个不是错误,
            #   但留着它 policy 就能读到一个没有声明过的判定 ——
            #   而问题清单是**我们**生成的那一份,它是权威。
            continue
        if not isinstance(a, dict):
            dropped.append(str(qid))
            continue

        # noul：容忍缺标签,不容忍缺值
        if a.get("type") == "noul" or a.get("noul") is not None:
            if not _is_num(a.get("noul")):
                dropped.append(str(qid))
                continue
            answers[str(qid)] = Answer(kind="noul", noul=min(1.0, max(0.0, float(a["noul"]))),
                                       # ★ `confidence` 也填上 —— 对 `noul` 而言
                                       #   「置信度」就是 P(true)。留 0 的话,
                                       #   任何读 `confidence` 的消费方都会拿到一个假的 0。
                                       confidence=min(1.0, max(0.0, float(a["noul"]))))
            continue

        # score：序数
        if a.get("type") == "score" or a.get("score") is not None:
            if not _is_num(a.get("score")):
                dropped.append(str(qid))
                continue
            p = _num_map(a.get("probabilities"))
            answers[str(qid)] = Answer(
                kind="score", score=int(a["score"]), probabilities=p,
                legend={str(k): str(v) for k, v in (a.get("legend") or {}).items()},
                confidence=float(a["confidence"]) if _is_num(a.get("confidence"))
                else (max(p.values()) if p else 0.0),
            )
            continue

        # choice：空字符串**不是一个选项**
        if a.get("type") == "choice" or a.get("choice") is not None:
            choice = a.get("choice")
            if not isinstance(choice, str) or choice == "":
                # ★ 放过去会让 `pickInput` 返回 `''`,上游拿它去拼路径,
                #   错误要隔好几层才暴露出来
                dropped.append(str(qid))
                continue
            p = _num_map(a.get("probabilities"))
            answers[str(qid)] = Answer(
                kind="choice", choice=choice, probabilities=p,
                confidence=float(a["confidence"]) if _is_num(a.get("confidence"))
                else (max(p.values()) if p else 0.0),
            )
            continue

        dropped.append(str(qid))

    missing = [q for q in asked if q not in answers]
    return answers, dropped, missing


# ═══════════════════════════════════════════════════════════
# 三个实现 —— 由 `FallbackClient` 串成降级链
# ═══════════════════════════════════════════════════════════


class MockClient:
    """**永远给最保守的答案,不做启发式**（§8.6）。

    一律 `0.5` / 第一个选项 / 最低分 —— 让 policy 的置信度门限**自己**走到
    `escalate`。猜得越像,越容易让人误以为判定是对的。
    """

    name = "mock"

    def decide(self, request: DecideRequest) -> DecideResponse:
        out: dict[str, Answer] = {}
        for qid, q in request.questions.items():
            kind = (q or {}).get("kind", "noul") if isinstance(q, dict) else "noul"
            if kind == "choice":
                opts = list(((q or {}).get("options") or {}).keys()) if isinstance(q, dict) else []
                # ★ 0.5 平均分给所有选项 —— 「完全不确定」的样子
                prob = 1.0 / len(opts) if opts else 0.0
                out[qid] = Answer(kind="choice", choice=opts[0] if opts else "",
                                  probabilities={o: prob for o in opts}, confidence=prob)
            elif kind == "score":
                out[qid] = Answer(kind="score", score=0, confidence=0.5)
            else:
                out[qid] = Answer(kind="noul", noul=0.5)
        return DecideResponse(answers=out, provider=self.name,
                              degraded=True, warnings=["mock：保守答案，不代表任何判定"])


class HttpJevClient:
    """`POST <base_url>/v1/systemone`。**只用标准库。**

    ★ 时间拆两段:`handshake_ms`（连上）和 `compute_ms`（拿到答案）。
    实测同一件事这两段是 **254ms vs 78ms** —— 合成一个 latency 就归因不了,
    而「墙钟为什么慢」那个结论整个建立在它们分不分得开上（`AGENTS.md` §8.11）。
    """

    def __init__(self, base_url: str, api_key: str, *, model: str = "jev-1.13",
                 timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"http:{model}"

    def decide(self, request: DecideRequest) -> DecideResponse:
        body = json.dumps({
            "model": request.model or self.model,
            "state": request.state,
            "questions": request.questions,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            t_connected = time.perf_counter()
            payload = json.loads(resp.read().decode())
        t_done = time.perf_counter()

        asked = list(request.questions.keys())
        answers, dropped, missing = parse_answers(payload.get("answers"), asked)
        warnings = [str(w) for w in (payload.get("warnings") or [])]
        if missing:
            # ★ **少返回答案要说出来**,不静默丢（§8.10）
            warnings.append(f"后端未返回 {len(missing)}/{len(asked)} 个答案：{', '.join(missing)}")

        usage = payload.get("usage") or {}
        return DecideResponse(
            answers=answers,
            provider=self.name,
            model=payload.get("model"),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            handshake_ms=(t_connected - t0) * 1000,
            compute_ms=(t_done - t_connected) * 1000,
            latency_ms=(t_done - t0) * 1000,
            degraded=bool(payload.get("degraded")) or bool(dropped) or bool(missing),
            dropped=dropped,
            missing=missing,
            warnings=warnings,
        )


class FallbackClient:
    """按顺序试,**每降一级都报一次**（§8.10）。

    ★ 和 TS 侧 `provider-fallback.ts` 同一条规矩:**记的 provider 是「原计划用的那个」,
    不覆盖成实际用的那个** —— 否则日志上看不出发生过降级。
    """

    def __init__(self, clients: Sequence[DecisionClient]) -> None:
        if not clients:
            raise ValueError("降级链不能是空的")
        self.clients = list(clients)
        self.name = "→".join(c.name for c in self.clients)
        self.degradations: list[dict[str, str]] = []

    def decide(self, request: DecideRequest) -> DecideResponse:
        first = self.clients[0].name
        for client in self.clients:
            try:
                resp = client.decide(request)
            except Exception as exc:  # noqa: BLE001 —— 任何一层挂了都该往下走
                self.degradations.append({"from": client.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if client is not self.clients[0]:
                # ★ 降级发生了 —— 记进 warnings,而且**provider 保留原计划那个**
                resp.warnings.append(
                    f"降级：原计划用 {first}，实际用 {client.name}"
                )
            resp.provider = first
            return resp
        raise RuntimeError(
            f"判定后端全部不可用：{[c.name for c in self.clients]}；"
            f"降级经过 {self.degradations}"
        )


__all__ = [
    "Answer", "DecideRequest", "DecideResponse", "DecisionClient",
    "HttpJevClient", "MockClient", "FallbackClient", "parse_answers",
]
