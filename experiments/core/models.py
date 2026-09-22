"""生成模型客户端。**只定义「一次对话补全」,不认识 agent。**

★ 契约里有两条是踩出来的,别简化掉:

- `usage` 拆成 **缓存命中 / 未命中 / 推理 / 可见** 四份。合成一个
  `output_tokens` 之后,「思考更久」和「话更多」就分不开了,而附录里
  要测的正是**思考程度**那个轴。
- `timing` 拆成 **握手 / 首 token 前 / 首 token 后**。握手是我们量到
  254ms 的那一段,TTFT 和输出长度解耦 —— 合成一个 latency 就归因不了。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class ModelReply:
    text: str
    input_tokens_cached: int = 0
    input_tokens_uncached: int = 0
    output_tokens_reasoning: int = 0
    output_tokens_visible: int = 0
    handshake_ms: float = 0.0
    ttft_ms: float = 0.0
    after_ttft_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ms(self) -> float:
        return self.handshake_ms + self.ttft_ms + self.after_ttft_ms


class ModelClient(Protocol):
    """名字 + 身份 + 一次补全。"""

    name: str
    model_id: str
    model_version: str
    provider: str

    def chat(self, messages: list[Message], *, max_tokens: int, temperature: float) -> ModelReply:
        ...


# ═══════════════════════════════════════════════════════════
# Mock —— 离线、确定性。**自检和单测靠它**
# ═══════════════════════════════════════════════════════════


class ScriptedModel:
    """按顺序吐预先写好的回答。**没有网络,输出完全确定。**

    它存在的理由是:接口的自检必须能在**没有 key、没有网**的机器上跑通。
    真正的 baseline 用下面的 `OpenAICompatModel`。
    """

    name = "scripted"
    model_id = "scripted"
    model_version = "0"
    provider = "none"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._i = 0

    def chat(self, messages: list[Message], *, max_tokens: int, temperature: float) -> ModelReply:
        text = self._replies[self._i] if self._i < len(self._replies) else ""
        self._i += 1
        return ModelReply(
            text=text,
            input_tokens_uncached=sum(len(m.content) // 4 for m in messages),
            output_tokens_visible=len(text) // 4,
            handshake_ms=0.0,
            ttft_ms=0.0,
            after_ttft_ms=0.0,
            raw={"scripted": True},
        )


# ═══════════════════════════════════════════════════════════
# HTTP —— OpenAI 兼容（DeepSeek / 本地 vLLM / 托管 Jev 都是这个形状）
# ═══════════════════════════════════════════════════════════


class OpenAICompatModel:
    """`POST {base_url}/chat/completions`。

    **只依赖标准库** —— 这样「装不上 openai 包」不会变成别人跑不起来的理由。
    每个字段缺失时的行为都写明了,不许静默补 0:
    usage 缺失就报 `usage_reported=False`,让人**看得见**它没报。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        model_version: str = "unpinned",
        provider: str = "openai-compat",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.model_id = model_id
        self.model_version = model_version
        self.provider = provider
        self.name = f"{provider}:{model_id}"
        self.timeout_s = timeout_s
        # 上一次调用有没有真的报 usage。**没报就说没报,不补默认值。**
        self.last_usage_reported = False

    def chat(self, messages: list[Message], *, max_tokens: int, temperature: float) -> ModelReply:
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode())
        elapsed = (time.perf_counter() - t0) * 1000

        # 非流式：拿不到真正的 TTFT，整段都记在 after_ttft。
        # **不假装知道 TTFT** —— 想拆开就得走 SSE，那是另一件事。
        usage = payload.get("usage") or {}
        self.last_usage_reported = bool(usage)
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""

        return ModelReply(
            text=text,
            input_tokens_cached=int(usage.get("prompt_cache_hit_tokens", 0) or 0),
            input_tokens_uncached=int(usage.get("prompt_tokens", 0) or 0)
            - int(usage.get("prompt_cache_hit_tokens", 0) or 0),
            output_tokens_reasoning=int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0)
            if isinstance(usage.get("completion_tokens_details"), dict)
            else 0,
            output_tokens_visible=int(usage.get("completion_tokens", 0) or 0),
            handshake_ms=0.0,
            ttft_ms=0.0,
            after_ttft_ms=elapsed,
            raw=payload,
        )


class CallableModel:
    """把任意 `responder(messages) -> str` 包成一个模型。

    ★ 它让**整条链路可以在离线机器上验完** —— 接口、记账、落盘、汇总,
    全都不需要 key 和网络。真模型接进来只是换一个 `ModelClient`。

    **不是给实验用的**:它吐什么完全由调用方决定,所以拿它跑出来的分数
    没有任何意义。谁在用它,谁就要在报告里说清楚。
    """

    name = "callable"
    provider = "local"

    def __init__(self, responder: "Callable[[list[Message]], str]", *, model_id: str = "callable") -> None:
        self._responder = responder
        self.model_id = model_id
        self.model_version = "n/a"

    def chat(self, messages: list[Message], *, max_tokens: int, temperature: float) -> ModelReply:
        text = self._responder(messages)
        return ModelReply(
            text=text,
            input_tokens_uncached=sum(len(m.content) // 4 for m in messages),
            output_tokens_visible=max(1, len(text) // 4),
            raw={"callable": True},
        )


def describe(model: ModelClient) -> dict[str, str]:
    """写进 `meta.json` 的三元组。**版本不许省。**"""
    return {"id": model.model_id, "version": model.model_version, "provider": model.provider}


__all__ = ["Message", "ModelReply", "ModelClient", "ScriptedModel", "CallableModel",
           "OpenAICompatModel", "describe", "error_types"]

# 给调用方 catch 用 —— 别让 URL 层的异常类型泄漏到 baseline 里
error_types = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError)
