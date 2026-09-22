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
    """一条对话消息。

    ★★ `reasoning_content` **不是可选的装饰,漏了会 400。**

    DeepSeek 官方文档（思考模式 · 工具调用）原话:携带了 `tools` 参数的请求,
    在后续**所有**请求中**必须完整回传 `reasoning_content`** ——
    **即使该轮模型未实际进行工具调用**;未正确回传则返回 400。

    也就是说:**任何多轮 + 带工具的实现（ReAct 就是）在思考模型上都会因此挂掉**,
    而报错发生在第二轮,看起来像「工具定义有问题」。

    反过来,不带 `tools` 时该字段会被忽略、不拼进上下文 —— 所以它只在这一种
    组合下是必需的,而那恰好是我们最常跑的那一种。
    """

    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str | None = None


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
    # 这一轮的思维链原文。**下一轮要原样带回去**,见 `Message.reasoning_content`
    reasoning_content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ms(self) -> float:
        return self.handshake_ms + self.ttft_ms + self.after_ttft_ms

    def as_assistant_message(self) -> Message:
        """把这一轮的回答变成可以 append 回历史的 assistant 消息。

        ★ **用它,不要手拼。** 手拼最常见的就是漏掉 `reasoning_content`,
        而那个错误要等到下一轮才以 400 的形式出现。
        """
        return Message(
            role="assistant",
            content=self.text,
            reasoning_content=self.reasoning_content,
            tool_calls=self.tool_calls,
        )


@dataclass(frozen=True)
class Thinking:
    """模型原生思考控制。**这是「思考程度」那个轴真正的旋钮。**

    ⚠️ 两条实测出来的坑,写在这里免得每次都踩:

    1. **`medium` 不是一个档位。** DeepSeek 官方映射:
       `minimal→low · low→low · medium→high · high→high · xhigh→high · max→max`。
       照 {低,中,高} 跑,`中` 和 `高` 是**同一次运行** —— 表里两行一样,而人会以为测了两档。
       **用厂商真实档位,并把映射写进结果。**
    2. **思考模式静默忽略 `temperature`。** 官方原话「不会报错,但也不会生效」;
       `top_p` 反过来只在思考模式下生效（0.95–1.0）。
       所以「所有 arm 温度相同」在「一臂开思考、一臂不开」时**做不到** ——
       这是**混淆变量,要记进结果**,不能假设掉。
    """

    enabled: bool = True
    effort: str = "high"


class ModelClient(Protocol):
    """名字 + 身份 + 一次补全。"""

    name: str
    model_id: str
    model_version: str
    provider: str

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        thinking: "Thinking | None" = None,
    ) -> ModelReply:
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

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        thinking: "Thinking | None" = None,
    ) -> ModelReply:
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


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """把 `Message` 变成请求体里的 `messages`。

    ★ **抽成纯函数是为了能测。** 这一处最容易犯的错是漏掉 `reasoning_content`
    —— 而它的症状是**下一轮**才出现的 400，隔着网络根本测不到。
    这里不碰网络，于是「历史里有没有把思维链带回去」可以直接断言。
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.reasoning_content is not None:
            item["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            item["tool_calls"] = list(m.tool_calls)
        if m.tool_call_id is not None:
            item["tool_call_id"] = m.tool_call_id
        out.append(item)
    return out


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

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        thinking: Thinking | None = None,
    ) -> ModelReply:
        """一次补全。

        ★ `tools` 一旦不是 `None`，`reasoning_content` 就成了**必需回传**的字段
        （见 `Message`），而 `temperature` 在思考模式下**会被静默忽略**。
        两件事都不报错地发生在服务端，所以只能在这一侧记清楚。
        """
        payload_messages = to_wire_messages(messages)

        body_fields: dict[str, Any] = {
            "model": self.model_id,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools is not None:
            body_fields["tools"] = tools
        if thinking is not None:
            # DeepSeek 的 OpenAI 兼容格式：开关在 extra_body，强度是一等参数
            body_fields["reasoning_effort"] = thinking.effort
            body_fields["thinking"] = {"type": "enabled" if thinking.enabled else "disabled"}

        body = json.dumps(body_fields).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and tools is not None:
                # ★ 把「400」翻译成它真正的原因。不然这个错看起来像「工具定义写错了」，
                #   而实际原因是历史里少了 reasoning_content。
                raise urllib.error.HTTPError(
                    exc.url, exc.code,
                    f"{exc.reason} —— 带 tools 的请求要求历史里每一轮都回传 "
                    f"`reasoning_content`（用 ModelReply.as_assistant_message() 拼历史，别手拼）",
                    exc.headers, None,
                ) from exc
            raise
        elapsed = (time.perf_counter() - t0) * 1000

        # 非流式：拿不到真正的 TTFT，整段都记在 after_ttft。
        # **不假装知道 TTFT** —— 想拆开就得走 SSE，那是另一件事。
        usage = payload.get("usage") or {}
        self.last_usage_reported = bool(usage)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        return ModelReply(
            text=message.get("content") or "",
            reasoning_content=message.get("reasoning_content"),
            tool_calls=tuple(message.get("tool_calls") or ()),
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

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        thinking: "Thinking | None" = None,
    ) -> ModelReply:
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


__all__ = ["Message", "ModelReply", "Thinking", "ModelClient", "ScriptedModel",
           "CallableModel", "OpenAICompatModel", "describe", "to_wire_messages", "error_types"]

# 给调用方 catch 用 —— 别让 URL 层的异常类型泄漏到 baseline 里
error_types = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError)
