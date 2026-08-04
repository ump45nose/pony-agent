"""Provider adapters backed by Hermes' battle-tested wire clients."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

from pony_agent.core.kernel import UnsupportedProviderError
from pony_agent.core.types import (
    JsonDict,
    ProviderEvent,
    ProviderRequest,
    ProviderResponse,
    ToolCall,
    new_id,
)


_END = object()


def _usage_dict(usage: Any) -> JsonDict:
    if usage is None:
        return {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
        "cache_read_tokens": ("cache_read_tokens", "cached_tokens"),
        "cache_write_tokens": ("cache_write_tokens", "cache_creation_input_tokens"),
        "reasoning_tokens": ("reasoning_tokens",),
    }
    result: JsonDict = {}
    for canonical, names in aliases.items():
        for name in names:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if isinstance(value, (int, float)):
                result[canonical] = int(value)
                break
    return result


class LegacyProviderAdapter:
    # One adapter instance may cross protocol families after an explicitly
    # configured fallback. The concrete subclasses describe the primary wire
    # path; validation therefore follows the agent's current api_mode.
    expected_modes: frozenset[str] = frozenset(
        {"chat_completions", "codex_responses", "anthropic_messages"}
    )

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._stream_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return str(getattr(self.agent, "provider", None) or "unknown")

    @property
    def api_mode(self) -> str:
        return str(getattr(self.agent, "api_mode", None) or "chat_completions")

    @property
    def model(self) -> str:
        return str(getattr(self.agent, "model", None) or "")

    def _validate(self) -> None:
        if self.api_mode not in self.expected_modes:
            raise UnsupportedProviderError(
                f"Pony kernel does not support api_mode={self.api_mode!r}; "
                "retry with --agent-core legacy"
            )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        self._validate()
        async with self._stream_lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Any] = asyncio.Queue()

            def enqueue(event: ProviderEvent) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            worker = asyncio.create_task(
                asyncio.to_thread(self._call, request, enqueue),
                name=f"pony-provider:{request.session_id}",
            )

            async def finish_worker() -> None:
                try:
                    response = await worker
                    await queue.put(ProviderEvent(kind="response", response=response))
                except BaseException as exc:
                    await queue.put(exc)
                finally:
                    await queue.put(_END)

            finisher = asyncio.create_task(finish_worker())
            try:
                while True:
                    item = await queue.get()
                    if item is _END:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                if not worker.done():
                    await self.cancel()
                await asyncio.gather(worker, finisher, return_exceptions=True)

    def _call(self, request: ProviderRequest, enqueue) -> ProviderResponse:
        previous = {
            "stream_delta_callback": getattr(self.agent, "stream_delta_callback", None),
            "reasoning_callback": getattr(self.agent, "reasoning_callback", None),
            "tool_gen_callback": getattr(self.agent, "tool_gen_callback", None),
            "tools": getattr(self.agent, "tools", None),
        }
        self.agent._interrupt_requested = False
        self.agent._interrupt_message = None
        self.agent._execution_thread_id = threading.current_thread().ident
        self.agent._interrupt_thread_signal_pending = False
        try:
            from run_agent import _set_interrupt

            _set_interrupt(False, self.agent._execution_thread_id)
        except Exception:
            pass
        self.agent.tools = deepcopy(request.tools)
        self.agent.stream_delta_callback = lambda delta: (
            enqueue(ProviderEvent(kind="text.delta", delta=str(delta)))
            if delta
            else None
        )
        self.agent.reasoning_callback = lambda delta: (
            enqueue(ProviderEvent(kind="reasoning.delta", delta=str(delta)))
            if delta
            else None
        )
        self.agent.tool_gen_callback = lambda name: enqueue(
            ProviderEvent(kind="tool.progress", payload={"name": str(name)})
        )
        try:
            api_kwargs = self.agent._build_api_kwargs(deepcopy(request.messages))
            if self.api_mode == "codex_responses":
                api_kwargs = self.agent._get_transport().preflight_kwargs(
                    api_kwargs,
                    allow_stream=False,
                    is_github_responses=self.agent._is_copilot_url(),
                )
            try:
                from hermes_cli.middleware import apply_llm_request_middleware

                api_kwargs = apply_llm_request_middleware(
                    api_kwargs,
                    session_id=request.session_id,
                    model=request.model,
                    provider=self.name,
                    base_url=str(getattr(self.agent, "base_url", "") or ""),
                    api_mode=self.api_mode,
                    api_call_count=int(request.metadata.get("iteration") or 1),
                    task_id=str(request.metadata.get("run_id") or ""),
                    turn_id=str(request.metadata.get("run_id") or ""),
                    api_request_id=new_id("request"),
                ).payload
            except Exception:
                pass
            response = self.agent._interruptible_streaming_api_call(api_kwargs)
            kwargs: JsonDict = {}
            if self.api_mode == "anthropic_messages":
                kwargs["strip_tool_prefix"] = bool(
                    getattr(self.agent, "_is_anthropic_oauth", False)
                )
            normalized = self.agent._get_transport().normalize_response(response, **kwargs)
            calls = [
                ToolCall(
                    id=str(call.id or new_id("call")),
                    name=str(call.name),
                    arguments=str(call.arguments),
                    provider_state=deepcopy(call.provider_data),
                )
                for call in (normalized.tool_calls or [])
            ]
            return ProviderResponse(
                content=str(normalized.content or ""),
                reasoning=str(normalized.reasoning or normalized.reasoning_content or ""),
                tool_calls=calls,
                finish_reason=str(normalized.finish_reason or "stop"),
                usage=_usage_dict(normalized.usage or getattr(response, "usage", None)),
                provider_state=deepcopy(normalized.provider_data),
            )
        finally:
            try:
                from run_agent import _set_interrupt

                _set_interrupt(False, self.agent._execution_thread_id)
            except Exception:
                pass
            for name, value in previous.items():
                setattr(self.agent, name, value)

    async def cancel(self) -> None:
        await asyncio.to_thread(self.agent.interrupt)

    async def activate_fallback(self) -> bool:
        activate = getattr(self.agent, "_try_activate_fallback", None)
        if not callable(activate):
            return False
        activated = bool(await asyncio.to_thread(activate))
        if activated:
            self.agent._interrupt_requested = False
            self.agent._interrupt_message = None
        return activated


class ChatCompletionsAdapter(LegacyProviderAdapter):
    pass


class ResponsesAdapter(LegacyProviderAdapter):
    pass


class AnthropicMessagesAdapter(LegacyProviderAdapter):
    pass


class GeminiAdapter(ChatCompletionsAdapter):
    """Native Gemini uses Hermes' chat-compatible transport plus thought signatures."""


def provider_for_agent(agent: Any) -> LegacyProviderAdapter:
    mode = str(getattr(agent, "api_mode", "") or "chat_completions")
    if mode in {"bedrock_converse", "codex_app_server"}:
        raise UnsupportedProviderError(
            f"Pony kernel does not support api_mode={mode!r}; retry with --agent-core legacy"
        )
    if mode == "anthropic_messages":
        return AnthropicMessagesAdapter(agent)
    if mode == "codex_responses":
        return ResponsesAdapter(agent)
    if mode != "chat_completions":
        raise UnsupportedProviderError(
            f"Pony kernel does not support api_mode={mode!r}; retry with --agent-core legacy"
        )
    try:
        from agent.gemini_native_adapter import is_native_gemini_base_url

        if is_native_gemini_base_url(str(getattr(agent, "base_url", "") or "")):
            return GeminiAdapter(agent)
    except Exception:
        pass
    return ChatCompletionsAdapter(agent)
