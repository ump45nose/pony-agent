"""Async message/tool loop for Pony Agent."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from typing import Any

from pony_agent.core.context import ContextBudgetExceeded
from pony_agent.core.ports import ContextPolicy, ProviderAdapter, SessionStore, ToolRuntime
from pony_agent.core.types import (
    JsonDict,
    KernelConfig,
    KernelEvent,
    KernelEventKind,
    ProviderRequest,
    ProviderResponse,
    SessionSnapshot,
    ToolCall,
    new_id,
    tool_call_to_message,
)


class UnsupportedProviderError(RuntimeError):
    pass


class AgentKernel:
    def __init__(
        self,
        *,
        provider: ProviderAdapter,
        tools: ToolRuntime,
        store: SessionStore,
        context: ContextPolicy,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.store = store
        self.context = context

    def open_session(
        self,
        config: KernelConfig,
        *,
        session_id: str | None = None,
    ) -> "KernelSession":
        return KernelSession(
            kernel=self,
            config=config,
            session_id=session_id or new_id("session"),
        )


class KernelSession:
    def __init__(self, *, kernel: AgentKernel, config: KernelConfig, session_id: str) -> None:
        self.kernel = kernel
        self.config = config
        self.session_id = session_id
        self._inputs: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(config.queue_size)
        self._steering: asyncio.Queue[Any] = asyncio.Queue(config.queue_size)
        self._events: asyncio.Queue[KernelEvent] = asyncio.Queue(config.queue_size * 8)
        self._cancel_requested = asyncio.Event()
        self._closed = False
        self._runner: asyncio.Task[None] | None = None
        self._started = False
        self._messages: list[JsonDict] = []
        self._run_id: str | None = None
        self._run_api_calls = 0

    async def submit(self, message: Any) -> None:
        self._ensure_runner()
        await self._inputs.put(("submit", message))

    async def follow_up(self, message: Any) -> None:
        self._ensure_runner()
        await self._inputs.put(("follow_up", message))

    async def steer(self, message: Any) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        await self._steering.put(message)

    async def cancel(self) -> None:
        self._cancel_requested.set()
        await asyncio.gather(
            self.kernel.provider.cancel(),
            self.kernel.tools.cancel(),
            return_exceptions=True,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runner:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
        if self._started:
            await self._emit(KernelEventKind.SESSION_CLOSED, {})

    async def events(self):
        self._ensure_runner()
        while True:
            event = await self._events.get()
            yield event
            if event.kind == KernelEventKind.SESSION_CLOSED:
                return

    def _ensure_runner(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run(), name=f"pony:{self.session_id}")

    async def _emit(
        self,
        kind: str | KernelEventKind,
        payload: JsonDict,
        *,
        durable: bool = True,
    ) -> KernelEvent:
        event = KernelEvent(self.session_id, str(kind), payload)
        if durable:
            event = await self.kernel.store.append(event)
        await self._events.put(event)
        return event

    async def _run(self) -> None:
        try:
            snapshot = await self.kernel.store.rebuild(self.session_id)
            self._messages = list(snapshot.messages)
            if snapshot.version > 0:
                self._started = True
            if self.config.system_prompt and (
                not self._messages or self._messages[0].get("role") != "system"
            ):
                self._messages.insert(0, {"role": "system", "content": self.config.system_prompt})
            if not self._started:
                self._started = True
                await self._emit(
                    KernelEventKind.SESSION_STARTED,
                    {"model": self.config.model, "metadata": self.config.metadata},
                )
            while not self._closed:
                origin, message = await self._inputs.get()
                self._cancel_requested.clear()
                await self._run_turn(message, origin=origin)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # keep the event stream usable on runner defects
            await self._emit(
                KernelEventKind.RUN_FAILED,
                {"run_id": self._run_id, "error": str(exc), "type": type(exc).__name__},
            )

    async def _run_turn(self, message: Any, *, origin: str) -> None:
        self._run_id = new_id("run")
        self._run_api_calls = 0
        await self._emit(
            KernelEventKind.RUN_STARTED,
            {"run_id": self._run_id, "origin": origin, "provider": self.kernel.provider.name},
        )
        user = {"id": new_id("msg"), "role": "user", "content": message, "origin": origin}
        self._messages.append(user)
        await self._emit(KernelEventKind.MESSAGE_USER, deepcopy(user))

        usage_total: JsonDict = {}
        try:
            for iteration in range(1, self.config.max_iterations + 1):
                if self._cancel_requested.is_set():
                    await self._cancel_with_partial("")
                    return
                await self._drain_steering()
                snapshot = SessionSnapshot(self.session_id, messages=self._messages)
                prepared = await self.kernel.context.prepare(
                    snapshot,
                    self.config.context_budget,
                    self.config.compression_threshold,
                )
                if prepared.compacted:
                    self._messages = prepared.messages
                    await self._emit(
                        KernelEventKind.CONTEXT_COMPACTED,
                        {**prepared.metadata, "messages": deepcopy(prepared.messages)},
                    )

                tool_defs = await self.kernel.tools.definitions(
                    {"session_id": self.session_id, "run_id": self._run_id}
                )
                response = await self._call_provider(tool_defs, iteration)
                if response.usage:
                    for key, value in response.usage.items():
                        if isinstance(value, (int, float)):
                            usage_total[key] = usage_total.get(key, 0) + value
                    await self._emit(KernelEventKind.USAGE, dict(response.usage))

                assistant: JsonDict = {
                    "id": new_id("msg"),
                    "role": "assistant",
                    "content": response.content,
                    "reasoning": response.reasoning or None,
                    "tool_calls": [tool_call_to_message(call) for call in response.tool_calls],
                    "finish_reason": response.finish_reason,
                    "provider_state": response.provider_state,
                }
                if response.provider_state:
                    for key in (
                        "reasoning_content",
                        "reasoning_details",
                        "anthropic_content_blocks",
                        "codex_reasoning_items",
                        "codex_message_items",
                    ):
                        if key in response.provider_state:
                            assistant[key] = deepcopy(response.provider_state[key])
                self._messages.append(assistant)
                await self._emit(KernelEventKind.MESSAGE_ASSISTANT, deepcopy(assistant))

                if not response.tool_calls:
                    await self._emit(
                        KernelEventKind.RUN_COMPLETED,
                        {
                            "run_id": self._run_id,
                            "final_response": response.content,
                            "iterations": iteration,
                            "api_calls": self._run_api_calls,
                            "usage": usage_total,
                            "provider": self.kernel.provider.name,
                            "model": getattr(
                                self.kernel.provider, "model", self.config.model
                            ),
                        },
                    )
                    return

                for call in response.tool_calls:
                    await self._emit(
                        KernelEventKind.TOOL_REQUESTED,
                        {"call_id": call.id, "name": call.name, "arguments": call.arguments},
                    )
                outcomes = await self.kernel.tools.execute_batch(
                    response.tool_calls,
                    {
                        "session_id": self.session_id,
                        "run_id": self._run_id,
                        "iteration": iteration,
                        "messages": self._messages,
                    },
                    self._emit_tool_event,
                )
                for outcome in outcomes:
                    tool_message = {
                        "id": new_id("msg"),
                        "role": "tool",
                        "name": outcome.name,
                        "tool_name": outcome.name,
                        "tool_call_id": outcome.call_id,
                        "content": outcome.model_content,
                        "_tool_receipt": outcome.receipt,
                        "effect_disposition": outcome.effect,
                    }
                    self._messages.append(tool_message)
                    kind = (
                        KernelEventKind.TOOL_FAILED
                        if outcome.is_error
                        else KernelEventKind.TOOL_COMPLETED
                    )
                    await self._emit(
                        kind,
                        {
                            "call_id": outcome.call_id,
                            "name": outcome.name,
                            "model_content": outcome.model_content,
                            "ui_details": outcome.ui_details,
                            "receipt": outcome.receipt,
                            "effect": outcome.effect,
                            "is_error": outcome.is_error,
                            "message": tool_message,
                        },
                    )
            raise RuntimeError(f"maximum tool iterations exceeded ({self.config.max_iterations})")
        except (asyncio.CancelledError, InterruptedError):
            await self._cancel_with_partial("")
        except ContextBudgetExceeded as exc:
            await self._emit(
                KernelEventKind.RUN_FAILED,
                {"run_id": self._run_id, "error": str(exc), "type": "context_budget_exceeded"},
            )
        except BaseException as exc:
            await self._emit(
                KernelEventKind.RUN_FAILED,
                {"run_id": self._run_id, "error": str(exc), "type": type(exc).__name__},
            )

    async def _call_provider(self, tools: list[JsonDict], iteration: int) -> ProviderResponse:
        retries = 0
        while True:
            partial_text = ""
            partial_reasoning = ""
            request = ProviderRequest(
                session_id=self.session_id,
                messages=deepcopy(self._messages),
                tools=tools,
                model=getattr(self.kernel.provider, "model", self.config.model),
                metadata={"run_id": self._run_id, "iteration": iteration, "attempt": retries + 1},
            )
            attempted_model = request.model
            try:
                self._run_api_calls += 1
                async for event in self.kernel.provider.stream(request):
                    if self._cancel_requested.is_set():
                        await self.kernel.provider.cancel()
                        raise InterruptedError("run cancelled")
                    if event.kind == "text.delta" and event.delta:
                        partial_text += event.delta
                        await self._emit(
                            KernelEventKind.TEXT_DELTA,
                            {"run_id": self._run_id, "delta": event.delta},
                            durable=False,
                        )
                    elif event.kind == "reasoning.delta" and event.delta:
                        partial_reasoning += event.delta
                        await self._emit(
                            KernelEventKind.REASONING_DELTA,
                            {"run_id": self._run_id, "delta": event.delta},
                            durable=False,
                        )
                    elif event.kind == "retry":
                        await self._emit(KernelEventKind.PROVIDER_RETRY, event.payload)
                    elif event.kind == "response" and event.response is not None:
                        return event.response
                raise RuntimeError("provider stream ended without a final response")
            except InterruptedError:
                if partial_text or partial_reasoning:
                    await self._emit(
                        KernelEventKind.MESSAGE_PARTIAL,
                        {"run_id": self._run_id, "content": partial_text, "reasoning": partial_reasoning},
                    )
                raise
            except BaseException as exc:
                if self._cancel_requested.is_set():
                    if partial_text or partial_reasoning:
                        await self._emit(
                            KernelEventKind.MESSAGE_PARTIAL,
                            {
                                "run_id": self._run_id,
                                "content": partial_text,
                                "reasoning": partial_reasoning,
                            },
                        )
                    raise InterruptedError("run cancelled") from exc
                if retries < self.config.max_turn_retries:
                    retries += 1
                    await self._emit(
                        KernelEventKind.PROVIDER_RETRY,
                        {
                            "run_id": self._run_id,
                            "attempt": retries + 1,
                            "provider": self.kernel.provider.name,
                            "model": getattr(
                                self.kernel.provider, "model", self.config.model
                            ),
                            "error": str(exc),
                        },
                    )
                    continue
                if await self.kernel.provider.activate_fallback():
                    await self._emit(
                        KernelEventKind.MODEL_FALLBACK,
                        {
                            "run_id": self._run_id,
                            "provider": self.kernel.provider.name,
                            "from_model": attempted_model,
                            "to_model": getattr(
                                self.kernel.provider, "model", self.config.model
                            ),
                            "error": str(exc),
                        },
                    )
                    retries = 0
                    continue
                raise

    async def _emit_tool_event(self, kind: str, payload: JsonDict) -> None:
        mapping = {
            "tool.started": KernelEventKind.TOOL_STARTED,
            "tool.progress": KernelEventKind.TOOL_PROGRESS,
        }
        event_kind = mapping.get(kind, KernelEventKind.TOOL_PROGRESS)
        await self._emit(
            event_kind,
            payload,
            durable=event_kind == KernelEventKind.TOOL_STARTED,
        )

    async def _drain_steering(self) -> None:
        while not self._steering.empty():
            steer = self._steering.get_nowait()
            message = {
                "id": new_id("msg"),
                "role": "user",
                "content": steer,
                "origin": "steer",
            }
            self._messages.append(message)
            await self._emit(KernelEventKind.MESSAGE_USER, deepcopy(message))

    async def _cancel_with_partial(self, content: str) -> None:
        if content:
            await self._emit(
                KernelEventKind.MESSAGE_PARTIAL,
                {"run_id": self._run_id, "content": content},
            )
        await self._emit(KernelEventKind.RUN_CANCELLED, {"run_id": self._run_id})
