"""Narrow dependency ports for the Pony kernel."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Protocol

from pony_agent.core.types import (
    ContextPreparation,
    JsonDict,
    KernelEvent,
    ProviderEvent,
    ProviderRequest,
    SessionSnapshot,
    ToolCall,
    ToolOutcome,
)


Emit = Callable[[str, JsonDict], Awaitable[None]]


class ProviderAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def api_mode(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]: ...

    async def cancel(self) -> None: ...

    async def activate_fallback(self) -> bool: ...


class ToolRuntime(Protocol):
    async def definitions(self, context: JsonDict) -> list[JsonDict]: ...

    async def execute_batch(
        self,
        calls: Sequence[ToolCall],
        context: JsonDict,
        emit: Emit,
    ) -> list[ToolOutcome]: ...

    async def cancel(self) -> None: ...


class SessionStore(Protocol):
    async def append(self, event: KernelEvent) -> KernelEvent: ...

    async def load(self, session_id: str) -> SessionSnapshot: ...

    async def rebuild(self, session_id: str) -> SessionSnapshot: ...

    async def close(self) -> None: ...


class ContextPolicy(Protocol):
    async def prepare(
        self,
        snapshot: SessionSnapshot,
        budget: int,
        threshold: float,
    ) -> ContextPreparation: ...
