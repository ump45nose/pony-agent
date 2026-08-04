"""Provider-independent values used by the Pony agent kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


JsonDict = dict[str, Any]


class KernelEventKind(StrEnum):
    SESSION_STARTED = "session.started"
    SESSION_CLOSED = "session.closed"
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    MESSAGE_USER = "message.user"
    MESSAGE_ASSISTANT = "message.assistant"
    MESSAGE_PARTIAL = "message.partial"
    TEXT_DELTA = "text.delta"
    REASONING_DELTA = "reasoning.delta"
    PROVIDER_RETRY = "provider.retry"
    MODEL_FALLBACK = "model.fallback"
    USAGE = "usage"
    TOOL_REQUESTED = "tool.requested"
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    CONTEXT_COMPACTED = "context.compacted"


@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    provider_state: JsonDict | None = None


@dataclass(slots=True)
class ToolOutcome:
    call_id: str
    name: str
    model_content: Any
    ui_details: Any = None
    receipt: JsonDict | None = None
    effect: str | None = None
    is_error: bool = False


@dataclass(slots=True)
class ProviderRequest:
    session_id: str
    messages: list[JsonDict]
    tools: list[JsonDict]
    model: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ProviderResponse:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: JsonDict = field(default_factory=dict)
    provider_state: JsonDict | None = None


@dataclass(slots=True)
class ProviderEvent:
    kind: str
    delta: str | None = None
    response: ProviderResponse | None = None
    payload: JsonDict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class KernelEvent:
    session_id: str
    kind: str
    payload: JsonDict
    seq: int | None = None


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    status: str = "new"
    version: int = 0
    messages: list[JsonDict] = field(default_factory=list)
    events: list[KernelEvent] = field(default_factory=list)


@dataclass(slots=True)
class ContextPreparation:
    messages: list[JsonDict]
    compacted: bool = False
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class KernelConfig:
    model: str
    system_prompt: str = ""
    context_budget: int = 200_000
    compression_threshold: float = 0.50
    max_iterations: int = 90
    max_turn_retries: int = 1
    queue_size: int = 32
    metadata: JsonDict = field(default_factory=dict)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def tool_call_to_message(call: ToolCall) -> JsonDict:
    message: JsonDict = {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }
    if call.provider_state:
        message["provider_state"] = call.provider_state
        extra = call.provider_state.get("extra_content")
        if extra is not None:
            message["extra_content"] = extra
        for key in ("call_id", "response_item_id"):
            if key in call.provider_state:
                message[key] = call.provider_state[key]
    return message
