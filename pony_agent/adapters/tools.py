"""Adapter for the existing Hermes registry and tool lifecycle."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from agent.transports.types import NormalizedResponse, ToolCall as LegacyToolCall
from pony_agent.core.ports import Emit
from pony_agent.core.types import JsonDict, ToolCall, ToolOutcome


def _is_error(message: JsonDict) -> bool:
    receipt = message.get("_tool_receipt")
    if isinstance(receipt, dict) and receipt.get("result_status"):
        return str(receipt["result_status"]).lower() == "error"
    content = message.get("content")
    if isinstance(content, str):
        lowered = content.lstrip().lower()
        return lowered.startswith("error") or '"error"' in lowered[:200]
    return False


class LegacyToolRuntime:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def definitions(self, context: JsonDict) -> list[JsonDict]:
        return deepcopy(list(getattr(self.agent, "tools", None) or []))

    async def execute_batch(
        self,
        calls: Sequence[ToolCall],
        context: JsonDict,
        emit: Emit,
    ) -> list[ToolOutcome]:
        loop = asyncio.get_running_loop()
        pending: list[Any] = []
        display_results: dict[str, Any] = {}

        def schedule(kind: str, payload: JsonDict) -> None:
            pending.append(asyncio.run_coroutine_threadsafe(emit(kind, payload), loop))

        def progress(kind, name, preview, display_args, **kwargs):
            schedule(
                "tool.progress",
                {
                    "name": str(name),
                    "preview": preview,
                    "arguments": display_args,
                    **kwargs,
                },
            )

        def started(call_id, name, display_args):
            schedule(
                "tool.started",
                {"call_id": str(call_id), "name": str(name), "arguments": display_args},
            )

        def completed(call_id, name, display_args, result):
            display_results[str(call_id)] = result
            schedule(
                "tool.progress",
                {"call_id": str(call_id), "name": str(name), "status": "completed"},
            )

        def run() -> list[JsonDict]:
            from agent.tool_executor import execute_tool_calls_segmented

            legacy_calls = [
                LegacyToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    provider_data=deepcopy(call.provider_state),
                )
                for call in calls
            ]
            assistant = NormalizedResponse(
                content="",
                tool_calls=legacy_calls,
                finish_reason="tool_calls",
            )
            legacy_messages = list(context.get("messages") or [])
            before = len(legacy_messages)
            self.agent._session_messages = legacy_messages
            self.agent._current_task_id = str(context.get("run_id") or context.get("session_id"))
            self.agent._execution_thread_id = threading.current_thread().ident
            self.agent._interrupt_requested = False
            self.agent._interrupt_message = None
            self.agent._interrupt_thread_signal_pending = False
            self.agent._incremental_persistence_failed = False
            try:
                from run_agent import _set_interrupt

                _set_interrupt(False, self.agent._execution_thread_id)
            except Exception:
                pass
            previous = (
                getattr(self.agent, "tool_progress_callback", None),
                getattr(self.agent, "tool_start_callback", None),
                getattr(self.agent, "tool_complete_callback", None),
            )
            self.agent.tool_progress_callback = progress
            self.agent.tool_start_callback = started
            self.agent.tool_complete_callback = completed
            try:
                execute_tool_calls_segmented(
                    self.agent,
                    assistant,
                    legacy_messages,
                    self.agent._current_task_id,
                    api_call_count=int(context.get("iteration") or 0),
                )
            finally:
                try:
                    from run_agent import _set_interrupt

                    _set_interrupt(False, self.agent._execution_thread_id)
                except Exception:
                    pass
                (
                    self.agent.tool_progress_callback,
                    self.agent.tool_start_callback,
                    self.agent.tool_complete_callback,
                ) = previous
            return [
                message
                for message in legacy_messages[before:]
                if isinstance(message, dict) and message.get("role") == "tool"
            ]

        messages = await asyncio.to_thread(run)
        if pending:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in pending),
                return_exceptions=True,
            )
        by_id = {str(message.get("tool_call_id")): message for message in messages}
        outcomes: list[ToolOutcome] = []
        for call in calls:
            message = by_id.get(call.id)
            if message is None:
                message = {
                    "content": json.dumps(
                        {"error": "tool runtime returned no result"}, ensure_ascii=False
                    ),
                    "_tool_receipt": {"result_status": "error"},
                }
            outcomes.append(
                ToolOutcome(
                    call_id=call.id,
                    name=call.name,
                    model_content=message.get("content"),
                    ui_details=display_results.get(call.id),
                    receipt=deepcopy(message.get("_tool_receipt")),
                    effect=message.get("effect_disposition"),
                    is_error=_is_error(message),
                )
            )
        return outcomes

    async def cancel(self) -> None:
        await asyncio.to_thread(self.agent.interrupt)
