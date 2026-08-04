"""Pony-kernel implementation of CLI oneshot mode."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from pony_agent.adapters import LegacyToolRuntime, provider_for_agent
from pony_agent.core import AgentKernel, KernelConfig, MinimalContextPolicy
from pony_agent.store import SQLiteSessionStore


def _resolve_runtime(model: str | None, provider: str | None):
    from hermes_cli.config import load_config
    from hermes_cli.models import detect_provider_for_model
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
    else:
        cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""
    env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    effective_model = (model or "").strip() or env_model or cfg_model
    effective_provider = (provider or "").strip() or None
    explicit_base_url: str | None = None
    if effective_provider is None and (model or env_model):
        explicit_model = (model or "").strip() or env_model
        try:
            from hermes_cli import model_switch as model_switch

            model_switch._ensure_direct_aliases()
            direct = model_switch.DIRECT_ALIASES.get(explicit_model.lower())
        except Exception:
            direct = None
        if direct is not None:
            effective_model = direct.model
            effective_provider = direct.provider
            if direct.base_url:
                explicit_base_url = direct.base_url.rstrip("/")
        else:
            cfg_provider = ""
            if isinstance(model_cfg, dict):
                cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            current = (
                cfg_provider
                or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                or "auto"
            )
            detected = detect_provider_for_model(explicit_model, current)
            if detected:
                effective_provider, effective_model = detected
    runtime = resolve_runtime_provider(
        requested=effective_provider,
        target_model=effective_model or None,
        explicit_base_url=explicit_base_url,
    )
    return cfg, effective_model, runtime


def _build_agent(
    *,
    cfg: dict[str, Any],
    model: str,
    runtime: dict[str, Any],
    toolsets: list[str] | None,
    session_id: str,
):
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.oneshot import _oneshot_clarify_callback
    from run_agent import AIAgent

    return AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        enabled_toolsets=toolsets,
        quiet_mode=True,
        platform="cli",
        session_id=session_id,
        session_db=None,
        credential_pool=runtime.get("credential_pool"),
        fallback_model=get_fallback_chain(cfg) or None,
        clarify_callback=_oneshot_clarify_callback,
    )


async def _execute(
    *,
    prompt: str,
    cfg: dict[str, Any],
    agent: Any,
    store_path: Path,
):
    provider = provider_for_agent(agent)
    tools = LegacyToolRuntime(agent)
    store = SQLiteSessionStore(store_path)
    context = MinimalContextPolicy()
    kernel = AgentKernel(provider=provider, tools=tools, store=store, context=context)
    compression = cfg.get("compression") if isinstance(cfg.get("compression"), dict) else {}
    threshold = float((compression or {}).get("threshold", 0.50))
    context_budget = int(
        getattr(getattr(agent, "context_compressor", None), "context_length", 200_000)
        or 200_000
    )
    session = kernel.open_session(
        KernelConfig(
            model=str(agent.model),
            system_prompt=str(agent._build_system_prompt() or ""),
            context_budget=context_budget,
            compression_threshold=threshold,
            max_iterations=int(getattr(agent, "max_iterations", 90) or 90),
            max_turn_retries=1,
            metadata={"provider": provider.name, "api_mode": provider.api_mode, "platform": "cli"},
        ),
        session_id=str(agent.session_id),
    )
    final: dict[str, Any] | None = None
    try:
        await session.submit(prompt)
        async for event in session.events():
            if event.kind == "run.completed":
                final = event.payload
                break
            elif event.kind == "run.failed":
                raise RuntimeError(str(event.payload.get("error") or "Pony kernel run failed"))
            elif event.kind == "run.cancelled":
                raise InterruptedError("Pony kernel run cancelled")
    finally:
        await session.close()
        await store.close()
    if final is None:
        raise RuntimeError("Pony kernel produced no terminal event")
    return final


def run_pony_agent(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    toolsets: object = None,
    use_config_toolsets: bool = True,
) -> tuple[str, dict[str, Any]]:
    from hermes_cli.oneshot import _normalize_toolsets
    from hermes_cli.tools_config import _get_platform_tools
    from pony_agent.core.types import new_id

    cfg, effective_model, runtime = _resolve_runtime(model, provider)
    toolsets_list = _normalize_toolsets(toolsets)
    if toolsets_list is None and use_config_toolsets:
        toolsets_list = sorted(_get_platform_tools(cfg, "cli"))
    session_id = new_id("session")
    agent = _build_agent(
        cfg=cfg,
        model=effective_model,
        runtime=runtime,
        toolsets=toolsets_list,
        session_id=session_id,
    )
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    pony_home = Path(os.environ.get("PONY_HOME") or (Path.home() / ".pony"))
    try:
        terminal = asyncio.run(
            _execute(
                prompt=prompt,
                cfg=cfg,
                agent=agent,
                store_path=pony_home / "kernel.db",
            )
        )
        usage = terminal.get("usage") or {}
        result = {
            "final_response": terminal.get("final_response") or "",
            "completed": True,
            "failed": False,
            "partial": False,
            "session_id": session_id,
            "model": terminal.get("model") or agent.model,
            "provider": terminal.get("provider") or agent.provider,
            "api_calls": terminal.get("api_calls", 0),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cache_read_tokens": usage.get("cache_read_tokens"),
            "cache_write_tokens": usage.get("cache_write_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": None,
            "cost_status": "unavailable",
            "cost_source": "pony-kernel",
            "service_tier": getattr(agent, "service_tier", None),
        }
        return str(result["final_response"]), result
    finally:
        try:
            agent.shutdown_memory_provider()
        except Exception:
            logging.debug("Pony oneshot memory cleanup failed", exc_info=True)
        try:
            agent.close()
        except Exception:
            logging.debug("Pony oneshot agent cleanup failed", exc_info=True)
