"""SQLite event log and transactional projections for Pony sessions."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

from pony_agent.core.types import JsonDict, KernelEvent, SessionSnapshot


_SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "cookie",
    "secret",
}

_OPAQUE_REPLAY_KEYS = {
    "signature",
    "thought_signature",
    "encrypted_content",
    "reasoning_details",
    "anthropic_content_blocks",
    "codex_reasoning_items",
}


def _redact(value: Any, *, key_hint: str = "") -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_KEYS or normalized.endswith("_password"):
                clean[str(key)] = "[REDACTED]"
            elif normalized in _OPAQUE_REPLAY_KEYS:
                clean[str(key)] = item
            else:
                clean[str(key)] = _redact(item, key_hint=normalized)
        return clean
    if isinstance(value, list):
        return [_redact(item, key_hint=key_hint) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, key_hint=key_hint) for item in value]
    if isinstance(value, str) and key_hint not in _OPAQUE_REPLAY_KEYS:
        try:
            from agent.redact import redact_sensitive_text

            return redact_sensitive_text(value, force=True)
        except Exception:
            return value
    return value


def _json(value: Any) -> str:
    return json.dumps(_redact(value), ensure_ascii=False, separators=(",", ":"), default=str)


class SQLiteSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'new',
                version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, seq)
            );

            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                provider_state_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, message_id),
                UNIQUE (session_id, seq)
            );

            CREATE TABLE IF NOT EXISTS tool_runs (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                call_id TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                seq INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT,
                result_json TEXT,
                receipt_json TEXT,
                effect_json TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, call_id, attempt)
            );

            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(session_id, type, seq);
            CREATE INDEX IF NOT EXISTS idx_messages_order
                ON messages(session_id, seq);
            CREATE INDEX IF NOT EXISTS idx_tool_runs_order
                ON tool_runs(session_id, seq);
            """
        )
        self._conn.commit()

    async def append(self, event: KernelEvent) -> KernelEvent:
        if self._closed:
            raise RuntimeError("session store is closed")
        async with self._lock:
            return await asyncio.to_thread(self._append_sync, event)

    def _append_sync(self, event: KernelEvent) -> KernelEvent:
        payload = _redact(event.payload)
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(id, metadata_json) VALUES (?, '{}')",
                (event.session_id,),
            )
            row = self._conn.execute(
                "SELECT version FROM sessions WHERE id = ?",
                (event.session_id,),
            ).fetchone()
            seq = int(row["version"]) + 1
            self._conn.execute(
                "INSERT INTO events(session_id, seq, type, payload_json) VALUES (?, ?, ?, ?)",
                (event.session_id, seq, event.kind, _json(payload)),
            )
            self._project(event.session_id, seq, event.kind, payload)
            status = self._status_for(event.kind)
            if event.kind == "session.started":
                metadata = _json(payload.get("metadata") or {})
                self._conn.execute(
                    "UPDATE sessions SET metadata_json = ? WHERE id = ?",
                    (metadata, event.session_id),
                )
            self._conn.execute(
                "UPDATE sessions SET version = ?, status = COALESCE(?, status), "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (seq, status, event.session_id),
            )
        return replace(event, payload=payload, seq=seq)

    @staticmethod
    def _status_for(kind: str) -> str | None:
        return {
            "session.started": "ready",
            "run.started": "running",
            "run.completed": "ready",
            "run.failed": "failed",
            "run.cancelled": "cancelled",
            "session.closed": "closed",
        }.get(kind)

    def _project(self, session_id: str, seq: int, kind: str, payload: JsonDict) -> None:
        if kind in {"message.user", "message.assistant"}:
            message_id = str(payload.get("id") or f"event-{seq}")
            self._conn.execute(
                """
                INSERT INTO messages(
                    session_id, message_id, seq, role, content_json, provider_state_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, message_id) DO UPDATE SET
                    seq=excluded.seq, role=excluded.role,
                    content_json=excluded.content_json,
                    provider_state_json=excluded.provider_state_json
                """,
                (
                    session_id,
                    message_id,
                    seq,
                    str(payload.get("role") or kind.removeprefix("message.")),
                    _json(payload),
                    _json(payload.get("provider_state"))
                    if payload.get("provider_state") is not None
                    else None,
                ),
            )
            return

        if kind == "tool.requested":
            self._conn.execute(
                """
                INSERT INTO tool_runs(
                    session_id, call_id, attempt, seq, name, status, input_json
                ) VALUES (?, ?, 1, ?, ?, 'requested', ?)
                ON CONFLICT(session_id, call_id, attempt) DO UPDATE SET
                    seq=excluded.seq, name=excluded.name, status='requested',
                    input_json=excluded.input_json, updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    str(payload.get("call_id") or f"call-{seq}"),
                    seq,
                    str(payload.get("name") or "unknown"),
                    _json(payload.get("arguments")),
                ),
            )
            return

        if kind == "tool.started":
            call_id = str(payload.get("call_id") or f"call-{seq}")
            self._conn.execute(
                """
                UPDATE tool_runs SET status='started', seq=?, updated_at=CURRENT_TIMESTAMP
                WHERE session_id=? AND call_id=? AND attempt=1
                """,
                (seq, session_id, call_id),
            )
            return

        if kind in {"tool.completed", "tool.failed"}:
            call_id = str(payload.get("call_id") or f"call-{seq}")
            self._conn.execute(
                """
                INSERT INTO tool_runs(
                    session_id, call_id, attempt, seq, name, status,
                    result_json, receipt_json, effect_json
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, call_id, attempt) DO UPDATE SET
                    seq=excluded.seq, name=excluded.name, status=excluded.status,
                    result_json=excluded.result_json,
                    receipt_json=excluded.receipt_json,
                    effect_json=excluded.effect_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    call_id,
                    seq,
                    str(payload.get("name") or "unknown"),
                    "failed" if kind == "tool.failed" else "completed",
                    _json(payload.get("model_content")),
                    _json(payload.get("receipt")),
                    _json(payload.get("effect")),
                ),
            )
            message = payload.get("message")
            if isinstance(message, dict):
                message_id = str(message.get("id") or f"tool-{call_id}")
                self._conn.execute(
                    """
                    INSERT INTO messages(
                        session_id, message_id, seq, role, content_json
                    ) VALUES (?, ?, ?, 'tool', ?)
                    ON CONFLICT(session_id, message_id) DO UPDATE SET
                        seq=excluded.seq, content_json=excluded.content_json
                    """,
                    (session_id, message_id, seq, _json(message)),
                )

    async def load(self, session_id: str) -> SessionSnapshot:
        if self._closed:
            raise RuntimeError("session store is closed")
        async with self._lock:
            return await asyncio.to_thread(self._load_sync, session_id, True)

    async def rebuild(self, session_id: str) -> SessionSnapshot:
        if self._closed:
            raise RuntimeError("session store is closed")
        async with self._lock:
            return await asyncio.to_thread(self._load_sync, session_id, False)

    def _load_sync(self, session_id: str, include_events: bool) -> SessionSnapshot:
        session = self._conn.execute(
            "SELECT status, version FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            return SessionSnapshot(session_id=session_id)
        rows = self._conn.execute(
            "SELECT seq, type, payload_json FROM events WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        all_events = [
            KernelEvent(session_id, row["type"], json.loads(row["payload_json"]), row["seq"])
            for row in rows
        ]
        messages: list[JsonDict] = []
        for event in all_events:
            if event.kind in {"message.user", "message.assistant"}:
                messages.append(event.payload)
            elif event.kind in {"tool.completed", "tool.failed"}:
                message = event.payload.get("message")
                if isinstance(message, dict):
                    messages.append(message)
            elif event.kind == "context.compacted":
                compacted = event.payload.get("messages")
                if isinstance(compacted, list):
                    messages = [m for m in compacted if isinstance(m, dict)]
        events = all_events if include_events else []
        return SessionSnapshot(
            session_id=session_id,
            status=session["status"],
            version=int(session["version"]),
            messages=messages,
            events=events,
        )

    async def close(self) -> None:
        if self._closed:
            return
        async with self._lock:
            await asyncio.to_thread(self._close_sync)
            self._closed = True

    def _close_sync(self) -> None:
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.close()
