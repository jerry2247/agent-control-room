"""In-process session registry: live progress for the dashboard poller.

Durable artifacts (documents, embeddings, metrics) live in ClickHouse; this
registry only tracks in-flight pipeline state for the UI.
"""
from __future__ import annotations

import time
from typing import Any

_SESSIONS: dict[str, dict[str, Any]] = {}
_ORDER: list[str] = []
_MAX_SESSIONS = 200


def create(session_id: str, payload: dict) -> dict:
    view = {
        "session_id": session_id,
        "status": "running",
        "stage": "queued",
        "stages": [],
        "created_at": time.time(),
        "error_logs": [],
        **payload,
    }
    _SESSIONS[session_id] = view
    _ORDER.append(session_id)
    while len(_ORDER) > _MAX_SESSIONS:
        _SESSIONS.pop(_ORDER.pop(0), None)
    return view


def update(session_id: str, **fields) -> None:
    if session_id in _SESSIONS:
        _SESSIONS[session_id].update(fields)


def set_stage(session_id: str, stage: str) -> None:
    view = _SESSIONS.get(session_id)
    if view is None:
        return
    now = time.time()
    if view["stages"]:
        last = view["stages"][-1]
        if "ms" not in last:
            last["ms"] = int((now - last["t0"]) * 1000)
    view["stage"] = stage
    view["stages"].append({"stage": stage, "t0": now})


def complete(session_id: str, **fields) -> None:
    set_stage(session_id, "done")
    update(session_id, status="completed", **fields)


def fail(session_id: str, error: str) -> None:
    view = _SESSIONS.get(session_id)
    if view is not None:
        view["error_logs"].append(error)
    update(session_id, status="failed")


def get(session_id: str) -> dict | None:
    view = _SESSIONS.get(session_id)
    if view is None:
        return None
    out = {k: v for k, v in view.items()}
    out["stages"] = [
        {"stage": s["stage"], **({"ms": s["ms"]} if "ms" in s else {})} for s in view["stages"]
    ]
    return out


def list_sessions() -> list[dict]:
    return [
        {
            "session_id": sid,
            "original_query": _SESSIONS[sid].get("original_query", ""),
            "status": _SESSIONS[sid].get("status"),
            "created_at": _SESSIONS[sid].get("created_at"),
        }
        for sid in reversed(_ORDER)
        if sid in _SESSIONS
    ]
