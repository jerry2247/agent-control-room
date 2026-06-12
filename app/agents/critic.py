"""Pre-execution Critic: audits generated queries BEFORE any web tool runs.

Three deterministic gates (always on) plus an optional LLM audit:
  1. safety     : blocklist of harm/PII-seeking patterns
  2. relevance  : re-verifies the epsilon constraint ||x_i - c|| <= eps + slack
  3. dedup      : rejects near-duplicate queries (pairwise distance < delta)
"""
from __future__ import annotations

import re

import numpy as np

from app.agents.state import HarnessState
from app import deps

_RADIUS_SLACK = 0.05

_SAFETY_PATTERNS = [
    r"\b(make|build|synthesi[sz]e|manufactur\w*)\b.{0,40}\b(bomb|explosive|nerve agent|sarin|ricin|bioweapon|chemical weapon)\b",
    r"\b(home address|phone number|ssn|social security number)\b.{0,30}\bof\b",
    r"\bhow to (hack|break) into\b",
    r"\b(undetectable|untraceable)\b.{0,20}\b(poison|weapon)\b",
    r"\bchild\b.{0,30}\b(sexual|explicit)\b",
    r"\b(kill|harm|hurt)\b.{0,20}\b(myself|yourself)\b",
]

_LLM_SYSTEM = (
    "You audit web search queries for an autonomous agent. For each query answer "
    "whether it is SAFE to execute (no harm facilitation, no PII hunting, no illegal "
    "instructions) and ON-TOPIC for the user's subject. Output ONLY a JSON array of "
    'objects: [{"query": str, "safe": bool, "on_topic": bool, "reason": str}].'
)


def local_safety_check(query: str) -> str | None:
    q = query.lower()
    for pat in _SAFETY_PATTERNS:
        if re.search(pat, q):
            return f"safety pattern matched: /{pat}/"
    if not (3 <= len(query) <= 250):
        return "length out of bounds"
    return None


async def verify_query_safety(state: HarnessState) -> dict:
    settings = deps.get_settings()
    embedder = deps.get_embedder()
    queries = state.get("generated_queries", [])
    # Relevance is measured against the NEUTRALIZED topic core (same center the
    # generator optimized against), not the user's loaded phrasing.
    center_text = state.get("neutral_topic") or state["sanitized_query"]
    eps = state.get("epsilon_used", 1.3)

    vectors = await embedder.embed_batch([center_text] + queries)
    center = np.asarray(vectors[0])
    E = np.vstack(vectors[1:]) if queries else np.zeros((0, len(center)))

    approved: list[str] = []
    approved_vecs: list[np.ndarray] = []
    rejected = list(state.get("rejected_queries", []))
    errors = list(state.get("error_logs", []))

    # ---- optional LLM audit (vetoes only; deterministic gates still apply) ----
    llm_flags: dict[str, str] = {}
    if deps.get_llm().available and queries:
        try:
            listing = "\n".join(f"- {q}" for q in queries)
            arr = await deps.get_llm().chat_json(
                _LLM_SYSTEM, f"User subject: {center_text!r}\nQueries:\n{listing}"
            )
            for item in arr:
                if isinstance(item, dict) and not (item.get("safe", True) and item.get("on_topic", True)):
                    llm_flags[str(item.get("query", ""))] = str(item.get("reason", "LLM critic veto"))
        except Exception as exc:
            errors.append(f"LLM critic unavailable, deterministic gates only: {exc}")

    for i, q in enumerate(queries):
        reason = local_safety_check(q)
        if reason is None:
            dist = float(np.linalg.norm(E[i] - center))
            if dist > eps + _RADIUS_SLACK:
                reason = f"out of epsilon radius: d={dist:.3f} > {eps:.3f}"
        if reason is None and q in llm_flags:
            reason = f"LLM critic: {llm_flags[q]}"
        if reason is None:
            for v in approved_vecs:
                if float(np.linalg.norm(E[i] - v)) < settings.dedup_distance:
                    reason = "near-duplicate of an approved query"
                    break
        if reason is None:
            approved.append(q)
            approved_vecs.append(E[i])
        else:
            rejected.append({"query": q, "reason": reason})

    feedback = "; ".join(f"{r['query'][:60]!r}: {r['reason']}" for r in rejected[-5:])
    return {
        "approved_queries": approved,
        "rejected_queries": rejected,
        "critic_feedback": feedback,
        "error_logs": errors,
    }


def route_after_audit(state: HarnessState) -> str:
    approved = len(state.get("approved_queries", []))
    n = state.get("n_queries", 5)
    settings = deps.get_settings()
    if approved >= max(2, n // 2):
        return "execute"
    if state.get("retry_count", 0) < settings.max_retries:
        return "retry"
    return "execute" if approved >= 1 else "abort"
