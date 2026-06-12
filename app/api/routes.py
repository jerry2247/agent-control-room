"""Execution endpoints."""
from __future__ import annotations

import asyncio
from typing import Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.orchestrator import new_session_id, run_pipeline
from app.core import registry
from app.core.textproc import IngressRejection
from app import deps

router = APIRouter(prefix="/api")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=400)
    n_queries: int = Field(default=5, ge=2, le=8)
    epsilon: Optional[Union[float, Literal["auto"]]] = "auto"  # number = fixed radius
    include_baseline: bool = True
    sync: bool = False        # True: wait for completion (CLI/tests/judges' curl)


@router.post("/search")
async def start_search(req: SearchRequest):
    eps = 0.0 if (req.epsilon in (None, "auto")) else float(req.epsilon)
    if eps < 0 or eps > 2.0:
        raise HTTPException(400, "epsilon must be in (0, 2] or 'auto'")
    sid = new_session_id()
    registry.create(sid, {
        "original_query": req.query,
        "n_queries": req.n_queries,
        "epsilon_requested": req.epsilon,
        "include_baseline": req.include_baseline,
        "backends": deps.get_settings().resolved,
    })
    coro = run_pipeline(sid, req.query, req.n_queries, eps, req.include_baseline)
    if req.sync:
        try:
            await coro
        except IngressRejection as exc:
            raise HTTPException(422, str(exc))
        except Exception:
            pass  # status/detail captured in the registry
        return registry.get(sid)

    task = asyncio.create_task(coro)
    task.add_done_callback(lambda t: t.exception())  # swallow: registry holds the error
    return {"session_id": sid, "status": "running"}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    view = registry.get(session_id)
    if view is None:
        raise HTTPException(404, "unknown session id")
    return view


@router.get("/sessions")
async def list_sessions():
    return registry.list_sessions()


@router.get("/metrics/{session_id}")
async def live_metrics(session_id: str):
    """Recomputed in ClickHouse at request time (live OLAP, not cached)."""
    db = deps.get_db()
    try:
        harness = await asyncio.to_thread(db.corpus_metrics, session_id, 0)
        baseline = await asyncio.to_thread(db.corpus_metrics, session_id, 1)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"session_id": session_id, "computed_live_in": db.name,
            "harness": harness, "baseline": baseline}


@router.post("/epsilon/optimize")
async def epsilon_optimize(req: SearchRequest):
    """Run only the variance engine sweep: returns the diversity-vs-epsilon
    curve and the recommended epsilon (no searches executed)."""
    import numpy as np
    from app.agents.generator import template_candidates, llm_candidates
    from app.core.divergence import optimize_epsilon
    from app.core.textproc import sanitize_query

    try:
        query = sanitize_query(req.query)
    except IngressRejection as exc:
        raise HTTPException(422, str(exc))
    settings = deps.get_settings()
    cands = None
    if deps.get_llm().available:
        try:
            cands = await llm_candidates(query, max(12, 3 * req.n_queries), "")
        except Exception:
            cands = None
    cands = cands or template_candidates(query)
    vecs = await deps.get_embedder().embed_batch([query] + [c["query"] for c in cands])
    eps, curve = optimize_epsilon(
        np.vstack(vecs[1:]), np.asarray(vecs[0]), req.n_queries,
        grid_min=settings.epsilon_grid_min, grid_max=settings.epsilon_grid_max,
        steps=settings.epsilon_grid_steps,
    )
    return {"recommended_epsilon": eps, "curve": curve,
            "candidates": [c["query"] for c in cands]}


@router.get("/health")
async def health():
    s = deps.get_settings()
    return {
        "status": "ok",
        "service": "orthogonal-search-harness",
        "version": "1.1.0",
        "backends": s.resolved,
        "database_engine": deps.get_db().name,
        "embedding_dim": s.embedding_dim if s.embedding_backend == "local" else "provider-defined",
    }
