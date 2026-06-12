"""Compiled LangGraph state machine: the full pipeline as a cyclic graph.

    START -> ingress -> generator -> critic --(retry)--> generator
                                        |--(execute)--> executor -> etl
                                        |--(abort)----> END           |
                                                                      v
                          END <- synthesize <- evaluate <- persist <--

The critic->generator edge makes the graph CYCLIC (bounded by max_retries and
LangGraph's recursion_limit): rejected batches feed critic_feedback back into
regeneration. Every node is wrapped with a timeout and live progress updates.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
from langgraph.graph import StateGraph, START, END

from app.agents.critic import route_after_audit, verify_query_safety
from app.agents.generator import generate_variance_queries
from app.agents.state import HarnessState
from app.core import registry
from app.core.synthesis import pca_2d, synthesize_viewpoints
from app.core.textproc import sanitize_query
from app import deps


def _wrap(name: str, fn):
    """Timeout + timing + registry progress around a node."""

    async def node(state: HarnessState) -> dict:
        settings = deps.get_settings()
        sid = state.get("session_id", "")
        registry.set_stage(sid, name)
        t0 = time.time()
        update = await asyncio.wait_for(fn(state), timeout=settings.node_timeout_s)
        timings = dict(state.get("timings", {}))
        timings[name] = round((time.time() - t0) * 1000)
        update["timings"] = timings
        # Mirror interesting fields into the live registry for the dashboard
        mirror = {k: v for k, v in update.items() if k in (
            "generated_queries", "approved_queries", "rejected_queries", "query_distances",
            "query_axes", "frame", "neutral_topic",
            "epsilon_used", "epsilon_mode", "epsilon_curve", "metrics", "scatter",
            "synthesis", "error_logs", "execution_urls", "timings", "airbyte_job",
        )}
        registry.update(sid, **mirror)
        return update

    return node


# ---------------------------------------------------------------------------
# Nodes (generator and critic are imported; the rest are defined here)
# ---------------------------------------------------------------------------


async def ingress_guardrails(state: HarnessState) -> dict:
    """Stage 1: ingress sanitation. When LLM_BACKEND=truefoundry, all later LLM
    calls additionally pass through the gateway's server-side guardrails."""
    sanitized = sanitize_query(state["original_query"])
    return {"sanitized_query": sanitized}


async def execute_searches(state: HarnessState) -> dict:
    search = deps.get_search()
    out = await search.run(
        state["approved_queries"],
        control_query=state["sanitized_query"],
        include_baseline=state.get("include_baseline", True),
    )
    urls = [r.url for results in out["per_query"].values() for r in results]
    return {
        "search_results": out["per_query"],
        "baseline_results": out["baseline"],
        "execution_urls": urls,
        "error_logs": state.get("error_logs", []) + out["errors"],
    }


async def etl_pipeline(state: HarnessState) -> dict:
    etl = deps.get_etl()
    sid = state["session_id"]
    flat = [r for results in state["search_results"].values() for r in results]
    baseline = list(state.get("baseline_results", []))

    await etl.hydrate_contents(flat + baseline)

    rows = etl.to_chunks(flat, sid, control=False, id_start=0)
    rows += etl.to_chunks(baseline, sid, control=True, id_start=len(rows))

    update: dict = {"documents": rows}
    errors = list(state.get("error_logs", []))

    # Durable path: stage raw records + trigger the Airbyte connection sync.
    if getattr(etl, "name", "") == "airbyte":
        try:
            etl.stage_records(flat + baseline, sid)
            job = await etl.trigger_sync()
            update["airbyte_job"] = {"jobId": job.get("jobId"), "status": job.get("status")}
        except Exception as exc:
            errors.append(f"Airbyte sync trigger failed (inline ETL still applied): {exc}")
    update["error_logs"] = errors
    return update


async def persist_to_clickhouse(state: HarnessState) -> dict:
    embedder = deps.get_embedder()
    db = deps.get_db()
    rows = state["documents"]
    if rows:
        vectors = await embedder.embed_batch([r["content"] for r in rows])
        for r, v in zip(rows, vectors):
            r["embedding"] = [float(x) for x in np.asarray(v, dtype=np.float32)]
        await asyncio.to_thread(db.insert_documents, rows)
    await asyncio.to_thread(
        db.insert_session, state["session_id"], state["sanitized_query"],
        state.get("epsilon_used", 0.0), state.get("epsilon_mode", "fixed"), state["n_queries"],
    )
    return {"documents": rows}


async def evaluate_metrics(state: HarnessState) -> dict:
    """Pillars B + C + D, computed by SQL inside ClickHouse."""
    db = deps.get_db()
    sid = state["session_id"]
    harness = await asyncio.to_thread(db.corpus_metrics, sid, 0)
    baseline = await asyncio.to_thread(db.corpus_metrics, sid, 1)

    def delta(key):
        h, b = harness.get(key, 0.0), baseline.get(key, 0.0)
        return {"harness": h, "baseline": b,
                "lift_pct": round(100.0 * (h - b) / b, 1) if b else None}

    metrics = {
        "harness": harness,
        "baseline": baseline,
        "deltas": {
            "semantic_spread": delta("semantic_spread"),
            "shannon_entropy_bits": delta("shannon_entropy_bits"),
            "ecosystem_entropy_bits": delta("ecosystem_entropy_bits"),
            "n_domains": delta("n_domains"),
        },
    }

    # Pillar D: frame balance. 0 = evidence for/against the presupposed claim
    # equally represented; >0 = the corpus argues the frame the user asked in.
    frame = state.get("frame") or {}
    if frame.get("presupposition"):
        embedder = deps.get_embedder()
        a_vec, n_vec = await embedder.embed_batch(
            [frame["affirm_anchor"], frame["negate_anchor"]]
        )
        h_fb = await asyncio.to_thread(
            db.frame_balance, sid, 0, [float(x) for x in a_vec], [float(x) for x in n_vec]
        )
        b_fb = await asyncio.to_thread(
            db.frame_balance, sid, 1, [float(x) for x in a_vec], [float(x) for x in n_vec]
        )
        reduction = None
        if abs(b_fb["balance"]) > 1e-9:
            reduction = round(100.0 * (1 - abs(h_fb["balance"]) / abs(b_fb["balance"])), 1)
        h = h_fb["balance"]
        verdict = "balanced" if abs(h) < 0.08 else ("leans_affirm" if h > 0 else "leans_counter")
        metrics["frame_balance"] = {
            "presupposition": frame["presupposition"],
            "harness": h,
            "baseline": b_fb["balance"],
            "bias_reduction_pct": reduction,
            "verdict": verdict,
            "sql": h_fb.get("sql", ""),
        }
        harness["sql"]["frame_balance"] = h_fb.get("sql", "")
    return {"metrics": metrics}


async def synthesize_results(state: HarnessState) -> dict:
    """Cluster viewpoints, surface consensus vs conflict, build the scatter."""
    db = deps.get_db()
    sid = state["session_id"]
    rows = await asyncio.to_thread(db.fetch_documents, sid)
    if not rows:
        return {"synthesis": {"clusters": [], "consensus_terms": [], "conflict": None}, "scatter": []}

    # Document-level (mean chunk embedding per URL), harness group only
    from app.core.metrics import doc_level_embeddings
    harness_rows = [r for r in rows if r["control_group"] == 0]
    doc_embs = doc_level_embeddings(harness_rows)
    doc_meta: dict[str, dict] = {}
    for r in harness_rows:
        doc_meta.setdefault(r["url"], {
            "url": r["url"], "domain": r["domain"], "title": r["title"],
            "content": r["content"],
        })
    urls = list(doc_embs.keys())
    E = np.vstack([doc_embs[u] for u in urls]) if urls else np.zeros((0, 2))
    docs = [doc_meta[u] for u in urls]
    synthesis = synthesize_viewpoints(docs, E)

    # Optional LLM narrative on top of the deterministic structure
    if deps.get_llm().available and synthesis["clusters"]:
        try:
            payload = "\n\n".join(
                f"Viewpoint {c['cluster_id']} ({', '.join(c['label_terms'])}): "
                + " | ".join(e["text"][:200] for e in c["excerpts"])
                for c in synthesis["clusters"]
            )
            narrative = await deps.get_llm().chat(
                "Summarize for a reader who wants the full picture: where these viewpoint "
                "clusters agree, where they directly conflict, and what a balanced reader "
                "should check next. 120 words max, plain prose.",
                payload, max_tokens=400,
            )
            synthesis["narrative"] = narrative.strip()
        except Exception:
            pass

    # 2-D scatter of every chunk (harness colored by doc cluster, baseline gray)
    all_embs = np.vstack([r["embedding"] for r in rows])
    coords = pca_2d(all_embs)
    url_cluster = dict(zip(urls, synthesis.get("labels", [])))
    scatter = [
        {
            "x": round(float(coords[i][0]), 4), "y": round(float(coords[i][1]), 4),
            "group": "baseline" if rows[i]["control_group"] else "harness",
            "cluster": int(url_cluster.get(rows[i]["url"], -1)),
            "domain": rows[i]["domain"], "title": rows[i]["title"][:80],
        }
        for i in range(len(rows))
    ]
    synthesis.pop("labels", None)
    return {"synthesis": synthesis, "scatter": scatter}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(HarnessState)
    g.add_node("ingress", _wrap("ingress", ingress_guardrails))
    g.add_node("generator", _wrap("generator", generate_variance_queries))
    g.add_node("critic", _wrap("critic", verify_query_safety))
    g.add_node("executor", _wrap("executor", execute_searches))
    g.add_node("etl", _wrap("etl", etl_pipeline))
    g.add_node("persist", _wrap("persist", persist_to_clickhouse))
    g.add_node("evaluate", _wrap("evaluate", evaluate_metrics))
    g.add_node("synthesize", _wrap("synthesize", synthesize_results))

    g.add_edge(START, "ingress")
    g.add_edge("ingress", "generator")
    g.add_edge("generator", "critic")
    g.add_conditional_edges("critic", route_after_audit, {
        "retry": "generator",      # cyclic edge, bounded by max_retries
        "execute": "executor",
        "abort": END,
    })
    g.add_edge("executor", "etl")
    g.add_edge("etl", "persist")
    g.add_edge("persist", "evaluate")
    g.add_edge("evaluate", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


async def run_pipeline(
    session_id: str,
    query: str,
    n_queries: int,
    epsilon: float = 0.0,
    include_baseline: bool = True,
) -> dict:
    initial: HarnessState = {
        "session_id": session_id,
        "original_query": query,
        "n_queries": n_queries,
        "epsilon": epsilon,
        "include_baseline": include_baseline,
        "retry_count": 0,
        "error_logs": [],
    }
    try:
        final = await get_graph().ainvoke(initial, config={"recursion_limit": 50})
        if not final.get("approved_queries"):
            registry.fail(session_id, "Critic rejected all generated queries; nothing executed.")
        elif not final.get("metrics"):
            registry.fail(session_id, "Pipeline ended before metric evaluation.")
        else:
            registry.complete(
                session_id,
                metrics=final.get("metrics"),
                synthesis=final.get("synthesis"),
                scatter=final.get("scatter"),
                documents_summary=_doc_summary(final.get("documents", [])),
            )
        return final
    except Exception as exc:
        registry.fail(session_id, f"{type(exc).__name__}: {exc}")
        raise


def _doc_summary(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in rows:
        d = seen.setdefault(r["url"], {
            "url": r["url"], "domain": r["domain"], "ecosystem": r["ecosystem"],
            "title": r["title"], "group": "baseline" if r["control_group"] else "harness",
            "chunks": 0, "source_query": r["source_query"],
        })
        d["chunks"] += 1
    return list(seen.values())
