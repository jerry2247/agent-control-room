"""LangGraph state definition for the harness pipeline."""
from __future__ import annotations

from typing import Any, List, TypedDict


class HarnessState(TypedDict, total=False):
    # Identity / inputs
    session_id: str
    original_query: str
    sanitized_query: str
    n_queries: int
    epsilon: float          # 0.0 => auto-optimize
    include_baseline: bool

    # Frame analysis (Pillar D)
    frame: dict                        # {type, presupposition, neutral_topic, counter_queries, anchors}
    neutral_topic: str                 # de-biased epsilon-ball center text

    # Variance engine outputs
    candidate_queries: List[dict]      # [{query, axis}]
    generated_queries: List[str]
    query_distances: dict              # query -> ||x - c||_2
    query_axes: dict                   # query -> axis (incl. "counter_frame")
    epsilon_used: float
    epsilon_mode: str                  # "auto" | "fixed"
    epsilon_curve: List[dict]

    # Critic outputs
    approved_queries: List[str]
    rejected_queries: List[dict]       # [{query, reason}]
    critic_feedback: str
    retry_count: int

    # Execution / ETL
    search_results: dict               # query -> [SearchResult]
    baseline_results: List[Any]
    execution_urls: List[str]
    documents: List[dict]              # chunk rows ready for ClickHouse
    airbyte_job: dict

    # Analytics
    metrics: dict                      # {"harness": {...}, "baseline": {...}, "deltas": {...}}
    scatter: List[dict]
    synthesis: dict

    # Bookkeeping
    error_logs: List[str]
    timings: dict
