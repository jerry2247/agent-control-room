"""Lazy singletons for backends (swappable in tests via reset())."""
from __future__ import annotations

from app.config import get_settings as _get_settings, reset_settings

_embedder = None
_db = None
_llm = None
_search = None
_etl = None


def get_settings():
    return _get_settings()


def get_embedder():
    global _embedder
    if _embedder is None:
        from app.core.embeddings import build_embedder
        _embedder = build_embedder(get_settings())
    return _embedder


def get_db():
    global _db
    if _db is None:
        from app.database.clickhouse_client import build_database
        _db = build_database(get_settings())
        _db.ensure_schema()
    return _db


def get_llm():
    global _llm
    if _llm is None:
        from app.services.truefoundry_client import LLMClient
        _llm = LLMClient(get_settings())
    return _llm


def get_search():
    global _search
    if _search is None:
        from app.services.composio_client import SearchExecutor
        _search = SearchExecutor(get_settings())
    return _search


def get_etl():
    global _etl
    if _etl is None:
        from app.services.airbyte_client import build_etl
        _etl = build_etl(get_settings())
    return _etl


def reset() -> None:
    """Drop all singletons and re-read environment (tests)."""
    global _embedder, _db, _llm, _search, _etl
    if _db is not None and hasattr(_db, "close"):
        _db.close()
    _embedder = _db = _llm = _search = _etl = None
    reset_settings()
