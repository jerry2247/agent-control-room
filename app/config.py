"""Central configuration for the Orthogonal Search Harness.

Every external dependency is a pluggable backend selected via environment
variables, so the system runs end-to-end with ZERO keys (Tier 0) and each
sponsor integration can be switched on independently:

  LLM_BACKEND        local | anthropic | truefoundry
  EMBEDDING_BACKEND  local | api            (api = OpenAI-compatible, e.g. TrueFoundry gateway)
  SEARCH_BACKEND     mock  | composio
  ETL_BACKEND        inline | airbyte       (airbyte = stage + trigger sync, inline also runs)
  CLICKHOUSE_MODE    auto  | chdb | cloud | memory
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    # ----- LLM (query generation + critic + synthesis) -----
    llm_backend: str = "local"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    truefoundry_base_url: str = ""        # e.g. https://<org>.truefoundry.cloud/api/llm
    truefoundry_api_key: str = ""
    truefoundry_chat_model: str = ""      # e.g. anthropic-main/claude-sonnet-4-6

    # ----- Embeddings -----
    embedding_backend: str = "local"      # local = deterministic hash projection (256-d)
    embedding_api_base: str = ""          # OpenAI-compatible /embeddings endpoint base
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 256

    # ----- Search execution (Composio) -----
    search_backend: str = "mock"
    composio_api_key: str = ""
    composio_base_url: str = "https://backend.composio.dev"
    composio_tool_slug: str = "COMPOSIO_SEARCH_SEARCH"
    composio_user_id: str = "default"
    results_per_query: int = 5

    # ----- ETL (Airbyte) -----
    etl_backend: str = "inline"
    airbyte_api_base: str = "https://api.airbyte.com/v1"
    airbyte_client_id: str = ""
    airbyte_client_secret: str = ""
    airbyte_connection_id: str = ""
    airbyte_staging_dir: str = ".airbyte_staging"

    # ----- ClickHouse -----
    clickhouse_mode: str = "auto"
    clickhouse_host: str = ""
    clickhouse_port: int = 8443
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "default"
    chdb_path: str = ".chdb_data"

    # ----- Variance engine -----
    n_queries_default: int = 5
    n_queries_max: int = 8
    default_epsilon: float = 0.0          # 0.0 => auto-optimize epsilon
    epsilon_grid_min: float = 0.55
    epsilon_grid_max: float = 1.45
    epsilon_grid_steps: int = 11
    dedup_distance: float = 0.30
    max_retries: int = 3

    # ----- Execution limits -----
    node_timeout_s: float = 120.0
    fetch_timeout_s: float = 12.0
    max_chunks_per_doc: int = 3
    chunk_chars: int = 700
    max_total_chunks: int = 120

    resolved: dict = field(default_factory=dict)


def load_settings() -> Settings:
    s = Settings()

    s.anthropic_api_key = _env("ANTHROPIC_API_KEY")
    s.anthropic_model = _env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    s.truefoundry_base_url = _env("TRUEFOUNDRY_BASE_URL").rstrip("/")
    s.truefoundry_api_key = _env("TRUEFOUNDRY_API_KEY")
    s.truefoundry_chat_model = _env("TRUEFOUNDRY_CHAT_MODEL")

    s.embedding_api_base = _env("EMBEDDING_API_BASE").rstrip("/")
    s.embedding_api_key = _env("EMBEDDING_API_KEY")
    s.embedding_model = _env("EMBEDDING_MODEL")
    s.embedding_dim = _env_int("EMBEDDING_DIM", 256)

    s.composio_api_key = _env("COMPOSIO_API_KEY")
    s.composio_base_url = _env("COMPOSIO_BASE_URL", "https://backend.composio.dev").rstrip("/")
    s.composio_tool_slug = _env("COMPOSIO_TOOL_SLUG", "COMPOSIO_SEARCH_SEARCH")
    s.composio_user_id = _env("COMPOSIO_USER_ID", "default")
    s.results_per_query = min(_env_int("RESULTS_PER_QUERY", 5), 8)

    s.airbyte_api_base = _env("AIRBYTE_API_BASE", "https://api.airbyte.com/v1").rstrip("/")
    s.airbyte_client_id = _env("AIRBYTE_CLIENT_ID")
    s.airbyte_client_secret = _env("AIRBYTE_CLIENT_SECRET")
    s.airbyte_connection_id = _env("AIRBYTE_CONNECTION_ID")
    s.airbyte_staging_dir = _env("AIRBYTE_STAGING_DIR", ".airbyte_staging")

    s.clickhouse_host = _env("CLICKHOUSE_HOST")
    s.clickhouse_port = _env_int("CLICKHOUSE_PORT", 8443)
    s.clickhouse_user = _env("CLICKHOUSE_USER", "default")
    s.clickhouse_password = _env("CLICKHOUSE_PASSWORD")
    s.clickhouse_database = _env("CLICKHOUSE_DATABASE", "default")
    s.chdb_path = _env("CHDB_PATH", ".chdb_data")

    s.n_queries_default = _env_int("N_QUERIES_DEFAULT", 5)
    s.default_epsilon = _env_float("DEFAULT_EPSILON", 0.0)

    # ---- Resolve backend selection (explicit env wins, else infer from keys) ----
    s.llm_backend = _env("LLM_BACKEND") or (
        "truefoundry" if (s.truefoundry_api_key and s.truefoundry_base_url)
        else "anthropic" if s.anthropic_api_key
        else "local"
    )
    s.embedding_backend = _env("EMBEDDING_BACKEND") or (
        "api" if (s.embedding_api_base and s.embedding_model) else "local"
    )
    s.search_backend = _env("SEARCH_BACKEND") or ("composio" if s.composio_api_key else "mock")
    s.etl_backend = _env("ETL_BACKEND") or "inline"
    s.clickhouse_mode = _env("CLICKHOUSE_MODE", "auto")
    if s.clickhouse_mode == "auto":
        if s.clickhouse_host:
            s.clickhouse_mode = "cloud"
        else:
            try:
                import chdb  # noqa: F401
                s.clickhouse_mode = "chdb"
            except Exception:
                s.clickhouse_mode = "memory"

    s.resolved = {
        "llm": s.llm_backend,
        "embeddings": s.embedding_backend,
        "search": s.search_backend,
        "etl": s.etl_backend,
        "database": s.clickhouse_mode,
    }
    return s


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Re-read environment (used by tests)."""
    global _settings
    _settings = None
