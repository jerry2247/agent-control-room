import pathlib
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@pytest.fixture()
def fresh_env(monkeypatch):
    """Force Tier-0 backends with an isolated embedded ClickHouse per test.

    The chdb data dir lives OUTSIDE pytest's tmp_path (chdb store layouts break
    pytest's managed teardown); we close the session and best-effort remove it.
    """
    for var in [
        "ANTHROPIC_API_KEY", "TRUEFOUNDRY_API_KEY", "TRUEFOUNDRY_BASE_URL",
        "COMPOSIO_API_KEY", "AIRBYTE_CLIENT_ID", "CLICKHOUSE_HOST",
        "LLM_BACKEND", "SEARCH_BACKEND", "ETL_BACKEND", "EMBEDDING_BACKEND",
    ]:
        monkeypatch.delenv(var, raising=False)
    chdb_dir = tempfile.mkdtemp(prefix="osh_chdb_")
    monkeypatch.setenv("CLICKHOUSE_MODE", "chdb")
    monkeypatch.setenv("CHDB_PATH", chdb_dir)
    from app import deps
    deps.reset()
    yield deps
    deps.reset()
    shutil.rmtree(chdb_dir, ignore_errors=True)
