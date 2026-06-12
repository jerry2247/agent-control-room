"""ClickHouse OLAP layer (Pillar B + C computed IN the database).

Engines (selected by CLICKHOUSE_MODE):
  chdb   : embedded ClickHouse (pip install chdb). Default for local dev/CI.
           Identical SQL dialect to ClickHouse Cloud.
  cloud  : ClickHouse Cloud / self-hosted via clickhouse-connect (HTTPS :8443).
  memory : pure-NumPy fallback, same interface (no SQL).

The judge-facing point: Semantic Spread (avg pairwise cosineDistance) and
Shannon Entropy are computed by SQL inside the OLAP store at request time,
not by application code. The exact SQL executed is returned alongside every
metrics payload so it can be shown live in the demo.
"""
from __future__ import annotations

import json
import re
import threading

import numpy as np

from app.core import metrics as np_metrics

_SID_RE = re.compile(r"^[a-f0-9]{8,32}$")

DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS scraped_documents (
    id UInt32,
    query_session_id String,
    control_group UInt8,
    source_query String,
    url String,
    domain String,
    ecosystem String,
    title String,
    chunk_index UInt16,
    content String,
    embedding Array(Float32),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree ORDER BY (query_session_id, id)
"""

DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS search_sessions (
    session_id String,
    original_query String,
    epsilon Float32,
    epsilon_mode String,
    n_queries UInt8,
    status String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree ORDER BY (created_at)
"""

SQL_SPREAD_CHUNKS = """
SELECT count() AS pairs, avg(cosineDistance(a.embedding, b.embedding)) AS spread
FROM scraped_documents AS a
INNER JOIN scraped_documents AS b ON a.query_session_id = b.query_session_id
WHERE a.query_session_id = '{sid}'
  AND a.control_group = {grp} AND b.control_group = {grp}
  AND a.id < b.id
"""

SQL_SPREAD_DOCS = """
WITH docs AS (
    SELECT url, avgForEach(embedding) AS e
    FROM scraped_documents
    WHERE query_session_id = '{sid}' AND control_group = {grp}
    GROUP BY url
)
SELECT count() AS pairs, avg(cosineDistance(a.e, b.e)) AS spread
FROM docs AS a INNER JOIN docs AS b ON a.url < b.url
"""

SQL_ENTROPY = """
SELECT -sum(p * log2(p)) AS entropy_bits, count() AS n_groups
FROM (
    SELECT {col}, count() AS c, sum(count()) OVER () AS total, c / total AS p
    FROM scraped_documents
    WHERE query_session_id = '{sid}' AND control_group = {grp}
    GROUP BY {col}
)
"""

SQL_DOMAIN_COUNTS = """
SELECT domain, count() AS chunks, uniqExact(url) AS docs
FROM scraped_documents
WHERE query_session_id = '{sid}' AND control_group = {grp}
GROUP BY domain ORDER BY chunks DESC
"""

SQL_FETCH = """
SELECT id, url, domain, ecosystem, title, control_group, source_query, content, embedding
FROM scraped_documents
WHERE query_session_id = '{sid}'
ORDER BY id
"""

# Pillar D: stance symmetry toward the presupposed claim.
# cos(e,a) - cos(e,n) == cosineDistance(e,n) - cosineDistance(e,a)
SQL_FRAME_BALANCE = """
SELECT
    avg(cosineDistance(embedding, {neg}) - cosineDistance(embedding, {aff})) AS balance,
    count() AS n
FROM scraped_documents
WHERE query_session_id = '{sid}' AND control_group = {grp}
"""


def _check_sid(sid: str) -> str:
    if not _SID_RE.match(sid):
        raise ValueError(f"invalid session id: {sid!r}")
    return sid


def _esc(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")


def _fmt_embedding(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


class _BaseSQLEngine:
    """Shared SQL logic for chdb and cloud engines."""

    name = "sql"

    def query_rows(self, sql: str) -> list[dict]:  # pragma: no cover - overridden
        raise NotImplementedError

    def execute(self, sql: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def ensure_schema(self) -> None:
        self.execute(DDL_DOCUMENTS)
        self.execute(DDL_SESSIONS)

    def insert_documents(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        values = []
        for r in rows:
            values.append(
                "({id},'{sid}',{grp},'{sq}','{url}','{dom}','{eco}','{title}',{ci},'{content}',{emb})".format(
                    id=int(r["id"]),
                    sid=_check_sid(r["query_session_id"]),
                    grp=int(r["control_group"]),
                    sq=_esc(r["source_query"])[:600],
                    url=_esc(r["url"])[:800],
                    dom=_esc(r["domain"])[:200],
                    eco=_esc(r["ecosystem"])[:60],
                    title=_esc(r["title"])[:400],
                    ci=int(r["chunk_index"]),
                    content=_esc(r["content"])[:4000],
                    emb=_fmt_embedding(r["embedding"]),
                )
            )
        self.execute(
            "INSERT INTO scraped_documents "
            "(id, query_session_id, control_group, source_query, url, domain, ecosystem, "
            "title, chunk_index, content, embedding) VALUES " + ",".join(values)
        )
        return len(rows)

    def insert_session(self, sid: str, query: str, epsilon: float, mode: str, n: int) -> None:
        self.execute(
            "INSERT INTO search_sessions (session_id, original_query, epsilon, epsilon_mode, "
            "n_queries, status) VALUES ('{sid}','{q}',{e},'{m}',{n},'running')".format(
                sid=_check_sid(sid), q=_esc(query)[:600], e=float(epsilon), m=_esc(mode), n=int(n)
            )
        )

    def corpus_metrics(self, sid: str, group: int) -> dict:
        sid = _check_sid(sid)
        grp = int(group)
        executed_sql = {}

        sql = SQL_SPREAD_DOCS.format(sid=sid, grp=grp)
        executed_sql["semantic_spread_docs"] = sql
        doc_rows = self.query_rows(sql)
        spread_docs = float(doc_rows[0]["spread"] or 0.0) if doc_rows and doc_rows[0]["pairs"] else 0.0

        sql = SQL_SPREAD_CHUNKS.format(sid=sid, grp=grp)
        executed_sql["semantic_spread_chunks"] = sql
        chunk_rows = self.query_rows(sql)
        spread_chunks = float(chunk_rows[0]["spread"] or 0.0) if chunk_rows and chunk_rows[0]["pairs"] else 0.0

        sql = SQL_ENTROPY.format(col="domain", sid=sid, grp=grp)
        executed_sql["domain_entropy"] = sql
        ent_rows = self.query_rows(sql)
        entropy_bits = float(ent_rows[0]["entropy_bits"] or 0.0) if ent_rows else 0.0
        n_domains = int(ent_rows[0]["n_groups"] or 0) if ent_rows else 0

        sql = SQL_ENTROPY.format(col="ecosystem", sid=sid, grp=grp)
        eco_rows = self.query_rows(sql)
        eco_entropy = float(eco_rows[0]["entropy_bits"] or 0.0) if eco_rows else 0.0

        domain_counts = self.query_rows(SQL_DOMAIN_COUNTS.format(sid=sid, grp=grp))
        n_docs = sum(int(r["docs"]) for r in domain_counts)
        n_chunks = sum(int(r["chunks"]) for r in domain_counts)
        norm = float(np.log2(n_domains)) if n_domains > 1 else 0.0

        return {
            "engine": self.name,
            "semantic_spread": round(spread_docs, 4),
            "semantic_spread_chunks": round(spread_chunks, 4),
            "shannon_entropy_bits": round(entropy_bits, 4),
            "normalized_entropy": round(entropy_bits / norm, 4) if norm else 0.0,
            "ecosystem_entropy_bits": round(eco_entropy, 4),
            "n_documents": n_docs,
            "n_chunks": n_chunks,
            "n_domains": n_domains,
            "domain_counts": [
                {"domain": r["domain"], "chunks": int(r["chunks"]), "docs": int(r["docs"])}
                for r in domain_counts
            ],
            "sql": executed_sql,
        }

    def fetch_documents(self, sid: str) -> list[dict]:
        rows = self.query_rows(SQL_FETCH.format(sid=_check_sid(sid)))
        for r in rows:
            if isinstance(r["embedding"], str):
                r["embedding"] = json.loads(r["embedding"])
            r["embedding"] = np.asarray(r["embedding"], dtype=np.float32)
            r["control_group"] = int(r["control_group"])
        return rows

    def frame_balance(self, sid: str, group: int, affirm: list, negate: list) -> dict:
        sql = SQL_FRAME_BALANCE.format(
            sid=_check_sid(sid), grp=int(group),
            aff=_fmt_embedding(affirm), neg=_fmt_embedding(negate),
        )
        rows = self.query_rows(sql)
        if not rows or not rows[0]["n"]:
            return {"balance": 0.0, "n": 0, "sql": sql}
        return {"balance": round(float(rows[0]["balance"] or 0.0), 4),
                "n": int(rows[0]["n"]), "sql": sql}


class ChdbEngine(_BaseSQLEngine):
    name = "chdb-embedded"

    def __init__(self, path: str):
        from chdb import session as chs

        self._sess = chs.Session(path)
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self._sess.close()
        except Exception:
            pass

    def execute(self, sql: str) -> None:
        with self._lock:
            self._sess.query(sql)

    def query_rows(self, sql: str) -> list[dict]:
        with self._lock:
            out = self._sess.query(sql, "JSON")
        payload = str(out)
        if not payload.strip():
            return []
        return json.loads(payload).get("data", [])


class CloudEngine(_BaseSQLEngine):
    name = "clickhouse-cloud"

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        import clickhouse_connect

        self._client = clickhouse_connect.get_client(
            host=host, port=port, username=user, password=password,
            database=database, secure=port == 8443,
        )
        self._lock = threading.Lock()

    def execute(self, sql: str) -> None:
        with self._lock:
            self._client.command(sql)

    def query_rows(self, sql: str) -> list[dict]:
        with self._lock:
            result = self._client.query(sql)
        cols = result.column_names
        return [dict(zip(cols, row)) for row in result.result_rows]


class MemoryEngine:
    """NumPy fallback with the same interface (no SQL available)."""

    name = "memory-numpy"

    def __init__(self):
        self._docs: list[dict] = []
        self._sessions: list[dict] = []

    def ensure_schema(self) -> None:
        pass

    def insert_session(self, sid, query, epsilon, mode, n) -> None:
        self._sessions.append({"session_id": sid, "original_query": query})

    def insert_documents(self, rows: list[dict]) -> int:
        self._docs.extend(rows)
        return len(rows)

    def fetch_documents(self, sid: str) -> list[dict]:
        out = []
        for r in self._docs:
            if r["query_session_id"] == sid:
                c = dict(r)
                c["embedding"] = np.asarray(r["embedding"], dtype=np.float32)
                out.append(c)
        return out

    def corpus_metrics(self, sid: str, group: int) -> dict:
        rows = [r for r in self._docs
                if r["query_session_id"] == sid and int(r["control_group"]) == int(group)]
        domains = [r["domain"] for r in rows]
        ecosystems = [r["ecosystem"] for r in rows]
        doc_embs = list(np_metrics.doc_level_embeddings(rows).values())
        chunk_embs = [np.asarray(r["embedding"]) for r in rows]
        urls = {r["url"] for r in rows}
        from collections import Counter
        dc = Counter(domains)
        return {
            "engine": self.name,
            "semantic_spread": round(np_metrics.semantic_spread(doc_embs), 4),
            "semantic_spread_chunks": round(np_metrics.semantic_spread(chunk_embs), 4),
            "shannon_entropy_bits": round(np_metrics.shannon_entropy(domains), 4),
            "normalized_entropy": round(np_metrics.normalized_entropy(domains), 4),
            "ecosystem_entropy_bits": round(np_metrics.shannon_entropy(ecosystems), 4),
            "n_documents": len(urls),
            "n_chunks": len(rows),
            "n_domains": len(dc),
            "domain_counts": [{"domain": d, "chunks": c, "docs": c} for d, c in dc.most_common()],
            "sql": {"note": "memory engine: metrics computed in NumPy, no SQL available"},
        }

    def frame_balance(self, sid: str, group: int, affirm: list, negate: list) -> dict:
        rows = [r for r in self._docs
                if r["query_session_id"] == sid and int(r["control_group"]) == int(group)]
        if not rows:
            return {"balance": 0.0, "n": 0, "sql": "memory engine (NumPy)"}
        a = np.asarray(affirm, dtype=np.float64)
        nv = np.asarray(negate, dtype=np.float64)
        a /= (np.linalg.norm(a) or 1.0)
        nv /= (np.linalg.norm(nv) or 1.0)
        scores = []
        for r in rows:
            e = np.asarray(r["embedding"], dtype=np.float64)
            e /= (np.linalg.norm(e) or 1.0)
            scores.append(float(e @ a) - float(e @ nv))
        return {"balance": round(float(np.mean(scores)), 4), "n": len(rows),
                "sql": "memory engine (NumPy)"}


def build_database(settings):
    mode = settings.clickhouse_mode
    if mode == "cloud":
        return CloudEngine(
            settings.clickhouse_host, settings.clickhouse_port, settings.clickhouse_user,
            settings.clickhouse_password, settings.clickhouse_database,
        )
    if mode == "chdb":
        return ChdbEngine(settings.chdb_path)
    return MemoryEngine()
