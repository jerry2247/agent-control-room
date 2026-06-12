"""Cross-check: ClickHouse SQL metrics == NumPy reference implementations."""
import math

import numpy as np
import pytest

from app.core import metrics as np_metrics

pytest.importorskip("chdb")


def _insert_corpus(db, sid: str):
    rng = np.random.default_rng(42)
    domains = ["a.example", "a.example", "b.example", "c.example", "d.example",
               "b.example", "a.example", "e.example"]
    rows = []
    for i, dom in enumerate(domains):
        v = rng.standard_normal(24)
        v /= np.linalg.norm(v)
        rows.append({
            "id": i, "query_session_id": sid, "control_group": 0,
            "source_query": "q", "url": f"https://{dom}/doc{i}", "domain": dom,
            "ecosystem": "other_web", "title": f"doc {i}", "chunk_index": 0,
            "content": f"content {i}", "embedding": [float(x) for x in v],
        })
    db.insert_documents(rows)
    return rows


def test_sql_semantic_spread_matches_numpy(fresh_env):
    db = fresh_env.get_db()
    assert db.name == "chdb-embedded"
    sid = "ab12cd34ef56"
    rows = _insert_corpus(db, sid)
    got = db.corpus_metrics(sid, 0)
    embs = [np.asarray(r["embedding"]) for r in rows]
    expected_chunks = np_metrics.semantic_spread(embs)
    assert got["semantic_spread_chunks"] == pytest.approx(expected_chunks, abs=2e-3)
    doc_embs = list(np_metrics.doc_level_embeddings(rows).values())
    expected_docs = np_metrics.semantic_spread(doc_embs)
    assert got["semantic_spread"] == pytest.approx(expected_docs, abs=2e-3)


def test_sql_entropy_matches_numpy_and_known_values(fresh_env):
    db = fresh_env.get_db()
    sid = "ab12cd34ef56"
    rows = _insert_corpus(db, sid)
    got = db.corpus_metrics(sid, 0)
    domains = [r["domain"] for r in rows]
    assert got["shannon_entropy_bits"] == pytest.approx(
        np_metrics.shannon_entropy(domains), abs=1e-4   # client rounds to 4 dp
    )
    # known value: distribution {3,2,1,1,1} over 8 docs
    expected = -(3 / 8 * math.log2(3 / 8) + 2 / 8 * math.log2(2 / 8) + 3 * (1 / 8) * math.log2(1 / 8))
    assert got["shannon_entropy_bits"] == pytest.approx(expected, abs=1e-4)
    assert got["n_domains"] == 5


def test_uniform_and_degenerate_entropy(fresh_env):
    db = fresh_env.get_db()
    # 4 domains, perfectly uniform -> exactly 2 bits
    rows = []
    for i in range(4):
        rows.append({
            "id": i, "query_session_id": "ffff0000aaaa", "control_group": 0,
            "source_query": "q", "url": f"https://d{i}.example/x", "domain": f"d{i}.example",
            "ecosystem": "other_web", "title": "t", "chunk_index": 0,
            "content": "c", "embedding": [1.0, 0.0],
        })
    db.insert_documents(rows)
    m = db.corpus_metrics("ffff0000aaaa", 0)
    assert m["shannon_entropy_bits"] == pytest.approx(2.0, abs=1e-9)
    assert m["normalized_entropy"] == pytest.approx(1.0, abs=1e-6)

    # single domain -> 0 bits
    db.insert_documents([{
        "id": 0, "query_session_id": "eeee1111bbbb", "control_group": 0,
        "source_query": "q", "url": "https://one.example/x", "domain": "one.example",
        "ecosystem": "other_web", "title": "t", "chunk_index": 0,
        "content": "c", "embedding": [1.0, 0.0],
    }])
    m = db.corpus_metrics("eeee1111bbbb", 0)
    assert m["shannon_entropy_bits"] == 0.0


def test_identical_vs_orthogonal_spread(fresh_env):
    db = fresh_env.get_db()
    # identical embeddings -> spread 0; orthogonal pair -> spread 1
    rows = []
    for i in range(3):
        rows.append({
            "id": i, "query_session_id": "1234567890ab", "control_group": 0,
            "source_query": "q", "url": f"https://s{i}.example/x", "domain": f"s{i}.example",
            "ecosystem": "other_web", "title": "t", "chunk_index": 0,
            "content": "c", "embedding": [1.0, 0.0],
        })
    db.insert_documents(rows)
    assert db.corpus_metrics("1234567890ab", 0)["semantic_spread"] == pytest.approx(0.0, abs=1e-6)

    db.insert_documents([
        {"id": 0, "query_session_id": "ba0987654321", "control_group": 0, "source_query": "q",
         "url": "https://x.example/1", "domain": "x.example", "ecosystem": "o", "title": "t",
         "chunk_index": 0, "content": "c", "embedding": [1.0, 0.0]},
        {"id": 1, "query_session_id": "ba0987654321", "control_group": 0, "source_query": "q",
         "url": "https://y.example/2", "domain": "y.example", "ecosystem": "o", "title": "t",
         "chunk_index": 0, "content": "c", "embedding": [0.0, 1.0]},
    ])
    assert db.corpus_metrics("ba0987654321", 0)["semantic_spread"] == pytest.approx(1.0, abs=1e-6)


def test_sql_injection_guard(fresh_env):
    db = fresh_env.get_db()
    with pytest.raises(ValueError):
        db.corpus_metrics("x'; DROP TABLE scraped_documents; --", 0)
