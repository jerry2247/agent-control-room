"""Pillar D: presupposition detection, counter-framing, and frame balance."""
import asyncio

from app.core import reframe, registry


def test_loaded_why_detects_presupposition():
    f = reframe.analyze("why does coffee cause cancer")
    assert f.type == "loaded_why"
    assert f.presupposition == "coffee cause cancer"
    assert "cause" not in f.neutral_topic          # directional verb neutralized
    assert "coffee" in f.neutral_topic and "cancer" in f.neutral_topic
    assert any("against" in c for c in f.counter_queries)
    assert any("alternative explanations" in c for c in f.counter_queries)
    assert f.affirm_queries and "supporting evidence" in f.affirm_queries[0]
    assert f.affirm_anchor and f.negate_anchor


def test_polar_question_frame():
    f = reframe.analyze("is nuclear energy safe")
    assert f.type == "polar"
    assert f.presupposition == "nuclear energy safe"
    assert any(" not safe" in c for c in f.counter_queries)


def test_comparative_frame_reverses_cleanly():
    f = reframe.analyze("is rust better than c++")
    assert f.type == "comparative"
    assert f.presupposition == "rust better than c++"
    assert any(c.startswith("c++ better than rust") for c in f.counter_queries)


def test_asserted_proof_frame():
    f = reframe.analyze("proof that vaccines cause autism")
    assert f.type == "asserted"
    assert f.presupposition == "vaccines cause autism"


def test_neutral_query_gets_no_probes():
    f = reframe.analyze("history of the roman empire")
    assert f.type == "none"
    assert f.counter_queries == [] and f.affirm_queries == []
    assert f.presupposition == ""


def test_full_pipeline_balances_loaded_frame(fresh_env):
    """The user's exact complaint: 'why does X do Y' must NOT return a corpus
    that argues X->Y. Baseline does (balance > 0); harness must not."""
    from app.agents.orchestrator import new_session_id, run_pipeline

    sid = new_session_id()
    registry.create(sid, {"original_query": "x"})
    final = asyncio.run(run_pipeline(sid, "why does coffee cause cancer", 5, 0.0))

    # dialectic pair guaranteed in the selection
    axes = [final["query_axes"][q] for q in final["approved_queries"]]
    assert "counter_frame" in axes
    assert "affirm_frame" in axes

    fb = final["metrics"]["frame_balance"]
    assert fb["presupposition"] == "coffee cause cancer"
    # plain single-query search argues the asked frame
    assert fb["baseline"] > 0.03
    # the harness corpus sits in the neutral band and is less tilted than baseline
    assert abs(fb["harness"]) < 0.08
    assert abs(fb["harness"]) < abs(fb["baseline"])
    assert fb["verdict"] == "balanced"


def test_neutral_query_skips_frame_metric(fresh_env):
    from app.agents.orchestrator import new_session_id, run_pipeline

    sid = new_session_id()
    registry.create(sid, {"original_query": "x"})
    final = asyncio.run(run_pipeline(sid, "history of the roman empire", 4, 0.0))
    assert final["metrics"].get("frame_balance") is None
    assert "counter_frame" not in final["query_axes"].values()
