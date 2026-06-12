# Orthogonal Search Harness

One biased query in. A verified, multi-perspective corpus out, with the diversity gain AND the premise-neutrality of the corpus proven live by SQL inside an OLAP (Online Analytical Processing) database.

LLM search agents inherit the user's framing twice. First, where they look: a single query lands in one media ecosystem. Second, and deeper, what they ask: "why does X do Y" presupposes X->Y, so every retrieved document argues inside the user's premise no matter how diverse the sources. The harness breaks both loops mechanically. It detects and neutralizes the presupposition, expands the query into N mathematically orthogonal queries (including dialectic probes that test the premise itself) under a topical-relevance constraint, executes them concurrently against the live web, normalizes everything into ClickHouse, and benchmarks the result against a plain single-query baseline in real time.

## Pipeline

```
[ User Input ]
      |
      v
1. INGRESS & GUARDRAILS ............ TrueFoundry LLM Gateway (+ local sanitizer)
      |
      v
2. VARIANCE ENGINE ................. LangGraph state machine (cyclic: critic
      |   - frame analysis: detect       feedback re-enters the generator)
      |     presupposition P, recenter
      |     on the neutralized topic
      |   - dialectic probes: affirm-P
      |     and counter-P queries with
      |     guaranteed selection slots
      |   - epsilon-constrained max-
      |     dispersion selection
      v
3. AUTONOMOUS SEARCH EXECUTION ..... Composio tool execution, concurrent fan-out
      |
      v
4. ETL PIPELINE .................... Airbyte connection sync (+ inline scraper)
      |
      v
5. OLAP STORAGE & METRICS .......... ClickHouse: cosineDistance, entropy, and
      |                              frame-balance stance symmetry, all in SQL
      v
6. SYNTHESIS & DASHBOARD ........... viewpoint clusters, conflict pairs, live charts
```

## Mathematical foundation

**A. Constrained Query Divergence.** Candidate queries are embedded as unit vectors. We select the subset maximizing pairwise squared distance, subject to every query staying inside an epsilon-ball around the original query embedding c:

```
max  sum_{i<j} ||x_i - x_j||^2    s.t.  ||x_i - c||_2 < epsilon
```

Implementation detail worth judging: for unit vectors, `sum_{i<j} ||x_i - x_j||^2 = n^2 - ||sum_i x_i||^2`, so maximizing dispersion is exactly minimizing the resultant vector length ("balancing forces"). That identity makes the objective O(d) per subset, letting us brute-force the true optimum for hackathon-scale pools and fall back to greedy + 2-swap above `C(m, n) > 30k`. See `app/core/divergence.py` and `tests/test_divergence.py` (the identity is verified against the naive O(n^2 d) sum).

Epsilon is **user-customizable** (slider / API field) and **optimizable**: auto mode sweeps the radius, builds the diversity-vs-epsilon curve, and picks the knee (smallest epsilon capturing >= 95% of attainable diversity, i.e. minimal topical drift for near-maximal viewpoint spread). The curve renders in the dashboard.

**B. Real-Time Semantic Spread.** Average pairwise cosine distance over the scraped corpus, computed by ClickHouse itself:

```
Spread = 2 / (K(K-1)) * sum_{i<j} (1 - cos(e_i, e_j))
```

**C. Structural Information Entropy.** Shannon entropy over source domains and over media-ecosystem classes (government, academic, mainstream news, community, independent, ...):

```
H(S) = -sum_x p(x) log2 p(x)
```

Both B and C run as SQL (`cosineDistance`, window functions, `log2`) against `scraped_documents`, for the harness corpus and for a single-query control group, and the dashboard shows the lift. The exact SQL executed is returned in every metrics payload (`metrics.harness.sql`) so it can be displayed live.

**D. Frame Balance (presupposition debiasing).** The deepest bias is not where you look but what you ask: "why does X do Y" presupposes X->Y, so every result argues inside the user's frame regardless of source diversity. The harness (1) detects the loaded frame and extracts the presupposed claim P (loaded-why, asserted, comparative, causal, and polar patterns), (2) recenters the epsilon-ball on the NEUTRALIZED topic core (directional verbs and stance scaffolding stripped), so divergence is constrained around the topic rather than the user's framing, (3) injects a dialectic probe pair (counter-frame: negation, reversal, alternative explanations; affirm-frame: the claim itself) with guaranteed selection slots via min-loss swaps under the resultant objective, and (4) proves the result with a stance-symmetry metric computed in ClickHouse:

```
balance = mean_e [ cos(e, a) - cos(e, n) ]
```

where `a` embeds P plus confirmation vocabulary and `n` embeds P plus refutation vocabulary. balance >> 0 means the corpus argues the asked frame; ~0 means evidence for and against P is symmetrically represented. Measured on the Tier-0 corpus: plain search lands at +0.07 to +0.19 (it argues the frame the user asked in); the harness lands inside the +-0.08 neutral band on every loaded query tested, a 45-90% bias reduction. See `app/core/reframe.py` and `tests/test_reframe.py`.

## Quickstart (zero keys required)

```bash
pip install -r requirements-dev.txt
pytest                                   # 37 tests: math, SQL-vs-NumPy, agents, frame debiasing, e2e, API
python scripts/demo.py "why does coffee cause cancer"
uvicorn app.main:app --reload            # dashboard at http://localhost:8000
```

Tier 0 runs everything locally: deterministic template generator, mock search corpus, inline ETL, embedded ClickHouse (chdb). Mock mode exists for development and CI. Judged demos should run with real backends (see `DEPLOYMENT.md`).

Example Tier-0 output (`python scripts/demo.py "is nuclear energy safe"`):

```
frame   : polar | presupposes 'nuclear energy safe'
          neutral center: 'nuclear energy'

--- approved orthogonal queries ---
  d=1.0206  nuclear energy criticism risks failures documented problems
  d=0.9771  nuclear energy safe supporting evidence reasons documented [AFFIRM]
  d=1.0532  nuclear energy international comparison other countries global approach
  d=1.0656  nuclear energy case study firsthand account community experience
  d=1.072   nuclear energy safe debunked fact check no link [COUNTER]

  metric                         harness  baseline      lift
  semantic spread                 0.4234     0.087   +386.7%
  domain entropy (bits)           3.2929       1.0   +229.3%
  ecosystem entropy (bits)        1.8513       1.0    +85.1%
  unique domains                      10         2   +400.0%
  frame balance (0=unbiased)     -0.0240   +0.1712  balanced
```

The last row is the point of the project: the plain single-query baseline argues the question's own premise (+0.17); the harness corpus is statistically neutral on it (-0.02).

## API

| Endpoint | Description |
| --- | --- |
| `POST /api/search` | `{query, n_queries (2-8), epsilon (number or "auto"), include_baseline, sync}` -> session id (async) or full result (sync) |
| `GET /api/session/{id}` | Live pipeline state: stage, queries with distances and axes, rejections with reasons, frame analysis, metrics, clusters, scatter |
| `GET /api/metrics/{id}` | Recomputed at request time inside ClickHouse (live OLAP query) |
| `POST /api/epsilon/optimize` | Variance-engine-only sweep: diversity curve + recommended epsilon |
| `GET /api/health` | Active backend per pipeline stage |

Key response fields beyond the corpus itself: `frame` (detected presupposition type, claim P, neutral topic, generated probes), `query_axes` (which approved query plays which role, including `affirm_frame` / `counter_frame`), `metrics.frame_balance` (`harness`, `baseline`, `bias_reduction_pct`, `verdict`), `epsilon_used` / `epsilon_curve`, and `metrics.harness.sql` (the exact SQL ClickHouse executed). Notes: use `n_queries >= 3` for loaded queries so the dialectic pair does not crowd out lens diversity, and prefer `epsilon: "auto"` (a hand-set radius is respected strictly and can be too tight to admit the dialectic probes; auto widens just enough and logs it).

```bash
curl -s localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"query": "is nuclear energy safe", "epsilon": "auto", "sync": true}' | python3 -m json.tool
```

## Repository structure

```
app/
  main.py                  FastAPI entrypoint, serves dashboard
  config.py                env-driven backend selection (all sponsors optional)
  deps.py                  lazy singletons
  api/routes.py            /api/search, /api/session, /api/metrics, /api/epsilon/optimize
  agents/
    state.py               LangGraph HarnessState
    generator.py           variance queries: LLM or perspective templates, dialectic probe
                           injection with guaranteed slots, dispersion optimizer
    critic.py              pre-execution audit: safety blocklist, radius re-check vs the
                           neutral center, dedup, LLM veto
    orchestrator.py        compiled cyclic StateGraph + run_pipeline
  core/
    embeddings.py          local deterministic embedder | OpenAI-compatible API
    divergence.py          Pillar A: constrained max-dispersion + epsilon auto-tuner
    reframe.py             Pillar D: presupposition detection, query neutralization,
                           counter/affirm probes, stance anchors
    metrics.py             NumPy ground truth for Pillars B/C (cross-checks SQL in tests)
    synthesis.py           k-means viewpoint clusters, conflict pairs, PCA scatter
    textproc.py            ingress guard, HTML->markdown, chunking, ecosystem classifier
  services/
    truefoundry_client.py  gateway/Anthropic LLM client
    composio_client.py     Composio tool execution (SDK or REST) + mock backend
    airbyte_client.py      inline ETL + Airbyte staging/sync trigger
  database/
    clickhouse_client.py   chdb embedded | ClickHouse Cloud | memory; metrics SQL
dashboard/index.html       single-file dashboard (Chart.js)
scripts/demo.py            CLI end-to-end demo
tests/                     37 tests incl. SQL==NumPy cross-validation and frame-balance e2e
render.yaml                Render Blueprint (IaC)
DEPLOYMENT.md              sponsor-by-sponsor integration guide + demo runbook
```

## Sponsor integration matrix

| Sponsor | Pipeline placement | What it does here | Env switch |
| --- | --- | --- | --- |
| TrueFoundry | LLM gateway for generator/critic/synthesis | guardrails, rate limits, fallback, cost tracking on every LLM call | `TRUEFOUNDRY_*` |
| Composio | search tool execution | concurrent live web searches via tool platform (SDK or REST) | `COMPOSIO_API_KEY` |
| Airbyte | ETL of record | staged JSONL + programmatic connection sync into ClickHouse | `AIRBYTE_*`, `ETL_BACKEND=airbyte` |
| ClickHouse | OLAP store + metric engine | stores chunks/embeddings; computes spread, entropy, and frame balance in SQL | `CLICKHOUSE_*` |
| Render | hosting | blueprint deploy, health checks, public demo URL | `render.yaml` |

## Honest limitations

Local embeddings are lexical (hash projection), good enough to drive the math and tests; switch `EMBEDDING_BACKEND=api` for semantic-grade vectors. Mock search is synthetic and labeled as such. Frame detection is pattern-based English heuristics covering five loaded-frame shapes (loaded-why, asserted, comparative, causal, polar); enabling an LLM backend extends coverage to arbitrary phrasings, while the deterministic probes and the frame-balance metric apply in both modes. The verdict band (|balance| < 0.08 = balanced) is calibrated for the local embedder; recalibrate against your embedding model if you change it. The Airbyte path triggers a real sync but the inline ETL remains the latency path for live demos. Session progress state is in-process (single worker); durable artifacts live in ClickHouse.
