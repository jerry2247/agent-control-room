# Parallax

Parallax turns one loaded search prompt into a balanced, multi-perspective research corpus. It detects the premise hidden inside a question, rewrites the search space around the neutral topic, fans out across opposing angles, and measures whether the final corpus is actually broader than a normal single-query search.

Built by Jerry, Jack, Adam, and Ritwin for a hackathon.

We used the sponsor stack where it naturally fit the product: TrueFoundry for optional LLM gateway calls, Composio for live search execution, Airbyte for the durable ETL path, ClickHouse for OLAP metrics, and Render for deployment. The project also runs without keys for local demos and CI, and `/api/health` reports the active backend choices.

## What It Does

- Finds loaded framing in questions like `why does coffee cause cancer` or `is nuclear energy safe`.
- Recenters the work on the neutral topic before generating new search queries.
- Selects perspective-shifted queries that stay close enough to the original topic while spreading across different viewpoints.
- Adds affirming and counter-frame probes so the corpus does not only argue the user's premise.
- Runs the approved queries, normalizes results, embeds chunks, and compares the harness corpus against a plain baseline search.
- Computes semantic spread, domain entropy, ecosystem entropy, and frame balance in ClickHouse when a SQL backend is active.
- Serves a single-page dashboard with the pipeline state, query choices, metrics, SQL, source domains, clusters, and conflict pairs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python scripts/demo.py "is nuclear energy safe"
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard.

The default local mode uses deterministic query generation, mock search results, inline ETL, and embedded ClickHouse through `chdb` when available. If `chdb` is unavailable, the app falls back to an in-memory NumPy engine and reports that backend in `/api/health`.

## Demo Flow

1. Start the app and open the dashboard.
2. Run a loaded query, for example `why does coffee cause cancer`.
3. Point out the detected premise and the neutral topic.
4. Show the approved query set, especially the affirm-frame and counter-frame probes.
5. Compare harness metrics against the baseline search.
6. Open the SQL panel to show the ClickHouse query used for the metric.
7. End with the conflict view, which surfaces opposing clusters from the resulting corpus.

For a live sponsor demo, configure the relevant environment variables from `.env.example` and verify `/api/health` before presenting.

## How It Works

```text
User query
  -> ingress guardrails
  -> variance engine
  -> critic
  -> search execution
  -> ETL and chunking
  -> ClickHouse storage and metrics
  -> synthesis and dashboard
```

The variance engine embeds candidate queries as unit vectors and selects the set with the largest pairwise spread while enforcing an epsilon radius around the neutralized topic. Auto epsilon sweeps the candidate space and chooses a radius that captures most of the available diversity without drifting away from the topic.

For loaded queries, the generator also injects dialectic probes. One probe tests the user's premise, and another looks for counter-evidence or alternative explanations. The critic checks safety, topicality, and duplicate queries before any search tool runs.

Frame balance is the key output for premise neutrality. A baseline search usually leans toward the wording of the original question. The harness aims to move that score closer to zero by collecting evidence on both sides of the premise.

## Sponsor Integrations

| Sponsor | Where it fits | What the code does |
| --- | --- | --- |
| TrueFoundry | LLM gateway | Uses an OpenAI-compatible chat endpoint for generation, critic checks, and synthesis when `TRUEFOUNDRY_*` is configured. |
| Composio | Search execution | Executes the configured Composio search tool for each approved query when `COMPOSIO_API_KEY` is present. |
| Airbyte | Durable ETL path | Stages scraped records as JSONL and triggers a configured Airbyte sync when `ETL_BACKEND=airbyte`. Inline ETL still runs for demo latency. |
| ClickHouse | Metrics engine | Stores documents and computes spread, entropy, and frame balance through SQL in `chdb` or ClickHouse Cloud. |
| Render | Hosting | Uses `render.yaml` to deploy the FastAPI service and serve the dashboard. |

All sponsor integrations are optional. The app starts locally without credentials, and `/api/health` reports which backend is active for each stage.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/search` | Start a run. Use `sync: true` for a complete response in one request. |
| `GET /api/session/{id}` | Read live session state, selected queries, metrics, clusters, and warnings. |
| `GET /api/sessions` | List recent in-process sessions. |
| `GET /api/metrics/{id}` | Recompute corpus metrics from the active database backend. |
| `POST /api/epsilon/optimize` | Run only the epsilon sweep and candidate selection preview. |
| `GET /api/health` | Show service status and active backend choices. |

Example:

```bash
curl -s http://localhost:8000/api/search \
  -H 'content-type: application/json' \
  -d '{"query": "is nuclear energy safe", "epsilon": "auto", "sync": true}' \
  | python -m json.tool
```

## Configuration

Copy `.env.example` into your shell or deployment environment and fill in only the services you want to use.

Key switches:

| Variable | Effect |
| --- | --- |
| `TRUEFOUNDRY_BASE_URL`, `TRUEFOUNDRY_API_KEY`, `TRUEFOUNDRY_CHAT_MODEL` | Use TrueFoundry for LLM calls. |
| `ANTHROPIC_API_KEY` | Use direct Anthropic calls if TrueFoundry is not configured. |
| `COMPOSIO_API_KEY` | Use Composio instead of mock search. |
| `AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_CONNECTION_ID`, `ETL_BACKEND=airbyte` | Trigger Airbyte syncs in addition to inline ETL. |
| `CLICKHOUSE_HOST`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_MODE=cloud` | Use ClickHouse Cloud instead of embedded `chdb`. |
| `EMBEDDING_BACKEND=api`, `EMBEDDING_API_BASE`, `EMBEDDING_MODEL` | Use an OpenAI-compatible embedding endpoint instead of local hash embeddings. |

## Repository Structure

```text
app/
  main.py                  FastAPI entrypoint and dashboard route
  config.py                Environment-driven backend selection
  deps.py                  Lazy backend singletons
  api/routes.py            Search, session, metrics, epsilon, and health endpoints
  agents/
    generator.py           Perspective query generation and dialectic probe injection
    critic.py              Safety, relevance, and deduplication audit
    orchestrator.py        LangGraph pipeline from ingress through synthesis
    state.py               Shared pipeline state
  core/
    divergence.py          Epsilon sweep and max-dispersion selection
    reframe.py             Presupposition detection and neutral topic extraction
    embeddings.py          Local hash embeddings or API embeddings
    metrics.py             NumPy metric reference implementation
    synthesis.py           Clustering, conflict pairs, and scatter data
    textproc.py            Query sanitation, HTML cleanup, chunking, source labeling
  services/
    truefoundry_client.py  TrueFoundry and Anthropic chat client
    composio_client.py     Composio search client and deterministic mock backend
    airbyte_client.py      Inline ETL plus optional Airbyte sync trigger
  database/
    clickhouse_client.py   chdb, ClickHouse Cloud, and memory database engines
dashboard/index.html       Demo dashboard
scripts/demo.py            Terminal demo
tests/                     Unit, API, SQL, and end-to-end tests
DEPLOYMENT.md              Deployment and demo guide
render.yaml                Render Blueprint
```

## Development

```bash
pytest
python scripts/demo.py "why does coffee cause cancer" --json
uvicorn app.main:app --reload
```

The tests are designed to pass without external credentials. Mock search data is synthetic and labeled as mock output. Use Composio and provider keys when you want live web results.
