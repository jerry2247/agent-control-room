# Deployment & Sponsor Integration Guide

The harness boots with zero keys (Tier 0: local generator, mock search, inline ETL, embedded ClickHouse). Each sponsor switches on independently via environment variables, so integrate in any order and verify after each step with `GET /api/health`. Recommended order: Render first (so you have a URL), then ClickHouse, Composio, TrueFoundry, Airbyte.

Master switchboard:

| Variable | Effect |
| --- | --- |
| `COMPOSIO_API_KEY` | search backend flips mock -> composio |
| `TRUEFOUNDRY_BASE_URL` + `TRUEFOUNDRY_API_KEY` + `TRUEFOUNDRY_CHAT_MODEL` | LLM flips local -> truefoundry |
| `ANTHROPIC_API_KEY` | LLM flips local -> anthropic (if TrueFoundry not set) |
| `CLICKHOUSE_HOST` + `CLICKHOUSE_PASSWORD` (+ `CLICKHOUSE_MODE=cloud`) | DB flips chdb -> ClickHouse Cloud |
| `AIRBYTE_CLIENT_ID/SECRET/CONNECTION_ID` + `ETL_BACKEND=airbyte` | ETL stages records + triggers syncs |
| `EMBEDDING_API_BASE` + `EMBEDDING_MODEL` (+ key) | embeddings flip local -> API |

---

## 1. Render (hosting, ~10 min)

1. Push this repo to GitHub.
2. Render Dashboard -> **New** -> **Blueprint** -> select the repo. Render reads `render.yaml` (service type web, `runtime: python`, uvicorn start command, health check on `/api/health`).
3. First deploy works with no keys (Tier 0). Add sponsor env vars under **Environment** as you complete the steps below; each save triggers a redeploy.
4. Verify: `curl https://<your-app>.onrender.com/api/health` returns `"status": "ok"` plus the active backend per stage. Open the root URL for the dashboard.

Notes: keep `--workers 1` (in-process session registry). The starter instance is fine for `CLICKHOUSE_MODE=cloud` or `memory`; bump to standard if you insist on embedded chdb in production. Free-tier instances sleep, so use a paid instance or warm it before the demo.

## 2. ClickHouse Cloud (OLAP metrics, ~10 min)

1. Create a service at https://clickhouse.cloud (trial credits are fine). Region close to your Render region.
2. From the service Connect panel copy: host (`<service>.<region>.clickhouse.cloud`), port 8443, user `default`, password.
3. Set on Render: `CLICKHOUSE_HOST`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_MODE=cloud` (user/port/database defaults are fine). Tables are created automatically on boot.
4. Verify: `/api/health` shows `"database_engine": "clickhouse-cloud"`. Run one search, then in the ClickHouse Cloud SQL console:

```sql
SELECT query_session_id, count() AS chunks, uniqExact(domain) AS domains
FROM scraped_documents GROUP BY query_session_id ORDER BY max(created_at) DESC LIMIT 5;
```

Demo moment: paste the spread SQL from the dashboard's "View the SQL" panel into the Cloud console and run it live. Same numbers as the UI.

Local dev needs nothing: `chdb` (embedded ClickHouse) ships in requirements and speaks the identical SQL dialect.

## 3. Composio (live search execution, ~15 min)

1. Sign up at https://composio.dev, create a project, copy the API key from the dashboard.
2. Enable the **Composio Search** toolkit (managed auth; no Google/Serp keys needed). In the dashboard check the exact search tool slug under the toolkit's Tools tab.
3. Set on Render: `COMPOSIO_API_KEY` (and `COMPOSIO_TOOL_SLUG` if your slug differs from the default `COMPOSIO_SEARCH_SEARCH`).
4. Verify: `/api/health` shows `"search": "composio"`. Run a search with `"sync": true` and confirm `documents_summary` contains real domains instead of `*.example`.

Implementation: `app/services/composio_client.py` uses the official SDK if installed (`pip install composio`, then `composio.tools.execute(slug, user_id=..., arguments={"query": ...})`) and otherwise falls back to the raw REST endpoint `POST {base}/api/v3/tools/execute/{slug}` with the `x-api-key` header, so no extra dependency is required. Result parsing is shape-tolerant (`results` / `organic_results` / `items`).

## 4. TrueFoundry (LLM gateway: guardrails, fallback, cost control, ~15 min)

1. In your TrueFoundry account open **AI Gateway**, add a provider account (e.g. your Anthropic or OpenAI key) under Integrations.
2. From the gateway playground's code snippet copy the OpenAI-compatible base URL (looks like `https://<org>.truefoundry.cloud/api/llm`) and generate a Personal Access Token / API key.
3. Set on Render: `TRUEFOUNDRY_BASE_URL`, `TRUEFOUNDRY_API_KEY`, `TRUEFOUNDRY_CHAT_MODEL` (the gateway model id, e.g. `anthropic-main/claude-sonnet-4-6`).
4. Optional but high-value for judging: in the gateway console configure rate limits, fallback models, and guardrail policies; they now apply to every generator/critic/synthesis call without code changes. Show the gateway metrics page during the demo (request log, latency, cost per call).
5. Optional embeddings upgrade through the same gateway: `EMBEDDING_BACKEND=api`, `EMBEDDING_API_BASE=$TRUEFOUNDRY_BASE_URL`, `EMBEDDING_API_KEY=$TRUEFOUNDRY_API_KEY`, `EMBEDDING_MODEL=openai-main/text-embedding-3-small`.
6. Verify: `/api/health` shows `"llm": "truefoundry"`; generated queries become noticeably more natural than the template fallback, and a `narrative` field appears in synthesis.

No TrueFoundry account? `ANTHROPIC_API_KEY` alone gives the same LLM features direct (LLM_BACKEND auto-switches to anthropic); you lose the gateway governance story.

## 5. Airbyte (durable ETL of record, ~30 min, optional tier)

The inline ETL always runs (fetch -> clean -> chunk -> embed), so the demo never blocks on a sync. Airbyte mode adds the production data-engineering story: every scraped record is also staged as JSONL and a real Airbyte connection sync into ClickHouse is triggered programmatically per session.

1. Airbyte Cloud (https://cloud.airbyte.com): user Settings -> **Applications** -> Create application -> copy `client_id` and `client_secret`.
2. Create a connection: source = file-based source pointed at your staging bucket (recommended: set `AIRBYTE_STAGING_DIR` to a mounted/S3-synced path and use the S3 source), destination = ClickHouse (host/port/user/password from step 2), table `airbyte_scraped_raw`. Copy the connection id from the connection URL.
3. Set on Render: `AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_CONNECTION_ID`, `ETL_BACKEND=airbyte`.
4. Verify: run a search; the session payload gains `airbyte_job: {jobId, status}` and the job appears in the Airbyte UI. Token exchange (`POST /v1/applications/token`, grant-type client_credentials, 60-minute bearer) and job trigger (`POST /v1/jobs` with `{connectionId, jobType: "sync"}`) are implemented in `app/services/airbyte_client.py`.

If sync trigger fails the pipeline continues and logs a warning; you keep the demo.

---

## Demo-day runbook (3 minutes)

Before going on stage: hit the Render URL once to warm it; run one query end-to-end; keep a ClickHouse Cloud SQL console tab open with the spread query pasted.

1. 0:00 Frame the problem: "every search agent inherits your bias twice: where it looks, and what the question already assumes. Ask 'why does X do Y' and you get pages arguing X->Y. We measure and remove both."
2. 0:20 Type a loaded query ("why does coffee cause cancer"), epsilon on auto, hit RUN. Point at the presupposition banner the moment it appears ("the system caught the hidden claim and recentered on the neutral topic"), then narrate the stages as they tick: guardrails, variance engine with the [affirm-frame] and [counter-frame] probes, critic, concurrent Composio searches, ETL, ClickHouse.
3. 1:10 Metrics cards land: read the lifts out loud ("+170% semantic spread, +2 bits of source entropy vs the exact same question asked normally"). Point at the Frame Balance card: "the plain search argued the question's own premise at +0.17; our corpus sits at -0.02, statistically neutral." Open the epsilon curve: "the system solved for how far it could diverge without leaving the topic."
4. 1:50 Show the conflict panel: two clusters quoting opposite findings, side by side.
5. 2:20 Switch to the ClickHouse console, run the SQL, same number appears: "the metric is computed inside the OLAP store, not in our app code."
6. 2:45 Architecture one-liner: five sponsor systems, each owning one pipeline stage, all hot-swappable by env var.

Contingencies: venue wifi dies mid-demo -> set `SEARCH_BACKEND=mock` (or run locally `uvicorn app.main:app`), everything else identical and the UI labels the backend honestly. Composio quota exhausted -> same fallback. Render cold start -> you warmed it.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/api/health` shows `database_engine: memory-numpy` | chdb failed to import (rare) or `CLICKHOUSE_MODE` forced; metrics still correct via NumPy, set ClickHouse Cloud creds for the SQL story |
| Composio results empty | check tool slug in the Composio dashboard; set `COMPOSIO_TOOL_SLUG`; confirm toolkit enabled for your project |
| TrueFoundry 401/404 | base URL must be the gateway root that serves `/chat/completions`; regenerate the PAT; confirm model id string from the playground snippet |
| Airbyte 401 | tokens expire after 60 min; the client re-fetches automatically, so check client id/secret values |
| OOM on Render starter | set `CLICKHOUSE_MODE=cloud` (or `memory`) instead of embedded chdb, or upgrade the plan |
| Queries all rejected | epsilon too tight for local embedding geometry; use `"epsilon": "auto"` or 1.0-1.3 with local embeddings |
| Frame Balance card missing | the query had no detectable presupposition (`frame.type: "none"`); expected for neutral topics like "history of X". Loaded shapes: why-does, proof-that, X-better-than-Y, X-causes-Y, is-X-safe |
| No `[counter-frame]` probe in the selection | fixed epsilon set too tight (fixed mode respects your radius strictly and skips probes outside it); use `"epsilon": "auto"`, which widens just enough to admit the dialectic pair and logs the adjustment |
| Frame balance not ~0 on live data | expected variance on small corpora; raise `RESULTS_PER_QUERY` and `n_queries`, and use semantic embeddings (`EMBEDDING_BACKEND=api`) for sharper stance anchors |

## Reference docs

- TrueFoundry gateway chat completions: https://www.truefoundry.com/docs/ai-gateway/chat-completions-overview
- Composio tool execution: https://docs.composio.dev/docs/tools-direct/executing-tools
- Airbyte API auth + jobs: https://docs.airbyte.com/platform/using-airbyte/configuring-api-access and https://reference.airbyte.com/reference/createaccesstoken
- Render blueprints: https://render.com/docs/blueprint-spec
- ClickHouse distance functions: https://clickhouse.com/docs/en/sql-reference/functions/distance-functions
