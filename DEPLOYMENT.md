# Parallax Deployment and Demo Guide

This guide is for getting Parallax ready for a hackathon demo. The app runs locally with no credentials, and each sponsor integration can be turned on independently as credentials become available.

## Demo Modes

| Mode | Use it for | Backends |
| --- | --- | --- |
| Local rehearsal | Development, testing, fallback demo | local generator, mock search, inline ETL, embedded `chdb` if installed |
| Live search demo | Showing real web execution | Composio search, inline ETL, ClickHouse backend of your choice |
| Full sponsor demo | Showing the complete architecture | TrueFoundry, Composio, Airbyte, ClickHouse, Render |

Check the active mode with:

```bash
curl http://localhost:8000/api/health
```

The dashboard and API both report the active backend choices. Do not present mock search as live web search.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

Terminal demo:

```bash
python scripts/demo.py "why does coffee cause cancer"
python scripts/demo.py "is nuclear energy safe" --json
```

## Environment Variables

Start from `.env.example`. Everything is optional, so add one service at a time and check `/api/health` after each change.

| Variable | What changes |
| --- | --- |
| `COMPOSIO_API_KEY` | Search switches from deterministic mock data to Composio. |
| `TRUEFOUNDRY_BASE_URL`, `TRUEFOUNDRY_API_KEY`, `TRUEFOUNDRY_CHAT_MODEL` | LLM calls route through the TrueFoundry gateway. |
| `ANTHROPIC_API_KEY` | LLM calls route directly to Anthropic when TrueFoundry is not configured. |
| `CLICKHOUSE_HOST`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_MODE=cloud` | Metrics and document storage use ClickHouse Cloud. |
| `AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_CONNECTION_ID`, `ETL_BACKEND=airbyte` | The ETL step stages JSONL records and triggers an Airbyte sync. |
| `EMBEDDING_BACKEND=api`, `EMBEDDING_API_BASE`, `EMBEDDING_MODEL` | Embeddings come from an OpenAI-compatible endpoint instead of the local hash embedder. |

## Render Deployment

`render.yaml` defines the web service, Python runtime, install command, start command, and health check.

1. Push the repo to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Deploy once with no secrets to confirm the base app starts.
4. Add sponsor environment variables in Render as each integration is ready.
5. Verify the public URL:

```bash
curl https://<your-render-service>.onrender.com/api/health
```

Keep one worker for the demo because session state is stored in-process.

## ClickHouse

Local development uses embedded `chdb` when it imports successfully. For a public demo, ClickHouse Cloud gives you a separate SQL console you can show live.

1. Create a ClickHouse Cloud service.
2. Copy host, user, password, and port from the connection panel.
3. Set `CLICKHOUSE_HOST`, `CLICKHOUSE_PASSWORD`, and `CLICKHOUSE_MODE=cloud`.
4. Run one search.
5. In the dashboard, open the SQL panel and copy the metric query into the ClickHouse console.

Useful console check:

```sql
SELECT query_session_id, count() AS chunks, uniqExact(domain) AS domains
FROM scraped_documents
GROUP BY query_session_id
ORDER BY max(created_at) DESC
LIMIT 5;
```

## Composio

Composio powers live search execution.

1. Create a Composio project and copy the API key.
2. Enable the search toolkit for the project.
3. Confirm the search tool slug in the Composio dashboard.
4. Set `COMPOSIO_API_KEY`.
5. If needed, set `COMPOSIO_TOOL_SLUG`. The default is `COMPOSIO_SEARCH_SEARCH`.
6. Run a query and confirm result domains are real domains, not `*.example`.

The client uses the Composio SDK when installed and falls back to the REST endpoint already implemented in `app/services/composio_client.py`.

## TrueFoundry

TrueFoundry is used as the optional LLM gateway.

1. In TrueFoundry, configure an AI Gateway provider integration.
2. Copy the OpenAI-compatible gateway base URL from the playground or docs.
3. Create an API key or personal access token.
4. Set `TRUEFOUNDRY_BASE_URL`, `TRUEFOUNDRY_API_KEY`, and `TRUEFOUNDRY_CHAT_MODEL`.
5. Verify `/api/health` reports `llm: truefoundry`.

When enabled, generator, critic, and synthesis calls use the gateway. The deterministic local path remains available when no LLM backend is configured.

## Airbyte

Airbyte adds the durable ETL story. Inline ETL still runs during the request so the live demo does not wait on an external sync.

1. Create an Airbyte application and copy `client_id` and `client_secret`.
2. Create a connection from your staged JSONL source to ClickHouse.
3. Copy the Airbyte connection id.
4. Set `AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_CONNECTION_ID`, and `ETL_BACKEND=airbyte`.
5. Run a search and check the session payload for `airbyte_job`.

The code stages records to `AIRBYTE_STAGING_DIR` and triggers a sync through the Airbyte API.

## Demo Script

Before showing the project:

1. Warm the Render URL or start the local server.
2. Open the dashboard.
3. Keep `/api/health` ready in another tab.
4. If using ClickHouse Cloud, open the SQL console.
5. Run one test query before presenting.

Suggested three-minute flow:

1. Start with the problem: normal search agents inherit the user's framing.
2. Enter a loaded query.
3. Show the detected premise and neutral topic.
4. Show the generated query set, including affirm-frame and counter-frame probes.
5. Compare harness metrics against the baseline.
6. Open the SQL query behind the metric.
7. Show the conflict pair and source domains.
8. Close with the sponsor architecture: TrueFoundry for LLM gateway calls, Composio for search, Airbyte for durable ETL, ClickHouse for metrics, Render for hosting.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `/api/health` shows `search: mock` | `COMPOSIO_API_KEY` is not set or the app needs a restart. |
| Result URLs use `*.example` domains | The app is still using mock search. |
| `/api/health` shows `database: memory` | `chdb` did not import and ClickHouse Cloud is not configured. Metrics still run, but the SQL panel will not represent a ClickHouse backend. |
| Composio returns no results | Check the tool slug and confirm the search toolkit is enabled. |
| TrueFoundry returns 401 or 404 | Confirm the gateway base URL, API key, and model id. |
| Airbyte job is missing | Confirm `ETL_BACKEND=airbyte` and the client id, client secret, and connection id. |
| Fixed epsilon rejects too many queries | Use `epsilon: "auto"` for demo runs. |

## Reference Links

- TrueFoundry AI Gateway: https://www.truefoundry.com/docs/ai-gateway/chat-completions-overview
- Composio tools: https://docs.composio.dev/docs/tools-direct/executing-tools
- Airbyte API access: https://docs.airbyte.com/platform/using-airbyte/configuring-api-access
- ClickHouse distance functions: https://clickhouse.com/docs/en/sql-reference/functions/distance-functions
- Render Blueprints: https://render.com/docs/blueprint-spec
