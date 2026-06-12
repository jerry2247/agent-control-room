"""ETL & ingestion (Sponsor: Airbyte).

Two modes:

  inline  (default) : the harness fetches target URLs itself (async httpx),
                      strips boilerplate HTML to markdown, chunks, embeds.
                      Always runs so the demo never blocks on an external sync.

  airbyte           : in addition to the inline pass, every raw scraped record
                      is staged as JSONL (AIRBYTE_STAGING_DIR, point an Airbyte
                      File/S3 source at it) and a sync of the configured
                      Airbyte connection (source -> ClickHouse destination) is
                      triggered programmatically through the Airbyte API:
                        POST {AIRBYTE_API_BASE}/applications/token   (client credentials)
                        POST {AIRBYTE_API_BASE}/jobs                 {connectionId, jobType: sync}
                      Docs: https://reference.airbyte.com/reference/createaccesstoken
                            https://docs.airbyte.com/platform/using-airbyte/configuring-api-access

This makes Airbyte the durable pipeline of record while keeping the live demo
latency-proof.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import time

import httpx

from app.core.textproc import chunk_text, classify_ecosystem, html_to_markdown
from app.services.composio_client import SearchResult

USER_AGENT = "OrthogonalSearchHarness/1.0 (+research tool; respects robots directives)"


class InlineETL:
    """Fetch -> clean -> normalize -> chunk."""

    name = "inline"

    def __init__(self, settings):
        self.s = settings

    async def hydrate_contents(self, results: list[SearchResult]) -> list[SearchResult]:
        """Fill .content for results that only carry a URL (live search)."""
        need = [r for r in results if not r.content]
        if not need:
            return results
        sem = asyncio.Semaphore(8)

        async def fetch(r: SearchResult):
            async with sem:
                try:
                    async with httpx.AsyncClient(
                        timeout=self.s.fetch_timeout_s,
                        follow_redirects=True,
                        headers={"User-Agent": USER_AGENT},
                    ) as client:
                        resp = await client.get(r.url)
                        resp.raise_for_status()
                        ctype = resp.headers.get("content-type", "")
                        if "html" not in ctype and "text" not in ctype:
                            return
                        r.content = html_to_markdown(resp.text)
                except Exception:
                    r.content = r.snippet  # degrade gracefully to the snippet

        await asyncio.gather(*[fetch(r) for r in need])
        return results

    def to_chunks(self, results: list[SearchResult], session_id: str, control: bool,
                  id_start: int) -> list[dict]:
        rows: list[dict] = []
        next_id = id_start
        seen_urls = set()
        for r in results:
            if r.url in seen_urls or not (r.content or r.snippet):
                continue
            seen_urls.add(r.url)
            chunks = chunk_text(
                r.content or r.snippet,
                chunk_chars=self.s.chunk_chars,
                max_chunks=self.s.max_chunks_per_doc,
            )
            for ci, chunk in enumerate(chunks):
                rows.append({
                    "id": next_id,
                    "query_session_id": session_id,
                    "control_group": 1 if control else 0,
                    "source_query": r.query,
                    "url": r.url,
                    "domain": r.domain,
                    "ecosystem": classify_ecosystem(r.domain),
                    "title": r.title,
                    "chunk_index": ci,
                    "content": chunk,
                    "embedding": None,  # filled by the persist node
                })
                next_id += 1
                if len(rows) >= self.s.max_total_chunks:
                    return rows
        return rows


class AirbyteETL(InlineETL):
    """InlineETL plus durable staging + programmatic Airbyte sync trigger."""

    name = "airbyte"

    def __init__(self, settings):
        super().__init__(settings)
        self._token: str | None = None
        self._token_ts = 0.0

    async def _get_token(self) -> str:
        if self._token and time.time() - self._token_ts < 600:
            return self._token
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.s.airbyte_api_base}/applications/token",
                json={
                    "client_id": self.s.airbyte_client_id,
                    "client_secret": self.s.airbyte_client_secret,
                    "grant-type": "client_credentials",
                },
            )
            resp.raise_for_status()
            self._token = resp.json()["access_token"]
            self._token_ts = time.time()
            return self._token

    def stage_records(self, results: list[SearchResult], session_id: str) -> str:
        staging = pathlib.Path(self.s.airbyte_staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / f"{session_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "session_id": session_id,
                    "query": r.query,
                    "url": r.url,
                    "domain": r.domain,
                    "title": r.title,
                    "content": r.content or r.snippet,
                    "backend": r.backend,
                }) + "\n")
        return str(path)

    async def trigger_sync(self) -> dict:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.s.airbyte_api_base}/jobs",
                headers={"Authorization": f"Bearer {token}"},
                json={"connectionId": self.s.airbyte_connection_id, "jobType": "sync"},
            )
            resp.raise_for_status()
            return resp.json()


def build_etl(settings):
    if settings.etl_backend == "airbyte" and settings.airbyte_client_id:
        return AirbyteETL(settings)
    return InlineETL(settings)
