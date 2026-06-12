"""Autonomous search execution (Sponsor: Composio).

Backends:
  composio : executes the Composio Search tool for every approved query,
             concurrently. Uses the official SDK when installed
             (composio.tools.execute), otherwise falls back to the raw REST
             API (POST {base}/api/v3/tools/execute/{slug}, x-api-key header).
             Docs: https://docs.composio.dev/docs/tools-direct/executing-tools
  mock     : deterministic synthetic results for offline dev / CI / demos
             without keys. Clearly labeled; never use for judged live runs.

The mock models the phenomenon under test: a plain single-query search lands
in an "echo chamber" (2 domains, near-identical phrasing), while orthogonal
queries fan out across distinct media ecosystems with distinct vocabulary.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field

from app.core.reframe import AFFIRM_MARKERS, COUNTER_MARKERS
from app.core.textproc import domain_of

# --------------------------------------------------------------------------
# Result model
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    query: str
    url: str
    title: str
    snippet: str = ""
    content: str = ""           # filled by mock; live results get content via ETL
    domain: str = ""
    rank: int = 0
    backend: str = "mock"

    def __post_init__(self):
        if not self.domain:
            self.domain = domain_of(self.url)


# --------------------------------------------------------------------------
# Mock backend (deterministic)
# --------------------------------------------------------------------------

_AXIS_PROFILES: dict[str, dict] = {
    "evidence_for": {
        "markers": ["supporting evidence", "benefits", "documented outcomes"],
        "domains": ["progress-institute.example", "innovation-daily-news.example"],
        "vocab": ["adoption", "improvement", "gains", "efficiency", "breakthrough",
                  "success", "growth", "deployment", "scaling", "momentum"],
    },
    "evidence_against": {
        "markers": ["criticism", "risks", "failures", "problems", "against"],
        "domains": ["watchdog-foundation.example", "critical-times.example"],
        "vocab": ["shortfall", "hazard", "failure", "overrun", "incident",
                  "recall", "lawsuit", "warning", "defect", "backlash"],
    },
    "scientific": {
        "markers": ["peer reviewed", "meta-analysis", "study"],
        "domains": ["journal-archive.example", "openscience.edu.example"],
        "vocab": ["cohort", "randomized", "statistically", "confidence", "sample",
                  "replication", "hypothesis", "variance", "control", "peer"],
    },
    "economic": {
        "markers": ["economic impact", "costs", "market analysis"],
        "domains": ["fiscal-times.example", "macro-ledger-blog.example"],
        "vocab": ["capital", "subsidy", "levelized", "amortized", "investment",
                  "tariff", "externality", "pricing", "demand", "liquidity"],
    },
    "policy": {
        "markers": ["regulation", "policy debate", "government oversight"],
        "domains": ["regulatory-monitor.gov.example", "statehouse-brief.example"],
        "vocab": ["statute", "compliance", "licensing", "mandate", "oversight",
                  "rulemaking", "jurisdiction", "moratorium", "permitting", "framework"],
    },
    "historical": {
        "markers": ["historical precedent", "past outcomes", "lessons"],
        "domains": ["history-encyclopedia.example", "retrospect-quarterly.example"],
        "vocab": ["decade", "precedent", "era", "archival", "legacy",
                  "hindsight", "milestone", "chronology", "predecessor", "aftermath"],
    },
    "international": {
        "markers": ["international comparison", "other countries", "global"],
        "domains": ["world-affairs-council.example", "global-survey-post.example"],
        "vocab": ["bilateral", "comparative", "transnational", "treaty", "region",
                  "exporting", "harmonization", "delegation", "sovereign", "accord"],
    },
    "ethical": {
        "markers": ["ethical concerns", "controversy", "moral"],
        "domains": ["ethics-forum.example", "civic-conscience.example"],
        "vocab": ["consent", "equity", "dignity", "obligation", "fairness",
                  "stakeholder", "accountability", "transparency", "justice", "deliberation"],
    },
    "practitioner": {
        "markers": ["practitioner experience", "field report", "expert"],
        "domains": ["operator-notes-blog.example", "fieldwork-weekly.example"],
        "vocab": ["maintenance", "workflow", "downtime", "calibration", "crew",
                  "logistics", "tooling", "inspection", "procedure", "throughput"],
    },
    "data": {
        "markers": ["statistics", "data trends", "measurement"],
        "domains": ["statline.gov.example", "openmetrics-lab.example"],
        "vocab": ["dataset", "percentile", "regression", "baseline", "quarterly",
                  "aggregate", "median", "decile", "timeseries", "census"],
    },
    "skeptic": {
        "markers": ["debunked", "fact check", "myths"],
        "domains": ["claimcheck-news.example", "rumor-audit.example"],
        "vocab": ["misleading", "unverified", "exaggerated", "context", "misquoted",
                  "retraction", "sourcing", "attribution", "correction", "verdict"],
    },
    "firsthand": {
        "markers": ["case study", "firsthand account", "community"],
        "domains": ["townhall-forum.example", "resident-voices-blog.example"],
        "vocab": ["neighborhood", "testimony", "petition", "meeting", "resident",
                  "anecdote", "experience", "household", "volunteer", "local"],
    },
}

_BASELINE_DOMAINS = ["trending-now.example", "popular-news.example"]
_BASELINE_FILLER = [
    "Everyone is talking about this topic right now and the takes keep coming.",
    "Here is what you need to know about the story that is everywhere today.",
    "The conversation online has been dominated by the same viral talking points.",
    "Influencers and aggregators repeat the headline claims with little variation.",
]


def _seed_int(*parts: str) -> int:
    return int.from_bytes(
        hashlib.blake2b("||".join(parts).encode(), digest_size=8).digest(), "big"
    )


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "page"


def _topic_tokens(query: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in toks if len(t) > 2][:6] or ["topic"]


def _infer_axis(query: str) -> str:
    q = query.lower()
    for axis, prof in _AXIS_PROFILES.items():
        for marker in prof["markers"]:
            if marker in q:
                return axis
    axes = sorted(_AXIS_PROFILES)
    return axes[_seed_int(query) % len(axes)]


# NOTE: "alternative explanations"/"confounding" deliberately absent: that
# probe surfaces neutral confounder analyses, not refutations.
_COUNTER_QUERY_MARKERS = ["against", "debunked", "refuted", "no link", "fact check",
                          "counterarguments", " not ", "whether", "myths"]
_AFFIRM_QUERY_MARKERS = ["supporting evidence", "benefits", "documented outcomes",
                         "proof", "reasons", "because"]
_DIRECTIONAL_RE = re.compile(
    r"\b(causes?|leads? to|results? in|prevents?|cures?|ruins?|destroys?|improves?|"
    r"harms?|increases?|decreases?|makes?|kills?|boosts?|triggers?)\b"
)


def _stance_of_query(query: str) -> str:
    """How real search engines behave: the wording of the query selects the
    stance of what comes back. Counter-framed probes surface refutations;
    affirm-framed (incl. bare 'X causes Y') surface confirmations."""
    q = f" {query.lower()} "
    if any(m in q for m in _COUNTER_QUERY_MARKERS):
        return "counter"
    if any(m in q for m in _AFFIRM_QUERY_MARKERS) or _DIRECTIONAL_RE.search(q):
        return "affirm"
    return "neutral"


def _stance_sentence(stance: str, rng, topic: list[str]) -> str:
    vocab = AFFIRM_MARKERS if stance == "affirm" else COUNTER_MARKERS
    picks = [vocab[int(j)] for j in rng.integers(0, len(vocab), size=3)]
    verdict = "stands" if stance == "affirm" else "fails"
    return (
        f"Assessment: the central {topic[0]} claim is {picks[0]} and {picks[1]} "
        f"in the sources reviewed here, and is considered {picks[2]}; "
        f"on balance the asked framing {verdict} scrutiny."
    )


class MockSearchBackend:
    """Deterministic synthetic corpus generator (offline mode)."""

    name = "mock"

    async def search(self, query: str, n_results: int = 5, baseline: bool = False) -> list[SearchResult]:
        import numpy as np

        rng = np.random.default_rng(_seed_int("mock", query, str(baseline)))
        topic = _topic_tokens(query)
        results: list[SearchResult] = []

        if baseline:
            # Echo chamber: 2 domains, recycled phrasing, heavy topic repetition,
            # and the content ARGUES THE USER'S FRAME (models "why does X do Y"
            # returning results that favor X->Y).
            for i in range(max(n_results, 6)):
                domain = _BASELINE_DOMAINS[i % len(_BASELINE_DOMAINS)]
                paras = [_stance_sentence("affirm", rng, topic)]
                for _ in range(3):
                    base = " ".join(topic)
                    filler = _BASELINE_FILLER[int(rng.integers(len(_BASELINE_FILLER)))]
                    w1 = AFFIRM_MARKERS[int(rng.integers(len(AFFIRM_MARKERS)))]
                    w2 = AFFIRM_MARKERS[int(rng.integers(len(AFFIRM_MARKERS)))]
                    paras.append(
                        f"{base.capitalize()} update: {filler} "
                        f"The {topic[0]} story stays the {topic[-1]} story, "
                        f"and the same {base} angle is repeated again. "
                        f"Commentators call the claim {w1}, widely {w2} by the usual voices."
                    )
                results.append(SearchResult(
                    query=query,
                    url=f"https://{domain}/story/{_slug(query)}-{i}",
                    title=f"{' '.join(topic).title()}: What Everyone Is Saying ({i + 1})",
                    snippet=paras[0][:160],
                    content="\n\n".join(paras),
                    rank=i,
                    backend="mock",
                ))
            return results

        axis = _infer_axis(query)
        stance = _stance_of_query(query)
        prof = _AXIS_PROFILES[axis]
        for i in range(n_results):
            domain = prof["domains"][i % len(prof["domains"])]
            vocab = list(prof["vocab"])
            paras = [] if stance == "neutral" else [_stance_sentence(stance, rng, topic)]
            stance_vocab = AFFIRM_MARKERS if stance == "affirm" else COUNTER_MARKERS
            for p in range(3):
                picks = [vocab[int(j)] for j in rng.integers(0, len(vocab), size=5)]
                tail = ""
                if stance != "neutral":
                    word = stance_vocab[int(rng.integers(len(stance_vocab)))]
                    tail = f" Reviewers describe the claim as {word}."
                paras.append(
                    f"From the {axis.replace('_', ' ')} angle on {' '.join(topic)}: "
                    f"analysis of {picks[0]} and {picks[1]} indicates that {topic[0]} "
                    f"{picks[2]} considerations dominate, with {picks[3]} and {picks[4]} "
                    f"cited by independent reviewers in section {p + 1}.{tail}"
                )
            results.append(SearchResult(
                query=query,
                url=f"https://{domain}/articles/{_slug(query)}-{i}",
                title=f"{axis.replace('_', ' ').title()} perspective: {' '.join(topic[:3]).title()} ({i + 1})",
                snippet=paras[0][:160],
                content="\n\n".join(paras),
                rank=i,
                backend="mock",
            ))
        return results


# --------------------------------------------------------------------------
# Composio backend
# --------------------------------------------------------------------------


def _walk_for_results(obj, found: list[dict], depth: int = 0) -> None:
    """Defensively locate result lists in Composio tool output."""
    if depth > 6:
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and (item.get("link") or item.get("url")):
                found.append(item)
            else:
                _walk_for_results(item, found, depth + 1)
    elif isinstance(obj, dict):
        for key in ("results", "organic_results", "organic", "items", "news_results", "data", "response_data"):
            if key in obj:
                _walk_for_results(obj[key], found, depth + 1)
        if not found:
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _walk_for_results(v, found, depth + 1)


class ComposioSearchBackend:
    name = "composio"

    def __init__(self, settings):
        self.s = settings
        self._sdk = None
        try:
            from composio import Composio  # optional dependency
            self._sdk = Composio(api_key=settings.composio_api_key)
        except Exception:
            self._sdk = None  # use REST fallback

    async def search(self, query: str, n_results: int = 5, baseline: bool = False) -> list[SearchResult]:
        arguments = {"query": query}
        if self._sdk is not None:
            raw = await asyncio.to_thread(
                self._sdk.tools.execute,
                self.s.composio_tool_slug,
                user_id=self.s.composio_user_id,
                arguments=arguments,
            )
        else:
            raw = await self._rest_execute(arguments)
        found: list[dict] = []
        _walk_for_results(raw, found)
        results = []
        for i, item in enumerate(found[:n_results]):
            url = item.get("link") or item.get("url") or ""
            if not url.startswith("http"):
                continue
            results.append(SearchResult(
                query=query,
                url=url,
                title=(item.get("title") or url)[:300],
                snippet=(item.get("snippet") or item.get("description") or "")[:500],
                rank=i,
                backend="composio",
            ))
        return results

    async def _rest_execute(self, arguments: dict) -> dict:
        import httpx

        url = f"{self.s.composio_base_url}/api/v3/tools/execute/{self.s.composio_tool_slug}"
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                url,
                headers={"x-api-key": self.s.composio_api_key},
                json={"user_id": self.s.composio_user_id, "arguments": arguments},
            )
            resp.raise_for_status()
            return resp.json()


class SearchExecutor:
    """Concurrent fan-out across all approved queries plus the baseline control."""

    def __init__(self, settings):
        self.s = settings
        self.backend = (
            ComposioSearchBackend(settings) if settings.search_backend == "composio"
            else MockSearchBackend()
        )

    async def run(
        self, approved_queries: list[str], control_query: str, include_baseline: bool = True
    ) -> dict:
        sem = asyncio.Semaphore(5)

        async def guarded(q: str, baseline: bool):
            async with sem:
                try:
                    return await self.backend.search(
                        q, n_results=self.s.results_per_query, baseline=baseline
                    )
                except Exception as exc:  # keep pipeline alive on partial failure
                    return exc

        tasks = [guarded(q, False) for q in approved_queries]
        if include_baseline:
            tasks.append(guarded(control_query, True))
        raw = await asyncio.gather(*tasks)

        out = {"per_query": {}, "baseline": [], "errors": []}
        for q, res in zip(approved_queries, raw[: len(approved_queries)]):
            if isinstance(res, Exception):
                out["errors"].append(f"search failed for {q!r}: {res}")
            else:
                out["per_query"][q] = res
        if include_baseline:
            last = raw[-1]
            if isinstance(last, Exception):
                out["errors"].append(f"baseline search failed: {last}")
            else:
                out["baseline"] = last
        return out
