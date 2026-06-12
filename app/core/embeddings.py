"""Pluggable embedding backends.

local : deterministic hashed random-projection embedding (256-d, no network,
        no model download). Token -> seeded Gaussian vector; document vector is
        the tf-weighted sum, L2-normalized. Lexical-overlap similarity is
        preserved, which is sufficient for the dispersion math and for CI.
api   : any OpenAI-compatible /embeddings endpoint (TrueFoundry gateway,
        OpenAI, Voyage behind a proxy, ...). Production quality.

All vectors are L2-normalized, so for unit vectors:
    ||a - b||^2 = 2 * (1 - cos(a, b))   (Euclidean and cosine are equivalent)
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

# NOTE: "no"/"not" are deliberately NOT stopwords; negation is semantically
# load-bearing for the frame-balance metric (Pillar D).
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "that", "this", "these", "it",
    "as", "at", "from", "into", "about", "than", "then", "but",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "i", "you", "we", "they", "he", "she", "its", "their", "our",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VEC_CACHE: dict = {}
_VEC_CACHE_MAX = 60000


def tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _stable_seed(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


def _token_vector(token: str, dim: int) -> np.ndarray:
    key = (token, dim)
    v = _VEC_CACHE.get(key)
    if v is None:
        rng = np.random.default_rng(_stable_seed(token))
        v = rng.standard_normal(dim)
        if len(_VEC_CACHE) < _VEC_CACHE_MAX:
            _VEC_CACHE[key] = v
    return v


class LocalHashEmbedder:
    """Deterministic, dependency-free embedding. Same text -> same vector."""

    name = "local-hash-projection"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text) or ["__empty__"]
        v = np.zeros(self.dim, dtype=np.float64)
        for tok, count in Counter(tokens).items():
            v += (1.0 + math.log(count)) * _token_vector(tok, self.dim)
        norm = np.linalg.norm(v)
        if norm == 0:
            v[0] = 1.0
            norm = 1.0
        return (v / norm).astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_one(t) for t in texts]


class APIEmbedder:
    """OpenAI-compatible embeddings endpoint (POST {base}/embeddings)."""

    name = "openai-compatible-api"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        out: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), 64):
                batch = texts[start:start + 64]
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))
                for item in data:
                    v = np.asarray(item["embedding"], dtype=np.float32)
                    n = float(np.linalg.norm(v))
                    out.append(v / n if n > 0 else v)
        return out


def build_embedder(settings):
    if settings.embedding_backend == "api":
        return APIEmbedder(
            settings.embedding_api_base, settings.embedding_api_key, settings.embedding_model
        )
    return LocalHashEmbedder(dim=settings.embedding_dim)
