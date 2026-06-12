"""Text processing: ingress sanitation, HTML -> markdown, chunking, domain logic."""
from __future__ import annotations

import re
from urllib.parse import urlparse

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard (your|the) (system prompt|instructions)",
    r"you are now\b",
    r"reveal (your|the) (system )?prompt",
]

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class IngressRejection(ValueError):
    pass


def sanitize_query(raw: str) -> str:
    """Local ingress guard (runs in addition to gateway-side guardrails when
    the TrueFoundry gateway is enabled)."""
    q = _CTRL_RE.sub(" ", raw or "")
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < 3:
        raise IngressRejection("Query too short (min 3 characters).")
    if len(q) > 400:
        raise IngressRejection("Query too long (max 400 characters).")
    lowered = q.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, lowered):
            raise IngressRejection("Query rejected by ingress guardrails (prompt injection pattern).")
    return q


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return "unknown"
    netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "unknown"


_ECOSYSTEM_RULES = [
    ("government", [r"\.gov$", r"\.gov\.", r"\.mil$"]),
    ("academic", [r"\.edu$", r"\.ac\.", r"arxiv", r"pubmed", r"journal", r"nature\.com",
                  r"sciencedirect", r"springer", r"meta-analysis", r"scholar"]),
    ("community", [r"reddit", r"stackexchange", r"stackoverflow", r"\bforum", r"quora"]),
    ("independent", [r"substack", r"medium\.com", r"blogspot", r"wordpress", r"ghost\.io", r"blog"]),
    ("encyclopedic", [r"wikipedia", r"britannica", r"encyclopedia"]),
    ("mainstream_news", [r"nytimes", r"bbc", r"cnn", r"guardian", r"reuters", r"apnews",
                         r"washingtonpost", r"foxnews", r"news", r"times", r"daily", r"post"]),
    ("think_tank", [r"institute", r"\.org$", r"foundation", r"council", r"center"]),
]


def classify_ecosystem(domain: str) -> str:
    for label, patterns in _ECOSYSTEM_RULES:
        for pat in patterns:
            if re.search(pat, domain):
                return label
    return "other_web"


def html_to_markdown(html: str) -> str:
    """Lightweight HTML -> markdown-ish text (inline ETL fallback path)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form",
                     "noscript", "iframe", "svg", "button"]):
        tag.decompose()
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not text or len(text) < 30 and el.name not in ("h1", "h2", "h3"):
            continue
        if el.name in ("h1", "h2", "h3"):
            parts.append("#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            parts.append("- " + text)
        elif el.name == "blockquote":
            parts.append("> " + text)
        else:
            parts.append(text)
    if not parts:  # fall back to full visible text
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:8000]
    return "\n\n".join(parts)[:8000]


def chunk_text(text: str, chunk_chars: int = 700, max_chunks: int = 3) -> list[str]:
    """Split on paragraph boundaries into ~chunk_chars chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paras:
        return []
    chunks: list[str] = []
    current = ""
    for p in paras:
        if len(current) + len(p) + 2 <= chunk_chars or not current:
            current = (current + "\n\n" + p).strip()
        else:
            chunks.append(current)
            current = p
        if len(chunks) >= max_chunks:
            return chunks
    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return [c[: chunk_chars * 2] for c in chunks]
