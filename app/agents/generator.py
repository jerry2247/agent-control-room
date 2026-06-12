"""Variance Engine: orthogonal query generation node.

Candidate generation (LLM when configured, deterministic perspective
templates otherwise) followed by the constrained dispersion optimizer from
app.core.divergence.

Frame debiasing (Pillar D): the user's query is analyzed for presuppositions
("why does X do Y" presupposes X->Y). The epsilon-ball is centered on the
NEUTRALIZED topic core, not on the user's loaded phrasing, counter-framed
probe queries are injected into the candidate pool, and at least one
counter-frame query is guaranteed a slot in the final selection (via a
min-loss swap under the resultant-vector objective).
"""
from __future__ import annotations

import numpy as np

from app.agents.state import HarnessState
from app.core import reframe
from app.core.divergence import optimize_epsilon, pairwise_sq_sum, select_dispersed
from app import deps

AXIS_TEMPLATES: list[tuple[str, str]] = [
    ("evidence_for", "{t} supporting evidence benefits documented outcomes"),
    ("evidence_against", "{t} criticism risks failures documented problems"),
    ("scientific", "{t} peer reviewed study meta-analysis findings"),
    ("economic", "{t} economic impact costs market analysis"),
    ("policy", "{t} regulation policy debate government oversight"),
    ("historical", "{t} historical precedent past outcomes lessons"),
    ("international", "{t} international comparison other countries global approach"),
    ("ethical", "{t} ethical concerns controversy moral debate"),
    ("practitioner", "{t} expert practitioner experience field report"),
    ("data", "{t} statistics data trends measurement"),
    ("skeptic", "{t} fact check debunked myths claims"),
    ("firsthand", "{t} case study firsthand account community experience"),
]

_LLM_SYSTEM = (
    "You are the Variance Engine of an anti-confirmation-bias search system. "
    "Given one user query, produce diverse web search queries that examine the SAME topic "
    "from maximally different, mutually opposing angles (supporters, critics, scientists, "
    "economists, regulators, historians, other countries, ethicists, practitioners, "
    "statisticians, fact-checkers, affected communities). If the query presupposes a claim, "
    "at least three of your queries MUST probe the opposite: evidence against the claim, "
    "its negation, and alternative explanations. Every query MUST stay on the user's topic. "
    "Output ONLY a JSON array of query strings."
)


def extract_topic(query: str) -> str:
    """Neutral topic core (kept as a stable public helper; see core.reframe)."""
    return reframe.neutralize(query)


def template_candidates(query: str, exclude: set[str] | None = None) -> list[dict]:
    topic = extract_topic(query)
    out = []
    for axis, tmpl in AXIS_TEMPLATES:
        q = tmpl.format(t=topic)
        if exclude and q in exclude:
            continue
        out.append({"query": q, "axis": axis})
    return out


def frame_candidates(frame: reframe.Frame, exclude: set[str] | None = None) -> list[dict]:
    """Dialectic probes injected for loaded queries: counter pole + affirm pole."""
    out = [
        {"query": c, "axis": "counter_frame"}
        for c in frame.counter_queries
        if not (exclude and c in exclude)
    ]
    out += [
        {"query": a, "axis": "affirm_frame"}
        for a in frame.affirm_queries
        if not (exclude and a in exclude)
    ]
    return out


async def llm_candidates(query: str, m: int, feedback: str, frame: reframe.Frame | None = None) -> list[dict]:
    llm = deps.get_llm()
    user = f"User query: {query!r}\nGenerate {m} orthogonal search queries."
    if frame is not None and frame.presupposition:
        user += (
            f"\nDetected presupposition the user takes for granted: {frame.presupposition!r}. "
            "Include queries that directly test whether this claim is true, false, or confounded."
        )
    if feedback:
        user += f"\nA previous batch was rejected by the safety critic: {feedback}. Avoid those failure modes."
    arr = await llm.chat_json(_LLM_SYSTEM, user)
    cands = [{"query": str(q)[:200], "axis": "llm"} for q in arr if isinstance(q, str) and len(str(q)) > 5]
    if len(cands) < 4:
        raise ValueError("LLM returned too few usable queries")
    return cands[:m]


def _force_axis_representation(
    candidates: list[dict], idx: list[int], E: np.ndarray, dists: np.ndarray,
    eps: float, axis: str, locked: set[int],
) -> list[int]:
    """Guarantee >= 1 query of `axis` in the selection: swap in the feasible
    candidate of that axis maximizing the post-swap dispersion objective
    (computable in O(d) per swap via the resultant identity). Slots already
    forced for another axis are locked."""
    if any(candidates[i]["axis"] == axis for i in idx):
        locked.update(i for i in idx if candidates[i]["axis"] == axis)
        return idx
    feasible = [
        i for i, c in enumerate(candidates)
        if c["axis"] == axis and dists[i] <= eps and i not in idx
    ]
    if not feasible or len(idx) < 2:
        return idx
    s = E[idx].sum(axis=0)
    n = len(idx)
    best = None  # (objective, position_to_replace, candidate_index)
    for j in feasible:
        for pos, i_out in enumerate(idx):
            if i_out in locked:
                continue
            s_new = s - E[i_out] + E[j]
            obj = n * n - float(s_new @ s_new)
            if best is None or obj > best[0]:
                best = (obj, pos, j)
    if best is None:
        return idx
    _, pos, j = best
    out = list(idx)
    out[pos] = j
    locked.add(j)
    return out


async def generate_variance_queries(state: HarnessState) -> dict:
    settings = deps.get_settings()
    embedder = deps.get_embedder()
    query = state["sanitized_query"]
    n = state["n_queries"]
    exclude = {r["query"] for r in state.get("rejected_queries", [])}

    # ---- Frame analysis: presupposition + neutral center + counter probes ----
    frame = reframe.analyze(query)

    # ---- Candidate pool ----
    candidates: list[dict] = []
    errors: list[str] = []
    if deps.get_llm().available:
        try:
            candidates = await llm_candidates(query, max(12, 3 * n), state.get("critic_feedback", ""), frame)
        except Exception as exc:
            errors.append(f"LLM generation failed, using perspective templates: {exc}")
    if not candidates:
        candidates = template_candidates(query, exclude if state.get("retry_count") else None)
    # Dialectic probes are ALWAYS injected, regardless of generation mode.
    candidates = candidates + frame_candidates(frame, exclude)
    candidates = [c for c in candidates if c["query"] not in exclude] or candidates

    # ---- Embed neutral center + candidates ----
    # The ball is anchored on the de-biased topic core, NOT the loaded query.
    center_text = frame.neutral_topic or query
    texts = [center_text] + [c["query"] for c in candidates]
    vectors = await embedder.embed_batch(texts)
    center = np.asarray(vectors[0])
    E = np.vstack(vectors[1:])

    # ---- epsilon: user-fixed or auto-optimized ----
    requested_eps = float(state.get("epsilon") or 0.0)
    if requested_eps > 0:
        eps_used, curve, mode = requested_eps, None, "fixed"
    else:
        eps_used, curve = optimize_epsilon(
            E, center, n,
            grid_min=settings.epsilon_grid_min,
            grid_max=settings.epsilon_grid_max,
            steps=settings.epsilon_grid_steps,
        )
        mode = "auto"
        # The radius must admit the dialectic pair: probes are built FROM the
        # user's own presupposed claim, so their topicality is structural.
        # (Fixed mode respects the user's epsilon strictly.)
        if frame.presupposition:
            all_dists = np.linalg.norm(E - center, axis=1)
            probe_mins = []
            for ax in ("counter_frame", "affirm_frame"):
                ds = [float(all_dists[i]) for i, c in enumerate(candidates) if c["axis"] == ax]
                if ds:
                    probe_mins.append(min(ds))
            if probe_mins:
                need = min(max(probe_mins) + 1e-4, settings.epsilon_grid_max)
                if need > eps_used + 1e-3:
                    errors.append(
                        f"epsilon widened {eps_used:.3f} -> {need:.3f} to admit dialectic probes"
                    )
                if need > eps_used:
                    eps_used = need

    # ---- Constrained dispersion selection ----
    idx, dists = select_dispersed(E, center, eps_used, n)
    if len(idx) < min(2, n):  # radius too tight for this embedding geometry: widen once
        eps_widened = float(np.sort(dists)[min(n, len(dists)) - 1] + 1e-6)
        errors.append(
            f"epsilon={eps_used:.3f} admitted {len(idx)} candidates; widened to {eps_widened:.3f}"
        )
        eps_used = eps_widened
        idx, dists = select_dispersed(E, center, eps_used, n)

    # ---- Guaranteed dialectic representation: one counter + one affirm probe ----
    if frame.presupposition:
        locked: set[int] = set()
        idx = _force_axis_representation(candidates, idx, E, dists, eps_used, "counter_frame", locked)
        if state["n_queries"] >= 3:
            idx = _force_axis_representation(candidates, idx, E, dists, eps_used, "affirm_frame", locked)

    selected = [candidates[i]["query"] for i in idx]
    return {
        "frame": frame.to_dict(),
        "neutral_topic": center_text,
        "candidate_queries": candidates,
        "generated_queries": selected,
        "query_distances": {candidates[i]["query"]: round(float(dists[i]), 4) for i in idx},
        "query_axes": {candidates[i]["query"]: candidates[i]["axis"] for i in idx},
        "epsilon_used": round(float(eps_used), 4),
        "epsilon_mode": mode,
        "epsilon_curve": curve or state.get("epsilon_curve"),
        "retry_count": state.get("retry_count", 0) + 1,
        "error_logs": state.get("error_logs", []) + errors,
    }
