"""Frame analysis (Pillar D): presupposition detection, counter-framing,
and stance-symmetry anchors.

The deepest bias in agent search is not WHERE you look but WHAT you ask:
"why does X do Y" presupposes X->Y, so every result, however diverse the
source, argues inside the user's frame. This module:

  1. detects the loaded frame and extracts the presupposed claim P
  2. produces a NEUTRALIZED topic core (directional verbs, stance adjectives,
     and interrogative scaffolding removed) used as the epsilon-ball CENTER,
     so divergence is constrained around the topic, not around the user's bias
  3. generates COUNTER-FRAMED probe queries (negation, reversal, alternative
     explanations) that are injected into the candidate pool and guaranteed
     representation in the final selection
  4. builds affirm/negate anchor texts for the Frame Balance metric:

         balance = mean_e [ cos(e, a) - cos(e, n) ]

     where a embeds "P + confirmation vocabulary" and n embeds "P + refutation
     vocabulary". balance >> 0 means the corpus argues the asked frame;
     balance ~ 0 means evidence for and against P is symmetrically represented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_AUX = r"(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would|should)"
_DIRECTIONAL = (
    r"(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in|prevents?|cures?|"
    r"ruins?|destroys?|improves?|harms?|helps?|increases?|decreases?|makes?|"
    r"kills?|boosts?|wrecks?|fixes?|creates?|triggers?)"
)
_STANCE_ADJ = (
    r"(?:good|bad|great|terrible|safe|dangerous|effective|harmful|healthy|"
    r"unhealthy|reliable|unreliable|productive|ethical|unethical|legal|illegal|"
    r"worth\s+it|real|fake|overrated|underrated|slow|fast|better|worse|"
    r"addictive|toxic|risky|beneficial|necessary|expensive|corrupt|fair|unfair)"
)
_COMPARATOR = r"(?:better|worse|safer|cheaper|faster|stronger|healthier|smarter|more\s+\w+)"
# Trailing prepositional complement after a stance adjective:
# "bad FOR TEENAGERS", "better FOR THE ENVIRONMENT", "good TO HAVE IN SCHOOLS"
_COMPLEMENT = r"(?:for|to|in|on|at|with|among|of)\s+.+"
_POLAR_RE = None  # compiled lazily below (needs the f-string pieces above)


def _polar_re():
    global _POLAR_RE
    if _POLAR_RE is None:
        _POLAR_RE = re.compile(
            rf"^{_AUX}\s+(?P<x>.+?)\s+(?:so\s+)?(?P<adj>{_STANCE_ADJ})"
            rf"(?:\s+(?P<rest>{_COMPLEMENT}))?$"
        )
    return _POLAR_RE

_BIAS_PATTERNS = [
    rf"^\s*(?:why|how come|is|are|was|were|should|could|can|do|does|did|will)\b",
    rf"\b(?:so\s+)?{_STANCE_ADJ}\s*\??\s*$",
    r"\b(?:best|worst|top|amazing|horrible|overrated|underrated)\b",
    r"\bproof that\b", r"\bdebunk(?:ed|ing)?\b", r"\btruth about\b",
    r"\?+\s*$",
]

# Shared stance lexicons: used for the metric anchors AND by the mock corpus,
# and they match vocabulary real articles actually use.
AFFIRM_MARKERS = ["confirmed", "established", "demonstrated", "linked",
                  "supported", "validated", "proven"]
COUNTER_MARKERS = ["refuted", "disputed", "unsupported", "contradicted",
                   "overstated", "debunked", "unfounded"]


@dataclass
class Frame:
    type: str = "none"                 # loaded_why | asserted | comparative | causal | polar | none
    presupposition: str = ""           # the claim P the query takes for granted
    neutral_topic: str = ""            # de-biased topic core (epsilon-ball center)
    counter_queries: list = field(default_factory=list)   # probe NOT-P / alternatives
    affirm_queries: list = field(default_factory=list)    # probe P explicitly (dialectic pair)
    affirm_anchor: str = ""
    negate_anchor: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "presupposition": self.presupposition,
            "neutral_topic": self.neutral_topic,
            "counter_queries": list(self.counter_queries),
            "affirm_queries": list(self.affirm_queries),
            "affirm_anchor": self.affirm_anchor,
            "negate_anchor": self.negate_anchor,
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip(" ?.!").strip()


def neutralize(query: str) -> str:
    """Strip interrogative scaffolding, stance adjectives, and directional
    verbs to recover the neutral topic core."""
    t = query.lower()
    t = re.sub(rf"^\s*(?:why|how come|how)\s+(?:{_AUX}\s+)?", " ", t)
    for pat in _BIAS_PATTERNS:
        t = re.sub(pat, " ", t)
    # Stance adjective followed by a prepositional complement ("bad FOR teenagers",
    # "better FOR the environment"): drop the adjective, keep the complement.
    t = re.sub(
        rf"\b(?:so\s+)?{_STANCE_ADJ}\b(?=\s+(?:for|to|in|on|at|with|among|of)\b)", " ", t
    )
    t = re.sub(rf"\b{_DIRECTIONAL}\b", " ", t)
    t = _clean(t)
    return t or _clean(query.lower())


def analyze(query: str) -> Frame:
    q = _clean(query.lower())
    neutral = neutralize(query)
    f = Frame(neutral_topic=neutral)

    # 1. "why does X <verb> Y" / "why X is Y": presupposes the embedded claim
    m = re.match(rf"^(?:why|how come)\s+(?:{_AUX}\s+)?(?P<p>.+)$", q)
    if m:
        f.type = "loaded_why"
        f.presupposition = _clean(m.group("p"))
    # 2. "reasons/proof/evidence that P"
    if f.type == "none":
        m = re.match(r"^(?:reasons?|proof|evidence)\s+(?:that|why|for)\s+(?P<p>.+)$", q)
        if m:
            f.type = "asserted"
            f.presupposition = _clean(m.group("p"))
    # 3. comparative "A (is) better than B" (leading aux stripped from A)
    comp = re.match(
        rf"^(?:{_AUX}\s+)?(?P<a>.+?)\s+(?:{_AUX}\s+)?(?P<cmp>{_COMPARATOR})\s+than\s+(?P<b>.+)$", q
    )
    if f.type == "none" and comp:
        f.type = "comparative"
        f.presupposition = _clean(
            f"{comp.group('a')} {comp.group('cmp')} than {comp.group('b')}"
        )
    # 4. bare directional claim "X causes Y"
    if f.type == "none" and re.search(rf"\w.*\b{_DIRECTIONAL}\b.*\w", q):
        f.type = "causal"
        f.presupposition = q
    # 5. polar stance question, optionally with a complement:
    #    "is X safe", "is X bad for Y", "are X better for Y"
    if f.type == "none":
        m = _polar_re().match(q)
        if m:
            f.type = "polar"
            f.presupposition = _clean(" ".join(
                part for part in (m.group("x"), m.group("adj"), m.group("rest")) if part
            ))

    if not f.presupposition:
        return f

    P = f.presupposition
    counters = [
        f"{P} evidence against criticism",
        f"{P} debunked fact check no link",
    ]
    if f.type in ("loaded_why", "causal", "asserted") and re.search(rf"\b{_DIRECTIONAL}\b", P):
        counters.append(f"{neutral} correlation alternative explanations confounding factors")
    if f.type == "comparative" and comp:
        counters.append(f"{_clean(comp.group('b'))} {comp.group('cmp')} than {_clean(comp.group('a'))} evidence")
    if f.type == "polar":
        m = _polar_re().match(q)
        if m:
            rest = f" {_clean(m.group('rest'))}" if m.group("rest") else ""
            counters.append(
                f"{_clean(m.group('x'))} not {m.group('adj')}{rest} evidence concerns"
            )
    seen = set()
    f.counter_queries = [c for c in counters if not (c in seen or seen.add(c))][:4]

    # Dialectic pair: the goal is SYMMETRY, not the opposite bias. One probe
    # argues P explicitly so the corpus carries both poles plus neutral lenses.
    f.affirm_queries = [f"{P} supporting evidence reasons documented"]

    f.affirm_anchor = f"{P} " + " ".join(AFFIRM_MARKERS)
    f.negate_anchor = f"{P} no evidence " + " ".join(COUNTER_MARKERS)
    return f
