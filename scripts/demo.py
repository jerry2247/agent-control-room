"""CLI demo: run the full pipeline and print the judge-facing numbers.

Usage:
    python scripts/demo.py "is nuclear energy safe"
    python scripts/demo.py "is nuclear energy safe" --epsilon 1.1 --n 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.0, help="0 = auto-optimize")
    p.add_argument("--json", action="store_true", help="dump raw final state")
    args = p.parse_args()

    from app.agents.orchestrator import new_session_id, run_pipeline
    from app.core import registry
    from app import deps

    sid = new_session_id()
    registry.create(sid, {"original_query": args.query})
    s = deps.get_settings()
    print(f"backends: {s.resolved}")
    print(f"session : {sid}\nquery   : {args.query!r}\n")

    final = asyncio.run(run_pipeline(sid, args.query, args.n, args.epsilon))

    if args.json:
        out = {k: v for k, v in final.items() if k not in ("search_results", "baseline_results", "documents")}
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"epsilon : {final['epsilon_used']} ({final['epsilon_mode']})")
    frame = final.get("frame") or {}
    if frame.get("presupposition"):
        print(f"frame   : {frame['type']} | presupposes {frame['presupposition']!r}")
        print(f"          neutral center: {frame['neutral_topic']!r}")
    print("\n--- approved orthogonal queries ---")
    axes = final.get("query_axes", {})
    for q in final.get("approved_queries", []):
        tag = {"counter_frame": " [COUNTER]", "affirm_frame": " [AFFIRM]"}.get(axes.get(q, ""), "")
        print(f"  d={final['query_distances'].get(q, '?'):<7} {q}{tag}")
    for r in final.get("rejected_queries", []):
        print(f"  REJECTED ({r['reason']}): {r['query']}")

    m = final.get("metrics", {})
    if m:
        h, b = m["harness"], m["baseline"]
        print("\n--- corpus metrics (computed in ClickHouse: %s) ---" % h["engine"])
        fmt = "  {:<28} {:>9} {:>9} {:>9}"
        print(fmt.format("metric", "harness", "baseline", "lift"))
        for key, label in [
            ("semantic_spread", "semantic spread"),
            ("shannon_entropy_bits", "domain entropy (bits)"),
            ("ecosystem_entropy_bits", "ecosystem entropy (bits)"),
            ("n_domains", "unique domains"),
            ("n_documents", "documents"),
        ]:
            lift = m["deltas"].get(key, {}).get("lift_pct")
            print(fmt.format(label, h.get(key), b.get(key), f"+{lift}%" if lift and lift > 0 else (f"{lift}%" if lift is not None else "-")))
        fb = m.get("frame_balance")
        if fb:
            print(fmt.format("frame balance (0=unbiased)", f"{fb['harness']:+.4f}",
                             f"{fb['baseline']:+.4f}", fb["verdict"]))

    syn = final.get("synthesis", {})
    if syn.get("conflict"):
        c = syn["conflict"]
        print("\n--- strongest viewpoint conflict ---")
        print(f"  A [{', '.join(c['viewpoint_a']['label_terms'])}]: {c['viewpoint_a']['excerpt']['text'][:120]}...")
        print(f"  B [{', '.join(c['viewpoint_b']['label_terms'])}]: {c['viewpoint_b']['excerpt']['text'][:120]}...")
    if syn.get("consensus_terms"):
        print(f"\n  consensus terms: {', '.join(syn['consensus_terms'])}")
    errs = final.get("error_logs", [])
    if errs:
        print("\n--- warnings ---")
        for e in errs:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
