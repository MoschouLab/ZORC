"""
P14b — ZORC Agent CLI
======================
Run the LangGraph ZORC agent for one or more gene AGI codes.

Usage
-----
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python agent/run_agent.py AT5G47010
    python agent/run_agent.py AT5G47010 AT1G01470 AT3G22270
    python agent/run_agent.py AT5G47010 --no-llm     # skip LLM if no API key
    python agent/run_agent.py AT5G47010 --json        # machine-readable output

Environment variables
---------------------
    ANTHROPIC_API_KEY   Required for LLM report generation
    ZORC_API_URL        FastAPI base URL (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# Allow `python agent/run_agent.py` from any working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _separator(width: int = 72) -> str:
    return "─" * width


def _print_state(state: dict, width: int = 72) -> None:
    gene  = state.get("gene_id", "?")
    prob  = state.get("prob_pos")
    pred  = state.get("prediction", "unknown")
    conf  = state.get("confidence", "unknown")
    src   = state.get("lookup_source", "?")
    err   = state.get("error")
    lit   = state.get("literature") or []
    report = state.get("report", "")

    print(_separator(width))
    print(f"  ZORC Agent — {gene}")
    print(_separator(width))

    if prob is not None:
        label = "P-body enriched" if pred == "enriched" else "NOT enriched"
        print(f"  Prediction   : {label}")
        print(f"  prob_pos     : {prob:.4f}")
        print(f"  Confidence   : {conf}")
        print(f"  Lookup source: {src}")
    else:
        print(f"  Status: {err or 'Gene not found'}")

    if lit:
        print(f"\n  Literature retrieved: {len(lit)} chunks")
        for h in lit[:3]:
            snippet = textwrap.shorten(h["text"], width=60, placeholder="…")
            print(f"    [{h['rank']}] {h['source']} p.{h['page'] + 1}"
                  f"  score={h['score']:.3f}")
            print(f"         {snippet}")

    if report:
        print("\n" + _separator(width))
        print(report)

    if err and prob is not None:
        print(f"\n  Warning: {err}")

    print(_separator(width))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZORC LangGraph agent — prediction + literature report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "gene_ids",
        nargs="+",
        metavar="GENE_ID",
        help="AGI code(s) to analyse (e.g. AT5G47010)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM report generation (useful for testing without API key)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Print machine-readable JSON instead of formatted output",
    )
    args = parser.parse_args()

    if args.no_llm:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    from agent.zorc_agent import build_agent
    agent = build_agent()

    results = []
    for gene_id in args.gene_ids:
        print(f"\nRunning ZORC agent for {gene_id.upper()} …", file=sys.stderr)
        state = agent.invoke({"gene_id": gene_id.upper()})
        results.append(state)

        if not args.json_out:
            _print_state(state)

    if args.json_out:
        # Serialise: keep only JSON-safe fields
        out = []
        for s in results:
            out.append({
                "gene_id":      s.get("gene_id"),
                "prob_pos":     s.get("prob_pos"),
                "prediction":   s.get("prediction"),
                "confidence":   s.get("confidence"),
                "shap_features": s.get("shap_features"),
                "n_lit_chunks": len(s.get("literature") or []),
                "report":       s.get("report"),
                "lookup_source": s.get("lookup_source"),
                "error":        s.get("error"),
            })
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
