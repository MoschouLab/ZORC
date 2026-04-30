"""
P14b — ZORC LangGraph Agent
============================
Multi-step agent that integrates ZORC ML predictions with P-body literature
to produce structured gene-level reports.

Workflow
--------
START
  └─► get_prediction   (SQLite lookup, FastAPI fallback)
        │
        ├─ prob_pos > 0.8 ──► retrieve_literature  (ChromaDB RAG)
        │                            │
        │                            ▼
        └─ prob_pos ≤ 0.8 ──► generate_report  (Claude claude-sonnet-4-6)
                                     │
                                     ▼
                                    END

State keys
----------
gene_id          str       AGI code supplied by the caller
prob_pos         float     P(P-body enriched) ∈ [0, 1]
prediction       str       "enriched" | "not_enriched"
confidence       str       "high" | "medium" | "low"
shap_features    dict      {feature: shap_value} from DB or API
literature       list      top-k RAG chunks (only when prob_pos > 0.8)
report           str       final markdown report
error            str       set when a node fails non-fatally

Usage
-----
    from agent.zorc_agent import build_agent
    agent = build_agent()
    state = agent.invoke({"gene_id": "AT5G47010"})
    print(state["report"])
"""

from __future__ import annotations

import os
import sqlite3
import textwrap
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH   = _REPO_ROOT / "data" / "zorc_database.db"
FASTAPI_URL = os.environ.get("ZORC_API_URL", "http://localhost:8000")

# ── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    gene_id:      str
    prob_pos:     float | None
    prediction:   str | None
    confidence:   str | None
    shap_features: dict[str, float] | None
    literature:   list[dict[str, Any]] | None
    report:       str | None
    error:        str | None
    lookup_source: str | None   # "fastapi" | "sqlite" | "not_found"


# ── Node 1 — get_prediction ───────────────────────────────────────────────────


def get_prediction(state: AgentState) -> dict:
    """
    Retrieve the pre-computed ZORC prediction for gene_id.

    Tries the FastAPI /lookup/{gene_id} endpoint first; falls back to
    a direct SQLite read if the API is unreachable or returns an error.
    """
    gene_id = state["gene_id"].upper()

    # ── FastAPI attempt ────────────────────────────────────────────────────────
    try:
        import httpx
        resp = httpx.get(f"{FASTAPI_URL}/lookup/{gene_id}", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "gene_id":      gene_id,
                "prob_pos":     data["prob_pos"],
                "prediction":   data["prediction"],
                "confidence":   data["confidence"],
                "shap_features": data.get("top_shap_features", {}),
                "lookup_source": "fastapi",
            }
        # 404 means gene not in dataset — propagate that cleanly
        if resp.status_code == 404:
            return {
                "gene_id":      gene_id,
                "prob_pos":     None,
                "prediction":   "unknown",
                "confidence":   "unknown",
                "shap_features": {},
                "lookup_source": "not_found",
                "error": f"Gene {gene_id} not found in ZORC dataset.",
            }
    except Exception:
        pass  # API not running — fall through to SQLite

    # ── SQLite fallback ────────────────────────────────────────────────────────
    if not _DB_PATH.exists():
        return {
            "gene_id":    gene_id,
            "prob_pos":   None,
            "prediction": "unknown",
            "confidence": "unknown",
            "shap_features": {},
            "lookup_source": "not_found",
            "error": "ZORC database not found. Run scripts/10a_build_database.py.",
        }

    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT p.gene_id, p.prob_pos, p.pred,
                   p.shap_mrna_length, p.shap_cds_length,
                   p.shap_di_CG, p.shap_utr3_au_content,
                   p.shap_rrach_per_kb, p.shap_rmsf_nterm50
            FROM predictions p
            WHERE p.gene_id = ?
            """,
            (gene_id,),
        ).fetchone()
    finally:
        con.close()

    if row is None:
        return {
            "gene_id":    gene_id,
            "prob_pos":   None,
            "prediction": "unknown",
            "confidence": "unknown",
            "shap_features": {},
            "lookup_source": "not_found",
            "error": f"Gene {gene_id} not found in ZORC dataset (SQLite fallback).",
        }

    prob = float(row["prob_pos"])
    prediction = "enriched" if row["pred"] == 1 else "not_enriched"
    if prob >= 0.75:
        confidence = "high"
    elif prob >= 0.55:
        confidence = "medium"
    else:
        confidence = "low"

    shap_raw = {
        "mrna_length":     row["shap_mrna_length"],
        "cds_length":      row["shap_cds_length"],
        "di_CG":           row["shap_di_CG"],
        "utr3_au_content": row["shap_utr3_au_content"],
        "rrach_per_kb":    row["shap_rrach_per_kb"],
        "rmsf_nterm50":    row["shap_rmsf_nterm50"],
    }
    shap_features = {
        k: round(float(v), 6) for k, v in shap_raw.items() if v is not None
    }

    return {
        "gene_id":      gene_id,
        "prob_pos":     round(prob, 6),
        "prediction":   prediction,
        "confidence":   confidence,
        "shap_features": shap_features,
        "lookup_source": "sqlite",
    }


# ── Node 2 — retrieve_literature ─────────────────────────────────────────────


def retrieve_literature(state: AgentState) -> dict:
    """
    Query the P-body ChromaDB vector store for literature relevant to
    this gene's prediction.  Only reached when prob_pos > 0.8.
    """
    gene_id  = state["gene_id"]
    prob_pos = state.get("prob_pos", 0.0)
    shap     = state.get("shap_features") or {}

    # Build a biologically meaningful query from SHAP context
    top_features = sorted(shap, key=lambda k: abs(shap[k]), reverse=True)[:3]
    feature_text = ", ".join(top_features) if top_features else "mRNA sequence features"

    query = (
        f"P-body mRNA enrichment condensation {gene_id} "
        f"RNA sequence features {feature_text} "
        f"heat stress Arabidopsis"
    )

    try:
        # Import here so the module is usable without ChromaDB installed
        import sys
        sys.path.insert(0, str(_REPO_ROOT))
        from agent.rag_query import query_literature
        hits = query_literature(query, k=5)
    except Exception as exc:
        return {
            "literature": [],
            "error": f"RAG retrieval failed: {exc}",
        }

    return {"literature": hits}


# ── Node 3 — generate_report ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a plant molecular biology expert specialising in mRNA condensation,
P-bodies, and stress granules in *Arabidopsis thaliana*.
Your role is to generate concise, scientifically accurate reports that
integrate computational predictions with published experimental evidence.
Always cite the literature source when you reference experimental data.
Write in clear scientific English; use markdown formatting.
"""

_REPORT_TEMPLATE_WITH_LIT = """\
Gene: {gene_id}
ZORC prediction: {prediction} (prob_pos={prob_pos:.3f}, confidence={confidence})
Top SHAP features driving the prediction:
{shap_block}

Relevant literature passages retrieved from the P-body knowledge base:
{lit_block}

Task: Write a structured markdown report with exactly three sections:
1. **Prediction Summary** — state the gene, probability, and confidence level.
2. **Supporting Literature** — summarise the top-3 most relevant passages
   (cite source PDF and page number for each).
3. **Integrated Interpretation** — explain what the SHAP features and the
   literature evidence together suggest about this gene's role in P-body biology.

Be concise (300–400 words total). Use bullet points where appropriate.
"""

_REPORT_TEMPLATE_NO_LIT = """\
Gene: {gene_id}
ZORC prediction: {prediction} (prob_pos={prob_pos:.3f}, confidence={confidence})
Top SHAP features driving the prediction:
{shap_block}

Note: This gene has a lower P-body enrichment probability (prob_pos ≤ 0.8),
so targeted literature retrieval was not performed.

Task: Write a structured markdown report with exactly two sections:
1. **Prediction Summary** — state the gene, probability, and confidence level.
2. **Feature-Based Interpretation** — explain which sequence or structural
   features contributed most to the prediction, and what a low enrichment
   probability means biologically.

Be concise (150–200 words). Use bullet points where appropriate.
"""


def generate_report(state: AgentState) -> dict:
    """
    Call Claude claude-sonnet-4-6 to synthesise the prediction and literature
    into a structured markdown report.
    """
    gene_id    = state["gene_id"]
    prob_pos   = state.get("prob_pos")
    prediction = state.get("prediction", "unknown")
    confidence = state.get("confidence", "unknown")
    shap       = state.get("shap_features") or {}
    literature = state.get("literature")
    error_note = state.get("error", "")

    # ── Handle gene not found ─────────────────────────────────────────────────
    if prob_pos is None:
        report = (
            f"# ZORC Report — {gene_id}\n\n"
            f"**Status:** Gene not found in ZORC dataset.\n\n"
            f"{error_note}\n\n"
            "Use `POST /predict` with the mRNA sequence to obtain a new prediction."
        )
        return {"report": report}

    # ── Format SHAP block ─────────────────────────────────────────────────────
    if shap:
        shap_lines = [
            f"  - {feat}: {val:+.4f}" for feat, val in
            sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ]
        shap_block = "\n".join(shap_lines)
    else:
        shap_block = "  (SHAP values not available)"

    # ── Format literature block ───────────────────────────────────────────────
    if literature:
        lit_parts = []
        for hit in literature[:3]:
            snippet = textwrap.shorten(hit["text"], width=400, placeholder="…")
            lit_parts.append(
                f"[{hit['rank']}] {hit['source']} p.{hit['page'] + 1} "
                f"(score={hit['score']:.3f})\n  {snippet}"
            )
        lit_block = "\n\n".join(lit_parts)
        user_prompt = _REPORT_TEMPLATE_WITH_LIT.format(
            gene_id=gene_id,
            prediction=prediction,
            prob_pos=prob_pos,
            confidence=confidence,
            shap_block=shap_block,
            lit_block=lit_block,
        )
    else:
        user_prompt = _REPORT_TEMPLATE_NO_LIT.format(
            gene_id=gene_id,
            prediction=prediction,
            prob_pos=prob_pos,
            confidence=confidence,
            shap_block=shap_block,
        )

    # ── Call Anthropic API ────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        report = (
            f"# ZORC Report — {gene_id}\n\n"
            f"**Prediction:** {prediction}  \n"
            f"**P(P-body enriched):** {prob_pos:.3f}  \n"
            f"**Confidence:** {confidence}\n\n"
            "**Note:** `ANTHROPIC_API_KEY` not set — LLM report generation skipped.\n\n"
            f"**Top SHAP features:**\n{shap_block}"
        )
        return {"report": report}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        llm_text = message.content[0].text
    except Exception as exc:
        llm_text = (
            f"[LLM generation failed: {exc}]\n\n"
            f"**Prediction:** {prediction} | prob_pos={prob_pos:.3f} | {confidence}\n\n"
            f"**Top SHAP features:**\n{shap_block}"
        )

    report = f"# ZORC Report — {gene_id}\n\n{llm_text}"
    return {"report": report}


# ── Routing function ──────────────────────────────────────────────────────────


def _route_after_prediction(state: AgentState) -> str:
    prob = state.get("prob_pos")
    if prob is not None and prob > 0.8:
        return "retrieve_literature"
    return "generate_report"


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_agent():
    """Compile and return the ZORC LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("get_prediction",     get_prediction)
    workflow.add_node("retrieve_literature", retrieve_literature)
    workflow.add_node("generate_report",    generate_report)

    workflow.add_edge(START, "get_prediction")

    workflow.add_conditional_edges(
        "get_prediction",
        _route_after_prediction,
        {
            "retrieve_literature": "retrieve_literature",
            "generate_report":     "generate_report",
        },
    )

    workflow.add_edge("retrieve_literature", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# ── Module-level singleton ─────────────────────────────────────────────────────

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run(gene_id: str) -> AgentState:
    """Convenience wrapper — build agent, invoke, return final state."""
    return get_agent().invoke({"gene_id": gene_id})
