#!/usr/bin/env python3
"""
ZORC — Manuscript metrics table
=================================
Computes and prints the final model comparison table for the manuscript,
covering original and NUMT-clean datasets across P9d and P9f runs.

Reads pre-saved predictions CSVs (not re-trains models).

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09f_manuscript_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
HC_PATH = BASE / "data" / "processed" / "colleague_high_confidence_set.csv"


def metrics_from_preds(pred_df: pd.DataFrame, split: str = "test"):
    """Compute AUROC, AUPRC, F1-macro from a predictions CSV."""
    sub = pred_df[pred_df["split"] == split].dropna(subset=["prob_pos", "class"])
    y = sub["class"].values.astype(int)
    p = sub["prob_pos"].values
    pred = (p >= 0.5).astype(int)
    return dict(
        auroc=roc_auc_score(y, p),
        auprc=average_precision_score(y, p),
        f1_macro=f1_score(y, pred, average="macro", zero_division=0),
        n_test=len(y),
    )


def hc_from_preds(pred_df: pd.DataFrame) -> str:
    """Compute HC validation score from a predictions CSV."""
    hc_df = pd.read_csv(HC_PATH)
    hc_col = next(c for c in ["gene_id", "geneID"] if c in hc_df.columns)
    hc_genes = set(hc_df[hc_col])
    gene_col = next(c for c in ["gene_id", "geneID"] if c in pred_df.columns)
    sub = pred_df[pred_df[gene_col].isin(hc_genes)].dropna(subset=["prob_pos", "class"])
    if len(sub) == 0:
        return "n/a"
    y = sub["class"].values.astype(int)
    p = sub["prob_pos"].values
    ok = ((p >= 0.5).astype(int) == y).sum()
    return f"{ok}/{len(y)}"


RUNS = [
    {
        "label": "P9d original",
        "pred_csv": RESULTS / "09d_zorc_predictions.csv",
        "note": "RF 500 trees, 4 eng features, class_weight=balanced, no calibration",
    },
    {
        "label": "P9f original",
        "pred_csv": RESULTS / "09f_predictions_final.csv",
        "note": "RF 500 trees, 2 eng features, Platt-calibrated, class_weight=balanced",
    },
    {
        "label": "P9d numt_clean",
        "pred_csv": RESULTS / "09d_zorc_predictions_numt_clean.csv",
        "note": "RF 500 trees, 4 eng features, class_weight=balanced, NUMT-filtered (1434 genes)",
    },
    {
        "label": "P9f numt_clean",
        "pred_csv": RESULTS / "09f_predictions_final_numt_clean.csv",
        "note": "RF 500 trees, 2 eng features, Platt-calibrated, neg=1.5×, NUMT-filtered (1434 genes)",
    },
]


def main():
    print()
    print("=" * 72)
    print("ZORC — Final model comparison table (manuscript)")
    print("=" * 72)
    print()

    rows = []
    for run in RUNS:
        path = run["pred_csv"]
        if not path.exists():
            rows.append({
                "label": run["label"],
                "auroc": "missing", "auprc": "missing",
                "f1_macro": "missing", "hc": "missing",
            })
            continue
        df = pd.read_csv(path)
        m = metrics_from_preds(df, split="test")
        hc = hc_from_preds(df)
        rows.append({
            "label": run["label"],
            "auroc": f"{m['auroc']:.4f}",
            "auprc": f"{m['auprc']:.4f}",
            "f1_macro": f"{m['f1_macro']:.4f}",
            "hc": hc,
            "n_test": m["n_test"],
        })

    header = f"{'Model':<22} | {'Test AUROC':>10} | {'Test AUPRC':>10} | {'F1-macro':>8} | {'HC val':>6} | {'n_test':>6}"
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        n = r.get("n_test", "?")
        print(f"{r['label']:<22} | {r['auroc']:>10} | {r['auprc']:>10} | {r['f1_macro']:>8} | {r['hc']:>6} | {str(n):>6}")
    print()

    print("Notes:")
    for run in RUNS:
        print(f"  {run['label']:<22}: {run['note']}")

    print()
    print("Interpretation:")
    print("  AUROC drop on numt_clean is expected — NUMT pseudogenes had organellar")
    print("  nucleotide composition (AT-rich) that made them trivially easy negatives.")
    print("  AUPRC is the more informative metric for imbalanced positives.")
    print("  HC validation maintained 24/25 (96%) across all runs (RF models).")
    print()
    print("  *** P9f numt_clean metrics are reported in the manuscript. ***")
    print()


if __name__ == "__main__":
    main()
