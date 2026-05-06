#!/usr/bin/env python3
"""
09e_numt_impact.py
==================
ZORC — Phase 9e: NUMT Contamination Impact Analysis

Loads predictions from the RF and XGBoost runs (original vs NUMT-clean)
and prints a comparison table showing the metric delta after removing
AT2G07xxx NUMT contaminants from the negative class.

Metrics computed from predictions CSVs (test split):
  - AUROC, AUPRC, F1-macro
  - HC validation accuracy (25 lab-curated genes)

Also loads P9f Platt-calibrated final model predictions as reference baseline.

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09e_numt_impact.py --config config/zorc_config.yaml

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9e — NUMT contamination impact")
    p.add_argument("--config", required=True, help="Path to zorc_config.yaml")
    return p.parse_args()


def compute_metrics_from_csv(pred_csv: Path, hc_genes: set, prob_col: str = "prob_pos",
                              pred_col: str = "pred_class") -> dict | None:
    if not pred_csv.exists():
        return None

    df = pd.read_csv(pred_csv)

    # Normalise gene ID column
    if "geneID" in df.columns:
        df = df.rename(columns={"geneID": "gene_id"})
    if pred_col not in df.columns and "pred" in df.columns:
        df = df.rename(columns={"pred": pred_col})

    # Test split metrics
    test = df[df["split"] == "test"].copy()
    if len(test) == 0:
        return None

    y_true = test["class"].values
    y_prob = test[prob_col].values.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    auroc    = roc_auc_score(y_true, y_prob)
    auprc    = average_precision_score(y_true, y_prob)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # HC validation — use all splits (HC genes appear wherever they landed in split)
    hc_mask   = df["gene_id"].isin(hc_genes)
    df_hc     = df[hc_mask]
    n_hc      = len(df_hc)
    if n_hc > 0:
        hc_prob   = df_hc[prob_col].values.astype(float)
        hc_pred   = (hc_prob >= 0.5).astype(int)
        hc_true   = df_hc["class"].values
        hc_correct = (hc_pred == hc_true).sum()
        hc_str    = f"{hc_correct}/{n_hc}"
    else:
        hc_str = "n/a"

    return {
        "n_test":   len(test),
        "auroc":    auroc,
        "auprc":    auprc,
        "f1_macro": f1_macro,
        "hc":       hc_str,
    }


def fmt_delta(orig: float | None, clean: float | None) -> str:
    if orig is None or clean is None:
        return "   n/a"
    d = clean - orig
    return f"{d:+.4f}"


def print_table(rows: list[tuple[str, dict | None, dict | None]]) -> None:
    hdr = (f"{'Model':<28} {'Metric':<12} {'Original':>10} "
           f"{'NUMT-clean':>12} {'Delta':>10}")
    sep = "-" * len(hdr)

    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for model_name, orig, clean in rows:
        metrics = [
            ("Test AUROC",    "auroc"),
            ("Test AUPRC",    "auprc"),
            ("Test F1-macro", "f1_macro"),
            ("HC validation", "hc"),
        ]
        first = True
        for label, key in metrics:
            o_val = orig[key]  if orig  else None
            c_val = clean[key] if clean else None

            if key == "hc":
                o_str = f"{o_val:>10}" if o_val else f"{'n/a':>10}"
                c_str = f"{c_val:>12}" if c_val else f"{'n/a':>12}"
                d_str = f"{'n/a':>10}"
                # Quick pass/fail delta for HC
                if o_val and c_val and o_val != "n/a" and c_val != "n/a":
                    try:
                        oc = int(o_val.split("/")[0]); ot = int(o_val.split("/")[1])
                        cc = int(c_val.split("/")[0]); ct = int(c_val.split("/")[1])
                        diff = cc - oc
                        d_str = f"{'±0' if diff==0 else f'{diff:+d}':>10}"
                    except Exception:
                        pass
            else:
                o_str = f"{o_val:>10.4f}" if isinstance(o_val, float) else f"{'n/a':>10}"
                c_str = f"{c_val:>12.4f}" if isinstance(c_val, float) else f"{'n/a':>12}"
                d_str = fmt_delta(
                    o_val if isinstance(o_val, float) else None,
                    c_val if isinstance(c_val, float) else None
                ).rjust(10)

            prefix = f"  {model_name:<26}" if first else f"  {'':26}"
            print(f"{prefix} {label:<12} {o_str} {c_str} {d_str}")
            first = False
        print(sep)


def main() -> int:
    args = parse_args()
    cfg  = load_config(args.config)

    proj     = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    results  = proj / "results"
    proc     = proj / "data" / "processed"

    print("=" * 70)
    print("ZORC — Phase 9e: NUMT Contamination Impact Analysis")
    print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Load HC gene list ─────────────────────────────────────────────────────
    hc_path = proc / "colleague_high_confidence_set.csv"
    hc_genes: set = set()
    if hc_path.exists():
        hc_df = pd.read_csv(hc_path)
        id_col = "geneID" if "geneID" in hc_df.columns else hc_df.columns[0]
        hc_genes = set(hc_df[id_col])
        print(f"\n  HC set loaded: {len(hc_genes)} genes from {hc_path.name}")
    else:
        print(f"\n  WARNING: HC set not found at {hc_path}")

    # ── Prediction CSV paths ──────────────────────────────────────────────────
    runs = [
        (
            "RF (P9 baseline)",
            results / "09_zorc_predictions.csv",
            results / "09_zorc_predictions_numt_clean.csv",
            "pred_class",
        ),
        (
            "XGBoost (P9b)",
            results / "09b_zorc_xgb_predictions.csv",
            results / "09b_zorc_xgb_predictions_numt_clean.csv",
            "pred",
        ),
    ]

    # P9f (Platt-calibrated final) — original only, for reference
    p9f_path = results / "09f_predictions_final.csv"
    p9f_metrics = None
    if p9f_path.exists():
        p9f_metrics = compute_metrics_from_csv(p9f_path, hc_genes, pred_col="pred_class")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n  Loading predictions and computing metrics...\n")

    table_rows = []
    for model_name, orig_path, clean_path, pcol in runs:
        orig_m  = compute_metrics_from_csv(orig_path,  hc_genes, pred_col=pcol)
        clean_m = compute_metrics_from_csv(clean_path, hc_genes, pred_col=pcol)

        status_orig  = f"✓ {orig_path.name}"  if orig_m  else f"✗ missing: {orig_path.name}"
        status_clean = f"✓ {clean_path.name}" if clean_m else f"✗ missing: {clean_path.name}"
        print(f"  {model_name}:")
        print(f"    Original  : {status_orig}")
        print(f"    NUMT-clean: {status_clean}")

        table_rows.append((model_name, orig_m, clean_m))

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("NUMT CONTAMINATION IMPACT — TEST SET METRICS")
    print("=" * 70)
    print_table(table_rows)

    # ── P9f reference (Platt-calibrated) ─────────────────────────────────────
    if p9f_metrics:
        print(f"\n  P9f reference (Platt-calibrated RF, original dataset):")
        print(f"    Test AUROC    : {p9f_metrics['auroc']:.4f}")
        print(f"    Test AUPRC    : {p9f_metrics['auprc']:.4f}")
        print(f"    Test F1-macro : {p9f_metrics['f1_macro']:.4f}")
        print(f"    HC validation : {p9f_metrics['hc']}")

    # ── NUMT filter summary ───────────────────────────────────────────────────
    excluded_path = proc / "08_zorc_numt_excluded.csv"
    heldout_path  = proc / "08_zorc_numt_heldout.csv"
    if excluded_path.exists():
        excl = pd.read_csv(excluded_path)
        print(f"\n  Dataset size comparison:")
        print(f"    Original   : 1,510 genes  (pos=889, neg=621)")
        print(f"    NUMT-clean : {1510 - len(excl) - (pd.read_csv(heldout_path).__len__() if heldout_path.exists() else 0):,} genes "
              f"  ({len(excl)} excluded + {pd.read_csv(heldout_path).__len__() if heldout_path.exists() else 0} held-out)")
        print(f"\n  Interpretation:")
        print(f"    AUROC change reflects removal of 'easy' NUMT sequences from neg class.")
        print(f"    Lower AUROC on clean data = more honest estimate of biologically")
        print(f"    meaningful discrimination ability.")
        print(f"    AUPRC is less affected because it focuses on positive-class precision.")

    print(f"\n{'='*70}")
    print("✓ Phase 9e complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
