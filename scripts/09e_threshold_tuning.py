#!/usr/bin/env python3
"""
ZORC — Phase 9e: Threshold Tuning
====================================
Finds the optimal classification threshold for the RF model from P9d,
optimising for F1-macro (balancing positive and negative class performance).

The default threshold=0.5 yields F1-neg=0.655 vs F1-pos=0.790 (P9d test).
A lower threshold assigns more samples as positive, trading some F1-pos
for better F1-neg coverage.

Strategy:
  - Load RF model (09d_rf_eng_model.pkl) and enriched feature matrix
  - Scan thresholds 0.30–0.70 in steps of 0.01
  - Select threshold maximising F1-macro on VALIDATION set only
  - Report performance on TEST set at selected threshold
  - Also report at threshold=0.5 for direct comparison

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09e_threshold_tuning.py --config config/zorc_config.yaml

Outputs:
    results/09e_threshold_curve.csv     metrics at each threshold (val set)
    logs/09e_threshold_report.txt
"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, precision_score, recall_score
)

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9e — Threshold tuning")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


META_COLS = [
    "geneID", "gene_id", "transcript_id", "class",
    "condition", "qc_fail", "isoform_source", "event_type",
    "bioemu_tier", "bioemu_status", "feature_source",
    "cluster_id", "split",
]

EPS_RMSF = 0.1
EPS_CDS  = 1.0


def add_engineered_features(df):
    if "rmsf_nterm50" in df.columns and "rmsf_cterm50" in df.columns:
        df["rmsf_nterm_cterm_ratio"] = (
            df["rmsf_nterm50"] / (df["rmsf_cterm50"] + EPS_RMSF)
        )
    if "packing_density" in df.columns and "idr_percent" in df.columns:
        df["packing_x_idr"] = (
            df["packing_density"] * (df["idr_percent"] / 100.0)
        )
    if "rrach_count" in df.columns and "cds_length" in df.columns:
        df["rrach_per_cds_kb"] = (
            df["rrach_count"] / ((df["cds_length"] + EPS_CDS) / 1000.0)
        )
    if "utr3_au_content" in df.columns and "utr3_length" in df.columns:
        df["utr3_au_x_length"] = df["utr3_au_content"] * df["utr3_length"]
    return df


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    f1_pos  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_neg  = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return dict(threshold=threshold,
                f1_macro=f1_mac, f1_pos=f1_pos, f1_neg=f1_neg,
                precision=prec, recall=rec,
                tn=tn, fp=fp, fn=fn, tp=tp)


def print_metrics(m, label):
    print(f"\n  [{label}]  threshold={m['threshold']:.2f}")
    print(f"    F1-macro : {m['f1_macro']:.4f}")
    print(f"    F1-pos   : {m['f1_pos']:.4f}")
    print(f"    F1-neg   : {m['f1_neg']:.4f}")
    print(f"    Precision: {m['precision']:.4f}   Recall: {m['recall']:.4f}")
    print(f"    CM: TN={m['tn']:4d} FP={m['fp']:4d} / "
          f"FN={m['fn']:4d} TP={m['tp']:4d}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("ZORC — Phase 9e: Threshold Tuning")
    print(f"Config: {args.config}")
    print(f"Timestamp: {ts}")
    print("=" * 70)

    base_dir    = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    data_dir    = base_dir / "data" / "processed"
    results_dir = base_dir / "results"
    logs_dir    = base_dir / "logs"

    fm_path    = base_dir / cfg.get("outputs", {}).get(
                    "feature_matrix", "data/processed/08_zorc_feature_matrix.csv")
    model_path = results_dir / "09d_rf_eng_model.pkl"
    hc_path    = base_dir / cfg.get("outputs", {}).get(
                    "hc_set", "data/processed/colleague_high_confidence_set.csv")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading RF model: {model_path.name}")
    with open(model_path, "rb") as f:
        rf = pickle.load(f)
    print(f"  Loaded: RandomForestClassifier "
          f"(n_estimators={rf.n_estimators})")

    # ── 2. Load and prepare features ───────────────────────────────────────
    print(f"\n[2/4] Loading feature matrix: {fm_path.name}")
    df = pd.read_csv(fm_path)
    if "geneID" in df.columns:
        df = df.rename(columns={"geneID": "gene_id"})

    df = add_engineered_features(df)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Features: {len(feature_cols)}")

    mask_train = df["split"] == "train"
    mask_val   = df["split"] == "val"
    mask_test  = df["split"] == "test"

    X = df[feature_cols].values.astype(np.float32)
    y = df["class"].values.astype(int)

    X_train = X[mask_train]; y_train = y[mask_train]
    X_val   = X[mask_val];   y_val   = y[mask_val]
    X_test  = X[mask_test];  y_test  = y[mask_test]

    # Same imputation as P9d
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_val_imp  = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Get probabilities
    prob_val  = rf.predict_proba(X_val_imp)[:, 1]
    prob_test = rf.predict_proba(X_test_imp)[:, 1]

    auroc_val  = roc_auc_score(y_val,  prob_val)
    auroc_test = roc_auc_score(y_test, prob_test)
    auprc_val  = average_precision_score(y_val,  prob_val)
    auprc_test = average_precision_score(y_test, prob_test)

    print(f"  Val  AUROC={auroc_val:.4f}  AUPRC={auprc_val:.4f}")
    print(f"  Test AUROC={auroc_test:.4f}  AUPRC={auprc_test:.4f}")

    # ── 3. Threshold scan ──────────────────────────────────────────────────
    print(f"\n[3/4] Scanning thresholds on VALIDATION set (0.30–0.70)...")

    thresholds = np.arange(0.30, 0.71, 0.01)
    val_curve  = [metrics_at_threshold(y_val, prob_val, t) for t in thresholds]
    val_df     = pd.DataFrame(val_curve)

    # Best threshold: maximise F1-macro on val
    best_idx   = val_df["f1_macro"].idxmax()
    best_t     = val_df.loc[best_idx, "threshold"]
    best_val_m = val_df.loc[best_idx].to_dict()

    # Also show threshold that maximises F1-neg without dropping F1-pos below 0.75
    constrained = val_df[val_df["f1_pos"] >= 0.75]
    if len(constrained) > 0:
        best_neg_idx = constrained["f1_neg"].idxmax()
        best_neg_t   = constrained.loc[best_neg_idx, "threshold"]
        best_neg_m   = constrained.loc[best_neg_idx].to_dict()
    else:
        best_neg_t = best_t
        best_neg_m = best_val_m

    print(f"\n  Val curve (selected thresholds):")
    print(f"  {'Threshold':>10} {'F1-macro':>9} {'F1-pos':>7} "
          f"{'F1-neg':>7} {'Prec':>7} {'Recall':>7}")
    print(f"  {'-'*10} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    show_t = [0.35, 0.40, 0.42, 0.44, 0.45, 0.47, 0.50, 0.55, 0.60]
    for _, row in val_df[val_df["threshold"].isin(show_t)].iterrows():
        marker = ""
        if abs(row["threshold"] - best_t)    < 0.005: marker = " ← best F1-macro"
        if abs(row["threshold"] - best_neg_t) < 0.005 and best_neg_t != best_t:
            marker = " ← best F1-neg (F1-pos≥0.75)"
        print(f"  {row['threshold']:>10.2f} {row['f1_macro']:>9.4f} "
              f"{row['f1_pos']:>7.4f} {row['f1_neg']:>7.4f} "
              f"{row['precision']:>7.4f} {row['recall']:>7.4f}{marker}")

    # ── 4. Test set evaluation ─────────────────────────────────────────────
    print(f"\n[4/4] Test set evaluation at selected thresholds...")

    m_05  = metrics_at_threshold(y_test, prob_test, 0.50)
    m_opt = metrics_at_threshold(y_test, prob_test, best_t)
    m_neg = metrics_at_threshold(y_test, prob_test, best_neg_t)

    print_metrics(m_05,  "TEST threshold=0.50 (baseline)")
    print_metrics(m_opt, f"TEST threshold={best_t:.2f} (best val F1-macro)")
    if best_neg_t != best_t:
        print_metrics(m_neg,
                      f"TEST threshold={best_neg_t:.2f} (best val F1-neg, F1-pos≥0.75)")

    # ── HC validation at optimal threshold ────────────────────────────────
    print(f"\n  High-confidence validation (n=25) at threshold={best_t:.2f}:")
    hc_df    = pd.read_csv(hc_path)
    hc_col   = "gene_id" if "gene_id" in hc_df.columns else \
               "geneID"  if "geneID"  in hc_df.columns else \
               hc_df.columns[0]
    hc_genes = hc_df[hc_col].tolist()
    hc_mask  = df["gene_id"].isin(hc_genes)
    df_hc    = df[hc_mask].copy()
    X_hc_imp = imputer.transform(
                    df_hc[feature_cols].values.astype(np.float32))
    prob_hc  = rf.predict_proba(X_hc_imp)[:, 1]
    pred_hc  = (prob_hc >= best_t).astype(int)
    y_hc     = df_hc["class"].values.astype(int)
    ok_hc    = (pred_hc == y_hc).sum()
    print(f"  {ok_hc}/25 ({ok_hc/25*100:.0f}%)")

    # ── Save curve + report ────────────────────────────────────────────────
    curve_path = results_dir / "09e_threshold_curve.csv"
    val_df.to_csv(curve_path, index=False)

    report_path = logs_dir / "09e_threshold_report.txt"
    with open(report_path, "w") as f:
        f.write(f"ZORC Phase 9e — Threshold Tuning Report\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Model: {model_path.name}\n\n")
        f.write(f"Val  AUROC={auroc_val:.4f}  AUPRC={auprc_val:.4f}\n")
        f.write(f"Test AUROC={auroc_test:.4f}  AUPRC={auprc_test:.4f}\n\n")
        f.write(f"Optimal threshold (val F1-macro): {best_t:.2f}\n")
        f.write(f"Best-neg threshold (val F1-neg, F1-pos>=0.75): "
                f"{best_neg_t:.2f}\n\n")
        f.write("TEST at threshold=0.50:\n")
        for k, v in m_05.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTEST at threshold={best_t:.2f}:\n")
        for k, v in m_opt.items():
            f.write(f"  {k}: {v}\n")
        if best_neg_t != best_t:
            f.write(f"\nTEST at threshold={best_neg_t:.2f}:\n")
            for k, v in m_neg.items():
                f.write(f"  {k}: {v}\n")
        f.write(f"\nHC validation at {best_t:.2f}: {ok_hc}/25\n")
        f.write("\nFull val curve:\n")
        f.write(val_df.to_string(index=False))
    print(f"\n  Threshold curve: {curve_path}")
    print(f"  Report: {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("THRESHOLD TUNING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10} {'F1-macro':>9} {'F1-pos':>8} "
          f"{'F1-neg':>8}  Note")
    print(f"  {'-'*10} {'-'*9} {'-'*8} {'-'*8}  {'-'*25}")
    for m, note in [
        (m_05,  "baseline (0.50)"),
        (m_opt, f"best F1-macro ({best_t:.2f})"),
    ]:
        print(f"  {m['threshold']:>10.2f} {m['f1_macro']:>9.4f} "
              f"{m['f1_pos']:>8.4f} {m['f1_neg']:>8.4f}  {note}")
    if best_neg_t != best_t:
        print(f"  {m_neg['threshold']:>10.2f} {m_neg['f1_macro']:>9.4f} "
              f"{m_neg['f1_pos']:>8.4f} {m_neg['f1_neg']:>8.4f}  "
              f"best F1-neg ({best_neg_t:.2f})")
    print(f"\n  AUROC and AUPRC are threshold-independent:")
    print(f"  Val  AUROC={auroc_val:.4f}  AUPRC={auprc_val:.4f}")
    print(f"  Test AUROC={auroc_test:.4f}  AUPRC={auprc_test:.4f}")
    print(f"{'='*70}")
    print("\n✓ Phase 9e complete.")


if __name__ == "__main__":
    main()
