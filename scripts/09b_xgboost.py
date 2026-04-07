#!/usr/bin/env python3
"""
ZORC — Phase 9b: XGBoost Benchmark
====================================
Diagnostic benchmark against the Random Forest baseline (P9).

Key differences vs P9:
  - XGBoost handles NaN natively → NO imputation of BioEmu dynamic features
  - has_bioemu flag still included as a feature (for fair comparison)
  - Same CD-HIT split (08_zorc_split_assignments.csv) — identical train/val/test
  - SHAP TreeExplainer for interpretability comparison with RF

Primary diagnostic question:
  If rmsf_mean / rg_mean rank higher in XGBoost SHAP than in RF SHAP,
  the RF performance is being suppressed by 49.5% median imputation.
  → Fix = AF2 structure recovery (Step 2) or BioEmu Tier 1 (Step 3).

Usage:
    conda activate zorc_pipeline
    python scripts/09b_xgboost.py --config config/zorc_config.yaml

Outputs:
    results/09b_zorc_xgb_model.json       XGBoost booster (JSON)
    results/09b_zorc_xgb_shap_values.csv  SHAP values per sample
    results/09b_zorc_xgb_shap_importance.csv  Mean |SHAP| per feature
    results/09b_zorc_xgb_predictions.csv  Predictions + probabilities
    logs/09b_xgb_report.txt               Full run report
"""

import argparse
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9b — XGBoost benchmark")
    p.add_argument("--config", required=True, help="Path to zorc_config.yaml")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def update_config(config_path: str, key: str, value):
    """Append a top-level key to config.yaml if it does not exist."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if key not in cfg:
        cfg[key] = value
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  [config] Added key '{key}' = {value}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, y_pred, split_name: str) -> dict:
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  [{split_name}] n={len(y_true)}")
    print(f"    AUROC  : {auroc:.4f}")
    print(f"    AUPRC  : {auprc:.4f}")
    print(f"    F1 pos : {f1_pos:.4f}")
    print(f"    F1 neg : {f1_neg:.4f}")
    print(f"    F1 macro: {f1_mac:.4f}")
    print(f"    Confusion matrix (rows=true, cols=pred):")
    print(f"      TN={tn:4d}  FP={fp:4d}")
    print(f"      FN={fn:4d}  TP={tp:4d}")

    return dict(
        split=split_name, n=len(y_true),
        auroc=auroc, auprc=auprc,
        f1_pos=f1_pos, f1_neg=f1_neg, f1_macro=f1_mac,
        tn=tn, fp=fp, fn=fn, tp=tp
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X_train, feature_names, n_background=500, seed=42):
    """
    Compute SHAP values using TreeExplainer.
    Background = subsample of training set (same as P9 for comparability).
    """
    rng = np.random.default_rng(seed)
    n_bg = min(n_background, len(X_train))
    bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
    X_bg = X_train[bg_idx]

    print(f"  Computing SHAP values (TreeExplainer, n={n_bg} background samples)...")
    explainer   = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)

    # XGBoost binary → shap_values is 2D array (n_samples × n_features)
    if isinstance(shap_values, list):
        # Some XGBoost versions return list[array] for binary classification
        shap_arr = shap_values[1]
    else:
        shap_arr = shap_values

    mean_abs = np.abs(shap_arr).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    # Full SHAP matrix (train only — same as P9 convention)
    shap_df = pd.DataFrame(shap_arr, columns=feature_names)

    return shap_df, importance_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP diagnostic: compare RF vs XGBoost rank for key IDR features
# ─────────────────────────────────────────────────────────────────────────────

IDR_FEATURES = [
    "idr_percent", "rmsf_mean", "rmsf_std", "rmsf_max",
    "rg_mean", "rg_std", "rg_cv",
    "rmsf_nterm50", "rmsf_cterm50",
    "contact_density", "packing_density", "sasa_per_residue",
    "pass_rate", "has_bioemu"
]

# RF baseline ranks from P9 (for diagnostic comparison)
RF_BASELINE_RANKS = {
    "mrna_length": 1, "cds_length": 2, "n_residues": 3,
    "di_CG": 4, "di_UA": 5, "utr3_au_content": 6,
    "utr3_length": 7, "di_UG": 8, "utr5_length": 9,
    "utr5_fraction": 10, "cds_fraction": 11, "fU": 12,
    "mfe_per_nt": 13, "fG": 14, "di_CA": 15,
    "rrach_per_kb": 16, "di_UU": 17, "di_CU": 18,
    "di_AG": 19, "au_content": 20,
}


def print_idr_diagnostic(importance_df: pd.DataFrame):
    """
    Print rank comparison for IDR/dynamic features between RF (P9) and XGBoost (P9b).
    This is the core diagnostic: if XGBoost ranks these higher, imputation was the culprit.
    """
    print("\n  ──────────────────────────────────────────────────────")
    print("  DIAGNOSTIC: IDR/Dynamic feature ranks — RF (P9) vs XGBoost (P9b)")
    print("  ──────────────────────────────────────────────────────")
    print(f"  {'Feature':<28} {'XGB rank':>9} {'RF rank':>9} {'Δ rank':>8}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*8}")

    xgb_rank_map = dict(zip(importance_df["feature"], importance_df["rank"]))
    n_total = len(importance_df)

    moved_up   = []
    moved_down = []

    for feat in IDR_FEATURES:
        xgb_r = xgb_rank_map.get(feat, None)
        rf_r  = RF_BASELINE_RANKS.get(feat, None)

        xgb_str = f"{xgb_r:>9d}" if xgb_r is not None else f"{'n/a':>9}"
        rf_str  = f"{rf_r:>9d}"  if rf_r  is not None else f"{'>>20':>9}"

        if xgb_r is not None and rf_r is not None:
            delta = rf_r - xgb_r   # positive = moved up in XGBoost
            delta_str = f"{delta:>+8d}"
            if delta > 0:
                moved_up.append((feat, delta))
            elif delta < 0:
                moved_down.append((feat, abs(delta)))
        elif xgb_r is not None and rf_r is None:
            # Feature entered Top 20 in XGBoost but was not in RF Top 20
            delta_str = f"{'↑ NEW':>8}"
            moved_up.append((feat, n_total))  # treat as maximum improvement
        else:
            delta_str = f"{'n/a':>8}"

        print(f"  {feat:<28} {xgb_str} {rf_str} {delta_str}")

    print()
    if moved_up:
        names = ", ".join(f[0] for f in sorted(moved_up, key=lambda x: -x[1])[:5])
        print(f"  ✓ Features ranked HIGHER in XGBoost: {names}")
        print(f"    → Supports imputation-bias hypothesis (RF P9 underestimated these)")
    else:
        print(f"  ✗ No IDR/dynamic features ranked higher in XGBoost.")
        print(f"    → Dynamic features may genuinely not discriminate beyond idr_percent.")
        print(f"    → Reconsider priority of AF2 recovery / BioEmu Tier 1 runs.")
    print("  ──────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("ZORC — Phase 9b: XGBoost Benchmark")
    print(f"Config: {args.config}")
    print(f"Timestamp: {ts}")
    print("=" * 70)

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir    = Path(cfg.get("base_dir", "~/Documents/ZORC")).expanduser()
    data_dir    = base_dir / "data" / "processed"
    results_dir = base_dir / "results"
    logs_dir    = base_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix_path = data_dir / cfg.get(
        "feature_matrix_file", "08_zorc_feature_matrix.csv")
    split_path          = data_dir / cfg.get(
        "split_assignments_file", "08_zorc_split_assignments.csv")
    hc_path             = data_dir / cfg.get(
        "hc_set_file", "colleague_high_confidence_set.csv")

    # ── XGBoost hyperparameters (from config or defaults) ─────────────────
    xgb_cfg = cfg.get("xgboost", {})
    N_ESTIMATORS   = int(xgb_cfg.get("n_estimators",   500))
    MAX_DEPTH      = int(xgb_cfg.get("max_depth",        6))
    LEARNING_RATE  = float(xgb_cfg.get("learning_rate",  0.05))
    MIN_CHILD_W    = int(xgb_cfg.get("min_child_weight", 5))
    SUBSAMPLE      = float(xgb_cfg.get("subsample",      0.8))
    COLSAMPLE_BT   = float(xgb_cfg.get("colsample_bytree", 0.8))
    RANDOM_STATE   = int(xgb_cfg.get("random_state",    42))
    EARLY_STOP     = int(xgb_cfg.get("early_stopping_rounds", 30))
    SHAP_BG        = int(xgb_cfg.get("shap_background_n", 500))

    # ── 1. Load feature matrix ─────────────────────────────────────────────
    print(f"\n[1/6] Loading feature matrix: {feature_matrix_path}")
    df = pd.read_csv(feature_matrix_path)
    print(f"  {len(df):,} samples, {df.shape[1]} columns")

    pos = (df["class"] == 1).sum()
    neg = (df["class"] == 0).sum()
    print(f"  Classes: pos={pos}, neg={neg}")

    # ── 2. Prepare features (NO imputation — XGBoost handles NaN natively) ─
    print("\n[2/6] Preparing features (NO imputation — NaN preserved for XGBoost)...")

    # Normalise column name: pipeline uses 'geneID', guard against 'gene_id'
    if "geneID" in df.columns and "gene_id" not in df.columns:
        df = df.rename(columns={"geneID": "gene_id"})

    # Non-numeric / meta columns to exclude from feature matrix
    META_COLS = [
        "gene_id", "transcript_id", "class",
        "condition", "qc_fail", "isoform_source", "event_type",
        "bioemu_tier", "bioemu_status", "feature_source",
        "cluster_id", "split",
    ]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Add has_bioemu flag (same logic as P9, for fair comparison)
    # has_bioemu=1 if any BioEmu dynamic feature is non-NaN
    bioemu_dynamic = [c for c in feature_cols if c.startswith(("rmsf_", "rg_", "contact_", "packing_", "sasa_", "pass_rate"))]
    if "has_bioemu" not in df.columns:
        df["has_bioemu"] = (~df[bioemu_dynamic].isna().all(axis=1)).astype(int)
    if "has_bioemu" not in feature_cols:
        feature_cols.append("has_bioemu")

    nan_counts = df[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  has_bioemu=1: {(df['has_bioemu']==1).sum()} ({(df['has_bioemu']==1).mean()*100:.0f}%)")
    print(f"  Features with NaN: {len(nan_features)}")
    for feat, count in nan_features.items():
        print(f"    {feat:<35} NaN={count:4d} ({count/len(df)*100:.1f}%)")

    # ── 3. Split assignments ───────────────────────────────────────────────
    # Feature matrix already contains 'split' column (added by P8).
    print(f"\n[3/6] Using split column already present in feature matrix...")
    if "split" not in df.columns:
        print(f"  'split' not found — loading from: {split_path}")
        splits = pd.read_csv(split_path).rename(columns={"geneID": "gene_id"})
        df = df.merge(splits[["gene_id", "split"]], on="gene_id", how="left")

    mask_train = df["split"] == "train"
    mask_val   = df["split"] == "val"
    mask_test  = df["split"] == "test"

    print(f"  Train: {mask_train.sum()}  Val: {mask_val.sum()}  Test: {mask_test.sum()}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["class"].values.astype(int)

    X_train, y_train = X[mask_train], y[mask_train]
    X_val,   y_val   = X[mask_val],   y[mask_val]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    # XGBoost DMatrix — NaN handled natively
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols)

    # ── 4. Class weight (scale_pos_weight = neg/pos ratio) ────────────────
    scale_pos_weight = float(neg) / float(pos)
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}  (neg/pos = {neg}/{pos})")

    # ── 5. Train XGBoost ───────────────────────────────────────────────────
    print(f"\n[4/6] Training XGBoost...")
    print(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"lr={LEARNING_RATE}, min_child_weight={MIN_CHILD_W}")
    print(f"  subsample={SUBSAMPLE}, colsample_bytree={COLSAMPLE_BT}, "
          f"early_stopping={EARLY_STOP} rounds")

    params = {
        "objective":          "binary:logistic",
        "eval_metric":        ["logloss", "auc"],
        "max_depth":          MAX_DEPTH,
        "learning_rate":      LEARNING_RATE,
        "n_estimators":       N_ESTIMATORS,
        "min_child_weight":   MIN_CHILD_W,
        "subsample":          SUBSAMPLE,
        "colsample_bytree":   COLSAMPLE_BT,
        "scale_pos_weight":   scale_pos_weight,
        "use_label_encoder":  False,
        "seed":               RANDOM_STATE,
        "tree_method":        "hist",    # fast, GPU-compatible
        "device":             "cuda",    # RTX A5000; falls back to cpu if absent
    }

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        evals_result=evals_result,
        verbose_eval=50,
    )

    best_round = booster.best_iteration
    best_val_auc = max(evals_result["val"]["auc"])
    print(f"\n  Best round: {best_round + 1}  |  Best val AUC (xgb eval): {best_val_auc:.4f}")

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    print("\n[5/6] Evaluation...")

    def predict(dmat, X_arr):
        prob = booster.predict(dmat)
        pred = (prob >= 0.5).astype(int)
        return prob, pred

    results = []

    # Train
    prob_tr, pred_tr = predict(dtrain, X_train)
    results.append(evaluate(y_train, prob_tr, pred_tr, "TRAIN"))

    # Val
    prob_val, pred_val = predict(dval, X_val)
    results.append(evaluate(y_val, prob_val, pred_val, "VAL"))

    # Test
    prob_te, pred_te = predict(dtest, X_test)
    results.append(evaluate(y_test, prob_te, pred_te, "TEST"))

    # ── 7. SHAP ────────────────────────────────────────────────────────────
    print(f"\n[6/7] SHAP feature importance...")
    shap_df, importance_df = compute_shap(
        booster, X_train, feature_cols,
        n_background=SHAP_BG, seed=RANDOM_STATE
    )

    print(f"\n  Top 20 features by mean |SHAP|:")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'-'*6} {'-'*35} {'-'*12}")
    for _, row in importance_df.head(20).iterrows():
        print(f"  {int(row['rank']):<6} {row['feature']:<35} {row['mean_abs_shap']:>12.4f}")

    # Diagnostic comparison with RF P9
    print_idr_diagnostic(importance_df)

    # ── 8. High-confidence validation ─────────────────────────────────────
    print("\n[7/7] High-confidence validation...")
    hc_df = pd.read_csv(hc_path)
    if "gene_id" in hc_df.columns:
        hc_genes = hc_df["gene_id"].tolist()
    elif "geneID" in hc_df.columns:
        hc_genes = hc_df["geneID"].tolist()
    else:
        hc_genes = hc_df.iloc[:, 0].tolist()

    hc_mask    = df["gene_id"].isin(hc_genes)
    df_hc      = df[hc_mask].copy()
    X_hc       = df_hc[feature_cols].values.astype(np.float32)
    y_hc       = df_hc["class"].values.astype(int)
    dhc        = xgb.DMatrix(X_hc, feature_names=feature_cols)
    prob_hc    = booster.predict(dhc)
    pred_hc    = (prob_hc >= 0.5).astype(int)
    correct_hc = (pred_hc == y_hc).sum()

    print(f"\n  High-confidence validation set (n={len(y_hc)}):")
    print(f"    Correctly classified: {correct_hc}/{len(y_hc)} "
          f"({correct_hc/len(y_hc)*100:.0f}%)")
    print(f"\n  {'GeneID':<16} {'True':>4} {'Pred':>5} {'P(pos)':>8}  {'OK':>3}")
    print(f"  {'-'*40}")
    for i, row in df_hc.iterrows():
        gid  = row["gene_id"]
        idx  = list(df_hc["gene_id"]).index(gid)
        pp   = prob_hc[idx]
        pr   = pred_hc[idx]
        tr   = y_hc[idx]
        ok   = "✓" if pr == tr else "✗"
        print(f"  {gid:<16} {tr:>4} {pr:>5} {pp:>8.4f}  {ok:>3}")

    # ── 9. Save outputs ────────────────────────────────────────────────────
    print("\n  Saving outputs...")

    # Model
    model_path = results_dir / "09b_zorc_xgb_model.json"
    booster.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # SHAP values (train set)
    shap_path = results_dir / "09b_zorc_xgb_shap_values.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"  SHAP values saved: {shap_path}")

    # SHAP importance
    imp_path = results_dir / "09b_zorc_xgb_shap_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"  SHAP importance saved: {imp_path}")

    # Predictions (all splits)
    pred_df = df[["gene_id", "transcript_id", "class", "split"]].copy()
    pred_all = np.full(len(df), np.nan)
    for split_name, mask, probs in [
        ("train", mask_train, prob_tr),
        ("val",   mask_val,   prob_val),
        ("test",  mask_test,  prob_te),
    ]:
        pred_all[np.where(mask)[0]] = probs
    pred_df["prob_pos"] = pred_all
    pred_df["pred"] = (pred_df["prob_pos"] >= 0.5).astype("Int64")
    pred_path = results_dir / "09b_zorc_xgb_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # ── 10. Write report ───────────────────────────────────────────────────
    report_path = logs_dir / "09b_xgb_report.txt"
    test_res = [r for r in results if r["split"] == "TEST"][0]
    val_res  = [r for r in results if r["split"] == "VAL"][0]

    with open(report_path, "w") as f:
        f.write(f"ZORC Phase 9b — XGBoost Report\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"XGBoost params:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest round: {best_round + 1}\n\n")
        f.write(f"Val  AUROC={val_res['auroc']:.4f}  "
                f"AUPRC={val_res['auprc']:.4f}  "
                f"F1-macro={val_res['f1_macro']:.4f}\n")
        f.write(f"Test AUROC={test_res['auroc']:.4f}  "
                f"AUPRC={test_res['auprc']:.4f}  "
                f"F1-macro={test_res['f1_macro']:.4f}\n\n")
        f.write(f"HC validation: {correct_hc}/{len(y_hc)} ({correct_hc/len(y_hc)*100:.0f}%)\n\n")
        f.write("Top 30 SHAP features:\n")
        for _, row in importance_df.head(30).iterrows():
            f.write(f"  {int(row['rank']):2d}  {row['feature']:<35}  {row['mean_abs_shap']:.4f}\n")
    print(f"  Report: {report_path}")

    # ── 11. Update config.yaml ─────────────────────────────────────────────
    update_config(args.config, "xgboost_model_file",
                  str(model_path.relative_to(base_dir)))
    update_config(args.config, "xgboost_shap_importance_file",
                  str(imp_path.relative_to(base_dir)))
    update_config(args.config, "xgboost_predictions_file",
                  str(pred_path.relative_to(base_dir)))

    # ── Summary ────────────────────────────────────────────────────────────
    val_res  = [r for r in results if r["split"] == "VAL"][0]
    test_res = [r for r in results if r["split"] == "TEST"][0]

    print("\n" + "=" * 70)
    print("XGBOOST BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Features          : {len(feature_cols)}")
    print(f"  Best round        : {best_round + 1} / {N_ESTIMATORS}")
    print(f"  VAL  AUROC        : {val_res['auroc']:.4f}")
    print(f"  VAL  AUPRC        : {val_res['auprc']:.4f}")
    print(f"  VAL  F1 macro     : {val_res['f1_macro']:.4f}")
    print(f"  TEST AUROC        : {test_res['auroc']:.4f}")
    print(f"  TEST AUPRC        : {test_res['auprc']:.4f}")
    print(f"  TEST F1 macro     : {test_res['f1_macro']:.4f}")
    print(f"  HC validation     : {correct_hc}/{len(y_hc)} ({correct_hc/len(y_hc)*100:.0f}%)")
    print()
    print("  RF P9 baseline for comparison:")
    print("  VAL  AUROC: 0.7990  AUPRC: 0.8312  F1-macro: 0.7247")
    print("  TEST AUROC: 0.7974  AUPRC: 0.8469  F1-macro: 0.7382")
    print("=" * 70)
    print("\n✓ Phase 9b complete.")


if __name__ == "__main__":
    main()
