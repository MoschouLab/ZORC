#!/usr/bin/env python3
"""
ZORC — Phase 9f: Final Model
==============================
Clean retraining + probability calibration + asymmetric class weight exploration.

Three interventions combined:
  1. Clean feature set: drop packing_x_idr (rank 55) and rrach_per_cds_kb
     (rank 37, extreme outliers max=40000). Keep rmsf_nterm_cterm_ratio and
     utr3_au_x_length (both confirmed in Top 20 by RF and XGBoost).

  2. Platt calibration (CalibratedClassifierCV, method='sigmoid', cv=5):
     Fits a logistic regression on top of RF probability outputs via
     cross-validation on the training set. Improves probability calibration
     without touching AUROC. A better-calibrated model yields more informative
     threshold cuts and better F1 at threshold=0.5.

  3. Asymmetric class weight sweep {0: w, 1: 1.0} for w in [1.0, 1.5, 2.0, 2.5]:
     'balanced' gives w~1.43. We test higher negative weights and report
     F1-neg vs F1-pos tradeoff on val, then apply best to test.

All three are evaluated on the same CD-HIT split (train/val/test) from P8.
AUROC/AUPRC are threshold-independent and reported for completeness.

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09f_final_model.py --config config/zorc_config.yaml

Outputs:
    results/09f_rf_final_model.pkl          final calibrated RF
    results/09f_shap_final.csv              SHAP importance
    results/09f_predictions_final.csv       predictions all splits
    logs/09f_final_model_report.txt
"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9f — Final model")
    p.add_argument("--config", required=True)
    p.add_argument("--feature-matrix", default=None,
                   help="Override config feature matrix path")
    p.add_argument("--output-suffix", default="",
                   help="Suffix appended to all output filenames (e.g. _numt_clean)")
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

# Features confirmed bad in P9d — excluded from final model
DROP_FEATURES = {"packing_x_idr", "rrach_per_cds_kb"}

EPS_RMSF = 0.1
EPS_CDS  = 1.0


def add_engineered_features(df):
    """Add only the two validated engineered features."""
    if "rmsf_nterm50" in df.columns and "rmsf_cterm50" in df.columns:
        df["rmsf_nterm_cterm_ratio"] = (
            df["rmsf_nterm50"] / (df["rmsf_cterm50"] + EPS_RMSF)
        )
    if "utr3_au_content" in df.columns and "utr3_length" in df.columns:
        df["utr3_au_x_length"] = df["utr3_au_content"] * df["utr3_length"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, split_name, threshold=0.5):
    y_pred  = (y_prob >= threshold).astype(int)
    auroc   = roc_auc_score(y_true, y_prob)
    auprc   = average_precision_score(y_true, y_prob)
    f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_pos  = f1_score(y_true, y_pred, pos_label=1,     zero_division=0)
    f1_neg  = f1_score(y_true, y_pred, pos_label=0,     zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  [{split_name}] threshold={threshold:.2f}")
    print(f"    AUROC  : {auroc:.4f}")
    print(f"    AUPRC  : {auprc:.4f}")
    print(f"    F1-pos : {f1_pos:.4f}")
    print(f"    F1-neg : {f1_neg:.4f}")
    print(f"    F1-mac : {f1_mac:.4f}")
    print(f"    CM: TN={tn:4d} FP={fp:4d} / FN={fn:4d} TP={tp:4d}")

    return dict(split=split_name, threshold=threshold,
                auroc=auroc, auprc=auprc,
                f1_macro=f1_mac, f1_pos=f1_pos, f1_neg=f1_neg,
                tn=tn, fp=fp, fn=fn, tp=tp)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(base_rf, X_train, feature_cols, n_bg=500, seed=42):
    """Compute SHAP on the base RF (pre-calibration — TreeExplainer requires it)."""
    rng  = np.random.default_rng(seed)
    bg   = X_train[rng.choice(len(X_train), min(n_bg, len(X_train)),
                               replace=False)]
    expl = shap.TreeExplainer(base_rf, data=bg,
                               feature_perturbation="interventional")
    sv   = expl.shap_values(X_train)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("ZORC — Phase 9f: Final Model")
    print(f"Config: {args.config}")
    print(f"Timestamp: {ts}")
    print("=" * 70)

    base_dir    = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    results_dir = base_dir / "results"
    logs_dir    = base_dir / "logs"

    suffix = args.output_suffix

    if args.feature_matrix:
        fm_path = Path(args.feature_matrix)
        if not fm_path.is_absolute():
            fm_path = base_dir / fm_path
    else:
        fm_path = base_dir / cfg.get("outputs", {}).get(
                        "feature_matrix", "data/processed/08_zorc_feature_matrix.csv")
    hc_path  = base_dir / cfg.get("outputs", {}).get(
                    "hc_set", "data/processed/colleague_high_confidence_set.csv")

    # ── 1. Load + engineer features ────────────────────────────────────────
    print(f"\n[1/6] Loading feature matrix...")
    df = pd.read_csv(fm_path)
    if "geneID" in df.columns:
        df = df.rename(columns={"geneID": "gene_id"})

    df = add_engineered_features(df)

    feature_cols = [c for c in df.columns
                    if c not in META_COLS and c not in DROP_FEATURES]
    print(f"  Samples  : {len(df):,}")
    print(f"  Features : {len(feature_cols)} "
          f"(dropped: {sorted(DROP_FEATURES)})")
    print(f"  Engineered kept: rmsf_nterm_cterm_ratio, utr3_au_x_length")

    mask_train = df["split"] == "train"
    mask_val   = df["split"] == "val"
    mask_test  = df["split"] == "test"

    X = df[feature_cols].values.astype(np.float32)
    y = df["class"].values.astype(int)

    X_train, y_train = X[mask_train], y[mask_train]
    X_val,   y_val   = X[mask_val],   y[mask_val]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    imputer = SimpleImputer(strategy="median")
    X_tr  = imputer.fit_transform(X_train)
    X_va  = imputer.transform(X_val)
    X_te  = imputer.transform(X_test)

    print(f"  Train={mask_train.sum()} Val={mask_val.sum()} "
          f"Test={mask_test.sum()}")

    # ── 2. Class weight sweep on val ───────────────────────────────────────
    print(f"\n[2/6] Class weight sweep (val set)...")
    weights_to_test = [
        {"label": "balanced (1.43×)", "cw": "balanced"},
        {"label": "neg=1.5×",         "cw": {0: 1.5, 1: 1.0}},
        {"label": "neg=2.0×",         "cw": {0: 2.0, 1: 1.0}},
        {"label": "neg=2.5×",         "cw": {0: 2.5, 1: 1.0}},
    ]

    print(f"\n  {'Weight':<20} {'Val F1-mac':>10} {'Val F1-pos':>10} "
          f"{'Val F1-neg':>10} {'OOB':>7}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    best_cw    = "balanced"
    best_f1mac = 0.0

    for w in weights_to_test:
        rf_tmp = RandomForestClassifier(
            n_estimators=300, class_weight=w["cw"],
            max_features="sqrt", min_samples_leaf=2,
            oob_score=True, random_state=42, n_jobs=-1,
        )
        rf_tmp.fit(X_tr, y_train)
        prob_va = rf_tmp.predict_proba(X_va)[:, 1]
        pred_va = (prob_va >= 0.5).astype(int)
        f1m  = f1_score(y_val, pred_va, average="macro", zero_division=0)
        f1p  = f1_score(y_val, pred_va, pos_label=1,     zero_division=0)
        f1n  = f1_score(y_val, pred_va, pos_label=0,     zero_division=0)
        oob  = rf_tmp.oob_score_
        print(f"  {w['label']:<20} {f1m:>10.4f} {f1p:>10.4f} "
              f"{f1n:>10.4f} {oob:>7.4f}")
        if f1m > best_f1mac:
            best_f1mac = f1m
            best_cw    = w["cw"]
            best_label = w["label"]

    print(f"\n  → Best class weight on val: {best_label}")

    # ── 3. Train final base RF with best class weight ──────────────────────
    print(f"\n[3/6] Training final RF (n_estimators=500, best class weight)...")
    rf_base = RandomForestClassifier(
        n_estimators=500, class_weight=best_cw,
        max_features="sqrt", min_samples_leaf=2,
        oob_score=True, random_state=42, n_jobs=-1,
    )
    rf_base.fit(X_tr, y_train)
    print(f"  OOB score: {rf_base.oob_score_:.4f}")

    # ── 4. Platt calibration (cv=5 on training set) ───────────────────────
    print(f"\n[4/6] Platt calibration (sigmoid, cv=5)...")
    print(f"  This may take ~5-10 minutes...")
    rf_cal = CalibratedClassifierCV(
        estimator=rf_base,
        method="sigmoid",
        cv=5,
    )
    rf_cal.fit(X_tr, y_train)
    print(f"  Calibration complete.")

    # Calibration quality check on val
    prob_va_raw = rf_base.predict_proba(X_va)[:, 1]
    prob_va_cal = rf_cal.predict_proba(X_va)[:, 1]

    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_val, prob_va_raw, n_bins=10, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_val, prob_va_cal, n_bins=10, strategy="uniform")

    # Brier score (lower = better calibration)
    brier_raw = np.mean((prob_va_raw - y_val) ** 2)
    brier_cal = np.mean((prob_va_cal - y_val) ** 2)
    print(f"\n  Calibration quality (val set):")
    print(f"    Brier score — uncalibrated: {brier_raw:.4f}")
    print(f"    Brier score — calibrated  : {brier_cal:.4f}  "
          f"({'better' if brier_cal < brier_raw else 'worse'})")

    # ── 5. Evaluation — uncalibrated vs calibrated ─────────────────────────
    print(f"\n[5/6] Evaluation...")

    print(f"\n  --- Uncalibrated RF (base) ---")
    evaluate(y_val,  rf_base.predict_proba(X_va)[:, 1], "VAL  uncal")
    res_test_raw = evaluate(
        y_test, rf_base.predict_proba(X_te)[:, 1], "TEST uncal")

    print(f"\n  --- Calibrated RF ---")
    evaluate(y_val,  rf_cal.predict_proba(X_va)[:, 1], "VAL  cal")
    res_test_cal = evaluate(
        y_test, rf_cal.predict_proba(X_te)[:, 1], "TEST cal")

    # HC validation
    print(f"\n  High-confidence validation (n=25):")
    hc_df    = pd.read_csv(hc_path)
    hc_col   = next((c for c in ["gene_id", "geneID"] if c in hc_df.columns),
                    hc_df.columns[0])
    hc_genes = hc_df[hc_col].tolist()
    hc_mask  = df["gene_id"].isin(hc_genes)
    df_hc    = df[hc_mask].copy()
    X_hc     = imputer.transform(
                    df_hc[feature_cols].values.astype(np.float32))
    y_hc     = df_hc["class"].values.astype(int)

    prob_hc_raw = rf_base.predict_proba(X_hc)[:, 1]
    prob_hc_cal = rf_cal.predict_proba(X_hc)[:, 1]
    ok_raw = ((prob_hc_raw >= 0.5).astype(int) == y_hc).sum()
    ok_cal = ((prob_hc_cal >= 0.5).astype(int) == y_hc).sum()
    print(f"    Uncalibrated: {ok_raw}/25 ({ok_raw/25*100:.0f}%)")
    print(f"    Calibrated  : {ok_cal}/25 ({ok_cal/25*100:.0f}%)")

    # ── Label noise analysis ───────────────────────────────────────────────
    print(f"\n  Label noise candidates (negatives with P(pos) 0.35–0.50):")
    df_test = df[mask_test].copy()
    prob_te_cal = rf_cal.predict_proba(X_te)[:, 1]
    df_test["prob_pos"] = prob_te_cal
    suspicious = df_test[
        (df_test["class"] == 0) &
        (df_test["prob_pos"] >= 0.35) &
        (df_test["prob_pos"] <= 0.50)
    ][["gene_id", "prob_pos"]].sort_values("prob_pos", ascending=False)
    print(f"    {len(suspicious)} genes in test set (potential mislabelled negatives)")
    if len(suspicious) > 0:
        print(suspicious.head(10).to_string(index=False))

    # ── 6. SHAP (on base RF — TreeExplainer requires non-calibrated) ───────
    print(f"\n[6/6] Computing SHAP values (base RF)...")
    imp = compute_shap(rf_base, X_tr, feature_cols)

    print(f"\n  Top 20 features (final model):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'-'*6} {'-'*35} {'-'*12}")
    for _, row in imp.head(20).iterrows():
        eng = " ◄" if row["feature"] in \
              {"rmsf_nterm_cterm_ratio", "utr3_au_x_length"} else ""
        print(f"  {int(row['rank']):<6} {row['feature']:<35} "
              f"{row['mean_abs_shap']:>12.4f}{eng}")

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(results_dir / f"09f_rf_final_model{suffix}.pkl", "wb") as f:
        pickle.dump(rf_cal, f)
    with open(results_dir / f"09f_rf_base_model{suffix}.pkl", "wb") as f:
        pickle.dump(rf_base, f)
    imp.to_csv(results_dir / f"09f_shap_final{suffix}.csv", index=False)

    # Predictions all splits
    pred_df = df[["gene_id", "transcript_id", "class", "split"]].copy()
    prob_all = np.full(len(df), np.nan)
    for mask, X_imp in [(mask_train, X_tr), (mask_val, X_va), (mask_test, X_te)]:
        prob_all[np.where(mask)[0]] = rf_cal.predict_proba(X_imp)[:, 1]
    pred_df["prob_pos"]  = prob_all
    pred_df["pred"]      = (pred_df["prob_pos"] >= 0.5).astype("Int64")
    pred_df.to_csv(results_dir / f"09f_predictions_final{suffix}.csv", index=False)

    # ── Report ──────────────────────────────────────────────────────────────
    report_path = logs_dir / f"09f_final_model_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(f"ZORC Phase 9f — Final Model Report\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"Features: {len(feature_cols)} "
                f"(dropped: {sorted(DROP_FEATURES)})\n")
        f.write(f"Best class weight: {best_label}\n")
        f.write(f"Calibration: Platt sigmoid, cv=5\n\n")
        f.write(f"Brier score uncal/cal: {brier_raw:.4f} / {brier_cal:.4f}\n\n")
        f.write(f"TEST uncalibrated:\n")
        for k, v in res_test_raw.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTEST calibrated:\n")
        for k, v in res_test_cal.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nHC: uncal={ok_raw}/25  cal={ok_cal}/25\n\n")
        f.write("Top 30 SHAP:\n")
        for _, row in imp.head(30).iterrows():
            f.write(f"  {int(row['rank']):2d}  {row['feature']:<35}  "
                    f"{row['mean_abs_shap']:.4f}\n")
    print(f"\n  Report: {report_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"  Features          : {len(feature_cols)} "
          f"(clean set, 2 engineered)")
    print(f"  Class weight      : {best_label}")
    print(f"  Calibration       : Platt sigmoid cv=5")
    print(f"  OOB score         : {rf_base.oob_score_:.4f}")
    print()
    print(f"  {'Model':<16} {'AUROC':>7} {'AUPRC':>7} "
          f"{'F1-mac':>7} {'F1-pos':>7} {'F1-neg':>7}  HC")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  --")
    for res, label in [(res_test_raw, "RF uncal (9f)"),
                       (res_test_cal, "RF cal   (9f)")]:
        hc = ok_raw if "uncal" in label else ok_cal
        print(f"  {label:<16} {res['auroc']:>7.4f} {res['auprc']:>7.4f} "
              f"{res['f1_macro']:>7.4f} {res['f1_pos']:>7.4f} "
              f"{res['f1_neg']:>7.4f}  {hc}/25")
    print()
    print(f"  Baseline (P9d RF, all 4 eng features):")
    print(f"  Test AUROC=0.7963  AUPRC=0.8431  F1-mac=0.7229  "
          f"F1-neg=0.6554  HC=24/25")
    print(f"{'='*70}")
    print("\n✓ Phase 9f complete.")


if __name__ == "__main__":
    main()
