#!/usr/bin/env python3
"""
ZORC — Phase 9d: Feature Engineering
======================================
Adds biologically motivated interaction and ratio features to the
feature matrix, based on SHAP analysis from RF (P9) and XGBoost (P9b)
post-BioEmu Tier1 completion.

Rationale per feature:
  rmsf_nterm_cterm_ratio
    rmsf_nterm50 (RF rank 11) and rmsf_cterm50 (RF rank 15) contribute
    individually. Their ratio captures N/C-terminal flexibility asymmetry —
    proteins with flexible N-termini relative to C-termini may expose
    interaction motifs relevant for P-body condensation.
    Guard: epsilon=0.1 avoids div/0 for the 19 proteins where both=0.

  packing_x_idr
    packing_density (XGB rank 4) × idr_percent. Less-packed proteins with
    more disordered regions may have greater multivalent interaction capacity.
    The product is zero for IDR=0 (Tier1 structured proteins) — biologically
    meaningful, not an artefact.

  rrach_per_cds_kb
    rrach_count / (cds_length / 1000). m6A site density normalised by CDS
    length specifically (not total mRNA length). rrach_count is XGB rank 6;
    cds_length is RF rank 2. Their ratio may decouple the m6A signal from the
    length confound.
    Guard: epsilon=1 nt on cds_length avoids div/0 for 44 QC-flagged zeros.

  utr3_au_x_length
    utr3_au_content (RF rank 6, XGB rank 2) × utr3_length (RF rank 7).
    Product captures absolute AU-rich content: a long 3'UTR with high AU
    fraction has more ARE (AU-rich elements) than either feature alone.
    No guards needed — both features are clean.

Outputs:
    data/processed/08_zorc_feature_matrix_eng.csv   enriched feature matrix
    results/09d_shap_rf_eng.csv                     RF SHAP importance
    results/09d_shap_xgb_eng.csv                    XGBoost SHAP importance
    results/09d_rf_eng_model.pkl                    RF model
    results/09d_xgb_eng_model.json                  XGBoost model
    logs/09d_feature_engineering_report.txt

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09d_feature_engineering.py --config config/zorc_config.yaml
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import shap
import yaml
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9d — Feature Engineering")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

EPS_RMSF  = 0.1   # Å — avoids div/0 when both N and C-term RMSF = 0
EPS_CDS   = 1.0   # nt — avoids div/0 for QC-flagged cds_length=0

def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add 4 engineered features to the dataframe.
    Returns (df_with_new_features, list_of_new_feature_names).
    """
    new_cols = []

    # 1. rmsf_nterm_cterm_ratio — N/C terminal flexibility asymmetry
    if "rmsf_nterm50" in df.columns and "rmsf_cterm50" in df.columns:
        df["rmsf_nterm_cterm_ratio"] = (
            df["rmsf_nterm50"] / (df["rmsf_cterm50"] + EPS_RMSF)
        )
        new_cols.append("rmsf_nterm_cterm_ratio")

    # 2. packing_x_idr — packing density × IDR fraction
    if "packing_density" in df.columns and "idr_percent" in df.columns:
        df["packing_x_idr"] = (
            df["packing_density"] * (df["idr_percent"] / 100.0)
        )
        new_cols.append("packing_x_idr")

    # 3. rrach_per_cds_kb — m6A density per CDS kilobase
    if "rrach_count" in df.columns and "cds_length" in df.columns:
        df["rrach_per_cds_kb"] = (
            df["rrach_count"] / ((df["cds_length"] + EPS_CDS) / 1000.0)
        )
        new_cols.append("rrach_per_cds_kb")

    # 4. utr3_au_x_length — absolute AU-rich content in 3'UTR
    if "utr3_au_content" in df.columns and "utr3_length" in df.columns:
        df["utr3_au_x_length"] = df["utr3_au_content"] * df["utr3_length"]
        new_cols.append("utr3_au_x_length")

    return df, new_cols


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, split_name):
    y_pred = (y_prob >= 0.5).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  [{split_name}] n={len(y_true)}")
    print(f"    AUROC  : {auroc:.4f}")
    print(f"    AUPRC  : {auprc:.4f}")
    print(f"    F1 pos : {f1_pos:.4f}")
    print(f"    F1 neg : {f1_neg:.4f}")
    print(f"    F1 macro: {f1_mac:.4f}")
    print(f"    CM: TN={tn:4d} FP={fp:4d} / FN={fn:4d} TP={tp:4d}")

    return dict(split=split_name, auroc=auroc, auprc=auprc,
                f1_macro=f1_mac, f1_pos=f1_pos, f1_neg=f1_neg,
                tn=tn, fp=fp, fn=fn, tp=tp)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(model, X_train, feature_names,
                             n_bg=500, seed=42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    bg    = X_train[rng.choice(len(X_train), min(n_bg, len(X_train)),
                               replace=False)]
    expl  = shap.TreeExplainer(model, data=bg,
                                feature_perturbation="interventional")
    sv    = expl.shap_values(X_train)
    # Handle all SHAP output formats:
    # - list of 2 arrays (old shap + RF binary): take class 1
    # - 3D array (n_samples, n_features, n_classes): take [..., 1]
    # - 2D array (XGBoost binary): use directly
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    return imp


def print_top20(imp: pd.DataFrame, title: str):
    print(f"\n  {title}")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'-'*6} {'-'*35} {'-'*12}")
    for _, row in imp.head(20).iterrows():
        marker = " ◄ NEW" if row["feature"] in NEW_FEATURES else ""
        print(f"  {int(row['rank']):<6} {row['feature']:<35} "
              f"{row['mean_abs_shap']:>12.4f}{marker}")


NEW_FEATURES = []  # filled after add_engineered_features()

# RF baseline SHAP ranks (post-P6b, P9 run) for comparison
RF_BASELINE = {
    "mrna_length": 1, "cds_length": 2, "n_residues": 3,
    "di_CG": 4, "di_UA": 5, "utr3_au_content": 6,
    "utr3_length": 7, "di_UG": 8, "utr5_fraction": 9,
    "utr5_length": 10, "rmsf_nterm50": 11, "cds_fraction": 12,
    "rmsf_max": 13, "rmsf_std": 14, "fG": 15,
    "fU": 16, "di_CA": 17, "di_CU": 18,
    "rrach_per_kb": 19, "mfe_per_nt": 20,
}


def print_eng_diagnostic(imp_rf, imp_xgb, new_cols):
    print("\n  ──────────────────────────────────────────────────────────")
    print("  ENGINEERED FEATURES — rank in RF and XGBoost")
    print(f"  {'Feature':<30} {'RF rank':>8} {'XGB rank':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*9}")
    rf_map  = dict(zip(imp_rf["feature"],  imp_rf["rank"]))
    xgb_map = dict(zip(imp_xgb["feature"], imp_xgb["rank"]))
    for feat in new_cols:
        rf_r  = rf_map.get(feat,  "n/a")
        xgb_r = xgb_map.get(feat, "n/a")
        print(f"  {feat:<30} {str(rf_r):>8} {str(xgb_r):>9}")
    print()

    print("  PARENT FEATURE RANK SHIFT (RF baseline P9 → RF post-engineering)")
    parents = {
        "rmsf_nterm50": None, "rmsf_cterm50": None,
        "packing_density": None, "idr_percent": None,
        "rrach_count": None, "cds_length": None,
        "utr3_au_content": None, "utr3_length": None,
    }
    print(f"  {'Feature':<28} {'RF base':>8} {'RF eng':>8} {'Δ':>6}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6}")
    for feat in parents:
        base = RF_BASELINE.get(feat, ">20")
        eng  = rf_map.get(feat, "n/a")
        if isinstance(base, int) and isinstance(eng, (int, float)):
            delta = f"{int(base) - int(eng):+d}"
        else:
            delta = "n/a"
        print(f"  {feat:<28} {str(base):>8} {str(eng):>8} {delta:>6}")
    print("  ──────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

META_COLS = [
    "geneID", "gene_id", "transcript_id", "class",
    "condition", "qc_fail", "isoform_source", "event_type",
    "bioemu_tier", "bioemu_status", "feature_source",
    "cluster_id", "split",
]


def main():
    global NEW_FEATURES

    args = parse_args()
    cfg  = load_config(args.config)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("ZORC — Phase 9d: Feature Engineering")
    print(f"Config: {args.config}")
    print(f"Timestamp: {ts}")
    print("=" * 70)

    base_dir    = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    data_dir    = base_dir / "data" / "processed"
    results_dir = base_dir / "results"
    logs_dir    = base_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    fm_path  = base_dir / cfg.get("outputs", {}).get(
                    "feature_matrix", "data/processed/08_zorc_feature_matrix.csv")
    hc_path  = base_dir / cfg.get("outputs", {}).get(
                    "hc_set", "data/processed/colleague_high_confidence_set.csv")

    # ── 1. Load and engineer features ─────────────────────────────────────
    print(f"\n[1/5] Loading feature matrix: {fm_path.name}")
    df = pd.read_csv(fm_path)
    if "geneID" in df.columns:
        df = df.rename(columns={"geneID": "gene_id"})
    print(f"  {len(df):,} samples, {df.shape[1]} columns")

    print(f"\n[2/5] Engineering features...")
    df, new_cols = add_engineered_features(df)
    NEW_FEATURES = new_cols

    print(f"  Added {len(new_cols)} features: {new_cols}")
    for feat in new_cols:
        s = df[feat].dropna()
        print(f"    {feat:<30} min={s.min():.3f}  "
              f"median={s.median():.3f}  max={s.max():.3f}  "
              f"nan={df[feat].isna().sum()}")

    # Class separation for new features
    print(f"\n  Class separation (new features):")
    print(f"  {'Feature':<30} {'pos mean':>10} {'neg mean':>10} {'ratio':>7}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*7}")
    for feat in new_cols:
        pos_m = df.loc[df["class"] == 1, feat].mean()
        neg_m = df.loc[df["class"] == 0, feat].mean()
        ratio = pos_m / neg_m if neg_m != 0 else float("inf")
        print(f"  {feat:<30} {pos_m:>10.4f} {neg_m:>10.4f} {ratio:>7.3f}×")

    # Save enriched feature matrix
    eng_fm_path = data_dir / "08_zorc_feature_matrix_eng.csv"
    df_save = df.rename(columns={"gene_id": "geneID"})
    df_save.to_csv(eng_fm_path, index=False)
    print(f"\n  Enriched matrix saved: {eng_fm_path.name}")
    print(f"  Shape: {df.shape[0]} × {df.shape[1]} "
          f"({df.shape[1] - df_save.shape[1] + df.shape[1]} total cols)")

    # ── 3. Prepare train/val/test splits ───────────────────────────────────
    print(f"\n[3/5] Preparing splits...")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Feature columns: {len(feature_cols)} "
          f"(+{len(new_cols)} engineered)")

    mask_train = df["split"] == "train"
    mask_val   = df["split"] == "val"
    mask_test  = df["split"] == "test"

    X = df[feature_cols].values.astype(np.float32)
    y = df["class"].values.astype(int)

    X_train, y_train = X[mask_train], y[mask_train]
    X_val,   y_val   = X[mask_val],   y[mask_val]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    # RF imputation (global median) — same as P9
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_rf = imputer.fit_transform(X_train)
    X_val_rf   = imputer.transform(X_val)
    X_test_rf  = imputer.transform(X_test)

    print(f"  Train: {mask_train.sum()}  Val: {mask_val.sum()}  "
          f"Test: {mask_test.sum()}")

    # ── 4. RF with engineered features ─────────────────────────────────────
    print(f"\n[4/5] Training models with engineered features...")
    print(f"\n  --- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        max_features="sqrt", min_samples_leaf=2,
        oob_score=True, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_rf, y_train)
    print(f"  OOB score: {rf.oob_score_:.4f}")

    rf_results = []
    rf_results.append(evaluate(
        y_train, rf.predict_proba(X_train_rf)[:, 1], "TRAIN"))
    rf_results.append(evaluate(
        y_val,   rf.predict_proba(X_val_rf)[:, 1],   "VAL"))
    rf_results.append(evaluate(
        y_test,  rf.predict_proba(X_test_rf)[:, 1],  "TEST"))

    print(f"\n  Computing RF SHAP values...")
    imp_rf = compute_shap_importance(rf, X_train_rf, feature_cols)
    print_top20(imp_rf, "RF Top 20 (post-engineering):")

    # ── XGBoost with engineered features ───────────────────────────────────
    print(f"\n  --- XGBoost ---")
    xgb_cfg    = cfg.get("xgboost", {})
    scale_pw   = float((y_train == 0).sum()) / float((y_train == 1).sum())

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "auc"],
        "max_depth":        int(xgb_cfg.get("max_depth",       6)),
        "learning_rate":    float(xgb_cfg.get("learning_rate", 0.05)),
        "min_child_weight": 2,          # tuned: was 5, now 2
        "subsample":        float(xgb_cfg.get("subsample",     0.8)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
        "scale_pos_weight": scale_pw,
        "seed":             int(xgb_cfg.get("random_state",    42)),
        "tree_method":      "hist",
        "device":           "cuda",
    }
    early_stop = 50   # tuned: was 30, now 50

    evals_result = {}
    booster = xgb.train(
        params, dtrain,
        num_boost_round=int(xgb_cfg.get("n_estimators", 500)),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stop,
        evals_result=evals_result,
        verbose_eval=50,
    )
    print(f"\n  Best round: {booster.best_iteration + 1}  "
          f"Val AUC: {max(evals_result['val']['auc']):.4f}")

    xgb_results = []
    for split_name, dmat, y_true in [
        ("TRAIN", dtrain, y_train),
        ("VAL",   dval,   y_val),
        ("TEST",  dtest,  y_test),
    ]:
        prob = booster.predict(dmat)
        xgb_results.append(evaluate(y_true, prob, split_name))

    print(f"\n  Computing XGBoost SHAP values...")
    imp_xgb = compute_shap_importance(booster, X_train, feature_cols)
    print_top20(imp_xgb, "XGBoost Top 20 (post-engineering):")

    # ── Diagnostic ──────────────────────────────────────────────────────────
    print_eng_diagnostic(imp_rf, imp_xgb, new_cols)

    # ── HC validation ────────────────────────────────────────────────────────
    print(f"\n[5/5] High-confidence validation (n=25)...")
    hc_df    = pd.read_csv(hc_path)
    hc_col   = "gene_id" if "gene_id" in hc_df.columns else \
               "geneID"  if "geneID"  in hc_df.columns else \
               hc_df.columns[0]
    hc_genes = hc_df[hc_col].tolist()

    hc_mask  = df["gene_id"].isin(hc_genes)
    df_hc    = df[hc_mask].copy()
    X_hc_raw = df_hc[feature_cols].values.astype(np.float32)
    y_hc     = df_hc["class"].values.astype(int)

    # RF HC
    X_hc_rf   = imputer.transform(X_hc_raw)
    prob_hc_rf = rf.predict_proba(X_hc_rf)[:, 1]
    pred_hc_rf = (prob_hc_rf >= 0.5).astype(int)
    ok_rf      = (pred_hc_rf == y_hc).sum()
    print(f"  RF  : {ok_rf}/{len(y_hc)} ({ok_rf/len(y_hc)*100:.0f}%)")

    # XGB HC
    dhc        = xgb.DMatrix(X_hc_raw, feature_names=feature_cols)
    prob_hc_xgb = booster.predict(dhc)
    pred_hc_xgb = (prob_hc_xgb >= 0.5).astype(int)
    ok_xgb      = (pred_hc_xgb == y_hc).sum()
    print(f"  XGB : {ok_xgb}/{len(y_hc)} ({ok_xgb/len(y_hc)*100:.0f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    imp_rf.to_csv(results_dir / "09d_shap_rf_eng.csv", index=False)
    imp_xgb.to_csv(results_dir / "09d_shap_xgb_eng.csv", index=False)

    with open(results_dir / "09d_rf_eng_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    booster.save_model(str(results_dir / "09d_xgb_eng_model.json"))

    # ── Report ────────────────────────────────────────────────────────────────
    rf_val  = [r for r in rf_results  if r["split"] == "VAL"][0]
    rf_test = [r for r in rf_results  if r["split"] == "TEST"][0]
    xb_val  = [r for r in xgb_results if r["split"] == "VAL"][0]
    xb_test = [r for r in xgb_results if r["split"] == "TEST"][0]

    report_path = logs_dir / "09d_feature_engineering_report.txt"
    with open(report_path, "w") as f:
        f.write(f"ZORC Phase 9d — Feature Engineering Report\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"Engineered features: {new_cols}\n\n")
        f.write(f"RF  Val  AUROC={rf_val['auroc']:.4f}  "
                f"AUPRC={rf_val['auprc']:.4f}  "
                f"F1={rf_val['f1_macro']:.4f}\n")
        f.write(f"RF  Test AUROC={rf_test['auroc']:.4f}  "
                f"AUPRC={rf_test['auprc']:.4f}  "
                f"F1={rf_test['f1_macro']:.4f}\n")
        f.write(f"XGB Val  AUROC={xb_val['auroc']:.4f}  "
                f"AUPRC={xb_val['auprc']:.4f}  "
                f"F1={xb_val['f1_macro']:.4f}\n")
        f.write(f"XGB Test AUROC={xb_test['auroc']:.4f}  "
                f"AUPRC={xb_test['auprc']:.4f}  "
                f"F1={xb_test['f1_macro']:.4f}\n\n")
        f.write(f"RF  HC: {ok_rf}/{len(y_hc)}\n")
        f.write(f"XGB HC: {ok_xgb}/{len(y_hc)}\n\n")
        f.write("RF Top 30 SHAP:\n")
        for _, row in imp_rf.head(30).iterrows():
            f.write(f"  {int(row['rank']):2d}  {row['feature']:<35}  "
                    f"{row['mean_abs_shap']:.4f}\n")
        f.write("\nXGB Top 30 SHAP:\n")
        for _, row in imp_xgb.head(30).iterrows():
            f.write(f"  {int(row['rank']):2d}  {row['feature']:<35}  "
                    f"{row['mean_abs_shap']:.4f}\n")

    print(f"\n  Report: {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'='*70}")
    print(f"  Engineered features   : {new_cols}")
    print(f"  Total feature columns : {len(feature_cols)}")
    print()
    print(f"  {'Model':<8} {'Split':<6} {'AUROC':>7} {'AUPRC':>7} "
          f"{'F1-mac':>7} {'HC':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for tag, res, hc_ok in [
        ("RF",  rf_results,  ok_rf),
        ("XGB", xgb_results, ok_xgb),
    ]:
        for r in res:
            if r["split"] in ("VAL", "TEST"):
                hc_str = f"{hc_ok}/25" if r["split"] == "TEST" else ""
                print(f"  {tag:<8} {r['split']:<6} "
                      f"{r['auroc']:>7.4f} {r['auprc']:>7.4f} "
                      f"{r['f1_macro']:>7.4f} {hc_str:>6}")
    print()
    print("  Baseline (post-P6b, no engineering):")
    print("  RF  Val  AUROC=0.8035  AUPRC=0.8294  F1=0.7188")
    print("  RF  Test AUROC=0.7878  AUPRC=0.8327  F1=0.6942")
    print("  XGB Val  AUROC=0.8001  AUPRC=0.8229  F1=0.7271")
    print("  XGB Test AUROC=0.7879  AUPRC=0.8317  F1=0.6976")
    print(f"{'='*70}")
    print("\n✓ Phase 9d complete.")


if __name__ == "__main__":
    main()
