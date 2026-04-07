#!/usr/bin/env python3
"""
09_random_forest.py
===================
ZORC — Zip-code Of RNAs that Condense
Phase 9: Random Forest classifier + SHAP feature importance analysis.

Trains a Random Forest to predict P-body RNA enrichment (class=1) vs
depletion (class=0) from the 59-feature matrix assembled in P8.

Pipeline:
  1. Load feature matrix (P8) with train/val/test split assignments
  2. Impute missing values:
     - Dynamic BioEmu features (rmsf_*, rg_*, pass_rate, contact_density,
       packing_density, sasa_per_residue): global median imputation
     - Add binary flag `has_bioemu` (1=BioEmu completed, 0=Tier1/excluded)
  3. Train Random Forest on train split (class_weight='balanced')
  4. Evaluate on val and test splits:
     - AUROC, AUPRC, F1 per class, confusion matrix
  5. SHAP feature importance (TreeExplainer, train set)
  6. Validate on high-confidence set (25 colleague genes)
  7. Write model, SHAP values, evaluation report

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09_random_forest.py --config config/zorc_config.yaml

    # Quick test with reduced trees
    python scripts/09_random_forest.py --config config/zorc_config.yaml --n-estimators 100

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

def ensure_output_keys(cfg, config_path):
    defaults = {
        'rf_model':        'results/09_zorc_rf_model.pkl',
        'shap_values':     'results/09_zorc_shap_values.csv',
        'rf_report':       'logs/09_rf_report.txt',
        'rf_predictions':  'results/09_zorc_predictions.csv',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P9 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    return Path(os.path.expanduser(cfg['project_dir'])) / cfg['outputs'][key]


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Dynamic BioEmu features — will be imputed + flagged
DYNAMIC_FEATURES = [
    'rmsf_mean', 'rmsf_std', 'rmsf_max', 'rmsf_nterm50', 'rmsf_cterm50',
    'rg_mean', 'rg_std', 'rg_cv',
    'contact_density', 'packing_density', 'sasa_per_residue',
    'pass_rate',
]

# RNA features
RNA_FEATURES = [
    'mrna_length', 'fA', 'fU', 'fG', 'fC', 'au_content', 'gc_content',
    'di_AA', 'di_AU', 'di_AG', 'di_AC', 'di_UA', 'di_UU', 'di_UG', 'di_UC',
    'di_GA', 'di_GU', 'di_GG', 'di_GC', 'di_CA', 'di_CU', 'di_CG', 'di_CC',
    'utr5_length', 'utr3_length', 'cds_length',
    'utr5_fraction', 'utr3_fraction', 'cds_fraction',
    'utr3_au_content',
    'rrach_per_kb', 'aaach_per_kb',
    'mfe_per_nt', 'frac_paired', 'n_stemloops_per_kb',
    'long_3utr', 'ptc_proxy',
]

# Static protein / IDR features
STATIC_PROTEIN_FEATURES = [
    'idr_percent', 'mean_disorder', 'max_disorder_window',
    'n_idr_regions', 'longest_idr_region',
    'n_residues',
]


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare feature matrix:
    1. Add has_bioemu flag
    2. Impute dynamic features with global median
    3. Return X DataFrame and feature column list
    """
    df = df.copy()

    # Add has_bioemu flag BEFORE imputation
    df['has_bioemu'] = (df['bioemu_status'] == 'completed').astype(int)

    # Collect all feature columns present in df
    feature_cols = []
    for col in RNA_FEATURES + STATIC_PROTEIN_FEATURES + DYNAMIC_FEATURES + ['has_bioemu']:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()

    # Impute dynamic features with global median (computed on full dataset)
    for col in DYNAMIC_FEATURES:
        if col in X.columns and X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    # Impute any remaining NaN with column median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Final check — no NaN should remain
    n_remaining_nan = X.isna().sum().sum()
    if n_remaining_nan > 0:
        print(f"  WARNING: {n_remaining_nan} NaN values remain after imputation")
        X = X.fillna(0)

    return X, feature_cols


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_rf(X_train, y_train, n_estimators=500, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    print(f"  Training Random Forest (n_estimators={n_estimators}, "
          f"class_weight='balanced', seed={seed})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
        max_features='sqrt',
        min_samples_leaf=2,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    print(f"  OOB score: {rf.oob_score_:.4f}")
    return rf


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, X, y, split_name: str) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, confusion_matrix, classification_report
    )
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auroc = roc_auc_score(y, y_prob)
    auprc = average_precision_score(y, y_prob)
    f1_pos = f1_score(y, y_pred, pos_label=1)
    f1_neg = f1_score(y, y_pred, pos_label=0)
    f1_macro = f1_score(y, y_pred, average='macro')
    cm = confusion_matrix(y, y_pred)

    print(f"\n  [{split_name}] n={len(y)}")
    print(f"    AUROC  : {auroc:.4f}")
    print(f"    AUPRC  : {auprc:.4f}")
    print(f"    F1 pos : {f1_pos:.4f}")
    print(f"    F1 neg : {f1_neg:.4f}")
    print(f"    F1 macro: {f1_macro:.4f}")
    print(f"    Confusion matrix (rows=true, cols=pred):")
    print(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return {
        'split':     split_name,
        'n':         len(y),
        'auroc':     round(auroc, 4),
        'auprc':     round(auprc, 4),
        'f1_pos':    round(f1_pos, 4),
        'f1_neg':    round(f1_neg, 4),
        'f1_macro':  round(f1_macro, 4),
        'tn': int(cm[0,0]), 'fp': int(cm[0,1]),
        'fn': int(cm[1,0]), 'tp': int(cm[1,1]),
    }


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def compute_shap(model, X_train, feature_cols, max_samples=500):
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis")
        print("  Install with: pip install shap")
        return None, None

    print(f"\n  Computing SHAP values (TreeExplainer, "
          f"n={min(max_samples, len(X_train))} samples)...")

    # Subsample for speed if needed
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_shap = X_train.iloc[idx]
    else:
        X_shap = X_train

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # shap_values shape depends on SHAP version:
    # - older: list of 2 arrays (n_samples, n_features)
    # - newer: 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature':        feature_cols,
        'mean_abs_shap':  mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    print(f"\n  Top 20 features by mean |SHAP|:")
    print(f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|':>12}")
    print(f"  {'-'*50}")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {i+1:<5} {row['feature']:<30} {row['mean_abs_shap']:>12.4f}")

    # Full SHAP matrix
    shap_df = pd.DataFrame(sv, columns=feature_cols)
    shap_df.insert(0, 'sample_idx', X_shap.index)

    return importance_df, shap_df


# =============================================================================
# HIGH-CONFIDENCE VALIDATION
# =============================================================================

def validate_high_confidence(model, X_all, df_all, feature_cols, proj):
    hc_path = proj / 'data/processed/colleague_high_confidence_set.csv'
    if not hc_path.exists():
        print("  High-confidence set not found, skipping.")
        return {}

    df_hc = pd.read_csv(hc_path)
    hc_ids = set(df_hc['geneID'])

    # Find these genes in the full dataset
    mask = df_all['geneID'].isin(hc_ids)
    X_hc = X_all[mask]
    y_hc = df_all.loc[mask, 'class']

    if len(X_hc) == 0:
        print("  No high-confidence genes found in feature matrix.")
        return {}

    y_prob = model.predict_proba(X_hc)[:, 1]
    y_pred = model.predict(X_hc)
    correct = (y_pred == y_hc).sum()

    print(f"\n  High-confidence validation set (n={len(X_hc)}):")
    print(f"    Correctly classified: {correct}/{len(X_hc)} "
          f"({100*correct/len(X_hc):.0f}%)")

    results = []
    for idx, (_, row) in enumerate(df_all[mask].iterrows()):
        results.append({
            'geneID':      row['geneID'],
            'true_class':  int(y_hc.iloc[idx]),
            'pred_class':  int(y_pred[idx]),
            'prob_pos':    round(float(y_prob[idx]), 4),
            'correct':     bool(y_pred[idx] == y_hc.iloc[idx]),
        })

    df_res = pd.DataFrame(results).sort_values('prob_pos', ascending=False)
    print(f"\n  {'GeneID':<14} {'True':>5} {'Pred':>5} {'P(pos)':>8} {'OK':>4}")
    print(f"  {'-'*40}")
    for _, r in df_res.iterrows():
        ok = '✓' if r['correct'] else '✗'
        print(f"  {r['geneID']:<14} {r['true_class']:>5} "
              f"{r['pred_class']:>5} {r['prob_pos']:>8.4f} {ok:>4}")

    return {
        'n': len(X_hc),
        'n_correct': int(correct),
        'accuracy': round(correct / len(X_hc), 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 9 — Random Forest + SHAP.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-estimators', type=int, default=None,
                        help='Override n_estimators from config')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP computation (faster)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 9: Random Forest + SHAP")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    seed         = cfg['ml']['random_seed']
    n_estimators = args.n_estimators or cfg['ml']['rf_n_estimators']
    np.random.seed(seed)

    # --- Load feature matrix -------------------------------------------------
    matrix_path = rp(cfg, 'feature_matrix')
    print(f"\n[1/6] Loading feature matrix: {matrix_path}")
    df = pd.read_csv(matrix_path)
    print(f"  {len(df):,} samples, {len(df.columns)} columns")
    print(f"  Classes: pos={( df['class']==1).sum():,}, "
          f"neg={(df['class']==0).sum():,}")

    # Filter to labeled samples only
    df = df[df['class'].isin([0, 1])].copy()
    df['class'] = df['class'].astype(int)

    # --- Prepare features ----------------------------------------------------
    print(f"\n[2/6] Preparing features (imputation + has_bioemu flag)...")
    X_all, feature_cols = prepare_features(df)
    y_all = df['class'].values

    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  has_bioemu=1: {X_all['has_bioemu'].sum():,} "
          f"({100*X_all['has_bioemu'].mean():.0f}%)")
    print(f"  NaN after imputation: {X_all.isna().sum().sum()}")

    # --- Split ---------------------------------------------------------------
    train_mask = df['split'] == 'train'
    val_mask   = df['split'] == 'val'
    test_mask  = df['split'] == 'test'

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_val   = X_all[val_mask]
    y_val   = y_all[val_mask]
    X_test  = X_all[test_mask]
    y_test  = y_all[test_mask]

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}")

    # --- Train ---------------------------------------------------------------
    print(f"\n[3/6] Training Random Forest...")
    rf = train_rf(X_train, y_train, n_estimators=n_estimators, seed=seed)

    # --- Evaluate ------------------------------------------------------------
    print(f"\n[4/6] Evaluation...")
    metrics = {}
    metrics['train'] = evaluate(rf, X_train, y_train, 'TRAIN')
    metrics['val']   = evaluate(rf, X_val,   y_val,   'VAL')
    metrics['test']  = evaluate(rf, X_test,  y_test,  'TEST')

    # Save predictions
    df_pred = df[['geneID', 'transcript_id', 'class', 'split',
                  'condition', 'bioemu_tier']].copy()
    df_pred['prob_pos'] = rf.predict_proba(X_all)[:, 1]
    df_pred['pred_class'] = rf.predict(X_all)
    df_pred['correct'] = (df_pred['pred_class'] == df_pred['class']).astype(int)
    pred_path = rp(cfg, 'rf_predictions')
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(pred_path, index=False)

    # --- SHAP ----------------------------------------------------------------
    importance_df, shap_df = None, None
    if not args.skip_shap:
        print(f"\n[5/6] SHAP feature importance...")
        importance_df, shap_df = compute_shap(rf, X_train, feature_cols)
        if shap_df is not None:
            shap_path = rp(cfg, 'shap_values')
            shap_path.parent.mkdir(parents=True, exist_ok=True)
            shap_df.to_csv(shap_path, index=False)
            importance_df.to_csv(
                str(shap_path).replace('shap_values', 'shap_importance'),
                index=False
            )
            print(f"  SHAP values saved: {shap_path}")
    else:
        print(f"\n[5/6] Skipping SHAP (--skip-shap)")

    # --- High-confidence validation ------------------------------------------
    print(f"\n[6/6] High-confidence validation...")
    hc_metrics = validate_high_confidence(rf, X_all, df, feature_cols, proj)

    # --- Save model ----------------------------------------------------------
    model_path = rp(cfg, 'rf_model')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':        rf,
            'feature_cols': feature_cols,
            'config':       cfg,
            'timestamp':    datetime.now().isoformat(),
        }, f)
    print(f"\n  Model saved: {model_path}")

    # --- Write report --------------------------------------------------------
    _write_report(metrics, hc_metrics, importance_df, feature_cols,
                  n_estimators, str(rp(cfg, 'rf_report')))

    # --- Final summary -------------------------------------------------------
    print(f"\n{'='*70}")
    print("RANDOM FOREST SUMMARY")
    print(f"{'='*70}")
    print(f"  Features          : {len(feature_cols)}")
    print(f"  OOB score         : {rf.oob_score_:.4f}")
    print(f"  VAL  AUROC        : {metrics['val']['auroc']:.4f}")
    print(f"  VAL  AUPRC        : {metrics['val']['auprc']:.4f}")
    print(f"  VAL  F1 macro     : {metrics['val']['f1_macro']:.4f}")
    print(f"  TEST AUROC        : {metrics['test']['auroc']:.4f}")
    print(f"  TEST AUPRC        : {metrics['test']['auprc']:.4f}")
    print(f"  TEST F1 macro     : {metrics['test']['f1_macro']:.4f}")
    if hc_metrics:
        print(f"  HC validation     : {hc_metrics['n_correct']}/"
              f"{hc_metrics['n']} ({100*hc_metrics['accuracy']:.0f}%)")
    print(f"{'='*70}")
    print("\n✓ Phase 9 complete.")
    return 0


def _write_report(metrics, hc_metrics, importance_df,
                  feature_cols, n_estimators, out_path):
    lines = [
        "=" * 80,
        "ZORC — Random Forest Report",
        "Script: 09_random_forest.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
        "MODEL CONFIGURATION",
        f"  n_estimators : {n_estimators}",
        f"  class_weight : balanced",
        f"  max_features : sqrt",
        f"  Features     : {len(feature_cols)}",
        f"  Imputation   : global median + has_bioemu flag",
        "",
        "EVALUATION METRICS",
    ]
    for split in ['train', 'val', 'test']:
        m = metrics[split]
        lines += [
            f"\n  {split.upper()} (n={m['n']})",
            f"    AUROC   : {m['auroc']:.4f}",
            f"    AUPRC   : {m['auprc']:.4f}",
            f"    F1 pos  : {m['f1_pos']:.4f}",
            f"    F1 neg  : {m['f1_neg']:.4f}",
            f"    F1 macro: {m['f1_macro']:.4f}",
            f"    TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}",
        ]

    if hc_metrics:
        lines += [
            "",
            "HIGH-CONFIDENCE VALIDATION",
            f"  {hc_metrics['n_correct']}/{hc_metrics['n']} correct "
            f"({100*hc_metrics['accuracy']:.0f}%)",
        ]

    if importance_df is not None:
        lines += ["", "TOP 20 FEATURES (mean |SHAP|)",
                  f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}"]
        for i, row in importance_df.head(20).iterrows():
            lines.append(f"  {i+1:<5} {row['feature']:<30} "
                         f"{row['mean_abs_shap']:.4f}")

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
