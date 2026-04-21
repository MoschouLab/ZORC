#!/usr/bin/env python3
"""
ZORC — Phase 9g: Length Baseline
==================================
Trains a Random Forest using ONLY mrna_length and cds_length as features.
Provides a length-only baseline AUROC for reviewer comparison against the
full 63-feature model (P9d: AUROC 0.7963 val / 0.7963 test).

Rationale: mrna_length and cds_length are the top-2 SHAP features in the
full model. A reviewer may ask whether the model merely learns a length
bias. This script quantifies the length signal in isolation so the
improvement from adding 61 biological features is explicitly documented.

Outputs:
    results/09g_length_baseline_model.pkl
    results/09g_length_baseline_predictions.csv
    logs/09g_length_baseline_report.txt

MLflow: logged to experiment 'zorc_pbody_prediction' if mlflow is installed.

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/09g_length_baseline.py --config config/zorc_config.yaml
"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

ROOT = Path(__file__).parent.parent

LENGTH_FEATURES = ["mrna_length", "cds_length"]

META_COLS = {
    "geneID", "gene_id", "transcript_id", "class", "condition", "qc_fail",
    "isoform_source", "event_type", "bioemu_tier", "bioemu_status",
    "feature_source", "cluster_id", "split", "aiupred_status",
}


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9g — Length baseline")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, X, y, label: str) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auroc = roc_auc_score(y, prob)
    auprc = average_precision_score(y, prob)
    f1_mac = f1_score(y, pred, average="macro", zero_division=0)
    f1_pos = f1_score(y, pred, pos_label=1, zero_division=0)
    f1_neg = f1_score(y, pred, pos_label=0, zero_division=0)
    cm = confusion_matrix(y, pred)
    return {
        "label":   label,
        "n":       len(y),
        "auroc":   auroc,
        "auprc":   auprc,
        "f1_mac":  f1_mac,
        "f1_pos":  f1_pos,
        "f1_neg":  f1_neg,
        "cm":      cm,
        "prob":    prob,
        "pred":    pred,
    }


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    fm_path = ROOT / cfg["paths"]["feature_matrix"]
    df = pd.read_csv(fm_path)

    # Validate required columns
    for col in LENGTH_FEATURES + ["class", "split"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not in feature matrix.")

    # Splits
    id_col = "geneID" if "geneID" in df.columns else "gene_id"
    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    X_train = train[LENGTH_FEATURES].values.astype(np.float32)
    y_train = train["class"].values
    X_val   = val[LENGTH_FEATURES].values.astype(np.float32)
    y_val   = val["class"].values
    X_test  = test[LENGTH_FEATURES].values.astype(np.float32)
    y_test  = test["class"].values

    # No NaNs expected for mrna_length / cds_length; assert to be safe
    assert np.isnan(X_train).sum() == 0, "NaN found in training length features"

    # ── Train ─────────────────────────────────────────────────────────────────
    rf_params = dict(
        n_estimators=500,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    res_val  = evaluate(model, X_val,  y_val,  "val")
    res_test = evaluate(model, X_test, y_test, "test")

    # ── Save model ────────────────────────────────────────────────────────────
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    model_path = results_dir / "09g_length_baseline_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ── Save predictions ──────────────────────────────────────────────────────
    records = []
    for split_df, res in [(val, res_val), (test, res_test)]:
        sub = split_df[[id_col, "class", "split"]].copy()
        sub["prob_pos"] = res["prob"]
        sub["pred"]     = res["pred"]
        sub.rename(columns={id_col: "gene_id"}, inplace=True)
        records.append(sub)
    preds_df = pd.concat(records, ignore_index=True)
    preds_path = results_dir / "09g_length_baseline_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # ── Log to MLflow (if available) ──────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("zorc_pbody_prediction")
        with mlflow.start_run(run_name="length_baseline_p9g"):
            mlflow.set_tag("phase", "P9g")
            mlflow.set_tag("model", "length_only_baseline")
            mlflow.set_tag("features", "mrna_length,cds_length")
            mlflow.log_params({
                "n_estimators":    rf_params["n_estimators"],
                "class_weight":    str(rf_params["class_weight"]),
                "max_features":    rf_params["max_features"],
                "min_samples_leaf": rf_params["min_samples_leaf"],
                "n_features":      2,
                "features":        "mrna_length,cds_length",
            })
            mlflow.log_metrics({
                "oob_score":       model.oob_score_,
                "val_auroc":       res_val["auroc"],
                "val_auprc":       res_val["auprc"],
                "val_f1_macro":    res_val["f1_mac"],
                "test_auroc":      res_test["auroc"],
                "test_auprc":      res_test["auprc"],
                "test_f1_macro":   res_test["f1_mac"],
            })
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(preds_path))

    # ── Report ────────────────────────────────────────────────────────────────
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    report_path = logs_dir / "09g_length_baseline_report.txt"

    lines = [
        "=" * 70,
        "ZORC P9g — Length Baseline Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "Features used: mrna_length, cds_length (top-2 SHAP from P9d)",
        f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}",
        "",
        "── OOB score ──────────────────────────────────────────────────────",
        f"  {model.oob_score_:.4f}",
        "",
    ]
    for res in [res_val, res_test]:
        lines += [
            f"── {res['label'].upper()} ({res['n']} genes) ────────────────────────────────────",
            f"  AUROC : {res['auroc']:.4f}",
            f"  AUPRC : {res['auprc']:.4f}",
            f"  F1-mac: {res['f1_mac']:.4f}  (pos={res['f1_pos']:.4f}  neg={res['f1_neg']:.4f})",
            f"  Confusion matrix (rows=True, cols=Pred):",
            f"    {res['cm']}",
            "",
        ]
    lines += [
        "── Comparison vs full model P9d ─────────────────────────────────",
        "  P9g (2 features)  val AUROC: {:.4f}  test AUROC: {:.4f}".format(
            res_val["auroc"], res_test["auroc"]
        ),
        "  P9d (63 features) val AUROC: 0.7969  test AUROC: 0.7963",
        "  Delta test AUROC: {:.4f}  ({:+.2%} relative gain)".format(
            0.7963 - res_test["auroc"],
            (0.7963 - res_test["auroc"]) / res_test["auroc"],
        ),
        "",
        f"MLflow logged: {'yes' if MLFLOW_AVAILABLE else 'no (install mlflow to enable)'}",
        "=" * 70,
    ]
    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
