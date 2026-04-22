#!/usr/bin/env python3
"""
ZORC — Phase 12a: MLflow Retroactive Experiment Logging
========================================================
Logs all six ZORC model runs retroactively to a local MLflow tracking server.
After running this script, launch the UI with:

    mlflow ui --port 5000

and open http://localhost:5000 to browse the experiment history.

Six runs logged (in development order):
  1. rf_baseline_p9            — RF baseline, pre-AF2, 56 features
  2. xgb_benchmark_p9b_pre_af2 — XGB benchmark before AF2 recovery
  3. xgb_benchmark_p9b_post_af2— XGB after AF2 static feature recovery
  4. rf_feature_engineering_p9d— RF with 4 engineered interaction features
  5. rf_threshold_tuning_p9e   — RF P9d model, threshold scan 0.30–0.70
  6. rf_final_calibrated_p9f   — RF final, Platt sigmoid calibration, 61 feats

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/10b_mlflow_retroactive.py --config config/zorc_config.yaml
    mlflow ui --port 5000
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P12a — MLflow retroactive logging")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Run specifications (from logs + architecture doc)
# ─────────────────────────────────────────────────────────────────────────────

RUNS = [
    # ── 1. RF Baseline (P9) ────────────────────────────────────────────────
    {
        "run_name": "rf_baseline_p9",
        "tags": {
            "phase": "P9",
            "model_type": "RandomForest",
            "dataset": "pre_af2_recovery",
        },
        "params": {
            "n_estimators": 500,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "oob_score": True,
            "random_state": 42,
            "imputation": "global_median",
            "n_features": 56,
        },
        "metrics": {
            "oob_score":     0.7397,
            "val_auroc":     0.7990,
            "val_auprc":     0.8312,
            "val_f1_macro":  0.7247,
            "test_auroc":    0.7974,
            "test_auprc":    0.8469,
            "test_f1_macro": 0.7382,
            "hc_accuracy":   0.96,
        },
        "artifact_files": ["results/09_zorc_rf_model.pkl",
                           "results/09_zorc_shap_importance.csv"],
        "timestamp": "2026-04-07",
    },
    # ── 2. XGBoost benchmark pre-AF2 (P9b — first XGB run) ────────────────
    {
        "run_name": "xgb_benchmark_p9b_pre_af2",
        "tags": {
            "phase": "P9b",
            "model_type": "XGBoost",
            "dataset": "pre_af2_recovery",
        },
        "params": {
            "objective":          "binary:logistic",
            "max_depth":          6,
            "learning_rate":      0.05,
            "n_estimators":       500,
            "early_stopping":     30,
            "subsample":          0.8,
            "colsample_bytree":   0.8,
            "nan_handling":       "native",
            "n_features":         56,
        },
        "metrics": {
            "val_auroc":     0.7993,
            "test_auroc":    0.7835,
            "hc_accuracy":   0.96,
        },
        "artifact_files": [],
        "timestamp": "2026-04-10",
    },
    # ── 3. XGBoost benchmark post-AF2 (P9b — re-run after AF2 recovery) ───
    {
        "run_name": "xgb_benchmark_p9b_post_af2",
        "tags": {
            "phase": "P9b",
            "model_type": "XGBoost",
            "dataset": "post_af2_recovery",
        },
        "params": {
            "objective":          "binary:logistic",
            "max_depth":          6,
            "learning_rate":      0.05,
            "n_estimators":       500,
            "early_stopping":     30,
            "min_child_weight":   5,
            "subsample":          0.8,
            "colsample_bytree":   0.8,
            "nan_handling":       "native",
            "n_features":         60,
            "best_round":         75,
        },
        "metrics": {
            "val_auroc":     0.8001,
            "val_auprc":     0.8229,
            "val_f1_macro":  0.7271,
            "test_auroc":    0.7879,
            "test_auprc":    0.8317,
            "test_f1_macro": 0.6976,
            "hc_accuracy":   0.96,
        },
        "artifact_files": ["results/09b_zorc_xgb_model.json",
                           "results/09b_zorc_xgb_shap_importance.csv"],
        "timestamp": "2026-04-12",
    },
    # ── 4. RF Feature Engineering (P9d) ────────────────────────────────────
    {
        "run_name": "rf_feature_engineering_p9d",
        "tags": {
            "phase": "P9d",
            "model_type": "RandomForest",
            "dataset": "post_af2_recovery",
        },
        "params": {
            "n_estimators":     500,
            "class_weight":     "balanced",
            "max_features":     "sqrt",
            "min_samples_leaf": 2,
            "oob_score":        True,
            "random_state":     42,
            "imputation":       "global_median",
            "n_features":       63,
            "engineered_features": str([
                "rmsf_nterm_cterm_ratio",
                "packing_x_idr",
                "rrach_per_cds_kb",
                "utr3_au_x_length",
            ]),
        },
        "metrics": {
            "val_auroc":     0.7969,
            "val_auprc":     0.8269,
            "val_f1_macro":  0.7094,
            "test_auroc":    0.7963,
            "test_auprc":    0.8431,
            "test_f1_macro": 0.7229,
            "hc_accuracy":   0.96,
        },
        "artifact_files": ["results/09d_rf_eng_model.pkl",
                           "results/09d_shap_rf_eng.csv"],
        "timestamp": "2026-04-14",
    },
    # ── 5. Threshold Tuning (P9e) ───────────────────────────────────────────
    {
        "run_name": "rf_threshold_tuning_p9e",
        "tags": {
            "phase": "P9e",
            "model_type": "RandomForest",
            "dataset": "post_af2_recovery",
        },
        "params": {
            "base_model":          "09d_rf_eng_model.pkl",
            "threshold_scan_min":  0.30,
            "threshold_scan_max":  0.70,
            "threshold_step":      0.01,
            "optimal_threshold":   0.50,
            "n_features":          63,
        },
        "metrics": {
            "val_auroc":              0.7969,
            "val_auprc":              0.8269,
            "test_auroc":             0.7963,
            "test_auprc":             0.8431,
            "test_f1_macro_at_0.50":  0.7229,
            "test_f1_macro_at_0.56":  0.6984,
        },
        "artifact_files": ["results/09e_threshold_curve.csv"],
        "timestamp": "2026-04-15",
    },
    # ── 6. RF Final Calibrated (P9f) ────────────────────────────────────────
    {
        "run_name": "rf_final_calibrated_p9f",
        "tags": {
            "phase": "P9f",
            "model_type": "RandomForest+PlattCalibration",
            "dataset": "post_af2_recovery",
            "model": "final_calibrated",
        },
        "params": {
            "n_estimators":           500,
            "class_weight":           "balanced",
            "max_features":           "sqrt",
            "min_samples_leaf":       2,
            "calibration":            "platt_sigmoid",
            "calibration_cv":         5,
            "n_features":             61,
            "dropped_features":       str(["packing_x_idr", "rrach_per_cds_kb"]),
            "optimal_threshold":      0.50,
        },
        "metrics": {
            "val_auroc":              0.7979,
            "test_auroc_uncal":       0.7840,
            "test_auroc_cal":         0.7862,
            "test_auprc_cal":         0.8333,
            "test_f1_macro_cal":      0.7111,
            "brier_score_uncal":      0.1801,
            "brier_score_cal":        0.1776,
            "hc_accuracy_uncal":      0.96,
            "hc_accuracy_cal":        0.92,
        },
        "artifact_files": ["results/09f_rf_final_model.pkl",
                           "results/09f_shap_final.csv",
                           "results/09f_predictions_final.csv"],
        "timestamp": "2026-04-15",
    },
]


def log_run(client, experiment_id, run_spec, base_dir: Path):
    """Log a single retroactive run to MLflow."""
    ts_dt = datetime.strptime(run_spec["timestamp"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    start_time_ms = int(ts_dt.timestamp() * 1000)

    with mlflow.start_run(
        run_name=run_spec["run_name"],
        experiment_id=experiment_id,
        tags={**run_spec["tags"], "retroactive_logging": "true"},
    ) as run:
        mlflow.log_params(run_spec["params"])
        mlflow.log_metrics(run_spec["metrics"])

        for rel_path in run_spec["artifact_files"]:
            full_path = base_dir / rel_path
            if full_path.exists():
                mlflow.log_artifact(str(full_path))
            else:
                print(f"    [WARN] artifact not found: {rel_path}")

    return run.info.run_id


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    base_dir = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()

    mlruns_dir = base_dir / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    print("=" * 65)
    print("ZORC — Phase 12a: MLflow Retroactive Logging")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print("=" * 65)

    client = mlflow.tracking.MlflowClient()

    experiment_name = "zorc_pbody_prediction"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "ZORC",
                "organism": "Arabidopsis_thaliana",
                "task": "P-body_mRNA_enrichment",
                "lab": "MoschouLab",
                "grant": "ERC_PLANTEX",
            },
        )
        print(f"\nCreated experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = exp.experiment_id
        print(f"\nUsing existing experiment '{experiment_name}' (id={experiment_id})")

    print()
    for i, run_spec in enumerate(RUNS, 1):
        print(f"  [{i}/{len(RUNS)}] Logging: {run_spec['run_name']}")
        run_id = log_run(client, experiment_id, run_spec, base_dir)
        # Key metrics for console summary
        m = run_spec["metrics"]
        test_auroc = m.get("test_auroc") or m.get("test_auroc_cal", "n/a")
        test_f1    = m.get("test_f1_macro") or m.get("test_f1_macro_cal", "n/a")
        print(f"         run_id={run_id[:8]}...  "
              f"test_auroc={test_auroc}  f1={test_f1}")

    print(f"\n{'=' * 65}")
    print(f"  {len(RUNS)} runs logged to experiment '{experiment_name}'")
    print(f"\n  Launch UI:  mlflow ui --port 5000")
    print(f"  Open:       http://localhost:5000")
    print("✓ Phase 12a complete.")


if __name__ == "__main__":
    main()
