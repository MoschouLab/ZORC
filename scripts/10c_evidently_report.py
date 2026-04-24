"""
ZORC — P12f: EvidentlyAI Data Drift & Quality Reports
======================================================
Generates HTML monitoring reports comparing the training distribution
(reference) against the test split (current) for all numeric model features.

Three report presets:
  - DataDriftPreset   : per-feature distribution shift (Jensen-Shannon / χ²)
  - DataQualityPreset : missing values, outliers, duplicates
  - ClassificationPreset : precision/recall/ROC on test split (requires labels)

Outputs
-------
  monitoring/evidently_reports/drift_report.html
  monitoring/evidently_reports/quality_report.html
  monitoring/evidently_reports/classification_report.html

Usage
-----
    conda activate zorc_pipeline
    pip install evidently
    python scripts/10c_evidently_report.py --config config/zorc_config.yaml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate EvidentlyAI monitoring reports")
    p.add_argument("--config", default="config/zorc_config.yaml")
    p.add_argument(
        "--n-current", type=int, default=None,
        help="Number of rows to sample from test split as 'current' batch. "
             "Default: use entire test split."
    )
    return p.parse_args()


# ── Feature columns used by the P9d model ────────────────────────────────────

_META_COLS = {
    "geneID", "transcript_id", "condition", "qc_fail", "bioemu_status",
    "feature_source", "isoform_source", "event_type", "cluster_id", "split",
}

_LABEL_COL = "class"


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude meta and label)."""
    drop = _META_COLS | {_LABEL_COL}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(args.config).parent.parent
    matrix_path = repo_root / cfg.get(
        "feature_matrix_eng",
        "data/processed/08_zorc_feature_matrix_eng.csv",
    )
    split_path = repo_root / cfg.get(
        "split_assignments",
        "data/processed/08_zorc_split_assignments.csv",
    )
    model_path = repo_root / cfg.get(
        "final_model_path",
        "results/09d_rf_eng_model.pkl",
    )
    out_dir = repo_root / "monitoring" / "evidently_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[P12f] Loading feature matrix …")
    df = pd.read_csv(matrix_path, low_memory=False)

    splits = pd.read_csv(split_path)[["geneID", "split"]]
    df = df.drop(columns=["split"], errors="ignore").merge(splits, on="geneID", how="left")

    feat_cols = _get_feature_cols(df)
    print(f"[P12f] Feature columns for reports: {len(feat_cols)}")

    ref_df = df[df["split"] == "train"][feat_cols + [_LABEL_COL]].reset_index(drop=True)
    cur_df = df[df["split"] == "test"][feat_cols + [_LABEL_COL]].reset_index(drop=True)

    if args.n_current is not None:
        cur_df = cur_df.sample(min(args.n_current, len(cur_df)), random_state=42)

    print(f"[P12f] Reference (train): {len(ref_df)} rows | Current (test): {len(cur_df)} rows")

    # ── Load model + feature order for classification report ─────────────────
    model = None
    model_feat_cols = None
    medians_path = repo_root / "api" / "imputation_medians.json"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if medians_path.exists():
            with open(medians_path) as f:
                med_data = json.load(f)
            model_feat_cols = med_data["feature_order"]
        print(f"[P12f] Model loaded: {model_path.name}  |  {len(model_feat_cols or [])} features")
    else:
        print(f"[P12f] WARNING: model not found at {model_path} — skipping ClassificationPreset")

    # ── Import Evidently (0.7+ API) ───────────────────────────────────────────
    # evidently 0.7 changed API:
    #   - presets live in evidently.presets
    #   - report.run() returns a Snapshot with save_html
    #   - ClassificationPreset requires DataDefinition with classification list
    try:
        from evidently import Dataset, Report
        from evidently.core.datasets import (
            BinaryClassification as EvBinaryClassification,
            DataDefinition,
        )
        from evidently.presets import (
            ClassificationPreset,
            DataDriftPreset,
            DataSummaryPreset,
        )
    except ImportError:
        print("[P12f] ERROR: evidently not installed. Run: pip install evidently")
        sys.exit(1)

    # ── Report 1: Data Drift ──────────────────────────────────────────────────
    print("[P12f] Generating DataDriftPreset report …")
    drift_snapshot = Report(metrics=[DataDriftPreset()]).run(
        reference_data=ref_df[feat_cols],
        current_data=cur_df[feat_cols],
    )
    drift_path = out_dir / "drift_report.html"
    drift_snapshot.save_html(str(drift_path))
    print(f"[P12f] Saved: {drift_path}")

    # ── Report 2: Data Quality / Summary ─────────────────────────────────────
    print("[P12f] Generating DataSummaryPreset report …")
    quality_snapshot = Report(metrics=[DataSummaryPreset()]).run(
        reference_data=ref_df[feat_cols],
        current_data=cur_df[feat_cols],
    )
    quality_path = out_dir / "quality_report.html"
    quality_snapshot.save_html(str(quality_path))
    print(f"[P12f] Saved: {quality_path}")

    # ── Report 3: Classification (requires model predictions) ─────────────────
    if model is not None and model_feat_cols is not None:
        print("[P12f] Generating ClassificationPreset report …")

        # Use the exact feature order the model was trained on (63 features)
        clf_cols = [c for c in model_feat_cols if c in ref_df.columns]
        ref_x = ref_df[clf_cols].copy()
        cur_x = cur_df[clf_cols].copy()

        # Impute NaN with column median (same strategy as training)
        for col in clf_cols:
            med = ref_x[col].median()
            ref_x[col] = ref_x[col].fillna(med)
            cur_x[col] = cur_x[col].fillna(med)

        ref_preds = pd.DataFrame({
            "target":     ref_df[_LABEL_COL].values,
            "prediction": model.predict(ref_x.values),
            "prediction_proba": model.predict_proba(ref_x.values)[:, 1],
        })
        cur_preds = pd.DataFrame({
            "target":     cur_df[_LABEL_COL].values,
            "prediction": model.predict(cur_x.values),
            "prediction_proba": model.predict_proba(cur_x.values)[:, 1],
        })

        # evidently 0.7: ClassificationPreset requires DataDefinition
        clf_dd = DataDefinition(
            classification=[EvBinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="prediction_proba",
            )]
        )
        ref_ds = Dataset.from_pandas(ref_preds, data_definition=clf_dd)
        cur_ds = Dataset.from_pandas(cur_preds, data_definition=clf_dd)

        clf_snapshot = Report(metrics=[ClassificationPreset()]).run(
            reference_data=ref_ds,
            current_data=cur_ds,
        )
        clf_path = out_dir / "classification_report.html"
        clf_snapshot.save_html(str(clf_path))
        print(f"[P12f] Saved: {clf_path}")

    print("\n[P12f] All reports generated successfully.")
    print(f"  Open in browser: {out_dir}/drift_report.html")


if __name__ == "__main__":
    main()
