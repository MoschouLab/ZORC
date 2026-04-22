#!/usr/bin/env python3
"""
ZORC — Phase 11d: Tableau Public CSV Exports
=============================================
Generates three publication-quality CSVs for Tableau Public dashboards.

Dashboard 1 — ZORC Feature Importance (01_feature_importance.csv)
  SHAP importance for the final model (P9f), annotated by feature category.
  Drives: bar chart + top-5 scatter plots coloured by class.

Dashboard 2 — P-body Probability Landscape (02_probability_landscape.csv)
  Per-gene probability with key features (utr3_au_content, rrach_count,
  rrach_per_kb, mrna_length, idr_percent) and labels.
  Drives: 2D scatter (utr3_au_content vs rrach_count), probability heatmap.

Dashboard 3 — Pipeline Optimisation History (03_pipeline_history.csv)
  AUROC / AUPRC / F1-macro across all six model iterations.
  Drives: line chart showing impact of each pipeline intervention.

Outputs:
    tableau/01_feature_importance.csv
    tableau/02_probability_landscape.csv
    tableau/03_pipeline_history.csv

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/11d_tableau_export.py --config config/zorc_config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P11d — Tableau CSV Exports")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Feature category annotations for Dashboard 1
FEATURE_CATEGORIES = {
    # RNA — composition
    "fA": "RNA-composition", "fU": "RNA-composition",
    "fG": "RNA-composition", "fC": "RNA-composition",
    "au_content": "RNA-composition",
    # RNA — dinucleotides
    "di_AA": "RNA-dinucleotide", "di_AU": "RNA-dinucleotide",
    "di_AG": "RNA-dinucleotide", "di_AC": "RNA-dinucleotide",
    "di_UA": "RNA-dinucleotide", "di_UU": "RNA-dinucleotide",
    "di_UG": "RNA-dinucleotide", "di_UC": "RNA-dinucleotide",
    "di_GA": "RNA-dinucleotide", "di_GU": "RNA-dinucleotide",
    "di_GG": "RNA-dinucleotide", "di_GC": "RNA-dinucleotide",
    "di_CA": "RNA-dinucleotide", "di_CU": "RNA-dinucleotide",
    "di_CG": "RNA-dinucleotide", "di_CC": "RNA-dinucleotide",
    # RNA — length / structure
    "mrna_length": "RNA-length", "cds_length": "RNA-length",
    "utr3_length": "RNA-length", "utr5_length": "RNA-length",
    "utr3_fraction": "RNA-length", "utr5_fraction": "RNA-length",
    "cds_fraction": "RNA-length",
    "utr3_au_content": "RNA-structure",
    "mfe": "RNA-structure", "mfe_per_nt": "RNA-structure",
    "n_stemloops": "RNA-structure", "n_stemloops_per_kb": "RNA-structure",
    # RNA — m6A
    "rrach_count": "RNA-m6A", "rrach_per_kb": "RNA-m6A",
    "rrach_5utr_count": "RNA-m6A", "rrach_3utr_count": "RNA-m6A",
    "rrach_cds_count": "RNA-m6A",
    # Protein — static
    "n_residues": "Protein-static",
    "idr_percent": "Protein-IDR",
    "has_bioemu": "Protein-flag",
    # Protein — dynamic (BioEmu)
    "rmsf_mean": "Protein-dynamic", "rmsf_std": "Protein-dynamic",
    "rmsf_max": "Protein-dynamic",
    "rmsf_nterm50": "Protein-dynamic", "rmsf_cterm50": "Protein-dynamic",
    "rg_mean": "Protein-dynamic", "rg_std": "Protein-dynamic",
    "rg_cv": "Protein-dynamic",
    "contact_density": "Protein-dynamic",
    "packing_density": "Protein-dynamic",
    "sasa_per_residue": "Protein-dynamic",
    "pass_rate": "Protein-dynamic",
    # Engineered
    "rmsf_nterm_cterm_ratio": "Engineered",
    "packing_x_idr": "Engineered",
    "rrach_per_cds_kb": "Engineered",
    "utr3_au_x_length": "Engineered",
}


def build_feature_importance(shap_path: Path) -> pd.DataFrame:
    df = pd.read_csv(shap_path)
    df["category"] = df["feature"].map(FEATURE_CATEGORIES).fillna("Other")
    # Normalise to % of total SHAP
    total = df["mean_abs_shap"].sum()
    df["shap_pct"] = (df["mean_abs_shap"] / total * 100).round(3)
    return df[["rank", "feature", "mean_abs_shap", "shap_pct", "category"]]


def build_probability_landscape(
    predictions_path: Path,
    feature_matrix_path: Path,
) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    fm   = pd.read_csv(feature_matrix_path)

    # Normalise gene_id column name
    if "geneID" in fm.columns:
        fm = fm.rename(columns={"geneID": "gene_id"})
    if "geneID" in pred.columns:
        pred = pred.rename(columns={"geneID": "gene_id"})

    keep_features = [
        "gene_id", "utr3_au_content", "rrach_count", "rrach_per_kb",
        "mrna_length", "cds_length", "utr3_length", "idr_percent",
        "rmsf_mean", "rmsf_nterm50", "rmsf_nterm_cterm_ratio",
        "di_CG", "di_UA", "mfe_per_nt",
    ]
    fm_sub = fm[[c for c in keep_features if c in fm.columns]]
    # Feature matrix has 9 genes with duplicate gene_ids (multiple isoforms);
    # keep first occurrence so the merge stays 1:1 with the predictions table.
    fm_sub = fm_sub.drop_duplicates(subset="gene_id", keep="first")

    merged = pred.merge(fm_sub, on="gene_id", how="left")

    # Confidence tier
    def confidence(p):
        if p >= 0.75:
            return "High (≥0.75)"
        elif p >= 0.55:
            return "Medium (0.55–0.75)"
        else:
            return "Low (<0.55)"

    merged["confidence"] = merged["prob_pos"].apply(confidence)
    merged["class_label"] = merged["class"].map({1: "P-body enriched", 0: "Not enriched"})

    return merged.round(5)


def build_pipeline_history() -> pd.DataFrame:
    """
    Hardcoded from log files and architecture document.
    Each row = one model evaluation event.
    """
    rows = [
        # P9 — RF baseline (pre-AF2 recovery)
        dict(phase="P9", run_label="RF Baseline", split="val",
             auroc=0.7990, auprc=0.8312, f1_macro=0.7247,
             n_features=56, notes="Global median imputation, pre-AF2"),
        dict(phase="P9", run_label="RF Baseline", split="test",
             auroc=0.7974, auprc=0.8469, f1_macro=0.7382,
             n_features=56, notes="Global median imputation, pre-AF2"),
        # P9g — length-only baseline (reference ceiling)
        dict(phase="P9g", run_label="Length-only Baseline", split="val",
             auroc=0.6864, auprc=0.7221, f1_macro=0.6283,
             n_features=2, notes="mrna_length + cds_length only"),
        dict(phase="P9g", run_label="Length-only Baseline", split="test",
             auroc=0.6434, auprc=0.6900, f1_macro=0.5870,
             n_features=2, notes="mrna_length + cds_length only"),
        # P9b — XGBoost post-AF2
        dict(phase="P9b", run_label="XGBoost (post-AF2)", split="val",
             auroc=0.8001, auprc=0.8229, f1_macro=0.7271,
             n_features=60, notes="NaN-native, AF2 static features"),
        dict(phase="P9b", run_label="XGBoost (post-AF2)", split="test",
             auroc=0.7879, auprc=0.8317, f1_macro=0.6976,
             n_features=60, notes="NaN-native, AF2 static features"),
        # P9d — RF feature engineering
        dict(phase="P9d", run_label="RF + Feature Eng.", split="val",
             auroc=0.7969, auprc=0.8269, f1_macro=0.7094,
             n_features=63, notes="4 interaction features added"),
        dict(phase="P9d", run_label="RF + Feature Eng.", split="test",
             auroc=0.7963, auprc=0.8431, f1_macro=0.7229,
             n_features=63, notes="4 interaction features added"),
        # P9d — XGB feature engineering
        dict(phase="P9d", run_label="XGB + Feature Eng.", split="val",
             auroc=0.8036, auprc=0.8302, f1_macro=0.7294,
             n_features=63, notes="4 interaction features added"),
        dict(phase="P9d", run_label="XGB + Feature Eng.", split="test",
             auroc=0.7811, auprc=0.8181, f1_macro=0.7027,
             n_features=63, notes="4 interaction features added"),
        # P9f — RF final Platt-calibrated
        dict(phase="P9f", run_label="RF Final (Platt cal.)", split="val",
             auroc=0.7979, auprc=None, f1_macro=None,
             n_features=61, notes="Platt calibration cv=5, Brier=0.1776"),
        dict(phase="P9f", run_label="RF Final (Platt cal.)", split="test",
             auroc=0.7862, auprc=0.8333, f1_macro=0.7111,
             n_features=61, notes="Platt calibration cv=5, Brier=0.1776"),
    ]
    df = pd.DataFrame(rows)
    # Add phase order for Tableau sorting
    phase_order = {"P9": 1, "P9g": 0, "P9b": 2, "P9d": 3, "P9f": 4}
    df["phase_order"] = df["phase"].map(phase_order)
    return df


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    base_dir    = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    results_dir = base_dir / "results"
    tableau_dir = base_dir / "tableau"
    tableau_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ZORC — Phase 11d: Tableau Public CSV Exports")
    print("=" * 65)

    # ── Dashboard 1: Feature Importance ─────────────────────────────────────
    shap_path = results_dir / "09f_shap_final.csv"
    print(f"\n[1/3] Feature importance  ← {shap_path.name}")
    fi = build_feature_importance(shap_path)
    out1 = tableau_dir / "01_feature_importance.csv"
    fi.to_csv(out1, index=False)
    print(f"  {len(fi)} features → {out1.name}")

    # ── Dashboard 2: Probability Landscape ──────────────────────────────────
    pred_path = results_dir / "09f_predictions_final.csv"
    fm_path   = base_dir / cfg.get("outputs", {}).get(
                    "feature_matrix_eng",
                    "data/processed/08_zorc_feature_matrix_eng.csv")
    if not fm_path.exists():
        fm_path = base_dir / cfg.get("outputs", {}).get(
                      "feature_matrix",
                      "data/processed/08_zorc_feature_matrix.csv")
    print(f"\n[2/3] Probability landscape ← {pred_path.name} + {fm_path.name}")
    pl = build_probability_landscape(pred_path, fm_path)
    out2 = tableau_dir / "02_probability_landscape.csv"
    pl.to_csv(out2, index=False)
    print(f"  {len(pl)} genes → {out2.name}")

    # ── Dashboard 3: Pipeline History ───────────────────────────────────────
    print(f"\n[3/3] Pipeline optimisation history (hardcoded from logs)")
    ph = build_pipeline_history()
    out3 = tableau_dir / "03_pipeline_history.csv"
    ph.to_csv(out3, index=False)
    print(f"  {len(ph)} rows → {out3.name}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("OUTPUT FILES")
    print(f"  {out1.relative_to(base_dir)}")
    print(f"  {out2.relative_to(base_dir)}")
    print(f"  {out3.relative_to(base_dir)}")
    print(f"\nNext step: upload to Tableau Public via Tableau Desktop.")
    print("✓ Phase 11d complete.")


if __name__ == "__main__":
    main()
