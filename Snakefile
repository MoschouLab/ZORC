# =============================================================================
# Snakefile — ZORC: Zip-code Of RNAs that Condense
# Phase 10: Reproducible Snakemake workflow (P1 → P9f)
#
# Usage:
#   # Dry-run (show plan without executing):
#   snakemake --cores 4 --use-conda -n
#
#   # Full pipeline (GPU phases P6/P6b require NVIDIA GPU ≥25 GB VRAM):
#   snakemake --cores 4 --use-conda
#
#   # ML-only (assumes BioEmu outputs already exist):
#   snakemake --cores 4 --use-conda results/09f_rf_final_model.pkl
#
#   # Single rule:
#   snakemake --cores 1 --use-conda logs/04_rna_features_report.txt
#
# Environments:
#   envs/zorc_pipeline.yml  — main pipeline (P1–P4, P7–P9f)
#   envs/bioemu_ref.yml     — BioEmu + AIUPred (P5–P6b); copy from
#                             ~/Documents/TRIP-isoform-lncoding-pipeline/
#                             before running P5/P6 for the first time
#
# External dependencies (must exist before pipeline starts):
#   config.reference.gtf           — TAIR10.59 GTF annotation
#   config.reference.genome_fasta  — TAIR10 genome FASTA
#   config.reference.rmats_*_dir   — rMATS output directories (TRIP pipeline)
#   data/raw/                      — APEAL + T-RIP Excel files (not redistributed)
#
# Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
# Project: github.com/MoschouLab/ZORC
# =============================================================================

import os

configfile: "config/zorc_config.yaml"

# ---------------------------------------------------------------------------
# Expand ~ in reference paths (outside the ZORC repo)
# ---------------------------------------------------------------------------
GTF          = os.path.expanduser(config["reference"]["gtf"])
GENOME_FASTA = os.path.expanduser(config["reference"]["genome_fasta"])
RMATS_NS_DIR = os.path.expanduser(config["reference"]["rmats_ns_dir"])
RMATS_HS_DIR = os.path.expanduser(config["reference"]["rmats_hs_dir"])

ENV_MAIN   = "envs/zorc_pipeline.yml"
ENV_BIOEMU = "envs/bioemu_ref.yml"


# =============================================================================
# RULE ALL — primary targets
# =============================================================================
rule all:
    input:
        # ── Final deliverable ────────────────────────────────────────────────
        "results/09f_rf_final_model.pkl",
        "logs/09f_final_model_report.txt",
        # ── Benchmark models ─────────────────────────────────────────────────
        "logs/09_rf_report.txt",           # RF baseline (pre-AF2 imputation)
        "logs/09b_xgb_report.txt",         # XGBoost benchmark
        # ── Feature engineering & threshold tuning (inform P9f) ──────────────
        "logs/09d_feature_engineering_report.txt",
        "logs/09e_threshold_report.txt",
        # ── AF2 recovery (patches P7/P8 outputs for P9d–P9f) ────────────────
        "logs/09c_af2_recovery_report.txt",


# =============================================================================
# P1 — Build coregulon gene list
# Intersects T-RIP RNA lists (Liu 2024 Plant Cell) with APEAL proteomics
# (Liu 2023 EMBO J). Positive class = enriched in P-bodies AND protein
# log2FC ≥ protein_enrichment_threshold in any APEAL condition.
# =============================================================================
rule build_coregulon:
    input:
        apeal = config["input_files"]["apeal_excel"],
        trip  = config["input_files"]["trip_excel"],
    output:
        csv    = "data/processed/01_zorc_coregulon_list.csv",
        report = "logs/01_coregulon_build_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/01_build_coregulon.py --config config/zorc_config.yaml"


# =============================================================================
# P2 — Map isoforms
# For genes with rMATS splicing events: transcript containing the AS region.
# For all others: Ensembl_canonical transcript (Strategy B fallback).
# See docs/ZORC_methodological_decisions_v2.md §2 for rationale.
# =============================================================================
rule map_isoforms:
    input:
        coregulon = rules.build_coregulon.output.csv,
        gtf       = ancient(GTF),
        rmats_ns  = ancient(RMATS_NS_DIR),
        rmats_hs  = ancient(RMATS_HS_DIR),
    output:
        csv    = "data/processed/02_zorc_isoform_map.csv",
        report = "logs/02_isoform_map_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/02_map_isoforms.py --config config/zorc_config.yaml"


# =============================================================================
# P3 — Fetch sequences
# Extracts spliced mRNA (-w) and translated CDS (-y) via gffread v0.12.7.
# Produces per-class FASTAs + combined all_mrna.fa / all_protein.fa.
# QC flags: no_CDS, short_protein <30aa, internal_stop.
# =============================================================================
rule fetch_sequences:
    input:
        isoform_map  = rules.map_isoforms.output.csv,
        gtf          = ancient(GTF),
        genome_fasta = ancient(GENOME_FASTA),
    output:
        manifest    = "data/processed/03_sequence_manifest.csv",
        all_mrna    = "data/processed/sequences/all_mrna.fa",
        all_protein = "data/processed/sequences/all_protein.fa",
        report      = "logs/03_sequence_qc_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/03_fetch_sequences.py --config config/zorc_config.yaml"


# =============================================================================
# P4 — RNA feature engineering (requires ViennaRNA/RNAfold ≥2.6)
# 41 features: nucleotide/dinucleotide composition, UTR fractions, m6A motifs
# (RRACH), RNAfold MFE (sequences ≤3000 nt only; 68 excluded from structure).
# Add --no-rnafold to skip structure features during development.
# =============================================================================
rule rna_features:
    input:
        manifest = rules.fetch_sequences.output.manifest,
        mrna_fa  = rules.fetch_sequences.output.all_mrna,
    output:
        csv    = "data/processed/04_rna_features.csv",
        report = "logs/04_rna_features_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/04_rna_features.py --config config/zorc_config.yaml"


# =============================================================================
# P5 — AIUPred IDR prediction + BioEmu tier assignment
# ENV: bioemu (AIUPred v2.0, --force-cpu avoids CUDA/NCCL conflict with torch)
# Tier thresholds (configurable in zorc_config.yaml under bioemu_tiers):
#   Tier 1 IDR <10%  → AlphaFold2 static structure only (AF2 fallback P9c)
#   Tier 2 IDR 10–30%→ BioEmu 50 conformations (P6)
#   Tier 3 IDR ≥30% → BioEmu 100 conformations (P6)
# =============================================================================
rule aiupred_idr:
    input:
        protein_fa = rules.fetch_sequences.output.all_protein,
        manifest   = rules.fetch_sequences.output.manifest,
    output:
        idr_summary = "data/processed/05_aiupred_idr_summary.csv",
        tier_list   = "data/processed/05_bioemu_tier_assignments.csv",
        report      = "logs/05_aiupred_report.txt",
    conda:
        ENV_BIOEMU
    shell:
        "python scripts/05_aiupred_idr.py --config config/zorc_config.yaml"


# =============================================================================
# P6 — BioEmu conformational sampling (Tier 2 + Tier 3 proteins)
# ENV: bioemu (BioEmu v1.1, torch 2.5.1+cu121)
# HARDWARE: NVIDIA GPU ≥25.4 GB VRAM required (tested on RTX A5000)
# RUNTIME: ~5 days continuous for 759 proteins
# CHECKPOINTING: Resumable — completed proteins stored in checkpoint.json.
#                Re-running this rule skips already-completed proteins.
# KNOWN BUGS (see docs/ZORC_methodological_decisions_v2.md §6.1):
#   unphysical_filter_bug → add --filter_samples=False to the shell command
#   timeout (>1h)         → add --timeout 10800
#   OOM (>3000aa proteins)→ AT2G28290.5, AT2G45540.2 marked excluded in P5
# =============================================================================
rule bioemu_batch:
    input:
        tier_list  = rules.aiupred_idr.output.tier_list,
        protein_fa = rules.fetch_sequences.output.all_protein,
    output:
        checkpoint = "data/processed/bioemu/checkpoint.json",
        report     = "logs/06_bioemu_report.txt",
    resources:
        gpu = 1,
    conda:
        ENV_BIOEMU
    shell:
        "python scripts/06_bioemu_batch.py --config config/zorc_config.yaml"


# =============================================================================
# P6b — BioEmu for Tier 1 proteins (AF2 fallback structures)
# ENV: bioemu
# HARDWARE: GPU required (same as P6). Runs AFTER P6 completes.
# Processes 698 Tier 1 proteins (IDR <10%) with 100 conformations each.
# =============================================================================
rule bioemu_tier1:
    input:
        tier_list  = rules.aiupred_idr.output.tier_list,
        protein_fa = rules.fetch_sequences.output.all_protein,
        p6_done    = rules.bioemu_batch.output.checkpoint,
    output:
        checkpoint = "data/processed/bioemu/checkpoint_tier1.json",
        report     = "logs/06b_bioemu_tier1_report.txt",
    resources:
        gpu = 1,
    conda:
        ENV_BIOEMU
    shell:
        "python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml"


# =============================================================================
# P7 — Protein feature extraction (MDAnalysis 2.10.0)
# 18 features from BioEmu .xtc trajectories:
#   RMSF (mean, std, max, Nterm50, Cterm50), Rg (mean, std, CV),
#   contact_density (Cα-Cα 8Å), packing_density, sasa_per_residue.
# Tier 1 proteins have BioEmu dynamic features set to NaN (→ imputed in P9).
# NOTE: AF2 API fallback silently failed for most Tier 1 proteins in the
#       original implementation. P9c (af2_recovery) patches this file.
# =============================================================================
rule protein_features:
    input:
        tier_list      = rules.aiupred_idr.output.tier_list,
        bioemu_ckpt    = rules.bioemu_batch.output.checkpoint,
        bioemu_ckpt_t1 = rules.bioemu_tier1.output.checkpoint,
        manifest       = rules.fetch_sequences.output.manifest,
    output:
        csv    = "data/processed/07_protein_features.csv",
        report = "logs/07_protein_features_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/07_protein_features.py --config config/zorc_config.yaml"


# =============================================================================
# P8 — Feature matrix assembly + anti-leakage train/val/test split
# Merges P4 (RNA) + P5 (IDR) + P7 (protein) into 1,510 × 59 feature matrix.
# CD-HIT at 40% protein identity → cluster-based split (70/15/15).
# Prevents paralog leakage (Arabidopsis has many high-identity gene families).
# =============================================================================
rule feature_matrix:
    input:
        rna_features     = rules.rna_features.output.csv,
        idr_summary      = rules.aiupred_idr.output.idr_summary,
        protein_features = rules.protein_features.output.csv,
    output:
        matrix = "data/processed/08_zorc_feature_matrix.csv",
        splits = "data/processed/08_zorc_split_assignments.csv",
        report = "logs/08_feature_matrix_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/08_feature_matrix.py --config config/zorc_config.yaml"


# =============================================================================
# P9 — Random Forest baseline (pre-AF2, global median imputation)
# BENCHMARK model — NOT the final deliverable (see P9f).
# 500 trees, class_weight='balanced'. Dynamic BioEmu features for Tier 1
# proteins (49.5% of dataset) imputed with global median + has_bioemu flag.
# Test AUROC: 0.7974  AUPRC: 0.8469  F1-macro: 0.7382
# =============================================================================
rule random_forest:
    input:
        matrix = rules.feature_matrix.output.matrix,
        splits = rules.feature_matrix.output.splits,
    output:
        model       = "results/09_zorc_rf_model.pkl",
        predictions = "results/09_zorc_predictions.csv",
        shap_values = "results/09_zorc_shap_values.csv",
        shap_imp    = "results/09_zorc_shap_importance.csv",
        report      = "logs/09_rf_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09_random_forest.py --config config/zorc_config.yaml"


# =============================================================================
# P9b — XGBoost benchmark
# Handles NaN natively (no imputation needed for dynamic features).
# Post-AF2 recovery: Val AUROC 0.8001, Test AUROC 0.7879 (best_round=75).
# NOTE: After af2_recovery (P9c) patches the feature matrix, re-running this
#       rule will use the post-AF2 matrix automatically (updated timestamp).
# =============================================================================
rule xgboost_benchmark:
    input:
        matrix = rules.feature_matrix.output.matrix,
        splits = rules.feature_matrix.output.splits,
    output:
        model       = "results/09b_zorc_xgb_model.json",
        predictions = "results/09b_zorc_xgb_predictions.csv",
        shap_values = "results/09b_zorc_xgb_shap_values.csv",
        shap_imp    = "results/09b_zorc_xgb_shap_importance.csv",
        report      = "logs/09b_xgb_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09b_xgboost.py --config config/zorc_config.yaml"


# =============================================================================
# P9c — AF2 structure recovery for Tier 1 proteins
# Fetches AlphaFold2 PDB structures via EBI API for 698 Tier 1 proteins
# (IDR <10%) and extracts Rg / contact_density.
# IMPORTANT: This rule PATCHES data/processed/07_protein_features.csv
# in-place and REGENERATES data/processed/08_zorc_feature_matrix.csv.
# It must run BEFORE P9d, P9e, and P9f (they use the patched matrix).
# P9 and P9b can run before or after P9c (they serve as benchmarks).
# =============================================================================
rule af2_recovery:
    input:
        protein_features = rules.protein_features.output.csv,
        feature_matrix   = rules.feature_matrix.output.matrix,
    output:
        report = "logs/09c_af2_recovery_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09c_af2_recovery.py --config config/zorc_config.yaml"


# =============================================================================
# P9d — Feature engineering
# Adds 4 candidate features; trains RF + XGBoost to rank them via SHAP.
# Confirmed valid (Top 20 in both models):
#   rmsf_nterm_cterm_ratio (N/C flexibility asymmetry)
#   utr3_au_x_length       (3'UTR AU-content × length interaction)
# Dropped (SHAP ranking inconsistent between RF and XGB):
#   packing_x_idr          → excluded in P9f
#   rrach_per_cds_kb       → excluded in P9f
# Depends on af2_recovery: uses the post-AF2 patched feature matrix.
# =============================================================================
rule feature_engineering:
    input:
        af2_recovery   = rules.af2_recovery.output.report,
        feature_matrix = rules.feature_matrix.output.matrix,
    output:
        eng_matrix = "data/processed/08_zorc_feature_matrix_eng.csv",
        shap_rf    = "results/09d_shap_rf_eng.csv",
        shap_xgb   = "results/09d_shap_xgb_eng.csv",
        rf_model   = "results/09d_rf_eng_model.pkl",
        xgb_model  = "results/09d_xgb_eng_model.json",
        report     = "logs/09d_feature_engineering_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09d_feature_engineering.py --config config/zorc_config.yaml"


# =============================================================================
# P9e — Threshold tuning
# Scans classification thresholds 0.30–0.70 on the validation set.
# Confirmed: threshold=0.50 is optimal (no gain from shifting).
# Uses P9d RF model (09d_rf_eng_model.pkl) on the post-AF2 feature matrix.
# =============================================================================
rule threshold_tuning:
    input:
        af2_recovery   = rules.af2_recovery.output.report,
        feature_matrix = rules.feature_matrix.output.matrix,
        rf_model       = rules.feature_engineering.output.rf_model,
    output:
        curve  = "results/09e_threshold_curve.csv",
        report = "logs/09e_threshold_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09e_threshold_tuning.py --config config/zorc_config.yaml"


# =============================================================================
# P9f — Final model (Platt-calibrated RF) — PRIMARY DELIVERABLE
# 500 trees, class_weight='balanced', 61 features (59 post-AF2 base +
# 2 validated engineered: rmsf_nterm_cterm_ratio, utr3_au_x_length).
# Platt calibration (logistic CV=5) reduces Brier score: 0.1801 → 0.1776.
# Test AUROC: 0.7963  AUPRC: 0.8431  F1-macro: 0.7229
# HC validation (25 colleague-curated genes): 24/25 (96%)
# Threshold: 0.50 (confirmed optimal by P9e)
# Reads the base feature matrix (post-P9c) and engineers features on-the-fly.
# =============================================================================
rule final_model:
    input:
        af2_recovery   = rules.af2_recovery.output.report,
        feature_matrix = rules.feature_matrix.output.matrix,
        p9d_done       = rules.feature_engineering.output.report,
        p9e_done       = rules.threshold_tuning.output.report,
    output:
        model       = "results/09f_rf_final_model.pkl",
        base_model  = "results/09f_rf_base_model.pkl",
        shap        = "results/09f_shap_final.csv",
        predictions = "results/09f_predictions_final.csv",
        report      = "logs/09f_final_model_report.txt",
    conda:
        ENV_MAIN
    shell:
        "python scripts/09f_final_model.py --config config/zorc_config.yaml"
