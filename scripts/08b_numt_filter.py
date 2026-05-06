#!/usr/bin/env python3
"""
08b_numt_filter.py
==================
ZORC — Phase 8b: NUMT Contamination Filter

Removes AT2G07xxx pericentromeric NUMT pseudogene contaminants from the
feature matrix. See data/raw/2026_Pbody_NUMTcontam_REPORT_v1.md for the
full contamination analysis.

Classification logic (driven by data/raw/2026_Pbody_NUMTcontam_CSV_v1.csv):
  NUMT_contaminant   → EXCLUDE (mitochondrial pseudogenes / TE-derived loci)
  flanking_real_gene → RETAIN  (6 confirmed flanking protein-coding genes)
  ambiguous          → HELD_OUT (10 loci inside NUMT core, manual review needed)
  AT2G06xxx/AT2G07xxx NOT in CSV → EXCLUDE (precautionary; pericentromeric range)
  All other genes    → RETAIN

Outputs:
  data/processed/08_zorc_feature_matrix_numt_clean.csv  — filtered matrix
  data/processed/08_zorc_numt_excluded.csv              — excluded genes (audit)
  data/processed/08_zorc_numt_heldout.csv               — ambiguous genes (review)

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/08b_numt_filter.py --config config/zorc_config.yaml

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="ZORC P8b — NUMT contamination filter")
    p.add_argument("--config", required=True, help="Path to zorc_config.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg  = load_config(args.config)

    proj = Path(cfg.get("project_dir", "~/Documents/ZORC")).expanduser()
    processed_dir = proj / "data" / "processed"
    raw_dir       = proj / "data" / "raw"

    # Input paths
    matrix_path   = proj / cfg["outputs"]["feature_matrix"]
    numt_csv_path = raw_dir / "2026_Pbody_NUMTcontam_CSV_v1.csv"

    # Output paths
    out_clean    = processed_dir / "08_zorc_feature_matrix_numt_clean.csv"
    out_excluded = processed_dir / "08_zorc_numt_excluded.csv"
    out_heldout  = processed_dir / "08_zorc_numt_heldout.csv"

    print("=" * 70)
    print("ZORC — Phase 8b: NUMT Contamination Filter")
    print(f"Feature matrix : {matrix_path}")
    print(f"NUMT CSV       : {numt_csv_path}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df      = pd.read_csv(matrix_path)
    numt_df = pd.read_csv(numt_csv_path)

    gene_col = "geneID" if "geneID" in df.columns else "gene_id"
    print(f"  Feature matrix : {len(df):,} genes × {df.shape[1]} columns  (col: '{gene_col}')")
    print(f"  NUMT CSV       : {len(numt_df):,} entries")
    print(f"\n  Original class distribution:")
    print(f"    pos (class=1): {(df['class'] == 1).sum():,}")
    print(f"    neg (class=0): {(df['class'] == 0).sum():,}")

    # ── 2. Build NUMT lookup ──────────────────────────────────────────────────
    print("\n[2/4] Building NUMT classification lookup...")
    numt_lookup = dict(zip(numt_df["gene_id"], numt_df["classification"]))
    print(f"  NUMT CSV entries       : {len(numt_lookup)}")
    print(f"    NUMT_contaminant     : {sum(v == 'NUMT_contaminant' for v in numt_lookup.values())}")
    print(f"    flanking_real_gene   : {sum(v == 'flanking_real_gene' for v in numt_lookup.values())}")
    print(f"    ambiguous            : {sum(v == 'ambiguous' for v in numt_lookup.values())}")

    # ── 3. Classify genes ─────────────────────────────────────────────────────
    print("\n[3/4] Classifying genes in feature matrix...")

    decisions = []
    n_csv_contam = n_csv_flanking = n_csv_ambig = n_precaution = n_other = 0

    for _, row in df.iterrows():
        gid = row[gene_col]
        in_pericentromeric = gid.startswith("AT2G06") or gid.startswith("AT2G07")

        if gid in numt_lookup:
            cls = numt_lookup[gid]
            if cls == "NUMT_contaminant":
                decision, reason = "EXCLUDE",   "NUMT_contaminant (CSV)"
                n_csv_contam += 1
            elif cls == "flanking_real_gene":
                decision, reason = "RETAIN",    "flanking_real_gene (CSV)"
                n_csv_flanking += 1
            elif cls == "ambiguous":
                decision, reason = "HELD_OUT",  "ambiguous (CSV) — manual review required"
                n_csv_ambig += 1
            else:
                decision, reason = "RETAIN",    f"unknown_csv_class={cls}"
        elif in_pericentromeric:
            decision, reason = "EXCLUDE", "AT2G06/07xxx not in NUMT CSV — precautionary (pericentromeric)"
            n_precaution += 1
        else:
            decision, reason = "RETAIN", "not_pericentromeric"
            n_other += 1

        decisions.append({
            gene_col:         gid,
            "class":          row["class"],
            "split":          row.get("split", ""),
            "numt_decision":  decision,
            "numt_reason":    reason,
        })

    dec_df = pd.DataFrame(decisions)

    retain_mask  = dec_df["numt_decision"] == "RETAIN"
    exclude_mask = dec_df["numt_decision"] == "EXCLUDE"
    heldout_mask = dec_df["numt_decision"] == "HELD_OUT"

    n_retain  = retain_mask.sum()
    n_exclude = exclude_mask.sum()
    n_heldout = heldout_mask.sum()

    print(f"\n  Classification breakdown:")
    print(f"    From CSV — NUMT_contaminant  : {n_csv_contam}")
    print(f"    From CSV — flanking_real_gene: {n_csv_flanking}")
    print(f"    From CSV — ambiguous         : {n_csv_ambig}")
    print(f"    Not in CSV, pericentromeric  : {n_precaution}  (precautionary EXCLUDE)")
    print(f"    Not pericentromeric          : {n_other}  (RETAIN)")

    print(f"\n  Decision totals:")
    print(f"    RETAIN   : {n_retain:,}")
    print(f"    EXCLUDE  : {n_exclude:,}")
    print(f"    HELD_OUT : {n_heldout:,}")

    print(f"\n  Excluded genes class distribution:")
    ex = dec_df[exclude_mask]
    print(f"    class=0 (neg): {(ex['class'] == 0).sum()}")
    print(f"    class=1 (pos): {(ex['class'] == 1).sum()}")

    print(f"\n  Held-out (ambiguous) class distribution:")
    ho = dec_df[heldout_mask]
    print(f"    class=0 (neg): {(ho['class'] == 0).sum()}")
    print(f"    class=1 (pos): {(ho['class'] == 1).sum()}")

    # Build output DataFrames
    df_clean    = df[retain_mask.values].copy().reset_index(drop=True)
    df_excluded = df[exclude_mask.values].copy().reset_index(drop=True)
    df_heldout  = df[heldout_mask.values].copy().reset_index(drop=True)

    # Add audit columns to excluded/heldout
    df_excluded["numt_decision"] = dec_df.loc[exclude_mask, "numt_decision"].values
    df_excluded["numt_reason"]   = dec_df.loc[exclude_mask, "numt_reason"].values
    df_heldout["numt_decision"]  = dec_df.loc[heldout_mask, "numt_decision"].values
    df_heldout["numt_reason"]    = dec_df.loc[heldout_mask, "numt_reason"].values

    print(f"\n  BEFORE filtering: {len(df):,} genes  "
          f"(pos={(df['class']==1).sum()}, neg={(df['class']==0).sum()})")
    print(f"  AFTER filtering : {len(df_clean):,} genes  "
          f"(pos={(df_clean['class']==1).sum()}, neg={(df_clean['class']==0).sum()})")

    if "split" in df_clean.columns:
        print(f"\n  Clean set split distribution:")
        for split in ["train", "val", "test"]:
            s  = df_clean["split"] == split
            print(f"    {split:6s}: {s.sum():4d}  "
                  f"(pos={(s & (df_clean['class']==1)).sum()}, "
                  f"neg={(s & (df_clean['class']==0)).sum()})")

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    print("\n[4/4] Saving outputs...")
    df_clean.to_csv(out_clean,    index=False)
    df_excluded.to_csv(out_excluded, index=False)
    df_heldout.to_csv(out_heldout,  index=False)

    print(f"  Clean matrix  : {out_clean}  ({len(df_clean):,} genes)")
    print(f"  Excluded genes: {out_excluded}  ({len(df_excluded):,} genes)")
    print(f"  Held-out genes: {out_heldout}  ({len(df_heldout):,} genes)")

    print(f"\n{'='*70}")
    print("NUMT FILTER SUMMARY")
    print(f"{'='*70}")
    print(f"  Original dataset : {len(df):,} genes  "
          f"(pos={(df['class']==1).sum()}, neg={(df['class']==0).sum()})")
    print(f"  Excluded (NUMT)  : {n_exclude}  "
          f"(class=0: {(ex['class']==0).sum()}, class=1: {(ex['class']==1).sum()})")
    print(f"  Held-out (ambig) : {n_heldout}  (manual review → {out_heldout.name})")
    print(f"  Clean dataset    : {len(df_clean):,} genes  "
          f"(pos={(df_clean['class']==1).sum()}, neg={(df_clean['class']==0).sum()})")
    print(f"{'='*70}")
    print("\n✓ Phase 8b complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
