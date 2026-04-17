"""
P11a — Build ZORC SQLite database
==================================
Creates data/zorc_database.db with three normalised tables:

  genes       — gene metadata + BioEmu tier + train/val/test split
  features    — RNA, protein/IDR, and engineered features (subset)
  predictions — P9f model probabilities, predictions, per-gene SHAP values

Sources
-------
  data/processed/02_zorc_isoform_map.csv          → transcript, condition, description
  data/processed/05_bioemu_tier_assignments.csv   → bioemu_tier
  data/processed/08_zorc_split_assignments.csv    → split
  data/processed/08_zorc_feature_matrix_eng.csv   → all features
  results/09f_predictions_final.csv               → prob_pos, pred  (P9f)
  results/09_zorc_shap_values.csv                 → per-gene SHAP  (P9 baseline; 500 genes)

Output
------
  data/zorc_database.db    — SQLite relational database
  logs/10a_build_database_report.txt

Usage
-----
  conda activate zorc_pipeline
  python scripts/10a_build_database.py --config config/zorc_config.yaml
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def ensure_key(cfg: dict, section: str, key: str, value) -> None:
    if key not in cfg.get(section, {}):
        cfg.setdefault(section, {})[key] = value
        log.info("Added missing config key: %s.%s = %s", section, key, value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sources(cfg: dict) -> dict:
    processed = Path(cfg["data"]["processed_dir"])
    results = Path(cfg["data"]["results_dir"])

    log.info("Loading isoform map …")
    isoform = pd.read_csv(processed / "02_zorc_isoform_map.csv")

    log.info("Loading BioEmu tier assignments …")
    tiers = pd.read_csv(processed / "05_bioemu_tier_assignments.csv")

    log.info("Loading split assignments …")
    splits = pd.read_csv(processed / "08_zorc_split_assignments.csv")

    log.info("Loading engineered feature matrix …")
    feats = pd.read_csv(processed / "08_zorc_feature_matrix_eng.csv")

    log.info("Loading P9f predictions …")
    preds = pd.read_csv(results / "09f_predictions_final.csv")

    log.info("Loading per-gene SHAP values (P9 baseline) …")
    shap_vals = pd.read_csv(results / "09_zorc_shap_values.csv")

    return {
        "isoform": isoform,
        "tiers": tiers,
        "splits": splits,
        "feats": feats,
        "preds": preds,
        "shap_vals": shap_vals,
    }


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_genes(sources: dict) -> pd.DataFrame:
    """
    genes (gene_id PK, transcript_id, gene_name, class, condition,
           bioemu_tier, split)
    """
    isoform = sources["isoform"][
        ["geneID", "transcript_id", "class", "condition", "description_prot"]
    ].copy()
    isoform = isoform.rename(columns={"geneID": "gene_id"})
    # Drop exact duplicate rows (some geneIDs appear multiple times in isoform_map)
    isoform = isoform.drop_duplicates(subset="gene_id")

    # Extract short gene name from description_prot  (e.g. "DCP1:p, ..." → "DCP1")
    isoform["gene_name"] = (
        isoform["description_prot"]
        .str.split(":p,").str[0]
        .str.split(":p").str[0]
        .str.strip()
    )
    isoform = isoform.drop(columns=["description_prot"])

    tiers = (sources["tiers"][["geneID", "bioemu_tier"]]
             .rename(columns={"geneID": "gene_id"})
             .drop_duplicates(subset="gene_id"))
    splits = (sources["splits"][["geneID", "split"]]
              .rename(columns={"geneID": "gene_id"})
              .drop_duplicates(subset="gene_id"))

    genes = (
        isoform
        .merge(tiers, on="gene_id", how="left")
        .merge(splits, on="gene_id", how="left")
    )

    # Coerce types
    genes["class"] = genes["class"].astype(int)
    genes["bioemu_tier"] = genes["bioemu_tier"].astype("Int64")  # nullable int

    log.info("genes table: %d rows", len(genes))
    return genes


def build_features(sources: dict) -> pd.DataFrame:
    """
    features (gene_id PK, <RNA features>, <protein/IDR features>,
              <engineered features>)
    Stores the subset defined in ZORC_P10_P14_architecture.md §P11a.
    """
    feats = (sources["feats"]
             .rename(columns={"geneID": "gene_id"})
             .drop_duplicates(subset="gene_id"))

    rna_cols = [
        "mrna_length", "cds_length", "utr3_length", "utr5_length",
        "utr3_au_content", "rrach_count", "rrach_per_kb",
        "di_CG", "di_UA", "di_UG",
        "mfe", "mfe_per_nt",
        # extra RNA features useful for dashboards
        "au_content", "gc_content", "frac_paired", "n_stemloops_per_kb",
        "utr3_fraction", "utr5_fraction", "cds_fraction",
        "long_3utr", "ptc_proxy",
    ]
    prot_cols = [
        "idr_percent", "rmsf_mean", "rmsf_nterm50", "rmsf_cterm50",
        "rmsf_std", "rmsf_max",
        "rg_mean", "rg_std", "rg_cv",
        "packing_density", "sasa_per_residue",
        "contact_density", "pass_rate",
        "n_residues",
        "mean_disorder", "max_disorder_window", "n_idr_regions",
        "longest_idr_region",
    ]
    eng_cols = ["rmsf_nterm_cterm_ratio", "utr3_au_x_length",
                "packing_x_idr", "rrach_per_cds_kb"]

    keep = ["gene_id"] + [c for c in rna_cols + prot_cols + eng_cols
                          if c in feats.columns]
    missing = [c for c in rna_cols + prot_cols + eng_cols
               if c not in feats.columns]
    if missing:
        log.warning("Features not found in matrix, skipping: %s", missing)

    features = feats[keep].copy()
    log.info("features table: %d rows × %d columns", len(features), len(features.columns))
    return features


def build_predictions(sources: dict, feats_df: pd.DataFrame) -> pd.DataFrame:
    """
    predictions (gene_id PK, prob_pos, pred,
                 shap_mrna_length, shap_cds_length, shap_di_CG,
                 shap_utr3_au_content, shap_rrach_per_kb,
                 shap_rmsf_nterm50)
    Per-gene SHAP available only for the 500 genes in 09_zorc_shap_values.csv.
    Others get NULL.
    """
    preds = (sources["preds"][["gene_id", "prob_pos", "pred"]]
             .drop_duplicates(subset="gene_id")
             .copy())

    # Map sample_idx → gene_id using the feature matrix row order
    feat_mat = sources["feats"]
    shap_vals = sources["shap_vals"].copy()

    # sample_idx is the integer row position (iloc) in the feature matrix
    idx_to_gene = feat_mat["geneID"].reset_index().rename(
        columns={"index": "sample_idx", "geneID": "gene_id"}
    )
    shap_vals = shap_vals.merge(idx_to_gene, on="sample_idx", how="left")

    # SHAP features to store (subset matching architecture spec, using actual column names)
    shap_feature_cols = {
        "mrna_length":    "shap_mrna_length",
        "cds_length":     "shap_cds_length",
        "di_CG":          "shap_di_CG",
        "utr3_au_content":"shap_utr3_au_content",
        "rrach_per_kb":   "shap_rrach_per_kb",
        "rmsf_nterm50":   "shap_rmsf_nterm50",
    }

    shap_keep = ["gene_id"] + [c for c in shap_feature_cols.keys()
                               if c in shap_vals.columns]
    shap_sub = (shap_vals[shap_keep]
                .rename(columns=shap_feature_cols)
                .drop_duplicates(subset="gene_id"))

    # Merge predictions with SHAP (left join → NULL for unsampled genes)
    pred_table = preds.merge(shap_sub, on="gene_id", how="left")
    pred_table = pred_table.drop_duplicates(subset="gene_id")

    log.info(
        "predictions table: %d rows, SHAP available for %d/%d genes",
        len(pred_table),
        pred_table["shap_mrna_length"].notna().sum(),
        len(pred_table),
    )
    return pred_table


# ---------------------------------------------------------------------------
# SQLite writer
# ---------------------------------------------------------------------------

def write_sqlite(db_path: Path, tables: dict) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
        log.info("Removed existing database at %s", db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE genes (
        gene_id       TEXT PRIMARY KEY,
        transcript_id TEXT NOT NULL,
        gene_name     TEXT,
        class         INTEGER NOT NULL,
        condition     TEXT,
        bioemu_tier   INTEGER,
        split         TEXT
    );

    CREATE TABLE features (
        gene_id                  TEXT PRIMARY KEY
                                 REFERENCES genes(gene_id),
        -- RNA features
        mrna_length              REAL,
        cds_length               REAL,
        utr3_length              REAL,
        utr5_length              REAL,
        utr3_au_content          REAL,
        rrach_count              REAL,
        rrach_per_kb             REAL,
        di_CG                    REAL,
        di_UA                    REAL,
        di_UG                    REAL,
        mfe                      REAL,
        mfe_per_nt               REAL,
        au_content               REAL,
        gc_content               REAL,
        frac_paired              REAL,
        n_stemloops_per_kb       REAL,
        utr3_fraction            REAL,
        utr5_fraction            REAL,
        cds_fraction             REAL,
        long_3utr                INTEGER,
        ptc_proxy                INTEGER,
        -- Protein / IDR features
        idr_percent              REAL,
        rmsf_mean                REAL,
        rmsf_nterm50             REAL,
        rmsf_cterm50             REAL,
        rmsf_std                 REAL,
        rmsf_max                 REAL,
        rg_mean                  REAL,
        rg_std                   REAL,
        rg_cv                    REAL,
        packing_density          REAL,
        sasa_per_residue         REAL,
        contact_density          REAL,
        pass_rate                REAL,
        n_residues               REAL,
        mean_disorder            REAL,
        max_disorder_window      REAL,
        n_idr_regions            REAL,
        longest_idr_region       REAL,
        -- Engineered features
        rmsf_nterm_cterm_ratio   REAL,
        utr3_au_x_length         REAL,
        packing_x_idr            REAL,
        rrach_per_cds_kb         REAL
    );

    CREATE TABLE predictions (
        gene_id               TEXT PRIMARY KEY
                              REFERENCES genes(gene_id),
        prob_pos              REAL,
        pred                  INTEGER,
        -- Per-gene SHAP values (P9 baseline RF; NULL if not in SHAP sample)
        shap_mrna_length      REAL,
        shap_cds_length       REAL,
        shap_di_CG            REAL,
        shap_utr3_au_content  REAL,
        shap_rrach_per_kb     REAL,
        shap_rmsf_nterm50     REAL
    );
    """)

    # Insert rows from DataFrames
    for table_name, df in tables.items():
        # Only insert columns that exist in both the DataFrame and the schema
        cur.execute(f"PRAGMA table_info({table_name})")
        schema_cols = {row[1] for row in cur.fetchall()}
        df_cols = [c for c in df.columns if c in schema_cols]
        df_sub = df[df_cols].where(pd.notnull(df[df_cols]), None)
        df_sub.to_sql(table_name, con, if_exists="append", index=False)
        log.info("  Inserted %d rows into %s", len(df_sub), table_name)

    con.commit()
    con.close()
    log.info("SQLite database written to %s (%.1f KB)",
             db_path, db_path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    db_path: Path,
    tables: dict,
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "10a_build_database_report.txt"

    genes = tables["genes"]
    feats = tables["features"]
    preds = tables["predictions"]

    n_pos = (genes["class"] == 1).sum()
    n_neg = (genes["class"] == 0).sum()
    tier_counts = genes["bioemu_tier"].value_counts().sort_index().to_dict()
    split_counts = genes["split"].value_counts().to_dict()
    shap_covered = preds["shap_mrna_length"].notna().sum()

    lines = [
        "=" * 60,
        "ZORC P11a — Database Build Report",
        "=" * 60,
        f"Database   : {db_path}",
        f"Size       : {db_path.stat().st_size / 1024:.1f} KB",
        "",
        "-- genes table --",
        f"  Total    : {len(genes)}",
        f"  Positive : {n_pos}",
        f"  Negative : {n_neg}",
        f"  BioEmu tiers: {tier_counts}",
        f"  Splits       : {split_counts}",
        "",
        "-- features table --",
        f"  Rows     : {len(feats)}",
        f"  Columns  : {len(feats.columns)}",
        "",
        "-- predictions table --",
        f"  Rows     : {len(preds)}",
        f"  SHAP covered : {shap_covered}/{len(preds)} genes",
        f"  Prob range : {preds['prob_pos'].min():.3f} – {preds['prob_pos'].max():.3f}",
        f"  Predicted pos: {(preds['pred']==1).sum()} / neg: {(preds['pred']==0).sum()}",
        "",
        "Tables: genes, features, predictions",
        "FK constraints: features.gene_id → genes, predictions.gene_id → genes",
        "=" * 60,
    ]

    report_path.write_text("\n".join(lines) + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="P11a — Build ZORC SQLite database")
    parser.add_argument("--config", default="config/zorc_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Ensure output key exists
    ensure_key(cfg, "data", "database", "data/zorc_database.db")

    db_path = Path(cfg["data"]["database"])
    log_dir = Path(cfg["data"].get("logs_dir", "logs"))

    sources = load_sources(cfg)

    log.info("Building tables …")
    genes_df = build_genes(sources)
    feats_df = build_features(sources)
    preds_df = build_predictions(sources, feats_df)

    tables = {
        "genes": genes_df,
        "features": feats_df,
        "predictions": preds_df,
    }

    log.info("Writing SQLite …")
    write_sqlite(db_path, tables)

    write_report(db_path, tables, log_dir)

    log.info("P11a complete. Database: %s", db_path)


if __name__ == "__main__":
    main()
