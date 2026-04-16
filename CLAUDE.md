# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**ZORC** (Zip-code Of RNAs that Condense) is a bioinformatics ML pipeline predicting P-body RNA enrichment in *Arabidopsis thaliana* using isoform mRNA sequence features, protein conformational ensembles (BioEmu), and IDR features (AIUPred). The pipeline is complete through Phase 9f (15 scripts).

**Lab:** MoschouLab, IMBB-FORTH / University of Crete  
**ERC Grant:** Consolidator Grant PLANTEX

## Environment setup

Two conda environments — keep them separate (BioEmu has strict CUDA/torch requirements):

```bash
# Main pipeline (P1–P5, P7–P9)
conda env create -f envs/zorc_pipeline.yml
conda activate zorc_pipeline

# BioEmu conformational sampling only (P6)
# bioemu_ref.yml lives in ~/Documents/TRIP-isoform-lncoding-pipeline/
conda activate bioemu
```

## Running the pipeline

Every script accepts `--config config/zorc_config.yaml`. Run from the project root:

```bash
python scripts/01_build_coregulon.py --config config/zorc_config.yaml
python scripts/02_map_isoforms.py --config config/zorc_config.yaml
python scripts/03_fetch_sequences.py --config config/zorc_config.yaml
python scripts/04_rna_features.py --config config/zorc_config.yaml   # add --no-rnafold to skip structure
python scripts/05_aiupred_idr.py --config config/zorc_config.yaml    # run in `bioemu` env
python scripts/06_bioemu_batch.py --config config/zorc_config.yaml   # GPU required, RTX A5000
python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml  # AF2 recovery for Tier1
python scripts/07_protein_features.py --config config/zorc_config.yaml
python scripts/08_feature_matrix.py --config config/zorc_config.yaml
python scripts/09_random_forest.py --config config/zorc_config.yaml
python scripts/09b_xgboost.py --config config/zorc_config.yaml
python scripts/09f_final_model.py --config config/zorc_config.yaml
```

There is no test suite or linter yet (planned for P10 via GitHub Actions).

## Architecture

### Configuration-driven design
All parameters live in `config/zorc_config.yaml` — paths, ML hyperparameters, BioEmu tier thresholds, class weights. Scripts read this file at startup. **Never hardcode paths or parameters in scripts.**

### Linear phase dependencies
Scripts are numbered `01_` through `09f_`. Each reads from `data/processed/` outputs of prior phases and writes its own CSV outputs + a structured text report to `logs/`. The numbered prefix maps directly to planned Snakemake rules (P10).

### Data flow
```
P1 (coregulon: 884 pos / 688 neg, 1,572 total)
  → P2 (isoform selection: rMATS-guided or canonical fallback)
    → P3 (sequence extraction via gffread)
      → P4 (41 RNA features; RNAfold MFE-only, max 3000 nt)
      → P5 (AIUPred IDR% → BioEmu tier assignment)
        → P6 (BioEmu ensembles, GPU; Tier2=50 conf, Tier3=100 conf)
        → P6b (AF2 structures for Tier1 proteins)
          → P7 (18 protein features from RMSF, IDR, length)
            → P8 (63-feature matrix; CD-HIT 40% identity anti-leakage split)
              → P9/P9b/P9d/P9f (RF + XGBoost, SHAP, Platt calibration)
```

### ML details
- **63 features:** 41 RNA (nucleotide/dinucleotide composition, m6A motifs, UTR fractions, RNAfold MFE/structure), 18 protein (RMSF N/C-terminal dynamics, IDR%, length), 2 binary flags
- **Split:** CD-HIT cluster-based 70/15/15 (train/val/test) — entire clusters stay together to prevent paralog leakage
- **Class imbalance:** `class_weight="balanced"` on RF; no SMOTE (BioEmu features are physically meaningful, interpolation is biologically nonsensical)
- **Final model:** RF 500 trees with Platt calibration (`09f_rf_final_model.pkl`); AUROC 0.7963, AUPRC 0.8431 on test set
- **High-confidence validation:** 24/25 colleague-curated genes predicted correctly (file: `data/processed/colleague_high_confidence_set.csv`)

### Missing values
- BioEmu dynamic features → NaN for Tier1/OOM-excluded proteins (AT2G28290, AT2G45540)
- RNAfold features → NaN for sequences > 3000 nt (68 sequences)
- Imputed in P9 via class-conditional median (not global median)

## Key methodological decisions

**Isoform vs. canonical protein (CRITICAL):** APEAL proteomics AGI codes are used only as matching keys to confirm P-body protein presence. All downstream features (P2–P7) use the *specific isoform sequence* from rMATS or the canonical transcript — never the canonical protein from databases. This is the core biological premise.

**Positive class:** T-RIP enriched (NS or HS) AND protein log2FC ≥ 0 in ANY of {AP_NS, AP_HS, PDL_NS, PDL_HS}. The `protein_enrichment_threshold` in config is deliberately 0.0 (relaxed) to retain borderline cases.

**BioEmu known bugs** (see `docs/ZORC_methodological_decisions.md` §6.2):
- `unphysical_filter_bug` → rerun with `--filter_samples=False`
- Timeout (>1h) → rerun with `--timeout 10800`
- OOM for proteins >3000 aa → mark as `excluded`, treat as Tier1

**RNAfold:** MFE-only (`--noPS`); partition function was too slow (O(n³) vs O(n²)). Dot-plot `.ps` files are written to CWD regardless of `--noPS` — they are gitignored.

**rMATS parsing:** GeneID fields are double-quoted; some gene symbol fields contain embedded tabs. Use `quoting=csv.QUOTE_NONE` + manual quote stripping.

## External dependencies (non-Python)

| Tool | Version | Used in |
|------|---------|---------|
| ViennaRNA / RNAfold | 2.6+ | P4 |
| CD-HIT | 4.8 | P8 |
| gffread | 0.12.7 | P3 |
| AIUPred | v2.0 | P5 (run in `bioemu` env with `--force-cpu`) |
| BioEmu | v1.1 | P6 (GPU required) |
| AlphaFold2 | — | P6b |

Reference genome files are in `~/Documents/TRIP-isoform-lncoding-pipeline/05_reference/` (outside this repo).

## Important files

| File | Purpose |
|------|---------|
| `config/zorc_config.yaml` | Single source of truth for all parameters and output paths |
| `docs/ZORC_methodological_decisions.md` | Full rationale for every non-obvious decision; consult before changing pipeline logic |
| `docs/ZORC_P10_P14_architecture.md` | Roadmap: Snakemake (P10), dashboards (P11), MLOps/Docker/FastAPI (P12), RAG agent (P14) |
| `docs/ZORC_continuation_prompt.md` | Session context snapshot (current pipeline state, data summaries) |
| `data/processed/colleague_high_confidence_set.csv` | 25 lab-curated high-confidence P-body genes; primary validation set |

## What is gitignored

- `data/raw/` — third-party source data (not redistributed)
- `data/processed/bioemu/AT*/` — BioEmu trajectory files (`.xtc`, `.pdb`; large, regenerable)
- `*.ps`, `*.eps` — RNAfold dot-plot PostScript files
- Large intermediate FASTAs and AF2 cache entries
