# ZORC Pipeline — Continuation Prompt
## Session context: MoschouLab ZORC ML pipeline (PLANTEX / ERC ConG)

**User:** Pepe (José Moya-Cuevas), lead bioinformatics postdoc, MoschouLab / IMBB-FORTH, Rethymno, Crete.
**PI:** Panagiotis Moschou
**Project dir:** `~/Documents/ZORC/`
**Conda envs:**
- `zorc_pipeline` — main pipeline (P1–P5, P7–P9, P10)
- `bioemu` — BioEmu v1.1 + AIUPred v2.0, torch 2.5.1+cu121
**Hardware:** ThinkStation P5 (RTX A5000 25.4GB VRAM, CUDA 12.2)
**GitHub:** owner: jomocue, organisation: MoschouLab (repo not yet published — P10)

---

## Project name & acronym
**ZORC — Zip-code Of RNAs that Condense**
Predictive ML pipeline for P-body mRNA enrichment in *Arabidopsis thaliana*.

---

## Pipeline status (as of April 2026)

| Phase | Script | Status | Key output |
|-------|--------|--------|------------|
| P1 | `01_build_coregulon.py` | ✅ complete | `data/processed/01_zorc_coregulon_list.csv` |
| P2 | `02_map_isoforms.py` | ✅ complete | `data/processed/02_zorc_isoform_map.csv` |
| P3 | `03_fetch_sequences.py` | ✅ complete | `data/processed/sequences/` |
| P4 | `04_rna_features.py` | ✅ complete | `data/processed/04_rna_features.csv` |
| P5 | `05_aiupred_idr.py` | ✅ complete | `data/processed/05_aiupred_idr_summary.csv` |
| P6 | `06_bioemu_batch.py` | ✅ complete | `data/processed/bioemu/` (checkpoint.json) |
| P7 | `07_protein_features.py` | ✅ complete | `data/processed/07_protein_features.csv` |
| P8 | `08_feature_matrix.py` | ✅ complete | `data/processed/08_zorc_feature_matrix.csv` |
| P9–P9f | `09_random_forest.py` … `09f_final_model.py` | ✅ complete | `results/09f_rf_final_model.pkl` |
| P10 | Snakemake workflow (15 rules) + `pipeline_dag.png` | ✅ complete | `Snakefile` |
| P11a–d | SQLite+DuckDB+notebooks, Streamlit, Dash, Tableau CSVs | ✅ complete | `dashboard/`, `notebooks/` |
| P12a | MLflow (6 runs retroactive) | ✅ complete | `mlruns/` |
| P12b | DVC (9 artefacts) | ✅ complete | `dvc/` |
| P12c | FastAPI (`/health`, `/predict`, `/lookup`) | ✅ complete | `api/main.py` |
| P12d | Docker (`moschoulab/zorc-predictor:1.0`, ViennaRNA from source) | ✅ complete | `docker/Dockerfile` |
| P12e | GitHub Actions CI (lint + test + docker-build) | ✅ complete | `.github/workflows/ci.yml` |
| P12f | EvidentlyAI + Prometheus monitoring | 🔄 next | `monitoring/` |

---

## Dataset overview

### Coregulon (P1)
- **Positives (class=1):** 884 genes — RNA enriched in T-RIP (NS or HS) AND protein log2FC ≥ 0 in any APEAL condition (AP or PDL)
- **Negatives (class=0):** 688 genes — RNA depleted from P-bodies
- **Total:** 1,572 genes (ratio 1:0.78 — near-balanced)
- Conflict resolution: "strongest" signal (10 genes resolved)
- Sources: Liu et al. 2024 Plant Cell (T-RIP) + Liu et al. 2023 EMBO J (APEAL proteomics)

### Isoform map (P2)
- 1,510 genes with transcript assigned (68 dropped: pseudogenes/TEs/tRNAs)
- 303 rMATS-guided isoforms (20%), 1,201 canonical fallback (80%)

### Sequences (P3)
- 1,501 mRNA sequences, 1,457 protein sequences
- 45 QC-flagged (no_CDS / short_protein / internal_stop) — included with flag

### Feature matrix (P8)
- **1,510 samples × 59 features**
- RNA features (41): composition, dinucleotides, UTR lengths, m6A motifs, RNAfold MFE
- Protein/IDR features (18): IDR%, rmsf_*, rg_*, contact_density, pass_rate, n_residues
- Split: train=1,064 / val=212 / test=234 (CD-HIT 40% identity anti-leakage)

---

## P6 — BioEmu status
- Completed: 757/759 (99.7%)
- Tier 1 (IDR<10%, AF2 only): 698 proteins — **NO BioEmu dynamic features**
- Tier 2 (IDR 10-30%, 50 conf): 328 proteins
- Tier 3 (IDR≥30%, 100 conf): 431 proteins
- Excluded (OOM >3000aa): AT2G28290.5 (3575aa), AT2G45540.2 (3001aa)
- Runtime: ~5 days continuous on RTX A5000

---

## P9 — Random Forest results (BASELINE)

```
Model: RandomForestClassifier, n_estimators=500, class_weight='balanced'
Features: 56 (59 - 3 meta columns)
Imputation: global median for dynamic features + has_bioemu flag (0/1)

OOB score : 0.7397
Val  AUROC: 0.7990  AUPRC: 0.8312  F1-macro: 0.7247
Test AUROC: 0.7974  AUPRC: 0.8469  F1-macro: 0.7382
HC validation (25 colleague genes): 24/25 (96%)
  Only miss: AT3G55280 (RPL23aB, ribosomal protein, P(pos)=0.42)
```

### SHAP Top 20 features
1. mrna_length (0.0389)
2. cds_length (0.0387)
3. n_residues (0.0243)
4. di_CG (0.0242)
5. di_UA (0.0229)
6. utr3_au_content (0.0210)
7. utr3_length (0.0169)
8. di_UG (0.0150)
9. utr5_length (0.0146)
10. utr5_fraction (0.0136)
11. cds_fraction (0.0121)
12. fU (0.0114)
13. mfe_per_nt (0.0105)
14. fG (0.0102)
15. di_CA (0.0097)
16. rrach_per_kb (0.0088)
17. di_UU (0.0086)
18. di_CU (0.0083)
19. di_AG (0.0082)
20. au_content (0.0070)

### Critical limitation identified
`idr_percent`, `rmsf_mean`, `rg_mean`, `rg_cv` are **absent from Top 20**
despite showing 1.25–1.28× class separation in P7. Root cause:
**49.5% imputation** — 754/1,510 Tier 1 proteins have global median substituted
for dynamic features, diluting their SHAP contributions by ~50%.

---

## P12e — CI/CD summary (completed 2026-04-24)

- **47 tests** across `tests/test_pipeline.py` (31) and `tests/test_api.py` (16)
- **86% coverage** on `api/` (well above 60% minimum)
- **3 CI jobs:** `lint` (ruff --select=E,F on api/ + tests/), `test` (pytest --cov),
  `docker-build` (multi-stage Dockerfile, no push)
- RNAfold mocked via `unittest.mock.patch` — tests run without ViennaRNA in runner
- `requirements.txt` added at repo root (pip-compatible, no conda-only packages)
- CI badge active in `README.md`

---

## Tarea actual — P12f: EvidentlyAI + Prometheus

Siguiente paso según `docs/ZORC_P10_P14_architecture.md` §P12f:

### P12f-a — EvidentlyAI data drift report

**Script:** `scripts/10c_evidently_report.py`

```bash
conda activate zorc_pipeline
pip install evidently
python scripts/10c_evidently_report.py --config config/zorc_config.yaml
```

Generates HTML reports in `monitoring/evidently_reports/`:
- `DataDriftPreset` — training vs new prediction requests feature distribution
- `DataQualityPreset` — missing values, outliers
- Reference = `data/processed/08_zorc_feature_matrix.csv` (X_train split)
- Current = synthetic batch of 50 random genes from the test split

### P12f-b — Prometheus metrics in FastAPI

Add to `api/main.py` (5 lines):

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
# Metrics available at GET /metrics
```

Install: `pip install prometheus-fastapi-instrumentator`

Create `monitoring/prometheus_config.yml` for local scraping.

### Deliverables
- `scripts/10c_evidently_report.py`
- `monitoring/evidently_reports/drift_report.html`
- `monitoring/prometheus_config.yml`
- Updated `api/main.py` with Prometheus instrumentation
- Commit: `feat(monitoring): add EvidentlyAI drift report and Prometheus metrics`

---

## Pending after P12f

- **P14a** — ChromaDB + LangChain RAG on P-body literature (`agent/literature_agent.py`)
- **P14b** — LangGraph multi-step ZORC prediction + literature agent (`agent/zorc_agent.py`)
- **P13** — Xenium 10X probe design + facility submission (manual, before summer 2026)

---

## Key reference files

```
~/Documents/ZORC/
├── config/zorc_config.yaml              ← all parameters (thresholds, paths, ML params)
├── scripts/01_build_coregulon.py        ← P1
├── scripts/02_map_isoforms.py           ← P2
├── scripts/03_fetch_sequences.py        ← P3
├── scripts/04_rna_features.py           ← P4
├── scripts/05_aiupred_idr.py            ← P5 (run in bioemu env)
├── scripts/06_bioemu_batch.py           ← P6 (run in bioemu env)
├── scripts/07_protein_features.py       ← P7
├── scripts/08_feature_matrix.py         ← P8
├── scripts/09_random_forest.py          ← P9 baseline RF
├── data/raw/
│   ├── EMBOJ-2022-111885_SourceFile1_.xlsx      ← APEAL proteomics
│   ├── tpc_23_01160Supplemental_Data_Sets_1XXX10.xlsx  ← T-RIP RNA lists
│   └── RSBs_DCP1_related_under_heat_stress.xlsx ← colleague HC set
├── data/processed/
│   ├── 01_zorc_coregulon_list.csv
│   ├── 02_zorc_isoform_map.csv
│   ├── 03_sequence_manifest.csv
│   ├── 04_rna_features.csv
│   ├── 05_aiupred_idr_summary.csv
│   ├── 05_bioemu_tier_assignments.csv
│   ├── 07_protein_features.csv
│   ├── 08_zorc_feature_matrix.csv       ← 1510 × 59 features
│   ├── 08_zorc_split_assignments.csv
│   ├── colleague_high_confidence_set.csv ← 25 HC genes
│   └── sequences/                       ← FASTAs (gitignored)
│   └── bioemu/                          ← BioEmu outputs (gitignored)
├── results/
│   ├── 09_zorc_rf_model.pkl
│   ├── 09_zorc_shap_values.csv
│   ├── 09_zorc_shap_importance.csv
│   └── 09_zorc_predictions.csv
├── logs/                                ← audit reports per phase
├── docs/
│   ├── ZORC_continuation_prompt.md      ← THIS FILE
│   └── ZORC_methodological_decisions.md ← permanent decisions log
└── envs/
    ├── zorc_pipeline.yml
    └── bioemu_ref.yml
```

---

## Working style
- Pepe runs all code locally on ThinkStation P5; Claude generates code
- specs-driven / vibe-coding hybrid
- Every completed phase → git commit before next phase
- All parameters in config.yaml (FAIR principle)
- Scripts auto-add their output keys to config.yaml if missing
- Target publication: MoschouLab/ZORC on GitHub + Zenodo DOI

---

*Prompt updated: 2026-04-24 · Pipeline state: P1–P12e complete · Next: P12f (EvidentlyAI + Prometheus)*
