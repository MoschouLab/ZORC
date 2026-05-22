# ZORC Pipeline — Continuation Prompt
## Session context: MoschouLab ZORC ML pipeline (PLANTEX / ERC ConG)

**User:** Pepe (José Moya-Cuevas), lead bioinformatics postdoc, MoschouLab / IMBB-FORTH, Rethymno, Crete.
**PI:** Panagiotis Moschou
**Project dir:** `~/Documents/ZORC/`
**Conda envs:**
- `zorc_pipeline` — main pipeline (P1–P5, P7–P9, P10)
- `bioemu` — BioEmu v1.1 + AIUPred v2.0, torch 2.5.1+cu121
**Hardware:** ThinkStation P5 (RTX A5000 25.4GB VRAM, CUDA 12.2)
**GitHub:** owner: jomocue, organisation: MoschouLab
**Repository:** https://github.com/MoschouLab/ZORC ✅ PUBLISHED (2026-05-22)
**Release:** v1.0.1 — https://github.com/MoschouLab/ZORC/releases/tag/v1.0.1
**Zenodo DOI:** 10.5281/zenodo.20342217 ✅ — https://doi.org/10.5281/zenodo.20342217

---

## Project name & acronym
**ZORC — Zip-code Of RNAs that Condense**
Predictive ML pipeline for P-body mRNA enrichment in *Arabidopsis thaliana*.

---

## Pipeline status (as of 2026-04-30 — PORTFOLIO COMPLETO)

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
| P12f | EvidentlyAI + Prometheus monitoring | ✅ complete | `monitoring/`, `scripts/10c_evidently_report.py` |
| P14a | ChromaDB + LangChain RAG (10 PDFs, 1244 vectors) | ✅ complete | `agent/ingest.py`, `agent/rag_query.py`, `agent/notebook_rag.ipynb` |
| P14b | LangGraph agent (3 nodes + conditional edge, Claude claude-sonnet-4-6) | ✅ complete | `agent/zorc_agent.py`, `agent/run_agent.py`, `agent/notebook_agent.ipynb` |

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

## NUMT fix — commit 830d87f (2026-05-06)

**68 NUMT contaminants removed from negative class** (AT2G07xxx pericentromeric
pseudogenes/TE-derived loci). 8 ambiguous loci moved to held-out set.

| | Genes | pos | neg |
|---|---|---|---|
| Original | 1,510 | 889 | 621 |
| Clean (`numt_clean`) | 1,434 | 888 | 546 |

Metric impact (test set, rerun on `08_zorc_feature_matrix_numt_clean.csv`):

| Model | AUROC | AUPRC | F1-macro | HC |
|---|---|---|---|---|
| RF original | 0.7878 | 0.8327 | 0.6942 | 24/25 |
| RF numt_clean | 0.7740 | **0.8447** | 0.6647 | 24/25 |
| XGB original | 0.7879 | 0.8317 | 0.6976 | 23/25 |
| XGB numt_clean | 0.7639 | 0.8235 | 0.6872 | **24/25** |

AUROC drop expected (NUMT sequences were "easy" negatives with organellar
nucleotide composition). AUPRC RF improves +0.012 — more honest estimate.
HC validation maintained 24/25 (RF) and improved 23→24/25 (XGB).

**⚠ PENDING — P9d + P9f rerun on numt_clean dataset:**
`data/processed/08_zorc_feature_matrix_numt_clean.csv` is the correct input
for all final model comparisons and the manuscript. P9d (feature engineering)
and P9f (Platt-calibrated final RF) need to be rerun with this clean matrix
to produce publication-ready metrics. Commit 830d87f contains only the P9
baseline and P9b (XGBoost) reruns.

Files produced (commit 830d87f):
- `scripts/08b_numt_filter.py` — filter script
- `scripts/09e_numt_impact.py` — metric comparison table
- `data/processed/08_zorc_feature_matrix_numt_clean.csv`
- `data/processed/08_zorc_numt_excluded.csv` (audit trail)
- `data/processed/08_zorc_numt_heldout.csv` (8 ambiguous for review)
- `results/09_zorc_rf_model_numt_clean.pkl`
- `results/09b_zorc_xgb_model_numt_clean.json`

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

## P12f — Completado (2026-04-24)

### Deliverables entregados
- `scripts/10c_evidently_report.py` — genera 3 reports HTML con evidently 0.7 API:
  - `monitoring/evidently_reports/drift_report.html` (7.5 MB)
  - `monitoring/evidently_reports/quality_report.html` (4.2 MB)
  - `monitoring/evidently_reports/classification_report.html` (3.6 MB)
- `api/main.py` — instrumentado con `prometheus-fastapi-instrumentator`;
  endpoint `/metrics` activo; gauge `zorc_model_loaded=1.0` en startup
- `monitoring/prometheus_config.yml` — scrape config para localhost:8000/metrics
- Commit: `60deb0d feat(monitoring): add EvidentlyAI drift reports and Prometheus API metrics`

### Nota técnica: evidently 0.7 API
evidently 0.7 cambió la API respecto a 0.4:
- Presets: `from evidently.presets import DataDriftPreset, DataSummaryPreset, ClassificationPreset`
- `Report.run()` retorna un `Snapshot` con `save_html()`
- `ClassificationPreset` requiere `DataDefinition(classification=[BinaryClassification(...)])`
  pasado via `Dataset.from_pandas(df, data_definition=dd)`

---

## P14a — Completado (2026-04-24)

### Deliverables entregados
- `agent/ingest.py` — ingesta 10 PDFs desde `data/papers/` con PyPDFLoader +
  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200);
  embeddings locales `sentence-transformers/all-MiniLM-L6-v2`;
  persiste 1244 vectores en `agent/chroma_db/` (ChromaDB, gitignored)
- `agent/rag_query.py` — función `query_literature(question, k)` que devuelve
  top-k chunks con rank, cosine similarity, PDF source y número de página;
  módulo importable + CLI con `--demo` para las 3 queries estándar
- `agent/notebook_rag.ipynb` — demo ejecutado: 3 queries + score plot
- `agent/ingest_manifest.json` — metadata del corpus (chunks por fuente)
- `data/papers/` — 10 PDFs tracked en git
- Commit: `a4cd868 feat(P14a): add ChromaDB + LangChain RAG over P-body literature`

### Nota técnica: LangChain 1.x imports
LangChain 1.x cambió los imports respecto a 0.2:
- Embeddings: `from langchain_huggingface import HuggingFaceEmbeddings`
- Vector store: `from langchain_chroma import Chroma`
- Paquetes requeridos: `langchain-huggingface`, `langchain-chroma` (además de `langchain-community`)

### Corpus (10 PDFs, 231 páginas, 1244 chunks)
| Fichero | Chunks |
|---------|--------|
| 2025_PLANT-COMMUNICATIONS-D-25-01149... | 297 |
| journal.pbio.3002305.pdf | 229 |
| Bio-protocol5587.pdf | 147 |
| TPC_paper.pdf | 143 |
| embj.2022111885.pdf | 139 |
| koad127.pdf | 97 |
| 1-s2.0-S0006349525034472-main.pdf | 88 |
| 1-s2.0-S1360138523001322-main.pdf | 61 |
| erac497.pdf | 32 |
| s41422-025-01133-4.pdf | 11 |

---

## P14b — Completado (2026-04-30)

### Deliverables entregados
- `agent/zorc_agent.py` — StateGraph LangGraph 1.1.9 con 3 nodos:
  - `get_prediction`: FastAPI `/lookup/{gene_id}` → SQLite fallback directo
  - `retrieve_literature`: `query_literature()` de P14a, k=5 chunks
  - `generate_report`: Claude claude-sonnet-4-6 via Anthropic SDK 0.97
  - Conditional edge: `prob_pos > 0.8` → `retrieve_literature`, else → `generate_report` directo
  - Fallback graceful si `ANTHROPIC_API_KEY` no está en entorno
- `agent/run_agent.py` — CLI: `python agent/run_agent.py AT5G47010`
  - Flags: `--no-llm`, `--json`
  - Multi-gene: `python agent/run_agent.py AT5G47010 AT3G22270 AT1G01470`
- `agent/notebook_agent.ipynb` — demo con 3 genes (AT5G47010, AT1G01470, AT3G22270)

### Verificación (sin API key, FastAPI activa)
```
AT5G47010: prob=0.9303, pred=enriched, conf=high, lit=5 chunks (ruta completa)
AT1G01470: prob=0.2525, pred=not_enriched, conf=low, lit=0 chunks (ruta directa)
```

### Para activar LLM completo
```bash
conda activate zorc_pipeline
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn api.main:app --port 8000 &
python agent/run_agent.py AT5G47010
```

### Nota técnica: LangGraph 1.1.9 API
- `StateGraph(TypedDict)` — `total=False` para campos opcionales del estado
- `workflow.add_conditional_edges(source, fn, {label: dest})` — routing function retorna string
- `agent.get_graph().nodes.keys()` — introspección del grafo
- `agent.get_graph().draw_mermaid_png()` — visualización (requiere pillow)

---

## PORTFOLIO COMPLETO — 2026-04-30

El portfolio computacional ZORC está terminado. Todas las fases de software completadas.

### Resumen de habilidades demostradas

| Área | Fases | Herramientas clave |
|------|-------|--------------------|
| ML end-to-end | P1–P9f | scikit-learn, XGBoost, SHAP, Platt calibration |
| Bioinformatics pipeline | P1–P8 | BioEmu, AIUPred, RNAfold, MDAnalysis, gffread |
| Workflow reproducibility | P10 | Snakemake (15 rules), conda env directives |
| Data Analytics | P11a–d | SQLite, DuckDB, Streamlit, Plotly Dash, Tableau (CSVs prepared for Tableau Public import) |
| MLOps | P12a–b | MLflow (6 runs), DVC (9 artefacts) |
| ML Engineering | P12c–e | FastAPI, Docker Hub, GitHub Actions CI (47 tests, 86% cov) |
| Monitoring | P12f | EvidentlyAI (3 HTML reports), Prometheus /metrics |
| Generative AI / RAG | P14a | ChromaDB, LangChain 1.x, all-MiniLM-L6-v2, 1244 chunks |
| LLM Agents | P14b | LangGraph 1.1.9 StateGraph, Anthropic SDK, conditional routing |

### Métricas finales del modelo (P9f — NUMT-clean, manuscrito)
```
RandomForestClassifier — 500 trees — Platt calibrated — neg=1.5× class weight
Test AUROC  : 0.7695   ← manuscript value (numt_clean dataset)
Test AUPRC  : 0.8350   ← manuscript value
Test F1-macro: 0.6732 (F1-pos=0.7826, F1-neg=0.5638)
HC validation: 24/25 (96%) — lab-curated high-confidence P-body genes
Dataset     : 1,434 genes × 61 features (41 RNA + 18 protein/IDR + 2 engineered)
Model file  : results/09f_rf_final_model_numt_clean.pkl
```

Comparativa completa: `python scripts/09f_manuscript_metrics.py`

### Completado (2026-05-06) ✅
- **api/ + Docker v1.1** ✅ — `api/main.py` carga `09f_rf_final_model_numt_clean.pkl`
  (61 features, CalibratedClassifierCV); `imputation_medians.json` regenerado;
  `docker/Dockerfile` → `moschoulab/zorc-predictor:1.1`; `/health` ✓ `/lookup/AT5G47010` ✓
- **README.md** ✅ — métricas actualizadas a AUROC=0.7695, AUPRC=0.8350, F1=0.6732;
  nota NUMT añadida en tabla de métricas; Docker tag 1.1
- **Dashboards Plotly Dash** ✅ — Pages 4-6 añadidas al `dashboard/dash_app.py`:
  Page 4 (Feature Importance SHAP numt_clean), Page 5 (Probability Landscape),
  Page 6 (Pipeline History original vs numt_clean). Layout refactorizado en dcc.Tabs.

### Commits clave (2026-05-06)
```
8ef8513  docs: update continuation prompt — all three tasks completed
aa96333  feat(dashboard): add Pages 4-6 to Dash app (numt_clean analytics)
3ec7f75  docs(README): update model metrics to NUMT-clean honest values
360233c  feat(api): upgrade to NUMT-clean P9f model — v1.1
2537714  feat(model): rerun P9d+P9f on NUMT-clean dataset — final manuscript metrics
830d87f  fix(data): remove NUMT contaminants from feature matrix
```

### GitHub publicado ✅ — 2026-05-22

**https://github.com/MoschouLab/ZORC** — repo público bajo MoschouLab.

Commits de publicación (sesión 2026-05-22):
```
4e4a5e2  chore: finalize .gitignore for public release
37b5667  docs: publication-ready README with visual design and badges
ea5e3a9  chore: add MIT License
b873cd9  chore(dvc): track large model files via DVC
```
Release v1.0.0 creado vía GitHub REST API — https://github.com/MoschouLab/ZORC/releases/tag/v1.0.0

**Checklist FAIR completado:**
- [x] Snakefile con 15 reglas y conda directives
- [x] README.md profesional en inglés con badges, métricas, confusion matrix, BibTeX
- [x] config/zorc_config.yaml como único entry point de parámetros
- [x] envs/zorc_pipeline.yml + envs/bioemu_ref.yml
- [x] .gitignore finalizado (.env, *.egg-info, *.pkl, chroma_db, mlruns, etc.)
- [x] 11 archivos de modelo grandes trackeados via DVC (no subidos a git)
- [x] MIT LICENSE 2026 José Moya-Cuevas / MoschouLab
- [x] Push exitoso — 624 objetos
- [x] GitHub Release v1.0.0 + v1.0.1 creados
- [x] Zenodo DOI asignado: **10.5281/zenodo.20342217**
- [x] README + BibTeX actualizados con DOI real
- [x] CI verde: Lint + Test (47/47, 85% cov) + Docker build ✅

### Estado final del proyecto — COMPLETO (excepto P13)

**Todo el trabajo computacional está terminado y publicado.**

| Componente | Estado |
|---|---|
| Pipeline P1–P9f | ✅ completo |
| Snakemake P10 | ✅ completo |
| Dashboards P11 | ✅ completo |
| MLOps P12 | ✅ completo |
| LLM Agent P14 | ✅ completo |
| GitHub público | ✅ https://github.com/MoschouLab/ZORC |
| Zenodo DOI | ✅ 10.5281/zenodo.20342217 |
| CI/CD | ✅ verde (Lint + Test + Docker) |
| **P13 espacial** | ⏳ pendiente experimental (antes verano 2026) |

### Pendiente (no computacional)
- **P13** — spatial transcriptomics validation: STOmics/MERFISH + expansion microscopy + padlock probes; facility submission before summer 2026

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
- Published: https://github.com/MoschouLab/ZORC | DOI: 10.5281/zenodo.20342217

---

*Prompt updated: 2026-05-22 · **PORTFOLIO COMPLETO + PUBLICADO EN GITHUB + ZENODO** · P1–P9f + P10 + P11a–d + P12a–f + P14a–b ✅ · https://github.com/MoschouLab/ZORC ✅ · DOI 10.5281/zenodo.20342217 ✅ · CI verde ✅ · **Solo pendiente: P13 validación espacial experimental (manual, antes verano 2026)***
