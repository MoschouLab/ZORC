# ZORC — Zip-code Of RNAs that Condense

[![CI](https://github.com/MoschouLab/ZORC/actions/workflows/ci.yml/badge.svg)](https://github.com/MoschouLab/ZORC/actions/workflows/ci.yml)

Predictive ML pipeline for P-body mRNA enrichment in *Arabidopsis thaliana*,
using isoform sequences, conformational ensembles (BioEmu), and IDR features (AIUPred).

**Project:** ERC Consolidator Grant PLANTEX  
**Lab:** MoschouLab, IMBB-FORTH / University of Crete  
**Author:** José Moya-Cuevas

## Model performance

| Metric | Value |
|--------|-------|
| Test AUROC | 0.7963 |
| Test AUPRC | 0.8431 |
| Test F1-macro | 0.7229 |
| HC validation | 24/25 (96%) |

Final model: RandomForestClassifier (500 trees, Platt-calibrated), 63 features
(41 RNA + 18 protein/IDR + 4 engineered).

## Pipeline phases

| Phase | Script | Status |
|-------|--------|--------|
| P1 | `01_build_coregulon.py` | ✅ complete |
| P2 | `02_map_isoforms.py` | ✅ complete |
| P3 | `03_fetch_sequences.py` | ✅ complete |
| P4 | `04_rna_features.py` | ✅ complete |
| P5 | `05_aiupred_idr.py` | ✅ complete |
| P6 | `06_bioemu_batch.py` | ✅ complete |
| P7 | `07_protein_features.py` | ✅ complete |
| P8 | `08_feature_matrix.py` | ✅ complete |
| P9–P9f | `09_random_forest.py` … `09f_final_model.py` | ✅ complete |
| P10 | Snakemake workflow | ✅ complete |
| P11a–d | SQLite/DuckDB, Streamlit, Dash, Tableau | ✅ complete |
| P12a | MLflow experiment tracking | ✅ complete |
| P12b | DVC data & model versioning | ✅ complete |
| P12c | FastAPI prediction service | ✅ complete |
| P12d | Docker image (moschoulab/zorc-predictor:1.0) | ✅ complete |
| P12e | GitHub Actions CI/CD | ✅ complete |
| P12f | EvidentlyAI drift reports + Prometheus metrics | ✅ complete |

## Monitoring

### Data drift reports (EvidentlyAI)

```bash
conda activate zorc_pipeline
pip install evidently
python scripts/10c_evidently_report.py --config config/zorc_config.yaml
# Reports saved to monitoring/evidently_reports/
#   drift_report.html          — feature distribution shift (train vs test)
#   quality_report.html        — missing values, outliers, data summary
#   classification_report.html — F1, ROC-AUC, precision/recall on test split
```

### API metrics (Prometheus)

The FastAPI service exposes a `/metrics` endpoint (Prometheus text format):

```bash
# Start the API
uvicorn api.main:app --port 8000

# Scrape metrics manually
curl http://localhost:8000/metrics | grep zorc_model_loaded
# zorc_model_loaded 1.0

# Key metrics:
#   zorc_model_loaded              — 1 when RF model is loaded
#   http_requests_total{handler}   — per-endpoint request counter
#   http_request_duration_seconds  — latency histogram
```

Run Prometheus locally with the provided config:

```bash
docker run -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus_config.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
# Dashboard: http://localhost:9090
```

## Environments

```bash
# Main pipeline (P1–P5, P7–P9, P10)
conda env create -f envs/zorc_pipeline.yml
conda activate zorc_pipeline

# BioEmu conformational sampling only (P6)
conda activate bioemu
```

## Running the pipeline

All scripts accept `--config config/zorc_config.yaml`. Run from project root:

```bash
python scripts/01_build_coregulon.py --config config/zorc_config.yaml
# ... through ...
python scripts/09f_final_model.py --config config/zorc_config.yaml
```

Or via Snakemake (P10):

```bash
snakemake --cores 4 --use-conda
```

## Prediction API

```bash
conda activate zorc_pipeline
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Docker:

```bash
docker pull moschoulab/zorc-predictor:1.0
docker run -p 8000:8000 moschoulab/zorc-predictor:1.0
```

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ --cov=api
```

## Reproducibility

Full parameter specification in `config/zorc_config.yaml`.  
Conda environments: `envs/zorc_pipeline.yml`, `envs/bioemu_ref.yml`.  
Data availability statement: raw data from Liu et al. 2024 (*Plant Cell*) and
Liu et al. 2023 (*EMBO J*).
