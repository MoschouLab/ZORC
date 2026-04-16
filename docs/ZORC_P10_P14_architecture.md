# ZORC — P10–P14 Complete Architecture & Portfolio Roadmap

**Project:** Zip-code Of RNAs that Condense  
**Lab:** MoschouLab / IMBB-FORTH / ERC PLANTEX  
**Lead analyst:** José Moya-Cuevas  
**Document version:** April 2026  
**Purpose:** Engineering & deployment roadmap. Converts ZORC into a
full-stack Data Analyst / AI / ML Engineer portfolio project.

---

## ML Pipeline Status (P1–P9f complete)

| Metric | Value |
|---|---|
| Final model | RandomForestClassifier, 500 trees, Platt-calibrated |
| Features | 63 (41 RNA + 18 protein/IDR + 2 engineered) |
| Test AUROC | 0.7963 |
| Test AUPRC | 0.8431 |
| Test F1-macro | 0.7229 (F1-pos=0.7904, F1-neg=0.6554) |
| HC validation | 24/25 (96%) |
| Dataset | 1,510 genes, CD-HIT 40% anti-leakage split |

**Key scientific findings:**
- P-body mRNA condensation is driven primarily by RNA sequence features
  (di_CG, utr3_au_content, rrach_count, mrna_length)
- Protein conformational dynamics contribute real signal (rmsf_nterm50
  rank 11, rmsf_nterm_cterm_ratio rank 16)
- F1-neg ceiling ~0.66 is structural: ~22% of negative labels are
  biologically ambiguous (T-RIP label noise, not a model limitation)
- AUROC ceiling ~0.80 confirmed across RF + XGBoost + feature engineering
  + threshold tuning + calibration

---

## Portfolio Skills Map

| Skill | Phase | Demonstrated by |
|---|---|---|
| ML end-to-end (classification) | P1–P9f | Complete pipeline, SHAP, anti-leakage CV |
| scikit-learn, XGBoost, SHAP | P9–P9f | RF + XGB benchmarking, feature engineering |
| Pandas, NumPy | P1–P9f | All pipeline scripts |
| Snakemake (workflow) | P10 | Snakefile reproducing P1→P9f |
| GitHub + Zenodo (FAIR) | P10 | MoschouLab/ZORC public repo + DOI |
| SQL — SQLite + DuckDB | P11a | Relational DB + analytical queries |
| Jupyter notebooks | P11a | SQL exploration notebook |
| Streamlit | P11b | Gene Explorer lab tool |
| Plotly Dash | P11c | ML analytical dashboard |
| Tableau Public | P11d | 3 publication-quality dashboards |
| MLflow | P12a | All 6 model runs tracked retroactively |
| DVC | P12b | Feature matrix + model versioning |
| FastAPI | P12c | `/predict`, `/health`, `/model/info` |
| Docker + Docker Hub | P12d | `moschoulab/zorc-predictor:1.0` |
| GitHub Actions CI/CD | P12e | lint + test_pipeline + test_api |
| EvidentlyAI | P12f | Data drift + model performance reports |
| Prometheus | P12f | API latency + prediction count metrics |
| ChromaDB + LangChain | P14a | RAG on P-body literature |
| LangGraph | P14b | Multi-step literature + prediction agent |
| Anthropic API | P14a/b | Claude as LLM backend |

---

## P10 — Reproducibility: Snakemake + GitHub + Zenodo

**Goal:** Convert numbered scripts to a fully reproducible Snakemake
workflow. Publish to MoschouLab/ZORC on GitHub with Zenodo DOI.  
**Primary tool:** VS Code + Claude Code (direct filesystem access)  
**Estimated effort:** 2–3 days

### Why Snakemake

Snakemake solves reproducibility of the *analysis*: which scripts run
in which order, what files they consume and produce, and which steps
need rerunning when inputs change. Docker solves portability of the
*environment*. For a research pipeline, Snakemake is what reviewers
and the community need to reproduce results. Docker enters in P12 for
model serving.

Snakemake advantages specific to ZORC:
- Native conda integration: each rule specifies its environment
  (`zorc_pipeline.yml` or `bioemu_ref.yml`)
- Smart reruns: modifying P9d config automatically triggers P9f rerun
- Single entry point: `snakemake --cores 4 --use-conda`
- All parameters in `config/zorc_config.yaml` (FAIR compliant)

### Snakemake rule map

```
rule all
  ├── P1:  build_coregulon     [zorc_pipeline] → 01_zorc_coregulon_list.csv
  ├── P2:  map_isoforms        [zorc_pipeline] → 02_zorc_isoform_map.csv
  ├── P3:  fetch_sequences     [zorc_pipeline] → sequences/
  ├── P4:  rna_features        [zorc_pipeline] → 04_rna_features.csv
  ├── P5:  aiupred_idr         [bioemu]        → 05_aiupred_idr_summary.csv
  ├── P6:  bioemu_batch        [bioemu, GPU]   → bioemu/ + checkpoint.json
  ├── P6b: bioemu_tier1        [bioemu, GPU]   → checkpoint_tier1.json
  ├── P7:  protein_features    [zorc_pipeline] → 07_protein_features.csv
  ├── P9c: af2_recovery        [zorc_pipeline] → 07_protein_features.csv (patch)
  ├── P8:  feature_matrix      [zorc_pipeline] → 08_zorc_feature_matrix.csv
  ├── P9:  random_forest       [zorc_pipeline] → 09_zorc_rf_model.pkl
  ├── P9b: xgboost             [zorc_pipeline] → 09b_zorc_xgb_model.json
  ├── P9d: feature_engineering [zorc_pipeline] → 08_zorc_feature_matrix_eng.csv
  ├── P9e: threshold_tuning    [zorc_pipeline] → 09e_threshold_curve.csv
  └── P9f: final_model         [zorc_pipeline] → 09f_rf_final_model.pkl ← FINAL
```

GPU-intensive rules (P6, P6b) are flagged as `localrules` with
documentation that they require manual execution with checkpointing.

### GitHub repository structure (final)

```
MoschouLab/ZORC/
├── README.md                         ← installation, usage, results, citation
├── Snakefile                         ← main workflow
├── config/
│   └── zorc_config.yaml
├── scripts/                          ← P1–P9f pipeline scripts
├── envs/
│   ├── zorc_pipeline.yml
│   └── bioemu_ref.yml
├── docs/
│   ├── ZORC_continuation_prompt_v2.md
│   ├── ZORC_methodological_decisions_v2.md
│   └── ZORC_P10_P14_architecture.md  ← THIS FILE
├── notebooks/
│   ├── 01_sql_exploration.ipynb      ← P11a DuckDB + SQL
│   ├── 02_feature_analysis.ipynb     ← P11a EDA
│   └── 03_model_comparison.ipynb     ← P11a model runs comparison
├── dashboard/
│   ├── streamlit_app.py              ← P11b Gene Explorer
│   └── dash_app.py                   ← P11c ML Analytical Dashboard
├── tableau/
│   └── README_tableau.md             ← P11d links + export instructions
├── api/
│   ├── main.py                       ← P12c FastAPI predictor
│   ├── feature_compute.py            ← RNA feature computation on-the-fly
│   └── requirements_api.txt
├── docker/
│   └── Dockerfile                    ← P12d
├── monitoring/
│   ├── evidently_reports/            ← P12f data drift reports
│   └── prometheus_config.yml         ← P12f API metrics
├── mlruns/                           ← P12a MLflow (summary tracked, runs gitignored)
├── dvc/                              ← P12b DVC config
├── agent/
│   ├── literature_agent.py           ← P14a RAG agent
│   ├── zorc_agent.py                 ← P14b LangGraph agent
│   └── chroma_db/                    ← P14a vector store (gitignored)
├── tests/
│   ├── test_pipeline.py              ← P12e CI tests
│   └── test_api.py                   ← P12e API tests
├── .github/
│   └── workflows/
│       └── ci.yml                    ← P12e GitHub Actions
└── data/
    ├── raw/                          ← gitignored
    └── processed/                   ← key CSVs tracked, binaries gitignored
```

### FAIR publication checklist

- [ ] Snakefile with all rules and conda directives
- [ ] README.md: installation, usage, results summary, citation
- [ ] config/zorc_config.yaml as single parameter entry point
- [ ] envs/*.yml for both conda environments
- [ ] Data availability statement
- [ ] GitHub release → Zenodo DOI (target: before manuscript submission)

---

## P11 — Data Analytics Layer

**Goal:** Demonstrate Data Analyst skills through SQL-based exploration,
interactive dashboards, and publication-quality visualizations.  
**Showcases:** SQL (SQLite + DuckDB), Jupyter, Streamlit, Plotly Dash,
Tableau Public  
**Estimated effort:** 4–5 days

### P11a — Database construction + SQL notebooks

**Script:** `scripts/10a_build_database.py`

Two complementary tools built from ZORC outputs:

#### SQLite relational database (`data/zorc_database.db`)

Three normalized tables with foreign key relationships:

```sql
CREATE TABLE genes (
    gene_id       TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    class         INTEGER NOT NULL,   -- 1=P-body enriched, 0=excluded
    condition     TEXT,               -- NS, HS, depleted
    bioemu_tier   INTEGER,            -- 1, 2, 3
    split         TEXT                -- train / val / test
);

CREATE TABLE features (
    gene_id                  TEXT PRIMARY KEY
                             REFERENCES genes(gene_id),
    -- RNA features (top predictors)
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
    -- Protein / IDR features
    idr_percent              REAL,
    rmsf_mean                REAL,
    rmsf_nterm50             REAL,
    rmsf_cterm50             REAL,
    rmsf_std                 REAL,
    rg_mean                  REAL,
    packing_density          REAL,
    n_residues               REAL,
    -- Engineered features
    rmsf_nterm_cterm_ratio   REAL,
    utr3_au_x_length         REAL
);

CREATE TABLE predictions (
    gene_id          TEXT PRIMARY KEY
                     REFERENCES genes(gene_id),
    prob_pos         REAL,    -- P(P-body enriched), final model P9f
    pred             INTEGER, -- 0/1 at threshold=0.50
    -- Top 5 SHAP contributions for this gene
    shap_mrna_length        REAL,
    shap_di_CG              REAL,
    shap_utr3_au_content    REAL,
    shap_rrach_count        REAL,
    shap_rmsf_nterm50       REAL
);
```

#### DuckDB analytical queries (`notebooks/01_sql_exploration.ipynb`)

DuckDB queries the SQLite database and reads CSVs directly.
Demonstrates advanced SQL: window functions, CTEs, aggregations.

```sql
-- Q1: Top 20 P-body candidates by model probability
SELECT g.gene_id, p.prob_pos,
       f.rrach_count, f.idr_percent, f.utr3_au_content,
       f.rmsf_nterm_cterm_ratio
FROM predictions p
JOIN genes g USING (gene_id)
JOIN features f USING (gene_id)
WHERE g.class = 1
ORDER BY p.prob_pos DESC
LIMIT 20;

-- Q2: Feature distributions by class using window functions
SELECT
    gene_id, class,
    utr3_au_content,
    NTILE(4) OVER (ORDER BY utr3_au_content) AS au_quartile,
    AVG(prob_pos) OVER (
        PARTITION BY NTILE(4) OVER (ORDER BY utr3_au_content)
    ) AS mean_prob_by_quartile
FROM features f
JOIN genes g USING (gene_id)
JOIN predictions p USING (gene_id);

-- Q3: Proteins with high N/C flexibility asymmetry AND high m6A density
-- (biologically: exposed N-termini + multiple m6A sites = strong P-body signal)
SELECT g.gene_id, f.rmsf_nterm_cterm_ratio,
       f.rrach_per_kb, f.idr_percent, p.prob_pos
FROM features f
JOIN genes g USING (gene_id)
JOIN predictions p USING (gene_id)
WHERE f.rmsf_nterm_cterm_ratio > 2.0
  AND f.rrach_per_kb > 30.0
ORDER BY p.prob_pos DESC;

-- Q4: Xenium probe panel candidates
-- High-confidence positives for spatial transcriptomics validation
WITH ranked AS (
    SELECT g.gene_id, p.prob_pos,
           f.utr3_au_content, f.rrach_count, f.idr_percent,
           ROW_NUMBER() OVER (ORDER BY p.prob_pos DESC) AS rank
    FROM predictions p
    JOIN genes g USING (gene_id)
    JOIN features f USING (gene_id)
    WHERE p.prob_pos > 0.85
      AND f.rrach_count > 15
)
SELECT * FROM ranked LIMIT 50;

-- Q5: Label noise analysis — ambiguous negatives
-- Genes labeled negative but predicted positive with high confidence
SELECT g.gene_id, p.prob_pos,
       f.utr3_au_content, f.rrach_count,
       g.condition
FROM predictions p
JOIN genes g USING (gene_id)
JOIN features f USING (gene_id)
WHERE g.class = 0
  AND p.prob_pos BETWEEN 0.35 AND 0.65
ORDER BY p.prob_pos DESC;

-- Q6: Model performance by BioEmu tier (structured vs disordered proteins)
SELECT g.bioemu_tier,
       COUNT(*)                                          AS n_genes,
       AVG(CASE WHEN p.pred = g.class THEN 1.0 ELSE 0.0 END) AS accuracy,
       AVG(p.prob_pos)                                  AS mean_prob,
       AVG(f.idr_percent)                               AS mean_idr
FROM predictions p
JOIN genes g USING (gene_id)
JOIN features f USING (gene_id)
WHERE g.split = 'test'
GROUP BY g.bioemu_tier
ORDER BY g.bioemu_tier;
```

Additional notebook: `notebooks/02_feature_analysis.ipynb`
- Correlation matrix of all 63 features
- Feature distributions by class (violin plots)
- RNA vs protein feature contribution analysis
- SHAP beeswarm global summary

### P11b — Streamlit: Gene Explorer (lab tool)

**File:** `dashboard/streamlit_app.py`  
**Launch:** `streamlit run dashboard/streamlit_app.py`  
**Target user:** Biologist searching for a specific gene  
**Rationale:** Streamlit excels at rapid prototyping and simple
interactive tools — minimal code, immediate result, deployable as
an internal lab resource.

Three pages:

**Page 1 — Gene Search:** Input AGI code → shows:
- Probability gauge 0–1 with confidence label (High/Medium/Low)
- SHAP waterfall plot (top 10 feature contributions for that gene)
- Feature values vs population violin plots (gene highlighted)
- External links: TAIR, UniProt, AlphaFold2 entry

**Page 2 — Xenium Probe Candidates:** Results of SQL Q4:
- Table of top 50 candidates by P(pos)
- Downloadable CSV for probe design submission
- Feature profiles comparison

**Page 3 — Model Card:** Static performance metrics, confusion matrix,
HC validation results, dataset description. Reproducible documentation
following ML model card best practices.

### P11c — Plotly Dash: Analytical Dashboard (ML explorer)

**File:** `dashboard/dash_app.py`  
**Launch:** `python dashboard/dash_app.py`  
**Target user:** Data Scientist exploring model behaviour  
**Rationale:** Plotly Dash supports complex reactive callbacks and
multi-panel linked charts — better suited than Streamlit for
analytical exploration of model internals.

Four linked panels with reactive Dash callbacks:

**Panel 1 — Feature Space UMAP:** 2D UMAP of 63 features.
Color dropdown: class / BioEmu tier / probability / prediction error.
Clicking a point highlights that gene across all other panels.

**Panel 2 — Model Performance Explorer:**
ROC + PR curves with interactive threshold slider (0.30–0.70).
Moving the slider updates confusion matrix + F1 metrics in real time.
Toggle: RF baseline P9 vs XGB P9b vs RF final P9f overlay.

**Panel 3 — Feature Importance Deep-dive:**
SHAP beeswarm with category filter dropdown:
(All / RNA / Protein-static / Protein-dynamic / Engineered).
Linked scatter: selected feature value vs P(pos), colored by class.

**Panel 4 — Experiment Comparison:**
Table of all model runs (P9, P9b, P9c, P9d, P9e, P9f) with sortable
metrics. Selecting a run updates the SHAP panel with that run's
importance rankings. Connects to MLflow logged artifacts via
`mlflow.search_runs()`.

### P11d — Tableau Public: Publication Visualizations

**Target:** Three permanent public dashboards on tableau.public.com  
**Workflow:** Export CSVs → Tableau Desktop (Windows/Mac) → Publish  
**Result:** Public URLs embeddable in README.md and shareable in CV

Dashboard 1 — **ZORC Feature Importance:**
SHAP importance bar chart for final model + scatter plots of top 5
features colored by class. Shows RNA dominance vs protein contribution.

Dashboard 2 — **P-body Probability Landscape:**
2D scatter (utr3_au_content vs rrach_count) with probability color
scale and class shape encoding. Interactive tooltip shows gene_id.

Dashboard 3 — **Pipeline Optimization History:**
Line chart of Val AUROC / Test AUROC / F1-macro across all model
iterations (P9 → P9b pre-AF2 → P9b post-AF2 → P9d → P9f).
Shows the impact of each intervention quantitatively.

---

## P12 — ML Engineering Layer

**Goal:** Demonstrate ML Engineer skills through experiment tracking,
model versioning, serving, containerization, monitoring, and CI/CD.  
**Showcases:** MLflow, DVC, FastAPI, Docker, GitHub Actions,
EvidentlyAI, Prometheus  
**Estimated effort:** 4–5 days

### P12a — MLflow: Experiment Tracking

**Script:** `scripts/10b_mlflow_retroactive.py`  
**Launch UI:** `mlflow ui --port 5000` → http://localhost:5000

Retroactive logging of all 6 model runs:

```
Experiment: zorc_pbody_prediction/
  ├── run: rf_baseline_p9
  │     params: n_estimators=500, imputation=median,
  │             class_weight=balanced, features=56
  │     metrics: val_auroc=0.7990, test_auroc=0.7974,
  │              test_f1_macro=0.7382, hc_accuracy=0.96
  │     artifacts: rf_model.pkl, shap_importance.csv
  │     tags: phase=P9, dataset=pre_af2_recovery
  │
  ├── run: xgb_benchmark_p9b_pre_af2
  │     params: max_depth=6, lr=0.05, early_stopping=30,
  │             features=60, nan_handling=native
  │     metrics: val_auroc=0.7993, test_auroc=0.7835,
  │              best_round=36
  │     tags: phase=P9b, dataset=pre_af2_recovery
  │
  ├── run: xgb_benchmark_p9b_post_af2
  │     params: max_depth=6, lr=0.05, early_stopping=30,
  │             features=60
  │     metrics: val_auroc=0.8001, test_auroc=0.7879,
  │              best_round=75
  │     tags: phase=P9b, dataset=post_af2_recovery
  │
  ├── run: rf_feature_engineering_p9d
  │     params: engineered=[rmsf_nterm_cterm_ratio, utr3_au_x_length,
  │                          packing_x_idr, rrach_per_cds_kb],
  │             features=63
  │     metrics: val_auroc=0.7969, test_auroc=0.7963,
  │              test_f1_macro=0.7229  ← BEST AUROC+F1
  │     tags: phase=P9d, dataset=post_af2_recovery
  │
  ├── run: rf_threshold_tuning_p9e
  │     params: threshold_scan=0.30-0.70, optimal_threshold=0.50
  │     metrics: confirmed threshold=0.50 optimal
  │     tags: phase=P9e
  │
  └── run: rf_final_calibrated_p9f
        params: calibration=platt_sigmoid_cv5,
                dropped=[packing_x_idr, rrach_per_cds_kb],
                features=61
        metrics: val_auroc=0.7979, test_auroc=0.7862,
                 brier_uncal=0.1801, brier_cal=0.1776
        tags: phase=P9f, model=final_calibrated
```

The MLflow UI table is directly demostrable — shows full experiment
history with sortable metrics columns.

### P12b — DVC: Data & Model Versioning

**Config:** `dvc/` directory  
**Purpose:** Version large files that cannot go to git

DVC tracks:
```
data/processed/08_zorc_feature_matrix.csv    (>10MB)
data/processed/08_zorc_feature_matrix_eng.csv
results/09f_rf_final_model.pkl
results/09f_rf_base_model.pkl
results/09b_zorc_xgb_model.json
```

DVC remote: local storage initially (`/mnt/zorc_dvc_remote/`), 
upgradeable to S3/GCS when cloud access available.

Usage:
```bash
dvc pull          # download latest model artifacts
dvc push          # push new artifacts after training
dvc repro         # rerun pipeline stages if inputs changed
```

Complements MLflow: MLflow tracks metrics + params, DVC tracks
the actual data and model files.

### P12c — FastAPI: Prediction Service

**File:** `api/main.py`  
**Swagger UI:** http://localhost:8000/docs (auto-generated)

Three endpoints:

```
POST /predict
  Input:  {
    "gene_id": "AT1G01470",
    "mrna_seq": "ATCGATCG...",      # required
    "protein_seq": "MASTKL..."       # optional, falls back to median
  }
  Output: {
    "gene_id": "AT1G01470",
    "prob_p_body": 0.847,
    "prediction": "enriched",        # enriched / not_enriched
    "confidence": "high",            # high >0.75 / medium 0.55-0.75 / low
    "top_features": {
      "utr3_au_content": {"value": 0.78, "shap": 0.031},
      "rrach_count":     {"value": 23,   "shap": 0.027},
      "mrna_length":     {"value": 1842, "shap": 0.024}
    },
    "model_version": "1.0.0",
    "disclaimer": "Prediction based on T-RIP training data..."
  }

GET /predict/batch
  Input:  list of gene objects
  Output: list of prediction objects

GET /model/info
  Output: {
    "model": "RandomForestClassifier",
    "version": "1.0.0",
    "phase": "P9f",
    "test_auroc": 0.7963,
    "test_auprc": 0.8431,
    "n_features": 63,
    "training_date": "2026-04-15"
  }

GET /health
  Output: {"status": "ok", "model_loaded": true}
```

The API computes RNA features on-the-fly from `mrna_seq` using the
same functions as P4 (RNAfold MFE, dinucleotides, UTR lengths, m6A
motifs). Protein features are optional — if absent, median imputation
is applied (documented in response).

### P12d — Docker: Containerization

**File:** `docker/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
COPY api/ ./api/
COPY results/09f_rf_final_model.pkl ./models/
COPY config/zorc_config.yaml .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", \
     "--port", "8000", "--workers", "2"]
```

Build and publish:
```bash
docker build -t moschoulab/zorc-predictor:1.0 .
docker push moschoulab/zorc-predictor:1.0
# Anyone can now run:
docker pull moschoulab/zorc-predictor:1.0
docker run -p 8000:8000 moschoulab/zorc-predictor:1.0
```

### P12e — GitHub Actions: CI/CD

**File:** `.github/workflows/ci.yml`

Three jobs triggered on every push to `main` and on PRs:

```yaml
jobs:
  lint:
    # flake8 on all scripts + api + dashboard
    # black --check formatting
    # isort import ordering

  test_pipeline:
    # Run P1→P4→P8→P9 on synthetic mini-dataset (50 genes)
    # Validates pipeline runs end-to-end without errors
    # Checks output file schemas match expectations
    # Runtime: ~2 min

  test_api:
    # Start FastAPI with test model
    # POST /predict with synthetic sequence
    # Verify response format and probability in [0,1]
    # GET /health → 200 OK
    # Runtime: ~1 min
```

CI status badge in README:
`![CI](https://github.com/MoschouLab/ZORC/actions/workflows/ci.yml/badge.svg)`

### P12f — Monitoring: EvidentlyAI + Prometheus

**Purpose:** Production model monitoring — demonstrates MLOps maturity.

#### EvidentlyAI data drift reports

**Script:** `scripts/10c_evidently_report.py`

Generates HTML reports comparing:
- Training set feature distribution vs new prediction requests
- Model performance report (if ground truth becomes available)
- Data quality report (missing values, outliers)

```python
# Example: detect if new genes being predicted differ
# significantly from training distribution
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=X_train_df, current_data=X_new_df)
report.save_html("monitoring/evidently_reports/drift_report.html")
```

Reports stored in `monitoring/evidently_reports/` and linked from README.

#### Prometheus API metrics

Added to FastAPI with 5 lines:
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
# Metrics available at GET /metrics
# Tracks: request_count, request_latency, prediction_distribution
```

`monitoring/prometheus_config.yml` for local Prometheus scraping.
Demonstrates observability instrumentation — directly relevant to
Datadope's observability domain.

---

## P13 — Xenium 10X Spatial Transcriptomics Validation

**Goal:** Experimental validation of ZORC predictions using spatial
transcriptomics.  
**Timeline:** Probe design 1 week → facility submission → turnaround 4–6 weeks

### Candidate selection (SQL query P11a-Q4)

Top 50 genes with P(pos) > 0.85 AND rrach_count > 15.
Cross-reference with HC validation set (25 genes).
Final probe panel: ~30 genes prioritised for Xenium design.

### Experimental setup

- *Arabidopsis thaliana* root cross-sections
- DCP1-GFP marker line (existing in lab)
- Conditions: Heat stress (37°C, 1h) vs NS
- Platform: Xenium In Situ (10X Genomics), external facility

### Success criteria

ZORC-predicted P-body mRNAs (P(pos) > 0.80) show ≥2× higher
co-localisation with DCP1-GFP foci than predicted-negative mRNAs
(P(pos) < 0.30) under heat stress.

---

## P14 — Generative AI Layer: ZORC Literature Agent

**Goal:** Demonstrate LLM integration, RAG, and agent design using
a scientifically meaningful use case.  
**Showcases:** ChromaDB, LangChain, LangGraph, Anthropic API  
**Estimated effort:** 2–3 days

### P14a — RAG: P-body Literature Knowledge Base

**Script:** `agent/literature_agent.py`

Build a semantic search index over P-body / biomolecular condensate
literature, queryable by natural language.

**Step 1 — Corpus collection:**
Download abstracts + key sentences from ~500 papers on P-bodies,
stress granules, and mRNA condensation from PubMed (using Entrez API).
Target papers: all papers citing Liu et al. 2024 (T-RIP) +
Liu et al. 2023 (APEAL) + key P-body reviews.

**Step 2 — Vector index (ChromaDB):**
```python
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Embed paper chunks with scientific embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/allenai-specter"
    # SPECTER: scientific paper embeddings, better than generic models
)
vectorstore = Chroma(
    persist_directory="agent/chroma_db",
    embedding_function=embeddings
)
# Index: ~500 papers, ~2000 chunks of 500 tokens each
```

**Step 3 — RAG queries:**
```python
from langchain.chains import RetrievalQA
from anthropic import Anthropic

qa_chain = RetrievalQA.from_chain_type(
    llm=Claude_via_Anthropic_API,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Example queries:
qa_chain("Which plant mRNAs have been experimentally validated in P-bodies?")
qa_chain("What is the evidence for m6A modification promoting P-body condensation?")
qa_chain("How does heat stress affect DCP1 foci formation in Arabidopsis?")
```

### P14b — LangGraph: ZORC Prediction + Literature Agent

**Script:** `agent/zorc_agent.py`

A multi-step agent that given a gene AGI code:
1. Calls the ZORC FastAPI `/predict` endpoint
2. Retrieves relevant literature from ChromaDB
3. Generates an integrated report: "Gene X has P(P-body)=0.84.
   This is supported by evidence that [literature context].
   Key features driving this prediction: [SHAP explanation]."

```python
from langgraph.graph import StateGraph

# Agent workflow graph:
# START → get_prediction → retrieve_literature → generate_report → END
#                  ↓ (if prob > 0.8)
#          retrieve_experimental_evidence

workflow = StateGraph(AgentState)
workflow.add_node("get_prediction", call_zorc_api)
workflow.add_node("retrieve_literature", query_chromadb)
workflow.add_node("generate_report", call_claude_api)
workflow.add_conditional_edges(
    "get_prediction",
    lambda s: "high_confidence" if s["prob"] > 0.8 else "standard",
    {"high_confidence": "retrieve_literature",
     "standard": "generate_report"}
)
```

**Demo:** A Jupyter notebook `notebooks/03_zorc_agent_demo.ipynb`
showing the agent answering questions about specific genes, with
sources cited from the literature.

### Why this is scientifically non-trivial

The combination of ZORC predictions + literature RAG creates a tool
that can answer questions like "among all Arabidopsis mRNAs predicted
to condense in P-bodies, which have experimental support from
independent studies?" — a query that would otherwise require manual
literature review. This is directly useful for manuscript preparation
and for prioritising Xenium probe candidates.

---

## Implementation Order & Timeline

| Phase | Content | Tool | Days |
|---|---|---|---|
| P10 | Snakemake + GitHub + Zenodo | Claude Code | 2–3 |
| P11a | SQLite + DuckDB + notebooks | Claude Code | 1–2 |
| P11b | Streamlit Gene Explorer | Claude Code | 1–2 |
| P11c | Plotly Dash ML dashboard | Claude Code | 2 |
| P11d | Tableau Public (3 dashboards) | Tableau Desktop | 1 |
| P12a | MLflow retroactive | Claude Code | 1 |
| P12b | DVC versioning | Claude Code | 0.5 |
| P12c | FastAPI predictor | Claude Code | 1–2 |
| P12d | Docker + Docker Hub | Claude Code | 1 |
| P12e | GitHub Actions CI | Claude Code | 1 |
| P12f | EvidentlyAI + Prometheus | Claude Code | 1 |
| P14a | ChromaDB + LangChain RAG | Claude Code | 1–2 |
| P14b | LangGraph agent | Claude Code | 1 |
| P13 | Xenium probe design + submission | Manual | 1 week |

**Total computational work:** ~3 weeks at normal pace  
**Result:** Complete Data Analyst + AI/ML Engineer portfolio project

---

## Working Mode for P10–P14

**Claude.ai (this chat):**
Strategic decisions, architecture review, interpretation of results,
manuscript methods, planning what to do and why.

**Claude Code (VS Code, ~/Documents/ZORC/):**
All code writing, editing, execution, debugging, git operations.
Claude Code has direct filesystem access — no copy/paste needed.

**Session startup protocol for Claude Code:**
```bash
cd ~/Documents/ZORC
claude
```
Always share at session start:
- `docs/ZORC_continuation_prompt_v2.md`
- `docs/ZORC_methodological_decisions_v2.md`
- `docs/ZORC_P10_P14_architecture.md`

Claude Code does NOT have access to claude.ai conversation history.

---

*Document generated: April 2026*  
*Status: P1–P9f complete. P10 begins after VS Code + Claude Code installation.*
