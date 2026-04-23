FROM python:3.11-slim-bookworm

# ViennaRNA (RNAfold via subprocess in feature_compute.py)
# curl required by the HEALTHCHECK command
# vienna-rna is the correct package name in Debian Bookworm (12)
RUN apt-get update && apt-get install -y --no-install-recommends \
        vienna-rna \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies — separate layer so rebuilds are fast
COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Application code (includes main.py, feature_compute.py, imputation_medians.json)
COPY api/ ./api/

# Model artifact — path must match _MODEL_PATH in api/main.py
#   _REPO_ROOT / "results" / "09d_rf_eng_model.pkl"  →  /app/results/09d_rf_eng_model.pkl
COPY results/09d_rf_eng_model.pkl ./results/09d_rf_eng_model.pkl

# SQLite database for GET /lookup/{gene_id}
#   _DB_PATH → /app/data/zorc_database.db
COPY data/zorc_database.db ./data/zorc_database.db

# Config for documentation / future use
COPY config/zorc_config.yaml ./config/zorc_config.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
