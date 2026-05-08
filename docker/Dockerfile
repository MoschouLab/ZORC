# ZORC P-body Prediction API — moschoulab/zorc-predictor:1.1
#
# Multi-stage build required for ViennaRNA (RNAfold):
#   - Not available in Debian apt (Bookworm or Trixie)
#   - PyPI wheel does not bundle the RNAfold binary on Linux x86_64
#   - conda-forge package unavailable outside a conda environment
#   Solution: compile ViennaRNA 2.6.4 from source in a builder stage,
#   copy only the RNAfold binary + libgsl27 runtime into the final image.
#   ca-certificates required in the builder (debian:bookworm-slim ships without it).

# ── Stage 1: compile ViennaRNA from source ────────────────────────────────────
FROM debian:bookworm-slim AS viennarna-builder

ARG VIENNARNA_VERSION=2.6.4

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        build-essential \
        pkg-config \
        libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://github.com/ViennaRNA/ViennaRNA/releases/download/v${VIENNARNA_VERSION}/ViennaRNA-${VIENNARNA_VERSION}.tar.gz" \
    && tar xzf ViennaRNA-${VIENNARNA_VERSION}.tar.gz \
    && cd ViennaRNA-${VIENNARNA_VERSION} \
    && ./configure \
        --prefix=/opt/viennarna \
        --without-python3 \
        --without-perl \
        --disable-openmp \
    && make -j$(nproc) \
    && make install

# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# libgsl27: runtime shared library that RNAfold links against
# curl: required by the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgsl27 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the RNAfold binary from the builder stage
COPY --from=viennarna-builder /opt/viennarna/bin/RNAfold /usr/local/bin/RNAfold

WORKDIR /app

# Python dependencies — separate layer so rebuilds skip this when unchanged
COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Application code (includes main.py, feature_compute.py, imputation_medians.json)
COPY api/ ./api/

# Model artifact — path must match _MODEL_PATH in api/main.py
#   _REPO_ROOT / "results" / "09f_rf_final_model_numt_clean.pkl"
COPY results/09f_rf_final_model_numt_clean.pkl ./results/09f_rf_final_model_numt_clean.pkl

# SQLite database for GET /lookup/{gene_id}
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
