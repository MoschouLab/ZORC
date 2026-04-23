# ── Stage 1: compile ViennaRNA from source ────────────────────────────────────
FROM debian:bookworm-slim AS viennarna-builder

ARG VIENNARNA_VERSION=2.6.4

RUN apt-get update && apt-get install -y --no-install-recommends \
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
#   _REPO_ROOT / "results" / "09d_rf_eng_model.pkl"  →  /app/results/09d_rf_eng_model.pkl
COPY results/09d_rf_eng_model.pkl ./results/09d_rf_eng_model.pkl

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
