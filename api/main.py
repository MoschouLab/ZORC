"""
ZORC FastAPI Prediction Service — Phase 12c
============================================
Serves the P9d RandomForest model (results/09d_rf_eng_model.pkl) via HTTP.

Endpoints
---------
POST /predict           Predict P-body enrichment for one mRNA sequence
POST /predict/batch     Batch version (list of sequences)
GET  /model/info        Model metadata and training metrics
GET  /health            Liveness probe

Usage
-----
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    uvicorn api.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
"""

import json
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.feature_compute import compute_features, PROTEIN_FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).parent
_REPO_ROOT  = _HERE.parent
_MODEL_PATH = _REPO_ROOT / "results" / "09d_rf_eng_model.pkl"
_MEDIANS_PATH = _HERE / "imputation_medians.json"

# ── Global model state ────────────────────────────────────────────────────────

_state: dict = {}


def _load_state() -> None:
    # Model
    with open(_MODEL_PATH, "rb") as f:
        _state["model"] = pickle.load(f)

    # Feature order + medians
    with open(_MEDIANS_PATH) as f:
        med_data = json.load(f)
    _state["feature_order"] = med_data["feature_order"]
    _state["medians"]       = med_data["medians"]   # dict {feature: value}

    # SHAP explainer (TreeExplainer is fast for RF)
    _state["explainer"] = shap.TreeExplainer(_state["model"])

    n = len(_state["feature_order"])
    print(f"[ZORC API] Model loaded: {_MODEL_PATH.name}  |  {n} features")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ZORC P-body Prediction API",
    description=(
        "Predicts P-body mRNA enrichment probability for *Arabidopsis thaliana* "
        "transcripts using a RandomForestClassifier trained on T-RIP + APEAL "
        "data (P1–P9d)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    gene_id:     str           = Field("", description="AGI code (optional, for tracking)")
    mrna_seq:    str           = Field(..., min_length=50,
                                       description="Full spliced mRNA sequence (DNA or RNA notation)")
    protein_seq: Optional[str] = Field(None, description=(
        "Translated protein sequence. Used to estimate CDS length "
        "for UTR features. If absent, median imputation is applied."))
    use_rnafold: bool          = Field(True, description=(
        "Run RNAfold for MFE/structure features. Set false for speed; "
        "structure features will be imputed from training median."))

    model_config = {"json_schema_extra": {"examples": [{
        "gene_id":  "AT1G01470",
        "mrna_seq": "ATGAAACCCGGGTTTTAA",
        "protein_seq": None,
        "use_rnafold": False,
    }]}}


class FeatureContribution(BaseModel):
    value: float
    shap:  float


class PredictResponse(BaseModel):
    gene_id:       str
    prob_p_body:   float = Field(..., description="P(P-body enriched) ∈ [0, 1]")
    prediction:    str   = Field(..., description="'enriched' or 'not_enriched'")
    confidence:    str   = Field(..., description="'high' (≥0.75) / 'medium' (0.55–0.75) / 'low' (<0.55)")
    top_features:  dict[str, FeatureContribution]
    model_version: str
    imputed_features: list[str] = Field(...,
        description="Features that were median-imputed (not computed from input sequence)")
    disclaimer:    str


class BatchPredictRequest(BaseModel):
    sequences: list[PredictRequest] = Field(..., max_length=500)


class BatchPredictResponse(BaseModel):
    results:    list[PredictResponse]
    n_success:  int
    n_error:    int
    errors:     list[dict]


class ModelInfoResponse(BaseModel):
    model:          str
    version:        str
    phase:          str
    n_estimators:   int
    n_features:     int
    feature_matrix: str
    test_auroc:     float
    test_auprc:     float
    test_f1_macro:  float
    hc_accuracy:    float
    training_date:  str
    note:           str


# ── Core prediction logic ─────────────────────────────────────────────────────

def _predict_one(req: PredictRequest) -> PredictResponse:
    model        = _state["model"]
    feature_order = _state["feature_order"]
    medians      = _state["medians"]
    explainer    = _state["explainer"]

    # 1. Compute features
    feats = compute_features(
        mrna_seq=req.mrna_seq,
        protein_seq=req.protein_seq,
        use_rnafold=req.use_rnafold,
    )

    # 2. Track which features are imputed
    imputed = [f for f in feature_order if feats.get(f) is None]

    # 3. Build feature vector in training order, imputing None → median
    x = np.array(
        [feats[f] if feats.get(f) is not None else medians[f]
         for f in feature_order],
        dtype=np.float32,
    ).reshape(1, -1)

    # 4. Predict
    prob = float(model.predict_proba(x)[0, 1])

    # 5. SHAP (top-5 by absolute contribution)
    shap_vals = explainer.shap_values(x)
    # shap_values shape varies by SHAP version:
    #   new API: ndarray (n_samples, n_features, n_classes) → take [:, :, 1]
    #   old API: list[ndarray per class]                    → take [1][0]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]     # shape (63,)
    elif isinstance(shap_vals, list):
        sv = shap_vals[1][0]        # shape (63,)
    else:
        sv = shap_vals[0]           # fallback

    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    top_features = {
        feature_order[i]: FeatureContribution(
            value=round(float(x[0, i]), 6),
            shap=round(float(sv[i]), 6),
        )
        for i in top_idx
    }

    # 6. Labels
    prediction = "enriched" if prob >= 0.50 else "not_enriched"
    if prob >= 0.75:
        confidence = "high"
    elif prob >= 0.55:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictResponse(
        gene_id=req.gene_id,
        prob_p_body=round(prob, 6),
        prediction=prediction,
        confidence=confidence,
        top_features=top_features,
        model_version="1.0.0-p9d",
        imputed_features=imputed,
        disclaimer=(
            "Prediction based on T-RIP (Liu et al. 2024 Plant Cell) training "
            "data. Protein conformational features are median-imputed when no "
            "protein sequence is provided. Validated on 24/25 lab-curated "
            "high-confidence P-body genes (Arabidopsis thaliana)."
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict P-body enrichment probability for a single mRNA sequence.

    RNA features are computed on-the-fly from `mrna_seq`.
    Protein/conformational features are median-imputed from the training set
    unless a protein_seq is provided (used only for CDS length estimation).
    """
    try:
        return _predict_one(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """
    Predict P-body enrichment for up to 500 sequences in one call.
    Failed predictions are collected in `errors`; successful ones in `results`.
    """
    results, errors = [], []
    for item in req.sequences:
        try:
            results.append(_predict_one(item))
        except Exception as exc:
            errors.append({"gene_id": item.gene_id, "error": str(exc)})
    return BatchPredictResponse(
        results=results,
        n_success=len(results),
        n_error=len(errors),
        errors=errors,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return metadata and training metrics for the loaded model."""
    model = _state["model"]
    return ModelInfoResponse(
        model="RandomForestClassifier",
        version="1.0.0-p9d",
        phase="P9d (feature-engineered RF, pre-Platt calibration)",
        n_estimators=model.n_estimators,
        n_features=model.n_features_in_,
        feature_matrix="data/processed/08_zorc_feature_matrix_eng.csv",
        test_auroc=0.7963,
        test_auprc=0.8431,
        test_f1_macro=0.7229,
        hc_accuracy=0.96,
        training_date="2026-04-22",
        note=(
            "63 features: 41 RNA + 18 protein/IDR + 4 engineered. "
            "Protein features are median-imputed from the training set when "
            "not available. "
            "For Platt-calibrated probabilities use results/09f_rf_final_model.pkl."
        ),
    )


@app.get("/health", tags=["Ops"])
def health():
    """Liveness probe. Returns 200 when the model is loaded."""
    return {"status": "ok", "model_loaded": "model" in _state}
