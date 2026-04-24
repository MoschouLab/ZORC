"""
tests/test_api.py — Integration tests for the ZORC FastAPI prediction service.

The model, SHAP explainer, and feature medians are mocked so these tests run
without requiring the trained model artefacts or ViennaRNA in the CI environment.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Feature order (63 features from P9d model) ───────────────────────────────

FEATURE_ORDER = [
    "mrna_length", "fA", "fU", "fG", "fC", "au_content", "gc_content",
    "di_AA", "di_AU", "di_AG", "di_AC", "di_UA", "di_UU", "di_UG", "di_UC",
    "di_GA", "di_GU", "di_GG", "di_GC", "di_CA", "di_CU", "di_CG", "di_CC",
    "utr5_length", "utr3_length", "cds_length", "utr5_fraction", "utr3_fraction",
    "cds_fraction", "utr3_au_content", "rrach_count", "rrach_per_kb",
    "aaach_count", "aaach_per_kb", "mfe", "mfe_per_nt", "frac_paired",
    "n_stemloops", "n_stemloops_per_kb", "long_3utr", "ptc_proxy",
    "rmsf_mean", "rmsf_std", "rmsf_max", "rmsf_nterm50", "rmsf_cterm50",
    "rg_mean", "rg_std", "rg_cv", "contact_density", "packing_density",
    "sasa_per_residue", "pass_rate", "n_residues", "idr_percent",
    "mean_disorder", "max_disorder_window", "n_idr_regions", "longest_idr_region",
    "rmsf_nterm_cterm_ratio", "packing_x_idr", "rrach_per_cds_kb",
    "utr3_au_x_length",
]

N_FEATURES = len(FEATURE_ORDER)

# ── Synthetic mRNA (200 nt) ───────────────────────────────────────────────────

_MRNA = (
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
)


# ── Mock state builder ────────────────────────────────────────────────────────

def _build_mock_state():
    """Return a dict that mimics api.main._state with mocked ML objects."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    mock_model.n_estimators = 500
    mock_model.n_features_in_ = N_FEATURES

    # SHAP: old-API list format [class0_vals, class1_vals]
    shap_vals = np.zeros((1, N_FEATURES))
    shap_vals[0, 0] = 0.05  # mrna_length has the largest contribution

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [shap_vals * -1, shap_vals]

    return {
        "model":         mock_model,
        "feature_order": FEATURE_ORDER,
        "medians":       {f: 0.0 for f in FEATURE_ORDER},
        "explainer":     mock_explainer,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient with mocked model loading."""
    fake_state = _build_mock_state()

    def _fake_load():
        import api.main as m
        m._state.update(fake_state)

    with patch("api.main._load_state", side_effect=_fake_load):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ── POST /predict ─────────────────────────────────────────────────────────────

def test_predict_returns_200(client):
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_prob_in_unit_interval(client):
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    assert 0.0 <= body["prob_p_body"] <= 1.0


def test_predict_response_fields(client):
    payload = {"gene_id": "AT1G01470", "mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    for field in ("gene_id", "prob_p_body", "prediction",
                  "confidence", "top_features", "model_version",
                  "imputed_features", "disclaimer"):
        assert field in body, f"missing field: {field}"


def test_predict_prediction_label(client):
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    assert body["prediction"] in ("enriched", "not_enriched")


def test_predict_confidence_label(client):
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    assert body["confidence"] in ("high", "medium", "low")


def test_predict_top_features_have_value_and_shap(client):
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    top = client.post("/predict", json=payload).json()["top_features"]
    assert len(top) == 5
    for feat_name, contrib in top.items():
        assert "value" in contrib
        assert "shap" in contrib


def test_predict_too_short_sequence_returns_422(client):
    payload = {"mrna_seq": "ATGAAACCC", "use_rnafold": False}  # < 50 nt
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_mrna_seq_returns_422(client):
    resp = client.post("/predict", json={"use_rnafold": False})
    assert resp.status_code == 422


def test_predict_gene_id_echoed(client):
    payload = {"gene_id": "AT5G47010", "mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    assert body["gene_id"] == "AT5G47010"


def test_predict_high_confidence_label_when_prob_above_075(client):
    # mock returns prob_p_body = 0.75 — check boundary
    payload = {"mrna_seq": _MRNA, "use_rnafold": False}
    body = client.post("/predict", json=payload).json()
    if body["prob_p_body"] >= 0.75:
        assert body["confidence"] == "high"


# ── POST /predict/batch ───────────────────────────────────────────────────────

def test_predict_batch_returns_200(client):
    payload = {"sequences": [
        {"mrna_seq": _MRNA, "use_rnafold": False},
        {"gene_id": "AT2G01234", "mrna_seq": _MRNA, "use_rnafold": False},
    ]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200


def test_predict_batch_n_success(client):
    payload = {"sequences": [
        {"mrna_seq": _MRNA, "use_rnafold": False},
        {"mrna_seq": _MRNA, "use_rnafold": False},
    ]}
    body = client.post("/predict/batch", json=payload).json()
    assert body["n_success"] == 2
    assert body["n_error"] == 0


# ── GET /model/info ───────────────────────────────────────────────────────────

def test_model_info_returns_200(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200


def test_model_info_fields(client):
    body = client.get("/model/info").json()
    for field in ("model", "version", "phase", "n_estimators",
                  "n_features", "test_auroc", "test_auprc"):
        assert field in body, f"missing field: {field}"


def test_model_info_auroc_in_range(client):
    body = client.get("/model/info").json()
    assert 0.0 < body["test_auroc"] <= 1.0
    assert 0.0 < body["test_auprc"] <= 1.0


# ── GET /lookup/{gene_id} — no DB present ────────────────────────────────────

def test_lookup_unknown_gene_returns_404_or_503(client):
    # Either no DB (503) or gene not found (404); a random non-existent ID covers both.
    resp = client.get("/lookup/XX9G99999")
    assert resp.status_code in (404, 503)
