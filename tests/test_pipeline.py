"""
tests/test_pipeline.py — Unit tests for ZORC RNA feature computation.

Tests cover api/feature_compute.py functions.  RNAfold subprocess calls
are mocked so these tests run without ViennaRNA installed (CI environment).
"""

import pytest
from unittest.mock import patch, MagicMock
from collections import Counter

from api.feature_compute import (
    _to_rna,
    _nucleotide_features,
    _m6a_features,
    _utr_features,
    _estimate_utr_lengths,
    _run_rnafold,
    _add_engineered_features,
    compute_features,
    PROTEIN_FEATURE_NAMES,
    RNAFOLD_MAX_LEN,
)

# ── Short synthetic mRNA (200 nt, AU-rich 3'UTR) ──────────────────────────────
_SEQ_DNA = (
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
    "ATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAAATGAAACCCGGGTTTTAA"
)
_SEQ_RNA = _SEQ_DNA.replace("T", "U")
_PROTEIN = "MKPGF" * 10   # 50 aa → CDS ~153 nt


# ── _to_rna ───────────────────────────────────────────────────────────────────

def test_to_rna_converts_t_to_u():
    assert _to_rna("ATGC") == "AUGC"


def test_to_rna_uppercases():
    assert _to_rna("atgc") == "AUGC"


def test_to_rna_leaves_u_unchanged():
    assert _to_rna("AUGC") == "AUGC"


# ── _nucleotide_features ──────────────────────────────────────────────────────

def test_nucleotide_features_keys():
    feats = _nucleotide_features(_SEQ_RNA)
    assert "mrna_length" in feats
    assert "fA" in feats
    assert "au_content" in feats
    assert "di_CG" in feats


def test_nucleotide_features_mrna_length():
    feats = _nucleotide_features(_SEQ_RNA)
    assert feats["mrna_length"] == len(_SEQ_RNA)


def test_nucleotide_features_fractions_sum_to_1():
    feats = _nucleotide_features(_SEQ_RNA)
    total = feats["fA"] + feats["fU"] + feats["fG"] + feats["fC"]
    assert abs(total - 1.0) < 1e-6


def test_nucleotide_features_au_plus_gc_is_1():
    feats = _nucleotide_features(_SEQ_RNA)
    assert abs(feats["au_content"] + feats["gc_content"] - 1.0) < 1e-6


def test_nucleotide_features_16_dinucs():
    feats = _nucleotide_features(_SEQ_RNA)
    dinucs = [k for k in feats if k.startswith("di_")]
    assert len(dinucs) == 16


def test_nucleotide_features_pure_a_sequence():
    feats = _nucleotide_features("AAAAAAA")
    assert feats["fA"] == pytest.approx(1.0)
    assert feats["fU"] == pytest.approx(0.0)
    assert feats["di_AA"] == pytest.approx(1.0)
    assert feats["di_AU"] == pytest.approx(0.0)


# ── _m6a_features ─────────────────────────────────────────────────────────────

def test_m6a_features_rrach_count():
    # GGACU is a RRACH motif
    seq = "GGACUGGACUGGACU"
    feats = _m6a_features(seq, len(seq))
    assert feats["rrach_count"] == 3


def test_m6a_features_no_motif():
    seq = "CCCCCCCCCCCCCCCC"
    feats = _m6a_features(seq, len(seq))
    assert feats["rrach_count"] == 0
    assert feats["rrach_per_kb"] == pytest.approx(0.0)


def test_m6a_features_per_kb_scaling():
    seq = "GGACU" * 200   # 1000 nt, 200 RRACH → 200 per kb
    feats = _m6a_features(seq, len(seq))
    assert feats["rrach_per_kb"] == pytest.approx(200.0)


# ── _estimate_utr_lengths ─────────────────────────────────────────────────────

def test_estimate_utr_lengths_with_protein():
    # 50aa protein → CDS = 153, utr5 ≈ max(200*0.10, 50) = 50
    cds, utr5 = _estimate_utr_lengths(2000, _PROTEIN)
    assert cds == len(_PROTEIN) * 3 + 3
    assert utr5 >= 50


def test_estimate_utr_lengths_without_protein():
    cds, utr5 = _estimate_utr_lengths(1000, None)
    assert cds == 550   # 55 % of 1000
    assert utr5 >= 50


def test_estimate_utr_lengths_large_protein_clamps_utr5():
    # CDS longer than mRNA → utr5 clamped to 0
    cds, utr5 = _estimate_utr_lengths(100, "M" * 200)
    assert utr5 == 0


# ── _utr_features ─────────────────────────────────────────────────────────────

def test_utr_features_keys():
    feats = _utr_features(_SEQ_RNA, 120, 30)
    for key in ("utr5_length", "utr3_length", "cds_length",
                "utr3_au_content", "utr5_fraction", "long_3utr"):
        assert key in feats, f"missing key: {key}"


def test_utr_features_fractions_bounded():
    feats = _utr_features(_SEQ_RNA, 120, 30)
    assert 0.0 <= feats["utr3_au_content"] <= 1.0
    assert 0.0 <= feats["utr5_fraction"] <= 1.0
    assert 0.0 <= feats["cds_fraction"] <= 1.0


def test_utr_features_long_3utr_flag():
    long_seq = "A" * 2000
    feats = _utr_features(long_seq, 500, 50)
    assert feats["long_3utr"] == 1

    short_seq = "A" * 300
    feats2 = _utr_features(short_seq, 200, 30)
    assert feats2["long_3utr"] == 0


# ── _run_rnafold (mocked) ─────────────────────────────────────────────────────

_RNAFOLD_STDOUT = ">query\n{seq}\n.(((...))) (-5.30)\n"


def test_run_rnafold_returns_null_for_long_seq():
    long_seq = "A" * (RNAFOLD_MAX_LEN + 1)
    result = _run_rnafold(long_seq)
    assert result["mfe"] is None


@patch("api.feature_compute.subprocess.run")
def test_run_rnafold_parses_output(mock_run):
    seq = "A" * 100
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f">query\n{seq}\n.(((...))) (-5.30)\n"
    mock_run.return_value = mock_result

    result = _run_rnafold(seq)
    assert result["mfe"] == pytest.approx(-5.30)
    assert result["mfe_per_nt"] == pytest.approx(-5.30 / 100)
    assert result["frac_paired"] is not None
    assert result["n_stemloops"] is not None


@patch("api.feature_compute.subprocess.run", side_effect=FileNotFoundError)
def test_run_rnafold_handles_missing_binary(mock_run):
    result = _run_rnafold("AUGCAUGC" * 10)
    assert result["mfe"] is None
    assert result["mfe_per_nt"] is None


# ── _add_engineered_features ──────────────────────────────────────────────────

def test_add_engineered_features_computes_ratio():
    feats = {
        "rmsf_nterm50": 10.0, "rmsf_cterm50": 5.0,
        "packing_density": 2.0, "idr_percent": 0.3,
        "rrach_count": 20.0, "cds_length": 1000.0,
        "utr3_au_content": 0.6, "utr3_length": 500.0,
    }
    out = _add_engineered_features(feats)
    assert out["rmsf_nterm_cterm_ratio"] == pytest.approx(10.0 / (5.0 + 0.1))
    assert out["packing_x_idr"] == pytest.approx(2.0 * 0.3)
    assert out["rrach_per_cds_kb"] == pytest.approx(20.0 / 1.001)
    assert out["utr3_au_x_length"] == pytest.approx(0.6 * 500.0)


def test_add_engineered_features_none_inputs_give_none():
    feats = {
        "rmsf_nterm50": None, "rmsf_cterm50": None,
        "packing_density": None, "idr_percent": None,
        "rrach_count": 0.0, "cds_length": 0.0,
        "utr3_au_content": 0.0, "utr3_length": 0.0,
    }
    out = _add_engineered_features(feats)
    assert out["rmsf_nterm_cterm_ratio"] is None
    assert out["packing_x_idr"] is None


# ── compute_features (integration, no RNAfold) ───────────────────────────────

def test_compute_features_returns_63_keys():
    feats = compute_features(_SEQ_DNA, use_rnafold=False)
    assert len(feats) == 63


def test_compute_features_protein_features_are_none():
    feats = compute_features(_SEQ_DNA, use_rnafold=False)
    for pf in PROTEIN_FEATURE_NAMES:
        assert feats[pf] is None, f"expected None for {pf}"


def test_compute_features_structure_none_when_rnafold_off():
    feats = compute_features(_SEQ_DNA, use_rnafold=False)
    assert feats["mfe"] is None
    assert feats["mfe_per_nt"] is None


def test_compute_features_rna_notation_accepted():
    feats_dna = compute_features(_SEQ_DNA, use_rnafold=False)
    feats_rna = compute_features(_SEQ_RNA, use_rnafold=False)
    assert feats_dna["mrna_length"] == feats_rna["mrna_length"]
    assert feats_dna["fA"] == pytest.approx(feats_rna["fA"])


def test_compute_features_mrna_length_correct():
    feats = compute_features(_SEQ_DNA, use_rnafold=False)
    assert feats["mrna_length"] == len(_SEQ_DNA)


def test_compute_features_with_protein_seq():
    feats = compute_features(_SEQ_DNA, protein_seq=_PROTEIN, use_rnafold=False)
    assert feats["cds_length"] == len(_PROTEIN) * 3 + 3
