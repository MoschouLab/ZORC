"""
feature_compute.py — ZORC RNA feature computation for the prediction API.

Mirrors the feature engineering logic of scripts/04_rna_features.py and
scripts/09d_feature_engineering.py so the API produces identical inputs to
those seen during model training.

Public interface
----------------
compute_features(mrna_seq, protein_seq=None) -> dict[str, float | None]
    Returns a dict keyed by feature name.  Protein features are None when
    protein_seq is not supplied; they will be median-imputed by the caller.
    RNAfold features are None when RNAfold is not available in PATH.
"""

import re
import subprocess
import tempfile
import os
from collections import Counter
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

RRACH_RE = re.compile(r"[AG][AG]AC[ACU]")
AAACH_RE = re.compile(r"AAAC[ACU]")

RNAFOLD_MAX_LEN = 3000   # P4 default; longer sequences get NaN structure features
EPS_RMSF        = 0.1    # avoid div/0 in rmsf_nterm_cterm_ratio
EPS_CDS         = 1.0    # avoid div/0 in rrach_per_cds_kb


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_rna(seq: str) -> str:
    """Normalise to uppercase RNA (T→U, U kept)."""
    return seq.upper().replace("T", "U")


# ── Nucleotide composition ────────────────────────────────────────────────────

def _nucleotide_features(seq: str) -> dict:
    n = len(seq)
    counts = Counter(seq)
    feats = {
        "mrna_length": n,
        "fA": counts.get("A", 0) / n,
        "fU": counts.get("U", 0) / n,
        "fG": counts.get("G", 0) / n,
        "fC": counts.get("C", 0) / n,
        "au_content": (counts.get("A", 0) + counts.get("U", 0)) / n,
        "gc_content": (counts.get("G", 0) + counts.get("C", 0)) / n,
    }
    dinucs = ["AA","AU","AG","AC","UA","UU","UG","UC",
               "GA","GU","GG","GC","CA","CU","CG","CC"]
    n_di = n - 1
    if n_di > 0:
        di_counts = Counter(seq[i : i + 2] for i in range(n_di))
        for di in dinucs:
            feats[f"di_{di}"] = di_counts.get(di, 0) / n_di
    else:
        for di in dinucs:
            feats[f"di_{di}"] = 0.0
    return feats


# ── m6A motifs ────────────────────────────────────────────────────────────────

def _m6a_features(seq: str, mrna_length: int) -> dict:
    kb = mrna_length / 1000.0
    rrach = len(RRACH_RE.findall(seq))
    aaach = len(AAACH_RE.findall(seq))
    return {
        "rrach_count":  rrach,
        "rrach_per_kb": rrach / kb,
        "aaach_count":  aaach,
        "aaach_per_kb": aaach / kb,
    }


# ── UTR / length features ─────────────────────────────────────────────────────

def _utr_features(seq: str, cds_length: int, utr5_length: int) -> dict:
    """
    Derive UTR features given estimated CDS and 5'UTR lengths.
    utr3_length = mrna_length - cds_length - utr5_length (floored at 0).
    utr3_au_content is computed from the last utr3_length nucleotides;
    when utr3_length == 0 the last 20 % of the mRNA is used as a proxy.
    """
    n = len(seq)
    utr3_length = max(n - cds_length - utr5_length, 0)

    if utr3_length > 0:
        utr3_seq = seq[-min(utr3_length, n):]
    else:
        utr3_proxy = max(int(n * 0.20), 50)
        utr3_seq   = seq[-utr3_proxy:]
        utr3_length = utr3_proxy

    m = Counter(utr3_seq)
    utr3_au = (m.get("A", 0) + m.get("U", 0)) / max(len(utr3_seq), 1)

    utr5_fraction = utr5_length / n if n > 0 else 0.0
    utr3_fraction = utr3_length / n if n > 0 else 0.0
    cds_fraction  = cds_length  / n if n > 0 else 0.0

    long_3utr = int(utr3_length > 1000)

    return {
        "utr5_length":    utr5_length,
        "utr3_length":    utr3_length,
        "cds_length":     cds_length,
        "utr5_fraction":  utr5_fraction,
        "utr3_fraction":  utr3_fraction,
        "cds_fraction":   cds_fraction,
        "utr3_au_content": round(utr3_au, 6),
        "long_3utr":      long_3utr,
        "ptc_proxy":      0,   # cannot determine without GTF / exon map
    }


def _estimate_utr_lengths(mrna_length: int,
                           protein_seq: Optional[str]) -> tuple[int, int]:
    """
    Return (cds_length, utr5_length) estimates.
    If protein_seq is supplied: CDS ≈ len(protein) * 3 + 3 (stop codon).
    Otherwise: CDS ≈ 55 % of mRNA (P-body mRNA median from training data);
    UTR5 ≈ 10 % of mRNA (minimum 50 nt).
    """
    if protein_seq and len(protein_seq) > 0:
        cds_length  = len(protein_seq) * 3 + 3
        # Rough UTR5: 10 % of mRNA (minimum 50 nt, constrained to leave UTR3)
        utr5_length = max(int(mrna_length * 0.10), 50)
        # Guard: total UTR > mRNA length
        if cds_length + utr5_length >= mrna_length:
            utr5_length = max(mrna_length - cds_length - 50, 0)
    else:
        cds_length  = int(mrna_length * 0.55)
        utr5_length = max(int(mrna_length * 0.10), 50)
    return cds_length, utr5_length


# ── RNAfold ───────────────────────────────────────────────────────────────────

def _run_rnafold(seq: str) -> dict:
    """
    Call RNAfold on a single sequence.  Returns NaN dict when RNAfold is
    unavailable or the sequence exceeds RNAFOLD_MAX_LEN nucleotides.
    """
    null_result = {
        "mfe": None, "mfe_per_nt": None,
        "frac_paired": None, "n_stemloops": None,
        "n_stemloops_per_kb": None,
    }
    if len(seq) > RNAFOLD_MAX_LEN:
        return null_result

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fa", delete=False
    ) as tmp:
        tmp.write(f">query\n{seq}\n")
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS", "--infile", tmp_path],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return null_result
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        return null_result

    lines = result.stdout.strip().split("\n")
    # Expected: >query / SEQUENCE / STRUCTURE (MFE kcal/mol)
    for i, line in enumerate(lines):
        if line.startswith(">"):
            struct_line = lines[i + 2] if i + 2 < len(lines) else ""
            break
    else:
        return null_result

    m = re.search(r"\(\s*(-?\d+\.?\d*)\s*\)", struct_line)
    if not m:
        return null_result

    mfe        = float(m.group(1))
    dot_bracket = struct_line.split()[0]
    n_paired    = dot_bracket.count("(") + dot_bracket.count(")")
    frac_paired = n_paired / max(len(dot_bracket), 1)
    n_stemloops = len(re.findall(r"\([.]+\)", dot_bracket))
    n           = len(seq)

    return {
        "mfe":               mfe,
        "mfe_per_nt":        mfe / n if n > 0 else None,
        "frac_paired":       round(frac_paired, 6),
        "n_stemloops":       n_stemloops,
        "n_stemloops_per_kb": n_stemloops / (n / 1000) if n > 0 else None,
    }


# ── Engineered features ───────────────────────────────────────────────────────

def _add_engineered_features(feats: dict) -> dict:
    """
    Compute the 4 engineered features introduced in P9d.
    When inputs are None the result is also None (caller will impute).
    """
    # 1. rmsf_nterm_cterm_ratio
    nterm = feats.get("rmsf_nterm50")
    cterm = feats.get("rmsf_cterm50")
    if nterm is not None and cterm is not None:
        feats["rmsf_nterm_cterm_ratio"] = nterm / (cterm + EPS_RMSF)
    else:
        feats["rmsf_nterm_cterm_ratio"] = None

    # 2. packing_x_idr
    packing = feats.get("packing_density")
    idr     = feats.get("idr_percent")
    if packing is not None and idr is not None:
        feats["packing_x_idr"] = packing * idr
    else:
        feats["packing_x_idr"] = None

    # 3. rrach_per_cds_kb  (computable from RNA)
    rrach     = feats.get("rrach_count", 0.0)
    cds_len   = feats.get("cds_length", 0.0)
    feats["rrach_per_cds_kb"] = rrach / ((cds_len + EPS_CDS) / 1000.0)

    # 4. utr3_au_x_length  (computable from RNA)
    utr3_au  = feats.get("utr3_au_content", 0.0)
    utr3_len = feats.get("utr3_length", 0.0)
    feats["utr3_au_x_length"] = utr3_au * utr3_len

    return feats


# ── Main entry point ──────────────────────────────────────────────────────────

# Protein features that the API cannot compute and sets to None for imputation.
PROTEIN_FEATURE_NAMES = [
    "rmsf_mean", "rmsf_std", "rmsf_max",
    "rmsf_nterm50", "rmsf_cterm50",
    "rg_mean", "rg_std", "rg_cv",
    "contact_density", "packing_density",
    "sasa_per_residue", "pass_rate",
    "n_residues",
    "idr_percent", "mean_disorder",
    "max_disorder_window", "n_idr_regions", "longest_idr_region",
]


def compute_features(
    mrna_seq: str,
    protein_seq: Optional[str] = None,
    use_rnafold: bool = True,
) -> dict:
    """
    Compute all 63 model features for one mRNA sequence.

    Parameters
    ----------
    mrna_seq    : mRNA sequence (DNA or RNA notation, case-insensitive)
    protein_seq : Optional translated protein sequence.  Used only to
                  estimate CDS length for UTR feature computation.
    use_rnafold : Call RNAfold for MFE/structure features.  Set False
                  for speed; structure features will be imputed.

    Returns
    -------
    dict with 63 keys matching the training feature matrix column order.
    Values are float (or None where imputation is required).
    """
    seq = _to_rna(mrna_seq)
    n   = len(seq)

    # ── RNA composition ────────────────────────────────────────────────────
    feats = _nucleotide_features(seq)

    # ── m6A motifs ─────────────────────────────────────────────────────────
    feats.update(_m6a_features(seq, n))

    # ── UTR / length ───────────────────────────────────────────────────────
    cds_length, utr5_length = _estimate_utr_lengths(n, protein_seq)
    feats.update(_utr_features(seq, cds_length, utr5_length))

    # ── RNAfold structure ──────────────────────────────────────────────────
    if use_rnafold:
        feats.update(_run_rnafold(seq))
    else:
        feats.update({
            "mfe": None, "mfe_per_nt": None,
            "frac_paired": None, "n_stemloops": None,
            "n_stemloops_per_kb": None,
        })

    # ── Protein features → None (imputed by caller) ────────────────────────
    for pf in PROTEIN_FEATURE_NAMES:
        feats[pf] = None

    # ── Engineered features ─────────────────────────────────────────────────
    feats = _add_engineered_features(feats)

    return feats
