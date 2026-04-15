#!/usr/bin/env python3
"""
07_protein_features.py
======================
ZORC — Zip-code Of RNAs that Condense
Phase 7: Protein feature extraction from BioEmu conformational ensembles.

For each protein computes features from:
  - BioEmu ensemble (Tier 2/3): topology.pdb + samples.xtc via MDAnalysis
  - AlphaFold2 structure (Tier 1 + OOM excluded): fetched from EBI API

Features extracted:
  Conformational dynamics (BioEmu only):
    - rmsf_mean       : mean per-residue RMSF across ensemble (Å)
    - rmsf_std        : std of per-residue RMSF
    - rmsf_nterm50    : mean RMSF of first 50 residues
    - rmsf_cterm50    : mean RMSF of last 50 residues
    - rmsf_max        : maximum per-residue RMSF
    - rg_mean         : mean radius of gyration (Å)
    - rg_std          : std of radius of gyration
    - rg_cv           : coefficient of variation of Rg (rg_std/rg_mean)
    - pass_rate       : fraction of requested conformations retained

  Structure (all tiers):
    - contact_density : mean Cα-Cα contacts within 8Å per residue
    - packing_density : mean heavy atom contacts within 6Å per residue
    - sasa_per_residue: mean SASA per residue (Å²)
    - n_residues      : protein length

  IDR-specific (from AIUPred, merged from P5):
    - idr_percent, mean_disorder, max_disorder_window
    - n_idr_regions, longest_idr_region

Tier 1 / OOM proteins use AlphaFold2 structure only — dynamic features
(rmsf_*, rg_*, pass_rate) set to NaN and flagged as feature_source='af2_only'.

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/07_protein_features.py --config config/zorc_config.yaml

    # Test with first 10 proteins
    python scripts/07_protein_features.py --config config/zorc_config.yaml --limit 10

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

def ensure_output_keys(cfg, config_path):
    defaults = {
        'protein_features':        'data/processed/07_protein_features.csv',
        'protein_features_report': 'logs/07_protein_features_report.txt',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P7 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    return Path(os.path.expanduser(cfg['project_dir'])) / cfg['outputs'][key]


# =============================================================================
# BIOEMU FEATURE EXTRACTION
# =============================================================================

def extract_bioemu_features(topo_path: str, xtc_path: str,
                              n_requested: int) -> dict:
    """
    Extract conformational ensemble features from BioEmu topology.pdb + samples.xtc.
    Uses MDAnalysis for trajectory analysis.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align

    try:
        u = mda.Universe(topo_path, xtc_path)
    except Exception as e:
        return {'error': f'mda_load_failed: {e}'}

    n_frames = len(u.trajectory)
    if n_frames == 0:
        return {'error': 'empty_trajectory'}

    # Select Cα atoms for backbone analysis
    ca = u.select_atoms('name CA')
    if len(ca) == 0:
        return {'error': 'no_CA_atoms'}

    n_res = len(ca)
    pass_rate = n_frames / n_requested if n_requested > 0 else 0.0

    # --- RMSF (after alignment to mean structure) ---
    try:
        # Align all frames to first frame
        aligner = align.AlignTraj(u, u, select='name CA',
                                   in_memory=True).run()
        rmsf_calc = rms.RMSF(ca).run()
        rmsf_values = rmsf_calc.results.rmsf  # per-residue RMSF in Å

        rmsf_mean   = float(np.mean(rmsf_values))
        rmsf_std    = float(np.std(rmsf_values))
        rmsf_max    = float(np.max(rmsf_values))
        rmsf_nterm  = float(np.mean(rmsf_values[:min(50, n_res)]))
        rmsf_cterm  = float(np.mean(rmsf_values[max(0, n_res-50):]))
    except Exception as e:
        rmsf_mean = rmsf_std = rmsf_max = rmsf_nterm = rmsf_cterm = np.nan

    # --- Radius of gyration ---
    rg_values = []
    try:
        protein = u.select_atoms('protein')
        for ts in u.trajectory:
            rg_values.append(protein.radius_of_gyration())
        rg_arr  = np.array(rg_values)
        rg_mean = float(np.mean(rg_arr))
        rg_std  = float(np.std(rg_arr))
        rg_cv   = float(rg_std / rg_mean) if rg_mean > 0 else np.nan
    except Exception:
        rg_mean = rg_std = rg_cv = np.nan

    # --- Contact density and packing density (from mean structure) ---
    # Use first frame as representative
    contact_density = np.nan
    packing_density = np.nan
    sasa_per_res    = np.nan

    try:
        u.trajectory[0]  # go to first frame

        # Cα-Cα contact density (8Å cutoff)
        ca_pos = ca.positions  # (n_res, 3)
        from scipy.spatial.distance import cdist
        ca_dist = cdist(ca_pos, ca_pos)
        np.fill_diagonal(ca_dist, np.inf)
        contacts_per_res = (ca_dist < 8.0).sum(axis=1)
        contact_density  = float(np.mean(contacts_per_res))

        # Heavy atom packing density (6Å cutoff)
        heavy = u.select_atoms('not name H*')
        if len(heavy) > 0:
            heavy_pos = heavy.positions
            heavy_dist = cdist(heavy_pos, heavy_pos)
            np.fill_diagonal(heavy_dist, np.inf)
            pack_per_atom = (heavy_dist < 6.0).sum(axis=1)
            packing_density = float(np.mean(pack_per_atom))

        # SASA per residue (approximate using radius of atoms)
        # Simple proxy: exposed surface = total SASA / n_residues
        try:
            from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import \
                HydrogenBondAnalysis
        except ImportError:
            pass
        # Use basic SASA via freesasa if available, else skip
        try:
            import freesasa
            struct = freesasa.structureFromMDAnalysis(u.select_atoms('protein'))
            result = freesasa.calc(struct)
            sasa_per_res = float(result.totalArea()) / n_res
        except Exception:
            # Fallback: estimate from Rg (rough proxy)
            if not np.isnan(rg_mean):
                sasa_per_res = 4 * np.pi * (rg_mean ** 2) / n_res

    except Exception as e:
        pass

    return {
        'n_conformations':  n_frames,
        'n_residues':       n_res,
        'pass_rate':        round(pass_rate, 4),
        'rmsf_mean':        round(rmsf_mean, 4) if not np.isnan(rmsf_mean) else np.nan,
        'rmsf_std':         round(rmsf_std, 4)  if not np.isnan(rmsf_std)  else np.nan,
        'rmsf_max':         round(rmsf_max, 4)  if not np.isnan(rmsf_max)  else np.nan,
        'rmsf_nterm50':     round(rmsf_nterm, 4) if not np.isnan(rmsf_nterm) else np.nan,
        'rmsf_cterm50':     round(rmsf_cterm, 4) if not np.isnan(rmsf_cterm) else np.nan,
        'rg_mean':          round(rg_mean, 4)   if not np.isnan(rg_mean)   else np.nan,
        'rg_std':           round(rg_std, 4)    if not np.isnan(rg_std)    else np.nan,
        'rg_cv':            round(rg_cv, 4)     if not np.isnan(rg_cv)     else np.nan,
        'contact_density':  round(contact_density, 4) if not np.isnan(contact_density) else np.nan,
        'packing_density':  round(packing_density, 4) if not np.isnan(packing_density) else np.nan,
        'sasa_per_residue': round(sasa_per_res, 4) if not np.isnan(sasa_per_res) else np.nan,
        'feature_source':   'bioemu',
        'error':            None,
    }


# =============================================================================
# ALPHAFOLD2 FEATURE EXTRACTION
# =============================================================================

def fetch_af2_structure(uniprot_id: str, cache_dir: Path) -> str | None:
    """
    Fetch AlphaFold2 PDB from EBI API.
    Returns local path to cached PDB or None if not found.
    """
    import urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f'{uniprot_id}_AF2.pdb'
    if pdb_path.exists():
        return str(pdb_path)
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
    try:
        urllib.request.urlretrieve(url, str(pdb_path))
        return str(pdb_path)
    except Exception:
        return None


def tair_to_uniprot(gene_id: str, tx_id: str) -> str | None:
    """
    Convert TAIR gene ID to UniProt accession via UniProt ID mapping API.
    Uses gene name search as fallback.
    """
    import urllib.request, urllib.parse, json as _json
    # Try direct gene ID search in UniProt
    query = urllib.parse.quote(f'gene:{gene_id} AND organism_id:3702 AND reviewed:true')
    url = (f'https://rest.uniprot.org/uniprotkb/search?'
           f'query={query}&format=json&size=1&fields=accession')
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = _json.loads(r.read())
        results = data.get('results', [])
        if results:
            return results[0]['primaryAccession']
    except Exception:
        pass

    # Fallback: try transcript ID (strip version)
    base = tx_id.split('.')[0] if '.' in tx_id else tx_id
    query2 = urllib.parse.quote(f'gene:{base} AND organism_id:3702')
    url2 = (f'https://rest.uniprot.org/uniprotkb/search?'
            f'query={query2}&format=json&size=1&fields=accession')
    try:
        with urllib.request.urlopen(url2, timeout=10) as r:
            data = _json.loads(r.read())
        results = data.get('results', [])
        if results:
            return results[0]['primaryAccession']
    except Exception:
        pass
    return None


def extract_af2_features(pdb_path: str) -> dict:
    """
    Extract static structural features from a single PDB file (AF2 or BioEmu frame).
    """
    import MDAnalysis as mda
    from scipy.spatial.distance import cdist

    try:
        u = mda.Universe(pdb_path)
    except Exception as e:
        return {'error': f'af2_load_failed: {e}'}

    ca = u.select_atoms('name CA')
    if len(ca) == 0:
        return {'error': 'no_CA_atoms_af2'}

    n_res = len(ca)
    ca_pos = ca.positions

    try:
        ca_dist = cdist(ca_pos, ca_pos)
        np.fill_diagonal(ca_dist, np.inf)
        contacts_per_res = (ca_dist < 8.0).sum(axis=1)
        contact_density  = float(np.mean(contacts_per_res))
    except Exception:
        contact_density = np.nan

    packing_density = np.nan
    try:
        heavy = u.select_atoms('not name H*')
        if len(heavy) > 0:
            heavy_pos = heavy.positions
            heavy_dist = cdist(heavy_pos, heavy_pos)
            np.fill_diagonal(heavy_dist, np.inf)
            packing_density = float(np.mean((heavy_dist < 6.0).sum(axis=1)))
    except Exception:
        pass

    # Rg from static structure
    try:
        protein = u.select_atoms('protein')
        rg_static = float(protein.radius_of_gyration())
    except Exception:
        rg_static = np.nan

    return {
        'n_conformations':  1,
        'n_residues':       n_res,
        'pass_rate':        np.nan,
        'rmsf_mean':        np.nan,
        'rmsf_std':         np.nan,
        'rmsf_max':         np.nan,
        'rmsf_nterm50':     np.nan,
        'rmsf_cterm50':     np.nan,
        'rg_mean':          round(rg_static, 4) if not np.isnan(rg_static) else np.nan,
        'rg_std':           np.nan,
        'rg_cv':            np.nan,
        'contact_density':  round(contact_density, 4) if not np.isnan(contact_density) else np.nan,
        'packing_density':  round(packing_density, 4) if not np.isnan(packing_density) else np.nan,
        'sasa_per_residue': np.nan,
        'feature_source':   'af2_only',
        'error':            None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 7 — Protein feature extraction from BioEmu/AF2.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N proteins (testing)')
    parser.add_argument('--skip-af2', action='store_true',
                        help='Skip AF2 fetching for Tier1 (faster, NaN for static features)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 7: Protein Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load inputs ---------------------------------------------------------
    tier_path = proj / cfg['outputs']['bioemu_tier_list']
    idr_path  = proj / cfg['outputs']['aiupred_results']
    ckpt_path = proj / cfg['outputs']['bioemu_checkpoint']
    bioemu_dir = proj / cfg['outputs']['bioemu_dir']
    af2_cache  = proj / 'data/processed/af2_structures'

    df_tiers = pd.read_csv(tier_path)
    df_idr   = pd.read_csv(idr_path)
    with open(ckpt_path) as f:
        checkpoint = json.load(f)
    ckpt_tier1_path = proj / cfg["outputs"].get("bioemu_checkpoint_tier1", "data/processed/bioemu/checkpoint_tier1.json")
    if ckpt_tier1_path.exists():
        with open(ckpt_tier1_path) as f:
            checkpoint.update(json.load(f))

    # Merge tier + IDR info
    df = df_tiers.merge(
        df_idr[['geneID', 'transcript_id', 'idr_percent', 'mean_disorder',
                'max_disorder_window', 'n_idr_regions', 'longest_idr_region']],
        on=['geneID', 'transcript_id'], how='left'
    )

    if args.limit:
        df = df.head(args.limit)
        print(f"\n  [--limit {args.limit}] Testing mode")

    print(f"\n  Total proteins to process: {len(df):,}")
    tier_counts = df['bioemu_tier'].value_counts().sort_index()
    for t, n in tier_counts.items():
        print(f"    Tier {t}: {n:,}")

    # --- Process each protein -----------------------------------------------
    print(f"\n[Processing proteins...]")
    results = []
    n_bioemu_ok = 0
    n_af2_ok    = 0
    n_failed    = 0
    t_start     = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        gene_id = row['geneID']
        tx_id   = str(row['transcript_id'])
        tier    = int(row['bioemu_tier'])
        pkey    = f"{gene_id}_{tx_id}"
        ckpt_status = checkpoint.get(pkey, {}).get('status', 'not_run')

        if idx % 50 == 0:
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 and idx > 0 else 0
            eta  = (len(df) - idx) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(df)}] {gene_id} | Tier{tier} | "
                  f"BioEmu:{ckpt_status} | "
                  f"ETA: {eta/60:.0f} min")

        result = {
            'geneID':        gene_id,
            'transcript_id': tx_id,
            'class':         row.get('class', np.nan),
            'condition':     row.get('condition', ''),
            'bioemu_tier':   tier,
            'bioemu_status': ckpt_status,
            'idr_percent':   row.get('idr_percent', np.nan),
            'mean_disorder': row.get('mean_disorder', np.nan),
            'max_disorder_window': row.get('max_disorder_window', np.nan),
            'n_idr_regions': row.get('n_idr_regions', np.nan),
            'longest_idr_region': row.get('longest_idr_region', np.nan),
        }

        # --- BioEmu path ---
        if ckpt_status == 'completed':
            topo = bioemu_dir / pkey / 'topology.pdb'
            xtc  = bioemu_dir / pkey / 'samples.xtc'
            if topo.exists() and xtc.exists():
                if tier == 2:
                    n_req = cfg['bioemu_tiers']['tier2_n_conformations']
                elif tier == 1:
                    n_req = cfg['bioemu_tiers'].get('tier1_n_conformations', 100)
                else:
                    n_req = cfg['bioemu_tiers']['tier3_n_conformations']
                feats = extract_bioemu_features(str(topo), str(xtc), n_req)
                if feats.get('error') is None:
                    n_bioemu_ok += 1
                else:
                    n_failed += 1
                result.update(feats)
            else:
                result.update({'feature_source': 'bioemu_files_missing',
                               'error': 'missing_topo_or_xtc'})
                n_failed += 1

        # --- AF2 path (Tier1, excluded, or BioEmu failed) ---
        else:
            if not args.skip_af2:
                uniprot = tair_to_uniprot(gene_id, tx_id)
                if uniprot:
                    pdb_path = fetch_af2_structure(uniprot, af2_cache)
                    if pdb_path:
                        feats = extract_af2_features(pdb_path)
                        result.update(feats)
                        n_af2_ok += 1
                    else:
                        result.update({'feature_source': 'af2_not_found',
                                       'n_residues': int(row.get('protein_length', 0))})
                        n_failed += 1
                else:
                    result.update({'feature_source': 'uniprot_not_found',
                                   'n_residues': int(row.get('protein_length', 0))})
                    n_failed += 1
            else:
                result.update({'feature_source': 'af2_skipped',
                               'n_residues': int(row.get('protein_length', 0))})

        results.append(result)

    # --- Assemble output dataframe -------------------------------------------
    df_out = pd.DataFrame(results)

    # Ensure all feature columns exist
    feat_cols = ['n_conformations', 'n_residues', 'pass_rate',
                 'rmsf_mean', 'rmsf_std', 'rmsf_max',
                 'rmsf_nterm50', 'rmsf_cterm50',
                 'rg_mean', 'rg_std', 'rg_cv',
                 'contact_density', 'packing_density', 'sasa_per_residue',
                 'feature_source', 'error']
    for col in feat_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    # --- Summary -------------------------------------------------------------
    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print("PROTEIN FEATURE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total processed    : {len(df_out):,}")
    print(f"  BioEmu features    : {n_bioemu_ok:,}")
    print(f"  AF2 features       : {n_af2_ok:,}")
    print(f"  Failed/missing     : {n_failed:,}")
    print(f"  Runtime            : {total_elapsed/60:.1f} min")

    print(f"\n  Feature means (BioEmu proteins, positives vs negatives):")
    bio = df_out[df_out['feature_source'] == 'bioemu']
    for cls, label in [(1, 'Positives'), (0, 'Negatives')]:
        sub = bio[bio['class'] == cls]
        if len(sub):
            print(f"\n  {label} (n={len(sub):,}):")
            for feat in ['rmsf_mean', 'rg_mean', 'rg_cv', 'contact_density',
                         'pass_rate', 'idr_percent']:
                if feat in sub.columns:
                    vals = sub[feat].dropna()
                    if len(vals):
                        print(f"    {feat:<22}: {vals.mean():.4f} ± {vals.std():.4f}")

    # --- Write outputs -------------------------------------------------------
    out_path = rp(cfg, 'protein_features')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\n  Protein features: {out_path}")
    print(f"  Rows: {len(df_out):,}  |  Columns: {len(df_out.columns)}")

    # Write report
    _write_report(df_out, n_bioemu_ok, n_af2_ok, n_failed,
                  total_elapsed, str(rp(cfg, 'protein_features_report')))

    print("\n✓ Phase 7 complete.")
    return 0


def _write_report(df, n_bio, n_af2, n_fail, elapsed, out_path):
    bio = df[df['feature_source'] == 'bioemu']
    lines = [
        "=" * 80,
        "ZORC — Protein Feature Extraction Report",
        "Script: 07_protein_features.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
        "RESULTS",
        f"  Total proteins       : {len(df):,}",
        f"  BioEmu features      : {n_bio:,}",
        f"  AF2-only features    : {n_af2:,}",
        f"  Failed/missing       : {n_fail:,}",
        f"  Runtime              : {elapsed/60:.1f} min",
        "",
        "FEATURE SOURCE BREAKDOWN",
    ]
    for src, cnt in df['feature_source'].value_counts().items():
        lines.append(f"  {src:<30}: {cnt:,}")
    lines += [
        "",
        "RMSF DISTRIBUTION (BioEmu proteins)",
    ]
    for cls, label in [(1, 'Positives'), (0, 'Negatives')]:
        sub = bio[bio['class'] == cls]['rmsf_mean'].dropna()
        if len(sub):
            lines.append(
                f"  {label}: median={sub.median():.2f} Å  "
                f"mean={sub.mean():.2f} Å  "
                f"min={sub.min():.2f}  max={sub.max():.2f}"
            )
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
