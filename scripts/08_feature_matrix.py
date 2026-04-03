#!/usr/bin/env python3
"""
08_feature_matrix.py
====================
ZORC — Zip-code Of RNAs that Condense
Phase 8: Feature matrix assembly + anti-leakage train/CV/test split.

Merges RNA features (P4), protein features (P7), and IDR features (P5)
into a single feature matrix. Applies CD-HIT clustering at 40% protein
identity to prevent paralog leakage across train/CV/test splits.

Steps:
  1. Load P4 (RNA features), P7 (protein features), P5 (IDR summary)
  2. Merge on geneID + transcript_id
  3. Handle missing values:
     - BioEmu dynamic features (rmsf_*, rg_cv) → NaN for Tier1/excluded
     - RNAfold features → NaN for sequences > 3000 nt
     - Strategy: impute with class-conditional median in P9 (RF handles NaN
       via imputation; recorded here as-is for transparency)
  4. Run CD-HIT on all_protein.fa at 40% identity threshold
  5. Assign clusters to train/CV/test (70/15/15), stratified by class
  6. Write feature matrix with split assignments

Output:
  data/processed/08_zorc_feature_matrix.csv
  data/processed/08_zorc_split_assignments.csv
  logs/08_feature_matrix_report.txt

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/08_feature_matrix.py --config config/zorc_config.yaml

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
        'feature_matrix':       'data/processed/08_zorc_feature_matrix.csv',
        'split_assignments':    'data/processed/08_zorc_split_assignments.csv',
        'feature_matrix_report':'logs/08_feature_matrix_report.txt',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P8 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    return Path(os.path.expanduser(cfg['project_dir'])) / cfg['outputs'][key]


# =============================================================================
# CD-HIT CLUSTERING
# =============================================================================

def run_cdhit(fasta_path: str, identity: float,
              out_prefix: str, threads: int = 4) -> dict:
    """
    Run CD-HIT on protein FASTA at given identity threshold.
    Returns dict: sequence_id → cluster_id (int)
    """
    cmd = [
        'cd-hit',
        '-i', fasta_path,
        '-o', out_prefix,
        '-c', str(identity),
        '-n', '2',        # word length for 40% identity
        '-T', str(threads),
        '-M', '8000',     # memory limit MB
        '-d', '0',        # full sequence name in cluster file
        '-sc', '1',       # sort by cluster size
    ]
    print(f"  Running CD-HIT (identity={identity})...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CD-HIT ERROR: {result.stderr[:500]}")
        return {}

    # Parse .clstr file
    clstr_path = out_prefix + '.clstr'
    clusters = parse_cdhit_clusters(clstr_path)
    print(f"  → {len(set(clusters.values())):,} clusters for "
          f"{len(clusters):,} sequences")
    return clusters


def parse_cdhit_clusters(clstr_path: str) -> dict:
    """
    Parse CD-HIT .clstr file.
    Returns dict: sequence_id → cluster_id
    """
    seq_to_cluster = {}
    current_cluster = -1

    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster += 1
            else:
                # Extract sequence ID: ">seq_id..."
                m = re.search(r'>([^\s\.]+)', line)
                if m:
                    seq_id = m.group(1)
                    seq_to_cluster[seq_id] = current_cluster

    return seq_to_cluster


# =============================================================================
# ANTI-LEAKAGE SPLIT
# =============================================================================

def assign_cluster_split(clusters: dict, labels: dict,
                          train_frac: float = 0.70,
                          val_frac:   float = 0.15,
                          seed: int = 42) -> dict:
    """
    Assign clusters to train/val/test splits, stratified by majority class.

    clusters: {seq_id: cluster_id}
    labels:   {seq_id: class_label (0 or 1)}

    Returns: {seq_id: split_name}
    """
    np.random.seed(seed)

    # Group sequences by cluster
    cluster_seqs = defaultdict(list)
    for seq_id, cid in clusters.items():
        cluster_seqs[cid].append(seq_id)

    # Determine majority class per cluster
    cluster_info = []
    for cid, seqs in cluster_seqs.items():
        class_counts = defaultdict(int)
        for s in seqs:
            class_counts[labels.get(s, -1)] += 1
        majority_class = max(class_counts, key=class_counts.get)
        cluster_info.append({
            'cluster_id':     cid,
            'n_sequences':    len(seqs),
            'majority_class': majority_class,
            'seqs':           seqs,
        })

    # Separate by majority class for stratified assignment
    pos_clusters = [c for c in cluster_info if c['majority_class'] == 1]
    neg_clusters = [c for c in cluster_info if c['majority_class'] == 0]
    unk_clusters = [c for c in cluster_info if c['majority_class'] not in (0, 1)]

    def split_clusters(clist, train_f, val_f):
        np.random.shuffle(clist)
        n = len(clist)
        n_train = int(n * train_f)
        n_val   = int(n * val_f)
        return (clist[:n_train],
                clist[n_train:n_train+n_val],
                clist[n_train+n_val:])

    p_train, p_val, p_test = split_clusters(pos_clusters, train_frac, val_frac)
    n_train, n_val, n_test = split_clusters(neg_clusters, train_frac, val_frac)

    # Assign splits
    seq_splits = {}
    for split_name, cluster_list in [
        ('train', p_train + n_train),
        ('val',   p_val   + n_val),
        ('test',  p_test  + n_test),
        ('train', unk_clusters),  # unknowns go to train
    ]:
        for c in cluster_list:
            for seq in c['seqs']:
                seq_splits[seq] = split_name

    # Summary
    split_counts = defaultdict(lambda: defaultdict(int))
    for seq, split in seq_splits.items():
        split_counts[split][labels.get(seq, -1)] += 1

    print(f"  Split assignments:")
    for split in ['train', 'val', 'test']:
        sc = split_counts[split]
        print(f"    {split}: {sum(sc.values()):,} total "
              f"(pos={sc[1]:,}, neg={sc[0]:,})")

    return seq_splits


# =============================================================================
# FEATURE COLUMNS DEFINITION
# =============================================================================

# RNA features (from P4)
RNA_FEATURES = [
    'mrna_length', 'fA', 'fU', 'fG', 'fC', 'au_content', 'gc_content',
    'di_AA', 'di_AU', 'di_AG', 'di_AC', 'di_UA', 'di_UU', 'di_UG', 'di_UC',
    'di_GA', 'di_GU', 'di_GG', 'di_GC', 'di_CA', 'di_CU', 'di_CG', 'di_CC',
    'utr5_length', 'utr3_length', 'cds_length',
    'utr5_fraction', 'utr3_fraction', 'cds_fraction',
    'utr3_au_content',
    'rrach_count', 'rrach_per_kb', 'aaach_count', 'aaach_per_kb',
    'mfe', 'mfe_per_nt', 'frac_paired', 'n_stemloops', 'n_stemloops_per_kb',
    'long_3utr', 'ptc_proxy',
]

# Protein / IDR features (from P5 + P7)
PROTEIN_FEATURES = [
    'idr_percent', 'mean_disorder', 'max_disorder_window',
    'n_idr_regions', 'longest_idr_region',
    'rmsf_mean', 'rmsf_std', 'rmsf_max', 'rmsf_nterm50', 'rmsf_cterm50',
    'rg_mean', 'rg_std', 'rg_cv',
    'contact_density', 'packing_density', 'sasa_per_residue',
    'pass_rate', 'n_residues',
]

META_COLS = [
    'geneID', 'transcript_id', 'class', 'condition',
    'bioemu_tier', 'bioemu_status', 'feature_source',
    'isoform_source', 'event_type',
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 8 — Feature matrix assembly + anti-leakage split.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--skip-cdhit', action='store_true',
                        help='Skip CD-HIT clustering (use random split instead)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 8: Feature Matrix Assembly")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load all feature tables ---------------------------------------------
    print("\n[1/5] Loading feature tables...")

    df_rna  = pd.read_csv(proj / cfg['outputs']['rna_features'])
    df_prot = pd.read_csv(proj / cfg['outputs']['protein_features'])
    df_iso  = pd.read_csv(proj / cfg['outputs']['isoform_map'])[
        ['geneID', 'transcript_id', 'isoform_source', 'event_type']
    ]

    print(f"  RNA features:     {len(df_rna):,} rows, {len(df_rna.columns)} cols")
    print(f"  Protein features: {len(df_prot):,} rows, {len(df_prot.columns)} cols")

    # Standardise geneID + transcript_id types
    for df in [df_rna, df_prot, df_iso]:
        df['geneID']       = df['geneID'].astype(str).str.strip()
        df['transcript_id'] = df['transcript_id'].astype(str).str.strip()

    # --- Merge ---------------------------------------------------------------
    print("\n[2/5] Merging feature tables...")

    # RNA → use geneID as key (RNA features are per gene/transcript)
    df_merge = df_rna[['geneID', 'transcript_id', 'class', 'condition',
                        'qc_fail'] + [c for c in RNA_FEATURES if c in df_rna.columns]].copy()

    # Protein features → merge on geneID + transcript_id
    prot_cols = ['geneID', 'transcript_id', 'bioemu_tier', 'bioemu_status',
                 'feature_source'] + [c for c in PROTEIN_FEATURES
                                       if c in df_prot.columns]
    df_merge = df_merge.merge(
        df_prot[[c for c in prot_cols if c in df_prot.columns]],
        on=['geneID', 'transcript_id'], how='left'
    )

    # Isoform metadata
    df_merge = df_merge.merge(df_iso, on=['geneID', 'transcript_id'], how='left')

    print(f"  Merged: {len(df_merge):,} rows, {len(df_merge.columns)} cols")
    print(f"  Class distribution: pos={( df_merge['class']==1).sum():,}, "
          f"neg={(df_merge['class']==0).sum():,}")

    # --- Missing value summary -----------------------------------------------
    print("\n[3/5] Missing value analysis...")
    feature_cols = [c for c in RNA_FEATURES + PROTEIN_FEATURES
                    if c in df_merge.columns]
    missing = df_merge[feature_cols].isna().sum()
    missing_pct = missing / len(df_merge) * 100
    print(f"  Features with >10% missing:")
    for col, pct in missing_pct[missing_pct > 10].items():
        print(f"    {col:<30}: {pct:.1f}% missing")
    print(f"  Features with 0% missing: "
          f"{(missing_pct == 0).sum()}")

    # --- CD-HIT clustering ---------------------------------------------------
    print("\n[4/5] Anti-leakage split via CD-HIT clustering...")

    prot_fa = proj / cfg['outputs']['all_protein_fa']
    identity = cfg['ml']['cdhit_identity_threshold']
    seed     = cfg['ml']['random_seed']
    train_f  = cfg['ml']['train_fraction']
    val_f    = cfg['ml']['val_fraction']

    seq_splits = {}

    if not args.skip_cdhit:
        # Check CD-HIT is available
        check = subprocess.run(['cd-hit', '-h'],
                               capture_output=True, text=True)
        if check.returncode not in (0, 1):
            print("  WARNING: cd-hit not found. Using random split.")
            args.skip_cdhit = True

    if not args.skip_cdhit:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, 'cdhit_out')
            clusters = run_cdhit(str(prot_fa), identity, out_prefix)

            if clusters:
                # Map seq IDs — CD-HIT uses first token of header
                # Our FASTA headers: GeneID|TranscriptID|classN|condition
                # CD-HIT uses full header up to first space
                # Remap to transcript_id
                tx_clusters = {}
                for header, cid in clusters.items():
                    # Header format: GeneID|TranscriptID|...
                    parts = header.split('|')
                    tx_id = parts[1] if len(parts) > 1 else header
                    tx_clusters[tx_id] = cid

                labels = dict(zip(df_merge['transcript_id'],
                                  df_merge['class']))
                seq_splits = assign_cluster_split(
                    tx_clusters, labels, train_f, val_f, seed
                )
                df_merge['cluster_id'] = df_merge['transcript_id'].map(tx_clusters)
                print(f"  CD-HIT: {len(set(tx_clusters.values())):,} clusters")
            else:
                print("  CD-HIT returned empty results — using random split")
                args.skip_cdhit = True

    if args.skip_cdhit or not seq_splits:
        print("  Using stratified random split (no CD-HIT)...")
        np.random.seed(seed)
        df_merge['cluster_id'] = np.arange(len(df_merge))  # each seq = own cluster
        pos_idx = df_merge[df_merge['class'] == 1].index.tolist()
        neg_idx = df_merge[df_merge['class'] == 0].index.tolist()

        def random_split(idx_list):
            np.random.shuffle(idx_list)
            n = len(idx_list)
            n_tr = int(n * train_f)
            n_va = int(n * val_f)
            return idx_list[:n_tr], idx_list[n_tr:n_tr+n_va], idx_list[n_tr+n_va:]

        p_tr, p_va, p_te = random_split(pos_idx)
        n_tr, n_va, n_te = random_split(neg_idx)
        for idxs, name in [(p_tr+n_tr,'train'),(p_va+n_va,'val'),(p_te+n_te,'test')]:
            for i in idxs:
                tx = df_merge.loc[i, 'transcript_id']
                seq_splits[tx] = name

    df_merge['split'] = df_merge['transcript_id'].map(seq_splits).fillna('train')

    # --- Write outputs -------------------------------------------------------
    print("\n[5/5] Writing outputs...")

    # Full feature matrix
    out_matrix = rp(cfg, 'feature_matrix')
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    df_merge.to_csv(out_matrix, index=False)
    print(f"  Feature matrix: {out_matrix}")
    print(f"  Rows: {len(df_merge):,}  |  Feature columns: {len(feature_cols)}")

    # Split assignments (lightweight reference file)
    split_df = df_merge[['geneID', 'transcript_id', 'class',
                          'split', 'cluster_id']].copy()
    out_split = rp(cfg, 'split_assignments')
    split_df.to_csv(out_split, index=False)
    print(f"  Split assignments: {out_split}")

    # Write report
    _write_report(df_merge, feature_cols, missing_pct,
                  str(rp(cfg, 'feature_matrix_report')))

    # --- Final summary -------------------------------------------------------
    print(f"\n{'='*70}")
    print("FEATURE MATRIX SUMMARY")
    print(f"{'='*70}")
    print(f"  Total samples      : {len(df_merge):,}")
    print(f"  Feature columns    : {len(feature_cols)}")
    print(f"    RNA features     : {len([c for c in RNA_FEATURES if c in df_merge.columns])}")
    print(f"    Protein features : {len([c for c in PROTEIN_FEATURES if c in df_merge.columns])}")
    print(f"\n  Split distribution:")
    for split in ['train', 'val', 'test']:
        sub = df_merge[df_merge['split'] == split]
        print(f"    {split}: {len(sub):,} "
              f"(pos={( sub['class']==1).sum():,}, "
              f"neg={(sub['class']==0).sum():,})")
    print(f"{'='*70}")
    print("\n✓ Phase 8 complete.")
    return 0


def _write_report(df, feature_cols, missing_pct, out_path):
    lines = [
        "=" * 80,
        "ZORC — Feature Matrix Assembly Report",
        "Script: 08_feature_matrix.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
        f"DATASET: {len(df):,} samples, {len(feature_cols)} features",
        f"  Positives (class=1): {(df['class']==1).sum():,}",
        f"  Negatives (class=0): {(df['class']==0).sum():,}",
        "",
        "SPLIT DISTRIBUTION",
    ]
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        lines.append(f"  {split}: {len(sub):,} "
                     f"(pos={(sub['class']==1).sum():,}, "
                     f"neg={(sub['class']==0).sum():,})")
    lines += ["", "FEATURES WITH >5% MISSING"]
    for col, pct in missing_pct[missing_pct > 5].items():
        lines.append(f"  {col:<30}: {pct:.1f}%")
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
