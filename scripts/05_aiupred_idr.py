#!/usr/bin/env python3
"""
05_aiupred_idr.py
=================
ZORC — Zip-code Of RNAs that Condense
Phase 5: IDR prediction with AIUPred + BioEmu tier assignment.

Runs AIUPred (v0.9, Erdos & Dosztanyi) on all coregulon protein sequences
using --force-cpu to bypass CUDA/NCCL conflicts in the bioemu conda env.

For each protein computes:
  - IDR%: fraction of residues with AIUPred score > 0.5
  - mean_disorder: mean AIUPred score across all residues
  - max_disorder_window: max mean score in any 20-residue sliding window
  - n_idr_regions: number of contiguous IDR segments (score > 0.5)
  - longest_idr_region: length of longest contiguous IDR segment

BioEmu tier assignment (from config.yaml):
  Tier 1: IDR% <  tier1_max_idr  → AlphaFold2 static features only
  Tier 2: IDR% <  tier2_max_idr  → BioEmu 50 conformations
  Tier 3: IDR% >= tier2_max_idr  → BioEmu 50-100 conformations

Requires: bioemu conda env (conda activate bioemu before running)
AIUPred:  ~/Documents/bioemu_analysis/AIUPred/aiupred.py

Usage:
    conda activate bioemu
    python scripts/05_aiupred_idr.py --config config/zorc_config.yaml

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

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
        'aiupred_results':    'data/processed/05_aiupred_idr_summary.csv',
        'aiupred_report':     'logs/05_aiupred_report.txt',
        'bioemu_tier_list':   'data/processed/05_bioemu_tier_assignments.csv',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P5 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    proj = Path(os.path.expanduser(cfg['project_dir']))
    return proj / cfg['outputs'][key]


# =============================================================================
# FASTA PARSING
# =============================================================================

def parse_fasta(path):
    """Return list of (header, sequence) tuples."""
    records = []
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line.strip())
    if header is not None:
        records.append((header, ''.join(seq)))
    return records


def extract_gene_tx_from_header(header):
    """
    Parse ZORC FASTA header: GeneID|TranscriptID|classN|condition[|QC_FAIL:...]
    Returns (gene_id, transcript_id, class, condition, qc_fail)
    """
    parts = header.split('|')
    gene_id     = parts[0] if len(parts) > 0 else ''
    tx_id       = parts[1] if len(parts) > 1 else ''
    cls_str     = parts[2] if len(parts) > 2 else 'class0'
    condition   = parts[3] if len(parts) > 3 else ''
    qc_fail     = any('QC_FAIL' in p for p in parts)
    try:
        cls = int(cls_str.replace('class', ''))
    except ValueError:
        cls = -1
    return gene_id, tx_id, cls, condition, qc_fail


# =============================================================================
# AIUPRED RUNNER
# =============================================================================

def find_aiupred(cfg):
    """Locate aiupred.py from config or default paths."""
    # Check config first
    aiupred_path = cfg.get('tools', {}).get('aiupred_script', None)
    if aiupred_path and Path(os.path.expanduser(aiupred_path)).exists():
        return os.path.expanduser(aiupred_path)

    # Default location from previous pipeline work
    default = os.path.expanduser('~/Documents/bioemu_analysis/AIUPred/aiupred.py')
    if Path(default).exists():
        return default

    return None


def run_aiupred_batch(protein_fa: str, aiupred_script: str,
                      batch_size: int = 100) -> dict:
    """
    Run AIUPred on all proteins in batches.
    Returns dict: transcript_id → list of (residue, score) tuples.

    Batching avoids memory issues with 1500+ sequences.
    Uses --force-cpu to bypass CUDA/NCCL conflicts.
    """
    records = parse_fasta(protein_fa)
    # Filter out QC-fail sequences and those without valid protein sequence
    valid = [(h, s) for h, s in records
             if s and len(s) >= 10 and 'QC_FAIL' not in h]

    print(f"  Total protein sequences: {len(records):,}")
    print(f"  Valid for AIUPred (len>=10, QC pass): {len(valid):,}")

    all_results = {}
    n_batches = (len(valid) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch = valid[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"  Batch {batch_idx+1}/{n_batches} "
              f"({len(batch)} sequences)...", end=' ', flush=True)

        # Write temp FASTA for this batch
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa',
                                         delete=False) as tmp_in:
            for header, seq in batch:
                # Use only transcript_id as header to keep it clean
                parts = header.split('|')
                tx_id = parts[1] if len(parts) > 1 else header.split()[0]
                tmp_in.write(f'>{tx_id}\n{seq}\n')
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                          delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            result = subprocess.run(
                ['python', aiupred_script,
                 '-i', tmp_in_path,
                 '-o', tmp_out_path,
                 '--force-cpu'],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"ERROR (exit {result.returncode})")
                print(f"  stderr: {result.stderr[:300]}")
                continue

            # Parse AIUPred output
            # Format: residue_num  amino_acid  score  (tab-separated)
            # Sequences separated by blank lines or header lines
            batch_results = parse_aiupred_output(tmp_out_path, batch)
            all_results.update(batch_results)
            print(f"OK ({len(batch_results)} parsed)")

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT (batch {batch_idx+1})")
        finally:
            os.unlink(tmp_in_path)
            try:
                os.unlink(tmp_out_path)
            except FileNotFoundError:
                pass

    return all_results


def parse_aiupred_output(output_path: str, batch: list) -> dict:
    """
    Parse AIUPred v2.0 output file.

    Actual format observed:
      # ... (logo/header comment lines, may be empty comment lines too)
      # Position\tResidue\tDisorder       ← column header
      #>AT1G08370.1                        ← sequence header: starts with #>
      1\tM\t0.9563
      2\tS\t0.9533
      ...
      (sequences separated by next #> header or EOF)

    Returns dict: tx_id → list of float scores
    """
    results = {}
    current_tx = None
    current_scores = []

    with open(output_path) as f:
        for line in f:
            line = line.rstrip('\n')

            # Sequence header line: "#>TX_ID"
            if line.startswith('#>'):
                # Save previous sequence if any
                if current_tx and current_scores:
                    results[current_tx] = current_scores
                tx_candidate = line[2:].strip()  # strip "#>"
                # Take first whitespace-delimited token
                current_tx = tx_candidate.split()[0] if tx_candidate else None
                current_scores = []
                continue

            # Skip all other comment/header lines
            if line.startswith('#') or not line.strip():
                continue

            # Data line: "1\tM\t0.9563"
            parts = line.split()
            if len(parts) >= 3:
                try:
                    score = float(parts[2])
                    current_scores.append(score)
                except ValueError:
                    pass

    # Flush last sequence
    if current_tx and current_scores:
        results[current_tx] = current_scores

    return results


# =============================================================================
# IDR FEATURE COMPUTATION
# =============================================================================

def compute_idr_features(scores: list, threshold: float = 0.5,
                          window: int = 20) -> dict:
    """
    Compute IDR features from a list of per-residue AIUPred scores.
    """
    if not scores:
        return {
            'idr_percent':          0.0,
            'mean_disorder':        0.0,
            'max_disorder_window':  0.0,
            'n_idr_regions':        0,
            'longest_idr_region':   0,
            'protein_length':       0,
        }

    n = len(scores)
    idr_mask = [s > threshold for s in scores]

    # IDR%
    idr_percent = 100.0 * sum(idr_mask) / n

    # Mean disorder
    mean_disorder = sum(scores) / n

    # Max disorder window (sliding window mean)
    if n >= window:
        max_win = max(
            sum(scores[i:i+window]) / window
            for i in range(n - window + 1)
        )
    else:
        max_win = mean_disorder

    # Count contiguous IDR regions
    n_regions = 0
    longest = 0
    current_len = 0
    for is_idr in idr_mask:
        if is_idr:
            current_len += 1
            longest = max(longest, current_len)
        else:
            if current_len > 0:
                n_regions += 1
            current_len = 0
    if current_len > 0:
        n_regions += 1

    return {
        'idr_percent':         round(idr_percent, 2),
        'mean_disorder':       round(mean_disorder, 4),
        'max_disorder_window': round(max_win, 4),
        'n_idr_regions':       n_regions,
        'longest_idr_region':  longest,
        'protein_length':      n,
    }


# =============================================================================
# TIER ASSIGNMENT
# =============================================================================

def assign_tier(idr_percent: float, cfg: dict) -> int:
    """
    Assign BioEmu computational tier based on IDR%.
    Tier 1: IDR% < tier1_max_idr  → AlphaFold2 static features only
    Tier 2: IDR% < tier2_max_idr  → BioEmu 50 conformations
    Tier 3: IDR% >= tier2_max_idr → BioEmu 50-100 conformations
    """
    t1 = cfg['bioemu_tiers']['tier1_max_idr']
    t2 = cfg['bioemu_tiers']['tier2_max_idr']
    if idr_percent < t1:
        return 1
    elif idr_percent < t2:
        return 2
    else:
        return 3


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 5 — AIUPred IDR prediction + BioEmu tier assignment.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Sequences per AIUPred batch (default: 100)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Locate files and check AIUPred, no computation')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 5: AIUPred IDR Prediction + BioEmu Tier Assignment")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Locate AIUPred ------------------------------------------------------
    aiupred_script = find_aiupred(cfg)
    if not aiupred_script:
        print("ERROR: aiupred.py not found.")
        print("  Add 'tools.aiupred_script' to config.yaml or place at:")
        print("  ~/Documents/bioemu_analysis/AIUPred/aiupred.py")
        return 1
    print(f"\n  AIUPred script: {aiupred_script}")

    # Verify --force-cpu is available
    test = subprocess.run(['python', aiupred_script, '--help'],
                          capture_output=True, text=True)
    if '--force-cpu' not in test.stdout + test.stderr:
        print("  WARNING: --force-cpu flag not found in AIUPred help.")
        print("  Proceeding anyway — may fail if CUDA conflict is present.")
    else:
        print("  --force-cpu flag confirmed ✓")

    # --- Load protein FASTA and manifest ------------------------------------
    prot_fa   = proj / cfg['outputs']['all_protein_fa']
    manifest  = pd.read_csv(proj / cfg['outputs']['sequence_manifest'])

    print(f"\n[1/4] Input protein FASTA: {prot_fa}")
    print(f"  Manifest: {len(manifest):,} entries")

    if args.dry_run:
        print("\n[DRY RUN] Files located. Stopping before computation.")
        return 0

    # --- Run AIUPred ---------------------------------------------------------
    print(f"\n[2/4] Running AIUPred (--force-cpu, batch_size={args.batch_size})...")
    print(f"  This may take 5-15 minutes for ~1,500 proteins...")
    aiupred_results = run_aiupred_batch(
        str(prot_fa), aiupred_script, batch_size=args.batch_size
    )
    print(f"\n  AIUPred completed: {len(aiupred_results):,} proteins processed")

    # --- Compute IDR features ------------------------------------------------
    print(f"\n[3/4] Computing IDR features and assigning BioEmu tiers...")
    records = parse_fasta(str(prot_fa))
    rows = []

    for header, seq in records:
        gene_id, tx_id, cls, condition, qc_fail = \
            extract_gene_tx_from_header(header)

        scores = aiupred_results.get(tx_id, None)
        if scores is not None:
            feats = compute_idr_features(scores)
            feats['aiupred_status'] = 'computed'
        else:
            # Not computed (QC fail or batch error) — use protein length only
            feats = compute_idr_features([])
            feats['protein_length'] = len(seq)
            feats['aiupred_status'] = 'skipped_qc_fail' if qc_fail else 'failed'

        tier = assign_tier(feats['idr_percent'], cfg)

        rows.append({
            'geneID':               gene_id,
            'transcript_id':        tx_id,
            'class':                cls,
            'condition':            condition,
            'qc_fail':              qc_fail,
            **feats,
            'bioemu_tier':          tier,
            'bioemu_n_conf':        (
                0                                       if tier == 1 else
                cfg['bioemu_tiers']['tier2_n_conformations'] if tier == 2 else
                cfg['bioemu_tiers']['tier3_n_conformations']
            ),
        })

    df = pd.DataFrame(rows)

    # --- Tier summary --------------------------------------------------------
    tier_counts = df['bioemu_tier'].value_counts().sort_index()
    print(f"\n  BioEmu tier assignments:")
    tier_labels = {
        1: f"IDR < {cfg['bioemu_tiers']['tier1_max_idr']}%  → AlphaFold2 only",
        2: f"IDR {cfg['bioemu_tiers']['tier1_max_idr']}-"
           f"{cfg['bioemu_tiers']['tier2_max_idr']}% → BioEmu "
           f"{cfg['bioemu_tiers']['tier2_n_conformations']} conf",
        3: f"IDR >= {cfg['bioemu_tiers']['tier2_max_idr']}%   → BioEmu "
           f"{cfg['bioemu_tiers']['tier3_n_conformations']} conf",
    }
    total_bioemu = 0
    for tier, count in tier_counts.items():
        print(f"    Tier {tier} ({tier_labels[tier]}): {count:,}")
        if tier > 1:
            n_conf = (cfg['bioemu_tiers']['tier2_n_conformations'] if tier == 2
                      else cfg['bioemu_tiers']['tier3_n_conformations'])
            total_bioemu += count * n_conf

    print(f"\n  Total BioEmu conformations to generate: ~{total_bioemu:,}")

    # Top IDR proteins
    print(f"\n  Top 10 highest IDR% (positives):")
    top = df[(df['class']==1) & (df['aiupred_status']=='computed')]\
        .nlargest(10, 'idr_percent')
    for _, r in top.iterrows():
        print(f"    {r['geneID']:<14} {r['transcript_id']:<18} "
              f"IDR={r['idr_percent']:5.1f}%  Tier={r['bioemu_tier']}  "
              f"len={r['protein_length']} aa")

    # --- Write outputs -------------------------------------------------------
    print(f"\n[4/4] Writing outputs...")
    out_idr     = rp(cfg, 'aiupred_results')
    out_tiers   = rp(cfg, 'bioemu_tier_list')
    out_report  = rp(cfg, 'aiupred_report')

    df.to_csv(out_idr, index=False)
    print(f"  IDR summary: {out_idr}")

    # Tier list — just the columns needed for BioEmu batching
    tier_cols = ['geneID', 'transcript_id', 'class', 'condition',
                 'protein_length', 'idr_percent', 'bioemu_tier',
                 'bioemu_n_conf', 'aiupred_status']
    df[tier_cols].to_csv(out_tiers, index=False)
    print(f"  Tier list: {out_tiers}")

    # Write report
    _write_report(df, cfg, tier_counts, total_bioemu, str(out_report))

    print(f"\n{'='*70}")
    print("AIUPred / TIER SUMMARY")
    print(f"{'='*70}")
    print(f"  Proteins processed  : {(df['aiupred_status']=='computed').sum():,}")
    print(f"  Skipped / failed    : {(df['aiupred_status']!='computed').sum():,}")
    print(f"  BioEmu runs needed  : "
          f"{(df['bioemu_tier']>1).sum():,} proteins")
    print(f"  Est. conformations  : ~{total_bioemu:,}")
    print(f"{'='*70}")
    print("\n✓ Phase 5 complete.")
    return 0


def _write_report(df, cfg, tier_counts, total_bioemu, out_path):
    t1 = cfg['bioemu_tiers']['tier1_max_idr']
    t2 = cfg['bioemu_tiers']['tier2_max_idr']
    lines = [
        "=" * 80,
        "ZORC — AIUPred IDR Prediction Report",
        "Script: 05_aiupred_idr.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
        "PARAMETERS",
        f"  IDR threshold (score > 0.5) : AIUPred default",
        f"  Tier 1 (AlphaFold2 only)    : IDR% < {t1}",
        f"  Tier 2 (BioEmu 50 conf)     : IDR% {t1} - {t2}",
        f"  Tier 3 (BioEmu 100 conf)    : IDR% >= {t2}",
        "",
        "RESULTS",
        f"  Total proteins              : {len(df):,}",
        f"  AIUPred computed            : {(df['aiupred_status']=='computed').sum():,}",
        f"  Skipped/failed              : {(df['aiupred_status']!='computed').sum():,}",
        "",
        "TIER BREAKDOWN",
    ]
    for tier, count in tier_counts.items():
        lines.append(f"  Tier {tier}: {count:,} proteins")
    lines += [
        f"  Total BioEmu conformations  : ~{total_bioemu:,}",
        "",
        "IDR% DISTRIBUTION",
    ]
    for cls, label in [(1,'Positives'),(0,'Negatives')]:
        sub = df[(df['class']==cls) & (df['aiupred_status']=='computed')]['idr_percent']
        if len(sub):
            lines.append(
                f"  {label}: median={sub.median():.1f}%  "
                f"mean={sub.mean():.1f}%  "
                f"min={sub.min():.1f}%  max={sub.max():.1f}%"
            )
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
