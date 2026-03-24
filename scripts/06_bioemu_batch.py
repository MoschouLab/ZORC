#!/usr/bin/env python3
"""
06_bioemu_batch.py
==================
ZORC — Zip-code Of RNAs that Condense
Phase 6: BioEmu conformational ensemble generation.

Runs BioEmu (python -m bioemu.sample) on all Tier 2 and/or Tier 3 proteins
from the AIUPred tier assignment (P5). One protein per run, with automatic
checkpointing to resume interrupted jobs.

Run order for ZORC:
  First:  --tier 2  (328 proteins, ~50 conf each, ~3-4 days)  ← validate
  Second: --tier 3  (431 proteins, ~100 conf each, ~6-8 days)

Output structure per protein:
  data/processed/bioemu/<GeneID>_<TranscriptID>/
    input.fasta        ← protein sequence used
    topology.pdb       ← reference structure
    samples.xtc        ← conformational ensemble
    run.log            ← BioEmu stdout/stderr

Checkpointing:
  data/processed/bioemu/checkpoint.json
  Tracks status per protein: pending / running / completed / failed
  Resume by re-running the same command — completed proteins are skipped.

Requires: bioemu conda env (conda activate bioemu before running)

Usage:
    conda activate bioemu
    cd ~/Documents/ZORC

    # First run — Tier 2 only
    python scripts/06_bioemu_batch.py --config config/zorc_config.yaml --tier 2

    # After validation, Tier 3
    python scripts/06_bioemu_batch.py --config config/zorc_config.yaml --tier 3

    # Both tiers (full run)
    python scripts/06_bioemu_batch.py --config config/zorc_config.yaml --tier 2,3

    # Test with first 5 proteins
    python scripts/06_bioemu_batch.py --config config/zorc_config.yaml --tier 2 --limit 5

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
        'bioemu_dir':        'data/processed/bioemu',
        'bioemu_checkpoint': 'data/processed/bioemu/checkpoint.json',
        'bioemu_report':     'logs/06_bioemu_report.txt',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P6 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    proj = Path(os.path.expanduser(cfg['project_dir']))
    return proj / cfg['outputs'][key]


# =============================================================================
# FASTA UTILITIES
# =============================================================================

def parse_fasta(path):
    """Return dict: header_first_token → sequence."""
    seqs = {}
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header:
                    seqs[header] = ''.join(seq)
                # First token of header = transcript_id
                header = line[1:].split('|')[1] if '|' in line else line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if header:
        seqs[header] = ''.join(seq)
    return seqs


def write_single_fasta(gene_id, tx_id, sequence, out_path):
    """Write a single-sequence FASTA for BioEmu input."""
    with open(out_path, 'w') as f:
        f.write(f'>{gene_id}_{tx_id}\n')
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + '\n')


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_checkpoint(checkpoint_path):
    """Load checkpoint JSON. Returns dict: protein_key → status dict."""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint, checkpoint_path):
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def protein_key(gene_id, tx_id):
    return f"{gene_id}_{tx_id}"


# =============================================================================
# BIOEMU RUNNER
# =============================================================================

def run_bioemu_single(gene_id: str, tx_id: str, sequence: str,
                      n_samples: int, out_dir: Path,
                      filter_samples: bool = True) -> dict:
    """
    Run BioEmu for a single protein.

    Returns dict with keys:
      success, n_conformations, runtime_sec, error_msg
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write input FASTA
    fasta_path = out_dir / 'input.fasta'
    write_single_fasta(gene_id, tx_id, sequence, str(fasta_path))

    # Build command
    cmd = [
        sys.executable, '-m', 'bioemu.sample',
        '--sequence', str(fasta_path),
        '--num_samples', str(n_samples),
        '--output_dir', str(out_dir),
    ]
    if not filter_samples:
        cmd.append('--filter_samples=False')

    log_path = out_dir / 'run.log'
    t_start = time.time()

    try:
        with open(log_path, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour max per protein
            )
        runtime = time.time() - t_start

        if result.returncode != 0:
            return {
                'success': False,
                'n_conformations': 0,
                'runtime_sec': runtime,
                'error_msg': f'exit_code={result.returncode}',
            }

        # Verify output files exist
        topo = out_dir / 'topology.pdb'
        xtc  = out_dir / 'samples.xtc'
        if not topo.exists() or not xtc.exists():
            return {
                'success': False,
                'n_conformations': 0,
                'runtime_sec': runtime,
                'error_msg': 'missing topology.pdb or samples.xtc',
            }

        # Count conformations from log
        n_conf = _count_conformations_from_log(log_path)

        return {
            'success': True,
            'n_conformations': n_conf,
            'runtime_sec': round(runtime, 1),
            'error_msg': None,
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'n_conformations': 0,
            'runtime_sec': 3600,
            'error_msg': 'timeout_1h',
        }
    except Exception as e:
        return {
            'success': False,
            'n_conformations': 0,
            'runtime_sec': time.time() - t_start,
            'error_msg': str(e),
        }


def _count_conformations_from_log(log_path: Path) -> int:
    """
    Parse BioEmu log to count final conformations after filtering.
    Looks for: "Filtered N samples down to M based on structure criteria"
    or falls back to counting frames from sampling progress.
    """
    if not log_path.exists():
        return 0
    with open(log_path) as f:
        content = f.read()

    # Pattern: "Filtered 100 samples down to 22 based on structure criteria"
    import re
    m = re.search(r'Filtered \d+ samples down to (\d+)', content)
    if m:
        return int(m.group(1))

    # Pattern: "Completed. Your samples are in ..."
    # If no filtering line, use requested samples as estimate
    if 'Completed' in content:
        return -1  # unknown but completed

    return 0


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

def format_eta(elapsed_sec, done, total):
    if done == 0:
        return "ETA: calculating..."
    rate = elapsed_sec / done  # sec per protein
    remaining = rate * (total - done)
    eta = timedelta(seconds=int(remaining))
    return f"ETA: {eta} ({rate/60:.1f} min/protein)"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 6 — BioEmu conformational ensemble generation.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--tier', default='2',
                        help='Tier(s) to process: "2", "3", or "2,3" (default: 2)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N proteins (for testing)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Pass --filter_samples=False to BioEmu (use for IDR>90%%)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously failed proteins')
    parser.add_argument('--status', action='store_true',
                        help='Show checkpoint status and exit')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 6: BioEmu Conformational Sampling")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load tier assignments -----------------------------------------------
    tier_path = proj / cfg['outputs']['bioemu_tier_list']
    df_tiers  = pd.read_csv(tier_path)

    # Parse requested tiers
    tiers = [int(t.strip()) for t in args.tier.split(',')]
    df_work = df_tiers[
        (df_tiers['bioemu_tier'].isin(tiers)) &
        (df_tiers['aiupred_status'] == 'computed')
    ].copy()

    if args.limit:
        df_work = df_work.head(args.limit)
        print(f"\n  [--limit {args.limit}] Testing mode")

    print(f"\n  Tiers requested : {tiers}")
    print(f"  Proteins to run : {len(df_work):,}")
    for tier in tiers:
        sub = df_work[df_work['bioemu_tier'] == tier]
        n_conf = cfg['bioemu_tiers'][f'tier{tier}_n_conformations'] if tier <= 2 \
                 else cfg['bioemu_tiers']['tier3_n_conformations']
        print(f"    Tier {tier}: {len(sub):,} proteins × {n_conf} conf")

    # --- Load checkpoint -----------------------------------------------------
    ckpt_path = proj / cfg['outputs']['bioemu_checkpoint']
    checkpoint = load_checkpoint(str(ckpt_path))

    # Status report
    n_completed = sum(1 for v in checkpoint.values() if v.get('status') == 'completed')
    n_failed    = sum(1 for v in checkpoint.values() if v.get('status') == 'failed')
    n_pending   = len(df_work) - sum(
        1 for _, row in df_work.iterrows()
        if checkpoint.get(protein_key(row['geneID'], row['transcript_id']),
                          {}).get('status') == 'completed'
    )

    print(f"\n  Checkpoint status:")
    print(f"    Completed : {n_completed:,}")
    print(f"    Failed    : {n_failed:,}")
    print(f"    Pending   : {n_pending:,}")

    if args.status:
        print("\n  [--status] Exiting.")
        return 0

    if n_pending == 0:
        print("\n  All proteins already completed. Use --retry-failed to retry failures.")
        return 0

    # --- Load protein sequences ----------------------------------------------
    prot_fa_path = proj / cfg['outputs']['all_protein_fa']
    print(f"\n  Loading protein sequences from: {prot_fa_path}")
    prot_seqs = parse_fasta(str(prot_fa_path))
    print(f"  → {len(prot_seqs):,} sequences loaded")

    # --- BioEmu output directory ---------------------------------------------
    bioemu_dir = proj / cfg['outputs']['bioemu_dir']
    bioemu_dir.mkdir(parents=True, exist_ok=True)

    # --- Main loop -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Starting BioEmu runs — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    t_global_start = time.time()
    n_done_this_session = 0
    n_failed_this_session = 0
    results_log = []

    for idx, (_, row) in enumerate(df_work.iterrows()):
        gene_id = row['geneID']
        tx_id   = str(row['transcript_id'])
        tier    = int(row['bioemu_tier'])
        pkey    = protein_key(gene_id, tx_id)

        # Skip if already completed
        if checkpoint.get(pkey, {}).get('status') == 'completed':
            continue

        # Skip failed unless --retry-failed
        if checkpoint.get(pkey, {}).get('status') == 'failed' and not args.retry_failed:
            continue

        # Get sequence
        seq = prot_seqs.get(tx_id)
        if not seq:
            print(f"  [{idx+1}/{len(df_work)}] SKIP {gene_id} ({tx_id}) — sequence not found")
            checkpoint[pkey] = {
                'status': 'failed',
                'error': 'sequence_not_found',
                'timestamp': datetime.now().isoformat(),
            }
            save_checkpoint(checkpoint, str(ckpt_path))
            continue

        # Determine n_samples
        if tier == 2:
            n_samples = cfg['bioemu_tiers']['tier2_n_conformations']
        else:
            n_samples = cfg['bioemu_tiers']['tier3_n_conformations']

        out_dir = bioemu_dir / pkey
        elapsed = time.time() - t_global_start
        eta_str = format_eta(elapsed, n_done_this_session, n_pending)

        print(f"  [{idx+1}/{len(df_work)}] {gene_id} | {tx_id} | "
              f"Tier{tier} | {len(seq)} aa | {n_samples} samples")
        print(f"         IDR={row['idr_percent']:.1f}%  {eta_str}")

        # Mark as running
        checkpoint[pkey] = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'tier': tier,
            'protein_length': len(seq),
            'idr_percent': float(row['idr_percent']),
        }
        save_checkpoint(checkpoint, str(ckpt_path))

        # Run BioEmu
        run_result = run_bioemu_single(
            gene_id, tx_id, seq, n_samples, out_dir,
            filter_samples=not args.no_filter
        )

        # Update checkpoint
        status = 'completed' if run_result['success'] else 'failed'
        checkpoint[pkey].update({
            'status':          status,
            'n_conformations': run_result['n_conformations'],
            'runtime_sec':     run_result['runtime_sec'],
            'error':           run_result['error_msg'],
            'completed_at':    datetime.now().isoformat(),
        })
        save_checkpoint(checkpoint, str(ckpt_path))

        if run_result['success']:
            n_done_this_session += 1
            print(f"         ✓ {run_result['n_conformations']} conformations "
                  f"in {run_result['runtime_sec']/60:.1f} min")
        else:
            n_failed_this_session += 1
            print(f"         ✗ FAILED: {run_result['error_msg']}")

        results_log.append({
            'geneID':          gene_id,
            'transcript_id':   tx_id,
            'tier':            tier,
            'protein_length':  len(seq),
            'idr_percent':     row['idr_percent'],
            'n_samples_requested': n_samples,
            'n_conformations': run_result['n_conformations'],
            'runtime_sec':     run_result['runtime_sec'],
            'status':          status,
        })

    # --- Session summary -----------------------------------------------------
    total_elapsed = time.time() - t_global_start
    n_total_completed = sum(1 for v in checkpoint.values()
                            if v.get('status') == 'completed')
    n_total_failed    = sum(1 for v in checkpoint.values()
                            if v.get('status') == 'failed')

    print(f"\n{'='*70}")
    print("SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"  This session:")
    print(f"    Completed  : {n_done_this_session:,}")
    print(f"    Failed     : {n_failed_this_session:,}")
    print(f"    Runtime    : {timedelta(seconds=int(total_elapsed))}")
    if n_done_this_session > 0:
        avg = total_elapsed / n_done_this_session
        print(f"    Avg/protein: {avg/60:.1f} min")
    print(f"\n  Cumulative (all sessions):")
    print(f"    Completed  : {n_total_completed:,}")
    print(f"    Failed     : {n_total_failed:,}")
    print(f"    Remaining  : {len(df_work) - n_total_completed - n_total_failed:,}")
    print(f"{'='*70}")

    # Write session log
    if results_log:
        _write_session_report(results_log, cfg, proj, total_elapsed, tiers)

    print("\n✓ Phase 6 session complete.")
    print("  Re-run the same command to continue (completed proteins skipped).")
    return 0


def _write_session_report(results_log, cfg, proj, elapsed, tiers):
    report_path = proj / cfg['outputs']['bioemu_report']
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_log)
    n_ok   = (df['status'] == 'completed').sum()
    n_fail = (df['status'] == 'failed').sum()

    lines = [
        "=" * 80,
        "ZORC — BioEmu Batch Run Report",
        "Script: 06_bioemu_batch.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tiers processed: {tiers}",
        "=" * 80, "",
        f"Session results: {n_ok} completed, {n_fail} failed",
        f"Total runtime  : {timedelta(seconds=int(elapsed))}",
        "",
        "Per-protein results:",
        f"{'GeneID':<14} {'TxID':<20} {'Tier'} {'Len':>5} {'IDR%':>6} "
        f"{'Req':>5} {'Got':>5} {'Min':>6} {'Status'}",
        "-" * 80,
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['geneID']:<14} {str(r['transcript_id']):<20} "
            f"{r['tier']:>4}  {r['protein_length']:>5} {r['idr_percent']:>5.1f}% "
            f"{r['n_samples_requested']:>5} {r['n_conformations']:>5} "
            f"{r['runtime_sec']/60:>5.1f}  {r['status']}"
        )
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    with open(report_path, 'a') as f:  # append — multiple sessions
        f.write('\n'.join(lines) + '\n\n')
    print(f"\n  Session report appended to: {report_path}")


if __name__ == '__main__':
    sys.exit(main())
