#!/usr/bin/env python3
"""
06b_bioemu_tier1.py
===================
ZORC — Zip-code Of RNAs that Condense
Phase 6b: BioEmu conformational ensemble generation for Tier 1 proteins.

Tier 1 = IDR < 10% (structured proteins). These were skipped in P6 because
static AF2 features were assumed sufficient. After AF2 recovery (P9c) and
XGBoost SHAP analysis, rg_mean entered the Top 20 — but rmsf_* remain
suppressed because AF2 gives static (zero-variance) conformations.

This script runs BioEmu on 698 Tier 1 proteins with 100 conformations each,
producing real RMSF and Rg ensemble features to replace the AF2 static values.

Key differences from 06_bioemu_batch.py:
  - Tier 1 only (IDR < 10%) — no --tier argument needed
  - Separate checkpoint: data/processed/bioemu/checkpoint_tier1.json
    (preserves the original P6 checkpoint intact)
  - n_conformations read from config key: bioemu_tiers.tier1_n_conformations
    (default 100 — same as Tier 3, sufficient for structured protein ensembles)
  - Default timeout: 10800 s (3h) — same as P6 large-protein fix
  - Outputs go to data/processed/bioemu/<GeneID>_<TranscriptID>/
    (same structure as P6 — P7 finds them automatically)

Runtime estimate (RTX A5000):
  698 proteins × ~6-9 min/protein = ~70-105 h (~3-4 days)
  Comfortable within a 6-7 day continuous GPU window.

Usage:
    conda activate bioemu
    cd ~/Documents/ZORC

    # Test with first 3 proteins before full run
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml --limit 3

    # Check status of ongoing run
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml --status

    # Full run (leave running unattended)
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml

    # Resume after interruption (completed proteins skipped automatically)
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml

    # Retry failed proteins only
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml --retry-failed

    # If many proteins fail with unphysical_filter_bug, add --no-filter
    python scripts/06b_bioemu_tier1.py --config config/zorc_config.yaml --no-filter --retry-failed

After completion:
    Rerun P7 (07_protein_features.py) to extract BioEmu features for Tier 1,
    then P8 (08_feature_matrix.py) to rebuild the feature matrix,
    then P9/P9b for updated model evaluation.

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import json
import os
import re
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


def ensure_config_keys(cfg, config_path):
    """Add tier1_n_conformations and checkpoint_tier1 to config if absent."""
    changed = False

    # tier1_n_conformations — default 100 (same as Tier 3)
    if 'tier1_n_conformations' not in cfg.get('bioemu_tiers', {}):
        cfg.setdefault('bioemu_tiers', {})['tier1_n_conformations'] = 100
        changed = True
        print(f"  [config] Added bioemu_tiers.tier1_n_conformations = 100")

    # Separate checkpoint path for Tier 1
    ck_key = 'bioemu_checkpoint_tier1'
    if ck_key not in cfg.get('outputs', {}):
        cfg.setdefault('outputs', {})[ck_key] = \
            'data/processed/bioemu/checkpoint_tier1.json'
        changed = True
        print(f"  [config] Added outputs.bioemu_checkpoint_tier1")

    # Tier 1 report
    rpt_key = 'bioemu_report_tier1'
    if rpt_key not in cfg.get('outputs', {}):
        cfg.setdefault('outputs', {})[rpt_key] = \
            'logs/06b_bioemu_tier1_report.txt'
        changed = True
        print(f"  [config] Added outputs.bioemu_report_tier1")

    if changed:
        save_config(cfg, config_path)
    return cfg


def rp(cfg, key):
    proj = Path(os.path.expanduser(cfg['project_dir']))
    return proj / cfg['outputs'][key]


# =============================================================================
# FASTA UTILITIES  (identical to P6)
# =============================================================================

def parse_fasta(path):
    """Return dict: transcript_id → sequence."""
    seqs = {}
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header:
                    seqs[header] = ''.join(seq)
                header = line[1:].split('|')[1] if '|' in line \
                         else line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if header:
        seqs[header] = ''.join(seq)
    return seqs


def write_single_fasta(gene_id, tx_id, sequence, out_path):
    with open(out_path, 'w') as f:
        f.write(f'>{gene_id}_{tx_id}\n')
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + '\n')


# =============================================================================
# CHECKPOINT MANAGEMENT  (identical to P6 — but different file)
# =============================================================================

def load_checkpoint(checkpoint_path):
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
# BIOEMU RUNNER  (identical to P6)
# =============================================================================

def run_bioemu_single(gene_id: str, tx_id: str, sequence: str,
                      n_samples: int, out_dir: Path,
                      filter_samples: bool = True,
                      timeout_sec: int = 10800) -> dict:
    """
    Run BioEmu for a single protein.
    Returns dict: success, n_conformations, runtime_sec, error_msg
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = out_dir / 'input.fasta'
    write_single_fasta(gene_id, tx_id, sequence, str(fasta_path))

    cmd = [
        sys.executable, '-m', 'bioemu.sample',
        '--sequence',    str(fasta_path),
        '--num_samples', str(n_samples),
        '--output_dir',  str(out_dir),
    ]
    if not filter_samples:
        cmd.append('--filter_samples=False')

    log_path = out_dir / 'run.log'
    t_start  = time.time()

    try:
        with open(log_path, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
        runtime = time.time() - t_start

        if result.returncode != 0:
            error_msg = f'exit_code={result.returncode}'
            try:
                log_content = open(log_path).read()
                if "Invalid suffix '_unphysical.xtc'" in log_content:
                    error_msg = 'unphysical_filter_bug'
            except Exception:
                pass
            return {'success': False, 'n_conformations': 0,
                    'runtime_sec': runtime, 'error_msg': error_msg}

        topo = out_dir / 'topology.pdb'
        xtc  = out_dir / 'samples.xtc'
        if not topo.exists() or not xtc.exists():
            return {'success': False, 'n_conformations': 0,
                    'runtime_sec': runtime,
                    'error_msg': 'missing topology.pdb or samples.xtc'}

        n_conf = _count_conformations_from_log(log_path)
        return {'success': True, 'n_conformations': n_conf,
                'runtime_sec': round(runtime, 1), 'error_msg': None}

    except subprocess.TimeoutExpired:
        return {'success': False, 'n_conformations': 0,
                'runtime_sec': timeout_sec,
                'error_msg': f'timeout_{timeout_sec//3600}h'}
    except Exception as e:
        return {'success': False, 'n_conformations': 0,
                'runtime_sec': time.time() - t_start, 'error_msg': str(e)}


def _count_conformations_from_log(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with open(log_path) as f:
        content = f.read()
    m = re.search(r'Filtered \d+ samples down to (\d+)', content)
    if m:
        return int(m.group(1))
    if 'Completed' in content:
        return -1
    return 0


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

def format_eta(elapsed_sec, done, total):
    if done == 0:
        return "ETA: calculating..."
    rate = elapsed_sec / done
    remaining = rate * (total - done)
    eta = timedelta(seconds=int(remaining))
    return f"ETA: {eta} ({rate/60:.1f} min/protein)"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 6b — BioEmu for Tier 1 (IDR<10%) proteins.",
    )
    parser.add_argument('--config',       required=True)
    parser.add_argument('--limit',        type=int, default=None,
                        help='Process only first N proteins (for testing)')
    parser.add_argument('--no-filter',    action='store_true',
                        help='Pass --filter_samples=False to BioEmu '
                             '(use if unphysical_filter_bug errors appear)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously failed proteins')
    parser.add_argument('--error-filter', default=None,
                        help='Only retry failures matching this error string')
    parser.add_argument('--timeout',      type=int, default=10800,
                        help='Timeout per protein in seconds (default: 10800 = 3h)')
    parser.add_argument('--status',       action='store_true',
                        help='Show checkpoint status and exit')
    args = parser.parse_args()

    cfg  = load_config(args.config)
    cfg  = ensure_config_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    n_conf_tier1 = cfg['bioemu_tiers']['tier1_n_conformations']

    print("=" * 70)
    print("ZORC — Phase 6b: BioEmu Tier 1 (IDR<10%) Conformational Sampling")
    print(f"Config    : {args.config}")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N conf    : {n_conf_tier1} per protein")
    print(f"Timeout   : {args.timeout//3600}h per protein")
    print(f"Filter    : {'OFF (--no-filter)' if args.no_filter else 'ON (default)'}")
    print("=" * 70)

    # --- Load tier assignments ------------------------------------------------
    tier_path = proj / cfg['outputs']['bioemu_tier_list']
    df_tiers  = pd.read_csv(tier_path)
    if 'geneID' not in df_tiers.columns and 'gene_id' in df_tiers.columns:
        df_tiers = df_tiers.rename(columns={'gene_id': 'geneID'})

    df_work = df_tiers[
        (df_tiers['bioemu_tier'] == 1) &
        (df_tiers['aiupred_status'] == 'computed')
    ].copy()

    if args.limit:
        df_work = df_work.head(args.limit)
        print(f"\n  [--limit {args.limit}] Testing mode")

    print(f"\n  Tier 1 proteins (IDR<10%) : {len(df_work):,}")
    print(f"  Conformations per protein : {n_conf_tier1}")
    total_conf = len(df_work) * n_conf_tier1
    print(f"  Total conformations       : {total_conf:,}")
    est_days = len(df_work) * 8 / 60 / 24   # conservative 8 min/protein
    print(f"  Estimated runtime         : {est_days:.1f} days (conservative, "
          f"~8 min/protein on RTX A5000)")

    # --- Load checkpoint (TIER 1 SPECIFIC — does not touch P6 checkpoint) ---
    ckpt_path  = proj / cfg['outputs']['bioemu_checkpoint_tier1']
    checkpoint = load_checkpoint(str(ckpt_path))

    n_completed = sum(1 for v in checkpoint.values()
                      if v.get('status') == 'completed')
    n_failed    = sum(1 for v in checkpoint.values()
                      if v.get('status') == 'failed')
    n_pending   = len(df_work) - sum(
        1 for _, row in df_work.iterrows()
        if checkpoint.get(
            protein_key(row['geneID'], row['transcript_id']), {}
        ).get('status') == 'completed'
    )

    print(f"\n  Checkpoint: {ckpt_path.name}")
    print(f"    Completed : {n_completed:,}")
    print(f"    Failed    : {n_failed:,}")
    print(f"    Pending   : {n_pending:,}")

    if args.status:
        print("\n  [--status] Exiting.")
        return 0

    if n_pending == 0:
        print("\n  All Tier 1 proteins already completed.")
        print("  Use --retry-failed to retry failures.")
        return 0

    # --- Load protein sequences ----------------------------------------------
    prot_fa_path = proj / cfg['outputs']['all_protein_fa']
    print(f"\n  Loading protein sequences: {prot_fa_path}")
    prot_seqs = parse_fasta(str(prot_fa_path))
    print(f"  → {len(prot_seqs):,} sequences loaded")

    # --- BioEmu output directory (SAME as P6 — P7 finds outputs here) ------
    bioemu_dir = proj / cfg['outputs']['bioemu_dir']
    bioemu_dir.mkdir(parents=True, exist_ok=True)

    # --- Main loop -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Starting BioEmu Tier 1 runs — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    t_global_start        = time.time()
    n_done_this_session   = 0
    n_failed_this_session = 0
    results_log           = []

    for idx, (_, row) in enumerate(df_work.iterrows()):
        gene_id = row['geneID']
        tx_id   = str(row['transcript_id'])
        pkey    = protein_key(gene_id, tx_id)

        # Skip completed
        if checkpoint.get(pkey, {}).get('status') == 'completed':
            continue

        # Skip failed unless --retry-failed
        if checkpoint.get(pkey, {}).get('status') == 'failed' \
                and not args.retry_failed:
            continue

        # --error-filter: only retry failures matching the string
        if (args.retry_failed and args.error_filter and
                checkpoint.get(pkey, {}).get('status') == 'failed'):
            err = checkpoint.get(pkey, {}).get('error', '')
            if args.error_filter not in err:
                continue

        # Get sequence
        seq = prot_seqs.get(tx_id)
        if not seq:
            print(f"  [{idx+1}/{len(df_work)}] SKIP {gene_id} ({tx_id}) "
                  f"— sequence not found")
            checkpoint[pkey] = {
                'status':    'failed',
                'error':     'sequence_not_found',
                'timestamp': datetime.now().isoformat(),
            }
            save_checkpoint(checkpoint, str(ckpt_path))
            continue

        out_dir    = bioemu_dir / pkey
        elapsed    = time.time() - t_global_start
        eta_str    = format_eta(elapsed, n_done_this_session, n_pending)
        idr_val    = row.get('idr_percent', 0.0)

        print(f"  [{idx+1}/{len(df_work)}] {gene_id} | {tx_id} | "
              f"Tier1 | {len(seq)} aa | {n_conf_tier1} samples")
        print(f"         IDR={idr_val:.1f}%  {eta_str}")

        # Mark as running
        checkpoint[pkey] = {
            'status':         'running',
            'timestamp':      datetime.now().isoformat(),
            'tier':           1,
            'protein_length': len(seq),
            'idr_percent':    float(idr_val),
        }
        save_checkpoint(checkpoint, str(ckpt_path))

        # Run BioEmu
        run_result = run_bioemu_single(
            gene_id, tx_id, seq, n_conf_tier1, out_dir,
            filter_samples=not args.no_filter,
            timeout_sec=args.timeout,
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
            err = run_result['error_msg']
            print(f"         ✗ FAILED: {err}")
            if err == 'unphysical_filter_bug':
                print(f"           → Re-run with --no-filter --retry-failed "
                      f"to recover this protein")

        results_log.append({
            'geneID':              gene_id,
            'transcript_id':       tx_id,
            'tier':                1,
            'protein_length':      len(seq),
            'idr_percent':         float(idr_val),
            'n_samples_requested': n_conf_tier1,
            'n_conformations':     run_result['n_conformations'],
            'runtime_sec':         run_result['runtime_sec'],
            'status':              status,
        })

    # --- Session summary -----------------------------------------------------
    total_elapsed     = time.time() - t_global_start
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
        # Project remaining time
        remaining = avg * (n_pending - n_done_this_session)
        if remaining > 0:
            print(f"    Est. remaining: {timedelta(seconds=int(remaining))}")
    print(f"\n  Cumulative (all sessions):")
    print(f"    Completed  : {n_total_completed:,} / {len(df_work)}")
    print(f"    Failed     : {n_total_failed:,}")
    print(f"    Remaining  : {len(df_work) - n_total_completed - n_total_failed:,}")
    print(f"{'='*70}")

    if results_log:
        _write_session_report(results_log, cfg, proj, total_elapsed,
                              n_conf_tier1)

    print("\n✓ Phase 6b session complete.")
    print("  Re-run the same command to continue (completed proteins skipped).")
    print("\n  After full completion:")
    print("    conda activate zorc_pipeline")
    print("    python scripts/07_protein_features.py --config config/zorc_config.yaml")
    print("    python scripts/08_feature_matrix.py   --config config/zorc_config.yaml")
    print("    python scripts/09b_xgboost.py         --config config/zorc_config.yaml")
    return 0


def _write_session_report(results_log, cfg, proj, elapsed, n_conf_tier1):
    report_path = proj / cfg['outputs']['bioemu_report_tier1']
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df    = pd.DataFrame(results_log)
    n_ok  = (df['status'] == 'completed').sum()
    n_fail = (df['status'] == 'failed').sum()
    avg_rt = df.loc[df['status'] == 'completed', 'runtime_sec'].mean() / 60 \
             if n_ok > 0 else 0

    lines = [
        "=" * 80,
        "ZORC — BioEmu Tier 1 Batch Run Report",
        "Script: 06b_bioemu_tier1.py",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"N conf    : {n_conf_tier1} per protein",
        "=" * 80, "",
        f"Session results : {n_ok} completed, {n_fail} failed",
        f"Total runtime   : {timedelta(seconds=int(elapsed))}",
        f"Avg/protein     : {avg_rt:.1f} min",
        "",
        "Per-protein results:",
        f"{'GeneID':<14} {'TxID':<20} {'Len':>5} {'IDR%':>5} "
        f"{'Req':>5} {'Got':>5} {'Min':>6}  {'Status'}",
        "-" * 80,
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['geneID']:<14} {str(r['transcript_id']):<20} "
            f"{r['protein_length']:>5} {r['idr_percent']:>4.1f}% "
            f"{r['n_samples_requested']:>5} {r['n_conformations']:>5} "
            f"{r['runtime_sec']/60:>5.1f}  {r['status']}"
        )
    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    with open(report_path, 'a') as f:   # append across sessions
        f.write('\n'.join(lines) + '\n\n')
    print(f"\n  Session report appended to: {report_path}")


if __name__ == '__main__':
    sys.exit(main())
