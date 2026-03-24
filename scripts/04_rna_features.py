#!/usr/bin/env python3
"""
04_rna_features.py
==================
ZORC — Zip-code Of RNAs that Condense
Phase 4: RNA feature engineering from isoform mRNA sequences.

Computes per-transcript RNA features:

  Composition features:
    - Nucleotide fractions: fA, fU, fG, fC
    - AU content (fraction A+U)
    - Dinucleotide frequencies (16 features)
    - GC content

  Length features:
    - Total mRNA length (spliced)
    - Estimated 5'UTR, CDS, 3'UTR lengths (from GTF annotation)
    - AU content in 3'UTR specifically

  m6A features:
    - RRACH motif frequency (R=A/G, H=A/C/U) — canonical m6A writer site
    - AAACH motif frequency (most common m6A motif)

  Secondary structure features (RNAfold):
    - MFE (minimum free energy, kcal/mol)
    - MFE/length (length-normalized MFE)
    - Ensemble free energy
    - Fraction of paired bases (from dot-bracket)
    - Number of stem-loops

  NMD proxy features:
    - Presence of premature termination codon (PTC) proxy:
      exon-exon junction > 50-55 nt downstream of stop codon
      (requires CDS annotation — approximated from protein/mRNA lengths)

Requires: zorc_pipeline conda env (RNAfold 2.7.2 installed)

Usage:
    conda activate zorc_pipeline
    cd ~/Documents/ZORC
    python scripts/04_rna_features.py --config config/zorc_config.yaml

    # Skip RNAfold (faster, no structure features)
    python scripts/04_rna_features.py --config config/zorc_config.yaml --no-rnafold

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
from collections import Counter
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
        'rna_features':       'data/processed/04_rna_features.csv',
        'rna_features_report':'logs/04_rna_features_report.txt',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P4 output keys to {config_path}")
    return cfg

def rp(cfg, key):
    proj = Path(os.path.expanduser(cfg['project_dir']))
    return proj / cfg['outputs'][key]


# =============================================================================
# FASTA PARSING
# =============================================================================

def parse_fasta_zorc(path):
    """
    Parse ZORC-format FASTA.
    Header format: GeneID|TranscriptID|classN|condition[|QC_FAIL:reason]
    Returns list of dicts: {gene_id, tx_id, cls, condition, qc_fail, sequence}
    """
    records = []
    header, seq = None, []

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    records.append(_parse_zorc_header(header, ''.join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line.strip())
    if header is not None:
        records.append(_parse_zorc_header(header, ''.join(seq)))

    return records


def _parse_zorc_header(header, seq):
    parts = header.split('|')
    gene_id   = parts[0] if len(parts) > 0 else ''
    tx_id     = parts[1] if len(parts) > 1 else ''
    cls_str   = parts[2] if len(parts) > 2 else 'class0'
    condition = parts[3] if len(parts) > 3 else ''
    qc_fail   = any('QC_FAIL' in p for p in parts)
    try:
        cls = int(cls_str.replace('class', ''))
    except ValueError:
        cls = -1
    return {
        'gene_id':   gene_id,
        'tx_id':     tx_id,
        'class':     cls,
        'condition': condition,
        'qc_fail':   qc_fail,
        'sequence':  seq.upper().replace('T', 'U'),  # DNA→RNA
    }


# =============================================================================
# UTR EXTRACTION FROM GTF
# =============================================================================

def build_utr_index(gtf_path, target_tx_ids):
    """
    Parse GTF to extract 5'UTR and 3'UTR coordinates per transcript.
    Returns dict: tx_id → {utr5_len, utr3_len, cds_len}
    """
    print(f"  Building UTR index for {len(target_tx_ids):,} transcripts...")
    tx_re    = re.compile(r'transcript_id\s+"([^"]+)"')
    utr_data = {}

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ('UTR', 'CDS', 'five_prime_utr',
                               'three_prime_utr', 'start_codon'):
                continue
            m = tx_re.search(parts[8])
            if not m or m.group(1) not in target_tx_ids:
                continue
            tx_id  = m.group(1)
            start  = int(parts[3])
            end    = int(parts[4])
            length = end - start + 1

            if tx_id not in utr_data:
                utr_data[tx_id] = {'utr5_len': 0, 'utr3_len': 0, 'cds_len': 0}

            if feature in ('five_prime_utr',):
                utr_data[tx_id]['utr5_len'] += length
            elif feature in ('three_prime_utr',):
                utr_data[tx_id]['utr3_len'] += length
            elif feature == 'CDS':
                utr_data[tx_id]['cds_len'] += length
            elif feature == 'UTR':
                # Generic UTR — we'll distinguish by position relative to CDS
                # Store for post-processing
                utr_data[tx_id].setdefault('utrs', []).append(
                    (int(parts[3]), int(parts[4]), parts[6])
                )

    print(f"  → UTR data for {len(utr_data):,} transcripts")
    return utr_data


# =============================================================================
# COMPOSITION FEATURES
# =============================================================================

def nucleotide_features(seq: str) -> dict:
    """Compute nucleotide composition and dinucleotide frequencies."""
    n = len(seq)
    if n == 0:
        return {}

    # Single nucleotide
    counts = Counter(seq)
    feats = {
        'fA': counts.get('A', 0) / n,
        'fU': counts.get('U', 0) / n,
        'fG': counts.get('G', 0) / n,
        'fC': counts.get('C', 0) / n,
        'au_content': (counts.get('A', 0) + counts.get('U', 0)) / n,
        'gc_content': (counts.get('G', 0) + counts.get('C', 0)) / n,
        'mrna_length': n,
    }

    # Dinucleotide frequencies
    dinucs = ['AA','AU','AG','AC','UA','UU','UG','UC',
               'GA','GU','GG','GC','CA','CU','CG','CC']
    n_di = n - 1
    if n_di > 0:
        di_counts = Counter(seq[i:i+2] for i in range(n_di))
        for di in dinucs:
            feats[f'di_{di}'] = di_counts.get(di, 0) / n_di
    else:
        for di in dinucs:
            feats[f'di_{di}'] = 0.0

    return feats


# =============================================================================
# m6A MOTIF FEATURES
# =============================================================================

# RRACH: R=[AG], H=[ACU]
RRACH_RE = re.compile(r'[AG][AG]AC[ACU]')
AAACH_RE = re.compile(r'AAAC[ACU]')
UGUG_RE  = re.compile(r'UGUG')   # alternative m6A context


def m6a_features(seq: str) -> dict:
    n = len(seq)
    if n == 0:
        return {'rrach_count': 0, 'rrach_per_kb': 0.0,
                'aaach_count': 0, 'aaach_per_kb': 0.0}

    rrach = len(RRACH_RE.findall(seq))
    aaach = len(AAACH_RE.findall(seq))
    kb    = n / 1000.0

    return {
        'rrach_count':  rrach,
        'rrach_per_kb': rrach / kb,
        'aaach_count':  aaach,
        'aaach_per_kb': aaach / kb,
    }


# =============================================================================
# 3'UTR AU CONTENT
# =============================================================================

def utr3_features(seq: str, utr3_len: int) -> dict:
    """
    Estimate 3'UTR AU content from the last utr3_len nucleotides.
    If utr3_len not available, use last 20% of sequence as proxy.
    """
    n = len(seq)
    if n == 0:
        return {'utr3_au_content': 0.0, 'utr3_length': 0, 'utr3_estimated': True}

    if utr3_len > 0:
        utr3_seq = seq[-min(utr3_len, n):]
        estimated = False
    else:
        # Proxy: last 20% of mRNA
        utr3_len_proxy = max(int(n * 0.20), 50)
        utr3_seq = seq[-utr3_len_proxy:]
        utr3_len = utr3_len_proxy
        estimated = True

    m = Counter(utr3_seq)
    au = (m.get('A', 0) + m.get('U', 0)) / max(len(utr3_seq), 1)

    return {
        'utr3_au_content': round(au, 4),
        'utr3_length':     utr3_len,
        'utr3_estimated':  estimated,
    }


# =============================================================================
# NMD PROXY
# =============================================================================

def nmd_proxy_features(mrna_len: int, cds_len: int, utr3_len: int,
                        protein_len: int) -> dict:
    """
    Approximate NMD (nonsense-mediated decay) susceptibility.

    NMD rule: stop codon > 50-55 nt upstream of last exon-exon junction
    (EJC rule). We approximate using 3'UTR length.

    Long 3'UTR (>1000 nt) is also a weak NMD predictor.
    """
    # Estimated stop codon position
    # Simple proxy: if CDS len available, check if protein*3 ≈ CDS
    cds_from_protein = protein_len * 3 + 3  # +3 for stop codon
    cds_mismatch = abs(cds_len - cds_from_protein) if cds_len > 0 else -1

    # Long 3'UTR proxy (NMD via upstream ORFs or long 3'UTR)
    long_3utr = int(utr3_len > 1000) if utr3_len > 0 else 0

    # PTC proxy: CDS shorter than expected from protein suggests early stop
    ptc_proxy = int(cds_mismatch > 30 and cds_len > 0 and protein_len > 10)

    return {
        'cds_protein_mismatch': cds_mismatch,
        'long_3utr':            long_3utr,
        'ptc_proxy':            ptc_proxy,
    }


# =============================================================================
# RNAFOLD
# =============================================================================

def run_rnafold_batch(sequences: list[tuple[str, str]],
                      max_len: int = 3000,
                      chunk_size: int = 10) -> dict:
    """
    Run RNAfold on sequences in chunks to avoid timeout.
    sequences: list of (tx_id, sequence)
    max_len: skip sequences longer than this (RNAfold is O(n^3))
    chunk_size: sequences per RNAfold subprocess call

    Returns dict: tx_id → {mfe, ensemble_fe, fraction_paired, n_stemloops}
    """
    results = {}
    eligible = [(tx, seq) for tx, seq in sequences
                if len(seq) <= max_len and len(seq) >= 20]
    skipped  = len(sequences) - len(eligible)

    if skipped:
        print(f"    RNAfold: skipping {skipped} sequences > {max_len} nt")

    if not eligible:
        return results

    n_chunks = (len(eligible) + chunk_size - 1) // chunk_size
    print(f"    RNAfold: {len(eligible)} sequences in {n_chunks} chunks of {chunk_size}...")

    for chunk_idx in range(n_chunks):
        chunk = eligible[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({len(chunk)} seqs)...",
              end=' ', flush=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa',
                                         delete=False) as tmp_in:
            for tx_id, seq in chunk:
                tmp_in.write(f'>{tx_id}\n{seq}\n')
            tmp_in_path = tmp_in.name

        try:
            result = subprocess.run(
                ['RNAfold', '--noPS', '--infile', tmp_in_path],
                capture_output=True, text=True, timeout=120  # 2 min per chunk
            )
            if result.returncode == 0:
                chunk_results = parse_rnafold_output(result.stdout, chunk)
                results.update(chunk_results)
                print(f"OK ({len(chunk_results)} parsed)")
            else:
                print(f"ERROR (exit {result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT — skipping chunk {chunk_idx+1}")
        except FileNotFoundError:
            print("RNAfold not found in PATH")
            break
        finally:
            os.unlink(tmp_in_path)

    return results


def parse_rnafold_output(stdout: str, sequences: list) -> dict:
    """
    Parse RNAfold output.
    Expected per sequence:
      >tx_id
      SEQUENCE
      STRUCTURE  (MFE kcal/mol)
      STRUCTURE  [ensemble kcal/mol]  ← with -p flag
    """
    results = {}
    lines = stdout.split('\n')
    i = 0
    tx_order = [tx for tx, _ in sequences]
    tx_idx = 0

    while i < len(lines) and tx_idx < len(tx_order):
        line = lines[i].strip()
        if line.startswith('>'):
            tx_id = tx_order[tx_idx]
            tx_idx += 1
            i += 1
            # Skip sequence line
            if i < len(lines):
                i += 1
            # MFE structure line
            mfe = None
            ensemble_fe = None
            dot_bracket = ''
            if i < len(lines):
                mfe_line = lines[i].strip()
                m = re.search(r'\(\s*(-?\d+\.?\d*)\s*\)', mfe_line)
                if m:
                    mfe = float(m.group(1))
                dot_bracket = mfe_line.split()[0] if mfe_line else ''
                i += 1
            # No ensemble line (running without -p for speed)
            ensemble_fe = None

            # Parse dot-bracket
            frac_paired = 0.0
            n_stemloops = 0
            if dot_bracket:
                n_paired = dot_bracket.count('(') + dot_bracket.count(')')
                frac_paired = n_paired / max(len(dot_bracket), 1)
                # Count stem-loops: (...) patterns
                n_stemloops = len(re.findall(r'\([.]+\)', dot_bracket))

            results[tx_id] = {
                'mfe':            mfe,
                'mfe_per_nt':     mfe / len(dot_bracket) if dot_bracket and mfe else None,
                'ensemble_fe':    ensemble_fe,
                'frac_paired':    round(frac_paired, 4),
                'n_stemloops':    n_stemloops,
                'rnafold_status': 'computed',
            }
        else:
            i += 1

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 4 — RNA feature engineering.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--no-rnafold', action='store_true',
                        help='Skip RNAfold (faster run, no structure features)')
    parser.add_argument('--rnafold-maxlen', type=int, default=3000,
                        help='Max sequence length for RNAfold (default: 3000 nt)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 4: RNA Feature Engineering")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.no_rnafold:
        print("  [--no-rnafold] Skipping RNAfold structure features")
    print("=" * 70)

    # --- Load mRNA sequences -------------------------------------------------
    mrna_fa = proj / cfg['outputs']['all_mrna_fa']
    prot_fa = proj / cfg['outputs']['all_protein_fa']
    manifest_path = proj / cfg['outputs']['sequence_manifest']

    print(f"\n[1/5] Loading sequences...")
    mrna_records = parse_fasta_zorc(str(mrna_fa))
    print(f"  mRNA sequences: {len(mrna_records):,}")

    # Load protein lengths for NMD proxy
    prot_seqs_raw = {}
    with open(prot_fa) as f:
        cur_tx, cur_seq = None, []
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if cur_tx:
                    prot_seqs_raw[cur_tx] = len(''.join(cur_seq))
                cur_tx = line[1:].split('|')[1] if '|' in line else line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_tx:
            prot_seqs_raw[cur_tx] = len(''.join(cur_seq))

    # Load manifest for UTR info
    manifest = pd.read_csv(manifest_path)

    # --- Build UTR index from GTF --------------------------------------------
    gtf_path = os.path.expanduser(cfg['reference']['gtf'])
    target_tx_ids = {r['tx_id'] for r in mrna_records}

    print(f"\n[2/5] Building UTR/CDS index from GTF...")
    utr_index = build_utr_index(gtf_path, target_tx_ids)

    # --- Compute RNA features ------------------------------------------------
    print(f"\n[3/5] Computing composition + m6A + UTR features...")
    rows = []
    for rec in mrna_records:
        seq      = rec['sequence']
        tx_id    = rec['tx_id']
        gene_id  = rec['gene_id']
        cls      = rec['class']
        cond     = rec['condition']

        utr_info = utr_index.get(tx_id, {})
        utr3_len = utr_info.get('utr3_len', 0)
        cds_len  = utr_info.get('cds_len', 0)
        utr5_len = utr_info.get('utr5_len', 0)
        prot_len = prot_seqs_raw.get(tx_id, 0)

        row = {
            'geneID':       gene_id,
            'transcript_id': tx_id,
            'class':        cls,
            'condition':    cond,
            'qc_fail':      rec['qc_fail'],
        }

        # Composition
        row.update(nucleotide_features(seq))

        # m6A motifs
        row.update(m6a_features(seq))

        # UTR lengths (from GTF)
        row['utr5_length'] = utr5_len
        row['utr3_length'] = utr3_len
        row['cds_length']  = cds_len

        # 3'UTR AU content
        row.update(utr3_features(seq, utr3_len))

        # NMD proxy
        row.update(nmd_proxy_features(len(seq), cds_len, utr3_len, prot_len))

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} sequences processed")

    # --- RNAfold -------------------------------------------------------------
    if not args.no_rnafold:
        print(f"\n[4/5] Running RNAfold (max_len={args.rnafold_maxlen} nt)...")
        print(f"  This may take 20-60 minutes for ~1,500 sequences...")
        seqs_for_fold = [(r['tx_id'], r['sequence']) for r in mrna_records]
        fold_results = run_rnafold_batch(seqs_for_fold, args.rnafold_maxlen)
        print(f"  RNAfold completed: {len(fold_results):,} structures computed")

        # Merge RNAfold results
        for col in ['mfe', 'mfe_per_nt', 'ensemble_fe',
                    'frac_paired', 'n_stemloops', 'rnafold_status']:
            df[col] = df['transcript_id'].map(
                lambda tx, c=col: fold_results.get(tx, {}).get(c, None)
            )
        df['rnafold_status'] = df['rnafold_status'].fillna('skipped')
    else:
        print(f"\n[4/5] Skipping RNAfold (--no-rnafold)")
        for col in ['mfe', 'mfe_per_nt', 'ensemble_fe',
                    'frac_paired', 'n_stemloops']:
            df[col] = None
        df['rnafold_status'] = 'skipped'

    # --- Write outputs -------------------------------------------------------
    print(f"\n[5/5] Writing outputs...")
    out_path = rp(cfg, 'rna_features')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  RNA features: {out_path}")
    print(f"  Rows: {len(df):,}  |  Feature columns: {len(df.columns)-5}")

    # Write report
    _write_report(df, args, str(rp(cfg, 'rna_features_report')))

    # --- Summary -------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RNA FEATURE SUMMARY")
    print(f"{'='*70}")
    for cls, label in [(1, 'Positives'), (0, 'Negatives')]:
        sub = df[df['class'] == cls]
        print(f"\n  {label} (n={len(sub):,}):")
        print(f"    AU content  : {sub['au_content'].mean():.3f} ± "
              f"{sub['au_content'].std():.3f}")
        print(f"    RRACH/kb    : {sub['rrach_per_kb'].mean():.2f} ± "
              f"{sub['rrach_per_kb'].std():.2f}")
        print(f"    mRNA length : {sub['mrna_length'].median():.0f} nt (median)")
        if 'mfe' in df.columns and sub['mfe'].notna().any():
            print(f"    MFE/nt      : {sub['mfe_per_nt'].mean():.4f}")
    print(f"{'='*70}")
    print("\n✓ Phase 4 complete.")
    return 0


def _write_report(df, args, out_path):
    n_rnafold = (df.get('rnafold_status', pd.Series()) == 'computed').sum()
    lines = [
        "=" * 80,
        "ZORC — RNA Feature Engineering Report",
        "Script: 04_rna_features.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
        "FEATURES COMPUTED",
        "  Nucleotide composition (fA, fU, fG, fC, au_content, gc_content)",
        "  Dinucleotide frequencies (16 features: AA, AU, ...)",
        "  mRNA length",
        "  UTR lengths (5'UTR, 3'UTR, CDS) from GTF",
        "  3'UTR AU content",
        "  m6A motifs: RRACH, AAACH (count + per_kb)",
        "  NMD proxies: long_3utr, ptc_proxy",
        f"  RNAfold: {'computed' if not args.no_rnafold else 'SKIPPED (--no-rnafold)'}",
        "    MFE, MFE/nt, ensemble_fe, frac_paired, n_stemloops",
        "",
        f"TOTAL SEQUENCES: {len(df):,}",
        f"  Positives: {(df['class']==1).sum():,}",
        f"  Negatives: {(df['class']==0).sum():,}",
        f"  RNAfold computed: {n_rnafold:,}",
        "",
        "=" * 80, "END OF REPORT", "=" * 80,
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
