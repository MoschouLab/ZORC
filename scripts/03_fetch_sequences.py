#!/usr/bin/env python3
"""
03_fetch_sequences.py
=====================
ZORC — Zip-code Of RNAs that Condense
Phase 3: Extract mRNA and protein sequences for all isoforms in the map.

Strategy:
  1. Load isoform map from P2 (02_zorc_isoform_map.csv)
  2. Build a minimal filtered GTF containing only the required transcript IDs
  3. Run gffread to extract:
       -w  spliced mRNA sequences (all exons joined)
       -y  translated CDS / protein sequences
  4. Rename FASTA headers to:
       >GeneID|TranscriptID|class|condition
  5. Write four output FASTAs:
       positives_mrna.fa, positives_protein.fa
       negatives_mrna.fa, negatives_protein.fa
     plus combined files (all_mrna.fa, all_protein.fa) for downstream steps.
  6. QC: flag transcripts with no CDS (non-coding), short proteins (<30 aa),
     and sequences containing internal stop codons.

Dependencies: gffread (conda install -c bioconda gffread)

Usage:
    python 03_fetch_sequences.py --config config/zorc_config.yaml

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

import pandas as pd
import yaml


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def save_config(cfg: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

def ensure_output_keys(cfg: dict, config_path: str) -> dict:
    """Auto-add P3 output keys to config if missing."""
    defaults = {
        'sequences_dir':          'data/processed/sequences',
        'all_mrna_fa':            'data/processed/sequences/all_mrna.fa',
        'all_protein_fa':         'data/processed/sequences/all_protein.fa',
        'positives_mrna_fa':      'data/processed/sequences/positives_mrna.fa',
        'positives_protein_fa':   'data/processed/sequences/positives_protein.fa',
        'negatives_mrna_fa':      'data/processed/sequences/negatives_mrna.fa',
        'negatives_protein_fa':   'data/processed/sequences/negatives_protein.fa',
        'sequence_qc_report':     'logs/03_sequence_qc_report.txt',
        'sequence_manifest':      'data/processed/03_sequence_manifest.csv',
    }
    changed = False
    for k, v in defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            changed = True
    if changed:
        save_config(cfg, config_path)
        print(f"  [config] Auto-added P3 output keys to {config_path}")
    return cfg

def rp(cfg: dict, key: str) -> Path:
    proj = Path(os.path.expanduser(cfg['project_dir']))
    return proj / cfg['outputs'][key]


# =============================================================================
# FILTER GTF TO TARGET TRANSCRIPTS
# =============================================================================

def filter_gtf(gtf_path: str, target_tx_ids: set, out_path: str) -> int:
    """
    Write a filtered GTF containing only entries for target transcript IDs.
    Keeps 'gene', 'transcript', 'exon', 'CDS', 'start_codon', 'stop_codon'
    features for the target transcripts.

    Returns number of transcripts written.
    """
    tx_pattern = re.compile(r'transcript_id\s+"([^"]+)"')
    gene_id_pattern = re.compile(r'gene_id\s+"([^"]+)"')

    # First pass: collect gene_ids for our transcripts
    tx_to_gene = {}
    target_gene_ids = set()

    print(f"  Scanning GTF for {len(target_tx_ids):,} target transcripts...")
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            m_tx = tx_pattern.search(line)
            if m_tx and m_tx.group(1) in target_tx_ids:
                m_gene = gene_id_pattern.search(line)
                if m_gene:
                    tx_to_gene[m_tx.group(1)] = m_gene.group(1)
                    target_gene_ids.add(m_gene.group(1))

    found_txs = set(tx_to_gene.keys())
    missing = target_tx_ids - found_txs
    if missing:
        print(f"  WARNING: {len(missing)} transcript IDs not found in GTF")
        for tx in sorted(missing)[:10]:
            print(f"    {tx}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    # Second pass: write filtered GTF
    features_kept = {'gene', 'transcript', 'exon', 'CDS',
                     'start_codon', 'stop_codon', 'UTR'}
    n_lines = 0

    with open(gtf_path) as f_in, open(out_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('#'):
                f_out.write(line)
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in features_kept:
                continue

            m_tx = tx_pattern.search(line)
            if m_tx and m_tx.group(1) in found_txs:
                f_out.write(line)
                n_lines += 1
            elif feature == 'gene':
                m_gene = gene_id_pattern.search(line)
                if m_gene and m_gene.group(1) in target_gene_ids:
                    f_out.write(line)
                    n_lines += 1

    print(f"  → Filtered GTF: {len(found_txs):,} transcripts, {n_lines:,} lines → {out_path}")
    return len(found_txs)


# =============================================================================
# RUN GFFREAD
# =============================================================================

def run_gffread(gtf_path: str, genome_fa: str,
                out_mrna: str, out_protein: str) -> tuple[bool, str]:
    """
    Run gffread to extract spliced mRNA (-w) and protein (-y) sequences.
    Returns (success, stderr_output).
    """
    Path(out_mrna).parent.mkdir(parents=True, exist_ok=True)
    Path(out_protein).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'gffread', gtf_path,
        '-g', genome_fa,
        '-w', out_mrna,
        '-y', out_protein,
        '-F',       # preserve all attributes
        '-E',       # expose exon/CDS features
    ]
    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: gffread failed (exit {result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return False, result.stderr

    # Count sequences produced
    n_mrna = sum(1 for l in open(out_mrna) if l.startswith('>'))
    n_prot = sum(1 for l in open(out_protein) if l.startswith('>'))
    print(f"  → mRNA sequences: {n_mrna:,}")
    print(f"  → Protein sequences: {n_prot:,}")
    return True, result.stderr


# =============================================================================
# PARSE AND RENAME FASTA
# =============================================================================

def parse_fasta(path: str) -> dict[str, str]:
    """Return {header: sequence} dict from a FASTA file."""
    seqs = {}
    current_header = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if current_header is not None:
                    seqs[current_header] = ''.join(current_seq)
                current_header = line[1:].split()[0]  # first word after >
                current_seq = []
            else:
                current_seq.append(line)
    if current_header is not None:
        seqs[current_header] = ''.join(current_seq)
    return seqs


def write_fasta(records: list[tuple[str, str]], path: str):
    """Write list of (header, sequence) tuples to FASTA."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for header, seq in records:
            f.write(f'>{header}\n')
            # Wrap at 80 chars
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')


def rename_and_split_fastas(
    raw_mrna_path: str,
    raw_prot_path: str,
    isoform_map: pd.DataFrame,
) -> tuple[dict, dict, list]:
    """
    Rename FASTA headers to: GeneID|TranscriptID|class|condition
    Split into positives / negatives.
    Returns (mrna_records, prot_records, qc_flags)
      where each dict has keys: 'all', 'positives', 'negatives'
      and qc_flags is a list of dicts for the QC report.
    """
    mrna_seqs = parse_fasta(raw_mrna_path)
    prot_seqs  = parse_fasta(raw_prot_path)

    # Build tx_id → row mapping
    tx_map = {}
    for _, row in isoform_map.iterrows():
        tx_id = row['transcript_id']
        if pd.notna(tx_id):
            tx_map[str(tx_id)] = row

    mrna_records = {'all': [], 'positives': [], 'negatives': []}
    prot_records  = {'all': [], 'positives': [], 'negatives': []}
    qc_flags = []

    for tx_id, seq_mrna in mrna_seqs.items():
        row = tx_map.get(tx_id)
        if row is None:
            continue  # transcript not in our map (shouldn't happen)

        gid       = row['geneID']
        cls       = int(row['class'])
        condition = str(row.get('condition', 'unknown'))
        new_header = f"{gid}|{tx_id}|class{cls}|{condition}"

        seq_prot = prot_seqs.get(tx_id, '')

        # QC flags
        qc = {
            'geneID':         gid,
            'transcript_id':  tx_id,
            'class':          cls,
            'condition':      condition,
            'mrna_len':       len(seq_mrna),
            'protein_len':    len(seq_prot),
            'has_cds':        len(seq_prot) > 0,
            'short_protein':  0 < len(seq_prot) < 30,
            'internal_stop':  '.' in seq_prot[:-1] if seq_prot else False,
            'qc_pass':        True,
        }
        # Mark QC failures
        if not qc['has_cds']:
            qc['qc_pass'] = False
            qc['qc_flag'] = 'no_CDS'
        elif qc['short_protein']:
            qc['qc_pass'] = False
            qc['qc_flag'] = 'short_protein'
        elif qc['internal_stop']:
            qc['qc_pass'] = False
            qc['qc_flag'] = 'internal_stop_codon'
        else:
            qc['qc_flag'] = 'pass'

        qc_flags.append(qc)

        # Add to records (both pass and fail — flag in header for transparency)
        suffix = '' if qc['qc_pass'] else f'|QC_FAIL:{qc["qc_flag"]}'
        full_header = new_header + suffix

        split_key = 'positives' if cls == 1 else 'negatives'

        mrna_records['all'].append((full_header, seq_mrna))
        mrna_records[split_key].append((full_header, seq_mrna))

        if seq_prot:
            prot_records['all'].append((full_header, seq_prot))
            prot_records[split_key].append((full_header, seq_prot))

    return mrna_records, prot_records, qc_flags


# =============================================================================
# QC REPORT
# =============================================================================

def write_qc_report(qc_flags: list, cfg: dict, out_path: str) -> pd.DataFrame:
    df_qc = pd.DataFrame(qc_flags)

    n_total   = len(df_qc)
    n_pass    = df_qc['qc_pass'].sum()
    n_fail    = n_total - n_pass
    n_no_cds  = (df_qc['qc_flag'] == 'no_CDS').sum()
    n_short   = (df_qc['qc_flag'] == 'short_protein').sum()
    n_istop   = (df_qc['qc_flag'] == 'internal_stop_codon').sum()
    n_pos_fail = df_qc[(df_qc['class']==1) & (~df_qc['qc_pass'])].shape[0]
    n_neg_fail = df_qc[(df_qc['class']==0) & (~df_qc['qc_pass'])].shape[0]

    lines = [
        "=" * 80,
        "ZORC — Sequence Fetch QC Report",
        "Script: 03_fetch_sequences.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "SEQUENCE STATISTICS",
        "-------------------",
        f"  Total sequences processed : {n_total:,}",
        f"  QC pass                   : {n_pass:,}",
        f"  QC fail                   : {n_fail:,}",
        f"    No CDS (non-coding)     : {n_no_cds:,}",
        f"    Short protein (<30 aa)  : {n_short:,}",
        f"    Internal stop codon     : {n_istop:,}",
        f"  Positives with QC fail    : {n_pos_fail:,}",
        f"  Negatives with QC fail    : {n_neg_fail:,}",
        "",
        "  Note: QC-fail sequences ARE included in output FASTAs",
        "  with a |QC_FAIL:reason suffix. They are flagged here for",
        "  downstream filtering decisions (e.g. exclude no_CDS from",
        "  protein feature extraction but keep for RNA features).",
        "",
        "mRNA LENGTH DISTRIBUTION",
        "------------------------",
    ]

    for cls, label in [(1, 'Positives'), (0, 'Negatives')]:
        sub = df_qc[df_qc['class'] == cls]['mrna_len']
        if len(sub):
            lines.append(
                f"  {label}: n={len(sub):,}  "
                f"median={sub.median():.0f} nt  "
                f"mean={sub.mean():.0f} nt  "
                f"min={sub.min()}  max={sub.max()}"
            )

    lines += [
        "",
        "PROTEIN LENGTH DISTRIBUTION",
        "---------------------------",
    ]
    for cls, label in [(1, 'Positives'), (0, 'Negatives')]:
        sub = df_qc[(df_qc['class'] == cls) & (df_qc['protein_len'] > 0)]['protein_len']
        if len(sub):
            lines.append(
                f"  {label}: n={len(sub):,}  "
                f"median={sub.median():.0f} aa  "
                f"mean={sub.mean():.0f} aa  "
                f"min={sub.min()}  max={sub.max()}"
            )

    if n_fail > 0:
        lines += ["", "QC FAILURES (first 30)", "-" * 40]
        for _, r in df_qc[~df_qc['qc_pass']].head(30).iterrows():
            lines.append(
                f"  {r['geneID']:<14} {r['transcript_id']:<18} "
                f"class={r['class']}  flag={r['qc_flag']}  "
                f"prot_len={r['protein_len']}"
            )

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    report = "\n".join(lines)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n  QC report: {out_path}")
    return df_qc


# =============================================================================
# MANIFEST
# =============================================================================

def write_manifest(qc_flags: list, isoform_map: pd.DataFrame,
                   cfg: dict, out_path: str):
    """
    Write a per-gene manifest CSV linking geneID → transcript_id →
    sequence lengths → QC status → isoform_source.
    This is the master reference for all downstream steps.
    """
    df_qc = pd.DataFrame(qc_flags)
    df_qc = df_qc.merge(
        isoform_map[['geneID', 'transcript_id', 'isoform_source',
                     'event_type', 'transcript_biotype', 'n_exons',
                     'strongest_rna_log2fc']],
        on=['geneID', 'transcript_id'], how='left'
    )
    cols = ['geneID', 'transcript_id', 'class', 'condition',
            'isoform_source', 'event_type', 'transcript_biotype',
            'n_exons', 'strongest_rna_log2fc',
            'mrna_len', 'protein_len', 'has_cds',
            'short_protein', 'internal_stop', 'qc_pass', 'qc_flag']
    df_qc = df_qc[[c for c in cols if c in df_qc.columns]]
    df_qc = df_qc.sort_values(['class', 'strongest_rna_log2fc'],
                               ascending=[False, False])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_qc.to_csv(out_path, index=False)
    print(f"  Sequence manifest: {out_path}")
    return df_qc


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 3 — Fetch mRNA and protein sequences.",
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--dry-run', action='store_true',
                        help='Build filtered GTF and check gffread, no final output')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = ensure_output_keys(cfg, args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    print("=" * 70)
    print("ZORC — Phase 3: Sequence Fetching")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load isoform map ----------------------------------------------------
    isomap_path = proj / cfg['outputs']['isoform_map']
    print(f"\n[1/6] Loading isoform map: {isomap_path}")
    isoform_map = pd.read_csv(isomap_path)
    isoform_map = isoform_map[isoform_map['transcript_id'].notna()].copy()
    target_tx_ids = set(isoform_map['transcript_id'].astype(str))
    print(f"  → {len(isoform_map):,} genes, {len(target_tx_ids):,} unique transcripts")
    print(f"     Positives: {(isoform_map['class']==1).sum():,}  "
          f"Negatives: {(isoform_map['class']==0).sum():,}")

    # --- Filter GTF ----------------------------------------------------------
    gtf_path    = os.path.expanduser(cfg['reference']['gtf'])
    genome_path = os.path.expanduser(cfg['reference']['genome_fasta'])
    seq_dir     = proj / cfg['outputs']['sequences_dir']
    seq_dir.mkdir(parents=True, exist_ok=True)

    filtered_gtf = str(seq_dir / 'filtered_transcripts.gtf')
    print(f"\n[2/6] Building filtered GTF...")
    n_found = filter_gtf(gtf_path, target_tx_ids, filtered_gtf)

    if n_found == 0:
        print("ERROR: No transcripts found in GTF. Check transcript ID format.")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Filtered GTF built. Stopping before gffread.")
        return 0

    # --- Run gffread ---------------------------------------------------------
    raw_mrna   = str(seq_dir / 'raw_gffread_mrna.fa')
    raw_prot   = str(seq_dir / 'raw_gffread_protein.fa')

    print(f"\n[3/6] Running gffread...")
    success, stderr = run_gffread(filtered_gtf, genome_path, raw_mrna, raw_prot)
    if not success:
        print("ERROR: gffread failed. See stderr above.")
        return 1

    # --- Parse, rename, split ------------------------------------------------
    print(f"\n[4/6] Renaming headers and splitting by class...")
    mrna_recs, prot_recs, qc_flags = rename_and_split_fastas(
        raw_mrna, raw_prot, isoform_map
    )

    n_mrna_all  = len(mrna_recs['all'])
    n_prot_all  = len(prot_recs['all'])
    n_mrna_pos  = len(mrna_recs['positives'])
    n_mrna_neg  = len(mrna_recs['negatives'])
    n_prot_pos  = len(prot_recs['positives'])
    n_prot_neg  = len(prot_recs['negatives'])

    print(f"  mRNA  — all: {n_mrna_all:,}  pos: {n_mrna_pos:,}  neg: {n_mrna_neg:,}")
    print(f"  Protein — all: {n_prot_all:,}  pos: {n_prot_pos:,}  neg: {n_prot_neg:,}")

    # --- Write FASTAs --------------------------------------------------------
    print(f"\n[5/6] Writing FASTAs...")
    fasta_outputs = {
        'all_mrna_fa':          mrna_recs['all'],
        'all_protein_fa':       prot_recs['all'],
        'positives_mrna_fa':    mrna_recs['positives'],
        'positives_protein_fa': prot_recs['positives'],
        'negatives_mrna_fa':    mrna_recs['negatives'],
        'negatives_protein_fa': prot_recs['negatives'],
    }
    for key, records in fasta_outputs.items():
        out = proj / cfg['outputs'][key]
        write_fasta(records, str(out))
        print(f"  {key:<28}: {len(records):>5} sequences → {out.name}")

    # --- QC report and manifest ----------------------------------------------
    print(f"\n[6/6] Writing QC report and manifest...")
    qc_report_path   = proj / cfg['outputs']['sequence_qc_report']
    manifest_path    = proj / cfg['outputs']['sequence_manifest']
    df_qc  = write_qc_report(qc_flags, cfg, str(qc_report_path))
    df_man = write_manifest(qc_flags, isoform_map, cfg, str(manifest_path))

    # --- Final summary -------------------------------------------------------
    n_pass = df_qc['qc_pass'].sum()
    n_fail = len(df_qc) - n_pass
    print(f"\n{'='*70}")
    print("SEQUENCE FETCH SUMMARY")
    print(f"{'='*70}")
    print(f"  Total sequences    : {n_mrna_all:,}")
    print(f"  With protein (CDS) : {n_prot_all:,}")
    print(f"  QC pass            : {n_pass:,}")
    print(f"  QC fail (flagged)  : {n_fail:,}")
    print(f"  Output dir         : {seq_dir}")
    print(f"{'='*70}")
    print("\n✓ Phase 3 complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
