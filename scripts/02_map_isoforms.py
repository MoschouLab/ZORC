#!/usr/bin/env python3
"""
02_map_isoforms.py
==================
ZORC — Zip-code Of RNAs that Condense
Phase 2: Map rMATS splicing events to specific transcript IDs.

For each gene in the ZORC coregulon list, identifies the most relevant isoform:

  STRATEGY A — rMATS-guided (preferred):
    For genes with a rMATS event, intersect event coordinates with the GTF
    to find the transcript(s) containing the alternatively spliced region.
    The "inclusion isoform" is selected per event type:
      SE   : transcript containing the skipped exon
      RI   : transcript where the retained intron region is exonic
      A3SS : transcript containing the long exon (long isoform)
      A5SS : transcript containing the long exon (long isoform)
      MXE  : transcript containing the 1st or 2nd exon (higher |ΔΨ|)
    Coordinate convention: rMATS uses 0-based starts → convert +1 for GTF.

  STRATEGY B — canonical fallback:
    For genes WITHOUT a rMATS event (most coregulon genes — T-RIP enrichment
    is gene-level, not isoform-level), assign the Ensembl_canonical transcript
    from the GTF. If no canonical tag, use the transcript with most exons.

  NEGATIVES:
    Always use canonical transcript (they are depleted — no specific "enriched"
    isoform to identify).

Output:
    data/processed/02_zorc_isoform_map.csv
    logs/02_isoform_map_report.txt

Usage:
    python 02_map_isoforms.py --config config/zorc_config.yaml

Author: José Moya-Cuevas, MoschouLab / IMBB-FORTH / ERC PLANTEX
Project: ZORC — github.com/MoschouLab/ZORC
License: MIT
"""

import argparse
import os
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyranges as pr
import yaml


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def rp(cfg, key_path):
    """Resolve a dotted path in config to an absolute Path."""
    keys = key_path.split('.')
    val = cfg
    for k in keys:
        val = val[k]
    return Path(os.path.expanduser(val))


# =============================================================================
# GTF INDEX
# =============================================================================

def build_gtf_index(gtf_path: str) -> tuple[dict, dict, dict]:
    """
    Parse GTF and build three indexes:
      gene2transcripts : gene_id → list of transcript_ids
      tx2exons         : transcript_id → list of (start_1based, end_1based, chr, strand)
      tx_meta          : transcript_id → {gene_id, biotype, is_canonical, n_exons}

    Returns (gene2transcripts, tx2exons, tx_meta)
    """
    print(f"  Parsing GTF: {gtf_path}")
    gene2transcripts = defaultdict(list)
    tx2exons = defaultdict(list)
    tx_meta = {}

    attr_re = re.compile(r'(\w+)\s+"([^"]+)"')

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ('transcript', 'exon'):
                continue

            chrom  = parts[0]
            start  = int(parts[3])   # 1-based
            end    = int(parts[4])   # 1-based
            strand = parts[6]
            attrs  = dict(attr_re.findall(parts[8]))

            gene_id   = attrs.get('gene_id', '')
            tx_id     = attrs.get('transcript_id', '')
            biotype   = attrs.get('transcript_biotype', attrs.get('gene_biotype', ''))
            canonical = 'Ensembl_canonical' in parts[8]

            if not gene_id or not tx_id:
                continue

            if feature == 'transcript':
                if tx_id not in tx_meta:
                    tx_meta[tx_id] = {
                        'gene_id':     gene_id,
                        'biotype':     biotype,
                        'is_canonical': canonical,
                        'chr':         chrom,
                        'strand':      strand,
                        'tx_start':    start,
                        'tx_end':      end,
                        'n_exons':     0,
                    }
                if tx_id not in gene2transcripts[gene_id]:
                    gene2transcripts[gene_id].append(tx_id)

            elif feature == 'exon':
                tx2exons[tx_id].append((chrom, start, end, strand))
                if tx_id in tx_meta:
                    tx_meta[tx_id]['n_exons'] += 1

    n_genes = len(gene2transcripts)
    n_txs   = len(tx_meta)
    print(f"  → {n_genes:,} genes, {n_txs:,} transcripts indexed")
    return dict(gene2transcripts), dict(tx2exons), tx_meta


def get_canonical_transcript(gene_id: str,
                              gene2transcripts: dict,
                              tx_meta: dict) -> str | None:
    """
    Return the best representative transcript for a gene:
    1. Ensembl_canonical protein_coding
    2. Any Ensembl_canonical
    3. Most exons among protein_coding
    4. Most exons overall
    """
    txs = gene2transcripts.get(gene_id, [])
    if not txs:
        return None

    def score(tx):
        m = tx_meta.get(tx, {})
        is_pc  = int(m.get('biotype', '') == 'protein_coding')
        is_can = int(m.get('is_canonical', False))
        n_ex   = m.get('n_exons', 0)
        return (is_can + is_pc, n_ex)

    return max(txs, key=score)


# =============================================================================
# rMATS LOADING
# =============================================================================

EVENT_COORD_COLS = {
    'SE':   ['exonStart_0base', 'exonEnd'],
    'RI':   ['upstreamEE', 'downstreamES'],  # intron = upstreamEE..downstreamES
    'A3SS': ['longExonStart_0base', 'longExonEnd'],
    'A5SS': ['longExonStart_0base', 'longExonEnd'],
    'MXE':  ['1stExonStart_0base', '1stExonEnd',
              '2ndExonStart_0base', '2ndExonEnd'],
}

def load_rmats_events(rmats_dir: str,
                      coregulon_ids: set,
                      condition_label: str) -> pd.DataFrame:
    """
    Load all rMATS event files from a directory.
    Filter to genes present in coregulon.
    Strips quotes from GeneID column.

    Returns combined DataFrame with columns:
      geneID, event_type, condition, chr, strand,
      coord_start (1-based), coord_end (1-based),
      event_id, [raw coord columns...]
    """
    rmats_dir = Path(os.path.expanduser(rmats_dir))
    all_events = []

    for event_type, coord_cols in EVENT_COORD_COLS.items():
        fpath = rmats_dir / f"{event_type}.MATS.JC.txt"
        if not fpath.exists():
            print(f"    WARNING: {fpath} not found, skipping")
            continue

        try:
            # quoting=3 (QUOTE_NONE) required: some rMATS geneSymbol fields
            # contain embedded tabs inside double-quotes (e.g. "GAMMA-CA1"),
            # causing pandas to see extra fields. Strip quotes manually after.
            import csv as _csv
            df = pd.read_csv(fpath, sep='\t', dtype=str, low_memory=False,
                             quoting=_csv.QUOTE_NONE, encoding_errors='replace')
        except Exception as e:
            print(f"    WARNING: could not read {fpath}: {e}")
            continue

        # Clean GeneID — strip quotes
        df['GeneID'] = df['GeneID'].str.strip('"').str.strip()

        # Filter to coregulon genes
        df = df[df['GeneID'].isin(coregulon_ids)].copy()
        if df.empty:
            continue

        # Convert numeric columns
        num_cols = [c for c in coord_cols if c in df.columns]
        for c in num_cols + ['IncLevel1', 'IncLevel2', 'IncLevelDifference',
                              'PValue', 'FDR']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df['event_type'] = event_type
        df['condition']  = condition_label
        df['geneID']     = df['GeneID']

        # Rename ID to event_id to avoid conflicts
        if 'ID' in df.columns:
            df = df.rename(columns={'ID': 'event_id'})

        all_events.append(df)

    if not all_events:
        return pd.DataFrame()

    combined = pd.concat(all_events, ignore_index=True, sort=False)
    print(f"    {condition_label}: {len(combined):,} events across "
          f"{combined['geneID'].nunique():,} coregulon genes")
    return combined


# =============================================================================
# TRANSCRIPT IDENTIFICATION VIA COORDINATE INTERSECTION
# =============================================================================

def exon_overlaps_interval(exons: list, chrom: str, q_start: int, q_end: int,
                            min_overlap: int = 1) -> bool:
    """
    True if any exon in `exons` overlaps [q_start, q_end] (1-based, inclusive)
    by at least min_overlap bases.
    """
    for ex_chr, ex_start, ex_end, _ in exons:
        if ex_chr != chrom:
            continue
        overlap = min(ex_end, q_end) - max(ex_start, q_start) + 1
        if overlap >= min_overlap:
            return True
    return False


def exon_contains_interval(exons: list, chrom: str,
                            q_start: int, q_end: int) -> bool:
    """True if any exon fully contains [q_start, q_end] (1-based, inclusive)."""
    for ex_chr, ex_start, ex_end, _ in exons:
        if ex_chr != chrom:
            continue
        if ex_start <= q_start and ex_end >= q_end:
            return True
    return False


def intron_is_retained(exons: list, chrom: str,
                       intron_start: int, intron_end: int) -> bool:
    """
    For RI: returns True if the intron region [intron_start, intron_end]
    is covered by an exon (i.e., the intron is retained in this transcript).
    Uses exon_contains_interval with ±2bp tolerance for splice site variation.
    """
    return exon_contains_interval(exons, chrom,
                                  intron_start + 2, intron_end - 2)


def find_inclusion_transcript(event_row: pd.Series,
                               event_type: str,
                               gene_id: str,
                               gene2transcripts: dict,
                               tx2exons: dict,
                               tx_meta: dict) -> tuple[str | None, str]:
    """
    Find the transcript that INCLUDES the alternatively spliced region.
    Returns (transcript_id, method_description).

    Coordinate conversion: rMATS 0-based start → +1 for GTF comparison.
    """
    txs = gene2transcripts.get(gene_id, [])
    if not txs:
        return None, 'no_transcripts_in_gtf'

    chrom = str(event_row.get('chr', ''))
    # Normalise chromosome name: GTF may use '1' or 'Chr1'
    if chrom.startswith('chr') and len(chrom) <= 5:
        chrom_alt = chrom[3:]
    else:
        chrom_alt = 'chr' + chrom

    candidates = []

    for tx_id in txs:
        exons = tx2exons.get(tx_id, [])
        if not exons:
            continue

        # Try both chromosome name formats
        def _check(ch):
            if event_type == 'SE':
                s = int(event_row['exonStart_0base']) + 1
                e = int(event_row['exonEnd'])
                return exon_overlaps_interval(exons, ch, s, e, min_overlap=10)

            elif event_type == 'RI':
                s = int(event_row['upstreamEE']) + 1   # intron start (1-based)
                e = int(event_row['downstreamES'])       # intron end
                return intron_is_retained(exons, ch, s, e)

            elif event_type in ('A3SS', 'A5SS'):
                s = int(event_row['longExonStart_0base']) + 1
                e = int(event_row['longExonEnd'])
                return exon_overlaps_interval(exons, ch, s, e, min_overlap=10)

            elif event_type == 'MXE':
                s1 = int(event_row['1stExonStart_0base']) + 1
                e1 = int(event_row['1stExonEnd'])
                s2 = int(event_row['2ndExonStart_0base']) + 1
                e2 = int(event_row['2ndExonEnd'])
                return (exon_overlaps_interval(exons, ch, s1, e1, min_overlap=10) or
                        exon_overlaps_interval(exons, ch, s2, e2, min_overlap=10))
            return False

        matched = _check(chrom) or _check(chrom_alt)
        if matched:
            m = tx_meta.get(tx_id, {})
            candidates.append((tx_id,
                                int(m.get('is_canonical', False)),
                                int(m.get('biotype', '') == 'protein_coding'),
                                m.get('n_exons', 0)))

    if not candidates:
        return None, f'no_overlap_{event_type}'

    # Rank: canonical > protein_coding > most exons
    candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    best_tx = candidates[0][0]
    method = f'rmats_{event_type}_overlap'
    if len(candidates) > 1:
        method += f'_{len(candidates)}candidates'
    return best_tx, method


# =============================================================================
# MAIN MAPPING LOGIC
# =============================================================================

def map_isoforms(coregulon: pd.DataFrame,
                 rmats_ns: pd.DataFrame,
                 rmats_hs: pd.DataFrame,
                 gene2transcripts: dict,
                 tx2exons: dict,
                 tx_meta: dict) -> pd.DataFrame:
    """
    For each gene in coregulon, assign a transcript_id.

    Priority:
      1. rMATS-guided (HS preferred > NS, highest |ΔΨ| event)
      2. Canonical fallback
    """
    # Combine NS + HS rMATS events; prefer HS (heat stress context)
    if not rmats_ns.empty and not rmats_hs.empty:
        rmats_all = pd.concat([rmats_hs, rmats_ns], ignore_index=True)
    elif not rmats_ns.empty:
        rmats_all = rmats_ns.copy()
    elif not rmats_hs.empty:
        rmats_all = rmats_hs.copy()
    else:
        rmats_all = pd.DataFrame()

    # Build per-gene rMATS event index
    gene_events = {}
    if not rmats_all.empty and 'geneID' in rmats_all.columns:
        for gid, grp in rmats_all.groupby('geneID'):
            # Sort by |ΔΨ| descending — use highest-confidence event
            if 'IncLevelDifference' in grp.columns:
                grp = grp.copy()
                grp['abs_dpsi'] = grp['IncLevelDifference'].abs()
                grp = grp.sort_values('abs_dpsi', ascending=False)
            gene_events[gid] = grp

    results = []
    n_rmats   = 0
    n_canonical = 0
    n_missing   = 0
    method_counts = defaultdict(int)

    for _, row in coregulon.iterrows():
        gid    = row['geneID']
        cls    = row['class']
        result = {
            'geneID':          gid,
            'class':           cls,
            'condition':       row.get('condition', ''),
            'transcript_id':   None,
            'isoform_source':  None,
            'event_type':      None,
            'event_condition': None,
            'n_rmats_events':  0,
            'transcript_biotype':  None,
            'is_canonical':    None,
            'n_exons':         None,
        }

        # --- Strategy A: rMATS-guided (positives only) ---
        if cls == 1 and gid in gene_events:
            events = gene_events[gid]
            result['n_rmats_events'] = len(events)
            found_tx = None

            for _, ev in events.iterrows():
                etype = ev.get('event_type', '')
                tx, method = find_inclusion_transcript(
                    ev, etype, gid, gene2transcripts, tx2exons, tx_meta
                )
                if tx:
                    found_tx = tx
                    result['event_type']      = etype
                    result['event_condition'] = ev.get('condition', '')
                    result['isoform_source']  = method
                    break

            if found_tx:
                result['transcript_id'] = found_tx
                n_rmats += 1
                method_counts[result['isoform_source']] += 1

        # --- Strategy B: canonical fallback ---
        if result['transcript_id'] is None:
            canon = get_canonical_transcript(gid, gene2transcripts, tx_meta)
            result['transcript_id']  = canon
            result['isoform_source'] = 'canonical_fallback'
            if canon:
                n_canonical += 1
                method_counts['canonical_fallback'] += 1
            else:
                n_missing += 1
                method_counts['no_transcript_found'] += 1

        # Annotate transcript metadata
        if result['transcript_id']:
            m = tx_meta.get(result['transcript_id'], {})
            result['transcript_biotype'] = m.get('biotype', '')
            result['is_canonical']       = m.get('is_canonical', False)
            result['n_exons']            = m.get('n_exons', 0)

        results.append(result)

    df = pd.DataFrame(results)

    # --- Drop genes with no transcript assignment ----------------------------
    # These are overwhelmingly pseudogenes, transposable elements, and tRNAs
    # that lack proper transcript annotations in the GTF. They are not
    # biologically informative training examples.
    no_tx_mask = df['transcript_id'].isna()
    n_dropped  = no_tx_mask.sum()
    if n_dropped > 0:
        dropped_df = df[no_tx_mask][['geneID', 'class']].copy()
        n_dropped_pos = (dropped_df['class'] == 1).sum()
        n_dropped_neg = (dropped_df['class'] == 0).sum()
        print(f"\n  Dropping {n_dropped} genes with no GTF transcript:")
        print(f"    Positives dropped: {n_dropped_pos}")
        print(f"    Negatives dropped: {n_dropped_neg}")
        if n_dropped_pos > 0:
            print(f"    WARNING — dropped positive gene(s):")
            for g in dropped_df[dropped_df['class']==1]['geneID'].tolist():
                print(f"      {g}  ← investigate manually")
        df = df[~no_tx_mask].copy()
        method_counts['dropped_no_gtf_transcript'] = n_dropped
        n_missing = 0  # already accounted for in method_counts

    # Merge back the full coregulon columns
    keep_from_coregulon = [c for c in coregulon.columns
                           if c not in ('class', 'condition', 'geneID')]
    df = df.merge(coregulon[['geneID'] + keep_from_coregulon],
                  on='geneID', how='left')

    # Final column order
    front_cols = ['geneID', 'transcript_id', 'class', 'condition',
                  'isoform_source', 'event_type', 'event_condition',
                  'n_rmats_events', 'transcript_biotype',
                  'is_canonical', 'n_exons']
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    print(f"\n  Isoform assignment summary:")
    print(f"    rMATS-guided  : {n_rmats:,}")
    print(f"    Canonical fallback: {n_canonical:,}")
    print(f"    No transcript found: {n_missing:,}")
    print(f"\n  Method breakdown:")
    for method, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {method:<45}: {cnt:,}")

    return df, n_rmats, n_canonical, n_missing, method_counts


# =============================================================================
# AUDIT REPORT
# =============================================================================

def write_report(df, n_rmats, n_canonical, n_missing,
                 method_counts, cfg, out_path):
    n_pos = (df['class'] == 1).sum()
    n_neg = (df['class'] == 0).sum()
    no_tx = df['transcript_id'].isna().sum()
    biotype_counts = df['transcript_biotype'].value_counts().head(10)

    lines = [
        "=" * 80,
        "ZORC — Isoform Mapping Report",
        "Script: 02_map_isoforms.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "INPUT",
        "-----",
        f"  Coregulon genes : {len(df):,}  (positives={n_pos:,}, negatives={n_neg:,})",
        f"  rMATS NS dir    : {cfg['reference']['rmats_ns_dir']}",
        f"  rMATS HS dir    : {cfg['reference']['rmats_hs_dir']}",
        f"  GTF             : {cfg['reference']['gtf']}",
        "",
        "ISOFORM ASSIGNMENT STRATEGY",
        "----------------------------",
        "  Strategy A — rMATS-guided (positives with matching event):",
        "    SE   : transcript containing the skipped exon",
        "    RI   : transcript where retained intron region is exonic",
        "    A3SS : transcript containing the long exon (long isoform)",
        "    A5SS : transcript containing the long exon (long isoform)",
        "    MXE  : transcript containing either mutually exclusive exon",
        "    Priority: HS > NS; highest |ΔΨ| event selected first",
        "",
        "  Strategy B — Canonical fallback:",
        "    Ensembl_canonical + protein_coding > canonical > most exons",
        "",
        "RESULTS",
        "-------",
        f"  rMATS-guided assignments : {n_rmats:,}",
        f"  Canonical fallback       : {n_canonical:,}",
        f"  No transcript found      : {n_missing:,}",
        f"  Missing transcript_id    : {no_tx:,}",
        "",
        "  Method breakdown:",
    ]
    for method, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {method:<45}: {cnt:,}")

    lines += [
        "",
        "  Transcript biotype distribution (top 10):",
    ]
    for bt, cnt in biotype_counts.items():
        lines.append(f"    {bt:<40}: {cnt:,}")

    lines += [
        "",
        "  Positives with rMATS-guided isoform: "
        f"{(df[df['class']==1]['isoform_source'].str.startswith('rmats', na=False)).sum():,} / {n_pos:,}",
        "",
        "  Event type breakdown (rMATS-guided only):",
    ]
    for et, cnt in df[df['isoform_source'].str.startswith('rmats', na=False)
                     ]['event_type'].value_counts().items():
        lines.append(f"    {et:<10}: {cnt:,}")

    lines += [
        "",
        "  Sample assignments (first 15 genes):",
        f"  {'geneID':<14} {'transcript_id':<18} {'source':<35} {'biotype'}",
        "  " + "-" * 85,
    ]
    for _, r in df.head(15).iterrows():
        lines.append(
            f"  {r['geneID']:<14} {str(r['transcript_id']):<18} "
            f"{str(r['isoform_source']):<35} {str(r['transcript_biotype'])}"
        )

    lines += [
        "",
        f"OUTPUT: {cfg['outputs'].get('isoform_map', 'data/processed/02_zorc_isoform_map.csv')}",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ]

    report = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\n  Report written to: {out_path}")
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZORC Phase 2 — Map rMATS events to transcript IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', required=True, help='Path to zorc_config.yaml')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and map but do not write outputs')
    args = parser.parse_args()

    cfg = load_config(args.config)
    proj = Path(os.path.expanduser(cfg['project_dir']))

    # Auto-add output keys to config if missing (avoids manual yaml editing)
    _defaults = {
        'isoform_map':        'data/processed/02_zorc_isoform_map.csv',
        'isoform_map_report': 'logs/02_isoform_map_report.txt',
    }
    _changed = False
    for k, v in _defaults.items():
        if k not in cfg.get('outputs', {}):
            cfg.setdefault('outputs', {})[k] = v
            _changed = True
    if _changed:
        import yaml as _yaml
        with open(args.config, 'w') as _f:
            _yaml.dump(cfg, _f, default_flow_style=False,
                       allow_unicode=True, sort_keys=False)
        print(f"  [config] Auto-added missing output keys to {args.config}")

    print("=" * 70)
    print("ZORC — Phase 2: Isoform Mapping")
    print(f"Config: {args.config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Load coregulon ------------------------------------------------------
    coregulon_path = proj / cfg['outputs']['coregulon_list']
    print(f"\n[1/5] Loading coregulon list: {coregulon_path}")
    coregulon = pd.read_csv(coregulon_path)
    coregulon_ids = set(coregulon['geneID'])
    print(f"  → {len(coregulon):,} genes  "
          f"(pos={( coregulon['class']==1).sum():,}, "
          f"neg={(coregulon['class']==0).sum():,})")

    # --- Build GTF index -----------------------------------------------------
    gtf_path = os.path.expanduser(cfg['reference']['gtf'])
    print(f"\n[2/5] Building GTF index...")
    gene2transcripts, tx2exons, tx_meta = build_gtf_index(gtf_path)

    # --- Load rMATS events ---------------------------------------------------
    ns_dir = os.path.expanduser(cfg['reference']['rmats_ns_dir'])
    hs_dir = os.path.expanduser(cfg['reference']['rmats_hs_dir'])

    print(f"\n[3/5] Loading rMATS events for coregulon genes...")
    print(f"  NS directory: {ns_dir}")
    rmats_ns = load_rmats_events(ns_dir, coregulon_ids, 'NS')
    print(f"  HS directory: {hs_dir}")
    rmats_hs = load_rmats_events(hs_dir, coregulon_ids, 'HS')

    # --- Map isoforms --------------------------------------------------------
    print(f"\n[4/5] Assigning transcript IDs...")
    df_map, n_rmats, n_canonical, n_missing, method_counts = map_isoforms(
        coregulon, rmats_ns, rmats_hs,
        gene2transcripts, tx2exons, tx_meta
    )

    # --- Print summary -------------------------------------------------------
    n_pos = (df_map['class'] == 1).sum()
    n_neg = (df_map['class'] == 0).sum()
    no_tx = df_map['transcript_id'].isna().sum()

    print(f"\n{'='*70}")
    print("ISOFORM MAP SUMMARY")
    print(f"{'='*70}")
    print(f"  Total genes mapped : {len(df_map):,}")
    print(f"  Positives          : {n_pos:,}")
    print(f"  Negatives          : {n_neg:,}")
    print(f"  rMATS-guided       : {n_rmats:,}")
    print(f"  Canonical fallback : {n_canonical:,}")
    print(f"  No transcript      : {n_missing:,}  ← check these manually")
    print(f"{'='*70}")

    if no_tx > 0:
        missing_genes = df_map[df_map['transcript_id'].isna()]['geneID'].tolist()
        print(f"\n  WARNING: {no_tx} genes without transcript assignment:")
        for g in missing_genes[:20]:
            print(f"    {g}")
        if len(missing_genes) > 20:
            print(f"    ... and {len(missing_genes)-20} more (see report)")

    # --- Write outputs -------------------------------------------------------
    print(f"\n[5/5] Writing outputs...")

    if not args.dry_run:
        out_key = 'isoform_map'
        # Add to config outputs if not present
        out_path = proj / cfg['outputs'].get(
            out_key, 'data/processed/02_zorc_isoform_map.csv'
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_map.to_csv(out_path, index=False)
        print(f"  Isoform map: {out_path}")
        print(f"  Rows: {len(df_map):,}  |  Columns: {len(df_map.columns)}")

        report_path = proj / cfg['outputs'].get(
            'isoform_map_report', 'logs/02_isoform_map_report.txt'
        )
        write_report(df_map, n_rmats, n_canonical, n_missing,
                     method_counts, cfg, str(report_path))
    else:
        print("  [DRY RUN] — no files written")

    print("\n✓ Phase 2 complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
