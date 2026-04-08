#!/usr/bin/env python3
"""
ZORC — Phase 9c: AlphaFold2 Structure Recovery for Tier 1 proteins
=====================================================================
P7 silently failed to fetch AF2 structures for all 698 Tier 1 proteins
(IDR < 10%). Root cause: UniProt query used `reviewed:true` (Swiss-Prot
only, ~500 Arabidopsis entries), returning wrong/missing accessions.

Fix: query UniProt via `xref:tair-{gene_id}` — the TAIR cross-reference
field, which correctly maps AT codes to UniProt accessions.

Strategy:
  1. Load 07_protein_features.csv — identify Tier 1 with af2_not_found
  2. Map TAIR → UniProt via xref:tair query (with rate limit + cache)
  3. Download AF2 PDBs from EBI (AF-{uniprot}-F1-model_v4.pdb)
  4. Extract Rg, contact_density, packing_density, sasa_per_residue,
     n_residues using MDAnalysis (same functions as P7)
  5. Patch 07_protein_features.csv in place (no split/model changes yet)
  6. Regenerate 08_zorc_feature_matrix.csv with recovered features

Usage:
    conda activate zorc_pipeline
    python scripts/09c_af2_recovery.py --config config/zorc_config.yaml

    # Dry-run: test mapping only, no downloads
    python scripts/09c_af2_recovery.py --config config/zorc_config.yaml --dry-run

    # Resume interrupted run (uses cached mappings + skips existing PDBs)
    python scripts/09c_af2_recovery.py --config config/zorc_config.yaml

Outputs:
    data/processed/af2_cache/                    AF2 PDB files (gitignored)
    data/processed/af2_cache/tair_uniprot_map.json  cached TAIR→UniProt mapping
    data/processed/07_protein_features.csv       PATCHED in place
    data/processed/08_zorc_feature_matrix.csv    REGENERATED (merged features)
    logs/09c_af2_recovery_report.txt
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ZORC P9c — AF2 structure recovery")
    p.add_argument("--config",   required=True, help="Path to zorc_config.yaml")
    p.add_argument("--dry-run",  action="store_true",
                   help="Test UniProt mapping only; skip downloads and patching")
    p.add_argument("--limit",    type=int, default=None,
                   help="Process only first N Tier1 proteins (for testing)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# UniProt mapping  (corrected from P7)
# ─────────────────────────────────────────────────────────────────────────────

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

def tair_to_uniprot(gene_id: str, session_cache: dict,
                    rate_delay: float = 0.35) -> str | None:
    """
    Map a TAIR gene ID to UniProt accession.

    Strategy (in order of reliability):
      1. xref:tair-{gene_id}   — direct TAIR cross-reference (most reliable)
      2. gene:{gene_id} AND organism_id:3702  — gene name search, no reviewed filter
      3. xref:tair-{gene_id_upper}            — uppercase variant

    Results cached in session_cache to avoid redundant API calls.
    Rate-limited to ~3 requests/second by default.
    """
    if gene_id in session_cache:
        return session_cache[gene_id]

    queries = [
        # Q1: TAIR cross-reference (primary — this is the correct approach)
        (f'{UNIPROT_SEARCH}?query={urllib.parse.quote(f"xref:tair-{gene_id}")}'
         f'&format=json&size=1&fields=accession&organism_id=3702'),
        # Q2: gene name fallback, no reviewed filter
        (f'{UNIPROT_SEARCH}?query={urllib.parse.quote(f"gene:{gene_id} AND organism_id:3702")}'
         f'&format=json&size=1&fields=accession'),
        # Q3: uppercase (some TAIR IDs stored capitalised differently)
        (f'{UNIPROT_SEARCH}?query={urllib.parse.quote(f"xref:tair-{gene_id.upper()}")}'
         f'&format=json&size=1&fields=accession&organism_id=3702'),
    ]

    accession = None
    for url in queries:
        try:
            time.sleep(rate_delay)
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
            results = data.get("results", [])
            if results:
                accession = results[0]["primaryAccession"]
                break
        except Exception:
            continue

    session_cache[gene_id] = accession  # cache even if None
    return accession


# ─────────────────────────────────────────────────────────────────────────────
# AF2 PDB download
# ─────────────────────────────────────────────────────────────────────────────

AF2_API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"

def get_af2_pdb_url(uniprot_id: str) -> str | None:
    """
    Query the AF2 API to get the actual pdbUrl for a UniProt ID.
    This is version-agnostic — works with v4, v6, or any future version.
    """
    api_url = AF2_API_TEMPLATE.format(uid=uniprot_id)
    try:
        time.sleep(0.1)
        with urllib.request.urlopen(api_url, timeout=15) as r:
            data = json.loads(r.read())
        if data:
            return data[0].get("pdbUrl")
    except Exception:
        pass
    return None


def fetch_af2_pdb(uniprot_id: str, cache_dir: Path,
                  retries: int = 3) -> Path | None:
    """
    Download AF2 PDB from EBI using the API endpoint to resolve the
    correct version URL (v4, v6, etc.). Skips if already cached.
    Returns local Path or None on failure.
    """
    # Check cache for any version
    existing = list(cache_dir.glob(f"{uniprot_id}_AF2_*.pdb"))
    if existing and existing[0].stat().st_size > 1000:
        return existing[0]

    # Get the real URL from the API
    pdb_url = get_af2_pdb_url(uniprot_id)
    if not pdb_url:
        return None

    # Extract version from URL for filename (e.g. v6)
    version = "v?"
    for part in pdb_url.split("-"):
        if part.startswith("model_"):
            version = part.replace("model_", "").replace(".pdb", "")
            break
    pdb_path = cache_dir / f"{uniprot_id}_AF2_{version}.pdb"

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(pdb_url, str(pdb_path))
            if pdb_path.stat().st_size > 1000:
                return pdb_path
            pdb_path.unlink(missing_ok=True)
        except Exception:
            if pdb_path.exists():
                pdb_path.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Structural feature extraction (static — single PDB frame)
# ─────────────────────────────────────────────────────────────────────────────

def extract_static_features(pdb_path: Path) -> dict:
    """
    Extract Rg, contact_density, packing_density, sasa_per_residue, n_residues
    from a single PDB file (AF2 or single BioEmu frame).
    Uses MDAnalysis — same approach as P7 extract_af2_features().
    Returns dict with feature values or {'error': reason}.
    """
    import MDAnalysis as mda
    from scipy.spatial.distance import cdist

    try:
        u = mda.Universe(str(pdb_path))
    except Exception as e:
        return {"error": f"mda_load_failed: {e}"}

    try:
        protein = u.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "no_protein_atoms"}

        ca     = protein.select_atoms("name CA")
        heavy  = protein.select_atoms("not name H*")
        n_res  = len(ca)

        if n_res < 10:
            return {"error": f"too_short: {n_res} residues"}

        # Radius of gyration
        rg = float(protein.radius_of_gyration())

        # Cα–Cα contact density (8 Å cutoff)
        ca_pos  = ca.positions
        dist_ca = cdist(ca_pos, ca_pos)
        np.fill_diagonal(dist_ca, np.inf)
        contacts_ca = int((dist_ca < 8.0).sum()) // 2
        contact_density = contacts_ca / n_res if n_res > 0 else 0.0

        # Heavy-atom packing density (6 Å cutoff)
        hv_pos    = heavy.positions
        dist_hv   = cdist(hv_pos, hv_pos)
        np.fill_diagonal(dist_hv, np.inf)
        contacts_hv = int((dist_hv < 6.0).sum()) // 2
        packing_density = contacts_hv / n_res if n_res > 0 else 0.0

        # SASA proxy: solvent-exposed atoms per residue
        # (MDAnalysis SASA requires freesasa; use heavy-atom count as proxy)
        sasa_per_residue = len(heavy) / n_res if n_res > 0 else 0.0

        return {
            "n_residues":       float(n_res),
            "rg_mean":          rg,
            "rg_std":           0.0,   # static — no ensemble
            "rg_cv":            0.0,   # static — no ensemble
            "contact_density":  contact_density,
            "packing_density":  packing_density,
            "sasa_per_residue": sasa_per_residue,
            "feature_source":   "af2_static",
            "error":            None,
        }

    except Exception as e:
        return {"error": f"feature_extraction_failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("ZORC — Phase 9c: AF2 Structure Recovery")
    print(f"Config: {args.config}")
    print(f"Timestamp: {ts}")
    if args.dry_run:
        print("  *** DRY-RUN MODE — no downloads, no patching ***")
    print("=" * 70)

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir   = Path(cfg.get("base_dir",
                    cfg.get("project_dir", "~/Documents/ZORC"))).expanduser()
    data_dir   = base_dir / "data" / "processed"
    logs_dir   = base_dir / "logs"
    af2_cache  = data_dir / "af2_cache"
    af2_cache.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pf_path    = data_dir / "07_protein_features.csv"
    fm_path    = data_dir / "08_zorc_feature_matrix.csv"
    rna_path   = data_dir / "04_rna_features.csv"
    map_cache  = af2_cache / "tair_uniprot_map.json"
    report_path = logs_dir / "09c_af2_recovery_report.txt"

    # ── Load persistent UniProt mapping cache ──────────────────────────────
    if map_cache.exists():
        with open(map_cache) as f:
            uniprot_cache = json.load(f)
        print(f"\n  Loaded {len(uniprot_cache)} cached TAIR→UniProt mappings")
    else:
        uniprot_cache = {}

    # ── 1. Identify Tier 1 targets ─────────────────────────────────────────
    print(f"\n[1/5] Loading protein features: {pf_path}")
    pf = pd.read_csv(pf_path)
    if "geneID" in pf.columns:
        pf = pf.rename(columns={"geneID": "gene_id"})

    # Targets: Tier 1 that previously failed AF2 fetch
    target_mask = (
        (pf["bioemu_tier"] == 1) &
        (pf["feature_source"].isin(["af2_not_found", "uniprot_not_found",
                                     "af2_skipped", None]) |
         pf["feature_source"].isna())
    )
    targets = pf[target_mask].copy()

    if args.limit:
        targets = targets.head(args.limit)
        print(f"  --limit active: processing first {args.limit} proteins")

    print(f"  Tier 1 targets for AF2 recovery: {len(targets)}")
    print(f"  (Already recovered: {(pf['feature_source'] == 'af2_static').sum()})")

    if len(targets) == 0:
        print("  Nothing to do — all Tier 1 proteins already have AF2 features.")
        return

    # ── 2. TAIR → UniProt mapping ──────────────────────────────────────────
    print(f"\n[2/5] Mapping TAIR IDs to UniProt accessions...")
    print(f"  (Rate-limited to ~3 req/s — estimated time: "
          f"~{len(targets) * 0.35 / 60:.1f} min)")

    # Only query genes not already in cache
    to_query = [g for g in targets["gene_id"].tolist() if g not in uniprot_cache]
    print(f"  Cache hits: {len(targets) - len(to_query)} / {len(targets)}")
    print(f"  New API queries: {len(to_query)}")

    n_mapped = 0
    for i, gene_id in enumerate(to_query):
        uid = tair_to_uniprot(gene_id, uniprot_cache)
        if uid:
            n_mapped += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(to_query):
            pct = (i + 1) / max(len(to_query), 1) * 100
            print(f"  Progress: {i+1}/{len(to_query)} queried "
                  f"({pct:.0f}%) — mapped so far: {n_mapped}", flush=True)
            # Save cache incrementally every 50 queries
            with open(map_cache, "w") as f:
                json.dump(uniprot_cache, f, indent=2)

    # Final cache save
    with open(map_cache, "w") as f:
        json.dump(uniprot_cache, f, indent=2)

    # Mapping summary
    mapped   = {g: uniprot_cache[g] for g in targets["gene_id"]
                if uniprot_cache.get(g) is not None}
    unmapped = [g for g in targets["gene_id"] if uniprot_cache.get(g) is None]
    print(f"\n  Mapped:   {len(mapped)} / {len(targets)} "
          f"({len(mapped)/len(targets)*100:.1f}%)")
    print(f"  Unmapped: {len(unmapped)}")

    if args.dry_run:
        print("\n  [DRY-RUN] Stopping before downloads.")
        print(f"  UniProt mapping cache saved: {map_cache}")
        if unmapped[:10]:
            print(f"  Sample unmapped: {unmapped[:10]}")
        return

    # ── 3. Download AF2 PDBs ───────────────────────────────────────────────
    print(f"\n[3/5] Downloading AF2 PDBs from EBI...")
    print(f"  Cache dir: {af2_cache}")

    n_downloaded = 0
    n_cached     = 0
    n_dl_failed  = 0
    pdb_paths    = {}   # gene_id → Path

    for gene_id, uniprot_id in mapped.items():
        pdb = fetch_af2_pdb(uniprot_id, af2_cache)
        if pdb:
            pdb_paths[gene_id] = pdb
            if n_downloaded + n_cached == 0:
                pass
            # Count as cached if file existed before this run
            n_downloaded += 1
        else:
            n_dl_failed += 1

        done = n_downloaded + n_dl_failed
        if done % 50 == 0 or done == len(mapped):
            print(f"  Progress: {done}/{len(mapped)} — "
                  f"downloaded: {n_downloaded}, failed: {n_dl_failed}", flush=True)

    print(f"\n  PDB download summary:")
    print(f"    Downloaded: {n_downloaded}")
    print(f"    Failed:     {n_dl_failed}")
    print(f"    Unmapped:   {len(unmapped)}")

    if n_downloaded == 0:
        print("\n  WARNING: No PDBs downloaded. Check network and UniProt IDs.")
        print("  Run with --dry-run first to validate mapping.")
        return

    # ── 4. Extract structural features ────────────────────────────────────
    print(f"\n[4/5] Extracting structural features from {n_downloaded} PDBs...")

    feat_records = []
    n_feat_ok    = 0
    n_feat_fail  = 0

    for gene_id, pdb_path in pdb_paths.items():
        feats = extract_static_features(pdb_path)
        feats["gene_id"] = gene_id
        feat_records.append(feats)
        if feats.get("error") is None:
            n_feat_ok += 1
        else:
            n_feat_fail += 1

    feat_df = pd.DataFrame(feat_records)
    print(f"  Feature extraction: {n_feat_ok} OK, {n_feat_fail} failed")

    if n_feat_fail > 0:
        fails = feat_df[feat_df["error"].notna()][["gene_id", "error"]].head(10)
        print(f"  Sample failures:\n{fails.to_string(index=False)}")

    # ── 5. Patch 07_protein_features.csv ──────────────────────────────────
    print(f"\n[5/5] Patching {pf_path.name}...")

    # Backup original
    backup = pf_path.with_suffix(".csv.pre_af2_recovery")
    if not backup.exists():
        pf.to_csv(backup, index=False)
        print(f"  Backup saved: {backup.name}")

    # Apply recovered features row by row
    af2_cols = ["n_residues", "rg_mean", "rg_std", "rg_cv",
                "contact_density", "packing_density", "sasa_per_residue",
                "feature_source", "error"]

    pf_updated = pf.copy()
    if "gene_id" not in pf_updated.columns and "geneID" in pf_updated.columns:
        pf_updated = pf_updated.rename(columns={"geneID": "gene_id"})

    n_patched = 0
    for _, row in feat_df.iterrows():
        gid = row["gene_id"]
        if row.get("error") is not None:
            continue  # don't patch with failed extractions
        mask = pf_updated["gene_id"] == gid
        if mask.sum() == 0:
            continue
        for col in af2_cols:
            if col in row.index and col in pf_updated.columns:
                pf_updated.loc[mask, col] = row[col]
        n_patched += 1

    # Restore original column name if needed
    pf_updated = pf_updated.rename(columns={"gene_id": "geneID"})
    pf_updated.to_csv(pf_path, index=False)
    print(f"  Patched {n_patched} proteins in {pf_path.name}")

    # ── Regenerate feature matrix ──────────────────────────────────────────
    print(f"\n  Regenerating feature matrix: {fm_path.name}...")
    fm = pd.read_csv(fm_path)
    if "geneID" in fm.columns:
        fm = fm.rename(columns={"geneID": "gene_id"})

    pf_merge = pf_updated.rename(columns={"geneID": "gene_id"})
    merge_cols = ["gene_id"] + [c for c in af2_cols
                                if c in pf_merge.columns and c != "error"]

    # Drop old protein feature columns from fm, re-merge from patched pf
    fm_drop = fm.drop(columns=[c for c in merge_cols[1:] if c in fm.columns],
                      errors="ignore")
    fm_new = fm_drop.merge(pf_merge[merge_cols], on="gene_id", how="left")

    fm_new = fm_new.rename(columns={"gene_id": "geneID"})
    fm_new.to_csv(fm_path, index=False)
    print(f"  Feature matrix regenerated: {fm_path.name}")

    # ── Report ─────────────────────────────────────────────────────────────
    total_t1    = len(pf[pf["bioemu_tier"] == 1])
    pf_check    = pd.read_csv(pf_path)
    if "geneID" in pf_check.columns:
        pf_check = pf_check.rename(columns={"geneID": "gene_id"})
    af2_ok_now  = (pf_check["feature_source"] == "af2_static").sum()
    still_miss  = total_t1 - af2_ok_now

    print(f"\n{'='*70}")
    print(f"AF2 RECOVERY SUMMARY")
    print(f"{'='*70}")
    print(f"  Tier 1 total         : {total_t1}")
    print(f"  Mapped to UniProt    : {len(mapped)} ({len(mapped)/total_t1*100:.1f}%)")
    print(f"  PDBs downloaded      : {n_downloaded}")
    print(f"  Features extracted   : {n_feat_ok}")
    print(f"  Proteins patched     : {n_patched}")
    print(f"  AF2 static in P7 now : {af2_ok_now} / {total_t1}")
    print(f"  Still missing        : {still_miss}")
    print(f"  Unmapped TAIR IDs    : {len(unmapped)}")
    if unmapped[:5]:
        print(f"    (sample: {unmapped[:5]})")
    print(f"{'='*70}")
    print(f"\n  Imputation pool reduction:")
    print(f"    Before: 748 samples with NaN dynamic features (49.5%)")
    print(f"    After:  ~{max(0, 748 - n_patched)} samples with NaN "
          f"({max(0, 748 - n_patched) / 1510 * 100:.1f}%)")
    print(f"\n  Next steps:")
    print(f"    1. git commit this patch")
    print(f"    2. Rerun P9b (XGBoost) on updated feature matrix")
    print(f"    3. Compare SHAP ranks for rg_mean / contact_density")
    print(f"{'='*70}")

    with open(report_path, "w") as f:
        f.write(f"ZORC Phase 9c — AF2 Recovery Report\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"Tier 1 total: {total_t1}\n")
        f.write(f"Mapped to UniProt: {len(mapped)}\n")
        f.write(f"PDBs downloaded: {n_downloaded}\n")
        f.write(f"Features extracted OK: {n_feat_ok}\n")
        f.write(f"Proteins patched: {n_patched}\n")
        f.write(f"AF2 static in P7 now: {af2_ok_now}\n")
        f.write(f"Still missing: {still_miss}\n\n")
        f.write("Unmapped TAIR IDs:\n")
        for g in unmapped:
            f.write(f"  {g}\n")
    print(f"\n  Report: {report_path}")
    print("\n✓ Phase 9c complete.")


if __name__ == "__main__":
    main()
