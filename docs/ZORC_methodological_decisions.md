# ZORC — Methodological Decisions Log

**Project:** Zip-code Of RNAs that Condense  
**Lab:** MoschouLab / IMBB-FORTH / University of Crete  
**ERC Grant:** PLANTEX (Consolidator Grant)  
**Lead analyst:** José Moya-Cuevas  
**Last updated:** April 2026

This document records all non-obvious methodological decisions made during
pipeline development, with rationale. Intended as permanent context for
Claude Code sessions, code reviewers, and manuscript Methods section.

---

## 1. Dataset Construction (P1 — Coregulon)

### 1.1 Data sources
- **RNA (positive/negative labels):** Liu et al. 2024, *Plant Cell* — T-RIP
  enriched/depleted gene lists (Suppl. Data Set 2).
  DOI: 10.1093/plcell/koae015
- **Protein (coregulon filter):** Liu et al. 2023, *EMBO J* — APEAL proteomics
  (AP + PDL purification, DCP1-TurboID vs GFP-TurboID).
  DOI: 10.15252/embj.2022111885

### 1.2 Positive class definition
**Decision:** RNA enriched in T-RIP (NS or HS) AND protein log2FC ≥ 0 in ANY
of {AP_NS, AP_HS, PDL_NS, PDL_HS} — union of methods, not intersection.

**Rationale:** Using union (any method/condition) rather than intersection
maximizes dataset size while still requiring protein confirmation. The
protein filter reduces false positives from T-RIP background. The log2FC ≥ 0
threshold (rather than a stricter cutoff like ≥ 1) is deliberately relaxed
to retain borderline cases that are biologically valid but below arbitrary
fold-change cutoffs. **Threshold is exposed in `config.yaml` as
`protein_enrichment_threshold` and can be changed without modifying code.**

**Alternative considered:** RNA-only (no protein filter) → 3,874 positives.
Rejected for first iteration because the coregulon criterion (mRNA + protein
cogulate in P-bodies) is the biologically most meaningful positive definition.
Can be tested via `protein_enrichment_threshold: null` in config.

### 1.3 Negative class definition
**Decision:** RNA depleted from P-bodies (NS or HS), regardless of protein
detection status.

**Rationale:** Depleted RNAs are actively excluded from P-bodies — the
strongest possible negative signal. Protein status is irrelevant because the
prediction target is RNA localization, not protein presence.

### 1.4 Conflict resolution (genes in both enriched and depleted lists)
**Decision:** `strongest` — assign to the class with the higher absolute
log2FC signal.

**Rationale:** 10 genes appeared in both lists (different conditions).
Using the dominant signal rather than dropping them preserves dataset size
and is more informative than discarding ambiguous cases.

### 1.5 Final dataset
- Positives (class=1): 884 genes
- Negatives (class=0): 688 genes
- Total: 1,572 genes
- Class ratio: 1:0.78 (near-balanced — no SMOTE needed)

---

## 2. Isoform Selection Strategy (P2)

### 2.1 rMATS-guided vs canonical fallback
**Decision:** For genes with rMATS splicing events, use the transcript
containing the alternatively spliced region (coordinate intersection with GTF).
For all other genes, use the Ensembl_canonical transcript.

**Rationale:** T-RIP enrichment is measured at the gene level, not isoform
level. However, for genes with detected AS events, the specific isoform
carrying the event is the biologically relevant unit for ZORC features.
The canonical transcript is the best representative for genes without
isoform-specific data.

**Result:** 303 genes (20%) received rMATS-guided isoforms; 1,201 (80%)
used canonical fallback.

### 2.2 Dropped genes (68)
Genes absent from TAIR10.59 GTF: pseudogenes, transposable elements, tRNAs,
and one Araport11-only locus (AT2G36485). These lack meaningful transcript
annotations and were excluded rather than assigned arbitrary sequences.

### 2.3 rMATS file parsing fix
rMATS output files use double-quoted GeneID fields. Some gene symbol fields
(e.g. "GAMMA-CA1") contain embedded tabs, causing pandas to miscount columns.
Fix: `quoting=csv.QUOTE_NONE` + manual quote stripping.

---

## 3. Sequence Fetching (P3)

### 3.1 Tool choice
`gffread` (v0.12.7) for spliced mRNA extraction (`-w`) and CDS translation
(`-y`). Standard, reproducible, handles all GTF feature types correctly.

### 3.2 Isoform vs canonical protein — CRITICAL DISTINCTION
**The AGI code from APEAL proteomics is used ONLY as a matching key** to
confirm P-body protein presence. All downstream sequence features (P2–P7)
use the **specific isoform sequence** identified by rMATS or the canonical
transcript — NOT the canonical protein sequence from databases.

This is the core biological rationale of ZORC: predicting isoform-specific
P-body enrichment, not gene-level presence.

### 3.3 QC flags
45 sequences flagged (no_CDS, short_protein <30aa, internal_stop). Included
in FASTAs with `|QC_FAIL:reason` suffix for transparency. Used for RNA
features but excluded from BioEmu/AIUPred (protein features).

---

## 4. RNA Feature Engineering (P4)

### 4.1 RNAfold configuration
**Decision:** MFE-only (`--noPS`, no `-p` partition function flag),
chunked batches of 10 sequences, max_len=3000 nt.

**Rationale:** Partition function (`-p`) is O(n³) vs MFE O(n²). For
~1,500 sequences, `-p` caused timeout in all tested batch sizes. MFE,
fraction_paired, and n_stemloops provide sufficient structural information
for the RF model. Ensemble free energy dropped as feature.

**68 sequences > 3000 nt** excluded from RNAfold (structure computation only).
Composition + m6A + UTR features computed for all 1,501. `rnafold_status`
column records per-sequence status. `--rnafold-maxlen` argument exposes
threshold for reproducibility.

**Revisit:** if MFE ranks high in SHAP feature importance (P9), consider
computing MFE for first 500 nt of long sequences (most regulatory).

### 4.2 Length normalization
Raw length features (mrna_length, cds_length, etc.) show strong separation
between classes (ratio up to 1.70) but this is confounded with general
transcript complexity. Length-normalized features added post-hoc:
- `n_stemloops_per_kb`: stem-loop density (the raw ratio was a length artifact)
- `utr3_fraction`, `utr5_fraction`, `cds_fraction`: structural proportions

**Key finding:** after normalization, real signal is `cds_fraction` (1.14×)
and `utr3_fraction` (0.80×). Positives dedicate more mRNA to CDS and have
proportionally shorter 3'UTRs. This will be monitored in SHAP analysis.

### 4.3 RNAfold dot-plot files
RNAfold 2.7.2 generates `*_dp.ps` PostScript files in the working directory
regardless of `--noPS`. Added `*.ps` and `*.eps` to `.gitignore`.

---

## 5. IDR Prediction & BioEmu Tier Assignment (P5)

### 5.1 AIUPred configuration
AIUPred v2.0 (Erdos & Dosztanyi). Used `--force-cpu` to bypass CUDA/NCCL
conflict within the `bioemu` conda env. CPU runtime for 1,456 proteins:
~15 min in 15 batches of 100. Negligible time cost vs GPU benefit.

**AIUPred v2.0 output format:** sequence headers use `#>TX_ID` format
(not separate `#` and `>` lines as in v1.x). Parser updated accordingly.

### 5.2 BioEmu tier thresholds
| Tier | IDR% | BioEmu | N proteins |
|------|------|--------|-----------|
| 1 | < 10% | AlphaFold2 static only | 698 (48%) |
| 2 | 10–30% | 50 conformations | 328 (22%) |
| 3 | ≥ 30% | 100 conformations | 431 (30%) |

**Rationale:** Tier boundaries based on established IDR literature thresholds.
Tier 1 proteins are well-structured; BioEmu adds minimal information over AF2.
Tier 3 proteins have significant IDR and require larger ensembles to sample
conformational diversity adequately. Thresholds exposed in `config.yaml`
under `bioemu_tiers`.

---

## 6. BioEmu Conformational Sampling (P6)

### 6.1 Sampling strategy
One FASTA per protein (BioEmu requirement). Output: `topology.pdb` +
`samples.xtc` per protein directory. Checkpointing after each protein
via `checkpoint.json` — safe to interrupt and resume.

### 6.2 Known BioEmu bugs and workarounds

**Bug 1 — unphysical_filter_bug:** When all sampled conformations are
filtered as structurally invalid, BioEmu crashes trying to save
`samples_unphysical.xtc` because `pathlib.with_suffix('_unphysical.xtc')`
raises `ValueError` (suffix must start with `.`). Fix: rerun with
`--filter_samples=False`. Affected 36 proteins, all recovered.

**Bug 2 — timeout (1h limit):** Large proteins (>~960 aa with high IDR)
exceed 1h GPU time. Rerun with `--timeout 10800` (3h). 5 proteins pending.

**Bug 3 — OOM:** Proteins >3000 aa exceed 25.4 GB VRAM on RTX A5000.
AT2G28290 (3575 aa) and AT2G45540 (3001 aa) marked as `excluded` with
`oom_protein_too_large_for_gpu`. Treated as Tier 1 (AF2 features only)
in P7/P8.

### 6.3 Final BioEmu status
- Completed: 752/759 (99.1%)
- Timeout retry pending: 5
- OOM excluded: 2
- Average runtime: Tier 2 ~8 min/protein, Tier 3 ~14.6 min/protein
- Total GPU time: ~5 days continuous on RTX A5000

### 6.4 Large binary files
`*.xtc` and `*.pdb` outputs excluded from git repo (too large, regenerable).
Only `checkpoint.json` and session reports committed. See `.gitignore`.

---

## 7. Class Imbalance Strategy

**Decision:** Class weights (`rf_class_weight: "balanced"` in config),
NOT SMOTE or other oversampling.

**Rationale:** BioEmu features are physically meaningful — interpolating
between protein conformational ensembles has no biological interpretation.
SMOTE on these features would generate physically nonsensical synthetic
samples. Class weights tell the model that misclassifying the minority
class costs more, without generating synthetic data.

**Final class ratio:** 1:0.78 (near-balanced after coregulon filter).
Class weights will have minimal effect but are kept for correctness.

---

## 8. Anti-leakage Split Strategy (P8 — pending)

**Decision:** CD-HIT clustering at 40% protein identity threshold.
Entire clusters assigned to one partition. Never split clusters across
train/CV/test.

**Rationale:** Arabidopsis gene families have many paralogs with high
sequence similarity. Standard random splits would leak information between
train and test sets via paralog pairs. The 40% identity threshold is
conservative and standard for ML on protein sequences.

**Split ratio:** 70% train / 15% CV / 15% test, stratified by class
within cluster assignments.

---

## 9. High-Confidence Validation Set

25 genes curated by lab colleagues (independent cross-reference of the
same two datasets). All 25 present in ZORC coregulon as positives (100%
overlap). Used as internal validation set for RF model (P9) and as
primary candidates for Xenium 10X probe design (P11).

File: `data/processed/colleague_high_confidence_set.csv`
Source: `data/raw/RSBs_DCP1_related_under_heat_stress.xlsx`

Notable genes: PAT1 (AT3G22270, AT4G14990), LBA1/UPF1 (AT5G47010),
EF1B (AT5G53330), VPS28 (AT4G05000), ATG6 (AT3G61710).

---

## 10. Conda Environment Architecture

Two environments, deliberately separated:

| Env | Purpose | Key packages |
|-----|---------|-------------|
| `zorc_pipeline` | Main pipeline (P1–P5, P7–P9) | pandas, scikit-learn, RNAfold, pyranges, gffread |
| `bioemu` | Conformational sampling only (P6) | BioEmu v1.1, torch 2.5.1+cu121, AIUPred v2.0 |

**Rationale:** BioEmu has strict CUDA/torch version requirements that
conflict with general scientific Python packages. Keeping envs separate
prevents dependency conflicts and makes the pipeline more reproducible.
In the future Snakemake workflow, each rule will specify its env via
`conda:` directive.

Environment files: `envs/zorc_pipeline.yml`, `envs/bioemu_ref.yml`

---

## 11. Snakemake Conversion Plan (P10)

Scripts are numbered (`01_`, `02_`...) and accept `--config config/zorc_config.yaml`
for direct conversion to Snakemake rules. All parameters are in `config.yaml`.
Each script auto-adds its output keys to config if missing.

Planned conversion: after RF model validation (P9), before GitHub publication.
Each script → one Snakemake rule with `conda:` env specified.

---

*This document should be updated at the end of each major pipeline phase.*
*For session context (data summaries, current pipeline state), see*
*`docs/ZORC_continuation_prompt.md`.*
