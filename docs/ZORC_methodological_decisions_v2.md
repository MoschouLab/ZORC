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

**Rationale:** Using union maximizes dataset size while still requiring protein
confirmation. The log2FC ≥ 0 threshold is deliberately relaxed to retain
borderline cases. **Threshold exposed in `config.yaml` as
`protein_enrichment_threshold`.**

**Alternative:** RNA-only (no protein filter) → 3,874 positives.
Testable via `protein_enrichment_threshold: null` in config.

### 1.3 Negative class definition
RNA depleted from P-bodies (NS or HS), regardless of protein detection status.

### 1.4 Conflict resolution
10 genes appeared in both enriched and depleted lists.
Strategy: `strongest` — assign to class with higher absolute log2FC.

### 1.5 Final dataset
- Positives: 884, Negatives: 688, Total: 1,572
- Class ratio: 1:0.78 (near-balanced)

---

## 2. Isoform Selection Strategy (P2)

### 2.1 rMATS-guided vs canonical fallback
For genes with rMATS events → transcript containing the alternatively spliced
region (coordinate intersection with GTF).
For all other genes → Ensembl_canonical transcript.
Result: 303 rMATS-guided (20%), 1,201 canonical (80%).

### 2.2 Dropped genes (68)
Pseudogenes, transposable elements, tRNAs, and AT2G36485 (Araport11-only).
Excluded: no meaningful GTF annotation.

### 2.3 CRITICAL: Isoform vs canonical protein
**The AGI code from APEAL is used ONLY as a matching key.** All downstream
features use the SPECIFIC isoform sequence from T-RIP, not the canonical
protein. This is the biological core of ZORC.

### 2.4 rMATS parsing fix
rMATS files use tab-embedded gene symbols (e.g. "GAMMA-CA1"). Fix:
`quoting=csv.QUOTE_NONE` + manual quote stripping.

---

## 3. Sequence Fetching (P3)

gffread v0.12.7, `-w` (spliced mRNA) + `-y` (translated CDS).
45 QC-flagged sequences included with `|QC_FAIL:reason` suffix.

---

## 4. RNA Feature Engineering (P4)

### 4.1 RNAfold configuration
MFE-only (no `-p` partition function), chunked batches of 10, max_len=3000 nt.
68 sequences >3000 nt excluded from structure computation only.
Rationale: O(n²) MFE vs O(n³) partition function.

### 4.2 Length normalization
Raw length features show ratio up to 1.70 (cds_length pos vs neg).
Length-normalized features added: `n_stemloops_per_kb`, `utr3_fraction`,
`utr5_fraction`, `cds_fraction`.
After normalization: real signal is `cds_fraction` (1.14×) and
`utr3_fraction` (0.80×). n_stemloops ratio was a length artifact.

### 4.3 RNAfold dot-plot files
RNAfold 2.7.2 generates `*_dp.ps` files regardless of `--noPS`.
Added `*.ps` and `*.eps` to `.gitignore`.

---

## 5. IDR Prediction & BioEmu Tier Assignment (P5)

### 5.1 AIUPred configuration
AIUPred v2.0. `--force-cpu` bypasses CUDA/NCCL conflict in bioemu env.
v2.0 output format: `#>TX_ID` headers (not separate `#` and `>` lines).

### 5.2 BioEmu tier thresholds
| Tier | IDR% | BioEmu | N proteins |
|------|------|--------|-----------|
| 1 | < 10% | AlphaFold2 static only | 698 (48%) |
| 2 | 10–30% | 50 conformations | 328 (22%) |
| 3 | ≥ 30% | 100 conformations | 431 (30%) |

---

## 6. BioEmu Conformational Sampling (P6)

### 6.1 Known bugs and workarounds
**unphysical_filter_bug:** `pathlib.with_suffix('_unphysical.xtc')` crashes
when all conformations are filtered. Fix: `--filter_samples=False`.
Affected 36 proteins, all recovered.

**timeout_1h:** Large proteins >960aa exceed 1h. Fix: `--timeout 10800` (3h).
5 proteins recovered.

**OOM:** Proteins >3000aa exceed 25.4GB VRAM.
AT2G28290.5 (3575aa) and AT2G45540.2 (3001aa) marked `excluded`,
treated as Tier 1 in P7/P8.

### 6.2 Final status
757/759 completed (99.7%). Total GPU time: ~5 days on RTX A5000.
Large binaries (*.xtc, *.pdb) excluded from git repo.

---

## 7. Class Imbalance Strategy

**Decision:** Class weights (`balanced`), NOT SMOTE.
BioEmu features are physically meaningful; interpolating between protein
conformational ensembles has no biological interpretation.
Final ratio 1:0.78 — class weights have minimal effect but are kept
for correctness.

---

## 8. Protein Feature Extraction (P7)

MDAnalysis 2.10.0 for trajectory analysis of BioEmu `.xtc` + `.pdb`.
Features: RMSF (mean, std, max, Nterm50, Cterm50), Rg (mean, std, CV),
contact_density (Cα-Cα 8Å), packing_density (heavy atoms 6Å),
sasa_per_residue.

AF2 fallback for Tier 1 via EBI API — **silently failed for most proteins
in current implementation.** See Section 13 (pending fix).

Runtime: 8.8 min for 1,457 proteins.

Key findings:
- rmsf_mean: pos=21.3Å vs neg=17.0Å (1.25×)
- rg_mean: pos=41.9Å vs neg=32.7Å (1.28×)
- contact_density: pos≈neg (not discriminant)

---

## 9. Anti-Leakage Split Strategy (P8)

CD-HIT at 40% protein identity. 1,299 clusters from 1,457 sequences.
Split: 70/15/15, stratified by majority class per cluster.
Result: train=1,064 / val=212 / test=234.

CD-HIT header parsing fix: regex `[^\s\.]+` truncated transcript IDs at dot.
Fixed to `[^\s]+` with `.rstrip('.')`.

---

## 10. Random Forest Model (P9)

### 10.1 Configuration
```
n_estimators=500, class_weight='balanced', max_features='sqrt'
min_samples_leaf=2, oob_score=True, random_state=42
```

### 10.2 Imputation strategy
Dynamic BioEmu features (rmsf_*, rg_*, contact_density, pass_rate):
**global median imputation** + binary flag `has_bioemu` (1=BioEmu, 0=Tier1).
Rationale: has_bioemu flag allows RF to learn that Tier1 proteins
(IDR<10%) have a different conformational profile.

### 10.3 Results (baseline)
```
OOB: 0.7397
Val:  AUROC=0.799  AUPRC=0.831  F1-macro=0.725
Test: AUROC=0.797  AUPRC=0.847  F1-macro=0.738
HC validation (25 genes): 24/25 (96%)
```

### 10.4 SHAP findings
Top features: mrna_length, cds_length, n_residues, di_CG, di_UA,
utr3_au_content, utr3_length.

**Critical issue:** `idr_percent`, `rmsf_mean`, `rg_mean`, `rg_cv` absent
from Top 20 despite 1.25–1.28× class separation.
Root cause: 49.5% imputation dilutes SHAP contributions by ~50%.
754/1,510 Tier 1 proteins have global median substituted for real values.

### 10.5 Known weaknesses
1. **Length confound:** mrna_length and cds_length are top 2 features.
   Need length-only baseline comparison.
2. **Label quality:** T-RIP captures DCP1-proximal RNAs, not exclusively
   P-body residents (may include stress granule mRNAs).
3. **49.5% imputed dynamic features** — most critical issue for next iteration.
4. Train AUROC=1.0 (expected overfitting for unregularised RF; OOB is honest).

---

## 11. Planned Next Steps (post-P9)

### 11.1 XGBoost benchmark (IMMEDIATE NEXT STEP)
Script: `09b_xgboost.py` (to be written)
- Handles NaN natively — no imputation needed for dynamic features
- If rmsf_mean/rg_mean rank higher in XGBoost SHAP → confirms imputation bias
- Expected: +0.03–0.06 AUROC over RF baseline
- SHAP via `shap.TreeExplainer` — same interpretability as RF

### 11.2 AF2 structure recovery
Fix P7 UniProt ID mapping → download AF2 PDBs for 698 Tier 1 proteins
→ recover real Rg/contact_density without additional GPU compute.

### 11.3 BioEmu for Tier 1 (conditional)
Only if XGBoost SHAP confirms dynamic features are critical AND
AF2 static features are insufficient. ~3-4 days GPU.

### 11.4 Feature engineering
- `rmsf_nterm50 / rmsf_cterm50` ratio
- `idr_percent × rrach_per_kb` interaction term
- `cds_length_per_exon`

### 11.5 Hyperparameter tuning
After dataset completion. Grid search over max_depth, min_samples_leaf,
max_features. Expected: +0.02–0.04 AUROC.

### 11.6 Alternative models
**XGBoost** (Step 11.1 above) — primary alternative.
**MLP** — possible but risky at n=1,510; SHAP via KernelExplainer (slow).
Not recommended as primary model.

---

## 12. Conda Environment Architecture

| Env | Purpose | Key packages |
|-----|---------|-------------|
| `zorc_pipeline` | Main pipeline (P1–P5, P7–P9) | pandas, scikit-learn, MDAnalysis, RNAfold, pyranges, gffread, shap, xgboost |
| `bioemu` | Conformational sampling (P6) | BioEmu v1.1, torch 2.5.1+cu121, AIUPred v2.0 |

Environment files: `envs/zorc_pipeline.yml`, `envs/bioemu_ref.yml`

---

## 13. Pending Technical Fixes

### 13.1 AF2 API fetch failure (P7)
P7 `tair_to_uniprot()` function failed silently for most Tier 1 proteins.
Result: AF2 features missing for 698/1,457 proteins.
Fix needed: diagnose UniProt REST API response, handle rate limiting,
or use pre-downloaded Arabidopsis AF2 proteome from EBI bulk download.
URL: https://alphafold.ebi.ac.uk/download (Arabidopsis thaliana proteome)

### 13.2 RNAfold for sequences >3000 nt
68 sequences excluded from MFE computation.
Option: compute MFE for first 500 nt (most regulatory region).
Revisit if mfe_per_nt ranks high in SHAP after dataset completion.

---

## 14. Snakemake + GitHub + Claude Code Plan (P10)

### 14.1 VS Code + Claude Code installation
**Install Claude Code on ThinkStation P5 when starting P10:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```
After installation, run `claude` in terminal and authenticate.
Then open the ZORC project directory: `cd ~/Documents/ZORC && claude`

Claude Code will have direct file system access to the repo and can:
- Convert numbered scripts to Snakemake rules directly
- Write the complete README.md
- Configure GitHub Actions for CI
- Handle all git operations

**Important:** Claude Code does NOT have access to claude.ai conversation
history. Always open `docs/ZORC_continuation_prompt.md` and
`docs/ZORC_methodological_decisions.md` at the start of each Claude Code
session to provide full project context.

### 14.2 Snakemake conversion plan
Scripts are numbered and accept `--config config/zorc_config.yaml`.
Each script → one Snakemake rule with `conda:` env specified.
Two env files: `envs/zorc_pipeline.yml` (most rules) +
`envs/bioemu.yml` (P5 AIUPred, P6 BioEmu).

Target: convert after RF model validation, before GitHub publication.

### 14.3 FAIR publication checklist
- [ ] Snakemake workflow (`Snakefile`)
- [ ] `environment.yml` files for both envs
- [ ] Complete `README.md` with installation + usage
- [ ] `config/zorc_config.yaml` as single parameter entry point
- [ ] GitHub repository: MoschouLab/ZORC
- [ ] Zenodo DOI via GitHub release
- [ ] Data availability statement (raw data in public repositories)

---

## 15. High-Confidence Validation Set

25 genes curated by lab colleagues (independent cross-reference of the
same two datasets). All 25 present in ZORC coregulon as positives.

File: `data/processed/colleague_high_confidence_set.csv`
Source: `data/raw/RSBs_DCP1_related_under_heat_stress.xlsx`

RF performance on these 25: 24/25 correct (96%).
Only miss: AT3G55280 (RPL23aB), P(pos)=0.42 — biologically ambiguous.

Notable genes: PAT1 (AT3G22270, AT4G14990), LBA1/UPF1 (AT5G47010),
EF1B (AT5G53330), VPS28 (AT4G05000), ATG6 (AT3G61710).

Primary candidates for spatial transcriptomics validation (P13).

---

*This document should be updated at the end of each major pipeline phase.*
*For session context and pipeline state, see `docs/ZORC_continuation_prompt.md`.*
