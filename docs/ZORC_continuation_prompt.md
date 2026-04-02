# ZORC Pipeline — Continuation Prompt
## Session context: MoschouLab T-RIP RNA-seq isoform project (PLANTEX / ERC ConG)

**User:** Pepe (José Moya-Cuevas), lead bioinformatics postdoc, MoschouLab / IMBB-FORTH, Rethymno, Crete.
**PI:** Panagiotis Moschou
**Project dir:** `~/Documents/TRIP-isoform-lncoding-pipeline/`
**Conda envs:** `trip_rnaseq` (main pipeline), `bioemu` (BioEmu v1.1 + AIUPred, torch 2.5.1+cu121)
**Hardware:** ThinkStation P5 (GPU, CUDA 12.2)

---

## Dataset overview

### T-RIP RNA-seq (P-body associated RNAs)
- Platform: Illumina NovaSeq 150bp PE, n=2 replicates per condition
- Design: DCP1-TurboID vs GFP-TurboID × NS (22°C) vs HS (37°C)
- BAMs: `04_results/03_bam/trip_clean/` (a9-a16, ~97% rRNA contamination — use with caution)
- rMATS results (DCP1 vs GFP): `04_results/05c_rmats/NS/` and `HS/`

### Total transcriptome RNA-seq
- Platform: MGI/BGI DNBSEQ ~100bp PE, ~97% unique mapping, NO rRNA problem
- Samples: 489_NS (n=3), 489_HS (n=3), 490_NS (n=3), 490_HS (n=2, missing rep3)
- BAMs: `04_results/03_bam/transcriptome/` (11 BAMs, all indexed)
- rMATS NS vs HS results: `04_results/06_rmats_transcriptome/NS_vs_HS/`

### Key reference files
- GTF: `05_reference/Arabidopsis_thaliana.TAIR10.59.gtf`
- Genome FASTA: `05_reference/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa`
- AIUPred: `~/Documents/bioemu_analysis/AIUPred/aiupred.py`

---

## Completed analyses (Phase 5H and beyond)

### 9 NS↔HS inversion genes (from T-RIP rMATS, DCP1 vs GFP)
Saved in: `04_results/05f_rmats_candidates/NS_HS_genuine_inversions.csv`

| GeneID | Symbol | Event | ΔΨ_NS | ΔΨ_HS |
|---|---|---|---|---|
| AT2G43680 | IQD14 | A3SS | -0.412 | +1.000 |
| AT3G26570 | PHT2 | RI | +0.376 | -0.901 |
| AT5G61020 | ECT3 | A3SS | +0.371 | -0.500 |
| AT4G32285 | AT4G32285 | RI | -0.425 | +0.319 |
| AT3G17040 | HCF107 | A3SS | -0.385 | +0.344 |
| AT4G38760 | AT4G38760 | A3SS | -0.500 | +0.220 |
| AT1G48210 | AT1G48210 | A3SS | +0.145 | -0.505 |
| AT4G13940 | HOG1 | RI | -0.136 | +0.168 |
| AT2G32700 | LUH | A3SS | +0.067 | -0.218 |
| AT2G44140 | ATG4A | DTU | — | — |

### T-RIP vs Total transcriptome ΔΨ classification
Saved in: `04_results/06_rmats_transcriptome/NS_vs_HS/gene_classification_TRIP_vs_transcriptome.csv`

- **Class A (6/9) — ACTIVE P-body selectivity:** HOG1, IQD14, PHT2, HCF107, AT4G32285, AT1G48210
- **Class B (2/9) — OPPOSITE-direction selectivity:** ECT3, AT4G38760
- **Class C (1/9) — Transcriptome-driven:** LUH (ATG4A separate, DTU event)

### BioEmu conformational ensemble (completed)
Results in: `04_results/05h_isa_consequences/bioemu_ECT3/` and `bioemu_batch2/`

| Protein | n_conformations | Key finding |
|---|---|---|
| ECT3.1 (+LQ) | 14 | YTH domain +46.7% RMSF vs ECT3.2 — allosteric IDR→YTH effect |
| ECT3.2 (-LQ) | 10 | YTH more rigid; stronger m6A binding predicted |
| IQD14 | 0 (failed) | 91.5% IDR — extreme disorder confirmed |
| HOG1.1 canonical | 192 | Highly ordered kinase |
| HOG1.2 RI isoform | 182 | N-term +68% RMSF vs canonical |
| ATG4A.1 (+IDR, P-body enriched) | 67 | RMSF N-term = 41.77 Å — massive disorder |
| ATG4A.4 (-IDR, P-body excluded) | 180 | Compact, ordered; Rg = 22.03 ± 0.78 Å |

### AIUPred IDR analysis
Saved in: `04_results/05h_isa_consequences/AIUPred_IDR_summary.csv`
23 isoforms, 10 genes. Key values:
- IQD14: 91.5% IDR (highest — LLPS driver)
- ATG4A: ΔIDR = -10.5% (.1 vs .4) — only gene where IDR directly predicts P-body partitioning
- HOG1: RI isoform gains +11% IDR vs canonical

### Total transcriptome AS landscape (rMATS, NS vs HS)
- 196,127 total events detected; 11,177 significant (FDR<0.05, |ΔΨ|>0.1)
- RI: 4,502 sig (40.3%) — most abundant; 2,784 show DECREASED retention under HS vs 1,718 increased
- Consistent with Bailey & Adams 2026 (Annals of Botany, B. napus Iso-Seq)
- A3SS: 3,013 sig; SE: 1,945; A5SS: 1,512; MXE: 205

### Paper gene lists (from Liu et al. 2024 Plant Cell Suppl. Data Set 2)
Location: `02_config/paper_gene_lists/`
- `paper_enriched_NS.csv`: 1,670 genes (P-body enriched under NS)
- `paper_enriched_HS.csv`: 2,723 genes (P-body enriched under HS)
- `paper_depleted_NS.csv`: 506 genes
- `paper_depleted_HS.csv`: 287 genes
- `paper_common_enriched.csv`: 519 genes (enriched in both NS and HS)

### APEAL proteomics (Sheet 1 of Excel — TurboID DCP1 interactome)
Two purification methods: AP (affinity purification FLAG/STREP) and PDL (proximity-dependent labeling)
Conditions: NS and HS
Key numbers:
- AP_NS: 299 enriched proteins
- AP_HS: 157 enriched proteins
- PDL_NS: 582 enriched proteins
- PDL_HS: 196 enriched proteins
- Total unique: 1,095 proteins

---

## ZORC ML Pipeline — Current plan

### Objective
Build a predictive model (Random Forest baseline) for P-body isoform enrichment from
sequence/structure features. Name: ZORC (Zero-Order RNA Condensate predictor).

### Classes
- **Class 1 (enriched):** genes from paper_enriched_HS ∪ paper_enriched_NS (~3,874 unique)
- **Class 0 (depleted):** genes from paper_depleted_HS ∪ paper_depleted_NS (~650-700 unique)
- One gene = one row (use strongest enrichment condition as representative)
- Class imbalance handled by **class weights** (NOT SMOTE — no biological meaning for interpolating BioEmu features)

### Coregulon filter (NEXT STEP — immediate priority)
Intersect T-RIP RNA lists (Sheet 2 of Excel) with APEAL protein lists (Sheet 1 of Excel):
- Keep only genes where BOTH the mRNA is T-RIP enriched/depleted AND the protein appears in AP or PDL (union of both methods)
- This gives the coregulon list: mRNA-protein cogulate pairs enriched/depleted together in P-bodies
- Expected size: ~800-1,200 unique genes after filter

**Excel file:** `/home/moschou/Downloads/tpc.23.01160Supplemental Data Sets 1XXX10.xlsx`
- Sheet 1: APEAL proteomics (AP and PDL enriched proteins)
- Sheet 2: T-RIP RNA enriched/depleted lists

### Sequence fetching strategy (NEXT STEP after coregulon list)
For each gene in the coregulon list, identify the SPECIFIC enriched isoform:
1. Map rMATS event coordinates (chr, start, end) → transcript ID from TAIR10.59 GTF
2. Extract mRNA sequence using `gffread` from `05_reference/Arabidopsis_thaliana.TAIR10.59.gtf` + genome FASTA
3. Extract protein sequence by translating CDS from GTF annotations
4. Cross-validate protein sequence against TAIR database
5. Output: two FASTAs (mRNA and protein) with GeneID_TranscriptID_condition_direction in headers

### Feature engineering plan

**RNA features (from mRNA sequence):**
- Nucleotide composition (A/U/G/C fractions)
- Dinucleotide frequencies
- AU content in 3'UTR (known P-body targeting signal)
- Length (CDS length, UTR lengths, total mRNA length)
- In-frame stop codon presence (NMD prediction proxy)
- m6A motif frequency (RRACH motif)
- Secondary structure features (MFE via RNAfold)

**Protein features — tiered BioEmu strategy (based on AIUPred IDR%):**
- **Tier 1 (IDR < 10%, ~40% of proteins):** AlphaFold2 static features only
  → Rg from AF2 structure, secondary structure composition, solvent accessibility
  → No BioEmu needed
- **Tier 2 (IDR 10-30%, ~35% of proteins):** BioEmu 50 conformations
  → Mean Rg ± std, RMSF global, RMSF N-term50, RMSF C-term50, pass_rate
- **Tier 3 (IDR > 30%, ~25% of proteins):** BioEmu 50-100 conformations
  → Same metrics + flag pass_rate < 20% as extreme disorder feature

**Computational estimate (ThinkStation P5, single GPU):**
- BioEmu on ~600 proteins (60% of dataset, Tiers 2+3) at 50 conformations: ~3-4 days continuous
- AlphaFold2 feature extraction from precomputed database: negligible

### Split strategy (anti-leakage)
1. Cluster all genes by sequence similarity: CD-HIT at 40% protein identity threshold
2. Assign entire clusters to train/CV/test — NEVER split clusters across partitions
3. Stratified by class within cluster assignment
4. Split ratio: 70% train / 15% CV / 15% test
5. DO NOT use ΔΨ or fold-change as features (circular — these define the class labels)

### Model
- Random Forest baseline (scikit-learn)
- Class weights = inverse of class frequency
- Evaluation: AUROC, AUPRC (precision-recall), F1 per class
- Feature importance analysis (SHAP values)

---

## Pending items in priority order

1. **[IMMEDIATE] Build coregulon list:** Intersect Sheet 2 (T-RIP RNA) with Sheet 1 (APEAL protein)
   - Script: Python, read Excel with openpyxl, merge on GeneID
   - Output: `02_config/zorc_coregulon_list.csv` with columns: GeneID, Symbol, enriched_NS_RNA, enriched_HS_RNA, depleted_NS_RNA, depleted_HS_RNA, protein_AP_NS, protein_AP_HS, protein_PDL_NS, protein_PDL_HS, class (1/0), condition (HS/NS/both)

2. **[NEXT] Map rMATS events to specific transcript IDs** for genes in coregulon list
   - Use pyranges to intersect rMATS event coordinates with GTF transcript coordinates

3. **[NEXT] Sequence fetching pipeline** using gffread + Biopython

4. **[NEXT] AIUPred IDR% calculation** for all proteins in coregulon list (assign BioEmu Tier)

5. **[NEXT] BioEmu batch runs** (Tier 2 and Tier 3 proteins, 50 conformations each)

6. **[NEXT] Feature matrix assembly** and CD-HIT clustering for anti-leakage split

7. **[LATER] IQD14 BioEmu rerun** with `--filter_samples=False` to extract features despite extreme disorder

8. **[LATER] NMD analysis** — ISA data shows HCF107, AT1G48210 PTC+ isoforms; formal NMD enrichment analysis under HS pending

---

## Key methodological decisions already made

- **No SMOTE:** BioEmu features are physically meaningful; interpolating between protein structures has no biological meaning. Use class weights instead.
- **HS∪NS for both classes:** Maximizes dataset size, reduces imbalance ratio from ~9.5:1 to ~5.5:1
- **One gene = one row:** If gene detected in both NS and HS, use strongest enrichment signal
- **Class weights not SMOTE:** Tell model "misclassifying depleted genes costs more"
- **CD-HIT split:** Prevents paralog leakage; entire clusters assigned to one partition
- **AlphaFold2 for Tier 1:** Precomputed Arabidopsis proteome available at alphafold.ebi.ac.uk

---

## Files generated in previous sessions (slides etc.)

- `04_results/05h_isa_consequences/AIUPred_IDR_summary.csv` — IDR% for 23 isoforms
- `04_results/05h_isa_consequences/GC_differential_10genes.csv` — GC differential nuclear architecture
- `04_results/06_rmats_transcriptome/NS_vs_HS/gene_classification_TRIP_vs_transcriptome.csv` — 3-class classification
- `/mnt/user-data/outputs/Phase5H_BioEmu_AllProteins.pptx` — BioEmu results all proteins
- `/mnt/user-data/outputs/Phase5H_BioEmu_ECT3.pptx` — ECT3 allosteric mechanism
- `/mnt/user-data/outputs/Phase5H_transcriptome_AS.pptx` — Global AS landscape + selectivity classification

---

*Prompt generated: 2026-03-23 · Session: T-RIP isoform pipeline Phase 5H → ZORC transition*
