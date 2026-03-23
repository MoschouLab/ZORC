# ZORC — Zip-code Of RNAs that Condense

Predictive pipeline for P-body RNA enrichment using isoform sequences,
conformational ensembles (BioEmu), and IDR features (AIUPred).

**Project:** ERC Consolidator Grant PLANTEX  
**Lab:** MoschouLab, IMBB-FORTH / University of Crete  
**Author:** José Moya-Cuevas

## Pipeline phases
| Phase | Script | Status |
|-------|--------|--------|
| P1 | `01_build_coregulon.py` | ✅ complete |
| P2 | `02_map_isoforms.py` | ✅ complete |
| P3 | `03_fetch_sequences.py` | 🔄 in progress |

## Environments
- `zorc_pipeline` — main pipeline (see `envs/zorc_pipeline.yml`)
- `bioemu` — BioEmu + AIUPred (see `envs/bioemu_ref.yml`)

## Reproducibility
```bash
conda env create -f envs/zorc_pipeline.yml
conda activate zorc_pipeline
python scripts/01_build_coregulon.py --config config/zorc_config.yaml
python scripts/02_map_isoforms.py --config config/zorc_config.yaml
```
