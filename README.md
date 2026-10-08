# 5-HT2A pipeline (generación + chemprop + filter + admet)


## Frontera

Este workflow termina en `results/admet/admet_safety_passed.csv`, que es el
input manual de docking/Pharmit. No hay reglas de docking aquí.

## Estructura

```text
5ht2a-pipeline/
  workflow/Snakefile + rules/{chemprop,generacion,filter,admet}.smk
  workflow/envs/{chemprop,generacion,filter,admet}.yaml
  config/config.yaml.template -> config/config.yaml (local, no versionar)
  scripts/download_models.sh  # Zenodo -> modelos/ (reemplaza symlink)
  docs/METODOLOGIA.md
```

En modo transición, las rules llaman a los módulos hermanos
(`../AI_drugdiscovery/generacion`, `../chemprop/...`, `../filter`,
`../admet_ai_filter`) vía rutas de `config.yaml`. En clon fresco, esos
módulos ya viven dentro del monorepo.

## Uso

```bash
cp config/config.yaml.template config/config.yaml  # ya trae record 23249654
bash scripts/download_models.sh --dest modelos --record 23249654
(cd modelos && sha256sum -c ../scripts/checksums.txt)
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 --dry-run -n
snakemake --snakefile workflow/Snakefile --use-conda --cores 8
```

## Modelos (Zenodo, no Git)

DOI: **https://doi.org/10.5281/zenodo.23249654** (licencia CC-BY-4.0; el código
sigue MIT). 3 archivos (~130MB, `sha256sum -c scripts/checksums.txt`):
* `chemeleon/checkpoints/pki/seed_2/pki-best-epoch=16-val_loss=0.3912.ckpt` (107M,
  mejor test: RMSE 0.747, R² 0.669) + `scaler.pt`
* `reinvent_stage2.chkpt` (23M, agente generativo final para muestrear)
* El umbral p90 de incertidumbre (`0.29146`) va hardcodeado en `scorer_pki.py`
  (`ENSEMBLE_P90_FALLBACK`); no se descargan `train_predictions.csv`.
* NO se suben: `chemeleon_mp.pt` (citar record `15460715`), `reinvent.prior`
  (externo), `last.ckpt`, `stage1.chkpt`, `tl.model`, otras seeds, CSVs/logs/`tb_*/`.

## Datos versionados en Git (<1MB)

* `data/pki/{pki_curated.csv,split_indices.csv,curation_report.json}`
* `data/tl_smiles.smi`, `data/examples/stage2_1.csv`

## Reproducibilidad

Envs pineados por rule (`workflow/envs/*.yaml`), seeds fijas
(`chemprop.seeds: [2]` productivo, `seed: 42` solo para smoke), split scaffold persistido
(`split_indices.csv`).
