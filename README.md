# 5-HT2A pipeline (generación + chemprop + filter + admet)

Monorepo Snakemake. **Docking queda afuera** (repo aparte `docking_5ht2a`).

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
cp config/config.yaml.template config/config.yaml  # editar record Zenodo
bash scripts/download_models.sh --dest modelos
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 --dry-run -n
snakemake --snakefile workflow/Snakefile --use-conda --cores 8
```

## Modelos (Zenodo, no Git)

16 archivos (~605MB, `sha256sum -c scripts/checksums.txt`):
* `chemeleon/checkpoints/pki/seed_{1..5}/pki-best-*.ckpt` (107M c/u, ensemble productivo)
* `chemeleon/checkpoints/pki/seed_{1..5}/scaler.pt` + `train_predictions.csv`
  (el scorer exige los 5 `train_predictions.csv` para el umbral p90 de incertidumbre)
* `reinvent_stage2.chkpt` (23M, agente generativo final para muestrear)
* NO se suben: `chemeleon_mp.pt` (citar record `15460715`), `reinvent.prior`
  (externo), `last.ckpt`, `stage1.chkpt`, `tl.model`, CSVs/logs/`tb_*/`.

## Datos versionados en Git (<1MB)

* `data/pki/{pki_curated.csv,split_indices.csv,curation_report.json}`
* `data/tl_smiles.smi`, `data/examples/stage2_1.csv`

## Reproducibilidad

Envs pineados por rule (`workflow/envs/*.yaml`), seeds fijas
(`chemprop.seeds: [1,2,3,4,5]`, `seed: 42` solo para smoke), split scaffold persistido
(`split_indices.csv`).
