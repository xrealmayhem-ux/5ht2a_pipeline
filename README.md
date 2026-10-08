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

* `reinvent.prior` (23M), `chemeleon_mp.pt` (34M, Zenodo 15460715),
  `checkpoints/pki/seed_*/pki-best-*.ckpt` (107M c/u).
* Nunca commitear `*.ckpt/*.pt`. Publicar record Zenodo y
  rellenar `scripts/checksums.txt` (`sha256sum`).

## Reproducibilidad

Envs pineados por rule (`workflow/envs/*.yaml`), seeds fijas
(`chemprop.seeds: [42,1,2,3,4]`, `seed: 42`), split scaffold persistido
(`split_indices.csv`).
