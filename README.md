# 5-HT2A pipeline (curación → modelo_pki → generación → filtro)

Monorepo Snakemake. **Docking queda afuera** (repo aparte `docking_5ht2a`).

## Flujo

```text
curacion -> modelo_pki -> generacion -> filtro -> [docking]
```

| Etapa | Carpeta | Entry point | Output |
|---|---|---|---|
| 1. Curación | `curacion/` | `curacion/prepare_pki.py` (`rules/curacion.smk`) | `results/curacion/pki_curated.csv` |
| 2. Entrenamiento | `modelo_pki/` | `modelo_pki/train_pki.py` (`rules/modelo_pki.smk`) | `results/modelo_pki/checkpoints/...` |
| 3. Generación | `generacion/` | `generacion/run_generacion.py` (`rules/generacion.smk`) | `results/generacion/stage2_1.csv` |
| 4. Filtro | `filtro/` | `filtro/scripts/filter1.py` (`rules/filtro.smk`) | `results/filtro/pharmit_ready.csv` + `.smi.gz` |

El output final de este workflow es `results/filtro/pharmit_ready.csv`, el input
manual de docking/Pharmit.

## Estructura

```text
5ht2a-pipeline/
  workflow/Snakefile + rules/{curacion,modelo_pki,generacion,filtro}.smk
  workflow/envs/{curacion,modelopki,generacion,filter}.yaml
  curacion/          # etapa 1: curación de pKi (prepare_pki.py)
  modelo_pki/        # etapa 2: entrenamiento/fine-tune CheMeleon (train_pki.py)
  generacion/        # etapa 3: REINVENT4 staged (run_generacion.py, scorer_pki.py)
  filtro/            # etapa 4: filtro químico (scripts/filter1.py)
  scripts/           # download_models.sh, checksums.txt, download_data.sh
  data/              # tl_smiles.smi, pki/{pki_curated.csv,split_indices.csv,...}, raw/
  results/           # salidas del pipeline (no versionado)
  config/            # config.yaml.template -> config.yaml (local, no versionar)
  docs/              # METODOLOGIA.md
  tests/             # test_smoke.py
  .github/           # smoke CI
```

## Uso

```bash
cp config/config.yaml.template config/config.yaml  # ya trae record 23249654
bash scripts/download_models.sh --dest modelos --record 23249654
(cd modelos && sha256sum -c ../scripts/checksums.txt)
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 --dry-run -n
snakemake --snakefile workflow/Snakefile --use-conda --cores 8
```

Pasos del workflow (Snakemake encadena las etapas):

```bash
# 1. Curación
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 results/curacion/pki_curated.csv

# 2. Entrenamiento (ensemble de seeds)
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 results/modelo_pki/checkpoints/seed_2/DONE

# 3. Generación (S1 -> TL -> S2)
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 results/generacion/stage2_1.csv

# 4. Filtro
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 results/filtro/pharmit_ready.csv
```

Smoke (CI):

```bash
snakemake --snakefile workflow/Snakefile --use-conda --cores 8 --dry-run -n
python tests/test_smoke.py
```

## Modelos (Zenodo, no Git)

DOI: **https://doi.org/10.5281/zenodo.23249654** (licencia CC-BY-4.0; el código
sigue MIT). 3 archivos (~130MB, `sha256sum -c scripts/checksums.txt`):
...

## Datos versionados en Git (<1MB)

* `data/pki/{pki_curated.csv,split_indices.csv,curation_report.json}`
* `data/tl_smiles.smi`, `data/examples/stage2_1.csv`

## Reproducibilidad

Envs pineados por rule (`workflow/envs/*.yaml`), seeds fijas
(`modelo_pki.seeds: [2]` productivo, `smoke` con seed 42), split scaffold persistido
(`data/pki/split_indices.csv`).
