# Generación molecular 5-HT2A (stage `generacion`)

Pipeline de generación con REINVENT4 y un modelo Chemprop single-task para
predecir pKi (seed_2 productivo). Esta carpeta contiene únicamente los archivos
necesarios para ejecutar la generación; el checkpoint de Chemprop se
resuelve vía symlink `modelos/chemeleon -> ../modelo_pki`.

## Estructura

```text
generacion/
├── run_generacion.py       # Stage 1, transfer learning y Stage 2
├── scorer_pki.py           # Scorer externo REINVENT4 (seed_2; acepta ensemble legacy)
├── modelos/
│   ├── reinvent.prior
│   └── chemeleon -> ../modelo_pki
├── datos/
│   └── tl_smiles.smi
└── resultados/              # Se crea al ejecutar; no se versionan sus salidas
```

## Ejecución

Desde la raíz del monorepo:

```bash
conda run -n reinvent4 python generacion/run_generacion.py
```

Prueba corta en CPU:

```bash
DEVICE=cpu S1_STEPS=2 TL_EPOCHS=1 S2_STEPS=5 BATCH_SIZE=32 \
  conda run -n reinvent4 python generacion/run_generacion.py
```

Para retomar únicamente Stage 2 usando checkpoints existentes:

```bash
START_STEP=4 conda run -n reinvent4 python generacion/run_generacion.py
```

Para correr solo hasta un paso (usado por el workflow Snakemake para partir
Stage 1 de TL+Stage 2):

```bash
START_STEP=1 STOP_STEP=1 conda run -n reinvent4 python generacion/run_generacion.py
```

Sin `STOP_STEP` el pipeline corre hasta el paso 4.

## Scorer manual

```bash
conda run -n modelo_pki python generacion/scorer_pki.py compounds.csv
printf 'CCO\nc1ccccc1\n' | conda run -n modelo_pki python generacion/scorer_pki.py
```

El scorer devuelve `score`, `pki_pred`, `pki_std` y `n_models`, compatibles
con `ExternalProcess` de REINVENT4.
