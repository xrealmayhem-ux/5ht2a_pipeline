# CheMeleon pKi — entrenamiento (stage `modelo_pki`)

Fine-tuning del foundation model CheMeleon (`chemeleon_mp.pt`, Zenodo 15460715)
para regresión de **pKi** sobre datos de binding humano (GPCRdb + ChEMBL),
con split scaffold estratificado persistido.

## Ubicación en el monorepo

- Código: `modelo_pki/train_pki.py`, `modelo_pki/config.py`, `modelo_pki/utils/`
- Modelo: `modelo_pki/chemeleon_mp.pt`
- Checkpoints: `modelo_pki/checkpoints/pki/seed_<S>/...`
- Datos: `../data/pki/{pki_curated.csv,split_indices.csv}`

## Flujo

| Paso | Script | Qué hace |
|------|--------|----------|
| 1 | `prepare_pki.py` (stage `curacion`) | Curación GPCRdb + ChEMBL (binding Ki/pKi) → `../data/pki/pki_curated.csv` + reporte JSON |
| 2 | `train_pki.py --seed S` | Entrena una seed con split persistido → `checkpoints/pki/seed_S/` (best+last ckpt, `scaler.pt`, métricas y predicciones) |

## Comandos

```bash
conda run -n modelo_pki python train_pki.py --seed 2

# Smoke test (no toca el split persistido ni los checkpoints productivos)
conda run -n modelo_pki python train_pki.py --seed 42 --limit 200 --max-epochs 1 \
    --no-warmstart --out-dir /tmp/pki_smoke
```

El consumidor externo (`generacion/scorer_pki.py`) importa
`_build_mpnn_from_chemeleon` desde `train_pki.py`.

## Detalles

- Split: scaffold de Murcko estratificado por pKi (bins 5–9), 70/15/15; índices
  fijados en `../data/pki/split_indices.csv` para reproducibilidad entre seeds.
- Modelo productivo: seed `2`; su `scaler.pt` acompaña al checkpoint.
- Configuración compartida en `config.py`.
