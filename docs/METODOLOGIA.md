# Metodología (resumen para el monorepo)

Flujo completo: **curación → modelo_pki → generación → filtro** (entrenamiento → generación → filtro).

1. **Curación (`curacion/`)**: preparación curada de pKi a partir de GPCRdb + ChEMBL (Ki → pKi). Salida: `results/curacion/pki_curated.csv`.
2. **Entrenamiento / modelo (`modelo_pki/`)**: fine-tune del modelo CheMeleon sobre la curación (scaffold split persistido). Salida: `results/modelo_pki/checkpoints/...`.
3. **Generación (`generacion/`)**: REINVENT4 staged learning (S1: QED+TPSA → TL → S2: Chemprop pKi + filters). Salida: `results/generacion/stage2_1.csv`.
4. **Filtro (`filtro/`)**: sanitize, dedup, tautómeros, CNS, alertas, SA Score, stereo, export pharmit-ready. Salida: `results/filtro/pharmit_ready.csv` + `.smi.gz`.

El filtro prioriza drug-like/diversidad; no demuestra actividad experimental frente a 5-HT2A.

Docking queda en repo aparte `docking_5ht2a`; el output final de este workflow (filtro) es el input manual de docking/Pharmit.
