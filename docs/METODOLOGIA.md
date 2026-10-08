# Metodología (resumen para el monorepo)

1. **Chemprop pKi**: fine-tune CheMeleon (`chemeleon_mp.pt`) sobre binding
   humano GPCRdb+ChEMBL, split scaffold Murcko estratificado 70/15/15,
   ensemble 5 seeds. Métricas declaradas: RMSE~0.765, R²~0.653.
2. **Generación REINVENT4 staged**: S1 (QED+TPSA) -> TL (tl_smiles.smi) ->
   S2 (Chemprop pKi 0.35 + QED/TPSA/SlogP/HBD + alerts + Murcko).
3. **Filter**: sanitize RDKit, dedup InChIKey, tautómeros, CNS
   (MW≤500, LogP 1-5, HBD≤3, HBA≤7, RotB≤8, TPSA 15-120), PAINS/BRENK/REOS,
   SA≤4.5, enumeración estéreo ≤8, export `.smi.gz` Pharmit-ready.
4. **ADMET-AI v2.0.1**: predict + regla dura
   hERG/AMES/DILI/Carcinogens_Lagunin ≥0.50.

El filtro prioriza drug-like/diversidad; no demuestra actividad experimental.
Detalle completo en `filter/README.md` y `admet_ai_filter/README.md` originales.
