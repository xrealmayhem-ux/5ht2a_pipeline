#!/usr/bin/env bash
# Descarga datasets de respaldo (df_chembl, tl_smiles) si no están presentes.
set -euo pipefail
echo "Sin datasets remotos por ahora: los CSVs curados pequeños viven en el repo."
echo "Los crudos (df_chembl 18M, GPCRdb 801K) se documentan en docs/DATOS.md."
