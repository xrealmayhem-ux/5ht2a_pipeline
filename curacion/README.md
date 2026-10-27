# Curación de pKi (stage `curacion`)

Prepara curadas las mediciones de **Ki/pKi** de binding humano a partir de
GPCRdb + ChEMBL.

- Entrada: `../data/raw/{GPCRdb_pvalue.xlsx,df_chembl.csv}`
- Salida: `../data/pki/pki_curated.csv` (+ reporte JSON)

Entry point del stage en el monorepo: `workflow/rules/curacion.smk`.
