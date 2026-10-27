"""Curate GPCRdb and ChEMBL binding Ki/pKi measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "pki"
OUT_PATH = OUT_DIR / "pki_curated.csv"
REPORT_PATH = OUT_DIR / "curation_report.json"


def canonical(smi: object) -> tuple[str | None, str | None]:
    if not isinstance(smi, str) or not smi.strip():
        return None, None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)


def to_pki(value: object, units: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number) or number <= 0:
        return float("nan")
    unit = str(units).strip().lower().replace("\u03bc", "u").replace("\u00b5", "u")
    factors = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3,
               "m": 1.0, "molar": 1.0, "mol/l": 1.0, "mol/litre": 1.0}
    return float(-np.log10(number * factors[unit])) if unit in factors else float("nan")


def main(out_path: Path | None = None, raw_dir: Path | None = None,
         report_path: Path | None = None) -> None:
    out_path = Path(out_path) if out_path else OUT_PATH
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    report_path = Path(report_path) if report_path else REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gpcr = pd.read_excel(raw_dir / "GPCRdb_pvalue.xlsx")
    gpcr = gpcr[gpcr["Species"].astype(str).str.strip().str.lower().eq("human")
                & gpcr["Assay Type"].astype(str).str.strip().str.lower().eq("binding")
                & gpcr["Activity Type"].astype(str).str.strip().str.lower().isin({"ki", "pki"})]
    gpcr_rows = []
    for _, row in gpcr.iterrows():
        smiles, inchikey = canonical(row["Smiles"])
        try:
            value = float(row["p-value (-log)"])
        except (TypeError, ValueError):
            continue
        if smiles is not None and inchikey is not None and np.isfinite(value):
            gpcr_rows.append({"smiles": smiles, "inchikey": inchikey, "pki": value})

    chembl = pd.read_csv(raw_dir / "df_chembl.csv", sep=";", low_memory=False)
    chembl = chembl[chembl["Assay Type"].astype(str).str.strip().str.upper().eq("B")
                    & chembl["Standard Type"].astype(str).str.strip().str.lower().isin({"ki", "pki"})]
    chembl_rows = []
    converted = 0
    for _, row in chembl.iterrows():
        smiles, inchikey = canonical(row["Smiles"])
        if smiles is None or inchikey is None:
            continue
        value = row["pChEMBL Value"]
        if pd.isna(value):
            value = to_pki(row["Standard Value"], row["Standard Units"])
            converted += int(np.isfinite(value))
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
        if np.isfinite(value):
            chembl_rows.append({"smiles": smiles, "inchikey": inchikey, "pki": float(value)})

    rows = pd.DataFrame(gpcr_rows + chembl_rows)
    if rows.empty:
        raise RuntimeError("No valid Ki/pKi rows survived curation")
    curated = rows.groupby("inchikey", as_index=False).agg(
        smiles=("smiles", lambda s: s.value_counts().index[0]), pki=("pki", "median"))
    curated = curated[["smiles", "inchikey", "pki"]]
    curated.to_csv(out_path, index=False)
    report = {
        "inputs": {"gpcrdb_rows": int(len(gpcr)), "chembl_rows": int(len(chembl))},
        "valid_rows": {"gpcrdb": len(gpcr_rows), "chembl": len(chembl_rows),
                       "chembl_converted_ki": converted},
        "curated_molecules": int(len(curated)),
        "pki_range": {"min": float(curated["pki"].min()), "max": float(curated["pki"].max()),
                      "median": float(curated["pki"].median())},
        "output": str(out_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate GPCRdb + ChEMBL pKi data")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    main(out_path=args.out, raw_dir=args.raw_dir, report_path=args.report)
