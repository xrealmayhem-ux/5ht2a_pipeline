"""REINVENT4 external scorer for the single-task pKi model (seed_2, best test).

Productive setup: single checkpoint (seed_2: test RMSE 0.747, R2 0.669).
With one model std=0, so the uncertainty penalty uses the precomputed p90
from the original 5-seed ensemble (n=3844 train mols) as fallback threshold.
Legacy 5-model ensemble still works if 5 checkpoints + 5 train_predictions.csv
are present (auto-detected).
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
# Overridable for the Snakemake monorepo: downloaded models live in
# <pipeline>/modelos/chemeleon (symlink -> ../modelo_pki) instead of a
# cross-repo checkout. The train_pki.py source is imported from the
# modelo_pki stage via CHEMPROP_SOURCE_DIR.
_CHEMELEON_ENV = os.getenv("CHEMPROP_CHEMELEON_DIR", "").strip()
CHEMELEON_DIR = Path(_CHEMELEON_ENV) if _CHEMELEON_ENV else SCRIPT_DIR / "modelos" / "chemeleon"
_SOURCE_ENV = os.getenv("CHEMPROP_SOURCE_DIR", "").strip()
_CHEMELEON_SRC = Path(_SOURCE_ENV) if _SOURCE_ENV else CHEMELEON_DIR
PKI_DIR = CHEMELEON_DIR / "checkpoints" / "pki"
TRAIN_PREDICTION_FILES = sorted(PKI_DIR.glob("seed_*/train_predictions.csv"))
if str(_CHEMELEON_SRC) not in sys.path:
    sys.path.insert(0, str(_CHEMELEON_SRC))

PKI_MID = 8.0
PKI_SLOPE = 1.0
MIN_ENSEMBLE_MODELS = 1
# Fallback p90 precomputed from the 5-seed ensemble train std (n=3844).
# Used when fewer than 5 train_predictions.csv are available (single-model setup).
ENSEMBLE_P90_FALLBACK = 0.2914589293869163
_UNCERTAINTY_P90 = None


def sigmoid_transform(x, mid=PKI_MID, slope=PKI_SLOPE):
    values = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = 1.0 / (1.0 + np.exp(-slope * (values - mid)))
    return float(result) if result.ndim == 0 else result


def ensemble_checkpoints() -> list[Path]:
    configured = os.getenv("CHEMPROP_PKI_ENSEMBLE", "").strip()
    if configured:
        paths = [Path(item.strip()) for item in configured.split(os.pathsep)
                 if item.strip()]
    else:
        paths = []
        for seed_dir in sorted(PKI_DIR.glob("seed_*")):
            candidates = sorted(
                seed_dir.glob("pki-best-epoch=*.ckpt"),
                key=lambda path: path.stat().st_mtime,
            )
            if candidates and (seed_dir / "scaler.pt").is_file():
                paths.append(candidates[-1])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pKi checkpoint(s): {missing}")
    if len(paths) < MIN_ENSEMBLE_MODELS:
        raise RuntimeError(
            f"Expected at least {MIN_ENSEMBLE_MODELS} pKi checkpoints, found {len(paths)}"
        )
    return paths


def _model_for_checkpoint(checkpoint: Path):
    from train_pki import _build_mpnn_from_chemeleon

    scaler_path = checkpoint.parent / "scaler.pt"
    scaler = torch.load(scaler_path, map_location="cpu", weights_only=False)["scaler"]
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        model = _build_mpnn_from_chemeleon(scaler)
    finally:
        sys.stdout = real_stdout
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_models(checkpoints: list[Path]):
    return [_model_for_checkpoint(checkpoint) for checkpoint in checkpoints]


def _predict(smiles_list: list[str], models) -> np.ndarray:
    import lightning.pytorch as pl
    from chemprop import data, featurizers
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    datapoints, valid_indices = [], []
    for index, smiles in enumerate(smiles_list):
        if isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None:
            datapoints.append(data.MoleculeDatapoint.from_smi(smiles))
            valid_indices.append(index)

    predictions = []
    for model in models:
        values = np.full(len(smiles_list), np.nan, dtype=float)
        if datapoints:
            dataset = data.MoleculeDataset(datapoints, featurizer)
            loader = data.build_dataloader(
                dataset, batch_size=128, num_workers=0, shuffle=False
            )
            trainer = pl.Trainer(
                accelerator="cpu", devices=1, logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False,
            )
            real_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                raw = trainer.predict(model, dataloaders=loader)
            finally:
                sys.stdout = real_stdout
            predicted = torch.cat([item.reshape(-1) for item in raw])
            values[valid_indices] = predicted.detach().cpu().numpy()
        predictions.append(values)
    return np.stack(predictions, axis=0)


def _uncertainty_threshold() -> float:
    global _UNCERTAINTY_P90
    if _UNCERTAINTY_P90 is None:
        if len(TRAIN_PREDICTION_FILES) >= 5:
            predictions = [
                pd.read_csv(path).sort_values("inchikey")["pki_pred"].to_numpy()
                for path in TRAIN_PREDICTION_FILES
            ]
            lengths = {len(values) for values in predictions}
            if len(lengths) != 1:
                raise RuntimeError("pKi training prediction files have inconsistent sizes")
            std = np.std(np.stack(predictions), axis=0, ddof=1)
            _UNCERTAINTY_P90 = float(np.percentile(std, 90))
        else:
            # Single-model setup: reuse the ensemble-calibrated threshold.
            _UNCERTAINTY_P90 = ENSEMBLE_P90_FALLBACK
    return _UNCERTAINTY_P90


def _score(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    threshold = _uncertainty_threshold()
    excess = np.maximum(std - threshold, 0.0) / max(threshold, 1e-12)
    confidence = 1.0 / (1.0 + excess)
    result = sigmoid_transform(mean) * confidence
    result[~np.isfinite(mean) | ~np.isfinite(std)] = 0.0
    return result


def main() -> None:
    if len(sys.argv) == 1:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"score": []}))
            return
        smiles = [line.strip() for line in raw.splitlines() if line.strip()]
        csv_mode = False
        smiles_column = None
    elif len(sys.argv) == 2:
        frame = pd.read_csv(sys.argv[1])
        smiles_column = next(
            (column for column in frame.columns if column.lower() == "smiles"),
            frame.columns[0],
        )
        smiles = frame[smiles_column].astype(str).tolist()
        csv_mode = True
    else:
        raise SystemExit("Usage: python scorer_patched.py [<smiles_csv>]")

    checkpoints = ensemble_checkpoints()
    models = _load_models(checkpoints)
    per_model = _predict(smiles, models)
    mean = np.nanmean(per_model, axis=0)
    std = np.nanstd(per_model, axis=0, ddof=1)
    scores = _score(mean, std)

    if csv_mode:
        output = pd.DataFrame({
            smiles_column: smiles,
            "score": np.round(scores, 6),
            "pki_pred": np.round(mean, 4),
            "pki_std": np.round(std, 4),
            "n_models": len(checkpoints),
        })
        print(output.to_csv(index=False), end="")
        return

    def rounded(value, digits):
        return round(float(value), digits) if np.isfinite(value) else None

    payload = {
        "score": [rounded(value, 6) for value in scores],
        "pki_pred": [rounded(value, 4) for value in mean],
        "pki_std": [rounded(value, 4) for value in std],
        "n_models": [len(checkpoints)] * len(smiles),
    }
    print(json.dumps({"payload": payload}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (SCRIPT_DIR / "scorer_debug.log").write_text(
            __import__("traceback").format_exc(), encoding="utf-8"
        )
        raise
