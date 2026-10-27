"""Train a single-target pKi CheMeleon model with a persisted scaffold split."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from chemprop import data, featurizers, models, nn
from config import (ACCELERATOR, BATCH_SIZE, BATCH_NORM, CHEMELEON_PATH, CHEMELEON_URL,
                    DROPOUT, GRADIENT_CLIP_VAL, MAX_EPOCHS, NUM_WORKERS, PATIENCE,
                    SCHEDULER_FACTOR, SCHEDULER_MIN_LR, SCHEDULER_PATIENCE,
                    SPLIT_SIZES, WEIGHT_DECAY)
from utils.losses import compute_metrics, make_stratified_scaffold_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "pki" / "pki_curated.csv"
DEFAULT_OUT = BASE_DIR / "checkpoints" / "pki"
SPLIT_PATH = BASE_DIR.parent / "data" / "pki" / "split_indices.csv"


def _optimizer(mpnn: models.MPNN) -> dict:
    opt = torch.optim.AdamW(mpnn.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR)
    return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def _split(df: pd.DataFrame, split_seed: int, persist: bool) -> tuple[list[int], list[int], list[int]]:
    keys = df["inchikey"].astype(str).tolist()
    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH)
        if saved["inchikey"].astype(str).tolist() == keys:
            return tuple(saved.index[saved["split"].eq(name)].tolist()
                         for name in ("train", "val", "test"))  # type: ignore[return-value]
    points = [data.MoleculeDatapoint.from_smi(smi, [float(y)])
              for smi, y in zip(df["smiles"], df["pki"])]
    bins = np.digitize(df["pki"].to_numpy(dtype=np.float32), (5, 6, 7, 8, 9))
    indices = make_stratified_scaffold_split([p.mol for p in points],
                                             df["pki"].to_numpy(dtype=np.float32),
                                             SPLIT_SIZES, bins, seed=split_seed)
    labels = np.full(len(df), "", dtype=object)
    for name, values in zip(("train", "val", "test"), indices):
        labels[values] = name
    if (labels == "").any():
        raise RuntimeError("Scaffold split left molecules unassigned")
    if persist:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"inchikey": keys, "split": labels}).to_csv(SPLIT_PATH, index=False)
    return indices


def _predict(trainer: pl.Trainer, model: models.MPNN, loader: object) -> np.ndarray:
    outputs = trainer.predict(model, loader, ckpt_path="best", weights_only=False)
    prediction = torch.cat([output.reshape(-1, 1) for output in outputs], dim=0)
    return prediction.detach().cpu().numpy().reshape(-1, 1)


def _cold_model(scaler: object) -> models.MPNN:
    message_passing = nn.BondMessagePassing()
    predictor = nn.RegressionFFN(
        input_dim=message_passing.output_dim, dropout=DROPOUT,
        output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    return models.MPNN(message_passing, nn.MeanAggregation(), predictor,
                       batch_norm=BATCH_NORM,
                       metrics=[nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()])


MP_HPARAM_KEYS = {"depth", "d_h", "d_v", "d_e", "d_f", "d_readout", "activation", "dropout", "undirected"}


def _build_mpnn_from_chemeleon(scaler):
    if not Path(CHEMELEON_PATH).exists():
        print(f"Downloading CheMeleon from {CHEMELEON_URL} ...")
        urlretrieve(CHEMELEON_URL, CHEMELEON_PATH)
        print("  Done (80 MB)")

    ckpt = torch.load(CHEMELEON_PATH, weights_only=True)
    mp_kwargs = {k: v for k, v in ckpt["hyper_parameters"].items() if k in MP_HPARAM_KEYS}
    mp = nn.BondMessagePassing(**mp_kwargs)
    mp.load_state_dict(ckpt["state_dict"])
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, dropout=DROPOUT,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=BATCH_NORM,
                       metrics=[nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()])

    n_total = sum(p.numel() for p in mpnn.parameters())
    print(f"  MP backbone: {sum(p.numel() for p in mp.parameters()):>8,} params")
    print(f"  FFN head:    {sum(p.numel() for p in ffn.parameters()):>8,} params")
    print(f"  Total:       {n_total:>8,} params")
    return mpnn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-warmstart", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    df = pd.read_csv(DATA_PATH)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    train_idx, val_idx, test_idx = _split(df, args.split_seed, args.limit is None)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    points = [data.MoleculeDatapoint.from_smi(smi, [float(y)])
              for smi, y in zip(df["smiles"], df["pki"])]
    sets = data.split_data_by_indices(points, [train_idx], [val_idx], [test_idx])
    datasets = [data.MoleculeDataset(part[0], featurizer) for part in sets]
    scaler = datasets[0].normalize_targets()
    datasets[1].normalize_targets(scaler)
    datasets[2].normalize_targets(scaler)
    loaders = [data.build_dataloader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                     shuffle=i == 0) for i, ds in enumerate(datasets)]
    prediction_train_loader = data.build_dataloader(
        datasets[0], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    model = _cold_model(scaler) if args.no_warmstart else _build_mpnn_from_chemeleon(scaler)
    model.configure_optimizers = lambda: _optimizer(model)
    out = args.out_dir / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=str(out), filename="pki-best-{epoch}-{val_loss:.4f}",
                                monitor="val_loss", mode="min", save_last=True)
    trainer = pl.Trainer(logger=CSVLogger(str(out), name="logs"), accelerator=ACCELERATOR,
                         devices=1, max_epochs=args.max_epochs,
                         gradient_clip_val=GRADIENT_CLIP_VAL,
                         callbacks=[checkpoint, EarlyStopping(monitor="val_loss", patience=PATIENCE),
                                    TQDMProgressBar(refresh_rate=1)])
    trainer.fit(model, loaders[0], loaders[1])
    best = checkpoint.best_model_path or str(out / "last.ckpt")
    if not Path(best).exists():
        raise RuntimeError("No checkpoint was produced")
    test_pred = _predict(trainer, model, loaders[2])
    train_pred = _predict(trainer, model, prediction_train_loader)
    test_df = df.iloc[test_idx].copy()
    test_df["pki_pred"] = test_pred[:, 0]
    test_df["residual"] = test_df["pki"] - test_df["pki_pred"]
    test_df.to_csv(out / "predictions.csv", index=False)
    train_df = df.iloc[train_idx][["smiles", "inchikey", "pki"]].copy()
    train_df["pki_pred"] = train_pred[:, 0]
    train_df.to_csv(out / "train_predictions.csv", index=False)
    pd.DataFrame([compute_metrics(test_df["pki"].to_numpy(), test_pred[:, 0])]).to_csv(
        out / "metrics.csv", index=False)
    torch.save({"scaler": scaler, "target": "pki"}, out / "scaler.pt")
    print(f"Saved {len(df):,} molecules; checkpoint={best}")


if __name__ == "__main__":
    main()
