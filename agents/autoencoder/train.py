from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXCLUDE_COLUMNS = {"id", "label", "attack_cat"}


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _normalize_columns(cols: list[str]) -> list[str]:
    return [str(c).replace("\ufeff", "").strip() for c in cols]


def _load_training_matrix(dataset: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(dataset)
    df.columns = _normalize_columns(list(df.columns))

    numeric_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            df[col] = s.fillna(0.0).astype("float32")
            numeric_cols.append(col)

    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found in dataset")

    x = df[numeric_cols].to_numpy(dtype=np.float32)
    return x, numeric_cols


def train(
    dataset: Path,
    output: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_raw, feature_names = _load_training_matrix(dataset)

    feat_mean = x_raw.mean(axis=0)
    feat_std = x_raw.std(axis=0) + 1e-6
    x = (x_raw - feat_mean) / feat_std

    model = AutoencoderModel(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x)
        recon = model(x_t)
        losses = torch.mean((recon - x_t) ** 2, dim=1).cpu().numpy()

    payload = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "feature_mean": feat_mean.astype(np.float32),
        "feature_std": feat_std.astype(np.float32),
        "loss_mean": float(losses.mean()),
        "loss_std": float(losses.std() + 1e-9),
        "loss_p95": float(np.percentile(losses, 95.0)),
        "meta": {
            "dataset": str(dataset),
            "rows": int(x.shape[0]),
            "features": int(x.shape[1]),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    print(json.dumps({"saved_model": str(output), "rows": int(x.shape[0]), "features": int(x.shape[1])}, indent=2))


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "autoencoder.pt"

    p = argparse.ArgumentParser(description="Train autoencoder on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pt model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )


if __name__ == "__main__":
    main()
