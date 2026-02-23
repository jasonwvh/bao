from __future__ import annotations

import argparse
import os
import sys
# When running this script directly, ensure the project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agents.common.preprocessing import (
    UNSWPreprocessor,
    build_sequences,
    fit_preprocessor,
    load_csv,
    schema_to_json,
    transform_frame,
)


class HybridLSTMAutoencoder(nn.Module):
    def __init__(self, num_dim: int, cat_cardinalities: list[int], hidden_dim: int = 64):
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        self.cat_embedding_dims: list[int] = []
        for card in cat_cardinalities:
            emb_dim = int(min(16, max(4, card // 4)))
            self.cat_embedding_dims.append(emb_dim)
            self.cat_embeddings.append(nn.Embedding(int(max(2, card)), emb_dim))

        input_dim = num_dim + sum(self.cat_embedding_dims)
        self.num_dim = num_dim
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_dim)

    def _embed(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, :, i]))
        return torch.cat(parts, dim=-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self._embed(x_num, x_cat)
        _, (h_n, _) = self.encoder(embedded)
        seq_len = embedded.shape[1]
        dec_in = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.out(dec_out)
        return recon, x_num


def train(
    dataset: Path,
    output: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    window_size: int,
    stride: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)
    labels = df["label"].astype(int).to_numpy()

    pre = fit_preprocessor(df)
    num, cat = transform_frame(pre, df)
    x_num, x_cat, y = build_sequences(num, cat, labels, window_size=window_size, stride=stride)
    if len(y) == 0:
        raise RuntimeError("No sequences generated. Increase dataset size or reduce window size.")

    normal_mask = y == 0
    if int(normal_mask.sum()) == 0:
        raise RuntimeError("No benign windows found for LSTM-autoencoder training")

    x_num_train = torch.from_numpy(x_num[normal_mask])
    x_cat_train = torch.from_numpy(x_cat[normal_mask])

    cat_cardinalities = [len(pre.vocabularies[c]) for c in pre.categorical_cols]
    model = HybridLSTMAutoencoder(
        num_dim=x_num.shape[-1],
        cat_cardinalities=cat_cardinalities,
        hidden_dim=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_num_train, x_cat_train), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for b_num, b_cat in loader:
            optimizer.zero_grad(set_to_none=True)
            recon, target = model(b_num, b_cat)
            loss = criterion(recon, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_num_t = torch.from_numpy(x_num)
        all_cat_t = torch.from_numpy(x_cat)
        recon, target = model(all_num_t, all_cat_t)
        losses = torch.mean((recon - target) ** 2, dim=(1, 2)).cpu().numpy()

    calib = {}
    benign_losses = losses[y == 0] if np.any(y == 0) else losses
    mal_losses = losses[y == 1] if np.any(y == 1) else losses
    invert_score = float(np.mean(mal_losses)) < float(np.mean(benign_losses))
    score_used = -losses if invert_score else losses
    benign_scores = -benign_losses if invert_score else benign_losses
    mal_scores = -mal_losses if invert_score else mal_losses

    if len(np.unique(y)) > 1:
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(score_used.reshape(-1, 1), y)
        calib = {
            "coef": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
        }

    if len(mal_scores) > 0 and len(benign_scores) > 0:
        mapping_threshold = float((np.mean(benign_scores) + np.mean(mal_scores)) / 2.0)
    else:
        mapping_threshold = float(np.percentile(benign_scores, 95.0))
    mapping_scale = float(max(np.std(np.concatenate([benign_scores, mal_scores])), 1e-6))

    payload = {
        "state_dict": model.state_dict(),
        "preprocessor": pre.to_dict(),
        "model_config": {
            "num_dim": int(x_num.shape[-1]),
            "cat_cardinalities": cat_cardinalities,
            "hidden_dim": 64,
            "window_size": int(window_size),
            "stride": int(stride),
        },
        "loss_stats": {
            "mean": float(losses.mean()),
            "std": float(losses.std() + 1e-9),
            "p95": float(np.percentile(losses, 95.0)),
            "benign_mean": float(benign_losses.mean()),
            "benign_std": float(benign_losses.std() + 1e-9),
            "mal_mean": float(mal_losses.mean()),
            "mal_std": float(mal_losses.std() + 1e-9),
            "invert_score": bool(invert_score),
        },
        "score_mapping": {
            "threshold": mapping_threshold,
            "scale": mapping_scale,
            "benign_score_mean": float(np.mean(benign_scores)),
            "mal_score_mean": float(np.mean(mal_scores)),
        },
        "calibration": calib,
        "meta": {
            "dataset": str(dataset),
            "rows": int(len(df)),
            "windows": int(len(y)),
            "features_num": int(x_num.shape[-1]),
            "features_cat": int(x_cat.shape[-1]),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    schema_to_json(
        pre,
        output.with_suffix(".schema.json"),
        extra={"window_size": int(window_size), "stride": int(stride)},
    )

    print(
        json.dumps(
            {
                "saved_model": str(output),
                "saved_schema": str(output.with_suffix('.schema.json')),
                "rows": int(len(df)),
                "windows": int(len(y)),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "lstm_autoencoder.pt"

    p = argparse.ArgumentParser(description="Train hybrid LSTM-autoencoder on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pt model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def main() -> None:
    print("Starting LSTM-autoencoder training...")
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        window_size=int(args.window_size),
        stride=int(args.stride),
    )


if __name__ == "__main__":
    main()
