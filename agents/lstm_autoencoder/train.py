from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agents.common.calibration import fit_logistic_calibrator, logistic_probability, select_probability_threshold
from agents.common.preprocessing import fit_preprocessor, load_csv, schema_to_json, transform_frame
from agents.common.training_config import load_agent_training_config


class TabularAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _vectorize(num: np.ndarray, cat: np.ndarray, cat_cardinalities: list[int]) -> np.ndarray:
    num = num.astype(np.float32)
    if cat.size == 0 or len(cat_cardinalities) == 0:
        return num

    n = cat.shape[0]
    total_cat_dim = int(sum(cat_cardinalities))
    cat_oh = np.zeros((n, total_cat_dim), dtype=np.float32)

    offset = 0
    for i, card in enumerate(cat_cardinalities):
        idx = np.clip(cat[:, i].astype(np.int64), 0, max(card - 1, 0))
        rows = np.arange(n)
        cat_oh[rows, offset + idx] = 1.0
        offset += card

    return np.concatenate([num, cat_oh], axis=1)


def train(dataset: Path, output: Path, seed: int, config_path: Path | None) -> None:
    cfg = load_agent_training_config(config_path)
    shared = dict(cfg.get("shared", {}))
    pre_cfg = dict(shared.get("preprocessing", {}))
    cal_cfg = dict(shared.get("calibration", {}))
    agent_cfg = dict(cfg.get("lstm_autoencoder", {}))

    val_fraction = float(shared.get("validation_fraction", 0.2))
    val_fraction = min(0.4, max(0.05, val_fraction))

    probability_clip = list(cal_cfg.get("probability_clip", [0.001, 0.999]))
    p_lo = float(probability_clip[0])
    p_hi = float(probability_clip[1])

    latent_dim = int(agent_cfg.get("hidden_dim", 64))
    epochs = int(agent_cfg.get("epochs", 20))
    batch_size = int(agent_cfg.get("batch_size", 256))
    learning_rate = float(agent_cfg.get("learning_rate", 1e-3))
    dropout = float(agent_cfg.get("dropout", 0.1))

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)

    y = df["label"].astype(int).to_numpy()
    train_idx, cal_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    cal_df = df.iloc[cal_idx].reset_index(drop=True)

    pre = fit_preprocessor(
        train_df,
        categorical_cols=list(pre_cfg.get("categorical_cols", ["proto", "service", "state"])),
        iqr_floor=float(pre_cfg.get("iqr_floor", 1.0)),
        clip_min=float(pre_cfg.get("clip_min", -15.0)),
        clip_max=float(pre_cfg.get("clip_max", 15.0)),
    )

    train_num, train_cat = transform_frame(pre, train_df)
    cal_num, cal_cat = transform_frame(pre, cal_df)

    cat_cardinalities = [len(pre.vocabularies[c]) for c in pre.categorical_cols]
    x_train = _vectorize(train_num, train_cat, cat_cardinalities)
    x_cal = _vectorize(cal_num, cal_cat, cat_cardinalities)

    y_train = train_df["label"].astype(int).to_numpy()
    y_cal = cal_df["label"].astype(int).to_numpy()

    x_train_normal = x_train[y_train == 0]
    if len(x_train_normal) == 0:
        raise RuntimeError("No benign rows found for autoencoder training")

    model = TabularAutoencoder(in_dim=x_train.shape[1], latent_dim=latent_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_normal)), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_train_t = torch.from_numpy(x_train)
        x_cal_t = torch.from_numpy(x_cal)

        recon_train = model(x_train_t)
        recon_cal = model(x_cal_t)

        scores_train = torch.mean((recon_train - x_train_t) ** 2, dim=1).cpu().numpy()
        scores_cal = torch.mean((recon_cal - x_cal_t) ** 2, dim=1).cpu().numpy()

    calibrator = fit_logistic_calibrator(scores_cal, y_cal, seed=seed)
    probs_cal = logistic_probability(scores_cal, calibrator, clip_lo=p_lo, clip_hi=p_hi)
    threshold_prob, best_ba = select_probability_threshold(np.asarray(probs_cal), y_cal)

    benign = scores_cal[y_cal == 0] if np.any(y_cal == 0) else scores_cal
    malicious = scores_cal[y_cal == 1] if np.any(y_cal == 1) else scores_cal

    payload = {
        "state_dict": model.state_dict(),
        "preprocessor": pre.to_dict(),
        "cat_cardinalities": cat_cardinalities,
        "model_config": {
            "in_dim": int(x_train.shape[1]),
            "latent_dim": int(latent_dim),
            "dropout": float(dropout),
        },
        "calibration": calibrator,
        "threshold_probability": float(threshold_prob),
        "probability_clip": [p_lo, p_hi],
        "loss_stats": {
            "mean": float(scores_cal.mean()),
            "std": float(scores_cal.std() + 1e-9),
            "p95": float(np.percentile(scores_cal, 95.0)),
            "benign_mean": float(benign.mean()),
            "benign_std": float(benign.std() + 1e-9),
            "mal_mean": float(malicious.mean()),
            "mal_std": float(malicious.std() + 1e-9),
            "train_benign_mean": float(scores_train[y_train == 0].mean()),
            "train_benign_std": float(scores_train[y_train == 0].std() + 1e-9),
        },
        "meta": {
            "dataset": str(dataset),
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(cal_df)),
            "features": int(x_train.shape[1]),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "validation_fraction": float(val_fraction),
            "balanced_accuracy": float(best_ba),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    schema_to_json(pre, output.with_suffix(".schema.json"), extra={"model": "tabular_autoencoder"})

    print(
        json.dumps(
            {
                "saved_model": str(output),
                "saved_schema": str(output.with_suffix('.schema.json')),
                "rows": int(len(df)),
                "features": int(x_train.shape[1]),
                "balanced_accuracy": float(best_ba),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "lstm_autoencoder.pt"
    default_cfg = Path(__file__).resolve().parents[2] / "config" / "agent_training.yaml"

    p = argparse.ArgumentParser(description="Train tabular autoencoder for UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pt model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(default_cfg), help="Path to agent training YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
