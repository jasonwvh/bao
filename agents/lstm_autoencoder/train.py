from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agents.common.calibration import fit_best_calibrator
from agents.common.preprocessing import (
    build_sequences,
    fit_preprocessor,
    load_csv,
    schema_to_json,
    transform_frame,
)
from agents.common.streaming import derive_stream_id
from agents.common.training_config import load_agent_training_config
from agents.common.versioning import collect_library_versions


class SequenceLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        recurrent_dropout = float(dropout if int(num_layers) > 1 else 0.0)
        self.encoder = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=int(hidden_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.out_proj = nn.Linear(int(hidden_dim), int(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = int(x.shape[1])
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec, _ = self.decoder(repeated)
        return self.out_proj(dec)


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


def _sequence_error(model: SequenceLSTMAutoencoder, x_np: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_np.astype(np.float32))
        recon = model(x)
        return torch.mean((recon - x) ** 2, dim=(1, 2)).cpu().numpy()


def _extract_stream_ids(df) -> np.ndarray:
    rows = df.to_dict(orient="records")
    return np.asarray([derive_stream_id(flow_features=row) for row in rows], dtype=object)


def _split_temporal_by_stream(df, val_fraction: float):
    if len(df) < 2:
        raise RuntimeError("Dataset is too small for temporal train/calibration split")

    stream_ids = _extract_stream_ids(df)
    train_mask = np.zeros(len(df), dtype=bool)

    for sid in dict.fromkeys(stream_ids.tolist()).keys():
        idx = np.flatnonzero(stream_ids == sid)
        if len(idx) <= 1:
            train_mask[idx] = True
            continue
        cut = int(np.floor(len(idx) * (1.0 - val_fraction)))
        cut = max(1, min(len(idx) - 1, cut))
        train_mask[idx[:cut]] = True

    if train_mask.all() or (not train_mask.any()):
        cut = max(1, min(len(df) - 1, int(np.floor(len(df) * (1.0 - val_fraction)))))
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:cut] = True

    train_df = df.iloc[np.flatnonzero(train_mask)].reset_index(drop=True)
    cal_df = df.iloc[np.flatnonzero(~train_mask)].reset_index(drop=True)
    if len(train_df) == 0 or len(cal_df) == 0:
        raise RuntimeError("Temporal split produced empty train/calibration partition")
    return train_df, cal_df


def _build_sequences_per_stream(
    *,
    num: np.ndarray,
    cat: np.ndarray,
    labels: np.ndarray,
    stream_ids: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(num) != len(stream_ids):
        raise RuntimeError("num rows and stream_ids length mismatch")

    x_num_parts: list[np.ndarray] = []
    x_cat_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for sid in dict.fromkeys(stream_ids.tolist()).keys():
        idx = np.flatnonzero(stream_ids == sid)
        if len(idx) == 0:
            continue
        s_num, s_cat, s_y = build_sequences(
            num[idx],
            cat[idx],
            labels[idx],
            window_size=window_size,
            stride=stride,
        )
        if len(s_y) == 0:
            continue
        x_num_parts.append(s_num)
        x_cat_parts.append(s_cat)
        y_parts.append(s_y)

    if not x_num_parts:
        return (
            np.zeros((0, window_size, num.shape[1]), dtype=np.float32),
            np.zeros((0, window_size, cat.shape[1]), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(x_num_parts, axis=0),
        np.concatenate(x_cat_parts, axis=0),
        np.concatenate(y_parts, axis=0),
    )


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

    hidden_dim = int(agent_cfg.get("hidden_dim", 64))
    num_layers = int(agent_cfg.get("num_layers", 1))
    epochs = int(agent_cfg.get("epochs", 20))
    batch_size = int(agent_cfg.get("batch_size", 256))
    learning_rate = float(agent_cfg.get("learning_rate", 1e-3))
    dropout = float(agent_cfg.get("dropout", 0.1))
    window_size = max(1, int(agent_cfg.get("window_size", 8)))
    stride = max(1, int(agent_cfg.get("stride", 1)))

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)

    train_df, cal_df = _split_temporal_by_stream(df, val_fraction=val_fraction)

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

    z_train_cat = np.zeros((x_train.shape[0], 0), dtype=np.int64)
    z_cal_cat = np.zeros((x_cal.shape[0], 0), dtype=np.int64)

    train_stream_ids = _extract_stream_ids(train_df)
    cal_stream_ids = _extract_stream_ids(cal_df)

    x_train_seq, _, y_train_seq = _build_sequences_per_stream(
        num=x_train,
        cat=z_train_cat,
        labels=y_train,
        stream_ids=train_stream_ids,
        window_size=window_size,
        stride=stride,
    )
    x_cal_seq, _, y_cal_seq = _build_sequences_per_stream(
        num=x_cal,
        cat=z_cal_cat,
        labels=y_cal,
        stream_ids=cal_stream_ids,
        window_size=window_size,
        stride=stride,
    )
    if len(x_train_seq) == 0 or len(x_cal_seq) == 0:
        raise RuntimeError("No sequences built for LSTM autoencoder. Check window_size/stride and dataset length.")

    x_train_normal_seq = x_train_seq[y_train_seq == 0]
    if len(x_train_normal_seq) == 0:
        raise RuntimeError("No benign sequence windows found for LSTM autoencoder training")

    model = SequenceLSTMAutoencoder(
        in_dim=int(x_train_seq.shape[2]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_normal_seq.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    scores_train = _sequence_error(model, x_train_seq)
    scores_cal = _sequence_error(model, x_cal_seq)

    calibration_best = fit_best_calibrator(
        scores=np.asarray(scores_cal),
        labels=np.asarray(y_cal_seq),
        seed=int(seed),
        clip_lo=float(p_lo),
        clip_hi=float(p_hi),
    )
    calibrator = calibration_best["calibrator"]
    threshold_prob = float(calibration_best["threshold_probability"])
    best_ba = float(calibration_best["balanced_accuracy"])
    best_ece = float(calibration_best["ece"])
    best_brier = float(calibration_best["brier"])
    selected_calibrator = str(calibration_best["selected"])
    calibration_diagnostics = list(calibration_best["diagnostics"])

    benign = scores_cal[y_cal_seq == 0] if np.any(y_cal_seq == 0) else scores_cal
    malicious = scores_cal[y_cal_seq == 1] if np.any(y_cal_seq == 1) else scores_cal
    train_benign = scores_train[y_train_seq == 0] if np.any(y_train_seq == 0) else scores_train

    model_versions = collect_library_versions()
    payload = {
        "state_dict": model.state_dict(),
        "preprocessor": pre.to_dict(),
        "cat_cardinalities": cat_cardinalities,
        "model_config": {
            "in_dim": int(x_train_seq.shape[2]),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "window_size": int(window_size),
            "stride": int(stride),
        },
        "calibration": calibrator,
        "selected_calibrator": selected_calibrator,
        "calibration_diagnostics": calibration_diagnostics,
        "calibration_metrics": {
            "balanced_accuracy": float(best_ba),
            "ece": float(best_ece),
            "brier": float(best_brier),
        },
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
            "train_benign_mean": float(train_benign.mean()),
            "train_benign_std": float(train_benign.std() + 1e-9),
        },
        "meta": {
            "model_type": "sequence_lstm_autoencoder",
            "dataset": str(dataset),
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(cal_df)),
            "train_sequences": int(len(x_train_seq)),
            "calibration_sequences": int(len(x_cal_seq)),
            "features": int(x_train_seq.shape[2]),
            "window_size": int(window_size),
            "stride": int(stride),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "validation_fraction": float(val_fraction),
            "train_streams": int(len(set(train_stream_ids.tolist()))),
            "calibration_streams": int(len(set(cal_stream_ids.tolist()))),
            "balanced_accuracy": float(best_ba),
            "calibration_ece": float(best_ece),
            "calibration_brier": float(best_brier),
            "library_versions": model_versions,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    schema_to_json(
        pre,
        output.with_suffix(".schema.json"),
        extra={
            "model": "sequence_lstm_autoencoder",
            "window_size": int(window_size),
            "stride": int(stride),
        },
    )

    print(
        json.dumps(
            {
                "saved_model": str(output),
                "saved_schema": str(output.with_suffix(".schema.json")),
                "rows": int(len(df)),
                "features": int(x_train_seq.shape[2]),
                "train_sequences": int(len(x_train_seq)),
                "calibration_sequences": int(len(x_cal_seq)),
                "balanced_accuracy": float(best_ba),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "lstm_autoencoder.pt"
    default_cfg = Path(__file__).resolve().parents[2] / "config" / "agent_training.yaml"

    p = argparse.ArgumentParser(description="Train sequence LSTM autoencoder for UNSW-NB15")
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
