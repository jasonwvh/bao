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


class Generator(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) * 2, int(out_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim) * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand((real.size(0), 1), device=real.device, dtype=real.dtype)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    crit_interp = critic(interp)
    grad_outputs = torch.ones_like(crit_interp)
    grads = torch.autograd.grad(
        outputs=crit_interp,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def _score_samples(critic: Critic, x_np: np.ndarray) -> np.ndarray:
    critic.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_np.astype(np.float32))
        # Higher anomaly score should indicate more suspicious.
        return (-critic(x).squeeze(1)).cpu().numpy()


def train(dataset: Path, output: Path, seed: int, config_path: Path | None) -> None:
    cfg = load_agent_training_config(config_path)
    shared = dict(cfg.get("shared", {}))
    pre_cfg = dict(shared.get("preprocessing", {}))
    cal_cfg = dict(shared.get("calibration", {}))
    agent_cfg = dict(cfg.get("wgan_gp", {}))

    val_fraction = float(shared.get("validation_fraction", 0.2))
    val_fraction = min(0.4, max(0.05, val_fraction))

    probability_clip = list(cal_cfg.get("probability_clip", [0.001, 0.999]))
    p_lo = float(probability_clip[0])
    p_hi = float(probability_clip[1])

    z_dim = int(agent_cfg.get("z_dim", 32))
    hidden_dim = int(agent_cfg.get("hidden_dim", 128))
    epochs = int(agent_cfg.get("epochs", 20))
    batch_size = int(agent_cfg.get("batch_size", 256))
    learning_rate = float(agent_cfg.get("learning_rate", 2e-4))
    n_critic = max(1, int(agent_cfg.get("n_critic", 5)))
    lambda_gp = float(agent_cfg.get("lambda_gp", 10.0))

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
        raise RuntimeError("No benign rows found for WGAN-GP training")

    generator = Generator(z_dim=z_dim, out_dim=int(x_train.shape[1]), hidden_dim=hidden_dim)
    critic = Critic(in_dim=int(x_train.shape[1]), hidden_dim=hidden_dim)
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_normal.astype(np.float32))), batch_size=batch_size, shuffle=True)

    generator.train()
    critic.train()
    for _ in range(epochs):
        for (real_batch,) in loader:
            real = real_batch

            for _ in range(n_critic):
                z = torch.randn(real.size(0), z_dim, dtype=real.dtype)
                fake = generator(z).detach()
                c_real = critic(real).mean()
                c_fake = critic(fake).mean()
                gp = _gradient_penalty(critic, real, fake)
                loss_c = -(c_real - c_fake) + lambda_gp * gp

                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            z = torch.randn(real.size(0), z_dim, dtype=real.dtype)
            fake = generator(z)
            loss_g = -critic(fake).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

    scores_train = _score_samples(critic, x_train)
    scores_cal = _score_samples(critic, x_cal)

    calibrator = fit_logistic_calibrator(scores_cal, y_cal, seed=seed)
    probs_cal = logistic_probability(scores_cal, calibrator, clip_lo=p_lo, clip_hi=p_hi)
    threshold_prob, best_ba = select_probability_threshold(np.asarray(probs_cal), y_cal)

    benign = scores_cal[y_cal == 0] if np.any(y_cal == 0) else scores_cal
    malicious = scores_cal[y_cal == 1] if np.any(y_cal == 1) else scores_cal
    train_benign = scores_train[y_train == 0] if np.any(y_train == 0) else scores_train

    payload = {
        "generator_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "preprocessor": pre.to_dict(),
        "cat_cardinalities": cat_cardinalities,
        "model_config": {
            "in_dim": int(x_train.shape[1]),
            "z_dim": int(z_dim),
            "hidden_dim": int(hidden_dim),
            "n_critic": int(n_critic),
            "lambda_gp": float(lambda_gp),
        },
        "calibration": calibrator,
        "threshold_probability": float(threshold_prob),
        "probability_clip": [p_lo, p_hi],
        "score_stats": {
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
            "model_type": "wgan_gp",
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

    schema_to_json(pre, output.with_suffix(".schema.json"), extra={"model": "wgan_gp"})

    print(
        json.dumps(
            {
                "saved_model": str(output),
                "rows": int(len(df)),
                "features": int(x_train.shape[1]),
                "balanced_accuracy": float(best_ba),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "wgan_gp.pt"
    default_cfg = Path(__file__).resolve().parents[2] / "config" / "agent_training.yaml"

    p = argparse.ArgumentParser(description="Train WGAN-GP tabular anomaly model for UNSW-NB15")
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
