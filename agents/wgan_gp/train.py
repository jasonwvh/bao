from __future__ import annotations

import argparse
import os
import sys
# When running this script directly, ensure the project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agents.common.preprocessing import fit_preprocessor, load_csv, schema_to_json, transform_frame


class Generator(nn.Module):
    def __init__(self, z_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, num_dim: int, cat_cardinalities: list[int]):
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        self.emb_dims = []
        total_emb_dim = 0
        for card in cat_cardinalities:
            emb_dim = int(min(16, max(4, card // 4)))
            self.cat_embeddings.append(nn.Embedding(int(max(2, card)), emb_dim))
            self.emb_dims.append(emb_dim)
            total_emb_dim += emb_dim
        self.num_dim = num_dim
        input_dim = num_dim + total_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.cat_embeddings) == 0:
            return torch.empty(x_cat.size(0), 0, device=x_cat.device)
        parts = []
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, i]))
        return torch.cat(parts, dim=-1)

    def prepare_input(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embedded_cat = self.embed_categorical(x_cat)
        return torch.cat([x_num, embedded_cat], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    pred = critic(interp)
    grads = torch.autograd.grad(
        outputs=pred,
        inputs=interp,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def train(dataset: Path, output: Path, seed: int, epochs: int, batch_size: int, lr: float, z_dim: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)

    y = df["label"].astype(int).to_numpy()
    pre = fit_preprocessor(df)
    num, cat = transform_frame(pre, df)
    cat_cardinalities = [len(pre.vocabularies[c]) for c in pre.categorical_cols]

    num_normal = num[y == 0]
    cat_normal = cat[y == 0]
    if len(num_normal) == 0:
        raise RuntimeError("No benign rows for WGAN-GP training")

    device = torch.device("cpu")
    critic = Critic(num_dim=num.shape[1], cat_cardinalities=cat_cardinalities).to(device)
    
    total_emb_dim = sum(critic.emb_dims)
    generator = Generator(z_dim=z_dim, out_dim=num.shape[1] + total_emb_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    dataset_tensor = TensorDataset(
        torch.from_numpy(num_normal.astype(np.float32)),
        torch.from_numpy(cat_normal.astype(np.int64))
    )
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True, drop_last=True)

    lambda_gp = 10.0
    n_critic = 5

    generator.train()
    critic.train()
    for _ in range(epochs):
        for real_num, real_cat in loader:
            real_num = real_num.to(device)
            real_cat = real_cat.to(device)
            
            for _ in range(n_critic):
                real = critic.prepare_input(real_num, real_cat)
                z = torch.randn(real.size(0), z_dim, device=device)
                fake = generator(z).detach()

                c_real = critic(real).mean()
                c_fake = critic(fake).mean()
                gp = _gradient_penalty(critic, real, fake)
                loss_c = -(c_real - c_fake) + lambda_gp * gp

                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            z = torch.randn(real_num.size(0), z_dim, device=device)
            fake = generator(z)
            loss_g = -critic(fake).mean()

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

    generator.eval()
    critic.eval()
    with torch.no_grad():
        num_t = torch.from_numpy(num.astype(np.float32)).to(device)
        cat_t = torch.from_numpy(cat.astype(np.int64)).to(device)
        x_all = critic.prepare_input(num_t, cat_t)
        scores = -critic(x_all).squeeze(1).cpu().numpy()

    benign_mean = float(np.mean(scores[y == 0])) if np.any(y == 0) else float(np.mean(scores))
    mal_mean = float(np.mean(scores[y == 1])) if np.any(y == 1) else float(np.mean(scores))
    invert = mal_mean < benign_mean
    if invert:
        scores = -scores

    calib = {}
    if len(np.unique(y)) > 1:
        lr_model = LogisticRegression(max_iter=1000, random_state=seed)
        lr_model.fit(scores.reshape(-1, 1), y)
        calib = {"coef": float(lr_model.coef_[0][0]), "intercept": float(lr_model.intercept_[0])}

    benign = scores[y == 0] if np.any(y == 0) else scores
    malicious = scores[y == 1] if np.any(y == 1) else scores

    payload = {
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "preprocessor": pre.to_dict(),
        "cat_cardinalities": cat_cardinalities,
        "model_config": {
            "num_dim": int(num.shape[1]),
            "emb_dims": list(critic.emb_dims),
            "z_dim": int(z_dim),
            "invert_score": bool(invert),
        },
        "calibration": calib,
        "score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores) + 1e-9),
            "p95": float(np.percentile(scores, 95.0)),
            "benign_mean": float(np.mean(benign)),
            "benign_std": float(np.std(benign) + 1e-9),
            "mal_mean": float(np.mean(malicious)),
            "mal_std": float(np.std(malicious) + 1e-9),
        },
        "meta": {
            "dataset": str(dataset),
            "rows": int(len(df)),
            "features": int(num.shape[1]),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "seed": int(seed),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    schema_to_json(pre, output.with_suffix(".schema.json"))

    print(json.dumps({"saved_model": str(output), "rows": int(len(df)), "features": int(num.shape[1])}, indent=2))


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "wgan_gp.pt"

    p = argparse.ArgumentParser(description="Train WGAN-GP anomaly detector on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pt model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--z-dim", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        z_dim=int(args.z_dim),
    )


if __name__ == "__main__":
    main()
