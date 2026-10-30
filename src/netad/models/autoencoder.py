"""Autoencoder trained on normal traffic, scored by reconstruction error per flow."""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _mlp(sizes: list[int]) -> nn.Sequential:
    """Linear layers with ReLU between them, none after the last one."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class Autoencoder(nn.Module):
    """n_features -> 32 -> 16 -> bottleneck -> 16 -> 32 -> n_features.

    Bottleneck and output are linear: the inputs are standardized, so they can be negative.
    """

    def __init__(self, n_features: int, bottleneck: int = 8, hidden: tuple[int, ...] = (32, 16)):
        super().__init__()
        self.n_features = n_features
        self.bottleneck = bottleneck
        self.hidden = tuple(hidden)
        sizes = [n_features, *hidden, bottleneck]
        self.encoder = _mlp(sizes)
        self.decoder = _mlp(sizes[::-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _tensor(X) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float32))


def reconstruction_error(model: Autoencoder, X, batch_size: int = 4096) -> np.ndarray:
    """Mean squared error of each row: one score per flow, whatever the batch size."""
    X = _tensor(X)
    model.eval()
    with torch.no_grad():
        return torch.cat([((model(xb) - xb) ** 2).mean(dim=1) for xb in X.split(batch_size)]).numpy()


def train_autoencoder(X_train, X_val, bottleneck: int = 8, hidden: tuple[int, ...] = (32, 16),
                      epochs: int = 500, batch_size: int = 256, lr: float = 1e-3, patience: int = 10,
                      seed: int = 42) -> tuple[Autoencoder, list[dict]]:
    """Fit on X_train with early stopping on the reconstruction loss of X_val.

    Both should be normal traffic only, already preprocessed. Returns the model from the epoch with
    the lowest validation loss (in eval mode) and the loss of every epoch.
    """
    torch.manual_seed(seed)
    X_train, X_val = _tensor(X_train), _tensor(X_val)
    model = Autoencoder(X_train.shape[1], bottleneck, hidden)
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best_loss, best_state, waited = np.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        val_loss = float(reconstruction_error(model, X_val).mean())
        history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, waited = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_autoencoder(model: Autoencoder, preprocessor, threshold: float,
                     directory: Path | str = MODELS_DIR, name: str = "autoencoder") -> None:
    """Weights (.pt), fitted preprocessor (.joblib) and architecture + threshold (.json)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / f"{name}.pt")
    joblib.dump(preprocessor, directory / f"{name}_preprocessor.joblib")
    config = {
        "n_features": model.n_features,
        "bottleneck": model.bottleneck,
        "hidden": list(model.hidden),
        "threshold": float(threshold),
    }
    (directory / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_autoencoder(directory: Path | str = MODELS_DIR, name: str = "autoencoder"):
    """Inverse of save_autoencoder: (model in eval mode, preprocessor, threshold)."""
    directory = Path(directory)
    config = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    model = Autoencoder(config["n_features"], config["bottleneck"], tuple(config["hidden"]))
    model.load_state_dict(torch.load(directory / f"{name}.pt", weights_only=True))
    model.eval()
    return model, joblib.load(directory / f"{name}_preprocessor.joblib"), config["threshold"]
