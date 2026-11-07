"""Generación, partición y carga del dataset de clasificación binaria sintético."""
from pathlib import Path

import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_dataloaders(
    n_samples: int = 5000,
    n_features: int = 20,
    n_informative: int = 15,
    val_size: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    save_csv: bool = True,
):
    """Crea el dataset, lo divide en train/val (estratificado) y devuelve DataLoaders."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_classes=2,
        random_state=seed,
    )

    if save_csv:  # deja el dataset versionable en data/
        DATA_DIR.mkdir(exist_ok=True)
        cols = [f"x{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        df.to_csv(DATA_DIR / "synthetic_classification.csv", index=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # El scaler se ajusta SOLO con train para evitar data leakage hacia validación
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )

    g = torch.Generator().manual_seed(seed)  # shuffle reproducible
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_features
