"""Utilidades de entorno: reproducibilidad y selección de dispositivo."""
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fija todas las semillas de aleatoriedad para que el experimento sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detecta automáticamente el mejor dispositivo disponible: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
