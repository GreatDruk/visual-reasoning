import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS).

    Deterministic algorithms are requested with warn_only=True: some MPS kernels
    have no deterministic variant, so runs on a Mac are reproducible only up to
    floating-point noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path):
    """Resolve a relative path against the repository root; absolute paths pass through."""
    path = Path(path)
    base_path = Path(__file__).resolve().parents[1]
    return path if path.is_absolute() else base_path / path


def get_device(name="auto"):
    """Return a torch.device, falling back cuda -> mps -> cpu when the requested one is missing."""
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("auto", "cuda", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
