"""Small, strict YAML configuration loader with relative inheritance."""

import re
from copy import deepcopy
from pathlib import Path

import yaml

NAME = re.compile(r"^\d{3}_[a-z]+$")


def load_config(path, stack=()):
    path = Path(path).resolve()
    if not NAME.fullmatch(path.stem) or path.suffix != ".yaml":
        raise ValueError("Configuration name must be NNN_word.yaml")
    if path in stack:
        raise ValueError("Configuration inheritance cycle")
    item = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError("Configuration must be a mapping")
    parent = item.pop("parent", None)
    if parent:
        if not NAME.fullmatch(parent):
            raise ValueError("Parent must be a configuration name")
        result = load_config(path.with_name(parent + ".yaml"), (*stack, path))
        result.pop("name")
        _merge(result, item)
    else:
        result = deepcopy(item)
    result["name"] = path.stem
    required = {
        "name",
        "family",
        "hypothesis",
        "sources",
        "model",
        "train",
        "siglip",
        "evaluation",
    }
    if set(result) != required:
        raise ValueError(f"Unexpected or missing configuration fields: {set(result) ^ required}")
    if result["family"] not in ("base", "resnet", "siglip"):
        raise ValueError("Unsupported experiment family")
    if result["model"]["neck"] not in ("sum", "bifpn", "multi"):
        raise ValueError("Unsupported neck")
    if result["model"]["backbone"] not in ("resnet", "convnext"):
        raise ValueError("Unsupported classifier backbone")
    if result["siglip"]["mode"] not in ("frozen", "tail", "adapter"):
        raise ValueError("Unsupported SigLIP adaptation")
    if result["siglip"]["crop"] not in ("stretch", "letterbox", "context"):
        raise ValueError("Unsupported crop transform")
    if result["siglip"]["proposals"] not in ("thaw", "resnet"):
        raise ValueError("Unsupported proposal source")
    if result["siglip"]["objective"] not in ("attributes", "joint"):
        raise ValueError("Unsupported adaptation objective")
    if not 0 < result["evaluation"]["blend"] < 1:
        raise ValueError("Blend must be strictly between zero and one")
    if result["train"]["epochs"] < 1 or result["train"]["batch"] < 1:
        raise ValueError("Epochs and batch must be positive")
    if sum(result["train"]["stages"]) != result["train"]["epochs"]:
        raise ValueError("Stage lengths must sum to the training epochs")
    if result["siglip"]["pool_grid"] not in (8, 16):
        raise ValueError("Pool grid must be 8 or 16")
    return result


def _merge(base, patch):
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"Unknown configuration field: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must remain a mapping")
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
