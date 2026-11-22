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
    extension = item.pop("union", None)
    count_extension = item.pop("count", None)
    if count_extension is not None and not isinstance(count_extension, dict):
        raise ValueError("Count settings must be a mapping")
    if extension is not None and not isinstance(extension, dict):
        raise ValueError("Union settings must be a mapping")
    if parent:
        if not NAME.fullmatch(parent):
            raise ValueError("Parent must be a configuration name")
        result = load_config(path.with_name(parent + ".yaml"), (*stack, path))
        result.pop("name")
        _merge(result, item)
    else:
        result = deepcopy(item)
    result["name"] = path.stem
    if result.get("family") == "pipeline":
        if set(result) != {"name", "family", "hypothesis", "sources", "pipeline"}:
            raise ValueError("Unexpected pipeline configuration fields")
        return result
    if result.get("family") == "submission":
        required = {"name", "family", "hypothesis", "sources", "blend"}
        if set(result) != required:
            raise ValueError("Unexpected submission configuration fields")
        blend = result["blend"]
        blend.setdefault("swap_source", "count53")
        blend.setdefault("count_primary", "count54_421")
        blend.setdefault("count_secondary", "count54_853")
        blend.setdefault("confirm_primary", False)
        if set(blend) != {
            "base",
            "counter_weight",
            "tta_weight",
            "hidden_weight",
            "swap",
            "swap_source",
            "count_primary",
            "count_secondary",
            "confirm_primary",
        }:
            raise ValueError("Unexpected submission blend settings")
        if blend["swap_source"] not in ("count53", "count54"):
            raise ValueError("Unknown count agreement source")
        if not all(
            re.fullmatch(r"count\d+_\d+", blend[k]) for k in ("count_primary", "count_secondary")
        ):
            raise ValueError("Invalid count component role")
        if not isinstance(blend["confirm_primary"], bool) or (
            blend["confirm_primary"] and not blend["counter_weight"]
        ):
            raise ValueError("Primary confirmation requires a primary counter")
        if blend["base"] not in ("compact", "quality") or not isinstance(blend["swap"], bool):
            raise ValueError("Invalid submission base or swap")
        if any(not 0 <= blend[k] <= 1 for k in ("counter_weight", "tta_weight", "hidden_weight")):
            raise ValueError("Submission weights must be between zero and one")
        return result
    if extension is not None:
        result["union"] = {**result.get("union", {}), **extension}
    if count_extension is not None:
        result["count"] = {**result.get("count", {}), **count_extension}
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
    if set(result) - {"union", "count"} != required:
        raise ValueError(f"Unexpected or missing configuration fields: {set(result) ^ required}")
    if result["family"] not in ("base", "resnet", "siglip", "union", "count"):
        raise ValueError("Unsupported experiment family")
    if "union" in result and result["family"] != "union":
        raise ValueError("Union settings require the union experiment family")
    if "count" in result and result["family"] != "count":
        raise ValueError("Count settings require the count experiment family")
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
    if result["train"].get("soft_background_weak_only") and not result["train"]["soft_background"]:
        raise ValueError("Weak-only uncertainty requires soft background supervision")
    if result["family"] == "union":
        u = result.get("union", {})
        fields = {
            "source",
            "relational",
            "roi",
            "phase",
            "proposal_floor",
            "single_floor",
            "radius_factor",
            "radius_min",
            "radius_max",
            "extract_batch",
            "weight",
            "rescue_weight",
            "primary",
        }
        if set(u) != fields:
            raise ValueError(f"Unexpected or missing union fields: {set(u) ^ fields}")
        if u["phase"] not in ("zero", "scene", "epoch"):
            raise ValueError("Unsupported training phase policy")
        if not 0 < u["proposal_floor"] <= u["single_floor"] < 1:
            raise ValueError("Invalid proposal probabilities")
        if not 0 < u["radius_min"] <= u["radius_max"] or u["radius_factor"] <= 0:
            raise ValueError("Invalid matching radius")
        if not 0 < u["weight"] < 1 or not 0 <= u["rescue_weight"] <= 1:
            raise ValueError("Invalid recovery blend weights")
        if not isinstance(u["relational"], bool) or not isinstance(u["roi"], bool):
            raise ValueError("Head switches must be boolean")
        if u["extract_batch"] < 1 or not isinstance(u["source"], str):
            raise ValueError("Invalid extraction source or batch")
    if result["family"] == "count":
        c = result.get("count", {})
        c.setdefault("local_weight", 0.0)
        c.setdefault("export_test", False)
        c.setdefault("initial", None)
        c.setdefault("frozen_scale", 0)
        expected = {
            "source",
            "density",
            "paired",
            "difficulty",
            "attempts",
            "absolute_weight",
            "pair_weight",
            "weight",
            "primary",
            "local_weight",
            "export_test",
            "initial",
            "frozen_scale",
        }
        if set(c) != expected:
            raise ValueError(f"Unexpected or missing count fields: {set(c) ^ expected}")
        if c["density"] not in ("factorized", "joint") or c["difficulty"] not in ("mixed", "tiny"):
            raise ValueError("Unsupported density or donor visibility policy")
        if not isinstance(c["paired"], bool) or not 1 <= c["attempts"] <= 4:
            raise ValueError("Invalid counterfactual sampling policy")
        if not 0 < c["weight"] < 1 or min(c["absolute_weight"], c["pair_weight"]) < 0:
            raise ValueError("Invalid count loss or blend weight")
        if c["local_weight"] < 0 or not isinstance(c["export_test"], bool):
            raise ValueError("Invalid local count loss or test export")
        if c["local_weight"] and (not c["paired"] or c["density"] != "factorized"):
            raise ValueError("Local delta currently requires paired factorized density")
        if c["frozen_scale"] not in (0, 1, 2):
            raise ValueError("Frozen feature scale must be zero, one or two")
        if c["frozen_scale"] and (c["paired"] or not c["initial"]):
            raise ValueError("Frozen readout requires an initial counter and no RGB augmentation")
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
