"""Collect results.json files from artefacts/hydra into a markdown table.

    uv run python -m src.report            # main runs (name_exp starting with a digit)
    uv run python -m src.report --all      # include debug runs
"""

import json
import sys

from .utils import resolve_path

CATS = ["attribute", "binding", "comparison", "counting", "negation", "position", "spatial", "spatial_counting", "spatial_negation"]


def collect(include_all=False):
    rows = []
    for path in sorted(resolve_path("artefacts/hydra").rglob("results.json")):
        r = json.loads(path.read_text())
        if "val" not in r or (not include_all and not r["name_exp"][:1].isdigit()):
            continue
        rows.append((path, r))
    return rows


def table(rows):
    head = "| ID | val AUC | val acc | pair acc | val_para AUC | twin-norm AUC | epochs | min | " + " | ".join(CATS) + " |"
    lines = [head, "|" + "---|" * (8 + len(CATS))]
    for _, r in rows:
        v = r["val"]
        cats = " | ".join(f"{v.get(f'auc/{c}', float('nan')):.3f}" for c in CATS)
        lines.append(
            f"| {r['name_exp']} | {v['auc']:.4f} | {v['accuracy']:.4f} | {v['pair_accuracy']:.4f} | "
            f"{r['val_para']['auc']:.4f} | {r['val_twin_norm']['auc']:.4f} | {r['epochs_run']} | {r['total_min']} | {cats} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(collect("--all" in sys.argv)))
