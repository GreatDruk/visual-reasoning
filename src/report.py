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
    columns = ["ID", "architecture", "seed", "val AUC", "val acc", "pair acc", "val_para AUC", "epochs", "fit min", "total min", *CATS]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, r in rows:
        v = r["val"]
        fmt = lambda value, digits=4: "—" if value is None else f"{value:.{digits}f}"
        values = [
            r["name_exp"], r.get("architecture", "—"), str(r.get("seed", "—")),
            fmt(v.get("auc")), fmt(v.get("accuracy")), fmt(v.get("pair_accuracy")),
            fmt(r.get("val_para", {}).get("auc")), str(r.get("epochs_run", "—")),
            fmt(r.get("fit_min"), 2), fmt(r.get("total_min"), 2),
            *(fmt(v.get(f"auc/{c}"), 3) for c in CATS),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(collect("--all" in sys.argv)))
