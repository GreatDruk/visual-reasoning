"""Build the offline input bundle; model inputs never originate in artifact outputs."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .runtime import immutable_inputs


def build(
    project, inputs, output, code_only=False, foundation_runs=None, assets=None, anchors=None
):
    project = Path(project).resolve()
    if not code_only:
        inputs, manifest = immutable_inputs(inputs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {
        p.relative_to(project).as_posix(): p
        for folder in ("src", "configs")
        for p in (project / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if not code_only:
        files.update(
            {
                "inputs/" + p.relative_to(inputs).as_posix(): p
                for p in inputs.rglob("*")
                if p.is_file()
            }
        )
    hashes = {}
    foundation_manifest = None
    if foundation_runs:
        foundation_manifest = dict(
            version=1,
            models={},
            files={},
            baselines={
                "quality": {"convnext": 0.5, "sigft": 0.5},
                "fast": {"resnet": 0.5, "sigft": 0.5},
                "compact": {"convnext": 1.0},
            },
        )
        for name, run in foundation_runs.items():
            if name not in ("convnext", "resnet"):
                raise ValueError("Register convnext and resnet foundation roles explicitly")
            run = Path(run).resolve()
            report = json.loads((run / "report.json").read_text(encoding="utf8"))
            if report.get("smoke") or len(report["fit_scene_names"]) != 5000:
                raise ValueError("Foundation bundle requires full train-only research runs")
            if report["config"]["model"]["backbone"] != name:
                raise ValueError("Foundation role and backbone differ")
            weight = (
                project
                / "artefacts"
                / "weights"
                / run.relative_to(project / "artefacts" / "results")
                / "model.pt"
            )
            mapping = {f"{name}.pt": weight, f"{name}_val.npz": run / "branch.npz"}
            for filename, path in mapping.items():
                files["foundations/" + filename] = path
                with path.open("rb") as stream:
                    foundation_manifest["files"][filename] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
            foundation_manifest["models"][name] = dict(
                checkpoint=f"{name}.pt",
                scores=f"{name}_val.npz",
                config=report["config"],
                fit_scene_names=report["fit_scene_names"],
                validation_csv_sha256=report["validation_csv_sha256"],
                source_sha256=report["source_sha256"],
            )
        if set(foundation_runs) != {"convnext", "resnet"}:
            raise ValueError("Both fixed foundation roles are required")
    if assets or anchors:
        if foundation_manifest is None:
            raise ValueError("Extra inputs require a foundation manifest")
        for registry, items in [("assets", assets or {}), ("anchors", anchors or {})]:
            foundation_manifest[registry] = {}
            for name, path in items.items():
                if not name.isidentifier():
                    raise ValueError("Input role must be an identifier")
                path = Path(path)
                filename = f"{registry}_{name}{path.suffix}"
                files["foundations/" + filename] = path
                with path.open("rb") as stream:
                    foundation_manifest["files"][filename] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
                foundation_manifest[registry][name] = filename
    with zipfile.ZipFile(output.with_suffix(".tmp"), "w") as z:
        for name, path in sorted(files.items()):
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            z.write(
                path,
                name,
                compress_type=zipfile.ZIP_DEFLATED
                if path.suffix in (".py", ".yaml", ".json")
                else zipfile.ZIP_STORED,
            )
        if foundation_manifest is not None:
            content = json.dumps(foundation_manifest, indent=2).encode("utf8")
            z.writestr("foundations/manifest.json", content)
            hashes["foundations/manifest.json"] = hashlib.sha256(content).hexdigest()
        z.writestr(
            "runtime.json" if code_only else "package.json",
            json.dumps(dict(version=1, files=hashes), indent=2),
        )
    with zipfile.ZipFile(output.with_suffix(".tmp")) as z:
        if z.testzip():
            raise ValueError("Archive CRC failed")
    output.with_suffix(".tmp").replace(output)
    print(output, len(files), "files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--inputs", type=Path, default=Path("inputs"))
    p.add_argument("--output", type=Path, default=Path("artefacts/kaggle/base.zip"))
    p.add_argument(
        "--code-only",
        action="store_true",
        help="Small runtime overlay for an existing immutable input dataset",
    )
    p.add_argument(
        "--foundation",
        action="append",
        default=[],
        metavar="ROLE=RESULT_RUN",
        help="Package an explicit full train-only run for downstream research",
    )
    p.add_argument("--asset", action="append", default=[], metavar="ROLE=PATH")
    p.add_argument("--anchor", action="append", default=[], metavar="ROLE=NPZ")
    a = p.parse_args()
    runs = dict(item.split("=", 1) for item in a.foundation)
    build(
        a.project,
        a.inputs,
        a.output,
        a.code_only,
        runs or None,
        dict(item.split("=", 1) for item in a.asset),
        dict(item.split("=", 1) for item in a.anchor),
    )
