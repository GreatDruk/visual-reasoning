"""Build the offline input bundle; model inputs never originate in artifact outputs."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .runtime import immutable_inputs


def build(project, inputs, output):
    project = Path(project).resolve()
    inputs, manifest = immutable_inputs(inputs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {
        p.relative_to(project).as_posix(): p
        for folder in ("src", "configs")
        for p in (project / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    files.update(
        {"inputs/" + p.relative_to(inputs).as_posix(): p for p in inputs.rglob("*") if p.is_file()}
    )
    hashes = {}
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
        z.writestr("package.json", json.dumps(dict(version=1, files=hashes), indent=2))
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
    a = p.parse_args()
    build(a.project, a.inputs, a.output)
