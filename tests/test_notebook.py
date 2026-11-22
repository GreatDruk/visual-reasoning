"""Final notebook bootstrap: offline unpacked/ZIP inputs, hashes and ambiguity."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch


def files():
    content = {
        "src/__init__.py": b"",
        "configs/057_quality.yaml": b"family: submission",
        "inputs/manifest.json": b"{}",
    }
    spec = {
        "version": 1,
        "files": {
            k: hashlib.sha256(v).hexdigest()
            for k, v in content.items()
            if k.startswith(("src/", "configs/"))
        },
    }
    return {**content, "snapshot057.json": json.dumps(spec).encode()}


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    working = tmp_path / "working"
    data = inputs / "competition"
    data.mkdir()
    for split in ("train", "test"):
        (data / split).mkdir()
        (data / (split + ".csv")).write_text("id,image\n")
    (data / "sample_submission.csv").write_text("id,label\n")
    monkeypatch.setenv("KAGGLE_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("KAGGLE_WORK_ROOT", str(working))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    source = "".join(
        json.loads(Path("notebooks/base.ipynb").read_text(encoding="utf-8"))["cells"][2]["source"]
    )

    def unpack(folder):
        for name, content in files().items():
            p = folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        return folder

    def run():
        scope = {}
        exec(compile(source, "notebook", "exec"), scope)
        assert scope["data"] == data
        return scope

    return inputs, working, unpack, run


def test_unpacked_final_bundle_is_used_in_place(bootstrap):
    inputs, working, unpack, run = bootstrap
    root = unpack(inputs / "final")
    assert run()["project"] == root
    assert not working.exists()


def test_zip_fallback(bootstrap):
    inputs, working, _, run = bootstrap
    with zipfile.ZipFile(inputs / "057.zip", "w") as z:
        for name, content in files().items():
            z.writestr(name, content)
    assert run()["project"] == working / "project"


def test_modified_source_is_rejected(bootstrap):
    inputs, _, unpack, run = bootstrap
    root = unpack(inputs / "final")
    (root / "src/__init__.py").write_text("changed")
    with pytest.raises(AssertionError, match="Source checksum"):
        run()


def test_ambiguous_bundles_rejected(bootstrap):
    inputs, _, unpack, run = bootstrap
    unpack(inputs / "one")
    unpack(inputs / "two")
    with pytest.raises(AssertionError, match="Several final"):
        run()


def test_zip_traversal_rejected(bootstrap):
    inputs, working, _, run = bootstrap
    with zipfile.ZipFile(inputs / "057.zip", "w") as z:
        z.writestr("../escape.py", b"bad")
    with pytest.raises(AssertionError, match="Unsafe archive"):
        run()
    assert not (working / "escape.py").exists()
