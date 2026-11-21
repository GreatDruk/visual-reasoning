"""Execute the notebook bootstrap against Kaggle dataset layouts, without training."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def package_files():
    files = {
        "src/__init__.py": b'"""An existing project bundle without a new bootstrap module."""\n',
        "configs/019_base.yaml": b"family: base\n",
        "inputs/manifest.json": b'{"version": 1, "files": {}}\n',
    }
    manifest = dict(version=1, files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    return {**files, "package.json": json.dumps(manifest).encode("utf-8")}


def unpack(root):
    for name, data in package_files().items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    working = tmp_path / "working"
    competition = inputs / "competition"
    competition.mkdir()
    for split in ("train", "val"):
        (competition / split).mkdir()
        (competition / f"{split}.csv").write_text("id,image\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("KAGGLE_WORK_ROOT", str(working))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    notebook = json.loads(Path("notebooks/base.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][2]["source"])

    def execute():
        namespace = {}
        exec(compile(source, "base.ipynb/bootstrap", "exec"), namespace)
        assert namespace["data"] == competition
        execute.namespace = namespace
        return namespace["project"]

    return inputs, working, execute


def test_bootstrap_reads_kaggle_unpacked_dataset_without_copying(bootstrap):
    inputs, working, execute = bootstrap
    project = unpack(inputs / "visual-reasoning-kaggle-input")
    assert execute() == project
    assert not (working / "project").exists()
    assert not list(inputs.rglob("base.zip"))


def test_bootstrap_retains_support_for_zip_datasets(bootstrap):
    inputs, working, execute = bootstrap
    with zipfile.ZipFile(inputs / "base.zip", "w") as bundle:
        for name, data in package_files().items():
            bundle.writestr(name, data)
    assert execute() == working / "project"
    assert (working / "project/inputs/manifest.json").is_file()


def test_bootstrap_checks_unpacked_file_hashes(bootstrap):
    inputs, _, execute = bootstrap
    project = unpack(inputs / "project")
    (project / "src/__init__.py").write_bytes(b"modified")
    with pytest.raises(AssertionError, match="Package checksum mismatch"):
        execute()


def test_bootstrap_rejects_ambiguous_unpacked_datasets(bootstrap):
    inputs, _, execute = bootstrap
    unpack(inputs / "first")
    unpack(inputs / "second")
    with pytest.raises(AssertionError, match="Several unpacked project datasets"):
        execute()


def test_zip_bootstrap_rejects_paths_outside_the_project(bootstrap):
    inputs, working, execute = bootstrap
    with zipfile.ZipFile(inputs / "base.zip", "w") as bundle:
        bundle.writestr("../escape.py", "invalid")
    with pytest.raises(AssertionError, match="Unsafe archive path"):
        execute()
    assert not (working / "escape.py").exists()


def test_runtime_overlay_updates_code_without_copying_model_inputs(bootstrap):
    inputs, working, execute = bootstrap
    original = unpack(inputs / "original")
    overlay = inputs / "runtime"
    overlay.mkdir()
    source = b'"""Updated code."""\n'
    (overlay / "src").mkdir()
    (overlay / "src/__init__.py").write_bytes(source)
    (overlay / "runtime.json").write_text(
        json.dumps(dict(version=1, files={"src/__init__.py": hashlib.sha256(source).hexdigest()})),
        encoding="utf-8",
    )
    project = execute()
    assert project == working / "runtime_project"
    assert (project / "src/__init__.py").read_bytes() == source
    assert execute.namespace["inputs"] == original / "inputs"
    assert not (project / "inputs").exists()


def test_notebook_rejects_the_old_runtime_before_starting_any_jobs(monkeypatch):
    notebook = json.loads(Path("notebooks/base.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    old = ModuleType("src")
    old.runtime = SimpleNamespace(gradient_step=lambda: None)
    monkeypatch.setitem(sys.modules, "src", old)
    with pytest.raises(AssertionError, match="Update the runtime dataset"):
        exec(compile(source, "base.ipynb/run", "exec"), {})


def test_runtime_overlay_keeps_foundation_weights_in_the_input_dataset(bootstrap):
    inputs, working, execute = bootstrap
    unpack(inputs / "original")
    overlay = inputs / "runtime"
    (overlay / "foundations").mkdir(parents=True)
    content = b"checksummed frozen weights"
    (overlay / "foundations/model.pt").write_bytes(content)
    (overlay / "runtime.json").write_text(
        json.dumps(
            dict(version=1, files={"foundations/model.pt": hashlib.sha256(content).hexdigest()})
        ),
        encoding="utf8",
    )
    project = execute()
    assert execute.namespace["foundations"] == overlay / "foundations"
    assert not (project / "foundations").exists()
    assert not list(working.rglob("*.pt"))
