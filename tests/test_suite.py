"""A worker failure must be visible and retain its complete diagnostic log."""

import io
import json

import pytest

from src import suite


def test_failed_worker_streams_and_preserves_logs(tmp_path, monkeypatch, capsys):
    class FailedProcess:
        stdout = io.StringIO("PROGRESS stage started\nFloatingPointError: invalid gradient\n")

        def wait(self):
            return 1

    monkeypatch.setattr(suite.subprocess, "Popen", lambda *a, **kw: FailedProcess())
    monkeypatch.setattr(suite, "load_config", lambda _: {"train": {"seed": 421}})
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Failed experiments"):
        suite.run_suite(
            tmp_path,
            tmp_path,
            tmp_path,
            ["020_resnet"],
            tmp_path / "work",
            output,
            gpus=(0,),
            verified_inputs=True,
        )
    captured = capsys.readouterr().out
    assert "[020_resnet GPU0] PROGRESS stage started" in captured
    assert "FAILED 020_resnet" in captured
    assert "FloatingPointError" in (output / "results/logs/020_resnet_421.log").read_text()
    rows = json.loads((output / "results/suite.json").read_text())
    assert rows[0]["status"] == "failed"
