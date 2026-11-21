"""A worker failure must be visible and retain its complete diagnostic log."""

import io
import json
from pathlib import Path

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


def test_one_failed_configuration_does_not_cancel_the_rest(tmp_path, monkeypatch, capsys):
    class FailedProcess:
        def __init__(self):
            self.stdout = io.StringIO("failed independently\n")

        def wait(self):
            return 1

    monkeypatch.setattr(suite.subprocess, "Popen", lambda *a, **kw: FailedProcess())
    monkeypatch.setattr(suite, "load_config", lambda _: {"train": {"seed": 812}})
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Failed experiments"):
        suite.run_suite(
            tmp_path,
            tmp_path,
            tmp_path,
            ["021_resnet", "023_resnet"],
            tmp_path / "work",
            output,
            gpus=(0,),
            verified_inputs=True,
        )
    rows = json.loads((output / "results/suite.json").read_text())
    assert [r["name"] for r in rows] == ["021_resnet", "023_resnet"]
    assert "Starting 023_resnet" in capsys.readouterr().out


@pytest.mark.parametrize("failed_source", [False, True])
def test_recovery_prepares_each_source_once_before_starting_heads(
    tmp_path, monkeypatch, failed_source
):
    from src import foundations

    monkeypatch.setattr(foundations, "read_bundle", lambda *a, **kw: (tmp_path, {}))
    calls = []
    work = tmp_path / "work"
    output = tmp_path / "output"

    class Finished:
        def __init__(self, cmd, **kwargs):
            config = Path(cmd[cmd.index("--config") + 1])
            seed = int(cmd[cmd.index("--seed") + 1])
            prepare = "--prepare-only" in cmd
            self.failed = failed_source and prepare and config.stem == "046_union"
            calls.append((config.stem, prepare))
            if not prepare:
                run = work / config.stem / str(seed) / "run"
                run.mkdir(parents=True)
                (run / "receipt.json").write_text(
                    json.dumps(
                        dict(name=config.stem, metrics=dict(auc=1.0), elapsed_seconds=0, smoke=True)
                    )
                )
            self.stdout = io.StringIO("prepared\n" if prepare else "trained\n")

        def wait(self):
            return 1 if self.failed else 0

    monkeypatch.setattr(suite.subprocess, "Popen", Finished)

    def run():
        suite.run_suite(
            Path.cwd(),
            tmp_path,
            tmp_path,
            [f"{n:03d}_union" for n in range(45, 51)],
            work,
            output,
            verified_inputs=True,
            foundations=tmp_path,
        )

    if failed_source:
        with pytest.raises(RuntimeError, match="Failed experiments"):
            run()
        assert {name for name, prepare in calls if not prepare} == {"045_union", "050_union"}
    else:
        run()
    assert len(calls) == (5 if failed_source else 9) and all(prepare for _, prepare in calls[:3])
    assert {name for name, prepare in calls if prepare} == {"045_union", "046_union", "050_union"}
    assert json.loads((output / "results/timing.json").read_text())["wall_seconds"] >= 0
