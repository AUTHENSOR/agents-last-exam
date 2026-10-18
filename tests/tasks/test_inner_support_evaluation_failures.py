import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_inner_support_source_review import ProcessSession

from tasks.engineering.inner_support_elevation_optimization import main, native_task, source_review
from tasks.engineering.inner_support_elevation_optimization.verification import (
    EvaluationUnavailableError,
)

pytest_plugins = ("test_inner_support_resource_budget", "test_inner_support_source_review")


@pytest.fixture
def guest_submission(sources, tmp_path):
    assets = Path(source_review.__file__).parent / "assets"
    (source_review.ASSETS / "benchmark.json").write_bytes((assets / "benchmark.json").read_bytes())
    output = tmp_path / "guest/output"
    family = output / "model_family_support_position"
    family.mkdir(parents=True)
    for name in ("family.json", "model.mdpa", "Materials.json"):
        (family / name).write_text("fixture")
    return {"input_dir": str(sources), "remote_output_dir": str(output)}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["family.json", "model.mdpa", "Materials.json"])
@pytest.mark.parametrize("damage", ["missing", "symlink", "directory"])
async def test_invalid_submission_scores_zero_through_remote_boundary(
    guest_submission, monkeypatch, caplog, name, damage
):
    metadata = guest_submission
    path = Path(metadata["remote_output_dir"]) / "model_family_support_position" / name
    path.unlink()
    if damage == "symlink":
        path.symlink_to(Path(metadata["input_dir"]) / "notes.md")
    elif damage == "directory":
        path.mkdir()

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid submission started native replay")

    monkeypatch.setattr(main, "replay_submission", forbidden)
    with caplog.at_level(logging.INFO, logger=main.__name__):
        assert await main.evaluate(SimpleNamespace(metadata=metadata), ProcessSession()) == [0.0]
    gates = list(
        Path(metadata["remote_output_dir"]).parent.glob(
            ".support-evaluation/*/review/evaluation-gate.json"
        )
    )
    assert len(gates) == 1
    gate = json.loads(gates[0].read_text())
    assert gate["evaluation_status"] == "source_noncompliant"
    assert gate["findings"][0]["submitted_feature"] == "family/" + name
    assert not gate["native_replay_completed"]
    assert "Support evaluation result:" in caplog.text and name in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "changed"])
async def test_broken_official_input_is_not_hidden_by_missing_submission(guest_submission, damage):
    metadata = guest_submission
    (Path(metadata["remote_output_dir"]) / "model_family_support_position/family.json").unlink()
    source = Path(metadata["input_dir"]) / "notes.md"
    if damage == "missing":
        source.unlink()
    else:
        source.write_text("corrupt official input")
    with pytest.raises(EvaluationUnavailableError, match="notes.md"):
        await main.evaluate(SimpleNamespace(metadata=metadata), ProcessSession())


@pytest.mark.asyncio
async def test_resource_probe_failure_does_not_interrupt_complete_replay(
    host_replay, monkeypatch, caplog
):
    control = host_replay
    control.receipt = {
        "completed": True,
        "cases": {case: {"directory": f"/replay/{case}"} for case in native_task.CASES},
    }
    control.status.update(ExecMainStatus="0", Result="success", ControlGroup="/fixture")
    original_command = native_task.run_command
    probes = []

    async def command(session, command, **kwargs):
        if kwargs.get("timeout") == 15:
            probes.append(command)
            raise EvaluationUnavailableError("Support remote command exceeded 15s")
        return await original_command(session, command, **kwargs)

    monkeypatch.setattr(native_task, "run_command", command)
    with caplog.at_level(logging.INFO, logger=native_task.__name__):
        result = await native_task.replay_submission(
            control.metadata, control.session, control.review
        )
    assert probes and set(result["observations"]) == set(native_task.CASES)
    live = json.loads(control.files[str(control.evidence / "native-live.json")])
    assert "exceeded 15s" in live["resource_probe_error"]
    assert "Support resource probe failed" in caplog.text
    assert "Support native replay result" in caplog.text


@pytest.mark.asyncio
async def test_critical_process_status_failure_remains_evaluation_error(host_replay, monkeypatch):
    control = host_replay
    original_command = native_task.run_command

    async def command(session, command, **kwargs):
        if command.startswith("systemctl show"):
            raise EvaluationUnavailableError("Cannot confirm native process status")
        return await original_command(session, command, **kwargs)

    monkeypatch.setattr(native_task, "run_command", command)
    with pytest.raises(EvaluationUnavailableError, match="Cannot confirm"):
        await native_task.replay_submission(control.metadata, control.session, control.review)
    assert not control.reads
