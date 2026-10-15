from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments.task_data import mounted


class LocalSandbox:
    is_linux = True
    python = sys.executable

    def __init__(self, root):
        self.task_data_root = str(root)

    async def exists(self, path):
        return Path(path).exists()

    async def mkdir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    async def rm(self, paths):
        for path in map(Path, paths):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

    async def run_command(self, command, *, timeout=60):
        return await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    async def list_dir(self, path):
        return [{"relpath": str(entry.relative_to(path))} for entry in Path(path).rglob("*")]


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "mount with spaces"
    source = root / "domain/task/base"
    (source / "input").mkdir(parents=True)
    (source / "input/data.txt").write_text("input")
    (source / "software").mkdir()
    (source / "software/run.sh").write_text("exit 0\n")
    sandbox = LocalSandbox(tmp_path / "workspace")
    spec = TaskDataSpec(domain_name="domain", task_name="task", variant_name="base")
    return root, source, sandbox, spec


async def test_plaintext_reference_on_another_task_is_rejected_before_solver(data):
    root, source, sandbox, spec = data
    reference = root / "another/task/base/reference"
    reference.mkdir(parents=True)
    (reference / "answer.txt").write_text("answer")
    with pytest.raises(RuntimeError, match="Plaintext reference"):
        await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
    assert not (Path(sandbox.task_data_root) / "domain/task/base/input").exists()


async def test_input_alias_and_stale_reference_cleanup_use_real_subprocess(data):
    root, source, sandbox, spec = data
    spec.sandbox_task_name = "neutral"
    canonical = Path(sandbox.task_data_root) / "domain/task/base"
    (canonical / "reference").mkdir(parents=True)
    (canonical / "reference/answer.txt").write_text("stale")
    report = await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
    base = Path(sandbox.task_data_root) / "domain/neutral/base"
    assert report["staged"] == ["input", "software"]
    assert (base / "input/data.txt").read_text() == "input"
    assert os.access(base / "software/run.sh", os.X_OK)
    assert not (base / "reference").exists()
    assert not canonical.exists()


async def test_input_refresh_removes_stale_files_and_preserves_neutral_alias(data):
    root, source, sandbox, spec = data
    spec.sandbox_task_name = "neutral"
    base = Path(sandbox.task_data_root) / "domain/neutral/base"
    (base / "input").mkdir(parents=True)
    (base / "input/stale.txt").write_text("old")
    (base / "software").mkdir()
    (base / "software/stale.sh").write_text("old")
    (base / "output").mkdir()
    (base / "output/keep.txt").write_text("retained")
    await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
    assert not (base / "input/stale.txt").exists()
    assert not (base / "software/stale.sh").exists()
    assert (base / "input/data.txt").read_text() == "input"
    assert (base / "output/keep.txt").read_text() == "retained"
    assert not (Path(sandbox.task_data_root) / "domain/task").exists()


async def test_readonly_mount_copies_are_writable_without_changing_source(data):
    root, source, sandbox, spec = data
    source_input = source / "input"
    source_file = source_input / "data.txt"
    executable = source_input / "run.sh"
    executable.write_text("exit 0\n")
    executable.chmod(0o555)
    source_file.chmod(0o444)
    source_input.chmod(0o555)
    try:
        await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
        staged = Path(sandbox.task_data_root) / "domain/task/base/input"
        (staged / "secret.txt").write_text("generated during setup")
        (staged / "data.txt").write_text("edited copy")
        assert (staged / "run.sh").stat().st_mode & 0o111 == 0o111
        assert (staged / "data.txt").stat().st_mode & 0o111 == 0
        assert source_file.read_text() == "input"
        assert source_input.stat().st_mode & 0o222 == 0
        assert source_file.stat().st_mode & 0o222 == 0
        assert executable.stat().st_mode & 0o777 == 0o555
    finally:
        source_input.chmod(0o755)
        source_file.chmod(0o644)


@pytest.mark.parametrize("which", ["input", "destination"])
async def test_symlink_paths_are_rejected(data, which):
    root, source, sandbox, spec = data
    outside = root.parent / "outside"
    outside.mkdir()
    if which == "input":
        (source / "input/escape").symlink_to(outside)
    else:
        destination = Path(sandbox.task_data_root) / "domain/task/base"
        destination.mkdir(parents=True)
        (destination / "input").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlinks"):
        await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")


@pytest.mark.parametrize(
    "root",
    [
        "mounted:",
        "mounted:relative",
        "mounted:/",
        "mounted:/mnt/../data",
        "mounted://mnt/data",
        "/mnt/data",
    ],
)
def test_mounted_source_is_normalized_absolute(root):
    with pytest.raises(ValueError):
        mounted._mounted_task_dir(
            root, TaskDataSpec(domain_name="domain", task_name="task", variant_name="base")
        )


@pytest.mark.parametrize("part", ["", "..", "../escape", "/absolute", "back\\slash"])
def test_mounted_identity_cannot_escape_root(part):
    with pytest.raises(ValueError):
        mounted._mounted_task_dir(
            "mounted:/mnt/data",
            TaskDataSpec(domain_name="domain", task_name=part, variant_name="base"),
        )


async def test_mount_destination_overlap_is_rejected(data):
    root, source, sandbox, spec = data
    sandbox.task_data_root = str(root)
    with pytest.raises(RuntimeError, match="disjoint"):
        await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")


async def test_encrypted_reference_is_decrypted_only_during_evaluation(data, monkeypatch):
    if shutil.which("7z") is None:
        pytest.skip("requires local 7z")
    root, source, sandbox, spec = data
    payload = root.parent / "private-reference"
    payload.mkdir()
    (payload / "answer.txt").write_text("reference payload")
    subprocess.run(
        ["7z", "a", "-preview-only", "-mhe=on", str(source / "reference.7z"), "answer.txt"],
        cwd=payload,
        capture_output=True,
        check=True,
    )
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "review-only")
    await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
    base = Path(sandbox.task_data_root) / "domain/task/base"
    assert not (base / "reference").exists()
    assert not (base / "reference.7z").exists()
    report = await mounted.stage_reference(sandbox, spec, source=f"mounted:{root}")
    assert report["staged"] == ["reference"]
    assert (base / "reference/answer.txt").read_text() == "reference payload"


async def test_software_chmod_failure_is_not_ignored(data):
    root, source, sandbox, spec = data
    original = sandbox.run_command

    async def run(command, *, timeout=60):
        if command.startswith("find "):
            return subprocess.CompletedProcess(command, 1, "", "review failure")
        return await original(command, timeout=timeout)

    sandbox.run_command = run
    with pytest.raises(RuntimeError, match="executable"):
        await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")


async def test_references_from_late_mount_are_still_supported(data):
    root, source, sandbox, spec = data
    await mounted.stage_input(sandbox, spec, source=f"mounted:{root}")
    (source / "reference").mkdir()
    (source / "reference/answer.txt").write_text("late reference")
    await mounted.stage_reference(sandbox, spec, source=f"mounted:{root}")
    assert (
        Path(sandbox.task_data_root) / "domain/task/base/reference/answer.txt"
    ).read_text() == "late reference"
