"""``task_data_source: mounted:<dir>`` — cp-based staging inside the sandbox."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments import task_data as td
from ale_run.environments.task_data import mounted


@dataclass
class _FakeSandbox:
    """Records the sandbox I/O the backend performs; pretends `existing` paths exist."""

    existing: set[str]
    os: str = "linux"
    python: str = "python3"
    task_data_root: str = "/media/user/data/agenthle"
    cmds: list[str] = field(default_factory=list)
    mkdirs: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rc: int = 0

    @property
    def is_linux(self):
        return self.os == "linux"

    async def exists(self, p):
        return p in self.existing

    async def mkdir(self, p):
        self.mkdirs.append(p)

    async def rm(self, ps):
        self.removed.extend(ps)

    async def run_command(self, cmd, *, timeout=60):
        self.cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, self.rc, "", "boom" if self.rc else "")


def _spec():
    return TaskDataSpec(domain_name="demo", task_name="hello", variant_name="base")


def test_select_dispatches_mounted():
    assert td.select("mounted:/mnt/ale-task-data") is mounted


async def test_stage_input_copies_input_and_software():
    sb = _FakeSandbox(existing={"/mnt/d/demo/hello/base/input", "/mnt/d/demo/hello/base/software"})
    rep = await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d/")
    assert rep == {"staged": ["input", "software"], "source": "mounted"}
    base = "/media/user/data/agenthle/demo/hello/base"
    assert (
        f"{base}/input" in sb.mkdirs
        and f"{base}/output" in sb.mkdirs
        and f"{base}/software" in sb.mkdirs
    )
    assert (
        sb.cmds[1]
        == f"cp -a --no-preserve=ownership -- /mnt/d/demo/hello/base/input/. {base}/input/"
        f" && chmod -R u+rwX -- {base}/input"
    )
    assert (
        sb.cmds[2]
        == f"cp -a --no-preserve=ownership -- /mnt/d/demo/hello/base/software/. {base}/software/"
        f" && chmod -R u+rwX -- {base}/software"
    )
    assert sb.cmds[3].startswith(f"find {base}/software -type f -exec chmod +x")


async def test_stage_input_without_software_and_missing_input():
    sb = _FakeSandbox(existing={"/mnt/d/demo/hello/base/input"})
    rep = await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d")
    assert rep["staged"] == ["input"] and len(sb.cmds) == 2
    with pytest.raises(RuntimeError, match="expected input/"):
        await mounted.stage_input(_FakeSandbox(existing=set()), _spec(), source="mounted:/mnt/d")


async def test_stage_reference_after_agent_and_skip():
    sb = _FakeSandbox(existing={"/mnt/d/demo/hello/base/reference"})
    rep = await mounted.stage_reference(sb, _spec(), source="mounted:/mnt/d")
    ref = "/media/user/data/agenthle/demo/hello/base/reference"
    assert rep == {"staged": ["reference"], "source": "mounted"}
    assert (
        sb.removed == [ref]
        and sb.cmds[-1]
        == f"cp -a --no-preserve=ownership -- /mnt/d/demo/hello/base/reference/. {ref}/"
        f" && chmod -R u+rwX -- {ref}"
    )
    assert (
        await mounted.stage_reference(
            _FakeSandbox(existing=set()), _spec(), source="mounted:/mnt/d"
        )
    ) == {"skipped": True, "reason": "no_reference"}


async def test_cp_failure_raises():
    sb = _FakeSandbox(existing={"/mnt/d/demo/hello/base/input"}, rc=1)
    with pytest.raises(RuntimeError, match="validation failed"):
        await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d")


def test_config_loader_accepts_mounted_source(tmp_path):
    from ale_run.orchestration.config_loader import load_experiment

    (tmp_path / "env.yaml").write_text(
        "provider: hfsandbox\ntask_data_source: mounted:/mnt/ale-task-data\noutput_path: local\n"
    )
    (tmp_path / "agent.yaml").write_text("harness: dummy\nexecutor: local\nmodel: none\n")
    (tmp_path / "exp.yaml").write_text(
        "name: t\nagents: [agent.yaml]\nenvironment: env.yaml\ntasks:\n  - path: demo/hello\n"
    )
    spec = load_experiment(tmp_path / "exp.yaml")
    assert spec.artifacts.task_data_source == "mounted:/mnt/ale-task-data"
