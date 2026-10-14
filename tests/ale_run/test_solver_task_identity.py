from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.base_interface import SandboxHandle
from ale_run.executors.sandbox import _sandbox_to_kwargs
from ale_run.orchestration.lifecycle import _build_executor
from ale_run.tasks.loader import TaskLoader


@pytest.mark.parametrize("agent_name", ["codex", "kimi-code"])
@pytest.mark.parametrize("os_type", ["linux", "windows"])
def test_aliased_task_keeps_canonical_identity_out_of_guest_spec(
    tmp_path, monkeypatch, agent_name, os_type
):
    monkeypatch.setattr("ale_run.orchestration.lifecycle._collect_env_passthrough", dict)
    task_root = Path(__file__).resolve().parents[2] / "tasks/life_sciences/rgi_mcr1_colistin_v2"
    task_data = TaskLoader(str(task_root)).load()["task_data"]
    sandbox = SandboxHandle(
        id="ale-qemu-life-sciences-rgi-mcr1-colistin-v2-fixture",
        endpoint="http://127.0.0.1:55000",
        os=os_type,
        work_dir_base="/home/user/.ale" if os_type == "linux" else r"C:\Users\User\.ale",
        task_data_root="/media/user/data/agenthle",
        node="/usr/bin/node",
        python="/usr/bin/python3",
        mcp_server_dir="/opt/cua",
        metadata={
            "container_name": "ale-qemu-life-sciences-rgi-mcr1-colistin-v2-fixture",
            "slot_root": "/host/slots/rgi_mcr1_colistin_v2-fixture",
            "overlay_qcow2": "/host/slots/rgi_mcr1_colistin_v2-fixture/data.qcow2",
            "image_provenance": {"revision": "image-fixture"},
        },
    )
    original = asdict(sandbox)
    config = SimpleNamespace(model="fixture")
    host_artifacts = tmp_path / "rgi_mcr1_colistin_v2" / "origin_log"
    kwargs = dict(
        executor_type="sandbox", env=SimpleNamespace(sandbox=sandbox),
        config=config, agent_name=agent_name,
        run_id="codex__fixture__life_sciences__rgi_mcr1_colistin_v2__v0__fixture",
        host_artifacts_dir=host_artifacts,
    )
    executor = _build_executor(**kwargs, sandbox_task_name=task_data.sandbox_task_name)
    payload = json.dumps({
        "sandbox_kwargs": _sandbox_to_kwargs(executor.sandbox),
        "work_dir": executor.work_dir,
    })
    assert "mcr1" not in payload.casefold()
    assert "colistin" not in payload.casefold()
    assert executor.sandbox.endpoint == sandbox.endpoint
    assert executor.sandbox.task_data_root == sandbox.task_data_root
    assert executor.sandbox.python == sandbox.python
    assert executor.sandbox.cua_server_port == sandbox.cua_server_port
    assert executor.config is config
    assert asdict(sandbox) == original
    assert _build_executor(
        **kwargs, sandbox_task_name=task_data.sandbox_task_name
    ).work_dir != executor.work_dir
    ordinary = _build_executor(**kwargs)
    assert ordinary.sandbox is sandbox
    assert kwargs["run_id"] in ordinary.work_dir
