from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import huggingface_hub
import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.images import get as get_image
from ale_run.environments.providers import hfsandbox as hf


SPEC = SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo/hello")


class Api:
    def __init__(self):
        self.created = []
        self.cancelled = []
        self.stage = "RUNNING"
        self.inspect_error = None
        self.cancel_error = None
        self.creating = threading.Event()
        self.finish_creation = threading.Event()
        self.finish_creation.set()
        self.hardware = SimpleNamespace(
            name="cpu-xl", cpu="16 vCPU", ram="124 GB", ephemeral_storage="1000 GB"
        )

    def list_jobs_hardware(self):
        return [self.hardware]

    def run_job(self, *, image, command, expose, volumes, **kwargs):
        self.creating.set()
        assert self.finish_creation.wait(5)
        self.created.append(
            dict(image=image, command=command, expose=expose, volumes=volumes, **kwargs)
        )
        return SimpleNamespace(id="review-job", owner=SimpleNamespace(name="review-owner"))

    def inspect_job(self, **kwargs):
        if self.inspect_error is not None:
            raise self.inspect_error
        return SimpleNamespace(
            status=SimpleNamespace(
                stage=self.stage,
                message="Pulling container image",
                expose_urls=["https://review--5000.hf.jobs"],
            )
        )

    def cancel_job(self, *, job_id, namespace):
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled.append((job_id, namespace))


@pytest.fixture
def api(monkeypatch):
    client = Api()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: client)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "review-placeholder")
    return client


async def create(provider, spec=SPEC):
    image = get_image(provider.config.image)
    return await provider._create_via_job(image.docker_image, image, {}, spec)


@pytest.mark.parametrize("stage", ["ERROR", "COMPLETED", "CANCELED", "DELETED"])
async def test_terminal_startup_cancels_owned_job(api, stage):
    api.stage = stage
    with pytest.raises(RuntimeError):
        await create(hf.HfSandboxProvider({}))
    assert api.cancelled == [("review-job", "review-owner")]


async def test_inspection_failure_cancels_owned_job(api):
    api.inspect_error = OSError("review failure")
    with pytest.raises(OSError):
        await create(hf.HfSandboxProvider({}))
    assert api.cancelled == [("review-job", "review-owner")]


async def test_startup_timeout_preserves_stage_reason_and_caps(api):
    api.stage = "SCHEDULING"
    provider = hf.HfSandboxProvider({"start_timeout": 0.03, "job_timeout": "20m"})
    started = asyncio.get_running_loop().time()
    with pytest.raises(RuntimeError, match="startup timed out after 0.03s") as error:
        await create(provider)
    assert "job hard cap 20m" in str(error.value)
    assert "stage=SCHEDULING" in str(error.value)
    assert "Pulling container image" in str(error.value)
    assert asyncio.get_running_loop().time() - started < 0.3
    assert api.cancelled == [("review-job", "review-owner")]


async def test_startup_cancellation_cancels_owned_job(api):
    api.inspect_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await create(hf.HfSandboxProvider({}))
    assert api.cancelled == [("review-job", "review-owner")]


async def test_cancellation_during_blocking_creation_reclaims_late_job(api):
    api.finish_creation.clear()
    task = asyncio.create_task(create(hf.HfSandboxProvider({})))
    try:
        assert await asyncio.to_thread(api.creating.wait, 3)
        task.cancel()
        api.finish_creation.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        assert api.cancelled == [("review-job", "review-owner")]
    finally:
        api.finish_creation.set()


@pytest.mark.parametrize(
    "spec",
    [
        dataclasses.replace(SPEC, os="windows"),
        dataclasses.replace(SPEC, gpu="nvidia-t4"),
        dataclasses.replace(SPEC, snapshot="gpu-free-ubuntu"),
    ],
)
async def test_unsupported_workloads_fail_before_allocating(api, spec):
    with pytest.raises(ValueError):
        await hf.HfSandboxProvider({}).acquire(spec)
    assert not api.created


@pytest.mark.parametrize(
    "spec",
    [
        dataclasses.replace(SPEC, machine_type="c4-standard-32"),
        dataclasses.replace(SPEC, vcpus=32),
        dataclasses.replace(SPEC, memory_gb=256),
    ],
)
async def test_undersized_flavor_fails_before_allocating(api, spec):
    with pytest.raises(ValueError, match="too small"):
        await create(hf.HfSandboxProvider({}), spec)
    assert not api.created


@pytest.mark.parametrize("storage", ["50 GB", "199 GB", "unknown", ""])
async def test_insufficient_or_unverifiable_storage_fails_before_allocating(api, storage):
    api.hardware.ephemeral_storage = storage
    with pytest.raises(ValueError, match="storage"):
        await create(hf.HfSandboxProvider({}))
    assert not api.created


@pytest.mark.parametrize("storage", ["200 GB", "1 TB"])
async def test_sufficient_storage_allocates(api, storage):
    api.hardware.ephemeral_storage = storage
    await create(hf.HfSandboxProvider({}))
    assert len(api.created) == 1


async def test_verified_small_image_storage_override(api):
    api.hardware.ephemeral_storage = "50 GB"
    await create(hf.HfSandboxProvider({"min_storage_gb": 40}))
    assert len(api.created) == 1


@pytest.mark.parametrize(
    "raw",
    [
        {"start_timeout": 0},
        {"start_timeout": -1},
        {"min_storage_gb": 0},
        {"min_storage_gb": True},
        {"min_storage_gb": float("nan")},
        {"min_storage_gb": float("inf")},
        {"cua_ready_timeout": float("nan")},
        {"resolution": [0, 768]},
        {"resolution": "800x600"},
        {"enable_dind": "false"},
        {"enable_dind": True},
        {"volumes": "hf://buckets/ns/data:/mnt/data:ro"},
    ],
)
def test_invalid_config_fails_closed(raw):
    with pytest.raises(ValueError):
        hf.HfSandboxProvider(raw)


@pytest.mark.parametrize(
    "volume",
    [
        "hf://buckets/ns/data:/mnt/data:ro:ignored",
        "hf://buckets/ns/data:/mnt/../data",
        "hf://buckets//data:/mnt/data",
        "hf://buckets/ns/data/../other:/mnt/data",
        "hf://buckets/ns/data:/:ro",
        "hf://datasets/ns/data:/mnt/data:rw",
    ],
)
def test_invalid_volume_rejected(volume):
    with pytest.raises(ValueError):
        hf.parse_volume_spec(volume)


async def test_cua_timeout_is_a_bound_not_just_poll_condition():
    provider = hf.HfSandboxProvider({"cua_ready_timeout": 0.03})
    handle = SimpleNamespace(
        id="review-job", check_reachable=AsyncMock(side_effect=lambda **kwargs: asyncio.sleep(1))
    )

    async def slow(**kwargs):
        await asyncio.sleep(1)

    handle.check_reachable = slow
    started = asyncio.get_running_loop().time()
    with pytest.raises(RuntimeError):
        await provider._wait_cua_ready(handle)
    assert asyncio.get_running_loop().time() - started < 0.3


async def test_acquire_cancellation_stops_proxy_and_remote(api, monkeypatch):
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), url="http://127.0.0.1:1", port=1)
    monkeypatch.setattr(hf, "_AuthProxy", lambda *args: proxy)
    provider = hf.HfSandboxProvider({})
    monkeypatch.setattr(provider, "_wait_cua_ready", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await provider.acquire(SPEC)
    proxy.stop.assert_awaited_once()
    assert api.cancelled == [("review-job", "review-owner")]


async def test_release_failure_remains_retryable(api):
    provider = hf.HfSandboxProvider({})
    remote_id, _, _, kill, _, _ = await create(provider)
    proxy = SimpleNamespace(stop=AsyncMock())
    provider._live[remote_id] = hf._Live(proxy=proxy, kill=kill)
    handle = SimpleNamespace(id=remote_id, metadata={})
    api.cancel_error = OSError("review failure")
    with pytest.raises(OSError):
        await provider.release(handle)
    assert remote_id in provider._live
    api.cancel_error = None
    await provider.release(handle)
    assert not provider._live
    assert api.cancelled == [(remote_id, "review-owner")]


async def test_proxy_stop_failure_still_cancels_remote(api):
    provider = hf.HfSandboxProvider({})
    remote_id, _, _, kill, _, _ = await create(provider)
    provider._live[remote_id] = hf._Live(
        proxy=SimpleNamespace(stop=AsyncMock(side_effect=OSError)), kill=kill
    )
    with pytest.raises(OSError):
        await provider.release(SimpleNamespace(id=remote_id, metadata={}))
    assert api.cancelled == [(remote_id, "review-owner")]


async def test_keep_then_delete_preserves_original_cleanup(api):
    provider = hf.HfSandboxProvider({})
    remote_id, _, _, kill, _, _ = await create(provider)
    provider._live[remote_id] = hf._Live(proxy=SimpleNamespace(stop=AsyncMock()), kill=kill)
    handle = SimpleNamespace(id=remote_id, metadata={})
    await provider.release(handle, mode="keep")
    assert not api.cancelled
    assert remote_id in provider._live
    await provider.release(handle)
    assert api.cancelled == [(remote_id, "review-owner")]


async def test_invalid_release_mode_does_not_destroy(api):
    provider = hf.HfSandboxProvider({})
    with pytest.raises(ValueError):
        await provider.release(SimpleNamespace(id="review-job", metadata={}), mode="typo")
    assert not api.cancelled


async def test_all_volume_mounts_checked_even_without_task_data(api, monkeypatch):
    from ale_run.environments.task_data import mounted

    check = AsyncMock(side_effect=RuntimeError("Plaintext reference"))
    monkeypatch.setattr(mounted, "_check_mount", check, raising=False)
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), url="http://127.0.0.1:1", port=1)
    monkeypatch.setattr(hf, "_AuthProxy", lambda *args: proxy)
    provider = hf.HfSandboxProvider({"volumes": ["hf://buckets/ns/data:/mnt/data:ro"]})
    monkeypatch.setattr(provider, "_wait_cua_ready", AsyncMock())
    with pytest.raises(RuntimeError, match="Plaintext reference"):
        await provider.acquire(SPEC)
    check.assert_awaited_once()
    assert api.cancelled == [("review-job", "review-owner")]


async def test_acquired_handle_records_pinned_image_identity(api, monkeypatch):
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), url="http://127.0.0.1:1", port=1)
    monkeypatch.setattr(hf, "_AuthProxy", lambda *args: proxy)
    provider = hf.HfSandboxProvider({"image": "ale-ubuntu22-docker-v1-1"})
    monkeypatch.setattr(provider, "_wait_cua_ready", AsyncMock())
    handle = await provider.acquire(SPEC)
    try:
        recorded = handle.metadata["image_provenance"]
        assert recorded["image_ref"] == api.created[0]["image"]
        assert recorded["image_revision"] == recorded["image_ref"].rsplit("@", 1)[1]
        assert recorded["provider"] == "hfsandbox"
    finally:
        await provider.release(handle)


async def test_sandbox_proxy_setup_failure_kills_and_closes(monkeypatch):
    calls = []
    sandbox = SimpleNamespace(
        id="review-sandbox",
        run=lambda *args, **kwargs: None,
        proxy_url_for=lambda *args: (_ for _ in ()).throw(OSError("review failure")),
        kill=lambda: calls.append("kill"),
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(
        huggingface_hub, "Sandbox", SimpleNamespace(create=lambda **kwargs: sandbox), raising=False
    )
    provider = hf.HfSandboxProvider({"transport": "sandbox"})
    with pytest.raises(OSError):
        await provider._create_via_sandbox("review-image", get_image(provider.config.image), {})
    assert calls == ["kill", "close"]


def test_metadata_cards_are_best_effort(tmp_path, monkeypatch, capsys):
    from ale_run.cli import main

    task = tmp_path / "tasks/demo/review"
    task.mkdir(parents=True)
    (task / "main.py").write_text("raise RuntimeError('must never import')")
    monkeypatch.chdir(tmp_path)
    for card in (
        {"vm": "linux"},
        {"vm": ["linux"]},
        {"vm": {"snapshot": 1}},
        {"snapshot": "cpu-free-ubuntu"},
        {"vm_category": "cpu-free-ubuntu"},
    ):
        (task / "task_card.json").write_text(json.dumps(card))
        assert main(["list", "-v"]) == 0
        output = capsys.readouterr().out
        assert "demo/review" in output
        if "snapshot" in card or "vm_category" in card:
            assert "cpu-free-ubuntu" in output
    (task / "task_card.json").write_bytes(b"\xff")
    assert main(["list", "-v"]) == 0


def test_discovery_does_not_follow_directory_symlinks(tmp_path):
    from ale_run.tasks import list_tasks

    root = tmp_path / "tasks"
    task = root / "domain/review"
    task.mkdir(parents=True)
    (task / "main.py").write_text("raise RuntimeError('must never import')")
    (task / "cycle").symlink_to(root, target_is_directory=True)
    (root / "alias").symlink_to(task, target_is_directory=True)
    assert list_tasks(root) == ["domain/review"]
