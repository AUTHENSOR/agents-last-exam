"""``task_data_source: mounted:<sandbox_dir>`` — task data already visible
*inside* the sandbox at ``<sandbox_dir>/<domain>/<task>/<variant>/`` (e.g. a
read-only HF bucket mounted into the sandbox job by the ``hfsandbox`` provider,
or any NFS/FUSE share the sandbox can see). Copies with plain ``cp`` via the
sandbox's cua-server, so it works on every provider — nothing runs on the host.

Layout expected under ``<sandbox_dir>`` is the one produced by extracting the
published ``ale-tasks-data.tar.gz``::

    <domain>/<task>/<variant>/{input,software,reference}

Phase 1 copies input/software and rejects mounts containing plaintext
reference directories or symlinks. A persistent mount must contain encrypted
reference.7z files, with the password held only by the orchestrator until
evaluation. Read-only mounts do not hide plaintext answers from the solver.
Plaintext reference copying is supported only for a mount attached after the
solver exits. Linux-only.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import baked_in_sandbox, join, prepare_task_subdir, shell_q, task_subdir

logger = logging.getLogger(__name__)

_PREFIX = "mounted:"


def _mounted_task_dir(source: str, task_data: TaskDataSpec) -> str:
    if not source.startswith(_PREFIX):
        raise ValueError("expected mounted:<absolute sandbox directory>")
    root = source[len(_PREFIX) :].rstrip("/")
    if not root.startswith("/") or root.startswith("//") or posixpath.normpath(root) != root:
        raise ValueError("mounted root must be a normalized absolute directory other than /")
    parts = (task_data.domain_name, task_data.task_name, task_data.variant_name)
    if any(not part or part in (".", "..") or "/" in part or "\\" in part for part in parts):
        raise ValueError("mounted task identity must contain three plain directory names")
    return "/".join((root, *parts))


async def _validate_mount(sandbox, task_data, source, *, before_solver):
    src = _mounted_task_dir(source, task_data)
    root = source[len(_PREFIX) :].rstrip("/")
    base = task_subdir(sandbox, task_data)
    await _check_mount(sandbox, root, base=base, before_solver=before_solver)
    return src


async def _check_mount(sandbox, root, *, base=None, before_solver=True):
    script = """
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
base = Path(sys.argv[2]) if sys.argv[2] else None
paths = [root]
if base is not None:
    paths += [base, *(base / name for name in ('input', 'software', 'reference', 'reference.7z', '.reference_extract'))]
for path in paths:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError('Mounted staging paths must not traverse symlinks')
if not root.is_dir():
    raise RuntimeError('Mounted task-data root does not exist')
if base is not None and (root == base or root in base.parents or base in root.parents):
    raise RuntimeError('Mounted source and task destination must be disjoint')
if sys.argv[3] == 'before':
    def fail(error):
        raise error
    for directory, directories, files in os.walk(root, onerror=fail):
        if 'reference' in directories or 'reference' in files:
            raise RuntimeError('Plaintext reference is solver-visible on the mount; use encrypted reference.7z')
        if any((Path(directory) / name).is_symlink() for name in directories + files):
            raise RuntimeError('Mounted task data must not contain symlinks')
"""
    result = await sandbox.run_command(
        shlex.join(
            [sandbox.python, "-c", script, root, base or "", "before" if before_solver else "after"]
        ),
        timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(f"mounted task-data validation failed: {(result.stderr or '')[:1000]}")


async def _copy_tree(sandbox: SandboxHandle, src: str, dst: str, *, timeout: float) -> None:
    """Copy contents into a user-writable task workspace, preserving executable bits."""
    await sandbox.mkdir(dst)
    r = await sandbox.run_command(
        f"cp -a --no-preserve=ownership -- {shell_q(sandbox, src)}/. {shell_q(sandbox, dst)}/"
        f" && chmod -R u+rwX -- {shell_q(sandbox, dst)}",
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"task_data_source=mounted: cp {src} -> {dst} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '')[:300]}"
        )


async def stage_input(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Copy input/ (+ software/) from the mount into the task dir; make output/."""
    if not sandbox.is_linux:
        raise NotImplementedError("task_data_source=mounted is Linux-only")
    src = await _validate_mount(sandbox, task_data, source, before_solver=True)
    if not await sandbox.exists(f"{src}/input"):
        raise RuntimeError(
            f"task_data_source=mounted: expected input/ at {src!r} inside the sandbox, "
            f"not found. Is the task-data volume mounted at {source[len(_PREFIX) :]!r}?"
        )
    base = await prepare_task_subdir(sandbox, task_data)
    await sandbox.rm(
        [
            join(sandbox, base, name)
            for name in (
                "input",
                "software",
                "reference",
                "reference.7z",
                ".reference_extract",
            )
        ]
    )
    await _copy_tree(sandbox, f"{src}/input", join(sandbox, base, "input"), timeout=1800)
    await sandbox.mkdir(join(sandbox, base, "output"))
    staged = ["input"]
    if await sandbox.exists(f"{src}/software"):
        sw_dst = join(sandbox, base, "software")
        await _copy_tree(sandbox, f"{src}/software", sw_dst, timeout=1800)
        # mirror baked_in_sandbox / local: software wrappers must be executable.
        result = await sandbox.run_command(
            f"find {shell_q(sandbox, sw_dst)} -type f -exec chmod +x {{}} +",
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError("mounted: could not make staged software executable")
        staged.append("software")
    logger.info("mounted: staged %s for %s from %s", staged, task_data.task_name, src)
    return {"staged": staged, "source": "mounted"}


async def stage_reference(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Copy reference/ — AFTER the agent, right before evaluate."""
    if not sandbox.is_linux:
        raise NotImplementedError("task_data_source=mounted is Linux-only")
    src = await _validate_mount(sandbox, task_data, source, before_solver=False)
    base = task_subdir(sandbox, task_data)
    target = join(sandbox, base, "reference")
    await sandbox.rm([target])
    archive = f"{src}/reference.7z"
    if await sandbox.exists(archive):
        local_archive = join(sandbox, base, "reference.7z")
        await sandbox.mkdir(base)
        result = await sandbox.run_command(
            f"cp --no-preserve=ownership -- {shell_q(sandbox, archive)} {shell_q(sandbox, local_archive)}",
            timeout=1800,
        )
        if result.returncode:
            raise RuntimeError("mounted: could not copy encrypted reference archive")
        return await baked_in_sandbox.stage_reference(sandbox, task_data, source=source)
    if not await sandbox.exists(f"{src}/reference"):
        return {"skipped": True, "reason": "no_reference"}
    await _copy_tree(sandbox, f"{src}/reference", target, timeout=1800)
    logger.info("mounted: staged reference %s -> %s", src, target)
    return {"staged": ["reference"], "source": "mounted"}
