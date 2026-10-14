from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

TASK_PATH = Path("computing_math/ising_post_measurement_1/n10_critical_u01_correlators")
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "task-data-hf/extracted"
RAW = Path(os.environ.get("ALE_COMPUTING_RAW_DATA", DEFAULT_DATA)) / TASK_PATH
REPAIRED = Path(os.environ.get("ALE_COMPUTING_REPAIR_DATA", DEFAULT_DATA)) / TASK_PATH
pytestmark = pytest.mark.skipif(
    not (RAW / "reference/critical_state.npy").is_file()
    or not (REPAIRED / "input/task_specification.md").is_file(),
    reason="Requires the Ising reference and repaired public specification",
)


def test_clarification_preserves_all_physics_equations_and_output_shapes():
    original = (RAW / "input/task_specification.md").read_text()
    repaired = (REPAIRED / "input/task_specification.md").read_text()
    assert re.findall(r"\$\$(.*?)\$\$", repaired, re.DOTALL) == re.findall(
        r"\$\$(.*?)\$\$", original, re.DOTALL
    )
    original_outputs = original.split("## 5. Output Files (Agent Produces)")[1].split("## 6.")[0]
    repaired_outputs = repaired.split("## 5. Output Files (Agent Produces)")[1].split("## 6.")[0]
    assert repaired_outputs == original_outputs
    assert "effective approximation" in repaired
    assert "They are not the exact result" in repaired
    assert "produces a post-measurement state of the critical chain:" not in repaired


def test_effective_conditional_state_matches_reference_and_differs_from_literal_unitary():
    state = np.load(RAW / "reference/critical_state.npy", allow_pickle=False)
    reference_rdm = np.load(RAW / "reference/rdm_site1.npy", allow_pickle=False)
    qubits = 10
    outcome = 37
    coupling = 0.1
    masks = 1 << np.arange(qubits - 1, -1, -1)
    indices = np.arange(1 << qubits)
    spins = 1 - 2 * ((indices[:, None] & masks) != 0).astype(int)
    single = state[outcome ^ masks] / state[outcome]
    paired = state[outcome ^ masks[:, None] ^ masks[None, :]] / state[outcome]
    covariance = paired - np.outer(single, single)
    np.fill_diagonal(covariance, 0)
    field = 2 * (1 - single**2 + covariance.sum(axis=1))
    phase = coupling * spins @ single
    damping = coupling**2 * (spins @ field + np.einsum("bi,ij,bj->b", spins, covariance, spins))
    effective = state * np.exp(1j * phase - damping / 2)
    effective /= np.linalg.norm(effective)
    reshaped = effective.reshape(2, -1)
    effective_rdm = reshaped @ reshaped.conj().T
    np.testing.assert_allclose(effective_rdm, reference_rdm[outcome], atol=1e-10)
    exact = np.empty_like(effective)
    for index in indices:
        active = masks[(index & masks) == 0]
        subsets = np.array([0], dtype=int)
        coefficients = np.array([1 + 0j])
        for mask in active:
            subsets = np.concatenate((subsets, subsets ^ mask))
            coefficients = np.concatenate(
                (coefficients * np.cos(2 * coupling), coefficients * 1j * np.sin(2 * coupling))
            )
        exact[index] = state[index] * np.dot(coefficients, state[outcome ^ subsets])
    exact /= np.linalg.norm(exact)
    reshaped = exact.reshape(2, -1)
    exact_rdm = reshaped @ reshaped.conj().T
    assert np.max(np.abs(exact_rdm - effective_rdm)) > 0.05
