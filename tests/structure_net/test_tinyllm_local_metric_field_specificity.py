from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from experiments.structure_net import tinyllm_local_metric_field_specificity as audit


def _projectors(kernels: np.ndarray) -> np.ndarray:
    kernels = kernels / np.linalg.norm(kernels, axis=1, keepdims=True)
    identity = np.broadcast_to(np.eye(3), (len(kernels), 3, 3)).copy()
    return identity - kernels[:, :, None] * kernels[:, None, :]


def test_primary_configuration_and_corrective_shift_are_locked() -> None:
    config = audit.LocalMetricFieldSpecificityConfig()
    assert config.seeds == (7, 29, 53)
    assert config.phase_shift_bins == 5
    assert config.phase_count * config.nuisance_replicates == 64
    with pytest.raises(ValueError, match="five bins"):
        replace(config, phase_shift_bins=4)
    with pytest.raises(ValueError, match="seeds are fixed"):
        replace(config, seeds=(7,))


def test_control_indices_preserve_the_declared_coordinate() -> None:
    nuisance = audit.nuisance_equivalence_indices(3, 4)
    phase = audit.semantic_phase_shift_indices(16, 4, 5)
    phase_labels = np.repeat(np.arange(3), 4)
    replicate_labels = np.tile(np.arange(4), 3)
    assert np.array_equal(phase_labels, phase_labels[nuisance])
    assert not np.array_equal(replicate_labels, replicate_labels[nuisance])
    full_replicates = np.tile(np.arange(4), 16)
    full_phases = np.repeat(np.arange(16), 4)
    assert np.array_equal(full_replicates, full_replicates[phase])
    assert np.all((full_phases[phase] - full_phases) % 16 == 5)


def test_line_geometry_is_sign_invariant() -> None:
    kernels = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projectors = _projectors(kernels)
    np.testing.assert_allclose(audit.line_cosines(projectors, projectors), 1.0)
    np.testing.assert_allclose(
        audit.projector_distances(projectors, projectors), 0.0, atol=1e-12
    )


def test_corrected_geometry_separates_phase_and_random_controls() -> None:
    phase_count = 4
    replicates = 2
    angles = np.repeat(np.arange(phase_count) * math.pi / 2.0, replicates)
    kernels = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
    reference = _projectors(kernels)
    transformed = reference.copy()
    nuisance = audit.nuisance_equivalence_indices(phase_count, replicates)
    phase = audit.semantic_phase_shift_indices(phase_count, replicates, 1)
    random = _projectors(np.tile(np.array([[0.0, 0.0, 1.0]]), (len(kernels), 1)))
    result = audit.corrected_cell_geometry(
        reference, transformed, nuisance, phase, random
    )
    assert result["paired_median_kernel_cosine"] == pytest.approx(1.0)
    assert result["nuisance_equivalence_median_kernel_cosine"] == pytest.approx(1.0)
    assert result["semantic_phase_control_median_kernel_cosine"] == pytest.approx(0.0)
    assert result["random_control_median_kernel_cosine"] == pytest.approx(0.0)


def test_random_projectors_are_deterministic_rank_two() -> None:
    first = audit.random_rank_two_projectors(12, 123)
    second = audit.random_rank_two_projectors(12, 123)
    different = audit.random_rank_two_projectors(12, 124)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    np.testing.assert_allclose(first @ first, first, atol=1e-12)
    assert np.linalg.matrix_rank(first).tolist() == [2] * 12


@pytest.mark.parametrize(
    ("valid", "base", "nuisance", "specificity", "expected"),
    [
        (False, True, True, True, "invalid"),
        (True, True, True, True, "nuisance_invariant_phase_specific_field"),
        (True, True, True, False, "phase_nonspecific_field"),
        (True, False, True, True, "nuisance_variant_field"),
        (True, True, False, True, "mixed_metric_field_geometry"),
    ],
)
def test_classification_precedence(
    valid: bool, base: bool, nuisance: bool, specificity: bool, expected: str
) -> None:
    assert (
        audit.classify_checkpoint(
            valid=valid,
            base_geometry=base,
            nuisance_equivalence=nuisance,
            specificity=specificity,
        )
        == expected
    )


def test_source_campaign_and_arrays_are_locked() -> None:
    config = audit.LocalMetricFieldSpecificityConfig(
        seeds=(7,), allow_underpowered=True
    )
    campaign, path, details = audit._load_source(config)
    assert audit._sha256(path) == audit.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == audit.SOURCE_IMPLEMENTATION_SHA256
    assert set(details) == {7, 29, 53}
    assert details[7][2].is_file()


def test_fingerprint_changes_with_array_identity() -> None:
    config = audit.LocalMetricFieldSpecificityConfig(
        seeds=(7,), allow_underpowered=True
    )
    first = audit._fingerprint(config, 7, "result", "arrays")
    assert first == audit._fingerprint(config, 7, "result", "arrays")
    assert first != audit._fingerprint(config, 7, "result", "different")

