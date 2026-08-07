import math

import numpy as np

from experiments.structure_net.tinyllm_deck_action_descrambler import (
    action_metrics,
    canonical_decomposition,
    deck_shift,
    generate_exact_orbits,
    orbit_average,
    orthogonal_procrustes_action,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_procrustes_recovers_exact_z3_action_and_projector():
    angle = 2.0 * math.pi / 3.0
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    action_true = np.block([[np.ones((1, 1)), np.zeros((1, 2))], [np.zeros((2, 1)), rotation]])
    generator = np.random.default_rng(3)
    source = generator.normal(size=(1024, 3))
    target = source @ action_true
    action = orthogonal_procrustes_action(source, target)
    projector, canonical = canonical_decomposition(action, 3)
    metrics = action_metrics(source, target, action, 3)
    assert np.allclose(action, action_true, atol=1e-10)
    assert metrics["activation_weighted_closure_error"] < 1e-10
    assert canonical["invariant_dimension"] == 1
    assert canonical["nontrivial_dimension"] == 2
    assert np.allclose(projector, np.diag([1.0, 0.0, 0.0]), atol=1e-10)


def test_exact_orbits_share_nuisance_and_have_deck_invariant_targets():
    task = CircleTaskConfig()
    data = generate_exact_orbits(task, k=3, orbit_count=6, seed=11, regime="composition")
    targets = data.target_bins.numpy().reshape(6, 3)
    calibration = data.calibration.numpy().reshape(6, 3, -1)
    assert np.all(targets == targets[:, :1])
    assert np.allclose(calibration, calibration[:, :1, :])
    phase = data.phase.numpy().reshape(6, 3)
    assert np.allclose(np.remainder(3.0 * phase, 2.0 * math.pi), np.remainder(3.0 * phase[:, :1], 2.0 * math.pi), atol=1e-5)


def test_deck_shift_and_orbit_average_use_complete_fibers():
    values = np.arange(18, dtype=np.float64).reshape(6, 3)
    shifted = deck_shift(values, orbit_count=2, k=3).reshape(2, 3, 3)
    original = values.reshape(2, 3, 3)
    assert np.array_equal(shifted[:, 0], original[:, 1])
    averaged = orbit_average(values, orbit_count=2, k=3).reshape(2, 3, 3)
    assert np.allclose(averaged[:, 0], averaged[:, 1])
    assert np.allclose(averaged[:, 0], original.mean(1))
