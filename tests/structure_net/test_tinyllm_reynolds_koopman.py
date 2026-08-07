import numpy as np

from experiments.structure_net.tinyllm_reynolds_koopman import (
    ReynoldsDictionary,
    _front,
    orbit_tensor,
    reynolds_parts,
    scale_cover,
)


def test_reynolds_barycenter_and_invariant_dictionary_ignore_cyclic_deck_shift():
    generator = np.random.default_rng(7)
    orbit = generator.normal(size=(32, 3, 12))
    values = orbit.reshape(96, 12).astype(np.float32)
    shifted = np.roll(orbit, -1, axis=1).reshape(96, 12).astype(np.float32)
    dictionary = ReynoldsDictionary(3, barycenter_rank=8, sketch_rank=5, seed=11).fit(values, 32)
    original = dictionary.transform(values, 32)
    transformed = dictionary.transform(shifted, 32)
    for name in ("B", "B+Q", "B+Q+C"):
        np.testing.assert_allclose(original[name], transformed[name], atol=1e-9, rtol=1e-9)


def test_cover_scaling_preserves_barycenter_and_scales_variation():
    generator = np.random.default_rng(13)
    values = generator.normal(size=(40, 7)).astype(np.float32)
    original = orbit_tensor(values, 20, 2)
    scaled = orbit_tensor(scale_cover(values, 20, 2, 0.25), 20, 2)
    np.testing.assert_allclose(original.mean(1), scaled.mean(1), atol=2e-7)
    np.testing.assert_allclose(
        scaled - scaled.mean(1, keepdims=True),
        0.25 * (original - original.mean(1, keepdims=True)),
        atol=2e-7,
    )
    barycenter, characters = reynolds_parts(scale_cover(values, 20, 2, 0.0), 20, 2)
    assert barycenter.shape == (20, 7)
    np.testing.assert_allclose(characters[0], 0.0, atol=2e-7)


def test_task_front_uses_joint_numeric_gate_without_reinterpreting_failures():
    cuts = ("early", "middle", "late")

    def item(r2, gain, accuracy_loss=0.0):
        return {
            "task_closure": {
                "B": {
                    "composition": {
                        "moment_r2": r2,
                        "map_circular_alignment": 0.99,
                        "map_sampling_resolved": True,
                        "map_winding_degree": 2.0,
                        "accuracy_loss": accuracy_loss,
                    }
                },
                "B+Q+C": {"composition": {"positive_cover_gain": gain}},
            }
        }

    values = {
        "early": item(0.89, 0.0),
        "middle": item(0.95, 0.03),
        "late": item(0.95, 0.01),
    }
    assert _front(values, "composition", 2, cuts) == "late"
