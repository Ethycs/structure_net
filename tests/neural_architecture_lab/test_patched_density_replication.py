import torch

from neural_architecture_lab.experiments.patched_density_replication import (
    PatchedDensityClassifier,
    ReplicationConfig,
)


def test_replication_config_rejects_noncomparable_architecture():
    try:
        ReplicationConfig(architecture=(784, 128, 64, 10))
    except ValueError as error:
        assert "input, hidden, output" in str(error)
    else:
        raise AssertionError("expected architecture validation")


def test_dense_patch_changes_logits_and_receives_gradients():
    torch.manual_seed(7)
    model = PatchedDensityClassifier((4, 3, 2), connection_density=0.5)
    inputs = torch.randn(5, 4)
    before = model(inputs).detach().clone()

    assert model.add_high_extrema_patches([1]) == 1
    after = model(inputs)
    assert not torch.equal(before, after.detach())

    after.sum().backward()
    assert all(parameter.grad is not None for parameter in model.patches.parameters())


def test_parameter_counts_distinguish_effective_connections_from_storage():
    torch.manual_seed(11)
    model = PatchedDensityClassifier((20, 8, 3), connection_density=0.1)
    model.add_high_extrema_patches([0, 2])

    counts = model.parameter_counts()

    assert counts["patch_parameters"] > 0
    assert counts["effective_parameters"] < counts["stored_trainable_parameters"]
    assert counts["dense_scaffold_parameters"] < counts["stored_trainable_parameters"]
