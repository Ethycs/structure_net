import torch.nn as nn

from experiments.neural_architecture_lab.audit_mnist_checkpoints import build_dense_mlp


def test_build_dense_mlp_matches_archived_sequential_key_layout():
    model = build_dense_mlp([784, 64, 10])

    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.Linear)
    assert list(model.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias"]
