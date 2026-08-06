"""GPU visibility is controlled explicitly by the parent launcher."""

import os

from structure_net.config.environment import setup_cuda_devices


def test_default_setup_preserves_unset_visibility(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert setup_cuda_devices() is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_explicit_setup_does_not_replace_parent_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    assert setup_cuda_devices("0,1") == "2"
