"""
Tests for the core functionality of the structure_net library.
"""

import torch
import pytest
from src.structure_net.core.layers import StandardSparseLayer
from src.structure_net.core.network_factory import create_standard_network

def test_standard_sparse_layer_creation():
    """
    Tests the creation of the StandardSparseLayer.
    """
    layer = StandardSparseLayer(in_features=128, out_features=64, sparsity=0.1)
    assert layer is not None, "StandardSparseLayer should be created."
    assert isinstance(layer.linear, torch.nn.Linear), "Layer should contain a Linear module."
    assert hasattr(layer, 'mask'), "Layer should have a sparsity mask."

def test_standard_sparse_layer_forward_pass():
    """
    Tests a forward pass for the StandardSparseLayer.
    """
    layer = StandardSparseLayer(in_features=128, out_features=64, sparsity=0.1)
    input_tensor = torch.randn(32, 128) # Batch of 32
    output = layer(input_tensor)
    assert output is not None, "Layer should produce an output."
    assert output.shape == (32, 64), "Output shape is incorrect."

def test_create_standard_network():
    """
    Tests the creation of a standard network using the factory.
    """
    model = create_standard_network(architecture=[784, 256, 128, 10], sparsity=0.05)
    assert model is not None, "Standard network should be created."
    assert len(model) > 0, "Standard network should have layers."
    # Check if it's a sequence of layers
    assert isinstance(model, torch.nn.Sequential), "Factory should create a Sequential model."

def test_standard_network_forward_pass():
    """
    Tests a forward pass for a network created by the factory.
    """
    model = create_standard_network(architecture=[784, 256, 10], sparsity=0.1)
    input_tensor = torch.randn(16, 784) # Batch of 16
    output = model(input_tensor)
    assert output is not None, "Network should produce an output."
    assert output.shape == (16, 10), "Network output shape is incorrect."