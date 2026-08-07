import numpy as np
import torch

from experiments.structure_net.tinyllm_normal_kernel_radius import (
    NormalKernelConfig,
    parameter_jacobian,
)


def test_parameter_jacobian_has_two_rows():
    parameter = torch.tensor([0.2, -0.3], requires_grad=True)
    moment = torch.stack((parameter[0] + 2 * parameter[1], parameter[0] * parameter[1]))
    jacobian = parameter_jacobian(moment, parameter)
    assert np.allclose(jacobian.detach().numpy(), [[1.0, 2.0], [-0.3, 0.2]])


def test_normal_kernel_preregistered_control_count():
    assert NormalKernelConfig().random_directions == 32
