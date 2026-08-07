import numpy as np

from experiments.structure_net.tinyllm_defect_certification import (
    CertificationConfig,
    numerical_jacobian,
)


class _LinearField:
    def __call__(self, phase, path):
        return np.asarray((2.0 * phase + path, -phase + 3.0 * path))


def test_numerical_jacobian_recovers_orientation():
    jacobian = numerical_jacobian(_LinearField(), 0.2, 0.4, 1e-5)
    assert np.allclose(jacobian, [[2.0, 1.0], [-1.0, 3.0]], atol=1e-8)
    assert np.linalg.det(jacobian) > 0.0


def test_formal_gate_requires_actual_network_enclosure():
    config = CertificationConfig(boundary_samples=16, endpoint_phase_points=64, surrogate_degree=2, surrogate_audit_points=5, device="cpu")
    assert config.phase_interval[0] < config.phase_interval[1]
