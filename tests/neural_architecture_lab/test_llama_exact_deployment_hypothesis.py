from dataclasses import replace
import pytest

from neural_architecture_lab.llama_exact_deployment_hypothesis import ExactDeploymentEvidence
from neural_architecture_lab.llama_lossless_compression_hypothesis import PartialCompressionEvidence


def evidence():
    return ExactDeploymentEvidence("c.json", "a" * 64, "r.md", "b" * 64, True, (1.1, 1., 2.), 6425499648)


def test_speedup_scope_remains_conservative():
    record = evidence().record()
    assert record["faster_single_model_supported"]
    assert not record["confirmed"] and not record["two_for_one_supported"]
    assert not record["weight_compression_used"] and not record["inference_arithmetic_changed"]
    assert not replace(evidence(), all_bitwise=False).record()["faster_single_model_supported"]
    assert not replace(evidence(), speedups=(1.09, 1., 1.)).record()["faster_single_model_supported"]
    with pytest.raises(ValueError):
        replace(evidence(), parameter_bytes=1)
    with pytest.raises(ValueError):
        replace(evidence(), speedups=(1., float("nan"), 2.))


def test_partial_archive_cannot_be_registered_as_full_or_deployment_success():
    record = PartialCompressionEvidence("c", "a" * 64, "r", "b" * 64, 1000, 800).record()
    assert record["direct_experiment_count"] == 1
    assert record["requested_archives"] == 199
    assert record["failed_archives"] == 1 and record["unrun_archives"] == 197
    assert not record["full_campaign_completed"] and not record["confirmed"]
    assert not record["inference_speed_tested"]


def test_finalizer_rejects_duplicate_controls_even_with_matching_bits(tmp_path):
    import json
    import torch
    from experiments.neural_architecture_lab.finalize_llama_exact_deployment import validate_workloads
    from experiments.structure_net.prepare_llama_cotenancy import sha256
    control = {"ids": torch.tensor([[1, 2]]), "expected": torch.tensor([3.]),
               "actual": torch.tensor([3.]), "bitwise_equal": True}
    path = tmp_path / "controls.pt"
    torch.save([control, control], path)
    row = {"length": 128, "decode_steps": 0, "control_artifact": str(path), "control_sha256": sha256(path)}
    (tmp_path / "workload_128_0.json").write_text(json.dumps(row))
    campaign = {"status": "completed", "uncontended": True,
                "workloads": [row, {"length": 512, "decode_steps": 0}, {"length": 128, "decode_steps": 32}]}
    with pytest.raises(ValueError, match="two distinct"):
        validate_workloads(campaign, tmp_path)
