from dataclasses import replace

import pytest

from neural_architecture_lab.llama_lossless_compression_hypothesis import LosslessCompressionEvidence


def example():
    return LosslessCompressionEvidence("campaign.json", "a" * 64, "report.md", "b" * 64,
                                       "lossless_storage_reduction", 199, True, 1000, 700, 1600)


def test_exploratory_storage_does_not_claim_accuracy_or_speed():
    record = example().record()
    assert record["tested"] and not record["confirmed"]
    assert not record["application_accuracy_tested"] and not record["inference_speed_tested"]
    assert record["direct_experiment_count"] == 199


def test_exactness_and_complete_census_required():
    with pytest.raises(ValueError, match="classification"):
        replace(example(), all_tensors_bitwise=False)
    with pytest.raises(ValueError, match="classification"):
        replace(example(), original_archive_bytes=1000)
    with pytest.raises(ValueError, match="complete"):
        replace(example(), archive_count=198)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(example(), campaign_sha256="f" * 63)
    assert replace(example(), classification="lossless_no_storage_reduction", original_archive_bytes=1000)
