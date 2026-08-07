from pathlib import Path

from neural_architecture_lab.io_correspondence_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_io_correspondence_experiment_results,
    build_io_correspondence_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_io_correspondence/"
    "20260806_d8_corrected_equivariant/campaign_results.json"
)


def test_meta_record_preserves_partial_observation_only_verdict():
    record = build_io_correspondence_meta_hypothesis(RESULTS)
    arms = record["result"]["descriptive_metrics"]["arms"]
    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "partially_supported_observation_repair_only"
    )
    assert record["result"]["num_direct_experiments"] == 35
    assert arms["calibrated_absolute_analytic"]["joint_pass_count"] == 5
    assert arms["calibrated_absolute_equivariant"]["joint_pass_count"] == 5
    assert arms["uncalibrated_relative_equivariant"]["joint_pass_count"] == 0
    assert record["result"]["implementation_correction"]["outcomes_seen_before_correction"]


def test_thirty_five_cells_convert_to_content_addressed_results():
    record = build_io_correspondence_meta_hypothesis(RESULTS)
    results = build_io_correspondence_experiment_results(record, RESULTS)
    assert len(results) == 35
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_965_504}
    assert all(result.model_checkpoint for result in results)
