from pathlib import Path

from neural_architecture_lab.task_geometry_atlas_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_task_geometry_atlas_experiment_results,
    build_task_geometry_atlas_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_task_geometry_atlas/20260805_d6_d8_seed7/results.json"
)


def test_atlas_record_preserves_proxy_and_single_seed_boundaries():
    record = build_task_geometry_atlas_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["independent_seed_count"] == 1
    assert record["hypothesis"]["subclaims"]["chain_level_induced_map_rank"] == (
        "not_tested"
    )
    assert metrics["d6_phase_carrier_stage_index"] == 3
    assert metrics["d6_cosine_quotient_stage_index"] == 9
    assert metrics["d8_phase_carrier_stage_index"] == 1
    assert metrics["d8_cosine_quotient_stage_index"] == 5


def test_all_atlas_arms_convert_to_nal_results():
    record = build_task_geometry_atlas_meta_hypothesis(RESULTS)
    results = build_task_geometry_atlas_experiment_results(record, RESULTS)

    assert len(results) == 4
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {29_956_224, 50_964_992}
    assert all(result.model_checkpoint for result in results)
