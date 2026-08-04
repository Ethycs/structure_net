from pathlib import Path

from neural_architecture_lab.meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_experiment_results,
    build_meta_hypothesis_record,
)


REPLICATION = Path(
    "data/experiments/patched_density_replication/20260804_seed4/results.json"
)
AUDIT = Path("data/experiments/mnist_checkpoint_audit/20260804/results.json")


def test_meta_hypothesis_is_conservative_about_single_seed_and_scope():
    record = build_meta_hypothesis_record(REPLICATION, AUDIT)

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["tested"] is True
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["confidence_assessment"] == "not_estimable_single_seed"
    assert record["result"]["independent_seed_count"] == 1
    assert record["result"]["positive_paired_gain_count"] == 2
    assert "embedded variable-density patches" in record["hypothesis"][
        "explicitly_not_tested"
    ]


def test_meta_hypothesis_preserves_direct_and_provenance_evidence():
    record = build_meta_hypothesis_record(REPLICATION, AUDIT)
    results = build_experiment_results(record, REPLICATION, AUDIT)

    assert len(record["evidence"]["direct_tests"]) == 2
    assert len(record["evidence"]["provenance_audits"]) == 2
    assert len(results) == 4
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.experiment_id for result in results} == {
        "patched-density-mnist-seed-4",
        "patched-density-fashion_mnist-seed-4",
        "mnist-checkpoint-audit-seed-4-4e9efa74",
        "mnist-checkpoint-audit-seed-5-9ccb1517",
    }
    assert results[0].metrics["patched_minus_control"] == 0.14640000000000003
    assert results[-1].metrics["claim_reproduced"] == 0.0
