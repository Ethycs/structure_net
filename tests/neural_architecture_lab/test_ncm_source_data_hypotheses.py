from dataclasses import replace

import pytest

from neural_architecture_lab.ncm_source_data_hypotheses import CONTROLS, SourceDataEvidence


def test_complete_data_scope_never_becomes_student_quality_or_model_speed():
    for hyp, (classification, cases, measurements, _, _) in CONTROLS.items():
        evidence = SourceDataEvidence(hyp, classification, cases, measurements, 'campaign', 'a' * 64, 'report', 'b' * 64)
        row = evidence.record()
        assert row['tested'] and not row['confirmed'] and row['data_preparation_only']
        assert not row['student_task_accuracy_measured'] and not row['model_speed_measured']
        with pytest.raises(ValueError):
            replace(evidence, classification='95_percent_student_accuracy').record()
        with pytest.raises(ValueError):
            replace(evidence, measurements=measurements - 1).record()
        with pytest.raises(ValueError):
            replace(evidence, campaign_sha256='missing').record()
