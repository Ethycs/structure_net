import pytest
from neural_architecture_lab.llama_lossless_residual_hypothesis import LosslessResidualEvidence


def test_exactness_claim_requires_every_matrix_and_final_test():
    def make(**changes):
        values=dict(campaign="result.json",campaign_sha256="a"*64,report="report.md",report_sha256="b"*64,
                    classification="lossless_preserved_no_efficiency_gain",matrices=197,final_examples=10,
                    all_weights_bitwise=True,all_outputs_bitwise=True)
        return LosslessResidualEvidence(**(values|changes))
    record=make().record()
    assert record["tested"] and not record["confirmed"]
    assert not record["application_accuracy_tested"]
    for changes in ({"matrices":196},{"all_weights_bitwise":False},{"all_outputs_bitwise":False},{"final_examples":0}):
        with pytest.raises(ValueError):
            make(**changes)
