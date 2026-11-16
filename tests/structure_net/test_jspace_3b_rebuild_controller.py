import pytest

from experiments.structure_net import jspace_3b_rebuild_controller as controller


def completed(tmp_path, passed):
    (tmp_path/"run").mkdir()
    checkpoint=tmp_path/"run/selected.pt"
    checkpoint.write_bytes(b"saved scientific weights")
    controller.write(tmp_path/"plan.json",{"fingerprint":"planned"})
    controller.write(tmp_path/"run/result.json",{"status":"completed","fingerprint":"planned",
        "relative_retention_pass":passed,"selected":{"checkpoint":{
            "path":str(checkpoint),"sha256":controller.sha(checkpoint)}} if passed else None})
    controller.write(tmp_path/"run/manifest.json",{"status":"completed","fingerprint":"planned",
        "artifacts":{"result.json":controller.sha(tmp_path/"run/result.json")}})
    return checkpoint


def test_failed_retention_stops_before_next_stage(tmp_path):
    completed(tmp_path,False)
    assert controller.selected_parent(tmp_path) is None


def test_only_authenticated_selected_weights_can_advance(tmp_path):
    checkpoint=completed(tmp_path,True)
    assert controller.selected_parent(tmp_path)==str(checkpoint)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError,match="authenticated selected"):
        controller.selected_parent(tmp_path)


def test_modified_quality_receipt_cannot_advance(tmp_path):
    completed(tmp_path,True)
    (tmp_path/"run/result.json").write_text('{}')
    with pytest.raises(ValueError,match="authenticated completed"):
        controller.selected_parent(tmp_path)
