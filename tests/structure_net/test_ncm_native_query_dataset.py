import pytest
import torch
from experiments.structure_net.ncm_native_query_dataset import source_inputs,target_tensor


def test_native_windows_preserve_anchor_padding_targets_and_frame_boundaries():
    rows = [{"source_frame":8,"context_ids":[1,2,3],"target_ids":[4,5,9]},
            {"source_frame":8,"context_ids":[1,7],"target_ids":[8,9]}]
    x = source_inputs(rows,9,"cpu")
    assert x["input_ids"].tolist()==[[9,9,9,9,9,1,2,3],[9,9,9,9,9,9,1,7]]
    assert x["attention_mask"].sum(1).tolist()==[3,2]
    target = target_tensor(rows,"cpu")
    assert target.shape==(2,128) and torch.equal(target[0,:3],torch.tensor([4,5,9]))
    assert target[0,3:].eq(-100).all() and target[1,2:].eq(-100).all()
    with pytest.raises(ValueError):source_inputs([rows[0],{**rows[1],"source_frame":4}],9,"cpu")
    with pytest.raises(ValueError):target_tensor([{**rows[0],"target_ids":list(range(129))}],"cpu")
