import pytest
from experiments.structure_net import ncm_engineering_repair_continuation as run


@pytest.mark.parametrize('change',[
    {'step':511},
    {'producer_fingerprint':'wrong'},
    {'optimizer':{'param_groups':[{'lr':1e-4}]}},
    {'optimizer':{'param_groups':[]}},
])
def test_resume_rejects_wrong_checkpoint_or_optimizer_contract(change):
    saved={'step':512,'producer_fingerprint':'parent','optimizer':{'param_groups':[{'lr':3e-5}]}}
    run.resume_contract(saved,'parent')
    with pytest.raises(ValueError):run.resume_contract({**saved,**change},'parent')
