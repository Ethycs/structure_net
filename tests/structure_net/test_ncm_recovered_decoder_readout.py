from types import SimpleNamespace
import torch
from torch import nn
import pytest

from experiments.structure_net import ncm_recovered_decoder_readout as study


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.codec=nn.Module();self.codec.decoder=nn.Linear(2,2)
        self.word_memory=nn.Linear(2,2)


def test_strict_decoder_swap_preserves_source_parameters():
    model=Model();old={k:v.clone() for k,v in model.word_memory.state_dict().items()}
    new={k:torch.zeros_like(v) for k,v in model.codec.decoder.state_dict().items()}
    study.install_decoder(model,new)
    assert all(torch.equal(v,old[k]) for k,v in model.word_memory.state_dict().items())
    assert all(torch.equal(v,new[k]) for k,v in model.codec.decoder.state_dict().items())
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):study.install_decoder(model,{**new,'word_memory.weight':torch.ones(2,2)})


def test_paired_result_counts_regressions_and_rejects_missing_cases():
    rows=[{'id':i,'arm':arm,'split':'reserved_component','exact_rendered':exact}
        for i,a,b in [('1',True,False),('2',False,True),('3',True,True),('4',False,False)]
        for arm,exact in [('parent',a),('new',b)]]
    assert study.paired(rows,'parent','new','reserved_component')=={
        'cases':4,'exact_fixes':1,'exact_regressions':1,'both_exact':1,'neither_exact':1}
    with pytest.raises(ValueError):study.paired(rows[:-1],'parent','new','reserved_component')
