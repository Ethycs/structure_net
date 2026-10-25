import pytest
import torch

from experiments.structure_net import ncm_generated_code_decoder as study
from test_ncm_sequential_decoder import Model


def test_generated_code_input_ignores_reference_and_stale_codes():
    row={'id':'a','oracle_codes':[1]*641,'predicted_codes':[2]*641,'target_text':'irrelevant'}
    predictions={'a':{'source_id_used':'a','codes':[3]*641}}
    assert study.selected_codes(row,predictions,'cpu').tolist()==[[3]*641]
    row['oracle_codes']=[7]*641
    assert study.selected_codes(row,predictions,'cpu').tolist()==[[3]*641]
    with pytest.raises(ValueError,match='binding'):
        study.selected_codes(row,{'a':{'source_id_used':'b','codes':[3]*641}},'cpu')
    with pytest.raises(ValueError,match='hierarchy'):
        study.selected_codes(row,{'a':{'source_id_used':'a','codes':[8192]*641}},'cpu')


def test_decoder_gradient_updates_preserve_all_other_state():
    model=Model(); before=study.decoder.protected(model)
    study.decoder.configure(model,True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=study.offload.CPUAdamW(params,lr=.01)
    model.word_adapters(model.codec.decoder(torch.ones(1,4))).sum().backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    opt.step()
    assert study.decoder.protected(model)==before
    study.offload.assert_cpu_state(opt)


def test_word_loss_preserves_EOS_and_ignores_padding():
    logits=torch.randn(1,3,128,requires_grad=True)
    study.decoder.word_loss(logits,torch.tensor([[8,101,0]])).backward()
    assert logits.grad[0,0].abs().sum()>0 and logits.grad[0,1].abs().sum()>0
    assert logits.grad[0,2].abs().sum()==0
