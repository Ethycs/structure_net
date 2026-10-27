import torch
from torch import nn
import pytest

from experiments.structure_net import ncm_source_coverage as coverage
from experiments.structure_net.ncm_lm1b_source_attention import SourceAttention,SourceMemory


def toy():
    torch.manual_seed(71)
    model=nn.Module();head=nn.Module();model.concepts=head
    head.generator=nn.Module();head.generator.transformer=nn.Module()
    head.generator.transformer.blocks=nn.ModuleList([nn.Linear(8,8) for _ in range(12)])
    head.source_memory=SourceMemory(8,8,2)
    head.layers=(8,10,11);head._bound=None;head._handles=[]
    head.adapters=nn.ModuleDict({str(i):SourceAttention(8,2,0.) for i in head.layers})
    for i in head.layers:
        with torch.no_grad():head.adapters[str(i)].out.weight.normal_(std=.1)
        def hook(module,args,x,layer=i):
            if head._bound is None:return x
            return torch.cat((x[:,:1],x[:,1:]+head.adapters[str(layer)](x[:,1:],head._bound)),1)
        head._handles.append(head.generator.transformer.blocks[i].register_forward_hook(hook))
    return model.eval()


def forward(model,x,source):
    h=model.concepts;h._bound=h.source_memory(source,torch.ones(source.shape[:2],dtype=torch.bool))
    try:
        for block in h.generator.transformer.blocks:x=block(x)
        return x
    finally:h._bound=None


def test_expansion_preserves_outputs_old_tensors_rng_and_distinct_parameters():
    model=toy();x=torch.randn(1,4,8);source=torch.randn(1,7,8)
    before=forward(model,x,source);old={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state();meta=coverage.expand(model)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(before,forward(model,x,source))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.items())
    assert meta['new_parameters']==9*2*8*8 and meta['source_blocks']==list(range(12))
    assert len(model.concepts._handles)==12
    for i in meta['added_source_blocks']:
        new=model.concepts.adapters[str(i)];original=model.concepts.adapters[str(meta['query_templates'][str(i)])]
        assert new.q.weight.data_ptr()!=original.q.weight.data_ptr()
        assert torch.equal(new.q.weight,original.q.weight) and new.out.weight.count_nonzero()==0


def test_added_outputs_receive_gradients_and_hooks_can_all_be_removed():
    model=toy();meta=coverage.expand(model)
    forward(model,torch.randn(1,4,8),torch.randn(1,7,8)).square().sum().backward()
    for i in meta['added_source_blocks']:
        adapter=model.concepts.adapters[str(i)]
        assert adapter.out.weight.grad.isfinite().all() and adapter.out.weight.grad.abs().sum()>0
        assert adapter.q.weight.grad.count_nonzero()==0
    for handle in model.concepts._handles:handle.remove()
    assert not any(b._forward_hooks for b in model.concepts.generator.transformer.blocks)


def test_rejects_busy_changed_and_already_expanded_topology_before_mutation():
    model=toy();model.concepts._bound='busy'
    with pytest.raises(ValueError):coverage.expand(model)
    assert len(model.concepts.adapters)==3
    model.concepts._bound=None;coverage.expand(model)
    with pytest.raises(ValueError):coverage.expand(model)
    assert len(model.concepts.adapters)==12
