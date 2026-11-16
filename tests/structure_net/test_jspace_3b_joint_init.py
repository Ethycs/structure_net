from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_joint_init as init


def test_canonicalization_changes_parameters_only():
    stack=nn.Linear(4,3);stack.register_buffer('geometry',torch.randn(3,4))
    norm=nn.LayerNorm(3);before=stack.geometry.clone()
    init.canonicalize(stack,norm)
    assert all(p.dtype==torch.bfloat16 for p in (*stack.parameters(),*norm.parameters()))
    assert stack.geometry.dtype==torch.float32 and torch.equal(stack.geometry,before)


def test_embedding_boundary_preserves_exact_bits_offsets_and_rejects_overwrite(tmp_path):
    E=torch.arange(40).reshape(10,4).bfloat16()
    rows=[dict(input_ids=[2,0,7],offset=0,stop=3),dict(input_ids=[9,2,1,4],offset=3,stop=7)]
    path=tmp_path/'boundary.npy'
    result=init.write_boundary(path,rows,E,chunk_tokens=2)
    actual=torch.from_numpy(np.load(path)).view(torch.bfloat16)
    assert torch.equal(actual,E[[2,0,7,9,2,1,4]])
    assert result['shape']==[7,4]
    with pytest.raises(FileExistsError):init.write_boundary(path,rows,E)


def test_inventory_counts_actual_mixed_geometry_and_bf16_parameters():
    stack=nn.Linear(4,3).bfloat16();stack.register_buffer('geometry',torch.randn(3,4))
    norm=nn.LayerNorm(3).bfloat16();E=torch.randn(11,4).bfloat16()
    rotary=nn.Module();rotary.register_buffer('inv_freq',torch.ones(2))
    r=init.inventory(stack,norm,E,rotary)
    assert r['trainable_scalar_count']==21
    assert r['trainable_tensor_bytes']==42 and r['fixed_geometry_tensor_bytes']==48
    assert r['total_unpacked_tensor_bytes']==42+48+88+8
    assert r['cpu_optimizer_and_gradient_bank_bytes']==21*14


def test_final_coordinate_frame_requires_exact_actual_buffers_and_fit_identity():
    block=nn.Module()
    for name,shape in [('basis',(4,2)),('anchor',(3,4)),('anchor_energy',(3,1))]:
        block.register_buffer(name,torch.randn(shape).bfloat16())
    stack=nn.Module();stack.blocks=nn.ModuleList([block]);stack.layer_indices=[35]
    state={'stack':{'_extra_state':{'layer_indices':[35]},**{'blocks.0.'+n:v.clone() for n,v in block.named_buffers()}}}
    assert init.final_geometry(stack,state,fit_hashes_match=True)['passed']
    assert not init.final_geometry(stack,state,fit_hashes_match=False)['passed']
    altered=deepcopy(state);altered['stack']['blocks.0.anchor'][0,0]+=1
    assert not init.final_geometry(stack,altered,fit_hashes_match=True)['passed']


def cases():
    return [dict(identity=str(i),split='train' if i<4608 else 'development',
                 offset=3*i,stop=3*i+3,input_ids=[1,2,3],content_loss_positions=[0,1,1,2],
                 content_targets=[2,3,3,4]) for i in range(4864)]


def test_initializer_dataset_excludes_audit_and_checks_every_offset():
    rows=cases();assert init.validated_items(rows)==14592
    rows[0]['split']='audit'
    with pytest.raises(ValueError,match='no audit'):init.validated_items(rows)
    rows=cases();rows[10]['offset']+=1
    with pytest.raises(ValueError,match='offsets'):init.validated_items(rows)


def test_boundary_rejects_negative_embedding_index(tmp_path):
    rows=[dict(input_ids=[-1],offset=0,stop=1)]
    with pytest.raises(ValueError,match='vocabulary'):
        init.write_boundary(tmp_path/'boundary.npy',rows,torch.ones(5,4).bfloat16())


def test_real_native_projection_canonical_checkpoint_strictly_reloads(tmp_path):
    torch.manual_seed(42)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=16,
        max_position_embeddings=32,attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation='eager'
    native=Qwen2ForCausalLM(config).bfloat16().eval()
    fits={}
    for layer in range(2):
        directory=tmp_path/str(layer);directory.mkdir();fits[layer]=directory
        torch.save(dict(basis=torch.linalg.qr(torch.randn(8,8)).Q,means=torch.randn(8,8)/10,
            rope_pairs=torch.tensor([[0]]),value_basis=torch.eye(4)[None,:,:2]),directory/'fit.pt')
        width=layer+1;features=1+width+width*(width+1)//2
        torch.save(dict(anchors=list(range(width)),mean=torch.zeros(12),scale=torch.ones(12),
            coefficients={'2':torch.randn(12,features)/100},removed=torch.tensor([10,11])),directory/'relation.pt')
    def make():
        return init.builder.build_stack(tmp_path,fits,0,None,rank=4,config=config,
            native_loader=lambda _,layer,**kw:deepcopy(native.model.layers[layer]),
            readout_loader=lambda _,**kw:(native.model.embed_tokens.weight.detach(),deepcopy(native.model.norm),deepcopy(native.model.rotary_emb)))
    stack,norm,E,rotary,proof=make()
    assert proof['native_initialization'] and list(stack.layer_indices)==[0,1]
    buffers={name:value.clone() for name,value in stack.named_buffers()}
    init.canonicalize(stack,norm)
    assert all(p.dtype==torch.bfloat16 for p in (*stack.parameters(),*norm.parameters()))
    assert all(torch.equal(value,buffers[name]) for name,value in stack.named_buffers())
    checkpoint=tmp_path/'initializer.pt'
    torch.save({'stack':stack.state_dict(),'norm':norm.state_dict()},checkpoint)
    state=torch.load(checkpoint,weights_only=True)
    copy,copy_norm,*_=make();init.canonicalize(copy,copy_norm)
    copy.load_state_dict(state['stack'],strict=True);copy_norm.load_state_dict(state['norm'],strict=True)
    assert init.builder.state_equal(copy.state_dict(),stack.state_dict())
    assert init.builder.state_equal(copy_norm.state_dict(),norm.state_dict())
    positions=torch.arange(4)[None];x=E[torch.tensor([[1,2,3,4]])]
    with torch.no_grad():
        kwargs=dict(position_ids=positions,position_embeddings=rotary(x,positions))
        assert torch.equal(stack(x,**kwargs)[0],copy(x,**kwargs)[0])


def test_checked_accepts_diagnostic_frame_mismatch_but_rejects_feature_objective(tmp_path,monkeypatch):
    plan=dict(fingerprint='fp',fit_map={'0':'fit'},model={'revision':init.REVISION},
        item_digest='items',cache_root='cache',shape=[3,4])
    monkeypatch.setattr(init,'verify',lambda root:plan)
    for name in ('initializer.pt','boundary_00.npy','items.json'):(tmp_path/name).write_bytes(name.encode())
    result=dict(status='completed',**plan,initializer=str(tmp_path/'initializer.pt'),
        initializer_sha256=init.base.sha(tmp_path/'initializer.pt'),boundary_path=str(tmp_path/'boundary_00.npy'),
        boundary_sha256=init.base.sha(tmp_path/'boundary_00.npy'),final_geometry_matches_feature_teacher=False,
        final_geometry_control={'passed':False},feature_alpha=0.,parent_feature_receipt=None,
        teacher_slots=[4,5,6,7],confirmed=False)
    def receipts():
        import json
        (tmp_path/'result.json').write_text(json.dumps(result))
        (tmp_path/'manifest.json').write_text(json.dumps(dict(status='completed',fingerprint='fp',
            artifacts={name:init.base.sha(tmp_path/name) for name in ('initializer.pt','boundary_00.npy','items.json','result.json')})))
    receipts();assert init.checked(tmp_path)['feature_alpha']==0.
    result['feature_alpha']=.419;receipts()
    with pytest.raises(ValueError,match='qualification'):init.checked(tmp_path)
    result['feature_alpha']=0.;receipts();(tmp_path/'initializer.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact changed'):init.checked(tmp_path)
