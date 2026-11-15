"""CPU-only diagnostic admission, real compact traces and target row fidelity."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from experiments.structure_net import jspace_3b_hidden_feature_run as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def cases():
    return [dict(identity=f'case{i}',family=('arc','boolq','hellaswag','winogrande')[i%4],
        split='train' if i<512 else 'development',offset=3*i,stop=3*i+3,input_ids=[1,2,3],
        loss_positions=[0,0,1,2],targets=[1,1,2,3],choices=['a','b'],answer=0) for i in range(576)]


def tiny():
    torch.manual_seed(777)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=36,num_attention_heads=2,
        num_key_value_heads=1,attention_dropout=0.,max_position_embeddings=16)
    cfg._attn_implementation='eager'
    blocks=[]
    for layer in (33,34,35):
        native=Qwen2DecoderLayer(cfg,layer).eval()
        basis=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        blocks.append(AffineQwenBlock(native,basis,torch.randn(6,8)/4,trainable=False))
    return CompactQwenStack(blocks,geometry_mode='bf16',pad_to_multiple=0).eval().requires_grad_(False)


def inputs():
    x=torch.randn(2,5,8).bfloat16();ids=torch.arange(5)[None].expand(2,-1);lengths=[3,5]
    angles=torch.arange(5)[:,None]*torch.tensor([.1,.2])[None];angles=torch.cat((angles,angles),-1)[None]
    valid=ids<torch.tensor(lengths)[:,None];allowed=(ids[:,None,:]<=ids[:,:,None])&valid[:,None,:]
    mask=torch.zeros(2,1,5,5,dtype=torch.bfloat16).masked_fill(~allowed[:,None],torch.finfo(torch.bfloat16).min)
    return x,dict(indices=[512,513],lengths=lengths,position_ids=ids,attention_mask=mask,
        loss_positions=torch.tensor([[0,0,1,2],[0,1,2,4]])),(angles.cos(),angles.sin())


def test_admission_requires_completed_selected_fresh_verification_before_parent(tmp_path,monkeypatch):
    parent={'fingerprint':'parent'};stage=dict(parent_cut=24,parent_checkpoint_sha256=study.PARENT_SHA)
    monkeypatch.setattr(study.source,'verify_stage',lambda *_:(parent,stage))
    calls=[]
    def checked(directory,phase,_):
        calls.append((Path(directory).name,phase))
        if phase=='verify':raise ValueError('Fresh verification incomplete')
        return {}
    monkeypatch.setattr(study.source,'checked',checked)
    with pytest.raises(ValueError,match='incomplete'):study.admit(tmp_path)
    assert calls==[('block_23',p) for p in ('pilot','prealign','train','verify')]


def test_admission_does_not_promote_cut23_quality_failure_but_requires_accepted_parent(tmp_path,monkeypatch):
    parent={};stage=dict(parent_cut=24,parent_checkpoint_sha256=study.PARENT_SHA)
    monkeypatch.setattr(study.source,'verify_stage',lambda *_:(parent,stage))
    def checked(directory,phase,_):
        is_parent=Path(directory).name=='block_24'
        return dict(passed=True,quality_pass=is_parent,selected_checkpoint_sha256=study.PARENT_SHA if is_parent else 'selected')
    monkeypatch.setattr(study.source,'checked',checked)
    assert study.admit(tmp_path)[2]['train']['quality_pass'] is False
    stage['parent_checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='pinned'):study.admit(tmp_path)


def test_original_rows_preserve_nontrivial_mapping_and_reject_confirmation():
    items=cases();mapping=list(reversed(range(576)));rows=study.case_rows(items,mapping)
    assert rows[513]==dict(source_index=513,original_logit_index=62,identity='case513',family='boolq',split='development',
        native_offset=1539,native_stop=1542,length=3,answer_position=2)
    assert sum(r['split']=='train' for r in rows)==512
    items[-1]['split']='confirmation'
    with pytest.raises(ValueError,match='split'):study.case_rows(items,mapping)


def test_real_bf16_trace_bank_preserves_valid_rows_and_answer_targets(tmp_path):
    stack=tiny()
    for p in stack.parameters():p.data=p.data.bfloat16()
    stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True)
    assert study.trace_rank(stack,[33,34,35])==4
    with pytest.raises(ValueError,match='explicit common'):study.trace_rank(stack,[32,33])
    x,batch,rope=inputs()
    trace=study.diagnostic.trace_coordinates(stack,x,position_ids=batch['position_ids'],position_embeddings=rope,
        attention_mask=batch['attention_mask'])
    rows=[dict(source_index=i,identity=str(i),split='development',length=n) for i,n in zip(batch['indices'],batch['lengths'])]
    bank=study.TraceBank(tmp_path/'bank',rows,[33,34,35],study.trace_rank(stack,[33,34,35]),stack.width,create=True)
    bank.write(trace,batch);bank.close()
    bank=study.TraceBank(tmp_path/'bank',rows,[33,34,35],4,8)
    reordered=bank.batch([513,512],5,'cpu')
    for layer in (33,34,35):
        assert torch.equal(reordered['coordinates'][layer][0],trace['coordinates'][layer][1])
        assert torch.equal(reordered['coordinates'][layer][1,:3],trace['coordinates'][layer][0,:3])
        assert torch.count_nonzero(reordered['coordinates'][layer][1,3:])==0
    bank.close()
    wanted=trace['coordinates'][35][torch.tensor([0,1]),torch.tensor([2,4])]
    assert torch.equal(study.final_answers(trace,batch,[{'split':'train'}]*2),wanted)
    with pytest.raises(ValueError,match='Only training'):study.final_answers(trace,batch,[{'split':'train'},{'split':'development'}])
    assert np.load(tmp_path/'bank/coordinates.npy').shape==(8,3,4)
    assert all(p.grad is None and not p.requires_grad for p in stack.parameters())


def test_metrics_preserve_equal_case_weight_and_last_valid_position():
    collector=study.Metrics();items=[{'identity':'a','family':'arc'},{'identity':'b','family':'boolq'}]
    expected=torch.ones(2,3,2);actual=expected.clone();actual[0,0]+=1;actual[0,1:]=float('nan');actual[1]+=2
    collector.add('error',actual,expected,dict(indices=[9,2],lengths=[1,3]),items)
    value=collector.finish(['a','b'])['error']
    assert value['mse']==2.5 and value['relative_rmse']==pytest.approx(2.5**.5)
    assert value['answer_relative_rmse']==pytest.approx(2.5**.5) and value['valid_tokens']==4
    assert [r['source_index'] for r in value['per_case']]==[9,2]
    with pytest.raises(ValueError,match='alignment'):collector.finish(['b','a'])


def test_load_modes_canonicalize_bf16_without_optimizer_and_keep_parent_norm(tmp_path,monkeypatch):
    original=tiny();norm=Qwen2RMSNorm(8);E=torch.randn(16,8).bfloat16();rotary=torch.nn.Identity()
    raw=deepcopy(original)
    for p in raw.parameters():p.data=p.data.bfloat16()
    saved_norm=deepcopy(norm).bfloat16();saved_norm.weight.data.fill_(2.25)
    state=dict(stack=deepcopy(raw.state_dict()),norm=deepcopy(saved_norm.state_dict()))
    delta=dict(block=deepcopy(raw.blocks[0].state_dict()))
    first=next(iter(dict(raw.blocks[0].named_parameters())))
    delta['block'][first]=delta['block'][first]+.125
    root=tmp_path/'source';directory=root/'stages/block_23';(directory/'pilot').mkdir(parents=True)
    (directory/'prealign').mkdir();(directory/'pilot/initializer.pt').write_bytes(b'raw')
    parent_checkpoint=tmp_path/'parent.pt';parent_checkpoint.write_bytes(b'parent')
    monkeypatch.setattr(study.source,'make_stack',lambda *a,**k:(deepcopy(original),deepcopy(norm),E.clone(),deepcopy(rotary),{'passed':True}))
    monkeypatch.setattr(study.source.native,'_load',lambda path:delta if Path(path).name=='selected_entrance.pt' else state)
    parent={'_campaign':str(root)};stage={'parent_checkpoint':str(parent_checkpoint)}
    unpacked,norm_loaded,*_=study.load_mode(parent,stage,'parent_unpacked','cpu')
    assert not unpacked.packed and all(p.dtype==torch.bfloat16 and not p.requires_grad for p in unpacked.parameters())
    assert study.old.state_equal(unpacked.state_dict(),state['stack']) and torch.equal(norm_loaded.weight,saved_norm.weight)
    warm,norm_loaded,*_=study.load_mode(parent,stage,'warm','cpu')
    expected=deepcopy(raw);expected.blocks[0].load_state_dict(delta['block']);expected.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True)
    assert warm.packed and study.old.state_equal(warm.state_dict(),expected.state_dict())
    assert torch.equal(norm_loaded.weight,saved_norm.weight)


def test_prepare_freezes_sources_inputs_and_does_not_touch_incomplete_live_campaign(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(study.base,'ROOT',tmp_path)
    campaign=tmp_path/'campaign';directory=campaign/'stages/block_23';cache=tmp_path/'cache';cache.mkdir()
    items=tmp_path/'items.json';study.base.write(items,dict(items=cases()))
    producer=tmp_path/'source.py';producer.write_text('# test producer\n')
    parent=dict(fingerprint='parent',items_path=str(items),original_indices=list(range(576)),native_cache=str(cache),source_sha256={},input_identity={})
    stage=dict(fingerprint='stage',parent_checkpoint=str(campaign/'parent.pt'),parent_records=str(campaign/'parent.json'),fits={})
    evidence={'train':dict(selected_checkpoint_sha256='selected',selected_epoch=1,history=[{'epoch':i} for i in (-1,0,1)])}
    monkeypatch.setattr(study,'admit',lambda *_:(parent,stage,evidence))
    monkeypatch.setattr(study.source,'local_import_closure',lambda *_:[producer])
    files=[campaign/'plan.json',directory/'plan.json',Path(stage['parent_checkpoint']),Path(stage['parent_records']),
        directory/'pilot/initializer.pt',directory/'prealign/selected_entrance.pt',directory/'train/selected.pt',
        directory/'capture/packed_teacher_logits.npy',cache/'teacher_logits.npy']
    files.extend(cache/f'boundary_{i:02d}.npy' for i in (23,24,36))
    for cut in (23,24):
        d=campaign/'stages'/f'block_{cut:02d}';files.append(d/'plan.json')
        files.extend(d/p/n for p in study.source.PHASES for n in ('manifest.json','result.json'))
    files.extend(directory/'train'/f'epoch_{i:02d}_development.json' for i in (-1,0,1))
    for path in files:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    root=tmp_path/'diagnostic';study.prepare(root,campaign)
    plan=study.verify(root)
    assert plan['rows'][512]['split']=='development' and plan['confirmed'] is False
    assert (root/'producer/source.py').read_bytes()==producer.read_bytes()
    with pytest.raises(FileExistsError):study.prepare(root,campaign)
    (directory/'train/selected.pt').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):study.verify(root)


def test_run_reservation_precedes_any_cuda_or_model_construction(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    items=tmp_path/'items.json';study.base.write(items,dict(items=cases()))
    parent=dict(fingerprint='parent',items_path=str(items),original_indices=list(range(576)))
    stage=dict(fingerprint='stage')
    plan=dict(campaign=str(tmp_path/'campaign'),campaign_fingerprint='parent',stage_fingerprint='stage',rows=study.case_rows(cases(),list(range(576))))
    monkeypatch.setattr(study,'verify',lambda *_:plan)
    monkeypatch.setattr(study.source,'verify_stage',lambda *_:(parent,stage))
    def denied(*_):raise PermissionError('No reservation')
    monkeypatch.setattr(jspace_full_lease,'wait_ready',denied)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('CUDA initialized before guard'))
    monkeypatch.setattr(study,'load_mode',lambda *_:pytest.fail('Model loaded before guard'))
    with pytest.raises(PermissionError,match='reservation'):study.run(tmp_path/'diagnostic')
