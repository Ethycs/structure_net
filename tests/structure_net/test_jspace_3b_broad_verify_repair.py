"""Device-faithful rotary reconstruction without weakening replay evidence."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_broad_verify_repair as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net.jspace_incremental_suffix import fixed_state


def tiny_fixed():
    torch.manual_seed(8721)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=16,tie_word_embeddings=True)
    config._attn_implementation='eager'
    native=Qwen2ForCausalLM(config).bfloat16().eval()
    def stack(trainable):
        return CompactQwenStack([AffineQwenBlock(native.model.layers[0],torch.eye(8)[:,:4],
            torch.zeros(1,8),trainable=trainable,parameter_dtype=torch.bfloat16)],geometry_mode='bf16')
    return config,native,stack


def test_trainability_never_reclassifies_learned_weights_as_fixed_buffers():
    _,native,factory=tiny_fixed();trainable=factory(True);frozen=factory(False)
    frozen.load_state_dict(trainable.state_dict(),strict=True)
    assert list(dict(trainable.named_parameters()))==list(dict(frozen.named_parameters()))
    assert list(dict(trainable.named_buffers()))==list(dict(frozen.named_buffers()))
    assert fixed_state(trainable,native.lm_head.weight,native.model.rotary_emb)==fixed_state(frozen,native.lm_head.weight,native.model.rotary_emb)
    assert 'inv_freq' in dict(native.model.rotary_emb.named_buffers())
    assert 'inv_freq' not in native.model.rotary_emb.state_dict()


def test_rotary_factory_receives_original_model_configuration_and_requested_device(tmp_path,monkeypatch):
    config,_,_=tiny_fixed();config.save_pretrained(tmp_path/'model')
    seen=[]
    def factory(*,config,device):
        seen.append((config.hidden_size,config.rope_theta,str(device)))
        return Qwen2RotaryEmbedding(config=config,device='cpu')
    contexts=[]
    monkeypatch.setattr(study.trial,'_model_device',lambda device:contexts.append(str(device)) or nullcontext())
    rotary=study.restore_rotary({'model':{'path':str(tmp_path/'model')}},'cuda:0',guard=lambda:seen.append('guard'),rotary_factory=factory)
    assert seen==['guard',(8,config.rope_theta,'cuda:0')] and contexts==['cuda:0']
    assert not rotary.training and all(not p.requires_grad for p in rotary.parameters())


@pytest.mark.parametrize('component',['geometry','readout','rotary'])
def test_whole_fixed_hash_detects_each_component_without_excluding_rotary(component):
    _,native,factory=tiny_fixed();stack=factory(False);E=native.lm_head.weight.detach().clone()
    rotary=deepcopy(native.model.rotary_emb);before=fixed_state(stack,E,rotary)
    target={'geometry':stack.blocks[0].basis,'readout':E,'rotary':rotary.inv_freq}[component]
    with torch.no_grad():target.view(-1)[0]+=1
    with pytest.raises(ValueError,match='Complete fixed'):
        study.assert_fixed(stack,E,rotary,before)


def test_exact_whole_fixed_inventory_passes_and_retains_rope():
    _,native,factory=tiny_fixed();stack=factory(False)
    expected=fixed_state(stack,native.lm_head.weight,native.model.rotary_emb)
    assert study.assert_fixed(stack,native.lm_head.weight,native.model.rotary_emb,expected)==expected


def failure(arm='parent_legacy',fingerprint='original'):
    return dict(status='failed',arm=arm,phase='verify',fingerprint=fingerprint,error=study.FAILURE)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def repair_fixture(tmp_path,monkeypatch,*,with_failure=True):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    recovery=tmp_path/'recovery';arm='parent_legacy';train=recovery/arm/'train';train.mkdir(parents=True)
    write_json(recovery/'plan.json',dict(fingerprint='original'))
    fixed=dict(sha256='fixed-whole-inventory',tensor_bytes=24)
    candidates=[]
    for step in (144,288):
        checkpoint=train/f'update_{step}.pt';checkpoint.write_bytes(f'actual weights {step}'.encode())
        report=train/f'update_{step}_development.json'
        write_json(report,dict(metrics={'count':1,'correct':1,'ce':.5},records=[dict(identity='dev',answer=0,prediction=0,logits=[1.,0.])]))
        candidates.append(dict(updates=step,checkpoint=str(checkpoint),checkpoint_sha256=study.base.sha(checkpoint),
            report=str(report),report_sha256=study.base.sha(report)))
    trained=dict(candidate=candidates[1],diagnostic_best=candidates[0],fixed_state=fixed,development_passed=True)
    write_json(train/'result.json',trained);write_json(train/'manifest.json',dict(status='completed'))
    target=recovery/arm/'verify'
    if with_failure:write_json(target/'failure.json',failure())
    monkeypatch.setattr(study.trial,'verify',lambda *_:{'fingerprint':'original'})
    monkeypatch.setattr(study.trial,'checked',lambda *_:trained)
    source=Path(study.__file__).resolve()
    monkeypatch.setattr(study.trial.source.inherited,'local_import_closure',lambda *_:[source])
    repair=tmp_path/'repair';study.prepare(repair,recovery,arm)
    plan,_,actual=study.verify(repair);assert actual==trained
    checks={}
    for c in candidates:
        report=json.loads(Path(c['report']).read_text());digest=c['checkpoint_sha256']
        checks[digest]=dict(strict_state_load=True,fixed_state_unchanged=True,fixed_state=fixed,
            checkpoint_sha256=digest,metrics=report['metrics'],records=report['records'])
    result=dict(status='completed',arm=arm,passed=True,checks=checks,development_passed=True,confirmed=False)
    uuid='GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff'
    resources=dict(lease_id='test-only',allocation={uuid:11264},
        allocator=dict(physical_uuid=uuid,allocator_limit_bytes=11008*1024**2,lease_bytes=11264*1024**2),
        peak_reserved_bytes=1024,elapsed_seconds=1.)
    return repair,plan,result,resources


@pytest.mark.parametrize('with_failure',[False,True])
def test_publication_is_append_only_and_binds_distinct_repair_producer(tmp_path,monkeypatch,with_failure):
    repair,plan,result,resources=repair_fixture(tmp_path,monkeypatch,with_failure=with_failure)
    target=Path(plan['target']);prior=(target/'failure.json').read_bytes() if with_failure else None
    receipt=study.publish(repair,plan,result,resources)
    published=json.loads((target/'result.json').read_text());manifest=json.loads((target/'manifest.json').read_text())
    assert manifest['fingerprint']=='original' and manifest['repair_fingerprint']==plan['fingerprint']
    assert published['verification_producer']['fingerprint']==plan['fingerprint']
    assert (target/'repair/plan.json').read_bytes()==(repair/'plan.json').read_bytes()
    assert receipt['manifest_sha256']==study.base.sha(target/'manifest.json')
    if with_failure:
        assert (target/'failure.json').read_bytes()==prior
        assert manifest['artifacts']['failure.json']==plan['prior_destination']['failure_sha256']
    else:assert not (target/'failure.json').exists()
    assert 'repair/receipt.json' in manifest['artifacts']
    assert any(name.startswith('repair/producer/') for name in manifest['artifacts'])
    with pytest.raises(ValueError,match='overwrite'):study.publish(repair,plan,result,resources)


@pytest.mark.parametrize('defect',['wrong_failure','extra_file','completed'])
def test_unrelated_or_completed_verification_destination_is_not_repairable(tmp_path,defect):
    target=tmp_path/'verify';write_json(target/'failure.json',failure())
    if defect=='wrong_failure':write_json(target/'failure.json',dict(failure(),error='OutOfMemoryError'))
    elif defect=='extra_file':(target/'partial.pt').write_bytes(b'partial state')
    else:write_json(target/'manifest.json',dict(status='completed'))
    with pytest.raises(ValueError):study.destination_state(target,'parent_legacy','original')


@pytest.mark.parametrize('defect',['failure_tamper','missing_candidate','changed_fixed','changed_rows'])
def test_failed_or_tampered_checks_cannot_publish_success(tmp_path,monkeypatch,defect):
    repair,plan,result,resources=repair_fixture(tmp_path,monkeypatch)
    target=Path(plan['target'])
    if defect=='failure_tamper':write_json(target/'failure.json',dict(failure(),additional='changed after preparation'))
    elif defect=='missing_candidate':result['checks'].pop(next(iter(result['checks'])))
    elif defect=='changed_fixed':next(iter(result['checks'].values()))['fixed_state']={'sha256':'different','tensor_bytes':24}
    else:next(iter(result['checks'].values()))['records'][0]['logits'][0]+=1
    with pytest.raises(ValueError):study.publish(repair,plan,result,resources)
    assert set(p.name for p in target.iterdir())=={'failure.json'}
    assert not (repair/'result.json').exists()


def test_denied_lease_precedes_rotary_construction_and_cuda(tmp_path,monkeypatch):
    repair,_,_,_=repair_fixture(tmp_path,monkeypatch)
    def denied(*_):raise RuntimeError('lease denied')
    monkeypatch.setattr(study,'wait_ready',denied)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *_:pytest.fail('CUDA initialized'))
    monkeypatch.setattr(study,'restore_rotary',lambda *_a,**_kw:pytest.fail('rotary constructed before lease'))
    with pytest.raises(RuntimeError,match='lease denied'):study.run(repair)
    assert json.loads((repair/'run_failure.json').read_text())['allocator'] is None


@pytest.mark.parametrize('defect',['wrong_card','overrun'])
def test_invalid_resource_receipt_is_rejected_before_publication(tmp_path,monkeypatch,defect):
    repair,plan,result,resources=repair_fixture(tmp_path,monkeypatch)
    if defect=='wrong_card':
        uuid='GPU-7b1db3cb-fake'
        resources['allocation']={uuid:11264};resources['allocator']['physical_uuid']=uuid
    else:resources['peak_reserved_bytes']=resources['allocator']['allocator_limit_bytes']+1
    with pytest.raises(ValueError):study.publish(repair,plan,result,resources)
    assert {p.name for p in Path(plan['target']).iterdir()}=={'failure.json'}
    assert not (repair/'result.json').exists()
