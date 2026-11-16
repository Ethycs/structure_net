"""Full-pass coverage, EOS integrity, separate groups, and cache reuse."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_3b_usable_recovery_data as study


def record(index, *, split='train', group=None, response_length=3):
    return study.cache.data.ResponseRecord(example_id=f'{split}-{index}', group_id=group or f'{split}-group-{index}',
        split=split, task='fixture', prompt_ids=(10, index+100, 12),
        response_ids=(20,)*(response_length-1)+(99,), finish_reason='eos', eos_token_id=99,
        provenance=dict(source_uri='fixture://source', source_revision='fixture-v1', tokenizer_sha256='tokenizer',
            chat_template_sha256='template', prompt_provenance='observed', response_provenance='observed',
            assistant_boundary_verified=True, source_family=f'family{index%2}'))


def population():
    rows=[record(i, response_length=2+i%5) for i in range(33)]
    rows[1]=replace(rows[1],group_id=rows[0].group_id)
    return rows+[record(100+i, split='development') for i in range(4)]


def test_all_examples_survive_shared_train_groups_and_no_replacement_pass():
    rows=population();p=study.training_plan(rows, accumulation_examples=4, buffer_size=16, thresholds=(20,50,80))
    assert p['training_examples']==33 and p['training_groups']==32
    assert len(p['epochs'])==1 and len(p['training']['windows'])==9
    assert sorted(p['training']['order'])==list(range(33))
    assert p['training']['actual_supervised_tokens']==sum(len(r.response_ids)for r in rows[:33])
    assert p['training']['packing'] is False and p['training']['microbatch_examples']==1


def test_full_epochs_are_deterministic_and_freshly_shuffled_without_new_data_claim():
    rows=population();kwargs=dict(epochs=3, accumulation_examples=4, buffer_size=16, thresholds=(20,50,80))
    p=study.training_plan(rows,**kwargs)
    assert p==study.training_plan(rows,**kwargs)
    assert len({tuple(epoch['order'])for epoch in p['epochs']})==3
    assert all(sorted(epoch['order'])==list(range(33)) for epoch in p['epochs'])
    final=study.coverage_report(p,p['training']['order'])
    assert final['presentations']==99 and final['unique_examples']==33 and final['unique_source_groups']==32
    assert final['complete_epochs']==3 and final['planned_run_completed']


def test_early_checkpoint_has_honest_coverage_and_full_endpoint_is_always_saved():
    p=study.training_plan(population(),accumulation_examples=4,buffer_size=16,thresholds=(20,50,80))
    first=p['checkpoints'][0];last=p['checkpoints'][-1]
    assert first['early_candidate_allowed'] and not first['full_training_pass']
    assert last['full_training_pass'] and last['examples_seen']==33
    prefix=p['training']['order'][:first['examples_seen']]
    progress=study.coverage_report(p,prefix)
    assert progress['supervised_response_tokens']==first['supervised_tokens']
    assert not progress['all_training_examples_seen'] and not progress['planned_run_completed']
    assert study.coverage_report(p,p['training']['order'])['all_training_examples_seen']


def test_unadmitted_reordering_or_duplicate_receipts_cannot_claim_full_coverage():
    p=study.training_plan(population(),accumulation_examples=4,buffer_size=16)
    order=p['training']['order']
    with pytest.raises(ValueError,match='frozen'):
        study.coverage_report(p,[order[1],order[0],*order[2:]])
    with pytest.raises(ValueError): study.coverage_report(p,[*order,order[0]])
    with pytest.raises(ValueError): study.coverage_report({'training':{'order':[1]}},[True])
    empty=study.coverage_report(p,[])
    assert empty['supervised_response_tokens']==0 and empty['complete_epochs']==0


def test_dev_groups_and_test_material_cannot_enter_the_training_inventory():
    rows=population();rows[-1]=replace(rows[-1],group_id=rows[0].group_id)
    with pytest.raises(ValueError,match='Cross-split'):
        study.training_plan(rows)
    rows=population();rows[-1]=replace(rows[-1],split='test')
    with pytest.raises(ValueError,match='admitted'):
        study.training_plan(rows)


def test_non_eos_and_oversize_responses_are_rejected_not_trimmed_or_completed():
    rows=population();rows[0]=replace(rows[0],finish_reason='stop')
    with pytest.raises(ValueError,match='observed terminal EOS'):
        study.training_plan(rows)
    rows=population();rows[0]=record(0,response_length=129)
    with pytest.raises(ValueError,match='never truncates'):
        study.training_plan(rows)


def test_every_real_response_and_eos_remains_supervised_in_existing_batch_contract():
    rows=[record(0,response_length=5),record(1,response_length=2)]
    batch=study.cache.data.response_batch(rows,pad_token_id=0,max_length=416)
    assert batch['loss_mask'].sum()==7
    for i,row in enumerate(rows):
        positions=batch['loss_mask'][i].nonzero().flatten()
        assert tuple(batch['targets'][i,positions].tolist())==row.response_ids
        assert batch['targets'][i,positions[-1]]==99
        assert not batch['loss_mask'][i,:len(row.prompt_ids)-1].any()


def test_dataset_is_a_delegate_without_copying_banks_or_changing_masks(tmp_path,monkeypatch):
    sentinel={'loss_mask':torch.tensor([[False,True,True]]),'teacher_hidden':torch.ones(1,3,2)}
    class Native:
        def __init__(self,root,*,cut):
            assert root=='/immutable-native' and cut==33
            self.records=population();self.rows=[];self.width=2;self.cut=cut;self.pad_token_id=0;self.items=[]
            self.closed=False
        def batch(self,indices,**kwargs):
            assert indices==[0];return sentinel
        def close(self):self.closed=True
    p=dict(full_data_root='/immutable-native',cut=33,training_indices=list(range(33)),
           development_indices=list(range(33,37)),max_length=416)
    monkeypatch.setattr(study,'verify',lambda *a,**kw:(p,population()))
    monkeypatch.setattr(study.banks,'RebuildDataset',Native)
    dataset=study.RecoveryDataset(tmp_path)
    assert dataset.batch([0]) is sentinel and dataset.training==list(range(33))
    dataset.close();assert dataset.native.closed


def test_window_token_counts_and_checkpoint_thresholds_follow_complete_examples():
    rows=population();p=study.training_plan(rows,accumulation_examples=4,buffer_size=16,thresholds=(20,50,80))
    cumulative=0
    for indices,window in zip(p['training']['windows'],p['training']['window_inventory']):
        count=sum(len(rows[i].response_ids)for i in indices);cumulative+=count
        assert window['supervised_tokens']==count and window['cumulative_supervised_tokens']==cumulative
    for checkpoint in p['checkpoints']:
        assert checkpoint['supervised_tokens']==p['training']['window_inventory'][checkpoint['update']-1]['cumulative_supervised_tokens']
        assert all(checkpoint['epoch_supervised_tokens']>=threshold for threshold in checkpoint['crossed_thresholds'])


def test_actual_producer_closure_has_protocol_nal_and_no_mutable_worker():
    paths=study.source_files()
    assert study.PROTOCOL in paths and study.ROOT/'pyproject.toml' in paths
    assert study.ROOT/'src/neural_architecture_lab/distillation_data.py' in paths
    assert not any(path.stem=='jspace_3b_usable_recovery' for path in paths)


def test_preparation_references_completed_native_banks_without_hashing_or_copying_payloads(tmp_path,monkeypatch):
    full=tmp_path/'full';native=full/'native';native.mkdir(parents=True)
    response=tmp_path/'response';(response/'native').mkdir(parents=True)
    for path in (full/'plan.json',native/'manifest.json',response/'plan.json',response/'native/manifest.json'):
        path.write_text('{}')
    study.write(native/'result.json',dict(full_teacher_bit_exact=True,teacher_unchanged=True,
        existing_boundary_bit_exact=True,seed_boundary35_bit_exact=True))
    payloads=[native/f'{name}.npy'for name in ('boundary_33','boundary_34','mc_boundary_33')]
    payloads.append(response/'native/teacher_hidden.npy')
    for path in payloads:path.write_bytes(b'immutable-bank-placeholder')
    old={str(path):study.identity(path)for path in payloads}
    source=tmp_path/'producer.py';source.write_text('SOURCE=1\n')
    inventory=[dict(family=f'family{i//2}')for i in range(16)]
    schedule=dict(training_examples=4441,training_groups=4393,development_examples=128,
        records_sha256='records',development_indices=list(range(16)),record_inventory=inventory,
        training={'actual_supervised_tokens':105089,'windows':[[0]]})
    plan=dict(cuts=list(range(36)),fingerprint='native',records_sha256='records',
        response_cache_root=str(response),response_cache_fingerprint='response',model={'revision':'teacher'},
        corpus='/admitted',rows=[],width=2048,vocabulary=151936,total_tokens=1,
        generation_indices=list(range(16)),generation_eos_token_ids=[99],input_identity={})
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'source_files',lambda:{source})
    monkeypatch.setattr(study.banks,'verify',lambda *a,**kw:(plan,[]))
    calls=[]
    monkeypatch.setattr(study.banks,'_manifest_files',lambda *a,**kw:calls.append(kw))
    monkeypatch.setattr(study,'training_plan',lambda *a,**kw:schedule)
    original_sha=study.sha
    def metadata_sha(path):
        assert Path(path).suffix!='.npy','Rehashed a native payload'
        return original_sha(path)
    monkeypatch.setattr(study,'sha',metadata_sha)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    result=study.prepare(tmp_path/'recovery_data',full)
    assert result['bank_copy_bytes']==0 and calls==[{'hash_payloads':False}]
    assert not list((tmp_path/'recovery_data').rglob('*.npy'))
    assert old=={str(path):study.identity(path)for path in payloads}
