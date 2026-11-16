"""Full-depth continuation, bounded storage, and packed-only measurements."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_3b_full_progressive_recovery as study
from test_jspace_3b_rebuild_recovery import fixture
from test_jspace_3b_packed_response_model import parts


def candidate(label,correct,nll,tokens,update,path=None):
    return dict(label=label,mc_correct=correct,mc_count=256,nll=nll,response_tokens=tokens,update=update,
        mc_eligibility={'passed':False},checkpoint={'path':str(path),'sha256':study.sha(path) if path else 'none'})


def test_continuation_chooses_best_observed_even_when_every_quality_gate_fails():
    raw=candidate('prefit',190,4.,0,-1)
    early=candidate('early',195,3.,5000,10)
    later=candidate('later',195,3.,10000,19)
    lower_mc=candidate('low_nll',180,1.,25000,46)
    assert study.select([raw,later,lower_mc,early]) is early
    assert not early['mc_eligibility']['passed']
    with pytest.raises(ValueError,match='Finite'):
        study.select([candidate('bad',195,float('nan'),0,0)])


def test_best_three_pruning_is_bounded_new_attempt_only_and_preserves_metrics(tmp_path):
    root=tmp_path/'run';checkpoints=root/'checkpoints';checkpoints.mkdir(parents=True)
    rows=[]
    for index in range(5):
        path=checkpoints/f'update_{index:04d}.pt';path.write_bytes(f'weights{index}'.encode())
        rows.append(candidate(str(index),190+index,3.,index*5000,index,path))
    before=deepcopy(rows)
    retained=study.retain_best_three(root,rows)
    assert len(retained)==3 and all((checkpoints/f'update_{i:04d}.pt').exists() == (i>=2) for i in range(5))
    assert rows==before # Measurement/checksum evidence is never overwritten.
    deleted=[json.loads(line)for line in (root/'pruned_checkpoints.jsonl').read_text().splitlines()]
    assert len(deleted)==2 and all(row['sha256'] for row in deleted)
    assert study.retain_best_three(root,rows)==retained


def test_pruning_refuses_old_external_checkpoint_or_modified_provisional_payload(tmp_path):
    root=tmp_path/'run';(root/'checkpoints').mkdir(parents=True)
    outside=tmp_path/'old_immutable.pt';outside.write_bytes(b'prior experiment')
    rows=[candidate('old',1,9.,0,0,outside)]
    for i in range(3):
        path=root/'checkpoints'/f'{i}.pt';path.write_bytes(str(i).encode())
        rows.append(candidate(str(i),200,3.,1,i,path))
    with pytest.raises(ValueError,match='nonowned'):study.retain_best_three(root,rows)
    assert outside.read_bytes()==b'prior experiment'
    owned=root/'checkpoints'/'bad.pt';owned.write_bytes(b'original')
    rows[0]=candidate('changed',1,9.,0,0,owned);owned.write_bytes(b'replaced')
    with pytest.raises(ValueError,match='modified'):study.retain_best_three(root,rows)


def test_actual_packed_candidate_scores_mc_and_full_response_then_removes_only_scratch(tmp_path,monkeypatch):
    cfg,stack,norm,E,rope,dataset=fixture(0)
    model_path=tmp_path/'native';model_path.mkdir();cfg.save_pretrained(model_path)
    fit_root=tmp_path/'fit';fit_root.mkdir();(fit_root/'fit_receipt.json').write_text('{}')
    cache_root=tmp_path/'cache';(cache_root/'native').mkdir(parents=True)
    (cache_root/'native/development.json').write_text(json.dumps({'nll':.1}))
    directory=tmp_path/'run/candidates/update_0000';checkpoint=tmp_path/'run/checkpoints/update_0000.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint')
    p=dict(model={'path':str(model_path),'revision':'native-fixture'},fingerprint='study',
        data_fingerprint='cache',corpus_fingerprint='corpus',fit_root=str(fit_root),max_length=8)
    calls=[]
    def mc(model,_dataset):
        assert model.stack.packed and model.stack.stripped
        calls.append(model.readout(model.hidden_forward(dataset.batch([2]))[:,-1]))
        return dict(metrics={'correct':100,'count':256,'accuracy':100/256},eligibility={'passed':False})
    monkeypatch.setattr(study.base,'packed_mc',mc)
    row=study.score_candidate(stack,norm,E,rope,dataset,object(),p,{'response_cache_root':str(cache_root)},
        directory,{'path':str(checkpoint),'sha256':study.sha(checkpoint)},label='update_0000',update=0,tokens=0,guard=None)
    assert row['mc_correct']==100 and row['nll_ratio']>1.1
    assert row['nll_is_diagnostic'] and row['full_vocabulary_same_packed_replay']
    assert len(calls)==1 and calls[0].shape==(1,cfg.vocab_size)
    assert (directory/'development.json').is_file() and (directory/'multiple_choice.json').is_file()
    assert checkpoint.exists() and not (tmp_path/'run/scratch/update_0000').exists()
    assert not row['packed_artifact_retained']


def test_gpu_receipt_authorizes_unqualified_continuation_without_task_judgment(tmp_path):
    root=tmp_path;run=root/'run';run.mkdir();cp=run/'selected.pt';cp.write_bytes(b'weights')
    p={'fingerprint':'full'}
    value=dict(status='completed',fingerprint='full',systems_passed=True,quality_status='pending',
        relative_retention_pass=None,continuation_checkpoint=str(cp),continuation_checkpoint_sha256=study.sha(cp))
    (run/'gpu_result.json').write_text(json.dumps(value))
    (run/'gpu_manifest.json').write_text(json.dumps(dict(status='completed',fingerprint='full',
        artifacts={'gpu_result.json':study.sha(run/'gpu_result.json')},checkpoints={str(cp):study.sha(cp)})))
    assert study.gpu_result(root,p)==value
    assert not (run/'task_result.json').exists()
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):study.gpu_result(root,p)


@pytest.mark.parametrize('cut',[0,2])
def test_real_native_prefix_and_embedding_match_cached_full_sequence_and_same_packed_logits(tmp_path,cut):
    path,native,model=parts(tmp_path,cut)
    ids=torch.tensor([[3,7,11,4,6]]);captured=[]
    handle=native.model.layers[cut].register_forward_pre_hook(lambda _module,args:captured.append(args[0].detach().clone()))
    with torch.inference_mode():native(input_ids=ids,use_cache=False)
    handle.remove()
    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids,dtype=torch.bool),boundary_hidden=captured[0])
    with study.base.packed_source.attach_prefix(model,path):
        receipt=study.prefix_cache_control(model,batch,prompt_tokens=3)
        assert receipt['passed'] and receipt['full_sequence_boundary']['bit_exact']
        assert receipt['full_sequence_same_packed_logits']['bit_exact']
        assert not receipt['prompt_versus_cropped_cache']['hard_gate']
        assert receipt['execution']==('Original native prefix'if cut else 'Original tied embedding lookup')
        corrupted={**batch,'boundary_hidden':batch['boundary_hidden']+.1}
        failed=study.prefix_cache_control(model,corrupted,prompt_tokens=3)
        assert not failed['passed'] and failed['full_sequence_boundary']['max_absolute_difference']>0
