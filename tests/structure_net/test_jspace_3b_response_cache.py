from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_3b_response_cache as cache


def record(name, split, start):
    return cache.data.ResponseRecord(example_id=name,group_id=name,split=split,task='fixture',
        prompt_ids=(start,start+1),response_ids=(start+2,18),eos_token_id=18,finish_reason='eos',
        provenance=dict(source_uri='fixture://data',source_revision='v1',tokenizer_sha256='a'*64,
            chat_template_sha256='b'*64,prompt_provenance='fixture',response_provenance='fixture',
            assistant_boundary_verified=True))


def fixture(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir()
    records=[record('train','train',1),record('dev','development',5)]
    for split,filename in [('train','train.jsonl'),('development','development.jsonl')]:
        (corpus/filename).write_text(''.join(json.dumps(asdict(r))+'\n' for r in records if r.split==split))
    root=tmp_path/'cache';(root/'native').mkdir(parents=True)
    p=dict(schema=cache.SCHEMA,corpus=str(corpus),cut=20,width=3,total_tokens=8,
        records_sha256=cache.data.dataset_fingerprint(records),rows=cache.index_records(records),
        input_identity={},source_sha256={})
    p['fingerprint']=cache.data.digest(p);cache.write(root/'plan.json',p)
    values=torch.arange(24,dtype=torch.float32).reshape(8,3).to(torch.bfloat16)
    for name in ('teacher_hidden','boundary_20'):
        np.save(root/'native'/f'{name}.npy',values.view(torch.int16).numpy())
    cache.write(root/'native/manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
        artifacts={f.name:cache.sha(f) for f in (root/'native').iterdir()}))
    return root,records,values


def test_complete_eos_masks_and_lossless_targets(tmp_path):
    root,records,values=fixture(tmp_path)
    ds=cache.ResponseDataset(root,cut=20,pad_token_id=0)
    batch=ds.batch([0])
    assert batch['loss_mask'].tolist()==[[False,True,True,False]]
    assert batch['targets'][batch['loss_mask']].tolist()==[3,18]
    assert torch.equal(batch['teacher_hidden'][0],values[:4])
    assert torch.equal(batch['boundary_hidden'][0],values[:4])
    assert torch.equal(batch['teacher_input_ids'],batch['input_ids'])
    assert batch['teacher_record_ids']==['train']
    assert ds.training==[0] and ds.development==[1]
    ds.close()
    all36=cache.ResponseDataset(root,cut=0,pad_token_id=0)
    assert 'boundary_hidden' not in all36.batch([1])
    all36.close()


def test_cache_rejects_changed_target(tmp_path):
    root,_,_=fixture(tmp_path)
    with (root/'native/teacher_hidden.npy').open('r+b') as stream:
        stream.seek(-1,2);stream.write(b'\xff')
    with pytest.raises(ValueError,match='target changed'):
        cache.ResponseDataset(root,cut=20)


def test_split_group_and_context_limits():
    first=record('a','train',1);second=record('b','development',5)
    with pytest.raises(ValueError):
        cache.index_records([first,replace(second,group_id=first.group_id)])
    with pytest.raises(ValueError,match='288 prompt'):
        cache.index_records([replace(first,prompt_ids=(1,)*289),second])
    with pytest.raises(ValueError,match='128 response'):
        cache.index_records([replace(first,response_ids=(3,)*128+(18,)),second])


def test_batch_refuses_changed_cached_prefix(tmp_path):
    root,_,_=fixture(tmp_path);ds=cache.ResponseDataset(root,cut=20)
    ds.rows[0]['input_ids'][0]=12
    with pytest.raises(ValueError,match='prefix mismatch'):ds.batch([0])
    ds.close()
