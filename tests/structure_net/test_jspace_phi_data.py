"""Real local Phi tokenizer and source/response contract regressions; CPU only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.structure_net import jspace_phi_data as data


@pytest.fixture(scope='module')
def tokenizer():
    if not (data.DEFAULT_MODEL/'tokenizer.json').exists():pytest.skip('Local Phi tokenizer unavailable')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(data.DEFAULT_MODEL,local_files_only=True,trust_remote_code=False,use_fast=True)


def row(i=0,*,split='train',family='arc',permutation=None):
    question=f'Question {i}: Which material is a liquid at room temperature?'
    original=['iron','water','wood','stone'] if family in ('arc','hellaswag') else ['Yes','No']
    original_answer=1;permutation=permutation or list(range(len(original)))
    choices=[original[x] for x in permutation];answer=permutation.index(original_answer)
    return dict(identity=data.identity(question,choices),source_identity=data.identity(question,original),
        semantic_identity=data.semantic_identity(question,original),family=family,split=split,
        prompt=question+'\n\n'+'\n'.join(f'{chr(65+j)}. {choice}' for j,choice in enumerate(choices)),
        input_ids=[9000,9001,9002],choices=choices,answer=answer,source_answer=original_answer,
        source_split='train' if split=='train' else 'validation',source_row=i,
        permutation=permutation,permutation_preserved_reason=None)


def encode(rows,tokenizer,**kw):
    return data.retokenize(rows,tokenizer,tokenizer_sha256='a'*64,source_revision='b'*64,
        eos_token_ids=[100257,100265],**kw)


def test_real_phi_content_targets_and_template_complete_response(tokenizer):
    original=row(permutation=[2,0,3,1]);result=encode([original],tokenizer)
    item=result['items'][0];response=result['responses'][0]
    expected=tokenizer.apply_chat_template([dict(role='system',content=data.SYSTEM),
        dict(role='user',content=original['prompt'])],tokenize=True,add_generation_prompt=True)
    assert item['input_ids']==expected and item['input_ids']!=original['input_ids']
    assert item['permutation']==original['permutation'] and item['answer']==3
    positions=item['loss_positions'];assert len(set(positions))==4 and positions[-1]==len(expected)-1
    assert item['targets']==[expected[p+1] for p in positions[:3]]+[result['label_ids'][3]]
    start,end=item['user_content_token_span']
    assert all(start<=p<p+1<end for p in positions[:3])
    full=tokenizer.apply_chat_template([dict(role='system',content=data.SYSTEM),dict(role='user',content=original['prompt']),
        dict(role='assistant',content='D')],tokenize=True,add_generation_prompt=False)
    assert response['prompt_ids']+response['response_ids']==full
    assert response['response_ids']==[result['label_ids'][3],100265]
    assert response['finish_reason']=='eos' and response['eos_token_id']==100265
    assert 'gold_multiple_choice' in response['provenance']['response_provenance']


def test_response_records_really_satisfy_new_nal_contract(tokenizer):
    from experiments.neural_architecture_lab.distillation_recovery import module
    records=module('distillation_data')
    result=encode([row()],tokenizer)
    record=records.ResponseRecord(**result['responses'][0])
    batch=records.response_batch([record],pad_token_id=tokenizer.pad_token_id,max_length=416)
    assert int(batch['loss_mask'].sum())==2
    assert tuple(record.input_ids)==tuple(result['items'][0]['input_ids']+result['responses'][0]['response_ids'])


def test_overlength_rejected_without_truncation_and_offsets_rebuilt(tokenizer):
    originals=[row(i) for i in range(3)]
    long=originals[1];question=' '.join(['lengthycontext']*500)
    long.update(prompt=question+'\n\n'+'\n'.join(f'{chr(65+j)}. {c}' for j,c in enumerate(long['choices'])),
        identity=data.identity(question,long['choices']),source_identity=data.identity(question,long['choices']),
        semantic_identity=data.semantic_identity(question,long['choices']))
    result=encode(originals,tokenizer)
    assert len(result['items'])==2 and result['rejections'][0]['source_index']==1
    assert result['rejections'][0]['prompt_tokens']>384
    assert [c['source_index'] for c in result['items']]==[0,2]
    assert result['items'][1]['offset']==result['items'][0]['stop']
    assert result['tokens']==result['items'][-1]['stop']


@pytest.mark.parametrize('mutation',[
    lambda c:c.update(split='audit'),lambda c:c.update(answer=3,source_answer=1),
    lambda c:c.update(source_identity='x'*64),lambda c:c.update(semantic_identity='y'*64),
    lambda c:c.update(prompt=c['prompt']+' changed'),
])
def test_source_mapping_or_nontraining_evaluation_is_rejected(tokenizer,mutation):
    c=row();mutation(c)
    with pytest.raises(ValueError):encode([c],tokenizer)


def test_cross_split_semantic_alias_is_rejected(tokenizer):
    a=row();b=row(split='development',permutation=[1,0,2,3])
    with pytest.raises(ValueError,match='Cross-split'):encode([a,b],tokenizer)


def test_three_choice_arc_is_preserved_not_silently_discarded(tokenizer):
    c=row();question=data.question_from_case(c);choices=c['choices'][:3]
    c.update(choices=choices,permutation=[0,1,2],prompt=question+'\n\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(choices)),
        identity=data.identity(question,choices),source_identity=data.identity(question,choices),
        semantic_identity=data.semantic_identity(question,choices))
    result=encode([c],tokenizer)
    assert len(result['items'][0]['choices'])==3 and result['items'][0]['answer']==1


def test_pilot_is_balanced_source_order_not_label_selected(tokenizer):
    original=[row(1000*s+100*f+i,split=split,family=family)
        for s,split in enumerate(('train','development')) for f,family in enumerate(data.FAMILIES) for i in range(33 if s==0 else 17)]
    result=encode(original,tokenizer);indices=data.pilot_indices(result['items'])
    pilot=[result['items'][i] for i in indices];counts=data.population(pilot)
    assert len(pilot)==192 and indices==sorted(indices)
    assert all(counts['train'][f]['count']==32 and counts['development'][f]['count']==16 for f in data.FAMILIES)
    with pytest.raises(ValueError,match='Insufficient'):data.pilot_indices(result['items'][:10])


def test_changed_assistant_prefix_rejected(tokenizer,monkeypatch):
    original=tokenizer.apply_chat_template
    def wrong(messages,**kw):
        value=original(messages,**kw)
        return [42,*value[1:]] if messages[-1]['role']=='assistant' else value
    monkeypatch.setattr(tokenizer,'apply_chat_template',wrong)
    with pytest.raises(ValueError,match='assistant response'):encode([row()],tokenizer)


def test_original_source_authentication_never_decodes_sealed_files(monkeypatch):
    if not data.DEFAULT_SOURCE.exists():pytest.skip('Original broad data unavailable')
    original=data.read;opened=[]
    def guarded(path):
        opened.append(str(path));assert Path(path).name not in ('audit_tasks.json','exclusions.json','tasks.json')
        return original(path)
    monkeypatch.setattr(data,'read',guarded)
    plan,items=data.source_items(data.DEFAULT_SOURCE)
    assert len(items)==4864 and plan['counts']=={'train':4608,'development':256}
    assert len(opened)==3


def test_model_inventory_hashes_local_checkpoint_without_tensor_loading(tmp_path):
    files={'config.json':dict(model_type='phi3',num_hidden_layers=40,hidden_size=5120,vocab_size=100352,
        quantization_config=dict(bits=4,quant_method='gptq')),
        'model.safetensors.index.json':dict(weight_map={'w':'model-00001.safetensors'}),'generation_config.json':{}}
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    for name in (*data.TOKENIZER_FILES,'model-00001.safetensors'):(tmp_path/name).write_bytes(b'fixed local bytes')
    result=data.model_inventory(tmp_path)
    assert result['quantization_config']==dict(bits=4,quant_method='gptq')
    assert result['files']['model-00001.safetensors']==data.sha(tmp_path/'model-00001.safetensors')


def test_cpu_guard_precedes_source_or_model_access(tmp_path,monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(data,'source_items',lambda *_:pytest.fail('source read before CPU guard'))
    with pytest.raises(ValueError,match='CUDA hidden'):data.prepare(tmp_path/'out')


def test_actual_prepare_readback_and_artifact_model_tamper(tmp_path,monkeypatch,tokenizer):
    import transformers
    source=tmp_path/'source';source.mkdir()
    originals=[row(1000*s+100*f+i,split=split,family=family)
        for s,split in enumerate(('train','development')) for f,family in enumerate(data.FAMILIES) for i in range(32 if s==0 else 16)]
    for name,value in (('plan.json',{'fingerprint':'b'*64}),('manifest.json',{}),('items.json',{'items':originals})):
        (source/name).write_text(json.dumps(value))
    monkeypatch.setattr(data,'source_items',lambda p:({'fingerprint':'b'*64},originals))
    model=tmp_path/'model';model.mkdir()
    for name,value in {
        'config.json':dict(model_type='phi3',num_hidden_layers=40,hidden_size=5120,vocab_size=100352),
        'generation_config.json':{'eos_token_id':[100257,100265]},
        'model.safetensors.index.json':{'weight_map':{'w':'model-1.safetensors'}},
    }.items():(model/name).write_text(json.dumps(value))
    for name in (*data.TOKENIZER_FILES,'model-1.safetensors'):(model/name).write_bytes(b'local test identity')
    def load(path,**kwargs):
        assert Path(path)==model and kwargs==dict(local_files_only=True,trust_remote_code=False,use_fast=True)
        return tokenizer
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',load)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    out=tmp_path/'out';plan=data.prepare(out,source,model)
    assert data.verify(out)==plan and plan['accepted_count']==192
    pilot=data.read(out/'pilot_items.json')['items']
    assert len(pilot)==192 and pilot[0]['offset']==0 and pilot[-1]['stop']==sum(len(c['input_ids']) for c in pilot)
    assert len((out/'responses.jsonl').read_text().splitlines())==192
    responses=[json.loads(line) for line in (out/'pilot_responses.jsonl').read_text().splitlines()]
    assert [r['example_id'] for r in responses]==[c['identity'] for c in pilot]
    assert all(r['prompt_ids']==c['input_ids'] and r['response_ids'][-1]==100265 for r,c in zip(responses,pilot))
    with pytest.raises(FileExistsError):data.prepare(out,source,model)
    shard=model/'model-1.safetensors';original=shard.read_bytes();shard.write_bytes(b'changed')
    with pytest.raises(ValueError,match='model/tokenizer'):data.verify(out)
    shard.write_bytes(original)
    (out/'pilot_items.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact'):data.verify(out)
