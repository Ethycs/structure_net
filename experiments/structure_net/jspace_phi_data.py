"""Retokenize authenticated public MC TRAIN/DEV for the local Phi-4 checkpoint.

This is a compression-pilot dataset, not a generation-recovery corpus. Sealed
audit/confirmation files are neither decoded nor scanned. Gold-letter responses
use the actual chat template; no EOS or explanatory answer is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import shutil

from .jspace_3b_broad_recovery_data import ContentLayout, question_from_case, semantic_identity
from .jspace_recovery_data import SYSTEM, digest, identity, read, sha, write

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'jspace.phi-data.v1'
FAMILIES = ('arc', 'boolq', 'hellaswag', 'winogrande')
DEFAULT_SOURCE = ROOT/'data/experiments/jspace_3b_broad_recovery_data/20261115_v1'
DEFAULT_MODEL = Path('/data/models/Phi-4-GPTQ-int4')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt')
ARTIFACTS = ('plan.json','items.json','pilot_items.json','tasks.json','rejections.json','responses.jsonl','pilot_responses.jsonl')
SCOPE = 'Short multiple-choice compression pilot; gold letter plus template terminator; no general generation-quality claim'


def response_digest(value):
    """Canonical digest used by NAL ResponseRecord (different from old plans)."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def source_items(root):
    """Authenticate only the existing TRAIN/DEV payload and its producing chain."""
    root = Path(root).resolve()
    plan, manifest = read(root/'plan.json'), read(root/'manifest.json')
    if (plan.get('schema') != 'jspace.3b-broad-recovery-data.v1' or
            plan.get('fingerprint') != digest({k:v for k,v in plan.items() if k != 'fingerprint'}) or
            manifest.get('fingerprint') != plan['fingerprint'] or manifest.get('status') != 'frozen_unscored'):
        raise ValueError('Authenticated broad recovery source required')
    for name in ('plan.json', 'items.json'):
        if manifest.get('artifacts', {}).get(name) != sha(root/name):
            raise ValueError('Source TRAIN/DEV artifact changed')
    for path, expected in plan['source_sha256'].items():
        archived = root/'producer'/Path(path).relative_to(ROOT)
        if sha(archived) != expected:
            raise ValueError('Original data-producing archive changed')
    items = read(root/'items.json')['items']
    if (digest(items) != plan['items_digest'] or
            Counter(c['split'] for c in items) != {'train':4608, 'development':256} or
            Counter(c['family'] for c in items if c['split']=='development') != dict.fromkeys(FAMILIES,64)):
        raise ValueError('Original 4608 TRAIN and 256 DEV rows required')
    validate_sources(items)
    return plan, items


def validate_sources(items):
    seen, groups = set(), {}
    for c in items:
        if (c['split'] not in ('train','development') or c['family'] not in FAMILIES or
                c['identity'] in seen or type(c['answer']) is not int or
                not 0 <= c['answer'] < len(c['choices']) or not 2 <= len(c['choices']) <= 4):
            raise ValueError('Unique supported TRAIN/DEV MC rows required')
        question = question_from_case(c)
        permutation = c['permutation']
        if (sorted(permutation) != list(range(len(c['choices']))) or
                permutation[c['answer']] != c['source_answer'] or
                c['semantic_identity'] != semantic_identity(question,c['choices'])):
            raise ValueError('Source semantics or option-order answer mapping changed')
        original_choices = [None]*len(permutation)
        for current,original in enumerate(permutation):original_choices[original]=c['choices'][current]
        if identity(question,original_choices)!=c['source_identity']:
            raise ValueError('Original source identity differs from reversed option permutation')
        for group in (c['source_identity'], c['semantic_identity']):
            if group in groups and groups[group] != c['split']:
                raise ValueError('Cross-split source/semantic leakage')
            groups[group] = c['split']
        seen.add(c['identity'])


def model_inventory(path):
    """Hash exact local weights/config and tokenizer; never load model tensors."""
    path = Path(path).resolve()
    config = read(path/'config.json')
    if config.get('model_type') != 'phi3' or config.get('num_hidden_layers') != 40:
        raise ValueError('Expected the local forty-layer Phi-4 model')
    index = read(path/'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(Path(s).name != s or not s.endswith('.safetensors') for s in shards):
        raise ValueError('Invalid local model shard inventory')
    names = ['config.json','generation_config.json','model.safetensors.index.json',*TOKENIZER_FILES,*shards]
    files = {name:sha(path/name) for name in names}
    tokenizer = {name:files[name] for name in TOKENIZER_FILES}
    return dict(path=str(path),files=files,tokenizer_files=tokenizer,tokenizer_sha256=digest(tokenizer),
        quantization_config=config.get('quantization_config'),model_type=config['model_type'],
        layers=40,hidden_size=config['hidden_size'],vocabulary=config['vocab_size'],
        source_precision='exact local checkpoint; quantization metadata retained, no precision reinterpretation')


def retokenize(items, tokenizer, *, tokenizer_sha256, source_revision,
               max_prompt_tokens=384, max_total_tokens=416, eos_token_ids=None):
    """Preserve raw text, labels and existing permutations; filter, never truncate."""
    if not 3 <= max_prompt_tokens <= max_total_tokens <= 416:
        raise ValueError('Explicit bounded prompt/total limits required')
    validate_sources(items)
    labels = [tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(4)]
    if any(len(x)!=1 for x in labels) or len(set(x[0] for x in labels)) != 4:
        raise ValueError('Four distinct single-token answer labels required')
    labels = [x[0] for x in labels]
    layout = ContentLayout(tokenizer,labels)
    template_sha = digest(tokenizer.chat_template)
    accepted, rejected, responses = [], [], []
    offset = 0
    for index,c in enumerate(items):
        encoded = layout.encode(question_from_case(c),c['choices'],c['answer'])
        ids = encoded['input_ids']
        messages = [dict(role='system',content=SYSTEM),dict(role='user',content=c['prompt'])]
        full = list(tokenizer.apply_chat_template([*messages,dict(role='assistant',content=chr(65+c['answer']))],
            tokenize=True,add_generation_prompt=False))
        if full[:len(ids)] != ids or len(full) <= len(ids) or full[len(ids)] != labels[c['answer']]:
            raise ValueError('Actual assistant response does not extend exact generation prefix/answer label')
        response = full[len(ids):]
        if len(ids)>max_prompt_tokens or len(full)>max_total_tokens:
            rejected.append(dict(source_index=index,identity=c['identity'],family=c['family'],split=c['split'],
                prompt_tokens=len(ids),total_tokens=len(full),reason='overlength; no truncation'))
            continue
        retained = {k:c[k] for k in ('identity','source_identity','semantic_identity','family','split','choices','answer',
            'source_answer','source_row','source_split','permutation','permutation_preserved_reason')}
        row = dict(retained,**encoded,source_index=index,group_id=c['semantic_identity'],
            original_input_ids_sha256=digest(c['input_ids']),loss_positions=encoded['content_loss_positions'],
            targets=encoded['content_targets'],offset=offset,stop=offset+len(ids))
        accepted.append(row);offset += len(ids)
        eos_ids = [tokenizer.eos_token_id] if eos_token_ids is None else eos_token_ids
        eos = response[-1] if response[-1] in eos_ids else tokenizer.eos_token_id
        responses.append(dict(example_id=c['identity'],group_id=c['semantic_identity'],split=c['split'],task=c['family'],
            prompt_ids=ids,response_ids=response,finish_reason='eos' if response[-1]==eos else 'stop',eos_token_id=eos,
            provenance=dict(source_uri='local authenticated broad recovery TRAIN/DEV',source_revision=source_revision,
                tokenizer_sha256=tokenizer_sha256,chat_template_sha256=template_sha,
                prompt_provenance='public task text; original option permutation retained',
                response_provenance='gold_multiple_choice_letter_with_actual_template_terminator',
                assistant_boundary_verified=True,tokenized_prefix_sha256=response_digest(ids),
                source_identity=c['source_identity'],scope=SCOPE)))
    return dict(items=accepted,responses=responses,rejections=rejected,label_ids=labels,
        tokenizer_sha256=tokenizer_sha256,chat_template_sha256=template_sha,tokens=offset)


def pilot_indices(items, *, train_per_family=32, development_per_family=16):
    selected = []
    for split,count in (('train',train_per_family),('development',development_per_family)):
        for family in FAMILIES:
            pool = [i for i,c in enumerate(items) if c['split']==split and c['family']==family]
            if len(pool)<count:
                raise ValueError(f'Insufficient untruncated {split}/{family}: {len(pool)} < {count}')
            selected.extend(pool[:count])
    return sorted(selected)


def population(items):
    return {s:{f:dict(count=len(rows:=[c for c in items if c['split']==s and c['family']==f]),
        answer_counts=dict(Counter(str(c['answer']) for c in rows)),
        tokens=sum(len(c['input_ids']) for c in rows)) for f in FAMILIES} for s in ('train','development')}


def prepare(root,source_root=DEFAULT_SOURCE,model_path=DEFAULT_MODEL,*,max_prompt_tokens=384,max_total_tokens=416):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU data preparation requires CUDA hidden')
    root,source_root = Path(root).resolve(),Path(source_root).resolve()
    if root.exists():raise FileExistsError(root)
    original,rows=source_items(source_root);model=model_inventory(model_path)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model['path'],local_files_only=True,trust_remote_code=False,use_fast=True)
    eos_ids=read(Path(model['path'])/'generation_config.json').get('eos_token_id',tokenizer.eos_token_id)
    if type(eos_ids)is int:eos_ids=[eos_ids]
    if not isinstance(eos_ids,list) or any(type(x)is not int or x<0 for x in eos_ids):raise ValueError('Explicit real EOS IDs required')
    encoded=retokenize(rows,tokenizer,tokenizer_sha256=model['tokenizer_sha256'],source_revision=original['fingerprint'],
        max_prompt_tokens=max_prompt_tokens,max_total_tokens=max_total_tokens,eos_token_ids=eos_ids)
    indices=pilot_indices(encoded['items'])
    pilot=[];offset=0
    for i in indices:
        row=dict(encoded['items'][i],full_index=i,offset=offset,stop=offset+len(encoded['items'][i]['input_ids']))
        pilot.append(row);offset=row['stop']
    sources=[Path(__file__).resolve(),ROOT/'tests/structure_net/test_jspace_phi_data.py',
        ROOT/'experiments/structure_net/jspace_3b_broad_recovery_data.py',ROOT/'experiments/structure_net/jspace_recovery_data.py']
    plan=dict(schema=SCHEMA,source_root=str(source_root),source_fingerprint=original['fingerprint'],model=model,
        source_artifact_sha256={name:sha(source_root/name) for name in ('plan.json','manifest.json','items.json')},
        source_sha256={str(p):sha(p) for p in sources},label_ids=encoded['label_ids'],
        tokenizer_sha256=model['tokenizer_sha256'],chat_template_sha256=encoded['chat_template_sha256'],
        eos_token_ids=eos_ids,
        max_prompt_tokens=max_prompt_tokens,max_total_tokens=max_total_tokens,scope=SCOPE,
        source_count=len(rows),accepted_count=len(encoded['items']),rejected_count=len(encoded['rejections']),
        population=population(encoded['items']),pilot_population=population(pilot),pilot_indices=indices,
        training_order=[i for i,c in enumerate(encoded['items']) if c['split']=='train'],
        position_policy='Three meaningful lexical next-token positions within user content, excluding option labels/newlines, plus answer',
        selection_policy='Preserve source order and existing permutations; reject overlength; first32 TRAIN/16 DEV per family for pilot',
        response_policy='Template-rendered gold MC letter only; no generated explanation, appended EOS or general corpus claim',
        audit_or_confirmation_decoded=False,confirmed=False,
        runtime_packages={name:version(name) for name in ('transformers','tokenizers')})
    plan['fingerprint']=digest(plan);root.mkdir(parents=True)
    for path,expected in plan['source_sha256'].items():
        dest=root/'producer'/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        if sha(dest)!=expected:raise ValueError('Producer archive changed')
    write(root/'plan.json',plan)
    for name,value in (('items.json',dict(items=encoded['items'],label_ids=encoded['label_ids'],tokens=encoded['tokens'])),
        ('pilot_items.json',dict(items=pilot,label_ids=encoded['label_ids'],tokens=offset)),
        ('tasks.json',dict(splits={s:[c for c in encoded['items'] if c['split']==s] for s in ('train','development')})),
        ('rejections.json',dict(rows=encoded['rejections']))):write(root/name,value)
    for name,records in (('responses.jsonl',encoded['responses']),
                         ('pilot_responses.jsonl',[encoded['responses'][i] for i in indices])):
        with (root/name).open('x') as stream:
            for record in records:stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    write(root/'manifest.json',dict(status='prepared_unscored',fingerprint=plan['fingerprint'],
        artifacts={n:sha(root/n) for n in ARTIFACTS}))
    return verify(root,live=True)


def verify(root,*,live=False):
    root=Path(root);plan,manifest=read(root/'plan.json'),read(root/'manifest.json')
    if (plan.get('schema')!=SCHEMA or plan.get('fingerprint')!=digest({k:v for k,v in plan.items() if k!='fingerprint'}) or
            manifest.get('fingerprint')!=plan['fingerprint'] or manifest.get('status')!='prepared_unscored' or
            set(manifest.get('artifacts',{}))!=set(ARTIFACTS) or
            plan.get('confirmed')is not False or plan.get('audit_or_confirmation_decoded')is not False):
        raise ValueError('Phi data plan changed')
    for name,expected in manifest['artifacts'].items():
        if Path(name).name!=name or sha(root/name)!=expected:raise ValueError('Phi data artifact changed')
    for name,expected in plan['source_artifact_sha256'].items():
        if sha(Path(plan['source_root'])/name)!=expected:raise ValueError('Original TRAIN/DEV source changed')
    for name,expected in plan['model']['files'].items():
        if sha(Path(plan['model']['path'])/name)!=expected:raise ValueError('Local Phi model/tokenizer changed')
    for path,expected in plan['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=expected or live and sha(path)!=expected:
            raise ValueError('Producing source/archive changed')
    items=read(root/'items.json')['items'];pilot=read(root/'pilot_items.json')['items']
    if (len(items)!=plan['accepted_count'] or population(items)!=plan['population'] or
            pilot_indices(items)!=plan['pilot_indices'] or population(pilot)!=plan['pilot_population']):
        raise ValueError('Phi population/pilot selection differs from frozen plan')
    with (root/'pilot_responses.jsonl').open() as stream:responses=[json.loads(line) for line in stream]
    if (len(responses)!=len(pilot) or any(r['example_id']!=c['identity'] or r['split']!=c['split'] or
            r['prompt_ids']!=c['input_ids'] for r,c in zip(responses,pilot))):
        raise ValueError('Pilot response identity/order/prompt mismatch')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','verify'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
    args=parser.parse_args()
    value=prepare(args.root,args.source_root,args.model_path) if args.phase=='prepare' else verify(args.root)
    print(json.dumps(value,sort_keys=True,allow_nan=False))
