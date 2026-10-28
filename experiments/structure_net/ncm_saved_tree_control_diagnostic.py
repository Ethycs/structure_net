"""Post hoc replay of stored TRAIN code and decoder errors; no model execution."""
import hashlib
import json
from pathlib import Path

BASE=Path('data/experiments/ncm_paper_regime')
TREE=BASE/'20261027_source_hierarchy_tree_train_v1'
RENDER=BASE/'20261028_source_hierarchy_saved_TRAIN_outputs_v1'
TARGETS=BASE/'20261027_expanded_teacher_codes_full_v3/cases.json'
ROOT=BASE/'20261028_saved_tree_TRAIN_code_decoder_diagnostic_v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('saved evidence changed: '+path)


def run():
    if ROOT.exists():raise ValueError('fresh descriptive diagnostic root required')
    receipt=read(RENDER/'actual_tool_receipt.json');verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code']!=0 or receipt['artifacts'].get(str(RENDER/'result.json'))!=sha(RENDER/'result.json'):
        raise ValueError('actual completed saved-output rendering required')
    rendered=read(RENDER/'result.json');verify(rendered['artifacts'])
    paths=[Path(__file__).resolve(),RENDER/'result.json',RENDER/'actual_tool_receipt.json',
           TREE/'control_ids.json',TREE/'initial_outputs.json',TREE/'final_outputs.json',TARGETS]
    proof={'stage':'posthoc_saved_TRAIN_control_replay','selected_after_inspecting_saved_control_outputs':True,
           'inputs':{str(p):sha(p) for p in paths},'new_model_calls':0,'new_judge_calls':0,
           'task_accuracy':None,'heldout_generalization_measured':False,'final_test_opened':False}
    ROOT.mkdir();write(ROOT/'producer.json',proof)
    ids=read(TREE/'control_ids.json');initial=read(TREE/'initial_outputs.json');final=read(TREE/'final_outputs.json')
    targets={r['id']:r for r in read(TARGETS)}
    if len(ids)!=4 or [r['id'] for r in rendered['rows']]!=ids:
        raise ValueError('the four original controls and rendered order must match')
    rows=[]
    for key,before,after,texts in zip(ids,initial,final,rendered['rows'],strict=True):
        gold=targets[key];codes=gold['oracle_codes'];words=gold['native_word_ids']
        if (any(len(x)!=641 for x in (codes,before['codes'],after['codes']))
                or any(len(x)!=128 for x in (words,before['raw_word_ids'],after['raw_word_ids']))):
            raise ValueError('complete native code and word geometry required')
        lexical=[i for i,t in enumerate(words) if t not in (0,101)]
        rows.append({'id':key,'code_positions':641,
            'initial_exact_codes':sum(a==b for a,b in zip(before['codes'],codes,strict=True)),
            'trained_exact_codes':sum(a==b for a,b in zip(after['codes'],codes,strict=True)),
            'changed_code_positions':sum(a!=b for a,b in zip(before['codes'],after['codes'],strict=True)),
            'changed_raw_word_positions':[i for i,(a,b) in enumerate(zip(before['raw_word_ids'],after['raw_word_ids'],strict=True)) if a!=b],
            'lexical_positions':len(lexical),
            'trained_lexical_error_positions':[i for i in lexical if after['raw_word_ids'][i]!=words[i]],
            'stored_text_unchanged':texts['initial']==texts['trained_tree']})
    verify(proof['inputs'])
    value={'status':'posthoc_saved_TRAIN_codes_and_word_errors_replayed','cases':4,'rows':rows,
        'all_trained_codes_equal_targets':all(r['trained_exact_codes']==641 for r in rows),
        'initial_exact_codes':sum(r['initial_exact_codes'] for r in rows),'total_codes':2564,
        'trained_lexical_errors':sum(len(r['trained_lexical_error_positions']) for r in rows),
        'lexical_positions':sum(r['lexical_positions'] for r in rows),
        'new_model_calls':0,'new_judge_calls':0,'task_accuracy':None,
        'heldout_generalization_measured':False,'final_test_opened':False,
        'artifacts':{str(ROOT/'producer.json'):sha(ROOT/'producer.json')}}
    write(ROOT/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('rows','artifacts')}))


if __name__=='__main__':run()
