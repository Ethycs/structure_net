"""Independent CPU replay of all128 new source-only output bindings."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_level_conditioning_numeric as run

DEST=run.ROOT/'20261024_level_conditioning_numeric_audit_v1'


def main():
    run.native.vendor.corpus.quality.dependencies()
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('fresh CPU replay after actual complete numeric inference required')
    run.native.vendor.verify(run.q.read(run.WORKFLOW/'producer.json'))
    for root in (run.PREP,run.PILOT,run.OUTPUTS):
        run.native.vendor.terminal(root)
    whole=run.q.read(run.WORKFLOW/'result.json')
    if [r['stage'] for r in whole['stages']]!=['prepare','pilot','generate'] or any(r['exit_code']!=0 for r in whole['stages']):
        raise ValueError('all target-free numeric stages required')
    cases,manifest,proof=run.inputs()
    if run.q.read(run.PREP/'cases.json')!=cases or run.q.read(run.PREP/'split_proof.json')!=proof:
        raise ValueError('same original128 target-free sources and training split required')
    state,saved=run.checkpoint.checkpoint_contract()
    model=run.training.lifecycle.mechanism.enable(run.training.lifecycle.head.from_domain(device='cpu'))
    model.load_state_dict(saved['model'],strict=True)
    run.checkpoint.model_contract(model,state)
    model_hash=run.native.vendor.module_hash(model)
    model.close()
    del model,saved
    rows=[r for name in run.source.COHORTS for r in cases[name]]
    values=run.q.read(run.OUTPUTS/'outputs.json')
    pilot=run.q.read(run.PILOT/'outputs.json')
    if [r['id'] for r in values]!=[r['id'] for r in rows]:
        raise ValueError('all128 ordered predictions required')
    by_id={r['id']:r for r in values}
    if any(by_id[r['id']]!=r for r in pilot):
        raise ValueError('a first pilot prediction changed')
    expected_probe_ids=[r['id'] for r in run.source.inference_probes(cases,manifest)]
    if [r['id'] for r in pilot]!=expected_probe_ids or run.q.read(run.PREP/'selection.json')['pilot_ids']!=expected_probe_ids:
        raise ValueError('same first/longest inference pilot required')
    for root,size in ((run.PILOT,2),(run.OUTPUTS,128)):
        result,producer=run.q.read(root/'result.json'),run.q.read(root/'producer.json')
        if (result['cases']!=size or result['checkpoint_sha256']!=state['checkpoint_sha256']
                or result['model_sha256']!=model_hash or not result['model_unchanged']
                or result['outputs_sha256']!=run.file_hash(root/'outputs.json')
                or result['peak_allocated_bytes']>3_000_000_000
                or result['source_only_task_accuracy'] is not None or result['oracle_target_codes']
                or not result['source_only'] or not result['semantic_cache_untouched'] or result['final_test_opened']
                or producer['config']['precision']!='FP32' or producer['config']['TF32']
                or producer['config']['physical_GPU']!=2 or producer['config']['external_judgments']
                or producer['config']['target_fields_in_prediction']):
            raise ValueError('strict new model, source-only scope or precision differs')
    tokenizer=run.native.tokenizer()
    diagnostics={name:{'cases':0,'nonempty':0,'missing_EOS':0,'source_positions':0,'distinct_texts':0} for name in run.source.COHORTS}
    texts={name:set() for name in run.source.COHORTS}
    for row,value in zip(rows,values):
        entry=manifest[row['id']]
        if run.file_hash(entry['path'])!=entry['sha256']:
            raise ValueError('a complete source feature payload changed')
        payload=torch.load(entry['path'],map_location='cpu',weights_only=True)
        if (any(payload[k]!=row[k] for k in ('id','source_sha256','alignment_sha256','native_prompt_sha256'))
                or payload['features'].shape!=(entry['source_tokens'],3072)
                or payload['features'].dtype!=torch.float32 or not torch.isfinite(payload['features']).all()):
            raise ValueError('complete FP32 feature identity or source positions differ')
        if len(value['codes'])!=641 or any(type(v) is not int or not 0<=v<8192 for v in value['codes']):
            raise ValueError('complete predicted native hierarchy required')
        decoded=run.source.decode(value['raw_word_ids'],tokenizer)
        expected={'id':row['id'],'source_sha256':row['source_sha256'],'split':row['split'],
            'feature_sha256':entry['sha256'],'source_positions':entry['source_tokens'],
            'codes':value['codes'],'raw_word_ids':value['raw_word_ids'],**decoded}
        if value!=expected:
            raise ValueError('native whole output was altered, repaired or mismatched')
        name=row['split']
        diagnostics[name]['cases']+=1
        diagnostics[name]['nonempty']+=bool(value['text'].strip())
        diagnostics[name]['missing_EOS']+=value['missing_EOS']
        diagnostics[name]['source_positions']+=value['source_positions']
        texts[name].add(value['text'])
    for name in diagnostics:
        diagnostics[name]['distinct_texts']=len(texts[name])
    artifacts={str(Path(__file__)):run.file_hash(__file__)}
    for root in (run.WORKFLOW,run.PREP,run.PILOT,run.OUTPUTS):
        for path in root.rglob('*.json'):
            if 'producer_source' not in path.parts:
                artifacts[str(path)]=run.file_hash(path)
    DEST.mkdir()
    result={'status':'all128_new_source_only_state_feature_code_and_rendering_bindings_replayed_semantics_pending',
        'cases':128,'checkpoint_sha256':state['checkpoint_sha256'],'model_sha256':model_hash,
        'diagnostics_not_task_accuracy':diagnostics,'source_only_task_accuracy':None,
        'semantic_cache_untouched':True,'final_test_opened':False,'speed_tested':False,'artifacts':artifacts}
    run.write_json(DEST/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    main()
