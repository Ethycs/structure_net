"""Replay complete fixed64 oracle bindings and both native word renderings."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_expanded_codec_component as run

DEST=run.ROOT/'20261024_expanded_codec64_audit_v1'


def main():
    run.base.configure()
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW/'actual_tool_receipt.json')['actual_tool_exit_code']!=0:
        raise ValueError('fresh CPU audit after actual complete numeric inference required')
    run.native.vendor.verify(run.q.read(run.WORKFLOW/'producer.json'))
    for root in (run.PREP,run.PILOT,run.OUTPUTS):
        run.native.vendor.terminal(root)
    whole=run.q.read(run.WORKFLOW/'result.json')
    if [r['stage'] for r in whole['stages']]!=['prepare','pilot','generate'] or any(r['exit_code']!=0 for r in whole['stages']):
        raise ValueError('actual complete numeric lifecycle required')
    rows=run.cases()
    if rows!=run.q.read(run.PREP/'cases.json'):
        raise ValueError('all original64 sources and native codes required')
    codec,expanded,state=run.load_decoders('cpu')
    hashes={'codec':run.native.vendor.module_hash(codec),'expanded':run.native.vendor.module_hash(expanded)}
    del codec,expanded
    values=run.q.read(run.OUTPUTS/'outputs.json')
    if [r['id'] for r in values]!=[r['id'] for r in rows]:
        raise ValueError('every ordered64 output required')
    by_id={r['id']:r for r in values}
    if any(by_id[r['id']]!=r for r in run.q.read(run.PILOT/'outputs.json')):
        raise ValueError('retained pilot first changed')
    for root,size in ((run.PILOT,2),(run.OUTPUTS,64)):
        result,producer=run.q.read(root/'result.json'),run.q.read(root/'producer.json')
        if (result['cases']!=size or result['model_hashes']!=hashes or not result['models_unchanged']
                or result['checkpoint_sha256']!=state['checkpoint_sha256']
                or result['outputs_sha256']!=run.file_hash(root/'outputs.json')
                or result['peak_allocated_bytes']>1_500_000_000
                or result['component_semantic_accuracy'] is not None or result['student_task_accuracy'] is not None
                or result['source_prediction'] or result['final_test_opened']
                or producer['config']['precision']!='FP32' or producer['config']['TF32']
                or producer['config']['physical_GPU']!=1):
            raise ValueError('complete trained state or native inference boundary differs')
    original={r['id']:r for r in run.q.read(run.base.OUTPUTS/'outputs.json')}
    tokenizer=run.native.tokenizer()
    counts={arm:{'correct_active_word_IDs':0,'active_word_IDs':0,'exact_active_frames':0,'missing_EOS':0} for arm in run.ARMS}
    changed=0
    for row,value in zip(rows,values):
        if (value['source_sha256']!=row['source_sha256'] or value['codes']!=row['oracle_codes']
                or set(value['conditions'])!=set(run.ARMS)
                or value['conditions']['domain']!=original[row['id']]['conditions']['domain']):
            raise ValueError('same original source, codes and baseline outputs required')
        length=sum(v!=0 for v in row['native_word_ids'])
        for arm,output in value['conditions'].items():
            raw=output['raw_word_ids']
            if len(raw)!=128 or any(type(v) is not int or not 0<=v<30528 for v in raw):
                raise ValueError('all128 native full-vocabulary IDs required')
            text,ends,bos=run.base.framed.render([raw],tokenizer)
            if output!={'text':text,'raw_word_ids':raw,'predicted_BOS':bos[0],'predicted_EOS_position':ends[0],'missing_EOS':ends[0]==128}:
                raise ValueError('native complete rendering was edited or repaired')
            counts[arm]['correct_active_word_IDs']+=sum(a==b for a,b in zip(raw[:length],row['native_word_ids'][:length]))
            counts[arm]['active_word_IDs']+=length
            counts[arm]['exact_active_frames']+=raw[:length]==row['native_word_ids'][:length]
            counts[arm]['missing_EOS']+=output['missing_EOS']
        changed+=value['conditions']['domain']['raw_word_ids']!=value['conditions']['expanded']['raw_word_ids']
    artifacts={str(Path(__file__)):run.file_hash(__file__)}
    for root in (run.WORKFLOW,run.PREP,run.PILOT,run.OUTPUTS):
        for path in root.rglob('*.json'):
            if 'producer_source' not in path.parts:
                artifacts[str(path)]=run.file_hash(path)
    DEST.mkdir()
    result={'status':'all64_expanded_decoder_state_code_baseline_and_rendering_replayed_semantics_pending',
        'cases':64,'model_hashes':hashes,'checkpoint_sha256':state['checkpoint_sha256'],
        'word_ID_diagnostics_only':counts,'raw_outputs_changed':changed,
        'oracle_target_codes':True,'source_prediction':False,'student_task_accuracy':None,
        'component_semantic_accuracy':None,'final_test_opened':False,'speed_tested':False,'artifacts':artifacts}
    run.write_json(DEST/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    main()
