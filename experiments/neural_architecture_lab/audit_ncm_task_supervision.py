"""Replay fixed-arm sample positions, code corruption and committed checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_task_supervision_training as study

data,base = study.data,study.base
LEVELS = (1,8,16,24,32,40,52,64,76,92,108,128)


def replay(records,rows,start,end,config=study.CONFIG):
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= config.updates:
        raise ValueError('nonempty committed update interval required')
    sampler = study.schedule.Schedule(len(rows),config);stream = iter(records)
    visits = [0]*len(rows);noisy = 0;previous_scale = None;digest = hashlib.sha256()
    for step in range(start+1,end+1):
        try:value = next(stream)
        except StopIteration as error:raise ValueError('missing completed update') from error
        refs = sampler.update(step-1)
        if (value['step'] != step or value['epoch'] != 1 or value['examples'] != len(refs)
                or value['lr'] != study.schedule.shared.rate(step,config)
                or len(value['microbatches']) != config.effective_examples//config.microbatch
                or any(not math.isfinite(value[k]) or value[k] < 0 for k in ('weighted_CE','gradient_norm'))
                or not math.isfinite(value['scale']) or value['scale'] <= 0
                or previous_scale is not None and value['scale'] < previous_scale):
            raise ValueError('update geometry, schedule or finite optimizer statistics differ')
        previous_scale = value['scale']
        for offset,micro in zip(range(0,len(refs),config.microbatch),value['microbatches'],strict=True):
            selected = refs[offset:offset+config.microbatch];bound = [rows[e.index] for e in selected]
            if any(len(r['oracle_codes']) != 641 or any(type(c) is not int or not 0 <= c < 8192
                    for c in r['oracle_codes']) for r in bound):raise ValueError('complete native code labels required')
            gold = torch.tensor([r['oracle_codes'] for r in bound],dtype=torch.long)
            _,changes = study.engine.histories(gold,selected,LEVELS)
            expected = {'ids':[r['id'] for r in bound],'positions':[e.position for e in selected],
                'populations':[e.population for e in selected],'noisy':[e.noisy for e in selected],'history_changes':changes}
            if (any(micro.get(k) != v for k,v in expected.items())
                    or not math.isfinite(micro['weighted_CE']) or micro['weighted_CE'] < 0):
                raise ValueError('case identity, noise stream or corrupted history differs')
            for e in selected:visits[e.index] += 1;noisy += e.noisy
        if value['weighted_CE'] != sum(m['weighted_CE'] for m in value['microbatches']):
            raise ValueError('effective loss differs from its microbatch contributions')
        digest.update(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
    sentinel = object()
    if next(stream,sentinel) is not sentinel:raise ValueError('unaccounted update after committed interval')
    return {'updates':end-start,'presentations':sum(visits),'noisy_presentations':noisy,
        'minimum_visits':min(visits),'maximum_visits':max(visits),'distinct_sources_visited':sum(v>0 for v in visits),
        'sample_order_replayed':True,'recorded_history_changes_replayed':True,'audited_records_sha256':digest.hexdigest()}


def checkpoint_record(record,step,arm,proof):
    path = Path(record['path']);saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (record['completed_updates'] != step or record['arm'] != arm
            or record['producer_fingerprint'] != proof['fingerprint'] or data.sha(path) != record['sha256']
            or data.read(path.with_suffix('.json')) != record
            or saved['completed_updates'] != step or saved['producer_fingerprint'] != proof['fingerprint']
            or saved['curriculum'] != asdict(study.CONFIG)
            or set(saved) != {'model','optimizer','scaler','CPU_rng','CUDA_rng','completed_updates','producer_fingerprint','curriculum'}):
        raise ValueError('exact committed checkpoint, arm, position and training configuration required')
    return saved


def audit(arm,stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():raise ValueError('CPU-only audit required')
    source = study.ROOTS[arm,stage];root = study.ROOTS[arm,'pilot_audit' if stage == 'pilot' else 'audit']
    if root.exists():raise ValueError('fresh append-only audit required')
    result = base.require(source);proof = data.read(source/'producer.json');base.verify_proof(proof)
    limit = 2 if stage == 'pilot' else study.CONFIG.updates
    expected_status = 'task_supervision_lifecycle_passed' if stage == 'pilot' else 'task_supervision_training_completed'
    if (result['status'] != expected_status or result['arm'] != arm or result['stage'] != stage
            or result['config'] != asdict(study.CONFIG) or result['updates'] != limit
            or not result['strict_final_reload_passed'] or not result['protected_tensors_unchanged']
            or not result['fresh_optimizer_masters_match_initial_weights']
            or result['task_accuracy'] is not None or result['new_judge_requests'] != 0
            or result['final_test_opened'] or result['speed_tested']):raise ValueError('complete fixed training stage required')
    rows,index,manifest,paths = study.prerequisites(arm,stage)
    if result['cases'] != len(rows) or data.read(source/'control_ids.json') != study.control_ids(rows,index):
        raise ValueError('complete bound population and actual length extremes required')
    paths += [source/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [Path(p) for p in result['artifacts']]
    training,environment = study.broad.runtime.configure()
    audit_proof = study.freeze(root,arm,'pilot_audit' if stage == 'pilot' else 'audit',training,environment,paths)
    with (source/'trace.jsonl').open() as stream:replayed = replay((json.loads(s) for s in stream),rows,0,limit)
    recorded = data.read(source/'trace_replay.json')
    if any(recorded[k] != replayed[k] for k in ('updates','presentations','noisy_presentations',
            'minimum_visits','maximum_visits','distinct_sources_visited')):raise ValueError('producer visit counts differ')
    lengths = {r['id']:r['source_positions'] for r in index};sampler = study.schedule.Schedule(len(rows))
    if (recorded['source_feature_positions_presented'] != sum(lengths[rows[e.index]['id']]
            for i in range(limit) for e in sampler.update(i))
            or recorded['native_target_positions_presented'] != limit*study.CONFIG.effective_examples*128
            or recorded['target_code_positions_presented'] != limit*study.CONFIG.effective_examples*641
            or result['measured_exposure'] != recorded):raise ValueError('measured presentation exposure differs')
    checkpoints = data.read(source/'checkpoints.json')
    steps = [1,2] if stage == 'pilot' else list(range(study.CONFIG.recovery_interval,limit+1,study.CONFIG.recovery_interval))
    if [r['completed_updates'] for r in checkpoints] != steps:raise ValueError('complete checkpoint schedule required')
    for record,step in zip(checkpoints,steps,strict=True):
        saved = checkpoint_record(record,step,arm,proof)
        if step == limit:
            if record['sha256'] != result['checkpoint_sha256'] or data.sha(source/'model.pt') != record['sha256']:
                raise ValueError('final checkpoint binding differs')
        del saved
    memory = [json.loads(s) for s in (source/'memory.jsonl').read_text().splitlines()]
    if max(r['peak_allocated_bytes'] for r in memory) != result['peak_allocated_bytes'] or result['peak_allocated_bytes'] > study.CONFIG.peak_limit_bytes:
        raise ValueError('complete recorded lifecycle memory bound differs')
    if stage == 'pilot':
        if not all(result[k] for k in ('gradient_parity_passed','strict_resumed_update_passed','source_only_outputs_reproduced')):
            raise ValueError('actual gradient and recovery qualification required')
        for mode in ('clean','noisy'):
            parity = data.read(source/f'parity_{mode}.json')
            if not (parity['passed'] and parity['gradient_presence_equal'] and parity['finite'] and parity['logits_allclose']
                    and parity['relative_gradient_L2_error'] <= 5e-5 and parity['max_abs_gradient_difference'] <= 1e-4
                    and parity['argmax_disagreements'] == 0 and parity['trainable_parameters_compared'] == 105323136):
                raise ValueError('original-reference gradient parity failed')
        equality = data.read(source/'resumed_state_equality.json')
        if set(equality) != {'model','optimizer','scaler','CPU_rng','CUDA_rng','completed_updates','producer_fingerprint','curriculum'} or not all(v is True for v in equality.values()):
            raise ValueError('complete exact restart state required')
        restarted = data.read(source/'resumed_update_2.json');replay([restarted],rows,1,2)
        original = json.loads((source/'trace.jsonl').read_text().splitlines()[-1])
        if restarted['microbatches'] != original['microbatches'] or not data.read(source/'stream_parity.json')['all_records_equal']:
            raise ValueError('exact resumed update and streamed inputs differ')
    data.write(root/'replay.json',replayed);base.verify_proof(audit_proof)
    value = {'schema':study.SCHEMA,'hypothesis_id':study.HYPOTHESIS,
        'status':'task_supervision_lifecycle_replayed' if stage == 'pilot' else 'task_supervision_training_replayed',
        'arm':arm,'stage':stage,'cases':len(rows),'updates':limit,'replay':replayed,
        'source_checkpoint_sha256':result['checkpoint_sha256'],'source_result_sha256':data.sha(source/'result.json'),
        'systems_lifecycle_only_not_quality_evidence':stage == 'pilot','training_updates_added':0,
        'task_accuracy':None,'new_judge_requests':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in (root/'producer.json',root/'replay.json')}}
    data.write(root/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('--arm',choices=study.ARMS,required=True)
    parser.add_argument('--stage',choices=('pilot','train'),required=True);args = parser.parse_args()
    audit(args.arm,args.stage)
