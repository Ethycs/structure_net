"""Replay matched TRAIN evidence and independently check residual target labels."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_onpolicy_residual_training as study

ROOTS = {(a,s):study.broad.runtime.ROOT/f'20261025_onpolicy_residual_{a}_{s}_audit_v1'
         for a in study.objective.ARMS for s in ('train','label')}
LEVELS = (1,8,16,24,32,40,52,64,76,92,108,128)


def replay_trace(rows, trace, arm, config=study.CONFIG):
    if arm not in study.objective.ARMS or [r['step'] for r in trace] != list(range(1,config.updates+1)):
        raise ValueError('fixed arm and complete update sequence required')
    generator = torch.Generator().manual_seed(config.sampler_seed)
    permutation = []
    exposures = {r['id']:0 for r in rows}
    if len(exposures) != len(rows): raise ValueError('unique TRAIN cases required')
    policy_rows, errors, changes = [],[0]*12,[0]*12
    for record in trace:
        if (record['arm'] != arm or record['updates'] != config.updates
                or record['lr'] != study.broad.rate(record['step'],config)
                or len(record['metrics']) != config.accumulation
                or not math.isfinite(record['gradient_norm'])
                or record['peak_bytes'] > config.peak_limit_bytes
                or record['mean_CE'] != sum(m['mean_CE'] for m in record['metrics'])/config.accumulation):
            raise ValueError('schedule, memory or update-loss contract differs')
        for micro,metric in enumerate(record['metrics']):
            if not permutation: permutation = torch.randperm(len(rows),generator=generator).tolist()
            row = rows[permutation.pop()]
            if row['split'] != 'TRAIN' or metric['id'] != row['id'] or metric['onpolicy'] != bool(micro%2):
                raise ValueError('exact TRAIN sampling/history-mode stream differs')
            losses = metric['level_CE']
            if (len(losses) != 12 or any(not math.isfinite(x) or x < 0 for x in losses)
                    or not math.isclose(metric['mean_CE'],sum(losses)/12,rel_tol=2e-6,abs_tol=1e-6)):
                raise ValueError('complete finite equal-level objective required')
            gold = row['oracle_codes']
            if metric['onpolicy']:
                history,targets = metric['history_codes'],metric['target_codes']
                if (not isinstance(history,list) or not isinstance(targets,list)
                        or len(history) != 641 or len(targets) != 641
                        or any(type(x) is not int or not 0 <= x < 8192 for x in history+targets)
                        or metric['history_errors'] != sum(a != b for a,b in zip(history,gold))
                        or metric['target_changes'] != sum(a != b for a,b in zip(targets,gold))
                        or arm == 'fixed_targets' and targets != gold):
                    raise ValueError('recorded histories, labels or raw disagreement counts differ')
                offset = 0
                for i,length in enumerate(LEVELS):
                    errors[i] += sum(a != b for a,b in zip(history[offset:offset+length],gold[offset:offset+length]))
                    changes[i] += sum(a != b for a,b in zip(targets[offset:offset+length],gold[offset:offset+length]))
                    offset += length
                policy_rows.append((row,history,targets))
            elif (metric['history_codes'] is not None or metric['target_codes'] is not None
                    or metric['history_errors'] != 0 or metric['target_changes'] != 0):
                raise ValueError('clean-history anchor changed')
            exposures[row['id']] += 1
    return {'exposures':exposures,'permutation':permutation,'sampler_rng':generator.get_state(),
            'policy_rows':policy_rows,'history_errors_by_level':errors,'target_changes_by_level':changes}


def replay_outputs(rows, outputs, training, tokenizer, shuffle=False):
    if [r['id'] for r in rows] != [r['id'] for r in outputs]:
        raise ValueError('complete ordered diagnostic population required')
    for i,(row,out) in enumerate(zip(rows,outputs)):
        expected_source = rows[(i+1)%len(rows)]['id'] if shuffle else row['id']
        words,codes,gold = out['raw_word_ids'],out['codes'],row['oracle_codes']
        if (out['source_id_used'] != expected_source or len(words) != 128 or len(codes) != 641
                or any(type(x) is not int or not 0 <= x < 30528 for x in words)
                or any(type(x) is not int or not 0 <= x < 8192 for x in codes)
                or out['active_words'] != sum(v != 0 for v in row['native_word_ids'])
                or out['active_word_matches'] != sum(a == b for a,b in zip(words,row['native_word_ids']) if b != 0)
                or out['task_accuracy'] is not None):
            raise ValueError('raw source/frame/code diagnostic contract differs')
        if [r['positions'] for r in out['levels']] != list(LEVELS):
            raise ValueError('complete native hierarchy required')
        offset = 0
        for value,length in zip(out['levels'],LEVELS):
            if (not math.isfinite(value['CE']) or value['CE'] < 0
                    or not 0 <= value['teacher_correct'] <= length
                    or value['rollout_correct'] != sum(a == b for a,b in zip(codes[offset:offset+length],gold[offset:offset+length]))):
                raise ValueError('rollout agreement or recorded teacher metric differs')
            offset += length
        if any(out[k] != v for k,v in training.preparation.rendering.decode(words,tokenizer).items()):
            raise ValueError('raw diagnostic rendering differs')


def validate_audit(arm, training):
    root = ROOTS[arm,'train']
    if training.q.read(root/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed state audit required')
    value = training.q.read(root/'campaign_results.json')
    if value['status'] != 'onpolicy_training_state_and_trace_replayed' or value['arm'] != arm:
        raise ValueError('matching state audit required')
    for path,expected in value['artifacts'].items():
        if study.broad.file_hash(path) != expected: raise ValueError('audited artifact changed:'+path)
    return value


def state_audit(arm, training, environment):
    root,dest = study.ROOTS[arm,'train'],ROOTS[arm,'train']
    if dest.exists(): raise ValueError('fresh append-only state audit required')
    training.preparation.completed(root)
    q = training.q
    result,proof = [q.read(root/n) for n in ('result.json','producer.json')]
    if (proof['environment'] != environment or proof['config'] != asdict(study.CONFIG)
            or proof['arm'] != arm or result['arm'] != arm
            or proof['fingerprint'] != q.digest({k:v for k,v in proof.items() if k != 'fingerprint'})):
        raise ValueError('matched runtime, arm and scientific fingerprint required')
    rows = q.read(study.broad.RUNS['prepare']/'cases.json')
    if len(rows) != 1376: raise ValueError('all1376 TRAIN cases required')
    ids = q.read(study.broad.RUNS['prepare']/'selection.json')['diagnostic_ids']
    lookup = {r['id']:r for r in rows}
    controls = [lookup[i] for i in ids]
    tokenizer = training.native.tokenizer()
    output,initial,shuffled = [q.read(root/n) for n in ('outputs.json','initial_outputs.json','shuffled_source_outputs.json')]
    for selected,values,shift,key in ((rows,output,False,'final_TRAIN'),
            (controls,initial,False,'initial_controls'),(controls,shuffled,True,'shuffled_source16')):
        replay_outputs(selected,values,training,tokenizer,shift)
        if training.prior.summarize(values) != result[key]: raise ValueError('diagnostic aggregate differs')
    parent = {r['id']:r for r in q.read(study.PARENT/'outputs.json')}
    if initial != [parent[i] for i in ids]: raise ValueError('baseline outputs differ from initializer')
    trace = [json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    stream = replay_trace(rows,trace,arm)
    original = torch.load(study.PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    saved = torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(original['model']) != set(saved['model']): raise ValueError('architecture changed')
    changed,frozen = [],0
    for name,value in saved['model'].items():
        equal = torch.equal(value,original['model'][name])
        if not name.startswith(('concepts.generator.','concepts.source_memory.','concepts.adapters.')):
            if not equal: raise ValueError('protected tensor changed:'+name)
            frozen += 1
        elif not equal: changed.append(name)
    if (not changed or saved['step'] != study.CONFIG.updates or result['updates'] != study.CONFIG.updates
            or saved['arm'] != arm or saved['producer_fingerprint'] != proof['fingerprint']
            or saved['permutation'] != stream['permutation']
            or not torch.equal(saved['sampler_rng'],stream['sampler_rng'])
            or any(float(v['step']) != study.CONFIG.updates for v in saved['optimizer']['state'].values())
            or not result['strict_reload_passed'] or not result['oracle_unchanged']
            or study.broad.file_hash(root/'model.pt') != result['checkpoint_sha256']
            or result['feature_cache']['peak_resident_bytes'] > study.CONFIG.host_feature_cache_bytes):
        raise ValueError('optimizer, sampler, frozen state or checkpoint integrity differs')
    memory = [json.loads(line) for line in (root/'memory.jsonl').read_text().splitlines()]
    if (max(r['peak_allocated_bytes'] for r in memory) != result['peak_allocated_bytes']
            or result['peak_allocated_bytes'] > study.CONFIG.peak_limit_bytes):
        raise ValueError('whole-lifecycle memory contract differs')
    oracle = {r['id']:r for r in q.read(study.broad.RUNS['prepare']/'oracle_outputs.json')}
    if q.read(root/'oracle_outputs.json') != [oracle[i] for i in ids]:
        raise ValueError('clean reconstruction outputs changed')
    paths = [Path(__file__),*[root/n for n in ('producer.json','result.json','actual_tool_receipt.json',
        'model.pt','trace.jsonl','outputs.json','initial_outputs.json','shuffled_source_outputs.json',
        'oracle_outputs.json','checkpoint_ready.json','memory.jsonl')]]
    value = {'status':'onpolicy_training_state_and_trace_replayed','arm':arm,
        'updates':study.CONFIG.updates,'cases':1376,'presentations':sum(stream['exposures'].values()),
        'onpolicy_presentations':len(stream['policy_rows']),
        'exposures_min':min(stream['exposures'].values()),'exposures_max':max(stream['exposures'].values()),
        'history_errors_by_level':stream['history_errors_by_level'],
        'target_changes_by_level':stream['target_changes_by_level'],
        'frozen_tensors_identical':frozen,'changed_trainable_tensors':changed,
        'oracle_residual_label_replay_required':arm == 'residual_targets',
        'historical_policy_rollouts_independently_reexecuted':False,
        'teacher_forced_logits_independently_reexecuted':False,
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):study.broad.file_hash(p) for p in paths}}
    dest.mkdir();study.broad.write_json(dest/'campaign_results.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','changed_trainable_tensors')}),flush=True)


@torch.no_grad()
def check_nearest_labels(vq, latent, history, targets):
    """Check the native nearest-code condition at each recorded policy state."""
    residual = latent.float().clone()
    offset = 0
    for i,length in enumerate(vq.levels):
        remaining = vq._down_for_quant(residual,i,length) if i < len(vq.levels)-1 else residual
        distances = vq._distances(remaining.reshape(-1,vq.embed_dim),vq.embeddings[i].weight)
        expected = distances.argmin(-1).reshape(len(history),length)
        if not torch.isfinite(distances).all() or not torch.equal(expected,targets[:,offset:offset+length]):
            raise ValueError('recorded target is not the native nearest residual code')
        value = vq.embeddings[i](history[:,offset:offset+length])
        if i < len(vq.levels)-1: value = vq._up_to_full(value,i,vq.levels[-1])
        residual -= vq._apply_quant_resi(value,i)
        offset += length
    if offset != history.shape[1] or targets.shape != history.shape:
        raise ValueError('complete historical and target hierarchies required')


def label_audit(arm, training):
    if arm != 'residual_targets': raise ValueError('fixed labels are already replayed against gold by the state audit')
    dest = ROOTS[arm,'label']
    if dest.exists(): raise ValueError('fresh append-only label audit required')
    validate_audit(arm,training)
    study.broad.offload.require_rtx2060()
    model = study.load_initial(training).eval().requires_grad_(False)
    rows = training.q.read(study.broad.RUNS['prepare']/'cases.json')
    root = study.ROOTS[arm,'train']
    trace = [json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    stream = replay_trace(rows,trace,arm)
    for i,(row,h,t) in enumerate(stream['policy_rows'],1):
        words = torch.tensor([row['native_word_ids']],device='cuda',dtype=torch.long)
        latent = model.codec.encode_z(words)
        if model.codec.vq.z_to_idx(latent)[0].cpu().tolist() != row['oracle_codes']:
            raise ValueError('frozen reference encoder binding differs')
        check_nearest_labels(model.codec.vq,latent,
            torch.tensor([h],device='cuda'),torch.tensor([t],device='cuda'))
        if i%128 == 0: print(json.dumps({'residual_histories_checked':i,'total':len(stream['policy_rows'])}),flush=True)
    if training.native.vendor.module_hash(model) != study.MODEL_SHA:
        raise ValueError('reference-only label audit changed frozen model')
    peak = torch.cuda.max_memory_allocated()
    if peak > study.CONFIG.peak_limit_bytes: raise ValueError('label-audit CUDA memory cap exceeded')
    paths = [Path(__file__),ROOTS[arm,'train']/'campaign_results.json',ROOTS[arm,'train']/'actual_tool_receipt.json',
             root/'trace.jsonl',study.broad.RUNS['prepare']/'cases.json',study.PARENT/'model.pt']
    result = {'status':'all_recorded_onpolicy_residual_labels_replayed','arm':arm,
        'histories':len(stream['policy_rows']),'code_positions':len(stream['policy_rows'])*641,
        'reference_latent_from_TRAIN_only':True,'frozen_model_unchanged':True,
        'peak_allocated_bytes':peak,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(p):study.broad.file_hash(p) for p in paths}}
    dest.mkdir();study.broad.write_json(dest/'campaign_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}),flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('arm',choices=study.objective.ARMS)
    parser.add_argument('stage',choices=('state','labels'))
    args = parser.parse_args()
    training,environment = study.broad.runtime.configure()
    state_audit(args.arm,training,environment) if args.stage == 'state' else label_audit(args.arm,training)
