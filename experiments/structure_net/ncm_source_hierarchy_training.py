"""Fixed-budget task training and trace replay for dense and tree source attention."""
import argparse
from dataclasses import asdict,replace
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_source_hierarchy_mechanism as mechanism
from experiments.structure_net import ncm_source_tree_padded_lifecycle as qualified_tree
from experiments.structure_net import ncm_task_supervision_training as task
from experiments.neural_architecture_lab.audit_ncm_public_training import LEVELS

data,pipeline,engine,broad = task.data,task.pipeline,task.engine,task.broad
CONFIG = replace(task.CONFIG,physical_GPU=1)
ARMS = mechanism.ARMS
HYPOTHESIS = 'ncm-source-hierarchy-task-comparison-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-hierarchy-comparison.md')
TEST = Path('tests/structure_net/test_ncm_source_hierarchy_mechanism.py')
AUDIT_TEST = Path('tests/structure_net/test_ncm_source_hierarchy_training.py')
QUALIFICATION = data.ROOT/'20261027_source_hierarchy_CPU_v1.json'
ROOTS = {(a,s):data.ROOT/f'20261027_source_hierarchy_{a}_{s}_v1' for a in ARMS for s in ('pilot','train','audit')}
STORAGE = Path('/data2/structure_net/ncm_source_hierarchy_20261027_v1')


def require(root):
    receipt = data.read(root/'actual_tool_receipt.json');data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0: raise ValueError('actual successful execution required')
    value = data.read(root/'result.json')
    if receipt['artifacts'].get(str(root/'result.json')) != data.sha(root/'result.json'): raise ValueError('receipt/result binding differs')
    if 'artifacts' in value:data.verify(value['artifacts'])
    return value


def inputs(arm,stage):
    qualified = data.read(QUALIFICATION);data.verify(qualified['artifacts'])
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 5: raise ValueError('CPU mechanism and replay contracts required')
    tree = require(qualified_tree.ROOT)
    if not all(tree[k] for k in ('FP32_clean_and_noisy_gradient_parity_passed','FP16_full_accumulation_updates_completed',
                                'strict_resumed_update_passed','protected_tensors_unchanged')):
        raise ValueError('completed corrected tree lifecycle required')
    paths = [QUALIFICATION,*[qualified_tree.ROOT/n for n in ('producer.json','result.json','actual_tool_receipt.json','initial_outputs.json','control_ids.json')]]
    rows,index,manifest,other = task.dataset('existing');paths += other+task.require_parent()
    if stage == 'train':
        dense = require(ROOTS['dense','pilot'])
        if not dense['strict_resumed_update_passed'] or dense['cases'] != 1376: raise ValueError('dense lifecycle required')
        paths += [ROOTS['dense','pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','initial_outputs.json')]
        if arm == 'dense':
            audited = require(ROOTS['tree','audit'])
            if audited['status'] != 'source_hierarchy_training_replayed': raise ValueError('fixed tree-then-dense training order required')
            paths += [ROOTS['tree','audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    return rows,index,manifest,paths


def freeze(root,arm,stage,environment,paths,training):
    files = {Path(__file__).resolve(),Path(mechanism.__file__).resolve(),PLAN.resolve(),TEST.resolve(),AUDIT_TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or
                    p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths += [Path('pixi.lock'),Path('pyproject.toml')]
    proof = {'schema':'nal.ncm-source-hierarchy-task-training.v1','hypothesis_id':HYPOTHESIS,'arm':arm,'stage':stage,
        'environment':environment,'config':asdict(CONFIG),'cases':1376,'trainable_parameters':mechanism.PARAMETERS[arm],
        'parent_checkpoint_sha256':task.schedule.PARENT_CHECKPOINT_SHA,'parent_model_sha256':task.PARENT_MODEL_SHA,
        'systems_lifecycle_only_not_quality_evidence':stage == 'pilot',
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in paths}}
    proof['fingerprint'] = task.base.fingerprint(proof)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def load_initial(training,arm):
    model = task.load_initial(training);metadata = mechanism.install(model,arm)
    broad.mechanism.configure(model,False)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != mechanism.PARAMETERS[arm]:
        raise ValueError('complete declared trainable population required')
    return model,metadata


def run(arm,stage):
    if stage == 'pilot' and arm != 'dense': raise ValueError('reuse successful tree lifecycle v3')
    root,storage = ROOTS[arm,stage],STORAGE/arm/stage
    if root.exists() or storage.exists(): raise ValueError('fresh append-only attempt required')
    training,environment = broad.runtime.configure();pipeline.determinism.configure_runtime()
    broad.offload.require_rtx2060()
    if torch.cuda.mem_get_info()[0] < CONFIG.peak_limit_bytes: raise ValueError('RTX2060 lifecycle headroom required')
    before_services = qualified_tree.original.prior.services()
    rows,index,manifest,paths = inputs(arm,stage)
    sources = task.Sources(rows,index,manifest);sampler = task.schedule.Schedule(len(rows),CONFIG)
    ids = task.control_ids(rows,index);lookup = {r['id']:r for r in rows}
    proof = freeze(root,arm,stage,environment,paths,training);storage.mkdir(parents=True)
    data.write(root/'control_ids.json',ids)
    torch.cuda.reset_peak_memory_stats();model,metadata = load_initial(training,arm)
    values = [sources.cache.get(lookup[k]) for k in ids];codes = [lookup[k]['oracle_codes'] for k in ids]
    initial = pipeline.lifecycle.outputs(model,values)
    # All added branches have zero outputs; both arms must match the original controls.
    expected = data.read(qualified_tree.ROOT/'initial_outputs.json')
    if ids != data.read(qualified_tree.ROOT/'control_ids.json') or initial != expected: raise ValueError('same controls and initial function required')
    data.write(root/'initial_outputs.json',initial);data.write(root/'architecture.json',metadata)
    frozen = training.prior.protected(model);initializer = training.native.vendor.module_hash(model)
    if stage == 'pilot':
        data.write(root/'stream_parity.json',task.stream_parity(sources,sampler))
        previous = pipeline.lifecycle.ROOT;pipeline.lifecycle.ROOT = root
        try:
            for noisy in (False,True):pipeline.lifecycle.parity(model,values,codes,noisy)
        finally:pipeline.lifecycle.ROOT = previous
    if training.native.vendor.module_hash(model) != initializer: raise ValueError('pre-update checks changed weights')
    broad.mechanism.configure(model,True);opt,scaler = task.optimizer_for(model)
    if sum(p.numel() for p in opt.param_groups[0]['params']) != mechanism.PARAMETERS[arm]: raise ValueError('optimizer population differs')
    torch.manual_seed(CONFIG.dropout_seed);torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    updates = 2 if stage == 'pilot' else CONFIG.updates
    batches = task.schedule.shared.prefetch(sampler.batches(stop_update=updates),sources.fetch,capacity=CONFIG.prefetch_batches)
    started = time.monotonic();checkpoints = []
    try:
        with (root/'trace.jsonl').open('x') as trace:
            for step in range(1,updates+1):
                record = engine.update(model,batches,opt,scaler,step=step,config=CONFIG)
                broad.offload.assert_cpu_state(opt)
                record['elapsed_seconds'] = time.monotonic()-started
                record['peak_allocated_bytes'] = task.memory(root,'step_'+str(step))
                trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
                if stage == 'pilot' or step%CONFIG.recovery_interval == 0 or step == updates:
                    if training.prior.protected(model) != frozen: raise ValueError('protected tensors changed')
                    path = storage/f'step{step}.pt'
                    engine.save(path,model,opt,scaler,step=step,fingerprint=proof['fingerprint'],config=CONFIG)
                    checkpoints.append({'path':str(path),'sha256':data.sha(path),'completed_updates':step})
                if stage == 'pilot' or step%16 == 0:print(json.dumps({k:v for k,v in record.items() if k!='microbatches'}),flush=True)
    finally:batches.close()
    final_outputs = pipeline.lifecycle.outputs(model,values);data.write(root/'final_outputs.json',final_outputs)
    if stage == 'pilot':
        first = torch.load(storage/'step1.pt',map_location='cpu',weights_only=True,mmap=True)
        engine.restore(first,model,opt,scaler,fingerprint=proof['fingerprint'],config=CONFIG)
        replay_batches = ((i,n,refs,sources.fetch(refs)) for i,n,refs in sampler.batches(1,2))
        replay = engine.update(model,replay_batches,opt,scaler,step=2,config=CONFIG)
        data.write(root/'resumed_update_2.json',replay)
        if replay['microbatches'] != record['microbatches'] or pipeline.lifecycle.outputs(model,values) != final_outputs:
            raise ValueError('resumed update/output differs')
        del first
    final = Path(checkpoints[-1]['path']);saved = torch.load(final,map_location='cpu',weights_only=True,mmap=True)
    current = engine.snapshot(model,opt,scaler,step=updates,fingerprint=proof['fingerprint'],config=CONFIG)
    same = training.preparation.checkpoint.training.lifecycle.same
    if any(not same(current[k],v) for k,v in saved.items()): raise ValueError('model/optimizer/scaler/RNG differs from checkpoint')
    del current
    before_reload = training.native.vendor.module_hash(model)
    if engine.restore(saved,model,opt,scaler,fingerprint=proof['fingerprint'],config=CONFIG) != updates or training.native.vendor.module_hash(model) != before_reload:
        raise ValueError('strict final reload differs')
    if training.prior.protected(model) != frozen: raise ValueError('protected state changed')
    peak = task.memory(root,'final_verified');counts = task.verify_trace(root/'trace.jsonl',rows,CONFIG,expected_updates=updates)
    data.write(root/'trace_replay.json',counts);data.write(root/'checkpoints.json',checkpoints)
    (root/'model.pt').symlink_to(final.resolve())
    after_services = qualified_tree.original.prior.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services): raise ValueError('other worker identity changed')
    data.write(root/'services.json',{'before':before_services,'after':after_services})
    task.base.verify_proof(proof)
    artifacts = [p for p in root.iterdir() if p.is_file()]+[Path(r['path']) for r in checkpoints]
    result = {'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,'arm':arm,'stage':stage,
        'status':'source_hierarchy_lifecycle_passed' if stage == 'pilot' else 'source_hierarchy_training_completed',
        'cases':len(rows),'updates':updates,'config':asdict(CONFIG),'trainable_parameters':mechanism.PARAMETERS[arm],
        'strict_resumed_update_passed':stage == 'pilot','strict_final_reload_passed':True,'protected_tensors_unchanged':True,
        'frozen_hashes':frozen,'final_model_sha256':before_reload,'checkpoint_sha256':data.sha(final),
        'peak_allocated_bytes':peak,'elapsed_seconds':time.monotonic()-started,'measured_exposure':counts,
        'feature_cache':sources.cache.metrics(),'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'systems_lifecycle_only_not_quality_evidence':stage == 'pilot','artifacts':{str(p):data.sha(p) for p in artifacts}}
    data.write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
    model.close();del saved,model,opt,scaler;gc.collect();torch.cuda.empty_cache()


def replay_trace(path,rows,updates):
    counts = task.verify_trace(path,rows,CONFIG,expected_updates=updates)
    sampler = task.schedule.Schedule(len(rows),CONFIG);checked = 0
    with Path(path).open() as stream:
        for step,line in enumerate(stream,1):
            record = json.loads(line);refs = sampler.update(step-1)
            for offset,micro in zip(range(0,CONFIG.effective_examples,CONFIG.microbatch),record['microbatches'],strict=True):
                selected = refs[offset:offset+CONFIG.microbatch]
                gold = torch.tensor([rows[e.index]['oracle_codes'] for e in selected],dtype=torch.long)
                _,changes = engine.histories(gold,selected,LEVELS)
                if changes != micro['history_changes']: raise ValueError('recorded code corruption differs')
                checked += len(selected)
            if record['weighted_CE'] != sum(m['weighted_CE'] for m in record['microbatches']): raise ValueError('loss sum differs')
    if checked != updates*CONFIG.effective_examples: raise ValueError('complete update exposure required')
    return counts


def audit(arm):
    root,source = ROOTS[arm,'audit'],ROOTS[arm,'train']
    if root.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('fresh CPU-only audit required')
    training,environment = broad.runtime.configure();pipeline.determinism.configure_runtime()
    result = require(source);proof = data.read(source/'producer.json');task.base.verify_proof(proof)
    parent = data.read(task.PARENT/'result.json')
    if (result['status'] != 'source_hierarchy_training_completed' or result['arm'] != arm or result['updates'] != CONFIG.updates
            or result['config'] != asdict(CONFIG) or result['frozen_hashes'] != parent['frozen_hashes']
            or not result['strict_final_reload_passed'] or not result['protected_tensors_unchanged']):
        raise ValueError('complete fixed training and unchanged protected tensors required')
    rows,_,_,paths = task.dataset('existing')
    counts = replay_trace(source/'trace.jsonl',rows,CONFIG.updates)
    if counts != result['measured_exposure']: raise ValueError('complete recorded exposure differs')
    if counts['presentations'] != 65536 or counts['noisy_presentations'] != 32768 or counts['distinct_sources_visited'] != 1376:
        raise ValueError('complete same-case exposure required')
    checkpoint = source/'model.pt'
    if data.sha(checkpoint) != result['checkpoint_sha256']: raise ValueError('final checkpoint bytes differ')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if saved['completed_updates'] != CONFIG.updates or saved['producer_fingerprint'] != proof['fingerprint'] or saved['curriculum'] != asdict(CONFIG):
        raise ValueError('checkpoint metadata differs')
    paths += [source/n for n in ('producer.json','result.json','actual_tool_receipt.json','trace.jsonl','model.pt')]
    audit_proof = freeze(root,arm,'audit',environment,paths,training)
    task.base.verify_proof(audit_proof)
    value = {'schema':audit_proof['schema'],'hypothesis_id':HYPOTHESIS,'arm':arm,'status':'source_hierarchy_training_replayed',
        **counts,'all_corruption_counts_replayed':True,'checkpoint_sha256':result['checkpoint_sha256'],
        'task_accuracy':None,'final_test_opened':False,'artifacts':{str(root/'producer.json'):data.sha(root/'producer.json')}}
    data.write(root/'result.json',value);print(json.dumps(value),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','train','audit'))
    parser.add_argument('--arm',choices=ARMS,required=True);args = parser.parse_args()
    root = ROOTS[args.arm,args.stage];existed = root.exists()
    try:audit(args.arm) if args.stage == 'audit' else run(args.arm,args.stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
