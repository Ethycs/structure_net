"""Matched frozen-body NCM training on existing or expanded reviewed summaries."""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch

from experiments.structure_net import ncm_task_supervision_schedule as schedule
from experiments.structure_net import ncm_expanded_teacher_codes as targets
from experiments.structure_net import ncm_public_training as public

data, base, catalog = targets.data, targets.base, targets.catalog
pipeline, engine, broad = public.pipeline, public.engine, public.pipeline.broad
CONFIG, ARMS = schedule.CONFIG, schedule.ARMS
HYPOTHESIS = schedule.HYPOTHESIS
SCHEMA = 'nal.ncm-task-supervision-training.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-task-supervision-expansion-plan.md')
EXECUTION = Path('docs/07 - Status Reports/2026-10-27_ncm-task-supervision-execution.md')
TEST = Path('tests/structure_net/test_ncm_task_supervision_training.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_task_supervision_service.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_task_supervision.py')
PARENT = public.root_for(2,1)
PARENT_MODEL_SHA = '090ec8c105e47bd2412d4ab9b83411035fd73e2d8f52e575dfb7edea60043f22'
ROOTS = {(a,s):data.ROOT/f'20261027_task_supervision_{a}_{s}_v1' for a in ARMS for s in ('pilot','pilot_audit','train','audit')}
STORAGE = Path('/data2/structure_net/ncm_task_supervision_20261027_v1')


def require_expanded():
    for root,flag in ((catalog.ROOTS['audit'],'training_data_admitted'),
                      (targets.ROOTS['full_audit'],'training_targets_admitted')):
        state = base.require(root);base.verify_proof(data.read(root/'producer.json'))
        if state.get(flag) is not True:raise ValueError('complete expanded catalog and targets must be admitted')
    return [root/n for root in (catalog.ROOTS['prepare'],catalog.ROOTS['audit'],targets.ROOTS['full'],targets.ROOTS['full_audit'])
            for n in ('producer.json','result.json','actual_tool_receipt.json')]


def dataset(arm):
    if arm not in ARMS:raise ValueError('fixed paired arm required')
    paths = catalog.require_old()
    data.completed(broad.RUNS['prepare'])
    if arm == 'existing':
        rows = data.read(broad.RUNS['prepare']/'cases.json');root = broad.catalog.DEST
        if len(rows) != 1376:raise ValueError('entire unchanged existing TRAIN population required')
        paths += [broad.RUNS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','selection.json')]
    else:
        paths += require_expanded();root = catalog.ROOTS['prepare']
        rows = data.read(targets.ROOTS['full']/'cases.json')
        arms = data.read(root/'arm_ids.json')
        if [r['id'] for r in rows] != arms['expanded'] or len(arms['new_admitted']) < catalog.MIN_NEW_FOR_TRAINING:
            raise ValueError('complete fixed expanded arm required')
        if rows[:1376] != data.read(broad.RUNS['prepare']/'cases.json'):
            raise ValueError('all original targets and code hierarchies must remain exact')
        paths += [targets.ROOTS['full']/'cases.json',root/'arm_ids.json']
    index,manifest = [data.read(root/n) for n in ('source_index.json','feature_manifest.json')]
    paths += [root/n for n in ('source_index.json','feature_manifest.json')]
    return rows,index,manifest,paths


class Sources:
    def __init__(self,rows,index,manifest):
        ids = {r['id'] for r in rows}
        if (len(ids) != len(rows) or ids != {r['id'] for r in index} or ids != set(manifest)
                or any(r['split'] != 'TRAIN' or len(r['oracle_codes']) != 641 or len(r['native_word_ids']) != 128 for r in rows)):
            raise ValueError('complete unique TRAIN rows, full hierarchies and source features required')
        self.rows,self.index = rows,index
        self.cache = catalog.FeatureCache(index,manifest,max_bytes=CONFIG.cache_bytes)

    def fetch(self,examples):
        out = []
        for example in examples:
            if example.population != 'anchor' or not 0 <= example.index < len(self.rows):raise ValueError('bound reviewed source index required')
            row = self.rows[example.index]
            out.append({**{k:row[k] for k in ('id','source_sha256','native_word_ids','oracle_codes')},
                'features':self.cache.get(row),'presentation':asdict(example)})
        return out


def control_ids(rows,index):
    lookup = {r['id']:r for r in rows};lengths = {r['id']:r['source_positions'] for r in index}
    if len(lookup) != len(rows) or set(lookup) != set(lengths) or len(rows) < 4:
        raise ValueError('complete population and at least four distinct controls required')
    selected = [rows[0]['id'],min(lengths,key=lambda k:(lengths[k],k)),
        max(lengths,key=lambda k:(lengths[k],k)),max(rows,key=lambda r:(r['native_lexical_tokens'],r['id']))['id']]
    selected = list(dict.fromkeys(selected))
    selected += [k for k in sorted(lookup) if k not in selected][:4-len(selected)]
    return selected


def require_parent():
    result = data.completed(PARENT)
    if (result['status'] != 'public_curriculum_milestone_training_completed' or result['epoch'] != 2
            or result['updates'] != 3384 or not result['strict_reload_passed']
            or result['checkpoint_sha256'] != schedule.PARENT_CHECKPOINT_SHA
            or result['final_model_sha256'] != PARENT_MODEL_SHA
            or data.sha(PARENT/'model.pt') != schedule.PARENT_CHECKPOINT_SHA):
        raise ValueError('exact completed epoch2 model-only initializer required')
    return [PARENT/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]


def prerequisites(arm,stage):
    if arm not in ARMS or stage not in ('pilot','train'):raise ValueError('fixed arm and execution stage required')
    rows,index,manifest,paths = dataset(arm);paths += require_parent()
    if stage == 'train':
        paths += require_expanded()
        for other in ARMS:
            pilot = ROOTS[other,'pilot'];status = base.require(pilot)
            base.verify_proof(data.read(pilot/'producer.json'))
            if not all(status.get(k) is True for k in ('strict_resumed_update_passed','gradient_parity_passed',
                    'protected_tensors_unchanged','source_only_outputs_reproduced','fresh_optimizer_masters_match_initial_weights')):
                raise ValueError('both real complete-population lifecycles required')
            audit = ROOTS[other,'pilot_audit'];replayed = base.require(audit)
            base.verify_proof(data.read(audit/'producer.json'))
            if replayed['status'] != 'task_supervision_lifecycle_replayed':raise ValueError('both lifecycle traces must be independently audited')
            paths += [p/n for p in (pilot,audit) for n in ('producer.json','result.json','actual_tool_receipt.json')]
        if arm == 'expanded':
            before = base.require(ROOTS['existing','audit'])
            base.verify_proof(data.read(ROOTS['existing','audit']/'producer.json'))
            if before['status'] != 'task_supervision_training_replayed':raise ValueError('existing arm must complete and audit before expanded training')
            paths += [ROOTS['existing','audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    return rows,index,manifest,paths


def configure():
    training,environment = broad.runtime.configure();pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('physicalGPU0 RTX3060 required; leave the2060 public run unchanged')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    return training,environment


def load_initial(training):
    model = pipeline.parent.load_initial(training)
    saved = torch.load(PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    if training.native.vendor.module_hash(model) != PARENT_MODEL_SHA:raise ValueError('strict epoch2 tensor load differs')
    model.concepts.teacher_logits = types.MethodType(pipeline.lifecycle.serial.teacher_logits,model.concepts)
    broad.mechanism.configure(model,True)
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 105323136:raise ValueError('fixed generator/concept-source population required')
    return model


def optimizer_for(model):
    # CPUAdamW clones master weights in its constructor: create it after loading epoch2.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = broad.offload.CPUAdamW(params,lr=CONFIG.max_lr,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    if any(not torch.equal(master.detach(),p.detach().cpu()) for p,master in zip(params,opt.masters)):
        raise ValueError('fresh optimizer masters must equal the actual loaded initializer')
    scaler = torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=2000)
    return opt,scaler


def freeze(root,arm,stage,training,environment,paths):
    files = {Path(__file__).resolve(),Path(schedule.__file__).resolve(),PLAN.resolve(),EXECUTION.resolve(),
        TEST.resolve(),SERVICE.resolve(),AUDITOR.resolve(),Path('tests/structure_net/test_ncm_task_supervision_schedule.py').resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths = list(dict.fromkeys([*paths,Path('pixi.lock'),Path('pyproject.toml')]))
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'arm':arm,'stage':stage,
        'config':asdict(CONFIG),'environment':environment,'parent_checkpoint_sha256':schedule.PARENT_CHECKPOINT_SHA,
        'parent_model_sha256':PARENT_MODEL_SHA,'systems_lifecycle_only_not_quality_evidence':stage == 'pilot',
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in paths}}
    proof['fingerprint'] = base.fingerprint(proof)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def memory(root,phase):
    torch.cuda.synchronize()
    value = {'phase':phase,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
    with (root/'memory.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    if value['peak_allocated_bytes'] > CONFIG.peak_limit_bytes:raise ValueError('whole lifecycle exceeds4.9GB peak allocation')
    return value['peak_allocated_bytes']


def save_checkpoint(path,model,opt,scaler,step,proof):
    engine.save(path,model,opt,scaler,step=step,fingerprint=proof['fingerprint'],config=CONFIG)
    value = {'path':str(path),'sha256':data.sha(path),'completed_updates':step,'arm':proof['arm'],
        'producer_fingerprint':proof['fingerprint']}
    data.write(path.with_suffix('.json'),value);return value


def stream_parity(sources,sampler):
    serial = [(i,n,[asdict(e) for e in refs],engine.batch_digest(sources.fetch(refs)))
              for i,n,refs in sampler.batches(stop_update=2)]
    prefetched = [(i,n,[asdict(e) for e in refs],engine.batch_digest(rows)) for i,n,refs,rows in
        schedule.shared.prefetch(sampler.batches(stop_update=2),sources.fetch,capacity=CONFIG.prefetch_batches)]
    if serial != prefetched:raise ValueError('serial and prefetched full features/targets differ')
    return {'all_records_equal':True,'microbatches':serial}


def verify_trace(path,rows,config=CONFIG,expected_updates=None):
    sampler = schedule.Schedule(len(rows),config);counts = [0]*len(rows);noisy = 0;last = 0
    with Path(path).open() as f:
        for last,line in enumerate(f,1):
            value = json.loads(line);refs = sampler.update(last-1)
            micro = value['microbatches']
            expected = {'ids':[rows[e.index]['id'] for e in refs],'positions':[e.position for e in refs],
                'populations':[e.population for e in refs],'noisy':[e.noisy for e in refs]}
            if (value['step'] != last or value['epoch'] != 1 or value['examples'] != config.effective_examples
                    or value['lr'] != schedule.shared.rate(last,config)
                    or len(micro) != config.effective_examples//config.microbatch
                    or any([x for m in micro for x in m[k]] != expected[k] for k in expected)
                    or any(len(m['ids']) != config.microbatch for m in micro)
                    or any(not math.isfinite(value[k]) or value[k] < 0 for k in ('weighted_CE','gradient_norm'))):
                raise ValueError('complete fixed sampler/loss-denominator trace differs')
            for e in refs:counts[e.index] += 1;noisy += e.noisy
    expected_updates = config.updates if expected_updates is None else expected_updates
    if last != expected_updates:raise ValueError('missing or additional completed updates')
    return {'updates':last,'presentations':sum(counts),'noisy_presentations':noisy,
        'minimum_visits':min(counts),'maximum_visits':max(counts),'distinct_sources_visited':sum(n>0 for n in counts),
        'trace_sha256':data.sha(path)}


def run(arm,stage):
    root,storage = ROOTS[arm,stage],STORAGE/arm/stage
    if root.exists() or storage.exists():raise ValueError('fresh append-only arm/stage required')
    training,environment = configure();rows,index,manifest,paths = prerequisites(arm,stage)
    sources = Sources(rows,index,manifest);sampler = schedule.Schedule(len(rows))
    selected = control_ids(rows,index);lookup = {r['id']:r for r in rows}
    proof = freeze(root,arm,stage,training,environment,paths);storage.mkdir(parents=True)
    data.write(root/'control_ids.json',selected)
    torch.cuda.reset_peak_memory_stats()
    model = load_initial(training);frozen = training.prior.protected(model)
    memory(root,'loaded')
    values = [sources.cache.get(lookup[k]) for k in selected]
    codes = [lookup[k]['oracle_codes'] for k in selected]
    initial = pipeline.lifecycle.outputs(model,values);data.write(root/'initial_outputs.json',initial)
    if stage == 'pilot':
        data.write(root/'stream_parity.json',stream_parity(sources,sampler))
        old_root = pipeline.lifecycle.ROOT;pipeline.lifecycle.ROOT = root
        try:
            for noisy in (False,True):pipeline.lifecycle.parity(model,values,codes,noisy)
        finally:pipeline.lifecycle.ROOT = old_root
    elif initial != data.read(ROOTS[arm,'pilot']/'initial_outputs.json'):
        raise ValueError('full arm initializer outputs differ from its pilot')
    if training.native.vendor.module_hash(model) != PARENT_MODEL_SHA:raise ValueError('pre-update controls changed initializer weights')
    broad.mechanism.configure(model,True);opt,scaler = optimizer_for(model)
    torch.manual_seed(CONFIG.dropout_seed);torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    limit = 2 if stage == 'pilot' else CONFIG.updates
    batches = schedule.shared.prefetch(sampler.batches(stop_update=limit),sources.fetch,capacity=CONFIG.prefetch_batches)
    started = time.monotonic();checkpoints = []
    try:
        with (root/'trace.jsonl').open('x') as trace:
            for step in range(1,limit+1):
                value = engine.update(model,batches,opt,scaler,step=step,config=CONFIG)
                broad.offload.assert_cpu_state(opt)
                value['elapsed_seconds'] = time.monotonic()-started
                value['peak_allocated_bytes'] = memory(root,'step_'+str(step))
                trace.write(json.dumps(value,allow_nan=False)+'\n');trace.flush()
                if stage == 'pilot' or step%CONFIG.recovery_interval == 0 or step == limit:
                    if training.prior.protected(model) != frozen:raise ValueError('protected tensors changed')
                    checkpoints.append(save_checkpoint(storage/f'step_{step:06d}.pt',model,opt,scaler,step,proof))
                if stage == 'pilot' or step%16 == 0:
                    print(json.dumps({k:v for k,v in value.items() if k!='microbatches'}),flush=True)
    finally:batches.close()
    final_outputs = pipeline.lifecycle.outputs(model,values);data.write(root/'final_control_outputs.json',final_outputs)
    if stage == 'pilot':
        first = torch.load(storage/'step_000001.pt',map_location='cpu',weights_only=True,mmap=True)
        engine.restore(first,model,opt,scaler,fingerprint=proof['fingerprint'],config=CONFIG)
        replay_batches = ((i,n,refs,sources.fetch(refs)) for i,n,refs in sampler.batches(1,2))
        replay = engine.update(model,replay_batches,opt,scaler,step=2,config=CONFIG)
        data.write(root/'resumed_update_2.json',replay)
        second = torch.load(storage/'step_000002.pt',map_location='cpu',weights_only=True,mmap=True)
        current = engine.snapshot(model,opt,scaler,step=2,fingerprint=proof['fingerprint'],config=CONFIG)
        same = training.preparation.checkpoint.training.lifecycle.same
        equality = {key:same(value,second[key]) for key,value in current.items()}
        data.write(root/'resumed_state_equality.json',equality)
        original_last = json.loads((root/'trace.jsonl').read_text().splitlines()[-1])
        if (not all(equality.values()) or replay['microbatches'] != original_last['microbatches']
                or pipeline.lifecycle.outputs(model,values) != final_outputs):
            raise ValueError('exact resumed optimizer/RNG state or source-only outputs differ')
        del first,second,current
    if training.prior.protected(model) != frozen:raise ValueError('protected tensors changed during arm lifecycle')
    peak = memory(root,'final_verified');full_hash = training.native.vendor.module_hash(model)
    last_checkpoint = Path(checkpoints[-1]['path'])
    saved = torch.load(last_checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if engine.restore(saved,model,opt,scaler,fingerprint=proof['fingerprint'],config=CONFIG) != limit:
        raise ValueError('complete final checkpoint position differs')
    if training.native.vendor.module_hash(model) != full_hash:raise ValueError('strict final checkpoint reload differs')
    (root/'model.pt').symlink_to(last_checkpoint.resolve())
    counts = verify_trace(root/'trace.jsonl',rows,expected_updates=limit)
    lengths = {r['id']:r['source_positions'] for r in index}
    counts['source_feature_positions_presented'] = sum(lengths[rows[e.index]['id']]
        for i in range(limit) for e in sampler.update(i))
    counts['native_target_positions_presented'] = counts['presentations']*128
    counts['target_code_positions_presented'] = counts['presentations']*641
    data.write(root/'checkpoints.json',checkpoints);data.write(root/'trace_replay.json',counts)
    base.verify_proof(proof)
    artifacts = [p for p in root.iterdir() if p.is_file()] + [p for p in storage.iterdir() if p.is_file()]
    result = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'status':'task_supervision_lifecycle_passed' if stage == 'pilot' else 'task_supervision_training_completed',
        'arm':arm,'stage':stage,'cases':len(rows),'updates':limit,'config':asdict(CONFIG),'presentations':counts['presentations'],
        'gradient_parity_passed':stage == 'pilot','strict_resumed_update_passed':stage == 'pilot',
        'source_only_outputs_reproduced':stage == 'pilot','strict_final_reload_passed':True,
        'fresh_optimizer_masters_match_initial_weights':True,'protected_tensors_unchanged':True,
        'frozen_hashes':frozen,'final_model_sha256':full_hash,'checkpoint_sha256':data.sha(last_checkpoint),
        'peak_allocated_bytes':peak,'feature_cache':sources.cache.metrics(),'elapsed_seconds':time.monotonic()-started,
        'measured_exposure':counts,'FLOPs':None,'FLOPs_status':'not_measured',
        'systems_lifecycle_only_not_quality_evidence':stage == 'pilot','task_accuracy':None,'new_judge_requests':0,
        'final_test_opened':False,'speed_tested':False,'artifacts':{str(p):data.sha(p) for p in artifacts}}
    data.write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
    model.close();del saved,model,opt,scaler;gc.collect();torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','train'))
    parser.add_argument('--arm',choices=ARMS,required=True);args = parser.parse_args()
    root = ROOTS[args.arm,args.stage];existed = root.exists()
    try:run(args.arm,args.stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error)})
        raise
