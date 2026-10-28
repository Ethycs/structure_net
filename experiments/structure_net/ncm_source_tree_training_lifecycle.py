"""Qualify mixed-precision source-tree training and exact resumed updates."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_source_tree_checkpoint_pilot as prior
from experiments.structure_net import ncm_task_supervision_training as task

data, tree = prior.data, prior.tree
pipeline, engine, broad = task.pipeline, task.engine, task.broad
CONFIG = replace(task.CONFIG, physical_GPU=1)
ROOT = data.ROOT/'20261027_source_tree_training_lifecycle_v1'
STORAGE = Path('/data2/structure_net/ncm_source_tree_20261027_v1/lifecycle')
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-tree-training-lifecycle.md')
HYPOTHESIS = 'ncm-source-tree-training-lifecycle-v1'


def run():
    if ROOT.exists() or STORAGE.exists(): raise ValueError('fresh lifecycle required')
    receipt = data.read(prior.ROOT/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    initial_pilot = data.read(prior.ROOT/'result.json')
    if receipt['actual_tool_exit_code'] != 0 or not initial_pilot['all_initial_teacher_logits_exact']:
        raise ValueError('real-checkpoint initialization qualification required')
    cpu = data.read(prior.QUALIFICATION); data.verify(cpu['artifacts'])
    if cpu['actual_tool_exit_code'] != 0 or cpu['tests_passed'] != 4: raise ValueError('tree CPU qualification required')
    training, environment = broad.runtime.configure(); pipeline.determinism.configure_runtime()
    broad.offload.require_rtx2060()
    if torch.cuda.mem_get_info()[0] < CONFIG.peak_limit_bytes: raise ValueError('whole lifecycle GPU headroom required')
    before_services = prior.services()
    rows, index, manifest, paths = task.dataset('existing'); paths += task.require_parent()
    sources = task.Sources(rows, index, manifest); sampler = task.schedule.Schedule(len(rows), CONFIG)
    ids = task.control_ids(rows, index); lookup = {r['id']: r for r in rows}
    paths += [prior.ROOT/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [prior.QUALIFICATION, Path('pixi.lock'), Path('pyproject.toml')]
    files = {Path(__file__).resolve(), PLAN.resolve(), Path(tree.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())): files.add(p)
    proof = {'schema':'nal.ncm-source-tree-training-lifecycle.v1','hypothesis_id':HYPOTHESIS,
        'environment':environment,'config':asdict(CONFIG),'updates':2,'cases':len(rows),
        'systems_lifecycle_only_not_quality_evidence':True,'parent_model_sha256':task.PARENT_MODEL_SHA,
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof)
    ROOT.mkdir(); STORAGE.mkdir(parents=True); (ROOT/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)): (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(ROOT/'producer.json',proof); data.write(ROOT/'control_ids.json',ids)
    torch.cuda.reset_peak_memory_stats(); model = task.load_initial(training)
    values = [sources.cache.get(lookup[k]) for k in ids]; codes = [lookup[k]['oracle_codes'] for k in ids]
    original = pipeline.lifecycle.outputs(model,values); frozen = training.prior.protected(model)
    metadata = tree.install(model); broad.mechanism.configure(model,False)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 123063936: raise ValueError('complete tree parameter population required')
    initial = pipeline.lifecycle.outputs(model,values)
    if initial != original: raise ValueError('initial source-only codes or words changed')
    initializer = training.native.vendor.module_hash(model)
    data.write(ROOT/'initial_outputs.json',initial); data.write(ROOT/'tree.json',metadata)
    data.write(ROOT/'stream_parity.json',task.stream_parity(sources,sampler))
    old_root = pipeline.lifecycle.ROOT; pipeline.lifecycle.ROOT = ROOT
    try:
        for noisy in (False,True): pipeline.lifecycle.parity(model,values,codes,noisy)
    finally: pipeline.lifecycle.ROOT = old_root
    if training.native.vendor.module_hash(model) != initializer: raise ValueError('parity checks changed initializer')
    broad.mechanism.configure(model,True); opt,scaler = task.optimizer_for(model)
    if sum(p.numel() for p in opt.param_groups[0]['params']) != 123063936: raise ValueError('optimizer missed tree parameters')
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    batches = task.schedule.shared.prefetch(sampler.batches(stop_update=2),sources.fetch,capacity=CONFIG.prefetch_batches)
    started = time.monotonic()
    try:
        with (ROOT/'trace.jsonl').open('x') as trace:
            for step in (1,2):
                record = engine.update(model,batches,opt,scaler,step=step,config=CONFIG)
                broad.offload.assert_cpu_state(opt)
                record['elapsed_seconds'] = time.monotonic()-started
                record['peak_allocated_bytes'] = task.memory(ROOT,'step_'+str(step))
                trace.write(json.dumps(record,allow_nan=False)+'\n'); trace.flush()
                if training.prior.protected(model) != frozen: raise ValueError('protected tensors changed')
                engine.save(STORAGE/f'step{step}.pt',model,opt,scaler,step=step,fingerprint=proof['fingerprint'],config=CONFIG)
                print(json.dumps({k:v for k,v in record.items() if k!='microbatches'}),flush=True)
    finally: batches.close()
    final_outputs = pipeline.lifecycle.outputs(model,values); data.write(ROOT/'final_outputs.json',final_outputs)
    first = torch.load(STORAGE/'step1.pt',map_location='cpu',weights_only=True,mmap=True)
    engine.restore(first,model,opt,scaler,fingerprint=proof['fingerprint'],config=CONFIG)
    replay_batches = ((i,n,refs,sources.fetch(refs)) for i,n,refs in sampler.batches(1,2))
    replay = engine.update(model,replay_batches,opt,scaler,step=2,config=CONFIG)
    data.write(ROOT/'resumed_update_2.json',replay)
    second = torch.load(STORAGE/'step2.pt',map_location='cpu',weights_only=True,mmap=True)
    current = engine.snapshot(model,opt,scaler,step=2,fingerprint=proof['fingerprint'],config=CONFIG)
    same = training.preparation.checkpoint.training.lifecycle.same
    equality = {key:same(value,second[key]) for key,value in current.items()}
    data.write(ROOT/'resumed_state_equality.json',equality)
    if not all(equality.values()) or replay['microbatches'] != record['microbatches'] or pipeline.lifecycle.outputs(model,values) != final_outputs:
        raise ValueError('exact resumed update or source-only output differs')
    if training.prior.protected(model) != frozen: raise ValueError('protected tensors changed after restart')
    peak = task.memory(ROOT,'verified'); counts = task.verify_trace(ROOT/'trace.jsonl',rows,CONFIG,expected_updates=2)
    after_services = prior.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services): raise ValueError('other worker identity changed')
    data.write(ROOT/'services.json',{'before':before_services,'after':after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    paths = [p for p in ROOT.iterdir() if p.is_file()]+[STORAGE/'step1.pt',STORAGE/'step2.pt']
    result = {'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,'status':'source_tree_mixed_precision_training_and_restart_passed',
        'updates':2,'trainable_parameters':123063936,'cases':1376,'physical_GPU':1,
        'initial_source_only_outputs_exact':True,'FP32_clean_and_noisy_gradient_parity_passed':True,
        'FP16_full_accumulation_updates_completed':True,'strict_resumed_update_passed':True,
        'protected_tensors_unchanged':True,'peak_allocated_bytes':peak,'measured_exposure':counts,
        'systems_lifecycle_only_not_quality_evidence':True,'task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in paths}}
    data.write(ROOT/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)
    model.close(); del model,opt,scaler,first,second,current; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
