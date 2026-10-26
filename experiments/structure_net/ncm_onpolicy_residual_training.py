"""Matched generated-history NCM training with fixed or recomputed residual labels."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_broad_generator as broad
from experiments.structure_net import ncm_onpolicy_residual as objective

PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-onpolicy-residual-plan.md')
TESTS = (Path('tests/structure_net/test_ncm_onpolicy_residual.py'),
         Path('tests/structure_net/test_ncm_onpolicy_residual_training.py'))
SCHEMA = 'nal.ncm-onpolicy-residual.v1'
HYPOTHESIS = 'ncm-onpolicy-residual-v1'
PARENT = broad.RUNS['train']
PARENT_SHA = '74cdce0eb14bb99c801b6ec0a025c092f95249663becc05561af8dfbad6040ff'
MODEL_SHA = 'dbafe920914779d8099b6d5f56464f1ec41dd6a70a53d2022135cc1fd2968ac9'
PARENT_RESULTS = broad.runtime.ROOT/'20261025_broad_generator_results_v1'
PARENT_RESULTS_SHA = '2c0241923d497be9d92589fe4845d3e7013f7754e0d3588cf4c2094be4b77299'
ROOTS = {(a,s):broad.runtime.ROOT/f'20261025_onpolicy_residual_{a}_{s}_v1'
         for a in objective.ARMS for s in ('pilot','train')}


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    accumulation: int = 4
    warmup: int = 64
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    sampler_seed: int = 191142
    dropout_seed: int = 193142
    clean_fraction: float = .5
    history: str = 'current_policy_greedy_FP32_source_only'
    precision: str = 'FP32weights_FP16loss_backprop_FP32generation_evaluation'
    optimizer: str = 'CPU_FP32_AdamW_betas0.9_0.95_eps1e-8_wd0.01_clip1'
    peak_limit_bytes: int = 4_900_000_000
    host_feature_cache_bytes: int = 256*1024**2
    TF32: bool = False
    physical_GPU: int = 1
    codec_and_word_source_attention_frozen: bool = True
    final_test_opened: bool = False
    speed_tested: bool = False


CONFIG = Config()


def rate(step):
    return broad.rate(step,CONFIG)


def select_row(rows, lookup, selection, permutation, sampler, micro, stage):
    if stage == 'pilot':
        return lookup[selection['longest_source_id' if micro%2 else 'first_id']]
    if stage != 'train': raise ValueError('fixed pilot/train stage required')
    if not permutation:
        permutation.extend(torch.randperm(len(rows),generator=sampler).tolist())
    return rows[permutation.pop()]


def load_initial(training):
    training.preparation.completed(PARENT)
    if broad.file_hash(PARENT/'model.pt') != PARENT_SHA:
        raise ValueError('unchanged completed broad-generator checkpoint required')
    if broad.file_hash(PARENT_RESULTS/'campaign_results.json') != PARENT_RESULTS_SHA:
        raise ValueError('completed broad-generator semantic result changed')
    if training.q.read(PARENT_RESULTS/'backend_actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('independent parent registry readback required')
    model = broad.parent.load_parent(training)
    saved = torch.load(PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True)
    if training.native.vendor.module_hash(model) != MODEL_SHA:
        raise ValueError('full selected initializer tensors differ')
    return model


def freeze(root, arm, stage, training, environment, extra):
    files = {Path(__file__),Path(objective.__file__),PLAN,*TESTS}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [PARENT/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt','outputs.json')]
    paths += [PARENT_RESULTS/n for n in ('campaign_results.json','backend_exact_readback.json','backend_actual_tool_receipt.json')]
    paths += [broad.RUNS['prepare']/n for n in ('cases.json','selection.json','oracle_outputs.json','producer.json','result.json','actual_tool_receipt.json')]
    paths += [broad.catalog.DEST/n for n in ('source_index.json','feature_manifest.json')]
    paths += [Path('pyproject.toml'),Path('pixi.lock'),*extra]
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'arm':arm,'stage':stage,
        'conformance':'EXPLORATORY_UNDERPOWERED','config':asdict(CONFIG),
        'actual_updates':2 if stage == 'pilot' else CONFIG.updates,
        'systems_lifecycle_only_not_quality_evidence':stage == 'pilot',
        'environment':environment,'source_hashes':{str(p):broad.file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):broad.file_hash(p) for p in dict.fromkeys(paths)}}
    proof['fingerprint'] = training.q.digest(proof)
    root.mkdir(); (root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    broad.write_json(root/'producer.json',proof)
    return proof


def save(path, model, opt, scaler, sampler, permutation, step, proof):
    if path.exists(): raise ValueError('checkpoint overwrite forbidden')
    value = {'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':sampler.get_state(),
        'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state_all(),
        'permutation':list(permutation),'step':step,'arm':proof['arm'],
        'producer_fingerprint':proof['fingerprint']}
    temporary = path.with_suffix('.partial')
    torch.save(value,temporary)
    temporary.rename(path)


def run(arm, stage, training, environment):
    training.preparation.completed(broad.RUNS['prepare'])
    for path,sha in training.q.read(broad.RUNS['prepare']/'result.json')['artifacts'].items():
        if broad.file_hash(path) != sha: raise ValueError('prepared TRAIN evidence changed')
    extra = []
    if stage == 'train':
        for other in objective.ARMS:
            pilot = ROOTS[other,'pilot']
            training.preparation.completed(pilot)
            result = training.q.read(pilot/'result.json')
            if not all(result[k] for k in ('strict_reload_passed','gradient_parity_passed','oracle_unchanged')):
                raise ValueError('both arm lifecycle pilots must pass before full training')
            extra += [pilot/n for n in ('producer.json','result.json','actual_tool_receipt.json','gradient_parity.json')]
        if arm == 'residual_targets':
            first = ROOTS['fixed_targets','train']
            training.preparation.completed(first)
            extra += [first/n for n in ('result.json','actual_tool_receipt.json')]
    rows = training.q.read(broad.RUNS['prepare']/'cases.json')
    selection = training.q.read(broad.RUNS['prepare']/'selection.json')
    lookup = {r['id']:r for r in rows}
    if len(rows) != 1376 or len(lookup) != 1376 or any(r['split'] != 'TRAIN' for r in rows):
        raise ValueError('complete unchanged1376 TRAIN population required')
    controls = [lookup[i] for i in selection['diagnostic_ids']]
    index,manifest = [training.q.read(broad.catalog.DEST/n) for n in ('source_index.json','feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index,manifest,max_bytes=CONFIG.host_feature_cache_bytes)
    model = load_initial(training)
    root = ROOTS[arm,stage]
    proof = freeze(root,arm,stage,training,environment,extra)
    broad.memory(root,'model_loaded')
    frozen = training.prior.protected(model)
    parity = []
    if stage == 'pilot':
        for key,onpolicy in (('first_id',False),('longest_source_id',True)):
            row = lookup[selection[key]]
            f,m,gold = training.prior.tensors(row,loader)
            words = torch.tensor([row['native_word_ids']],device=f.device)
            parity.append(objective.gradient_parity(model,f,m,gold,words,arm,onpolicy))
            broad.memory(root,'gradient_parity_'+key)
        broad.write_json(root/'gradient_parity.json',parity)
    tokenizer = training.native.tokenizer()
    initial = training.prior.diagnostics(model,controls,loader,tokenizer)
    parent_outputs = {r['id']:r for r in training.q.read(PARENT/'outputs.json')}
    if initial != [parent_outputs[r['id']] for r in controls]:
        raise ValueError('fixed initializer source-only outputs differ from completed parent')
    broad.write_json(root/'initial_outputs.json',initial)
    if stage == 'train' and initial != training.q.read(ROOTS[arm,'pilot']/'initial_outputs.json'):
        raise ValueError('pilot/full initial inference differs')
    objective.sequential.configure(model,True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = broad.offload.CPUAdamW(params,lr=CONFIG.max_lr,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG.sampler_seed)
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    permutation = []
    updates = 2 if stage == 'pilot' else CONFIG.updates
    started = time.monotonic()
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1,updates+1):
            objective.sequential.configure(model,True); opt.zero_grad(set_to_none=True)
            for group in opt.param_groups: group['lr'] = rate(step)
            measured = []
            for micro in range(CONFIG.accumulation):
                row = select_row(rows,lookup,selection,permutation,sampler,micro,stage)
                f,m,gold = training.prior.tensors(row,loader)
                words = torch.tensor([row['native_word_ids']],device=f.device)
                metric = objective.backward_example(model,f,m,gold,words,arm,bool(micro%2),scaler,CONFIG.accumulation)
                measured.append({'id':row['id'],**metric})
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('nonfinite or frozen gradient')
            if step == 1 and (model.concepts.generator.l0_queries.grad is None
                             or not model.concepts.generator.l0_queries.grad.abs().sum()):
                raise ValueError('root queries must learn')
            old_scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); broad.offload.assert_cpu_state(opt)
            if scaler.get_scale() < old_scale: raise ValueError('skipped optimizer update')
            peak = broad.memory(root,'update_'+str(step))['peak_allocated_bytes']
            record = {'step':step,'updates':updates,'arm':arm,'lr':rate(step),
                'gradient_norm':float(norm),'metrics':measured,
                'mean_CE':sum(v['mean_CE'] for v in measured)/len(measured),
                'elapsed_seconds':time.monotonic()-started,'peak_bytes':peak}
            trace.write(json.dumps(record,allow_nan=False)+'\n'); trace.flush()
            if step <= 2 or step%16 == 0:
                broad.write_json(root/'progress.json',record)
                print(json.dumps({k:v for k,v in record.items() if k != 'metrics'}),flush=True)
    elapsed = time.monotonic()-started
    opt.zero_grad(set_to_none=True)
    final_hash = training.native.vendor.module_hash(model)
    if training.prior.protected(model) != frozen or final_hash == MODEL_SHA:
        raise ValueError('protected state changed or generator did not update')
    save(root/'model.pt',model,opt,scaler,sampler,permutation,updates,proof)
    broad.write_json(root/'checkpoint_ready.json',{'step':updates,'model_sha256':final_hash,
        'checkpoint_sha256':broad.file_hash(root/'model.pt'),'training_seconds':elapsed,
        'producer_fingerprint':proof['fingerprint'],'evaluation_pending':True})
    final = training.prior.diagnostics(model,controls if stage == 'pilot' else rows,loader,tokenizer)
    broad.write_json(root/'outputs.json',final)
    shuffled = training.prior.diagnostics(model,controls,loader,tokenizer,shuffle=True)
    broad.write_json(root/'shuffled_source_outputs.json',shuffled)
    clean = broad.oracle(model,controls,loader)
    original = {r['id']:r for r in training.q.read(broad.RUNS['prepare']/'oracle_outputs.json')}
    if clean != [original[r['id']] for r in controls]: raise ValueError('clean reconstruction changed')
    broad.write_json(root/'oracle_outputs.json',clean)
    del opt; gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True)
    opt = broad.offload.CPUAdamW(params,lr=CONFIG.max_lr,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng'])
    torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same = training.preparation.checkpoint.training.lifecycle.same
    expected = {r['id']:r for r in final}
    if (not same(model.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
            or not same(scaler.state_dict(),saved['scaler'])
            or not torch.equal(sampler.get_state(),saved['sampler_rng'])
            or not torch.equal(torch.get_rng_state(),saved['CPU_rng'])
            or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
            or permutation != saved['permutation'] or saved['arm'] != arm
            or saved['step'] != updates or saved['producer_fingerprint'] != proof['fingerprint']
            or training.prior.diagnostics(model,controls,loader,tokenizer) != [expected[r['id']] for r in controls]
            or training.prior.protected(model) != frozen
            or training.native.vendor.module_hash(model) != final_hash):
        raise ValueError('strict checkpoint/optimizer/scaler/RNG/output reload failed')
    peak = broad.memory(root,'strict_reload')['peak_allocated_bytes']
    training.native.vendor.verify(proof)
    result = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'arm':arm,
        'status':'onpolicy_residual_'+stage+'_completed_semantic_quality_pending',
        'updates':updates,'TRAIN_cases':len(rows),'trainable_parameters':sum(p.numel() for p in params),
        'initial_controls':training.prior.summarize(initial),'final_TRAIN':training.prior.summarize(final),
        'shuffled_source16':training.prior.summarize(shuffled),'strict_reload_passed':True,
        'gradient_parity_passed':all(r['passed'] for r in parity) if stage == 'pilot' else True,
        'oracle_unchanged':True,'frozen_hashes':frozen,'initial_model_sha256':MODEL_SHA,
        'final_model_sha256':final_hash,'checkpoint_sha256':broad.file_hash(root/'model.pt'),
        'producer_fingerprint':proof['fingerprint'],'training_seconds':elapsed,
        'peak_allocated_bytes':peak,'feature_cache':loader.metrics(),'task_accuracy':None,
        'final_test_opened':False,'speed_tested':False}
    broad.write_json(root/'result.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)
    model.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('arm',choices=objective.ARMS)
    parser.add_argument('stage',choices=('pilot','train'))
    args = parser.parse_args()
    root = ROOTS[args.arm,args.stage]
    if root.exists(): raise ValueError('fresh append-only artifact root required')
    training,environment = broad.runtime.configure()
    broad.offload.require_rtx2060(); torch.cuda.reset_peak_memory_stats()
    try:
        run(args.arm,args.stage,training,environment)
    except Exception as error:
        if root.exists(): broad.write_json(root/'failure.json',{'exception':type(error).__name__,
            'message':str(error),'task_accuracy':None})
        raise


if __name__ == '__main__':
    main()
