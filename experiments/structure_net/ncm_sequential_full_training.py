"""Fixed full732 next-scale training after decoder reconstruction qualifies."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_sequential_decoder as decoder
from experiments.structure_net import ncm_packed_sequential_objective as objective
from experiments.structure_net.ncm_memory_data import file_hash, write_json

prior, base, native, preparation, q, ROOT = decoder.prior, decoder.base, decoder.native, decoder.preparation, decoder.q, decoder.ROOT
RUNS = {s: ROOT / ('20261025_sequential_full732_' + s + '_v1') for s in ('pilot', 'train')}
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-sequential-full732-plan.md')
TEST = Path('tests/structure_net/test_ncm_packed_sequential_objective.py')
CONFIG = dict(updates=2048, accumulation=4, warmup=64, max_lr=1e-4, min_lr=1e-5,
    sampler_seed=161142, corruption_seed=162142, dropout_seed=163142,
    training='packed_native_block_diagonal_teacher_forcing', inference='explicit_sequential_source_only',
    precision='FP32weights_FP16training_FP32inference', optimizer='AdamW', betas=[.9,.95],
    eps=1e-8, weight_decay=.01, clip=1., physical_GPU=2, TF32=False,
    peak_limit_bytes=4_900_000_000, host_feature_cache_bytes=24*1024**3,
    codec_and_word_source_attention_frozen=True, task_accuracy=None, final_test_opened=False, speed_tested=False)


def rate(step):
    if not 1 <= step <= CONFIG['updates']:
        raise ValueError('step outside fixed budget')
    if step <= CONFIG['warmup']:
        return CONFIG['max_lr']*step/CONFIG['warmup']
    return CONFIG['min_lr'] + .5*(CONFIG['max_lr']-CONFIG['min_lr'])*(1+math.cos(math.pi*(step-CONFIG['warmup'])/(CONFIG['updates']-CONFIG['warmup'])))


def load_initial():
    root = decoder.RUNS['train']
    preparation.completed(root)
    state = q.read(root / 'result.json')
    if not state['full_decoder_reconstruction_pass'] or not state['strict_reload_passed']:
        raise ValueError('full732 decoder reconstruction gate must pass first')
    if file_hash(root / 'model.pt') != state['checkpoint_sha256']:
        raise ValueError('completed repaired decoder changed')
    model = decoder.load_parent()
    saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.codec.decoder.load_state_dict(saved['decoder'], strict=True)
    if (native.vendor.module_hash(model.codec.decoder) != state['final_decoder_sha256']
            or decoder.protected(model) != state['protected_sha256']):
        raise ValueError('compositional decoder checkpoint differs')
    return model


def freeze(root, stage):
    files = {Path(__file__), PLAN, TEST, Path(objective.__file__)}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and (path.is_relative_to(Path('experiments').resolve()) or path.is_relative_to(native.vendor.VENDOR.resolve())):
                files.add(path)
    files.update(native.vendor.VENDOR.rglob('*.py'))
    inputs = [base.DATA / n for n in ('cases.json','source_index.json','feature_manifest.json','selection.json','result.json','actual_tool_receipt.json')]
    inputs += [decoder.RUNS['train'] / n for n in ('producer.json','model.pt','result.json','actual_tool_receipt.json')]
    inputs += [decoder.PARENT / 'model.pt']
    if stage == 'train':
        inputs += [RUNS['pilot'] / n for n in ('producer.json','result.json','actual_tool_receipt.json','gradient_parity.json')]
    root.mkdir(); (root / 'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (root / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema':'nal.ncm-sequential-full732.v1','hypothesis_id':'ncm-sequential-full732-v1',
        'conformance':'EXPLORATORY','stage':stage,'config':{**CONFIG,'actual_updates':2 if stage=='pilot' else CONFIG['updates'],
            'systems_lifecycle_only_not_quality_evidence':stage=='pilot'},
        'source_hashes':{str(p):file_hash(p) for p in sorted(files)},'input_hashes':{str(p):file_hash(p) for p in inputs},
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()}}
    proof['fingerprint'] = q.digest(proof)
    write_json(root / 'producer.json',proof)
    return proof


def run(stage):
    native.vendor.corpus.quality.dependencies()
    preparation.checkpoint.training.require_gpu()
    preparation.completed(base.DATA)
    if stage == 'train':
        preparation.completed(RUNS['pilot'])
        pilot = q.read(RUNS['pilot'] / 'result.json')
        if not pilot['strict_reload_passed'] or not pilot['gradient_parity_passed']:
            raise ValueError('completed full732 CUDA and objective parity pilot required')
    rows = q.read(base.DATA / 'cases.json')
    if len(rows)!=732 or file_hash(base.DATA / 'cases.json') != q.read(base.DATA / 'result.json')['cases_sha256']:
        raise ValueError('exact complete732 TRAIN population required')
    lookup = {r['id']:r for r in rows}
    controls = [lookup[i] for i in q.read(base.DATA / 'selection.json')['diagnostic_ids']]
    index, manifest = q.read(base.DATA / 'source_index.json'), q.read(base.DATA / 'feature_manifest.json')
    loader = preparation.streaming.FeatureCache(index,manifest,max_bytes=CONFIG['host_feature_cache_bytes'])
    lengths = {r['id']:r['source_positions'] for r in index}
    longest = max(range(len(rows)),key=lambda i:lengths[rows[i]['id']])
    root = RUNS[stage]
    if root.exists(): raise ValueError('fresh full732 study root required')
    model = load_initial()
    proof = freeze(root,stage)
    frozen, initial_hash = prior.protected(model), native.vendor.module_hash(model)
    tokenizer = native.tokenizer()
    parity = []
    if stage == 'pilot':
        torch.cuda.reset_peak_memory_stats()
        for i,noisy in ((0,False),(longest,True)):
            f,m,gold = prior.tensors(rows[i],loader)
            parity.append(objective.gradient_parity(model,f,m,gold,noisy))
            parity[-1]['id'] = rows[i]['id']
            parity[-1]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            if parity[-1]['peak_allocated_bytes'] > CONFIG['peak_limit_bytes']:
                raise ValueError('FP32 gradient parity exceeds declared allocation cap')
        write_json(root / 'gradient_parity.json',parity)
    initial = prior.diagnostics(model,controls,loader,tokenizer)
    write_json(root / 'initial_outputs.json',initial)
    prior.mechanism.configure(model,True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG['sampler_seed'])
    noise = torch.Generator().manual_seed(CONFIG['corruption_seed'])
    torch.manual_seed(CONFIG['dropout_seed']);torch.cuda.manual_seed_all(CONFIG['dropout_seed'])
    permutation=[];steps=2 if stage=='pilot' else CONFIG['updates']
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with (root / 'trace.jsonl').open('x') as trace:
        for step in range(1,steps+1):
            prior.mechanism.configure(model,True)
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate(step)
            metrics=[]
            for micro in range(CONFIG['accumulation']):
                if not permutation:permutation=torch.randperm(len(rows),generator=sampler).tolist()
                i=([0,longest,0,longest][micro] if stage=='pilot' else permutation.pop())
                f,m,gold=prior.tensors(rows[i],loader)
                metric=objective.backward_example(model,f,m,gold,noise,scaler,CONFIG['accumulation'],bool(micro%2))
                metrics.append({'id':rows[i]['id'],**metric})
            scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite full732 gradients')
            if step==1 and (model.concepts.generator.l0_queries.grad is None or not model.concepts.generator.l0_queries.grad.abs().sum()):
                raise ValueError('native root queries must learn')
            if any(p.grad is not None for p in model.codec.parameters()):raise ValueError('frozen codec gradient')
            scaler.step(opt);scaler.update()
            peak=torch.cuda.max_memory_allocated()
            if peak>CONFIG['peak_limit_bytes']:raise ValueError('full732 memory cap exceeded')
            record={'step':step,'updates':steps,'lr':rate(step),'gradient_norm':float(norm),'metrics':metrics,
                'mean_CE':sum(x['mean_CE'] for x in metrics)/len(metrics),'elapsed_seconds':time.monotonic()-started,'peak_bytes':peak}
            trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
            if step<=2 or step%16==0:
                write_json(root / 'progress.json',record)
                print(json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)
    training_seconds=time.monotonic()-started
    # Complete TRAIN rollout is diagnostic; source-disjoint semantic grading is separate.
    final=prior.diagnostics(model,controls if stage=='pilot' else rows,loader,tokenizer)
    shuffled=prior.diagnostics(model,controls,loader,tokenizer,shuffle=True)
    final_hash=native.vendor.module_hash(model)
    if frozen!=prior.protected(model) or final_hash==initial_hash:
        raise ValueError('frozen decoder changed or generator did not learn')
    prior.save(root / 'model.pt',model,opt,scaler,sampler,noise,steps,proof)
    write_json(root / 'sampler_remaining.json',permutation)
    del opt;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(root / 'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True)
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng']);noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['CPU_rng']);torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same=preparation.checkpoint.training.lifecycle.same
    expected={r['id']:r for r in final}
    if (not same(model.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
            or not same(scaler.state_dict(),saved['scaler']) or not torch.equal(sampler.get_state(),saved['sampler_rng'])
            or not torch.equal(noise.get_state(),saved['noise_rng']) or not torch.equal(torch.get_rng_state(),saved['CPU_rng'])
            or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
            or prior.diagnostics(model,controls,loader,tokenizer)!=[expected[r['id']] for r in controls]
            or native.vendor.module_hash(model)!=final_hash):
        raise ValueError('strict model/optimizer/scaler/RNG/output reload failed')
    write_json(root / 'outputs.json',final);write_json(root / 'shuffled_source_outputs.json',shuffled)
    native.vendor.verify(proof)
    result={'schema':proof['schema'],'status':'full732_'+stage+'_completed_source_disjoint_quality_pending',
        'updates':steps,'TRAIN_cases':732,'trainable_parameters':sum(p.numel() for p in params),
        'initial_control16':prior.summarize(initial),'final_TRAIN':prior.summarize(final),
        'shuffled_source16':prior.summarize(shuffled),'strict_reload_passed':True,
        'gradient_parity_passed':all(r['passed'] for r in parity) if stage=='pilot' else True,
        'frozen_hashes':frozen,'initial_model_sha256':initial_hash,'final_model_sha256':final_hash,
        'checkpoint_sha256':file_hash(root / 'model.pt'),'producer_fingerprint':proof['fingerprint'],
        'training_seconds':training_seconds,'peak_allocated_bytes':peak,'feature_cache':loader.metrics(),
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False}
    write_json(root / 'result.json',result)
    print(json.dumps(result,allow_nan=False),flush=True);model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=tuple(RUNS))
    run(parser.parse_args().stage)
