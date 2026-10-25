"""Adapt only the existing text decoder to a frozen NCM's real generated codes."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_pixi as runtime
from experiments.structure_net import ncm_sequential_decoder as decoder
from experiments.structure_net import ncm_cpu_adamw as offload
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = runtime.ROOT
PARENT = runtime.RUNS['train']
PARENT_SHA = '37d8d70f14d81a63ecd63ca6b9a3f5bf1ac9301d866af1135e24404224e795ad'
MODEL_SHA = '9eb6e2c41a3a4e496c588a2ec0666c4cbad6f93ba0fc16c6c88b0ea2df2d339b'
HYPOTHESIS = 'ncm-generated-code-decoder-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-generated-code-decoder-plan.md')
RUNS = {s: ROOT / ('20261025_generated_code_decoder_' + s + '_v1') for s in ('pilot', 'train')}
CONFIG = dict(updates=1024, accumulation=4, sampler_seed=171142, dropout_seed=172142,
    max_lr=1e-4, min_lr=1e-5, warmup=32, betas=[.9,.95], eps=1e-8,
    weight_decay=.01, clip=1., precision='FP32weights_FP16training_FP32evaluation',
    optimizer='CPU_FP32_AdamW', physical_GPU=1, TF32=False,
    peak_limit_bytes=4_900_000_000, host_feature_cache_bytes=256*1024**2,
    only_decoder_trainable=True, input_codes='frozen_parent_source_only_predictions',
    loss='per_example_word_CE_ignore_PAD0_train_EOS101', source_positions_clipped=False,
    final_test_opened=False, speed_tested=False)


def selected_codes(row, predictions, device):
    value = predictions[row['id']]
    if value['source_id_used'] != row['id']:
        raise ValueError('source-only code binding differs')
    codes = torch.tensor([value['codes']], dtype=torch.long, device=device)
    if codes.shape != (1, 641) or codes.min() < 0 or codes.max() >= 8192:
        raise ValueError('complete generated hierarchy required')
    return codes


def load_parent(training):
    decoder.preparation.completed(PARENT)
    state = training.q.read(PARENT / 'result.json')
    if not state['strict_reload_passed'] or file_hash(PARENT / 'model.pt') != PARENT_SHA:
        raise ValueError('completed unchanged parent required')
    model = training.load_initial()
    saved = torch.load(PARENT / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    if training.native.vendor.module_hash(model) != MODEL_SHA:
        raise ValueError('full parent tensor identity differs')
    return decoder.configure(model, False)


def inputs(row, loader, predictions):
    features = loader.get(row).unsqueeze(0).cuda()
    valid = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
    return features, valid, selected_codes(row, predictions, features.device)


@torch.no_grad()
def evaluate(model, rows, loader, predictions, tokenizer):
    decoder.configure(model, False)
    out = []
    for row in rows:
        features, valid, codes = inputs(row, loader, predictions)
        raw = model.word_logits(features, valid, codes).argmax(-1).cpu()[0]
        words = torch.tensor(row['native_word_ids'])
        out.append({'id': row['id'], 'source_sha256': row['source_sha256'],
            'codes': codes.cpu()[0].tolist(), 'raw_word_ids': raw.tolist(),
            **decoder.preparation.rendering.decode(raw.tolist(), tokenizer),
            'active_word_matches': int(((raw == words) & (words != 0)).sum()),
            'active_words': int((words != 0).sum()),
            'rendered_exact': bool(torch.equal(raw[words != 0], words[words != 0])),
            'task_accuracy': None})
    return out


def freeze(root, stage, training, environment, baseline_hash):
    files = {Path(__file__), PLAN, Path('tests/structure_net/test_ncm_generated_code_decoder.py')}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and (
                    path.is_relative_to(Path('experiments').resolve()) or
                    path.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(path)
    paths = [PARENT/n for n in ('producer.json','result.json','outputs.json','model.pt','actual_tool_receipt.json')]
    paths += [training.base.DATA/n for n in ('cases.json','source_index.json','feature_manifest.json','selection.json','result.json')]
    paths += [Path('pyproject.toml'), Path('pixi.lock')]
    if stage == 'train':
        paths += [RUNS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema':'nal.ncm-generated-code-decoder.v1','hypothesis_id':HYPOTHESIS,
        'stage':stage,'conformance':'EXPLORATORY_UNDERPOWERED',
        'config':{**CONFIG,'actual_updates':2 if stage=='pilot' else CONFIG['updates'],
            'systems_lifecycle_only_not_quality_evidence':stage=='pilot'},
        'source_hashes':{str(p):file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):file_hash(p) for p in paths},
        'parent_model_sha256':MODEL_SHA,'parent_output_sha256':baseline_hash,'environment':environment}
    proof['fingerprint'] = training.q.digest(proof)
    write_json(root/'producer.json',proof)
    return proof


def run(stage):
    training, environment = runtime.configure()
    offload.require_rtx2060()
    if RUNS[stage].exists(): raise ValueError('fresh append-only root required')
    if stage == 'train':
        decoder.preparation.completed(RUNS['pilot'])
        if not training.q.read(RUNS['pilot']/'result.json')['strict_reload_passed']:
            raise ValueError('completed lifecycle pilot required')
    rows = training.q.read(training.base.DATA/'cases.json')
    if len(rows)!=732 or file_hash(training.base.DATA/'cases.json')!=training.q.read(training.base.DATA/'result.json')['cases_sha256']:
        raise ValueError('complete unchanged732 TRAIN cases required')
    # The registered independent audit binds the saved parent predictions.
    audit = training.q.read(ROOT/'20261025_sequential_full732_pixi_train_audit_v1/campaign_results.json')
    output_hash = file_hash(PARENT/'outputs.json')
    if audit['artifacts'].get(str(PARENT/'outputs.json')) != output_hash:
        raise ValueError('independently audited parent predictions required')
    predictions = {r['id']:r for r in training.q.read(PARENT/'outputs.json')}
    if set(predictions)!={r['id'] for r in rows}: raise ValueError('all732 generated code inputs required')
    index, manifest = [training.q.read(training.base.DATA/n) for n in ('source_index.json','feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index,manifest,max_bytes=CONFIG['host_feature_cache_bytes'])
    lengths = {r['id']:r['source_positions'] for r in index}
    longest = max(rows,key=lambda r:lengths[r['id']])
    ids = training.q.read(training.base.DATA/'selection.json')['diagnostic_ids']
    controls = [r for r in rows if r['id'] in set(ids)|{longest['id']}]
    model = load_parent(training)
    tokenizer = training.native.tokenizer()
    torch.cuda.reset_peak_memory_stats()
    proof = freeze(RUNS[stage],stage,training,environment,output_hash)
    root = RUNS[stage]
    frozen = decoder.protected(model)
    initial_hash = training.native.vendor.module_hash(model.codec.decoder)
    with torch.no_grad():
        for row in controls:
            features,valid,codes = inputs(row,loader,predictions)
            actual = training.prior.mechanism.generate(model,features,valid)
            words = model.word_logits(features,valid,actual).argmax(-1).cpu()[0].tolist()
            if not torch.equal(actual,codes) or words!=predictions[row['id']]['raw_word_ids']:
                raise ValueError('saved source-only code/output cache failed real regeneration')
    initial = evaluate(model,controls,loader,predictions,tokenizer)
    write_json(root/'initial_outputs.json',initial)
    decoder.configure(model,True)
    params = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in params}!={id(p) for p in model.codec.decoder.parameters()}:
        raise ValueError('only text decoder may learn')
    opt = offload.CPUAdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG['sampler_seed'])
    torch.manual_seed(CONFIG['dropout_seed']);torch.cuda.manual_seed_all(CONFIG['dropout_seed'])
    permutation=[]; updates=2 if stage=='pilot' else CONFIG['updates']; started=time.monotonic()
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1,updates+1):
            decoder.configure(model,True); opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=decoder.rate(step,CONFIG['updates'])
            metrics=[]
            for micro in range(CONFIG['accumulation']):
                if not permutation:permutation=torch.randperm(len(rows),generator=sampler).tolist()
                row=([rows[0],longest,rows[0],longest][micro] if stage=='pilot' else rows[permutation.pop()])
                features,valid,codes=inputs(row,loader,predictions)
                words=torch.tensor([row['native_word_ids']],device=features.device)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model.word_logits(features,valid,codes)
                    loss=decoder.word_loss(logits,words)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                scaler.scale(loss/CONFIG['accumulation']).backward()
                metrics.append({'id':row['id'],'word_CE':float(loss.detach())})
                del logits,loss
            scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            if step==1 and (model.codec.decoder.output_layer.weight.grad is None or not model.codec.decoder.output_layer.weight.grad.abs().sum()):
                raise ValueError('decoder output projection needs a gradient')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('frozen gradient')
            scaler.step(opt);scaler.update();offload.assert_cpu_state(opt)
            peak=torch.cuda.max_memory_allocated()
            if peak>CONFIG['peak_limit_bytes']:raise ValueError('GPU allocation limit exceeded')
            record={'step':step,'updates':updates,'lr':opt.param_groups[0]['lr'],
                'word_CE':sum(r['word_CE'] for r in metrics)/4,'examples':metrics,
                'gradient_norm':float(norm),'elapsed_seconds':time.monotonic()-started,'peak_bytes':peak}
            trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
            if step<=2 or step%16==0:
                write_json(root/'progress.json',record)
                print(json.dumps({k:v for k,v in record.items() if k!='examples'}),flush=True)
    elapsed=time.monotonic()-started
    final=evaluate(model,controls if stage=='pilot' else rows,loader,predictions,tokenizer)
    final_hash=training.native.vendor.module_hash(model)
    if decoder.protected(model)!=frozen or training.native.vendor.module_hash(model.codec.decoder)==initial_hash:
        raise ValueError('protected tensors changed or decoder did not change')
    saved={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'sampler_rng':sampler.get_state(),
        'sampler_permutation_remaining':permutation.copy(),'CPU_rng':torch.get_rng_state(),
        'CUDA_rng':torch.cuda.get_rng_state_all(),'step':updates,'producer_fingerprint':proof['fingerprint'],
        'parent_checkpoint_sha256':PARENT_SHA,'protected_sha256':frozen}
    torch.save(saved,root/'model.pt')
    del saved,opt;model.close();del model;gc.collect();torch.cuda.empty_cache()
    model=load_parent(training)
    saved=torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);decoder.configure(model,True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=offload.CPUAdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);sampler.set_state(saved['sampler_rng'])
    torch.set_rng_state(saved['CPU_rng']);torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same=training.preparation.checkpoint.training.lifecycle.same
    if (not same(model.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
            or not same(scaler.state_dict(),saved['scaler']) or not torch.equal(sampler.get_state(),saved['sampler_rng'])
            or not torch.equal(torch.get_rng_state(),saved['CPU_rng']) or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
            or permutation!=saved['sampler_permutation_remaining'] or decoder.protected(model)!=frozen
            or training.native.vendor.module_hash(model)!=final_hash):
        raise ValueError('strict checkpoint/optimizer/scaler/RNG reload failed')
    by_id={r['id']:r for r in final}
    if evaluate(model,controls,loader,predictions,tokenizer)!=[by_id[r['id']] for r in controls]:
        raise ValueError('strict output reload failed')
    write_json(root/'outputs.json',final)
    training.native.vendor.verify(proof)
    result={'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,'status':'decoder_'+stage+'_completed_quality_pending',
        'updates':updates,'TRAIN_cases':732,'trainable_parameters':sum(p.numel() for p in params),
        'initial_controls':decoder.summarize(initial),'final_TRAIN_diagnostic':decoder.summarize(final),
        'strict_reload_passed':True,'protected_sha256':frozen,'parent_checkpoint_sha256':PARENT_SHA,
        'checkpoint_sha256':file_hash(root/'model.pt'),'final_model_sha256':final_hash,
        'producer_fingerprint':proof['fingerprint'],'training_seconds':elapsed,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'feature_cache':loader.metrics(),
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False}
    if result['peak_allocated_bytes']>CONFIG['peak_limit_bytes']:raise ValueError('full lifecycle memory limit exceeded')
    write_json(root/'result.json',result);print(json.dumps(result),flush=True);model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=tuple(RUNS))
    run(p.parse_args().stage)
