"""Distill frozen Llama features into the released XSum raw-prefix interface."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn

from experiments.structure_net import ncm_public_training_bundle as bundle
from experiments.structure_net import ncm_public_training_probe as active
from experiments.structure_net import ncm_decode_policy as decoding
from experiments.structure_net.ncm_training_recipe import cumulative_latent

data = bundle.data
HYPOTHESIS = 'ncm-xsum-prefix-bridge-v1'
SCHEMA = 'nal.ncm-xsum-prefix-bridge.v1'
ROOTS = {s: data.ROOT/f'20261027_xsum_prefix_bridge_{s}_v1'
         for s in ('prepare', 'pilot', 'train', 'evaluate', 'audit', 'results')}
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-xsum-prefix-bridge-plan.md')
TEST = Path('tests/structure_net/test_ncm_xsum_prefix_bridge.py')
AUDITOR = Path('experiments/neural_architecture_lab/ncm_xsum_prefix_bridge_results.py')
RELEASE = Path('data/experiments/ncm_llama/inputs/hazy-xsum-d342ce8/xsum')
RELEASE_HASHES = dict(zip(('config.yaml', 'tokenizer.pt', 'generator.pt'), (
    '5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493',
    '1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7',
    'f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4')))


@dataclass(frozen=True)
class Config:
    fit_cases: int = 512
    probe_cases: int = 128
    prefix_positions: int = 512
    output_positions: int = 48
    feature_width: int = 3072
    hidden_width: int = 1024
    embedding_width: int = 384
    epochs: int = 10
    seed: int = 5827
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    peak_limit_bytes: int = 4_900_000_000
    word_agreement_gate: float = .90
    relative_MSE_gate: float = .50
    MSE_replay_rtol: float = 1e-10
    MSE_replay_atol: float = 1e-8


CONFIG = Config()


def fingerprint(proof):
    return hashlib.sha256(json.dumps({k: v for k, v in proof.items() if k != 'fingerprint'},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def select(entries, fit=CONFIG.fit_cases, probe=CONFIG.probe_cases):
    if len({r['id'] for r in entries}) != len(entries):
        raise ValueError('duplicate source identity')
    eligible = [r for r in entries if type(r['source_positions']) is int
                and 0 < r['source_positions'] <= CONFIG.prefix_positions]
    chosen = sorted(eligible, key=lambda r: hashlib.sha256(
        (HYPOTHESIS+':'+r['id']).encode()).hexdigest())[:fit+probe]
    if (len(chosen) != fit+probe or len({r['source_sha256'] for r in chosen}) != len(chosen)
            or len({r['source_component'] for r in chosen}) != len(chosen)):
        raise ValueError('complete source-disjoint bridge fit and probe sets required')
    return {'fit': chosen[:fit], 'probe': chosen[fit:]}


def require(root, names=('producer.json', 'result.json')):
    receipt = data.read(root/'actual_tool_receipt.json')
    if type(receipt['actual_tool_exit_code']) is not int or receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual successful stage receipt required')
    for name in names:
        if receipt['artifacts'].get(str(root/name)) != data.sha(root/name):
            raise ValueError('receipt does not bind the completed stage')
    data.verify(receipt['artifacts'])
    result = data.read(root/'result.json')
    data.verify(result['artifacts'])
    return result


def freeze(root, stage, inputs, environment=None):
    if root.exists():
        raise ValueError('fresh append-only root required')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), AUDITOR.resolve()}
    files.update(p.resolve() for p in Path('experiments/structure_net').glob('ncm*.py'))
    files.update(p.resolve() for p in active.broad.runtime.PROJECT.joinpath(
        'data/experiments/ncm_llama/vendor/ncm-20261021').rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    inputs = [*inputs, Path('pyproject.toml'), Path('pixi.lock'), *[RELEASE/n for n in RELEASE_HASHES]]
    for name, expected in RELEASE_HASHES.items():
        if data.sha(RELEASE/name) != expected:
            raise ValueError('released conditional checkpoint changed')
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'config': asdict(CONFIG), 'environment': environment, 'task_accuracy': None,
        'final_test_opened': False, 'new_judge_requests': 0,
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys(inputs)}}
    proof['fingerprint'] = fingerprint(proof)
    data.write(root/'producer.json', proof)
    return proof


def verify_proof(proof):
    if proof['fingerprint'] != fingerprint(proof):
        raise ValueError('producer fingerprint differs')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])


def finish(root, proof, value, files):
    verify_proof(proof)
    data.write(root/'result.json', {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
        **value, 'task_accuracy': None, 'final_test_opened': False,
        'new_judge_requests': 0, 'speed_tested': False,
        'artifacts': {str(p): data.sha(p) for p in [root/'producer.json', *files]}})
    print(json.dumps({'stage': proof['stage'], 'status': value['status'], 'task_accuracy': None}), flush=True)


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only selection required')
    admitted = data.completed(bundle.ROOT)
    manifest = data.read(bundle.ROOT/'manifest.json')
    if admitted['cases'] != 162353 or not admitted['features_admitted_for_training']:
        raise ValueError('complete already admitted public TRAIN corpus required')
    data.verify(manifest['index_hashes'])
    metadata = bundle.lines(manifest['source_index'])
    entries = bundle.lines(manifest['feature_index'])
    if len(metadata) != 162353 or len(entries) != 162353:
        raise ValueError('complete source-only indexes required')
    selected = select(metadata)
    root = ROOTS['prepare']
    proof = freeze(root, 'prepare', [bundle.ROOT/n for n in
        ('manifest.json', 'result.json', 'actual_tool_receipt.json')]+[
        Path(manifest['source_index']), Path(manifest['feature_index'])])
    source_path = bundle.feature.metadata.ROOTS['full']/'metadata.jsonl'
    cases = {}
    with source_path.open('rb') as stream:
        for split, rows in selected.items():
            cases[split] = []
            for entry in rows:
                stream.seek(entry['offset_bytes']); payload = stream.read(entry['length_bytes'])
                if hashlib.sha256(payload).hexdigest() != entry['record_sha256']:
                    raise ValueError('selected metadata record changed')
                row = json.loads(payload); ids = row['source_ids']
                feature = entries[entry['catalog_position']]
                binding = {**{k: entry[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
                           'original_body_sha256': bundle.BODY_SHA}
                if (row['id'] != entry['id'] or row['split'] != 'TRAIN' or row['source_truncated']
                        or len(ids) != entry['source_positions']
                        or any(type(v) is not int or not 0 <= v < 50257 for v in ids)
                        or feature['id'] != row['id'] or feature['header']['binding'] != binding
                        or feature['metadata_record_sha256'] != entry['record_sha256']):
                    raise ValueError('complete source-only metadata/feature binding differs')
                cases[split].append({'id': row['id'], 'split': split, 'source_sha256': row['source_sha256'],
                    'source_component': row['source_component'], 'source_positions': len(ids),
                    'source_ids': ids, 'binding': binding, 'metadata_record_sha256': entry['record_sha256'],
                    'feature': feature})
    data.write(root/'cases.json', cases)
    finish(root, proof, {'status': 'bridge_source_selection_bound', 'fit_cases': 512,
        'probe_cases': 128, 'summary_targets_read': False}, [root/'cases.json'])


def features(row, device='cuda'):
    return bundle.feature.store.load(row['feature'], row['binding'],
        source_positions=row['source_positions'],
        metadata_record_sha256=row['metadata_record_sha256']).to(device)


class Bridge(nn.Module):
    def __init__(self, stats, hidden=CONFIG.hidden_width):
        super().__init__()
        for name in ('input_mean', 'input_std', 'embedding_mean'):
            self.register_buffer(name, stats[name].clone())
        self.network = nn.Sequential(nn.Linear(len(self.input_mean), hidden), nn.GELU(),
                                     nn.Linear(hidden, len(self.embedding_mean)))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return self.embedding_mean+self.network((x-self.input_mean)/self.input_std)


def prefix_ids(ids, device):
    if not 0 < len(ids) <= CONFIG.prefix_positions:
        raise ValueError('complete native prefix required; truncation forbidden')
    return torch.tensor([list(ids)+[50257]*(CONFIG.prefix_positions-len(ids))], device=device)


@contextmanager
def replace_prefix(embedding, replacement, dummy):
    calls = []
    def hook(module, args, output):
        if calls or not torch.equal(args[0], dummy) or output.shape != replacement.shape:
            raise ValueError('exactly one isolated dummy-prefix embedding call required')
        calls.append(True)
        return replacement
    handle = embedding.register_forward_hook(hook)
    try:
        yield calls
        if len(calls) != 1:
            raise ValueError('prefix replacement was not consumed')
    finally:
        handle.remove()


@torch.no_grad()
def codes_native(net, ids):
    return net.generate(None, ids, temperature=1., top_k=1, top_p=0.,
                        use_kv_cache=False, cfg_scale=0.)


@torch.no_grad()
def codes_bridged(vae, net, embeddings):
    length = len(embeddings)
    dummy = prefix_ids([0]*length, embeddings.device)
    pad = vae.encoder.wte(torch.full((CONFIG.prefix_positions-length,), 50257,
                                   device=embeddings.device, dtype=torch.long))
    replacement = torch.cat((embeddings, pad), dim=0).unsqueeze(0)
    with replace_prefix(vae.encoder.wte, replacement, dummy):
        return codes_native(net, dummy)


@torch.no_grad()
def words(vae, codes, allowance):
    logits = vae.decoder(cumulative_latent(vae, codes))
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite released word logits')
    return decoding.pick_tokens(logits, frequency=2., deficit=1.,
                                adjacency=[5., 4., .5], allowance=allowance)[0].cpu().tolist()


def runtime():
    training, environment = active.broad.runtime.configure()
    active.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('physical GPU0 RTX3060 required')
    torch.manual_seed(CONFIG.seed); torch.cuda.manual_seed_all(CONFIG.seed)
    return training, environment


def models():
    from experiments.structure_net import ncm_llama as legacy
    vae, net, cfg = legacy.load_ncm()
    vae.eval().requires_grad_(False); net.eval().requires_grad_(False)
    if (getattr(cfg.model, 'contextual_prefix', False) or net.use_ctx_encoder
            or not cfg.model.raw_token_prefix or sum(net.levels) != 200):
        raise ValueError('unchanged released raw-embedding prefix and hierarchy required')
    saved = torch.load(RELEASE/'generator.pt', map_location='cpu', weights_only=True, mmap=True)
    allowance = saved['inference']['token_unigram'].double()
    return vae, net, allowance


def hashes(training, vae, net):
    return [training.native.vendor.module_hash(v) for v in (vae, net)]


def guard(training, vae, net, expected):
    if hashes(training, vae, net) != expected or any(p.grad is not None for m in (vae, net) for p in m.parameters()):
        raise ValueError('released model mutation or gradient')
    if torch.cuda.max_memory_allocated() > CONFIG.peak_limit_bytes:
        raise ValueError('declared GPU allocation cap exceeded')
    return active.live_training()


@torch.no_grad()
def normalization(rows, vae, net):
    sx = torch.zeros(3072, dtype=torch.float64, device='cuda'); xx = sx.clone()
    sy = torch.zeros(384, dtype=torch.float64, device='cuda')
    sz = torch.zeros(768, dtype=torch.float64, device='cuda'); zz = sz.new_zeros(())
    count = 0
    for row in rows:
        x = features(row).double()
        y = vae.encoder.wte(torch.tensor(row['source_ids'], device='cuda'))
        z = net.transformer.wte(y).double()
        sx += x.sum(0); xx += x.square().sum(0); sy += y.double().sum(0)
        sz += z.sum(0); zz += z.square().sum(); count += len(x)
    mean = (sx/count).float(); embedding_mean = (sy/count).float()
    projected_mean = net.transformer.wte(embedding_mean).double()
    variance = ((zz-2*(projected_mean*sz).sum()+count*projected_mean.square().sum())/(count*768)).float()
    if not torch.isfinite(variance) or variance <= 0:
        raise ValueError('positive finite fit-only projected variance required')
    return {'input_mean': mean.cpu(), 'input_std': ((xx/count-(sx/count).square()).clamp_min(0).sqrt().float().clamp_min(1e-5)).cpu(),
        'embedding_mean': embedding_mean.cpu(), 'projected_mean': projected_mean.float().cpu(),
        'projected_variance': variance.cpu(), 'fit_positions': count, 'fit_ids': [r['id'] for r in rows]}


def optimizer(bridge):
    return torch.optim.AdamW(bridge.parameters(), lr=CONFIG.max_lr, betas=(.9, .999),
                            eps=1e-8, weight_decay=.01, foreach=False)


def update(bridge, opt, row, vae, net, variance, step):
    total = CONFIG.fit_cases*CONFIG.epochs
    rate = CONFIG.min_lr+.5*(CONFIG.max_lr-CONFIG.min_lr)*(1+math.cos(math.pi*(step-1)/(total-1)))
    for group in opt.param_groups: group['lr'] = rate
    x = features(row)
    with torch.no_grad():
        target = net.transformer.wte(vae.encoder.wte(torch.tensor(row['source_ids'], device='cuda')))
    opt.zero_grad(set_to_none=True)
    loss = (net.transformer.wte(bridge(x))-target).square().mean()/variance
    if not torch.isfinite(loss): raise ValueError('nonfinite bridge loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in bridge.parameters()):
        raise ValueError('missing or nonfinite bridge gradient')
    norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.)
    opt.step()
    if any(not torch.isfinite(p).all() for p in bridge.parameters()):
        raise ValueError('nonfinite bridge weights')
    return {'step': step, 'id': row['id'], 'positions': len(x), 'normalized_MSE': float(loss.detach()),
            'gradient_norm': float(norm), 'learning_rate': rate}


def equal_state(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def pilot():
    require(ROOTS['prepare']); cases = data.read(ROOTS['prepare']/'cases.json')
    training, environment = runtime(); before = active.live_training()
    root = ROOTS['pilot']
    proof = freeze(root, 'pilot', [ROOTS['prepare']/n for n in ('producer.json', 'result.json', 'cases.json', 'actual_tool_receipt.json')], environment)
    vae, net, allowance = models(); expected = hashes(training, vae, net)
    stats = normalization(cases['fit'], vae, net); torch.save(stats, root/'normalizers.pt')
    torch.manual_seed(CONFIG.seed); bridge = Bridge(stats).cuda(); opt = optimizer(bridge)
    probes = [cases['fit'][0], min(cases['fit'][1:], key=lambda r: (-r['source_positions'], r['id']))]
    first = update(bridge, opt, probes[0], vae, net, stats['projected_variance'].cuda(), 1)
    torch.save({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()}, root/'step1.pt')
    second = update(bridge, opt, probes[1], vae, net, stats['projected_variance'].cuda(), 2)
    expected_state = deepcopy({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()})
    saved = torch.load(root/'step1.pt', map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True); opt.load_state_dict(saved['optimizer'])
    repeated = update(bridge, opt, probes[1], vae, net, stats['projected_variance'].cuda(), 2)
    if second != repeated or not equal_state(expected_state, {'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()}):
        raise ValueError('optimizer continuation does not repeat exactly')
    if any(not torch.count_nonzero(p.grad) for p in bridge.parameters()):
        raise ValueError('every bridge parameter needs nonzero gradient after update2')
    controls = []
    for row in probes:
        ids = prefix_ids(row['source_ids'], 'cuda')
        a = codes_native(net, ids); b = codes_native(net, ids)
        identity = codes_bridged(vae, net, vae.encoder.wte(ids[0, :row['source_positions']]))
        if not torch.equal(a, b) or not torch.equal(a, identity):
            raise ValueError('native repeat or exact prefix injection differs')
        if words(vae, a, allowance) != words(vae, identity, allowance):
            raise ValueError('exact prefix injection changed decoded IDs')
        controls.append({'id': row['id'], 'codes': a[0].cpu().tolist(), 'words': words(vae, a, allowance)})
    after = guard(training, vae, net, expected)
    data.write(root/'controls.json', controls)
    finish(root, proof, {'status': 'prefix_bridge_pilot_passed', 'strict_optimizer_replay': True,
        'identity_prefix_controls': len(controls), 'updates': [first, second],
        'frozen_model_hashes': expected, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after},
        [root/'normalizers.pt', root/'step1.pt', root/'controls.json'])


def train():
    prior = require(ROOTS['pilot']); require(ROOTS['prepare'])
    if prior['status'] != 'prefix_bridge_pilot_passed': raise ValueError('passing pilot required')
    cases = data.read(ROOTS['prepare']/'cases.json'); stats = torch.load(ROOTS['pilot']/'normalizers.pt', weights_only=True)
    if stats['fit_ids'] != [r['id'] for r in cases['fit']]: raise ValueError('fit-only normalization population differs')
    training, environment = runtime(); before = active.live_training()
    root = ROOTS['train']
    proof = freeze(root, 'train', [ROOTS[s]/n for s in ('prepare', 'pilot') for n in
        ('producer.json', 'result.json', 'actual_tool_receipt.json')]+[ROOTS['prepare']/'cases.json', ROOTS['pilot']/'normalizers.pt'], environment)
    vae, net, _ = models(); expected = hashes(training, vae, net)
    if expected != prior['frozen_model_hashes']: raise ValueError('released weights differ from pilot')
    torch.manual_seed(CONFIG.seed); bridge = Bridge(stats).cuda(); opt = optimizer(bridge)
    variance = stats['projected_variance'].cuda(); rng = torch.Generator().manual_seed(CONFIG.seed)
    step = 0
    with (root/'trace.jsonl').open('x') as stream:
        for epoch in range(CONFIG.epochs):
            for i in torch.randperm(len(cases['fit']), generator=rng).tolist():
                step += 1; row = update(bridge, opt, cases['fit'][i], vae, net, variance, step)
                stream.write(json.dumps({**row, 'epoch': epoch+1}, allow_nan=False)+'\n')
                if step % 128 == 0:
                    stream.flush(); print(json.dumps({'step': step, 'epoch': epoch+1, 'normalized_MSE': row['normalized_MSE']}), flush=True)
                if torch.cuda.max_memory_allocated() > CONFIG.peak_limit_bytes:
                    raise ValueError('GPU memory limit exceeded')
    torch.save({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict(), 'updates': step}, root/'model.pt')
    identity = training.native.vendor.module_hash(bridge)
    saved = torch.load(root/'model.pt', map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True)
    if identity != training.native.vendor.module_hash(bridge): raise ValueError('bridge strict reload differs')
    after = guard(training, vae, net, expected)
    finish(root, proof, {'status': 'prefix_bridge_training_completed', 'updates': step,
        'fit_cases': len(cases['fit']), 'bridge_sha256': identity, 'frozen_model_hashes': expected,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'strict_reload': True,
        'training_service_before': before, 'training_service_after': after}, [root/'model.pt', root/'trace.jsonl'])


@torch.no_grad()
def evaluate():
    trained = require(ROOTS['train']); require(ROOTS['prepare']); require(ROOTS['pilot'])
    if trained['updates'] != 5120 or not trained['strict_reload']: raise ValueError('complete fixed bridge training required')
    cases = data.read(ROOTS['prepare']/'cases.json'); stats = torch.load(ROOTS['pilot']/'normalizers.pt', weights_only=True)
    training, environment = runtime(); before = active.live_training(); root = ROOTS['evaluate']
    proof = freeze(root, 'evaluate', [ROOTS[s]/n for s in ('prepare', 'pilot', 'train') for n in
        ('producer.json', 'result.json', 'actual_tool_receipt.json')]+[ROOTS['prepare']/'cases.json', ROOTS['pilot']/'normalizers.pt', ROOTS['train']/'model.pt'], environment)
    vae, net, allowance = models(); expected = hashes(training, vae, net)
    if expected != trained['frozen_model_hashes']: raise ValueError('frozen teacher changed')
    bridge = Bridge(stats).cuda().eval(); saved = torch.load(ROOTS['train']/'model.pt', map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True)
    if training.native.vendor.module_hash(bridge) != trained['bridge_sha256']: raise ValueError('selected bridge differs')
    outputs = []; (root/'prefix_predictions').mkdir()
    for row in cases['probe']:
        x = features(row); learned = bridge(x)
        target = vae.encoder.wte(torch.tensor(row['source_ids'], device='cuda'))
        mean = bridge.embedding_mean.expand_as(target)
        predictions = {'target': net.transformer.wte(target).cpu(), 'learned': net.transformer.wte(learned).cpu(),
                       'mean': net.transformer.wte(bridge.embedding_mean).cpu()}
        path = root/'prefix_predictions'/(row['id']+'.pt'); torch.save(predictions, path)
        record = {'id': row['id'], 'source_positions': len(x), 'prediction_path': str(path),
                  'prediction_sha256': data.sha(path), 'projected_elements': len(x)*768}
        for name, embeddings in (('native', None), ('bridge', learned), ('mean', mean)):
            codes = codes_native(net, prefix_ids(row['source_ids'], 'cuda')) if name == 'native' else codes_bridged(vae, net, embeddings)
            record[name] = {'codes': codes[0].cpu().tolist(), 'words': words(vae, codes, allowance)}
        for name, key in (('learned', 'bridge_squared_error'), ('mean', 'mean_squared_error')):
            record[key] = float((predictions[name].double()-predictions['target'].double()).square().sum())
        outputs.append(record)
        if len(outputs) % 16 == 0: print(json.dumps({'probe_cases': len(outputs), 'total': 128}), flush=True)
        if torch.cuda.max_memory_allocated() > CONFIG.peak_limit_bytes: raise ValueError('GPU memory limit exceeded')
    data.write(root/'outputs.json', outputs)
    after = guard(training, vae, net, expected)
    if training.native.vendor.module_hash(bridge) != trained['bridge_sha256']: raise ValueError('evaluation changed bridge')
    finish(root, proof, {'status': 'prefix_bridge_predictions_completed', 'probe_cases': len(outputs),
        'bridge_sha256': trained['bridge_sha256'], 'frozen_model_hashes': expected,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after},
        [root/'outputs.json', *[Path(r['prediction_path']) for r in outputs]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'train', 'evaluate'))
    stage = parser.parse_args().stage
    root = ROOTS[stage]; existed = root.exists()
    try:
        globals()[stage]()
    except Exception as error:
        if not existed and root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json', {'exception': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise
