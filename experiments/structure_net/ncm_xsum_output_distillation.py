"""Final-feature and code-distribution training of the frozen XSum bridge."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.structure_net import ncm_xsum_prefix_bridge as base

data = base.data
HYPOTHESIS = 'ncm-xsum-output-distillation-v1'
SCHEMA = 'nal.ncm-xsum-output-distillation.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-xsum-output-distillation-plan.md')
TEST = Path('tests/structure_net/test_ncm_xsum_output_distillation.py')
ANALYSIS = Path('experiments/neural_architecture_lab/ncm_xsum_output_distillation_results.py')
ROOTS = {s: data.ROOT/f'20261027_xsum_output_distill_{s}_v1' for s in
         ('pilot', 'cache', 'train_mse', 'train_distill', 'evaluate', 'audit', 'results')}
LEVELS = (4, 6, 10, 14, 20, 26, 32, 40, 48)


@dataclass(frozen=True)
class Config:
    seed: int = 6519
    cases: int = 512
    epochs: int = 10
    feature_epochs: int = 5
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    native_cache_agreement: float = .999
    agreement_gate: float = .90
    content_improvement_gate: float = .05
    max_allocated_bytes: int = 4_900_000_000


CONFIG = Config()


def phase(arm, step):
    if arm not in ('mse', 'distill') or not 1 <= step <= CONFIG.cases*CONFIG.epochs:
        raise ValueError('fixed arm and update range required')
    return 'mse' if arm == 'mse' else ('features' if step <= CONFIG.cases*CONFIG.feature_epochs else 'KL')


def rate(step):
    phase('mse', step)
    return CONFIG.min_lr+.5*(CONFIG.max_lr-CONFIG.min_lr)*(1+math.cos(
        math.pi*(step-1)/(CONFIG.cases*CONFIG.epochs-1)))


def level_loss(student, teacher, kind, levels=LEVELS):
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[1] != sum(levels):
        raise ValueError('complete matching level features or distributions required')
    teacher = teacher.detach()
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('finite distillation inputs required')
    values = []
    for a, b in zip(student.split(levels, dim=1), teacher.split(levels, dim=1)):
        if kind == 'features': values.append(F.mse_loss(a, b))
        elif kind == 'KL':
            values.append(F.kl_div(F.log_softmax(a, -1), F.log_softmax(b, -1),
                                  log_target=True, reduction='none').sum(-1).mean())
        else: raise ValueError('unknown output objective')
    return torch.stack(values).mean()


@contextmanager
def capture_final(net):
    captured = []
    def hook(module, args, output): captured.append(output)
    handle = net.transformer.ln_f.register_forward_hook(hook)
    try:
        yield captured
        if len(captured) != 1: raise ValueError('one complete final-feature call required')
    finally: handle.remove()


def logits_from_features(net, hidden):
    if hidden.shape[1:] != (200, 768): raise ValueError('complete final features required')
    return torch.cat([head(h) for head, h in zip(net.lm_heads, hidden.split(LEVELS, dim=1))], dim=1)


def forward(vae, net, embeddings, history):
    """Student content arrives only as embeddings, never real source IDs."""
    length = embeddings.shape[0]
    if embeddings.shape != (length, 384) or not 0 < length <= 512 or history.shape != (1, 200):
        raise ValueError('complete source and code history required')
    with torch.no_grad():
        padding = vae.encoder.wte(torch.full((512-length,), 50257, dtype=torch.long, device=embeddings.device))
        previous = vae.vq.idx_to_ncm_input(history)
        query = net.build_l0_query_input(1, previous.dtype)
    prefix = torch.cat((embeddings, padding), 0).unsqueeze(0)
    x = torch.cat((prefix, query, previous), 1)
    mask = torch.arange(512, device=embeddings.device).unsqueeze(0) >= length
    with capture_final(net) as captured: logits = net(x, prefix_pad_mask=mask)
    hidden = captured[0][:, 512:, :]
    if hidden.shape != (1, 200, 768) or logits.shape != (1, 200, 4096):
        raise ValueError('released hierarchy geometry changed')
    return logits, hidden


def parent():
    for name in ('prepare', 'pilot', 'train', 'evaluate'):
        base.require(base.ROOTS[name])
        base.verify_proof(data.read(base.ROOTS[name]/'producer.json'))
    cases = data.read(base.ROOTS['prepare']/'cases.json')
    if len(cases['fit']) != 512 or len(cases['probe']) != 128:
        raise ValueError('unchanged parent population required')
    return cases


def freeze(root, stage, inputs, environment=None):
    if root.exists(): raise ValueError('fresh append-only stage required')
    prior = data.read(base.ROOTS['prepare']/'producer.json')
    files = {Path(p) for p in prior['source_hashes']}
    files.update((Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), ANALYSIS.resolve()))
    files.update(Path('experiments/structure_net').glob('ncm*.py'))
    parent_inputs = [base.ROOTS[s]/n for s in ('prepare', 'pilot', 'train', 'evaluate')
                     for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    parent_inputs += [base.ROOTS['prepare']/'cases.json', base.ROOTS['pilot']/'normalizers.pt',
                      base.ROOTS['train']/'model.pt', base.ROOTS['evaluate']/'outputs.json',
                      Path('pixi.lock'), Path('pyproject.toml')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage,
             'config': asdict(CONFIG), 'environment': environment,
             'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
             'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys(parent_inputs+list(inputs))}}
    proof['fingerprint'] = base.fingerprint(proof); data.write(root/'producer.json', proof)
    return proof


def finish(root, proof, value, files):
    base.verify_proof(proof)
    data.write(root/'result.json', {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
        **value, 'task_accuracy': None, 'final_test_opened': False, 'new_judge_requests': 0,
        'speed_tested': False, 'artifacts': {str(p): data.sha(p) for p in [root/'producer.json', *files]}})
    print(json.dumps({'stage': proof['stage'], 'status': value['status']}), flush=True)


def bridge_model():
    stats = torch.load(base.ROOTS['pilot']/'normalizers.pt', map_location='cpu', weights_only=True)
    bridge = base.Bridge(stats).cuda().eval()
    saved = torch.load(base.ROOTS['train']/'model.pt', map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True)
    return bridge, stats


def optimizer(bridge):
    return torch.optim.AdamW(bridge.parameters(), lr=CONFIG.max_lr, betas=(.9, .999),
                            eps=1e-8, weight_decay=.01, foreach=False)


@torch.no_grad()
def target(vae, net, row):
    ids = base.prefix_ids(row['source_ids'], 'cuda')
    codes = base.codes_native(net, ids)
    logits, hidden = forward(vae, net, vae.encoder.wte(ids[0, :row['source_positions']]), codes)
    if not torch.equal(logits, logits_from_features(net, hidden)):
        raise ValueError('saved final features do not reproduce exact head logits')
    decisions = base.fixed_greedy(logits)
    return {'codes': codes.cpu(), 'hidden': hidden.cpu(), 'teacher_decisions': decisions.cpu(),
            'native_matches': int((codes == decisions).sum()), 'id': row['id']}


def update(bridge, opt, row, cached, vae, net, stats, arm, step):
    objective = phase(arm, step); lr = rate(step)
    for group in opt.param_groups: group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    learned = bridge(base.features(row))
    if objective == 'mse':
        with torch.no_grad():
            desired = net.transformer.wte(vae.encoder.wte(torch.tensor(row['source_ids'], device='cuda')))
        loss = (net.transformer.wte(learned)-desired).square().mean()/stats['projected_variance'].cuda()
    else:
        codes = cached['codes'].cuda(); desired = cached['hidden'].cuda()
        logits, hidden = forward(vae, net, learned, codes)
        if objective == 'features': loss = level_loss(hidden, desired, 'features')
        else:
            with torch.no_grad(): teacher_logits = logits_from_features(net, desired)
            loss = level_loss(logits, teacher_logits, 'KL')
    if not torch.isfinite(loss): raise ValueError('nonfinite bridge objective')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in bridge.parameters()):
        raise ValueError('missing or nonfinite bridge gradient')
    norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.)
    opt.step()
    if any(not torch.isfinite(p).all() for p in bridge.parameters()): raise ValueError('nonfinite bridge')
    if torch.cuda.max_memory_allocated() > CONFIG.max_allocated_bytes: raise ValueError('allocation cap exceeded')
    return {'step': step, 'id': row['id'], 'arm': arm, 'phase': objective,
            'loss': float(loss.detach()), 'gradient_norm': float(norm), 'learning_rate': lr}


def pilot():
    cases = parent(); training, environment = base.runtime(); before = base.active.live_training()
    root = ROOTS['pilot']; proof = freeze(root, 'pilot', [], environment)
    vae, net, _ = base.models(); expected = base.hashes(training, vae, net)
    if not net.query_level0 or net.predict_level0 or net.l0_ar or net.training or vae.training:
        raise ValueError('fixed released query hierarchy in eval mode required')
    probes = [cases['fit'][0], max(cases['fit'], key=lambda r: r['source_positions'])]
    controls = []
    for row in probes:
        cached = target(vae, net, row)
        if cached['native_matches'] != 200: raise ValueError('pilot native full/level decisions differ')
        ids = base.prefix_ids(row['source_ids'], 'cuda')
        from experiments.structure_net.ncm_llama import generator_input
        with torch.no_grad():
            direct = net(generator_input(vae, net, ids, cached['codes'].cuda()), prefix_pad_mask=ids.eq(50257))
            injected, hidden = forward(vae, net, vae.encoder.wte(ids[0, :row['source_positions']]), cached['codes'].cuda())
        if not torch.equal(direct, injected) or not torch.equal(hidden.cpu(), cached['hidden']):
            raise ValueError('native/replacement final features or logits differ')
        controls.append({'id': row['id'], 'native_matches': cached['native_matches']})
    bridge, stats = bridge_model(); opt = optimizer(bridge)
    first = update(bridge, opt, probes[0], target(vae, net, probes[0]), vae, net, stats, 'distill', 1)
    torch.save({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()}, root/'step1.pt')
    cached = target(vae, net, probes[1])
    second = update(bridge, opt, probes[1], cached, vae, net, stats, 'distill', 2561)
    expected_state = deepcopy({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()})
    saved = torch.load(root/'step1.pt', map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True); opt.load_state_dict(saved['optimizer'])
    repeated = update(bridge, opt, probes[1], cached, vae, net, stats, 'distill', 2561)
    if second != repeated or not base.equal_state(expected_state, {'bridge': bridge.state_dict(), 'optimizer': opt.state_dict()}):
        raise ValueError('distillation optimizer continuation differs')
    if any(not torch.count_nonzero(p.grad) for p in bridge.parameters()): raise ValueError('zero bridge gradient')
    del bridge, opt, expected_state, saved
    bridge, stats = bridge_model(); opt = optimizer(bridge)
    mse = update(bridge, opt, probes[1], None, vae, net, stats, 'mse', 1)
    after = base.guard(training, vae, net, expected)
    finish(root, proof, {'status': 'output_distillation_pilot_passed', 'strict_optimizer_replay': True,
        'controls': controls, 'updates': [first, second, mse], 'frozen_model_hashes': expected,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after}, [root/'step1.pt'])


def cache():
    pilot_result = base.require(ROOTS['pilot']); cases = parent()
    training, environment = base.runtime(); before = base.active.live_training()
    root = ROOTS['cache']; proof = freeze(root, 'cache', [ROOTS['pilot']/n for n in
        ('producer.json', 'result.json', 'actual_tool_receipt.json')], environment)
    vae, net, _ = base.models(); expected = base.hashes(training, vae, net)
    if expected != pilot_result['frozen_model_hashes']: raise ValueError('frozen model changed')
    (root/'targets').mkdir(); index = []
    for row in cases['fit']:
        value = target(vae, net, row); p = root/'targets'/(row['id']+'.pt'); torch.save(value, p)
        index.append({'id': row['id'], 'path': str(p), 'sha256': data.sha(p), 'native_matches': value['native_matches']})
        if len(index) % 32 == 0: print(json.dumps({'cached': len(index), 'total': 512}), flush=True)
    agreement = sum(r['native_matches'] for r in index)/(512*200)
    data.write(root/'index.json', index)
    after = base.guard(training, vae, net, expected)
    finish(root, proof, {'status': 'teacher_cache_passed' if agreement >= CONFIG.native_cache_agreement else 'teacher_cache_failed',
        'cases': len(index), 'native_full_forward_agreement': agreement, 'frozen_model_hashes': expected,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after},
        [root/'index.json', *[Path(r['path']) for r in index]])


def load_target(row):
    if data.sha(row['path']) != row['sha256']: raise ValueError('teacher cache changed')
    value = torch.load(row['path'], map_location='cpu', weights_only=True)
    if value['id'] != row['id'] or value['hidden'].shape != (1, 200, 768) or value['hidden'].dtype != torch.float32:
        raise ValueError('teacher cache identity or precision differs')
    return value


def train(arm):
    prior = base.require(ROOTS['cache']); base.require(ROOTS['pilot']); cases = parent()
    if prior['status'] != 'teacher_cache_passed': raise ValueError('teacher cache validity failed')
    index = {r['id']: r for r in data.read(ROOTS['cache']/'index.json')}
    if set(index) != {r['id'] for r in cases['fit']}: raise ValueError('complete fit-only cache required')
    training, environment = base.runtime(); before = base.active.live_training()
    root = ROOTS['train_'+arm]; proof = freeze(root, 'train_'+arm, [ROOTS[s]/n for s in ('pilot', 'cache')
        for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]+[ROOTS['cache']/'index.json'], environment)
    vae, net, _ = base.models(); expected = base.hashes(training, vae, net)
    if expected != prior['frozen_model_hashes']: raise ValueError('frozen teacher changed')
    bridge, stats = bridge_model(); initial_sha = training.native.vendor.module_hash(bridge)
    if initial_sha != data.read(base.ROOTS['train']/'result.json')['bridge_sha256']: raise ValueError('warm start differs')
    opt = optimizer(bridge); rng = torch.Generator().manual_seed(CONFIG.seed); step = 0
    checkpoints = []
    with (root/'trace.jsonl').open('x') as stream:
        for epoch in range(CONFIG.epochs):
            for i in torch.randperm(512, generator=rng).tolist():
                step += 1; row = cases['fit'][i]
                record = update(bridge, opt, row, load_target(index[row['id']]) if arm == 'distill' else None,
                                vae, net, stats, arm, step)
                stream.write(json.dumps({**record, 'epoch': epoch+1}, allow_nan=False)+'\n')
                if step % 32 == 0:
                    stream.flush(); print(json.dumps(record), flush=True)
                if step % 512 == 0:
                    p = root/f'step_{step:06d}.pt'; tmp = p.with_suffix('.tmp')
                    torch.save({'bridge': bridge.state_dict(), 'optimizer': opt.state_dict(),
                        'updates': step, 'arm': arm, 'rng_state': rng.get_state()}, tmp)
                    tmp.rename(p); checkpoints.append(p)
    identity = training.native.vendor.module_hash(bridge)
    saved = torch.load(checkpoints[-1], map_location='cuda', weights_only=True)
    bridge.load_state_dict(saved['bridge'], strict=True)
    if identity != training.native.vendor.module_hash(bridge): raise ValueError('strict reload changed bridge')
    after = base.guard(training, vae, net, expected)
    finish(root, proof, {'status': 'output_distillation_training_completed', 'arm': arm, 'updates': step,
        'bridge_sha256': identity, 'initial_bridge_sha256': initial_sha, 'strict_reload': True,
        'frozen_model_hashes': expected, 'checkpoint': str(checkpoints[-1]),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after}, [root/'trace.jsonl', *checkpoints])


@torch.no_grad()
def evaluate():
    cases = parent(); prior = {arm: base.require(ROOTS['train_'+arm]) for arm in ('mse', 'distill')}
    if any(r['updates'] != 5120 or not r['strict_reload'] for r in prior.values()): raise ValueError('both fixed arms required')
    training, environment = base.runtime(); before = base.active.live_training()
    root = ROOTS['evaluate']; proof = freeze(root, 'evaluate', [ROOTS['train_'+a]/n for a in prior
        for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')], environment)
    vae, net, allowance = base.models(); expected = base.hashes(training, vae, net)
    if any(r['frozen_model_hashes'] != expected for r in prior.values()): raise ValueError('frozen teacher changed')
    initial = data.read(base.ROOTS['evaluate']/'outputs.json'); models = {}
    for arm, result in prior.items():
        model, _ = bridge_model()
        model.load_state_dict(torch.load(result['checkpoint'], map_location='cuda', weights_only=True)['bridge'], strict=True)
        if training.native.vendor.module_hash(model) != result['bridge_sha256']: raise ValueError('selected checkpoint differs')
        models[arm] = model
    controls = {cases['probe'][0]['id'], max(cases['probe'], key=lambda r: r['source_positions'])['id']}
    outputs = []
    for row, old in zip(cases['probe'], initial):
        if row['id'] != old['id']: raise ValueError('probe identity changed')
        if row['id'] in controls:
            codes = base.codes_native(net, base.prefix_ids(row['source_ids'], 'cuda'))
            if codes[0].tolist() != old['native']['codes'] or base.words(vae, codes, allowance) != old['native']['words']:
                raise ValueError('native reference replay differs')
        x = base.features(row); out = {'id': row['id'], 'native': old['native'], 'initial': old['bridge']}
        for arm, model in models.items():
            codes = base.codes_bridged(vae, net, model(x))
            out[arm] = {'codes': codes[0].tolist(), 'words': base.words(vae, codes, allowance)}
        outputs.append(out)
        if len(outputs) % 16 == 0: print(json.dumps({'evaluated': len(outputs), 'total': 128}), flush=True)
    data.write(root/'outputs.json', outputs); after = base.guard(training, vae, net, expected)
    if any(training.native.vendor.module_hash(m) != prior[a]['bridge_sha256'] for a, m in models.items()):
        raise ValueError('evaluation changed a bridge')
    finish(root, proof, {'status': 'output_distillation_predictions_completed', 'cases': len(outputs),
        'frozen_model_hashes': expected, 'native_controls': sorted(controls),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_service_before': before, 'training_service_after': after}, [root/'outputs.json'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'cache', 'train_mse', 'train_distill', 'evaluate'))
    stage = parser.parse_args().stage; root = ROOTS[stage]; existed = root.exists()
    try:
        if stage.startswith('train_'): train(stage.removeprefix('train_'))
        else: globals()[stage]()
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json', {'stage': stage, 'error_type': type(error).__name__, 'message': str(error)})
        raise
