"""Fixed-weight, TRAIN-only source guidance after the recovered history diagnostic."""
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_recovered_predictor_diagnostic as prior
from experiments.structure_net import ncm_saved_recovered_code_readout as readout
from experiments.structure_net import ncm_recovered_training_verified_cache as cache

evidence, mechanism = prior.evidence, prior.mechanism
ROOT = prior.BASE/'20261028_recovered_source_guidance_TRAIN_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-source-guidance.md')
TEST = Path('tests/structure_net/test_ncm_recovered_source_guidance.py')
CPU = prior.BASE/'20261028_recovered_source_guidance_CPU_v1.json'
ARMS = {'baseline': (0., 'constant'), 'author_ramp_0.5': (.5, 'ramp'),
        'constant_0.5': (.5, 'constant'), 'constant_1.0': (1., 'constant')}
LIMIT = 3_500_000_000


def weight(scale, schedule, level, levels):
    if (not math.isfinite(scale) or scale < 0 or schedule not in ('ramp', 'constant')
            or not 0 <= level < levels):
        raise ValueError('finite nonnegative guidance and valid level required')
    return scale * level/max(1, levels-1) if schedule == 'ramp' else scale


@contextmanager
def disabled_source(head):
    """Bypass source adapters for one forward; restore instance layout even on error."""
    if head.training or head._bound is not None:
        raise ValueError('eval mode and no active source context required')
    present = 'source_context' in head.__dict__
    previous = head.__dict__.get('source_context')
    head.source_context = lambda *args: nullcontext()
    try:
        yield
    finally:
        if present:
            head.source_context = previous
        else:
            del head.source_context


def mix(conditional, unconditional, strength):
    if (conditional.shape != unconditional.shape or not math.isfinite(strength) or strength < 0
            or not torch.isfinite(conditional).all() or not torch.isfinite(unconditional).all()):
        raise ValueError('same finite logit geometry and nonnegative strength required')
    # The zero arm must be bitwise unchanged, including cancellation near ties.
    return conditional if strength == 0 else (1.+strength)*conditional-strength*unconditional


@torch.no_grad()
def generate(model, source, mask, scale, schedule):
    if model.training or model.concepts.training:
        raise ValueError('deterministic eval mode required')
    levels = model.concepts.generator.levels
    cumulative = source.new_zeros(len(source), levels[-1], model.codec.embed_dim)
    previous, output, calls = None, [], 0
    for level in range(len(levels)):
        conditional = mechanism.level_logits(model, source, mask, previous, level)
        calls += 1
        strength = weight(scale, schedule, level, len(levels))
        if strength:
            with disabled_source(model.concepts):
                unconditional = mechanism.level_logits(model, source, mask, previous, level)
            logits = mix(conditional, unconditional, strength)
            calls += 1
        else:
            logits = conditional
        codes = logits.argmax(-1)
        output.append(codes)
        cumulative, previous = mechanism.advance(model.codec, cumulative, codes, level)
    return torch.cat(output, dim=1), calls


def run():
    if ROOT.exists():
        raise ValueError('fresh fixed-weight TRAIN guidance root required')
    proofs = cache.ProcessProofs()
    checked = proofs.read(CPU)
    proofs.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 3:
        raise ValueError('completed guidance and context-restoration CPU contracts required')
    cache_cpu = proofs.read(cache.CPU)
    proofs.verify(cache_cpu['artifacts'])
    if cache_cpu['actual_tool_exit_code'] != 0 or cache_cpu['tests_passed'] != 3:
        raise ValueError('qualified process-local hash guards required')
    training, environment = prior.ncm_pixi.configure()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060'
            or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physicalGPU1 RTX2060 with declared headroom required')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    diagnostic = proofs.completed(prior.ROOT)
    prior_proof = proofs.read(prior.ROOT/'producer.json')
    proofs.verify(prior_proof['source_hashes'])
    proofs.verify(prior_proof['inputs'])
    if not diagnostic['execution_consistent'] or diagnostic['model_sha256'] != prior.MODEL_SHA:
        raise ValueError('complete consistent latest-predictor diagnostic required')
    trained = proofs.completed(evidence.TRAIN)
    fitted = proofs.completed(readout.decoder.ROOTS['train'])
    proofs.completed(prior.BASE/'20261027_recovered_decoder_audit_v1')
    old_readout = proofs.completed(readout.ROOT)
    old_outputs = proofs.read(readout.ROOT/'outputs.json')
    expected_words = {r['id']: r['raw_word_ids'] for r in old_outputs if r['input_kind'] == 'after_source_prediction'}
    if (fitted['checkpoint_sha256'] != readout.CHECKPOINT_SHA
            or fitted['final_codec_sha256'] != readout.CODEC_SHA
            or old_readout['codec_sha256'] != readout.CODEC_SHA):
        raise ValueError('same fixed improved decoder required')
    decoder_path = Path(fitted['checkpoint'])
    proofs.bind(decoder_path, readout.CHECKPOINT_SHA)
    selected, matrices = prior_proof['selected_records'], []
    if len(selected) != 4 or set(expected_words) != {r['control']['id'] for r in selected}:
        raise ValueError('all original four TRAIN controls required')
    for row in selected:
        entry, meta = row['feature_entry'], row['metadata']
        binding = prior.source_binding(row['pair'], meta, entry, row['control']['source_positions'])
        matrices.append(prior.features.load(entry, binding, source_positions=meta['source_positions'],
                                             metadata_record_sha256=meta['record_sha256']))
    paths = (Path(__file__).resolve(), PLAN, TEST, CPU, Path(cache.__file__).resolve(), cache.PLAN,
             cache.TEST, cache.CPU, Path('pixi.lock'), Path('pyproject.toml'))
    for path in paths:
        proofs.bind(path)
    files = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(Path('experiments').resolve())}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    proof = {'hypothesis_id': 'ncm-recovered-source-guidance-TRAIN-v1',
        'stage': 'exploratory_fixed_weight_TRAIN_inference', 'environment': environment,
        'config': {'arms': ARMS, 'cases': 4, 'seed': 42, 'FP32': True, 'TF32': False,
            'physical_GPU': 1, 'batch': 1, 'peak_GPU_limit_bytes': LIMIT,
            'null_condition': 'bypass added source adapters; keep native CLS and same generated history',
            'trained_null_condition_branch': False, 'temperature': 0., 'single_fixed_readout': True,
            'task_accuracy': None, 'final_test_opened': False, 'input_validation_policy': cache.POLICY},
        'source_hashes': {str(p): proofs.sha(p) for p in sorted(files)}, 'input_hashes': dict(proofs.inputs)}
    proof['fingerprint'] = prior.task.base.fingerprint(proof)
    ROOT.mkdir()
    evidence.write(ROOT/'producer.json', proof)
    torch.cuda.reset_peak_memory_stats()
    model = prior.task.load_initial(training)
    saved = torch.load(evidence.TRAIN/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    if (training.native.vendor.module_hash(model) != prior.MODEL_SHA
            or training.prior.protected(model) != trained['frozen_hashes']):
        raise ValueError('strict latest predictor identity required')
    codec = training.native.load_codec()
    saved = torch.load(decoder_path, map_location='cpu', weights_only=True, mmap=True)
    if readout.decoder.old.frozen_hash(codec) != saved['frozen_state_sha256']:
        raise ValueError('unchanged frozen encoder and codebooks required')
    codec.decoder.load_state_dict(saved['decoder'], strict=True)
    del saved
    codec.eval().requires_grad_(False)
    if training.native.vendor.module_hash(codec) != readout.CODEC_SHA:
        raise ValueError('strict improved codec identity required')
    tokenizer, outputs = training.native.tokenizer(), []
    with torch.no_grad():
        for row, matrix in zip(selected, matrices, strict=True):
            source = matrix.unsqueeze(0).cuda()
            mask = torch.ones(source.shape[:2], dtype=torch.bool, device='cuda')
            identity = row['control']['id']
            for arm, (scale, schedule) in ARMS.items():
                codes, calls = generate(model, source, mask, scale, schedule)
                repeated, repeated_calls = generate(model, source, mask, scale, schedule)
                if not torch.equal(codes, repeated) or repeated_calls != calls:
                    raise ValueError('guided rollout failed exact repetition')
                actual = codes[0].cpu().tolist()
                if arm == 'baseline' and actual != row['saved']:
                    raise ValueError('zero-guidance saved code parity failed')
                if schedule == 'ramp' and actual[0] != row['saved'][0]:
                    raise ValueError('author ramp must leave root prediction unchanged')
                logits = codec.decode(codes)
                if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all():
                    raise ValueError('complete finite word readout required')
                words = logits.argmax(-1)[0].cpu().tolist()
                if arm == 'baseline' and words != expected_words[identity]:
                    raise ValueError('zero-guidance saved word parity failed')
                record = {'id': identity, 'arm': arm, 'codes': actual, 'raw_word_ids': words,
                    **evidence.rendering.decode(words, tokenizer), 'generator_calls': calls,
                    'repeat_generator_calls': repeated_calls, 'reference_root': row['gold'][0],
                    'reference_code_matches': sum(a == b for a, b in zip(actual, row['gold'], strict=True)),
                    'changed_from_baseline_codes': sum(a != b for a, b in zip(actual, row['saved'], strict=True)),
                    'reference_text': row['pair']['target_text'], 'TRAIN_only': True, 'task_accuracy': None}
                outputs.append(record)
                evidence.write(ROOT/f'output_{len(outputs)-1:02d}.json', record)
                print(json.dumps({'outputs': len(outputs), 'arm': arm, 'reference_code_matches': record['reference_code_matches']}), flush=True)
                if torch.cuda.max_memory_allocated() > LIMIT:
                    raise ValueError('declared inference memory limit exceeded')
            del source, mask
    if (training.native.vendor.module_hash(model) != prior.MODEL_SHA
            or training.native.vendor.module_hash(codec) != readout.CODEC_SHA
            or model.concepts._bound is not None or 'source_context' in model.concepts.__dict__
            or any(p.grad is not None for p in (*model.parameters(), *codec.parameters()))):
        raise ValueError('fixed model or source context changed')
    evidence.write(ROOT/'outputs.json', outputs)
    proofs.verify(proof['source_hashes'])
    proofs.verify(proof['input_hashes'])
    value = {'status': 'fixed_recovered_source_guidance_TRAIN_completed', 'cases': 4, 'outputs': 16,
        'baseline_codes_and_words_equal_saved': True, 'all_repeated_codes_equal': True,
        'author_ramp_root_unchanged': True, 'model_and_codec_unchanged': True,
        'code_matches_by_arm': {arm: sum(r['reference_code_matches'] for r in outputs if r['arm'] == arm) for arm in ARMS},
        'code_positions_per_arm': 2564, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_updates': 0, 'new_judge_calls': 0, 'heldout_cases': 0, 'TRAIN_only': True,
        'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): proofs.sha(p) for p in sorted(ROOT.iterdir()) if p.is_file()}}
    evidence.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed and not (ROOT/'failure.json').exists():
            evidence.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                                'task_accuracy': None, 'automatic_retry': False})
        raise
