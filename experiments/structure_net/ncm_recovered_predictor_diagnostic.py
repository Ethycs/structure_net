"""Separate correct-history learning from source-only rollout on four TRAIN cases."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_recovered_saved_examples as evidence
from experiments.structure_net import ncm_recovered_training_stream as stream
from experiments.structure_net import ncm_packed_feature_store as features
from experiments.structure_net import ncm_task_supervision_training as task
from experiments.structure_net import ncm_sequential_levels as mechanism
from experiments.structure_net import ncm_pixi

BASE = evidence.BASE
ROOT = BASE/'20261028_recovered_predictor_history_diagnostic_v1'
META = BASE/'20261028_recovered_paraphrase_metadata_full_v1'
CODES = BASE/'20261028_recovered_paraphrase_codes_full_v1'
FEATURES = BASE/'20261028_recovered_paraphrase_features_full_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-predictor-history-diagnostic.md')
TEST = Path('tests/structure_net/test_ncm_recovered_predictor_diagnostic.py')
CPU = BASE/'20261028_recovered_predictor_history_CPU_v1.json'
CHECKPOINT_SHA = 'edc88690abaa22b29989be5042269d86f011483a76c8f7b473e4e21b41e30799'
MODEL_SHA = 'e2c17018d5a1d38c867854d009bc3fc895f2bee566b338e7719adb539ba84205'
BODY_SHA = '75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c'
CODEC_SHA = '9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32'
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
LIMIT = 4_900_000_000


def selected_index(path, result, keys):
    if result['artifacts'].get(str(path)) != evidence.sha(path):
        raise ValueError('bound complete index required')
    selected = {}
    with Path(path).open() as entries:
        for row in map(json.loads, entries):
            if row['id'] in keys:
                if row['id'] in selected:
                    raise ValueError('duplicate selected identity')
                selected[row['id']] = row
    if set(selected) != keys:
        raise ValueError('all selected identities required')
    return selected


def source_binding(pair, meta, entry, positions):
    binding = {**{k: meta[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
               'original_body_sha256': BODY_SHA}
    if (pair['split'] != 'TRAIN_noisy_candidate' or pair['source_record_id'] != meta['id']
            or pair['source_sha256'] != meta['source_sha256']
            or pair['source_component'] != meta['source_component']
            or entry['id'] != meta['id'] or entry['header']['binding'] != binding
            or entry['header']['shape'] != [positions, 3072] or meta['source_positions'] != positions
            or entry['metadata_record_sha256'] != meta['record_sha256']):
        raise ValueError('exact source/context/feature/control binding required')
    return binding


def score(logits, gold, levels=LEVELS):
    if (logits.ndim != 3 or logits.shape[:2] != gold.shape
            or gold.shape != (1, sum(levels)) or not torch.isfinite(logits).all()):
        raise ValueError('one complete finite hierarchy required')
    prediction, rows, offset = logits.argmax(-1), [], 0
    for width in levels:
        target = gold[:, offset:offset+width]
        scores = logits[:, offset:offset+width].float()
        rows.append({'positions': width, 'CE': float(F.cross_entropy(scores.flatten(0, 1), target.flatten())),
                     'correct_codes': int((prediction[:, offset:offset+width] == target).sum())})
        offset += width
    return {'levels': rows, 'mean_level_CE': sum(r['CE'] for r in rows)/len(rows),
            'correct_codes': sum(r['correct_codes'] for r in rows), 'code_positions': sum(levels),
            'predicted_codes': prediction[0].cpu().tolist(), 'task_accuracy': None}


def root_comparison(left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('matching finite root logits required')
    return {'max_absolute_difference': float((left-right).abs().max()),
            'allclose': torch.allclose(left, right, rtol=1e-4, atol=1e-4),
            'argmax_equal': torch.equal(left.argmax(-1), right.argmax(-1))}


def run():
    if ROOT.exists():
        raise ValueError('fresh diagnostic root required')
    checked = evidence.read(CPU)
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 3:
        raise ValueError('completed focused CPU contracts required')
    for path, digest in checked['artifacts'].items():
        if evidence.sha(path) != digest:
            raise ValueError('qualified source changed')
    training, environment = ncm_pixi.configure()
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
    roots = (evidence.TRAIN, evidence.AUDIT, evidence.CATALOG, META, CODES, FEATURES)
    trained, audited, catalog, metadata, coded, packed = [evidence.result(r) for r in roots]
    if (trained['updates'] != 6416 or audited['status'] != 'recovered_full_training_replayed'
            or audited['source_checkpoint_sha256'] != CHECKPOINT_SHA
            or not audited['actual_saved_protected_tensors_verified']
            or trained['checkpoint_sha256'] != CHECKPOINT_SHA
            or trained['final_model_sha256'] != MODEL_SHA or packed['body_sha256'] != BODY_SHA
            or coded['codec_sha256'] != CODEC_SHA):
        raise ValueError('completed audited training and unchanged representations required')
    paths = [Path(__file__).resolve(), PLAN, TEST, CPU, evidence.TRAIN/'model.pt',
             *[r/n for r in roots for n in ('result.json', 'actual_tool_receipt.json')],
             evidence.TRAIN/'producer.json', evidence.CATALOG/'pairs_index.jsonl',
             META/'index.jsonl', Path(coded['binary']), Path(coded['index']),
             Path(packed['storage'])/'index.jsonl', Path('pixi.lock'), Path('pyproject.toml')]
    for name in ('controls.json', 'final_outputs.json'):
        path = evidence.TRAIN/name
        if trained['artifacts'].get(str(path)) != evidence.sha(path):
            raise ValueError('saved source-only control artifact changed')
        paths.append(path)
    controls = evidence.read(evidence.TRAIN/'controls.json')
    saved_outputs = evidence.read(evidence.TRAIN/'final_outputs.json')
    if (len(controls) != 4 or len(saved_outputs) != 4
            or any(c['population'] != 'recovered' for c in controls)):
        raise ValueError('all four preselected recovered TRAIN controls required')
    pairs = evidence.indexed_rows('pairs', {c['id'] for c in controls}, catalog)
    source_ids = {p['source_record_id'] for p in pairs.values()}
    metas = selected_index(META/'index.jsonl', metadata, source_ids)
    entries = selected_index(Path(packed['storage'])/'index.jsonl', packed, source_ids)
    if evidence.sha(coded['index']) != coded['artifacts'][coded['index']]:
        raise ValueError('bound singleton target-code index required')
    with Path(coded['index']).open() as f:
        codes = stream.CodeStore(coded['binary'], list(map(json.loads, f)),
            expected_sha256=coded['artifacts'][coded['binary']], codec_sha256=CODEC_SHA)
    selected = []
    for control, saved in zip(controls, saved_outputs, strict=True):
        pair = pairs[control['id']]
        meta, entry = metas[pair['source_record_id']], entries[pair['source_record_id']]
        binding = source_binding(pair, meta, entry, control['source_positions'])
        matrix = features.load(entry, binding, source_positions=control['source_positions'],
                               metadata_record_sha256=meta['record_sha256'])
        selected.append({'control': control, 'pair': pair, 'metadata': meta, 'feature_entry': entry,
                         'features': matrix, 'gold': codes.get(pair), 'saved': saved['codes']})
    if evidence.sha(evidence.TRAIN/'model.pt') != CHECKPOINT_SHA:
        raise ValueError('fixed step6416 checkpoint changed')
    original_producer = evidence.read(evidence.TRAIN/'producer.json')
    for path, digest in original_producer['source_hashes'].items():
        if evidence.sha(path) != digest:
            raise ValueError('original producing model source changed')
    source_paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(Path('experiments').resolve())}
    source_paths.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    proof = {'hypothesis_id': 'ncm-recovered-predictor-history-diagnostic-v1',
        'stage': 'exploratory_fixed_TRAIN_diagnostic', 'environment': environment,
        'config': {'cases': 4, 'seed': 42, 'FP32': True, 'TF32': False, 'batch': 1,
            'physical_GPU': 1, 'peak_GPU_limit_bytes': LIMIT, 'root_rtol': 1e-4, 'root_atol': 1e-4,
            'new_training': False, 'heldout_cases': 0, 'new_judge_calls': 0, 'final_test_opened': False},
        'source_hashes': {str(p): evidence.sha(p) for p in sorted(source_paths)},
        'inputs': {str(p): evidence.sha(p) for p in paths},
        'selected_records': [{k: v for k, v in r.items() if k != 'features'} for r in selected]}
    proof['fingerprint'] = hashlib.sha256(json.dumps(proof, sort_keys=True, allow_nan=False).encode()).hexdigest()
    ROOT.mkdir()
    evidence.write(ROOT/'producer.json', proof)
    torch.cuda.reset_peak_memory_stats()
    model = task.load_initial(training)
    saved = torch.load(evidence.TRAIN/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    if (training.native.vendor.module_hash(model) != MODEL_SHA
            or training.prior.protected(model) != trained['frozen_hashes']):
        raise ValueError('exact final model and protected tensors required')
    outputs, root_logits = [], {}
    with torch.no_grad():
        for i, row in enumerate(selected):
            f = row['features'].unsqueeze(0).cuda()
            mask = torch.ones(f.shape[:2], device='cuda', dtype=torch.bool)
            generated = mechanism.generate(model, f, mask)
            predicted = generated[0].cpu().tolist()
            if predicted != row['saved']:
                evidence.write(ROOT/'saved_rollout_mismatch.json', {'id': row['control']['id'],
                    'actual': predicted, 'expected': row['saved']})
                raise ValueError('source-only rollout must reproduce saved final codes exactly')
            gold = torch.tensor([row['gold']], device='cuda', dtype=torch.long)
            explicit_root = mechanism.level_logits(model, f, mask, None, 0).cpu()
            rng = torch.Generator().manual_seed(42+i)
            random_history = torch.randint(8192, gold.shape, generator=rng).cuda()
            metrics, roots_for_case = {}, {'explicit': explicit_root}
            for name, history in (('correct_history', gold), ('self_history', generated),
                                  ('random_history', random_history)):
                logits = model.concepts.teacher_logits(f, mask, history)
                metrics[name] = score(logits, gold)
                roots_for_case[name] = logits[:, :1].cpu()
                del logits
            other = selected[(i+1) % len(selected)]
            swapped = other['features'].unsqueeze(0).cuda()
            swapped_mask = torch.ones(swapped.shape[:2], device='cuda', dtype=torch.bool)
            logits = model.concepts.teacher_logits(swapped, swapped_mask, gold)
            metrics['swapped_source_correct_history'] = score(logits, gold)
            roots_for_case['swapped_source'] = logits[:, :1].cpu()
            del logits, swapped, swapped_mask
            checks = {name: root_comparison(explicit_root, value) for name, value in roots_for_case.items()
                      if name not in ('explicit', 'swapped_source')}
            out = {'id': row['control']['id'], 'source_positions': len(row['features']),
                'swapped_source_id': other['control']['id'], 'reference_codes': row['gold'],
                'source_only_codes': predicted, 'saved_rollout_equal': True,
                'rollout_correct_codes': int((generated == gold).sum()), 'metrics': metrics,
                'root_history_checks': checks,
                'packed_self_history_reproduces_rollout': metrics['self_history']['predicted_codes'] == predicted,
                'TRAIN_only': True, 'task_accuracy': None}
            outputs.append(out)
            root_logits[row['control']['id']] = roots_for_case
            evidence.write(ROOT/f'case_{i}.json', out)
            print(json.dumps({'cases_completed': i+1, 'teacher_correct': metrics['correct_history']['correct_codes'],
                              'rollout_correct': out['rollout_correct_codes'], 'root_checks': checks}), flush=True)
            del f, mask, generated, gold, random_history
            if torch.cuda.max_memory_allocated() > LIMIT:
                raise ValueError('declared GPU memory limit exceeded')
    if (training.native.vendor.module_hash(model) != MODEL_SHA
            or training.prior.protected(model) != trained['frozen_hashes']
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('model tensors changed during inference')
    torch.save(root_logits, ROOT/'root_logits.pt')
    evidence.write(ROOT/'outputs.json', outputs)
    for mapping in (proof['inputs'], proof['source_hashes']):
        for path, digest in mapping.items():
            if evidence.sha(path) != digest:
                raise ValueError('fixed diagnostic input changed')
    valid = all(r['packed_self_history_reproduces_rollout'] and
                all(c['allclose'] and c['argmax_equal'] for c in r['root_history_checks'].values()) for r in outputs)
    totals = {name: {'correct_codes': sum(r['metrics'][name]['correct_codes'] for r in outputs),
                     'mean_level_CE': sum(r['metrics'][name]['mean_level_CE'] for r in outputs)/4}
              for name in outputs[0]['metrics']}
    value = {'status': 'recovered_predictor_history_diagnostic_completed' if valid else 'predictor_execution_consistency_failed',
        'execution_consistent': valid, 'cases': 4, 'code_positions': 2564, 'totals': totals,
        'rollout_correct_codes': sum(r['rollout_correct_codes'] for r in outputs),
        'all_saved_rollouts_equal': True, 'model_unchanged': True, 'model_sha256': MODEL_SHA,
        'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'TRAIN_only': True,
        'heldout_cases': 0, 'task_accuracy': None, 'new_judge_calls': 0, 'training_updates': 0,
        'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): evidence.sha(p) for p in sorted(ROOT.iterdir())}}
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
