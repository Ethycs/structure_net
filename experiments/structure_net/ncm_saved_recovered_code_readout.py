"""Decode saved recovered TRAIN predictions with the fixed improved codec."""
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_recovered_saved_examples as examples
from experiments.structure_net import ncm_recovered_decoder_training as decoder
from experiments.structure_net import ncm_recovered_training_stream as stream
from experiments.structure_net import ncm_pixi

BASE = examples.BASE
ROOT = BASE/'20261028_saved_recovered_codes_new_readout_v1'
CODE_ROOT = BASE/'20261028_recovered_paraphrase_codes_full_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-saved-recovered-code-readout.md')
TEST = Path('tests/structure_net/test_ncm_saved_recovered_code_readout.py')
CPU = BASE/'20261028_saved_recovered_code_readout_CPU_v1.json'
CHECKPOINT_SHA = '7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b'
CODEC_SHA = '44b48863faed5238af13593856b44f41cfc26ab6d959107c670fd10838ec536d'
LIMIT = 1_500_000_000


def fixed_inputs(controls, initial, final, pairs):
    if (len(controls) != 4 or len({r['id'] for r in controls}) != 4
            or set(pairs) != {r['id'] for r in controls}):
        raise ValueError('all four fixed controls required')
    rows = []
    for control, before, after in zip(controls, initial, final, strict=True):
        if control['population'] != 'recovered':
            raise ValueError('original recovered TRAIN controls required')
        for prediction in (before, after):
            if (len(prediction['codes']) != 641
                    or any(type(v) is not int or not 0 <= v < 8192 for v in prediction['codes'])):
                raise ValueError('complete unchanged source-predicted hierarchy required')
        rows.append({'id': control['id'], 'before_source_prediction': list(before['codes']),
            'after_source_prediction': list(after['codes']), 'pair': pairs[control['id']]})
    return rows


def run():
    if ROOT.exists():
        raise ValueError('fresh fixed-code diagnostic root required')
    checked = examples.read(CPU)
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 2:
        raise ValueError('fixed-code pairing and no-oracle-substitution contracts required')
    for path, digest in checked['artifacts'].items():
        if examples.sha(path) != digest:
            raise ValueError('qualified diagnostic source changed')
    training, environment = ncm_pixi.configure()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060'
            or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physicalGPU1 RTX2060 with full readout memory headroom required')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained, audit = examples.result(examples.TRAIN), examples.result(examples.AUDIT)
    if trained['updates'] != 6416 or audit['source_checkpoint_sha256'] != trained['checkpoint_sha256']:
        raise ValueError('completed audited predictor training required')
    fitted = examples.result(decoder.ROOTS['train'])
    decoder_audit = examples.result(BASE/'20261027_recovered_decoder_audit_v1')
    examples.result(BASE/'20261027_recovered_decoder_replay_v1')
    path = Path(fitted['checkpoint'])
    if (examples.sha(path) != CHECKPOINT_SHA or fitted['checkpoint_sha256'] != CHECKPOINT_SHA
            or decoder_audit['checkpoint_sha256'] != CHECKPOINT_SHA
            or fitted['final_codec_sha256'] != CODEC_SHA):
        raise ValueError('fixed audited step6212 decoder required')
    for name in ('controls.json', 'initial_outputs.json', 'final_outputs.json'):
        if trained['artifacts'][str(examples.TRAIN/name)] != examples.sha(examples.TRAIN/name):
            raise ValueError('saved source-only control predictions changed')
    controls = examples.read(examples.TRAIN/'controls.json')
    catalog = examples.result(examples.CATALOG)
    pairs = examples.indexed_rows('pairs', {r['id'] for r in controls}, catalog)
    rows = fixed_inputs(controls, examples.read(examples.TRAIN/'initial_outputs.json'),
                        examples.read(examples.TRAIN/'final_outputs.json'), pairs)
    coded = examples.result(CODE_ROOT)
    with Path(coded['index']).open() as index:
        entries = list(map(json.loads, index))
    code_store = stream.CodeStore(coded['binary'], entries,
        expected_sha256=examples.sha(coded['binary']), codec_sha256=coded['codec_sha256'])
    for row in rows:
        row['reference_reconstruction_control'] = code_store.get(row['pair'])
    inputs = [Path(__file__).resolve(), PLAN, TEST, CPU, path, Path(coded['binary']), Path(coded['index']),
        *[examples.TRAIN/n for n in ('controls.json', 'initial_outputs.json', 'final_outputs.json')],
        examples.CATALOG/'pairs_index.jsonl',
        *[root/n for root in (examples.TRAIN, examples.AUDIT, examples.CATALOG, CODE_ROOT,
            decoder.ROOTS['train'], BASE/'20261027_recovered_decoder_audit_v1',
            BASE/'20261027_recovered_decoder_replay_v1') for n in ('result.json', 'actual_tool_receipt.json')],
        training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml',
        Path('pixi.lock'), Path('pyproject.toml')]
    files = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(Path('experiments').resolve())}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    proof = {'stage': 'fixed_saved_TRAIN_predictions_new_codec_readout', 'environment': environment,
        'config': {'cases': 4, 'physical_GPU': 1, 'FP32': True, 'TF32': False,
            'batch': 1, 'peak_GPU_limit_bytes': LIMIT, 'source_prediction_recomputed': False,
            'reference_codes_are_separate_reconstruction_controls': True,
            'new_training': False, 'new_judge_calls': 0, 'heldout_cases': 0, 'final_test_opened': False},
        'source_hashes': {str(p): examples.sha(p) for p in sorted(files)},
        'inputs': {str(p): examples.sha(p) for p in inputs}}
    ROOT.mkdir()
    examples.write(ROOT/'producer.json', proof)
    torch.cuda.reset_peak_memory_stats()
    codec = training.native.load_codec()
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if decoder.old.frozen_hash(codec) != saved['frozen_state_sha256']:
        raise ValueError('released frozen representation differs')
    codec.decoder.load_state_dict(saved['decoder'], strict=True)
    codec.eval().requires_grad_(False)
    del saved
    if training.native.vendor.module_hash(codec) != CODEC_SHA:
        raise ValueError('exact improved standalone codec required')
    tokenizer, outputs = training.native.tokenizer(), []
    with torch.no_grad():
        for row in rows:
            gold_words = torch.tensor([row['pair']['native_word_ids']], device='cuda', dtype=torch.long)
            if codec.encode(gold_words)[0].cpu().tolist() != row['reference_reconstruction_control']:
                raise ValueError('original singleton target-code identity changed')
            for kind in ('before_source_prediction', 'after_source_prediction', 'reference_reconstruction_control'):
                codes = torch.tensor([row[kind]], device='cuda', dtype=torch.long)
                logits = codec.decode(codes)
                if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all():
                    raise ValueError('complete finite native word output required')
                words = logits.argmax(-1)[0].cpu().tolist()
                if codec.decode(codes).argmax(-1)[0].cpu().tolist() != words:
                    raise ValueError('fixed-code decoder output did not repeat exactly')
                outputs.append({'id': row['id'], 'input_kind': kind, 'codes': row[kind],
                    'raw_word_ids': words, **examples.rendering.decode(words, tokenizer),
                    'reference_text': row['pair']['target_text'], 'TRAIN_only': True,
                    'source_prediction': kind != 'reference_reconstruction_control', 'task_accuracy': None})
                if torch.cuda.max_memory_allocated() > LIMIT:
                    raise ValueError('readout memory limit exceeded')
            print(json.dumps({'cases_completed': len(outputs)//3, 'total': 4}), flush=True)
    if training.native.vendor.module_hash(codec) != CODEC_SHA or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('codec changed during inference')
    examples.write(ROOT/'outputs.json', outputs)
    for mapping in (proof['inputs'], proof['source_hashes']):
        for source, digest in mapping.items():
            if examples.sha(source) != digest:
                raise ValueError('fixed diagnostic input changed')
    value = {'status': 'fixed_saved_recovered_codes_decoded_with_improved_codec',
        'cases': 4, 'outputs': 12, 'repeated_outputs_equal': True, 'source_generator_calls': 0,
        'reference_encoder_control_calls': 4, 'decoder_calls': 24, 'codec_unchanged': True,
        'codec_sha256': CODEC_SHA, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
        'TRAIN_only': True, 'heldout_cases': 0, 'task_accuracy': None, 'new_judge_calls': 0,
        'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(ROOT/n): examples.sha(ROOT/n) for n in ('producer.json', 'outputs.json')}}
    examples.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed and not (ROOT/'failure.json').exists():
            examples.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
