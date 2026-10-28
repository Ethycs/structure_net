"""Frozen native codec reconstruction of the complete fixed recovered sample."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import torch

from experiments.structure_net import ncm_recovered_label_sample as sample
from experiments.structure_net import ncm_public_code_targets as targets
from experiments.structure_net import ncm_source_tree_checkpoint_pilot as lifecycle

data, codec_source = targets.data, targets.codec_source
ROOT = data.ROOT/'20261027_recovered_label_codec_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-label-codec.md')
LIMIT = 3_500_000_000


def aggregate(rows):
    values = {}
    for name in ('all', 'long', 'short'):
        chosen = [r for r in rows if name == 'all' or r['stratum'] == name]
        sums = {k: sum(r[k] for r in chosen) for k in ('active_matches', 'active_tokens', 'content_matches',
            'content_tokens', 'exact_active_sequence', 'exact_rendered', 'has_digits', 'empty', 'invalid_outputs')}
        sums.update(cases=len(chosen), exact_rendered_fraction=sums['exact_rendered']/len(chosen),
            content_agreement=sums['content_matches']/sums['content_tokens'],
            digit_cases_preserved=sum(r['has_digits'] and r['digit_sequence_preserved'] for r in chosen),
            tokenizer_digit_changes=sum(not r['tokenizer_digit_sequence_preserved'] for r in chosen))
        values[name] = sums
    return values


def run():
    if ROOT.exists():
        raise ValueError('fresh append-only reconstruction root required')
    training, environment = codec_source.prior.runtime()
    lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physical GPU2 with sufficient free memory required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    before_services = lifecycle.services()
    saved_sample = data.read(sample.ROOT/'result.json'); receipt = data.read(sample.ROOT/'actual_tool_receipt.json')
    data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0 or saved_sample['cases'] != 128:
        raise ValueError('actual completed fixed sample required')
    cases = data.read(sample.ROOT/'cases.json')
    if len(cases) != 128 or len({r['id'] for r in cases}) != 128:
        raise ValueError('all128 unique targets required')
    if data.sha(codec_source.BROAD/'model.pt') != codec_source.CHECKPOINT_SHA:
        raise ValueError('unchanged repaired-codec checkpoint required')
    tokenizer = training.native.tokenizer()
    for row in cases:
        lexical = tokenizer.encode(row['target_text'], add_special_tokens=False, truncation=False)
        if len(lexical) != row['native_lexical_tokens']:
            raise ValueError('audited native target capacity differs')
        row['native_word_ids'] = targets.catalog.native_frame(lexical, tokenizer.all_special_ids)
        row['native_reference'] = tokenizer.decode(lexical, skip_special_tokens=True)
    control = data.ROOT/'20261025_broad_generator_prepare_v1'
    paths = [sample.ROOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'cases.json')]
    paths += [codec_source.BROAD/n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    paths += [control/n for n in ('cases.json', 'selection.json')]
    paths += [Path('pixi.lock'), Path('pyproject.toml'), training.native.RELEASE/'files/lm1b/tokenizer.pt',
              training.native.RELEASE/'files/lm1b/small/config.yaml']
    files = {Path(__file__).resolve(), PLAN.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and (path.is_relative_to(Path('experiments').resolve())
                    or path.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(path)
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (ROOT/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': 'nal.ncm-recovered-label-codec.v1', 'stage': 'fixed_target_aware_reconstruction',
        'environment': environment, 'config': {'cases': 128, 'FP32': True, 'TF32': False, 'physical_GPU': 2,
            'peak_limit_bytes': LIMIT, 'codec_sha256': codec_source.CODEC_SHA, 'model_updates': 0},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in paths}}
    data.write(ROOT/'producer.json', proof); data.write(ROOT/'cases.json', cases)
    torch.cuda.reset_peak_memory_stats()
    codec = training.native.load_codec().float().eval().requires_grad_(False)
    saved = torch.load(codec_source.BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    codec.load_state_dict(codec_source.codec_state(saved['model'], codec.state_dict()), strict=True); del saved
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA:
        raise ValueError('exact frozen repaired codec required')
    controls = {r['id']: r for r in data.read(control/'cases.json')}
    ids = data.read(control/'selection.json')['diagnostic_ids']
    repeat_ids = {cases[0]['id'], min(cases, key=lambda r: (-r['native_lexical_tokens'], r['id']))['id']}
    outputs = []
    with torch.no_grad():
        for identity in ids:
            row = controls[identity]
            if codec.encode(torch.tensor([row['native_word_ids']], device='cuda'))[0].cpu().tolist() != row['oracle_codes']:
                raise ValueError('original TRAIN control differs')
        if len(ids) != 16:
            raise ValueError('all16 original TRAIN controls required')
        for row in cases:
            gold = torch.tensor([row['native_word_ids']], device='cuda')
            codes = codec.encode(gold)
            if codes.shape != (1, 641) or not torch.equal(codes, codec.encode(gold)):
                raise ValueError('repeated complete hierarchy must match exactly')
            logits = codec.decode(codes)
            if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all():
                raise ValueError('finite complete native logits required')
            raw = logits.argmax(-1)[0].cpu().tolist()
            if row['id'] in repeat_ids and raw != codec.decode(codes).argmax(-1)[0].cpu().tolist():
                raise ValueError('repeated probe decoding differs')
            decoded = training.preparation.rendering.decode(raw, tokenizer)
            record = {'id': row['id'], 'stratum': row['stratum'], 'target_text': row['target_text'],
                'native_reference': row['native_reference'], 'codes': codes[0].cpu().tolist(),
                'raw_word_ids': raw, 'text': decoded['text'],
                'invalid_outputs': int(any(v >= len(tokenizer) for v in raw)),
                'tokenizer_digit_sequence_preserved': re.findall(r'\d+', row['target_text']) == re.findall(r'\d+', row['native_reference']),
                **codec_source.prior.metrics(row['native_word_ids'], raw, row['native_reference'], decoded['text'])}
            outputs.append(record)
            if len(outputs) % 16 == 0:
                print(json.dumps({'reconstructed': len(outputs), 'of': 128}), flush=True)
            if torch.cuda.max_memory_allocated() > LIMIT:
                raise ValueError('fixed own CUDA allocation limit exceeded')
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('frozen codec tensors or gradient state changed')
    after_services = lifecycle.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services):
        raise ValueError('other worker identity changed')
    data.write(ROOT/'outputs.json', outputs); data.write(ROOT/'services.json', {'before': before_services, 'after': after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    result = {'schema': proof['schema'], 'status': 'all128_frozen_native_reconstructions_completed',
        'cases': 128, 'metrics': aggregate(outputs), 'codec_sha256': codec_source.CODEC_SHA,
        'codec_unchanged': True, 'original_TRAIN_controls_exact': 16, 'repeated_encodes_exact': 128,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'model_updates': 0, 'new_judge_requests': 0,
        'task_accuracy': None, 'source_only_summary_accuracy': None, 'final_test_opened': False,
        'speed_tested': False, 'training_admitted': False,
        'artifacts': {str(p): data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    data.write(ROOT/'result.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
