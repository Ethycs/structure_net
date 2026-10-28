"""Matched released-versus-repaired codec diagnostic on recovered labels."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import torch

from experiments.structure_net import ncm_recovered_label_codec as prior

data = prior.data
ROOT = data.ROOT/'20261027_recovered_released_codec_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-released-codec.md')
EXPECTED = '9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32'


def run():
    if ROOT.exists(): raise ValueError('fresh matched diagnostic root required')
    receipt = data.read(prior.ROOT/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0: raise ValueError('actual repaired-codec completion required')
    training, environment = prior.codec_source.prior.runtime()
    prior.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0] < prior.LIMIT):
        raise ValueError('physicalGPU2 with adequate memory required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    before_services = prior.lifecycle.services()
    cases = data.read(prior.ROOT/'cases.json'); old = data.read(prior.ROOT/'outputs.json')
    if len(cases) != 128 or [r['id'] for r in cases] != [r['id'] for r in old]:
        raise ValueError('same complete ordered128 cases required')
    paths = [prior.ROOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'cases.json', 'outputs.json')]
    paths += [training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml', Path('pixi.lock'), Path('pyproject.toml')]
    files = {Path(__file__).resolve(), PLAN.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-recovered-released-codec.v1', 'environment': environment,
        'config': {'cases': 128, 'FP32': True, 'TF32': False, 'codec_sha256': EXPECTED, 'physical_GPU': 2},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)}, 'input_hashes': {str(p): data.sha(p) for p in paths}}
    data.write(ROOT/'producer.json', proof); data.write(ROOT/'cases.json', cases)
    torch.cuda.reset_peak_memory_stats(); codec = training.native.load_codec(); tokenizer = training.native.tokenizer()
    if training.native.vendor.module_hash(codec) != EXPECTED: raise ValueError('exact released codec required')
    repeat_ids = {cases[0]['id'], min(cases, key=lambda r: (-r['native_lexical_tokens'], r['id']))['id']}
    outputs = []
    with torch.no_grad():
        for row, comparison in zip(cases, old):
            gold = torch.tensor([row['native_word_ids']], device='cuda'); codes = codec.encode(gold)
            if codes[0].cpu().tolist() != comparison['codes'] or not torch.equal(codes, codec.encode(gold)):
                raise ValueError('released and repaired encoders must produce identical hierarchies')
            logits = codec.decode(codes)
            if logits.shape != (1, 128, 30528) or not torch.isfinite(logits).all(): raise ValueError('finite full native logits required')
            raw = logits.argmax(-1)[0].cpu().tolist()
            if row['id'] in repeat_ids and raw != codec.decode(codes).argmax(-1)[0].cpu().tolist(): raise ValueError('repeated decode differs')
            text = training.preparation.rendering.decode(raw, tokenizer)['text']
            outputs.append({'id': row['id'], 'stratum': row['stratum'], 'target_text': row['target_text'],
                'native_reference': row['native_reference'], 'codes': codes[0].cpu().tolist(), 'raw_word_ids': raw, 'text': text,
                'invalid_outputs': int(any(v >= len(tokenizer) for v in raw)),
                'tokenizer_digit_sequence_preserved': re.findall(r'\d+', row['target_text']) == re.findall(r'\d+', row['native_reference']),
                **prior.codec_source.prior.metrics(row['native_word_ids'], raw, row['native_reference'], text)})
            if torch.cuda.max_memory_allocated() > prior.LIMIT: raise ValueError('fixed allocation cap exceeded')
    if training.native.vendor.module_hash(codec) != EXPECTED or any(p.grad is not None for p in codec.parameters()): raise ValueError('released codec changed')
    after_services = prior.lifecycle.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services): raise ValueError('other worker changed')
    data.write(ROOT/'outputs.json', outputs); data.write(ROOT/'services.json', {'before': before_services, 'after': after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    comparison = {'released_fixes_repaired': sum(not a['exact_rendered'] and b['exact_rendered'] for a, b in zip(old, outputs)),
        'released_regresses_repaired': sum(a['exact_rendered'] and not b['exact_rendered'] for a, b in zip(old, outputs)),
        'both_exact': sum(a['exact_rendered'] and b['exact_rendered'] for a, b in zip(old, outputs)),
        'neither_exact': sum(not a['exact_rendered'] and not b['exact_rendered'] for a, b in zip(old, outputs))}
    value = {'schema': proof['schema'], 'status': 'matched_released_codec_reconstructions_complete', 'cases': 128,
        'metrics': prior.aggregate(outputs), 'repaired_metrics': prior.aggregate(old), 'comparison': comparison,
        'codec_sha256': EXPECTED, 'all128_encoded_hierarchies_identical_between_codecs': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'model_updates': 0, 'new_judge_requests': 0,
        'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False, 'training_admitted': False,
        'artifacts': {str(p): data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    data.write(ROOT/'result.json', value); print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
