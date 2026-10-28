"""Real-model qualification of two readouts sharing source-predicted codes."""
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_recovered_paired_readout as paired
from experiments.structure_net import ncm_recovered_decoder_readout as decoder
from experiments.structure_net import ncm_task_supervision_training as task
from experiments.structure_net import ncm_lm1b_native_development as rendering

data = task.data
ROOT = data.ROOT/'20261028_recovered_paired_readout_pilot_v1'
CPU = data.ROOT/'20261028_recovered_paired_readout_CPU_v1.json'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paired-readout-qualification.md')
TEST = Path('tests/structure_net/test_ncm_recovered_paired_readout.py')
LIMIT = 4_900_000_000


def freeze(training, environment, paths):
    files = {Path(__file__).resolve(), Path(paired.__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())): files.add(p)
    proof = {'schema': 'nal.ncm-recovered-paired-readout-qualification.v1',
        'hypothesis_id': 'ncm-recovered-paraphrase-training-v1', 'stage': 'readout_systems_pilot',
        'environment': environment, 'config': {'physical_GPU': 0, 'batch': 1, 'FP32': True,
            'TF32': False, 'arms': paired.ARMS, 'source_only': True, 'target_encoding': False,
            'reference_predictions_for_parity_only': True, 'training_updates': 0,
            'peak_limit_bytes': LIMIT, 'feature_cache_bytes': 256*1024**2,
            'final_test_opened': False, 'new_judge_calls': 0, 'speed_tested': False},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys([*paths, Path('pixi.lock'), Path('pyproject.toml')])}}
    proof['fingerprint'] = task.base.fingerprint(proof)
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(ROOT/'producer.json', proof)
    return proof


def run():
    if ROOT.exists(): raise ValueError('fresh readout qualification root required')
    qualification = data.read(CPU); data.verify(qualification['artifacts'])
    if qualification['actual_tool_exit_code'] != 0 or qualification['tests_passed'] != 5:
        raise ValueError('completed paired prediction and state-restoration contracts required')
    training, environment = task.configure()
    if torch.cuda.mem_get_info()[0] < LIMIT: raise ValueError('full GPU lifecycle headroom required')
    rows, index, manifest, paths = task.dataset('existing')
    paths += task.require_parent()+[CPU]
    ids = task.control_ids(rows, index); lookup = {r['id']: r for r in rows}
    selected = [lookup[k] for k in ids]
    sources = task.Sources(rows, index, manifest)
    fitted = decoder.study.require(decoder.study.ROOTS['train'])
    for stage in ('audit', 'replay'): decoder.study.require(decoder.evaluation.ROOTS[stage])
    if data.sha(fitted['checkpoint']) != '7bc8beea1357c15f344903720f1207086591f0223b2e9ec81690c1af468f789b':
        raise ValueError('fixed step6212 decoder required')
    paths += [Path(fitted['checkpoint'])]
    for r in (decoder.study.ROOTS['train'], decoder.evaluation.ROOTS['audit'], decoder.evaluation.ROOTS['replay']):
        paths += [r/n for n in ('result.json', 'actual_tool_receipt.json')]
    for row in index:
        if row['id'] in ids:
            paths += [Path(row['path']), Path(manifest[row['id']]['header_path']), Path(manifest[row['id']]['payload_path'])]
    proof = freeze(training, environment, paths)
    data.write(ROOT/'controls.json', [{'id': r['id'], 'source_sha256': r['source_sha256']} for r in selected])
    torch.cuda.reset_peak_memory_stats()
    model = task.load_initial(training).eval().requires_grad_(False)
    original_hash = training.native.vendor.module_hash(model)
    protected = decoder.parent.decoder.protected(model)
    saved = torch.load(fitted['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if saved['frozen_state_sha256'] != decoder.study.old.frozen_hash(model.codec):
        raise ValueError('new decoder uses a different frozen encoder or VQ')
    tokenizer = training.native.tokenizer(); outputs = {a: [] for a in paired.ARMS}; count = 0
    def generate(m, features, mask):
        nonlocal count
        count += 1
        return training.prior.mechanism.generate(m, features, mask)
    try:
        with torch.no_grad():
            for row in selected:
                features = sources.cache.get(row).unsqueeze(0).cuda()
                mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
                codes = training.prior.mechanism.generate(model, features, mask)
                words = model.word_logits(features, mask, codes).argmax(-1).cpu()[0].tolist()
                actual = paired.predict_parent(model, features, mask, generate)
                if actual != {'codes': codes.cpu()[0].tolist(), 'raw_word_ids': words}:
                    raise ValueError('shared source prediction differs from original sequential reference')
                outputs['parent_with_source'].append({'id': row['id'], **actual, **rendering.decode(words, tokenizer)})
                if torch.cuda.max_memory_allocated() > LIMIT: raise ValueError('readout memory ceiling exceeded')
                print(json.dumps({'source_predictions_verified': count, 'total': len(selected)}), flush=True)
            if count != len(selected): raise ValueError('exactly one shared prediction per source required')
            with paired.standalone_decoder(model, saved['decoder']):
                if (training.native.vendor.module_hash(model.codec) != fitted['final_codec_sha256']
                        or decoder.parent.decoder.protected(model) != protected):
                    raise ValueError('decoder replacement altered protected state')
                new_hash = training.native.vendor.module_hash(model)
                for parent in outputs['parent_with_source']:
                    actual = paired.decode_standalone(model, parent['codes'])
                    if actual != paired.decode_standalone(model, parent['codes']):
                        raise ValueError('standalone decoding did not repeat exactly')
                    outputs['new_codec_only'].append({'id': parent['id'], **actual,
                        **rendering.decode(actual['raw_word_ids'], tokenizer)})
                if training.native.vendor.module_hash(model) != new_hash:
                    raise ValueError('standalone readout changed model parameters')
            paired_identity = paired.paired_identity(outputs['parent_with_source'], outputs['new_codec_only'])
            if (paired_identity != 4 or training.native.vendor.module_hash(model) != original_hash
                    or decoder.parent.decoder.protected(model) != protected):
                raise ValueError('complete paired population or exact parent restoration failed')
            if torch.cuda.max_memory_allocated() > LIMIT: raise ValueError('readout memory ceiling exceeded')
        data.write(ROOT/'outputs.json', outputs); task.base.verify_proof(proof)
        value = {'schema': proof['schema'], 'hypothesis_id': proof['hypothesis_id'],
            'status': 'paired_source_predicted_readouts_qualified', 'cases': len(selected),
            'shared_predictor_calls': count, 'reference_predictor_calls': len(selected),
            'original_sequential_outputs_equal': True, 'all_code_pairs_identical': True,
            'new_readout_exact_repetitions': len(selected), 'nondecoder_state_unchanged': True,
            'exact_parent_restore': True, 'parent_model_sha256': original_hash,
            'new_readout_model_sha256': new_hash, 'decoder_checkpoint_sha256': data.sha(fitted['checkpoint']),
            'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
            'systems_lifecycle_only_not_quality_evidence': True, 'new_trained_predictor_evaluated': False,
            'task_accuracy': None, 'training_updates': 0, 'new_judge_calls': 0,
            'final_test_opened': False, 'speed_tested': False,
            'artifacts': {str(p): data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
        data.write(ROOT/'result.json', value)
        print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)
    finally: model.close()


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
