"""Test exact code collisions for every fixed held-out reconstruction confusion."""
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_recovered_decoder_evaluation as evaluation

study, data = evaluation.study, evaluation.data
INCIDENT = data.ROOT/'20261028_recovered_decoder_confusion_codes_v1'
ROOT = data.ROOT/'20261028_recovered_decoder_confusion_codes_v2'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-codec-confusion-codes.md')


def run():
    if ROOT.exists(): raise ValueError('fresh encoding diagnostic required')
    checked = study.require(evaluation.ROOTS['replay'])
    reconstructed = study.require(evaluation.ROOTS['evaluate'])
    if checked['status'] != 'all128_raw_outputs_and_metrics_replayed' or reconstructed['cases'] != 128:
        raise ValueError('complete independently replayed fixed128 reconstruction required')
    training, environment = study.runtime.codec_source.prior.runtime()
    study.runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER'):
        raise ValueError('physical GPU2 required')
    cases, outputs = [data.read(evaluation.ROOTS['evaluate']/n) for n in ('cases.json', 'outputs.json')]
    if [r['id'] for r in cases] != [r['id'] for r in outputs] or len(cases) != 128:
        raise ValueError('same ordered128 diagnostic cases required')
    paths = [Path(__file__), PLAN]
    for source in (evaluation.ROOTS['evaluate'], evaluation.ROOTS['replay']):
        paths += [source/n for n in ('result.json', 'actual_tool_receipt.json')]
    paths += [evaluation.ROOTS['evaluate']/n for n in ('producer.json', 'cases.json', 'outputs.json')]
    paths += [INCIDENT/n for n in ('failure.json', 'actual_tool_receipt.json', 'failed_producer_source.py')]
    paths += [study.runtime.codec_source.BROAD/'model.pt']
    proof = evaluation.freeze(ROOT, 'reference_vs_decoded_code_collision_diagnostic', training, environment, paths)
    torch.cuda.reset_peak_memory_stats(); codec, frozen = study.load_initial(training)
    codec.eval().requires_grad_(False); records = []
    levels = list(training.native.LEVELS)
    if sum(levels) != 641: raise ValueError('original641-code hierarchy required')
    with torch.no_grad():
        for row, output in zip(cases, outputs, strict=True):
            gold = torch.tensor([row['native_word_ids']], dtype=torch.long, device='cuda')
            predicted = torch.tensor([output['raw_word_ids']], dtype=torch.long, device='cuda')
            if gold.shape != (1, 128) or predicted.shape != gold.shape: raise ValueError('complete native frames required')
            predicted = torch.where(gold == 0, 0, predicted)
            if (not torch.equal(predicted[gold == 101], gold[gold == 101])
                    or (predicted[gold != 0] <= 0).any() or (predicted >= 30522).any()):
                raise ValueError('native active vocabulary or boundaries changed')
            codes = codec.encode(torch.cat((gold, predicted), dim=0))
            if codes.shape != (2, 641) or codes[0].tolist() != output['codes']:
                raise ValueError('same frozen reference encoding required')
            identical = torch.equal(gold, predicted)
            if identical != output['exact_active_sequence']: raise ValueError('active-frame control differs')
            counts = []; start = 0
            for width in levels:
                counts.append(int((codes[0,start:start+width] != codes[1,start:start+width]).sum())); start += width
            if identical and any(counts): raise ValueError('identical-frame control changed codes')
            records.append({'id': row['id'], 'identical_active_frame': identical,
                'changed_code_ids': sum(counts), 'changed_code_ids_by_level': counts,
                'reference_codes': codes[0].tolist(), 'decoded_frame_codes': codes[1].tolist(),
                'distinct_frame_same_codes': not identical and not any(counts)})
    if study.old.frozen_hash(codec) != frozen: raise ValueError('encoding changed frozen tensors')
    if torch.cuda.max_memory_allocated() > study.runtime.LIMIT: raise ValueError('fixed memory cap exceeded')
    data.write(ROOT/'cases.json', records)
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    identical = sum(r['identical_active_frame'] for r in records)
    if identical != 117: raise ValueError('all117 exact controls required')
    value = {'schema': proof['schema'], 'status': 'all128_reconstruction_confusions_reencoded',
        'cases': 128, 'identical_active_frame_controls': identical, 'distinct_active_frames': 128-identical,
        'distinct_frame_same_codes': sum(r['distinct_frame_same_codes'] for r in records),
        'distinct_frame_different_codes': sum(not r['identical_active_frame'] and r['changed_code_ids'] > 0 for r in records),
        'levels': levels, 'frozen_encoder_VQ_unchanged': True, 'new_requests': 0,
        'task_accuracy': None, 'global_injectivity_proven': False, 'final_test_opened': False,
        'artifacts': {str(p): data.sha(p) for p in (ROOT/'producer.json', ROOT/'cases.json')}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
