"""Match original singleton encoding and separately measure batching differences."""
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_recovered_decoder_confusions as previous

evaluation, study, data = previous.evaluation, previous.study, previous.data
ROOT = data.ROOT/'20261028_recovered_decoder_confusion_codes_v3'


def run():
    if ROOT.exists(): raise ValueError('fresh singleton encoding diagnostic required')
    for stage in ('replay', 'evaluate'): study.require(evaluation.ROOTS[stage])
    training, environment = study.runtime.codec_source.prior.runtime()
    study.runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER'):
        raise ValueError('physical GPU2 required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cases, outputs = [data.read(evaluation.ROOTS['evaluate']/n) for n in ('cases.json', 'outputs.json')]
    if len(cases) != 128 or [r['id'] for r in cases] != [r['id'] for r in outputs]: raise ValueError('same full128 sample required')
    paths = [Path(__file__), previous.PLAN, Path(previous.__file__), study.runtime.codec_source.BROAD/'model.pt']
    for stage in ('replay', 'evaluate'):
        paths += [evaluation.ROOTS[stage]/n for n in ('result.json', 'actual_tool_receipt.json')]
    paths += [evaluation.ROOTS['evaluate']/n for n in ('producer.json', 'cases.json', 'outputs.json')]
    paths += [previous.ROOT/n for n in ('producer.json', 'failure.json', 'actual_tool_receipt.json')]
    proof = evaluation.freeze(ROOT, 'singleton_confusions_and_batch_diagnostic', training, environment, paths)
    torch.cuda.reset_peak_memory_stats(); codec, frozen = study.load_initial(training)
    codec.eval().requires_grad_(False); records = []; levels = list(training.native.LEVELS)
    if sum(levels) != 641: raise ValueError('original641-code hierarchy required')
    with torch.no_grad():
        for row, output in zip(cases, outputs, strict=True):
            gold = torch.tensor([row['native_word_ids']], dtype=torch.long, device='cuda')
            predicted = torch.tensor([output['raw_word_ids']], dtype=torch.long, device='cuda')
            if gold.shape != (1,128) or predicted.shape != gold.shape: raise ValueError('complete native frames required')
            predicted = torch.where(gold == 0, 0, predicted)
            if (not torch.equal(predicted[gold == 101], gold[gold == 101])
                    or (predicted[gold != 0] <= 0).any() or (predicted >= 30522).any()):
                raise ValueError('native active vocabulary or boundaries changed')
            reference_codes, predicted_codes = codec.encode(gold), codec.encode(predicted)
            paired_codes = codec.encode(torch.cat((gold, predicted), dim=0))
            diagnostic = {'id': row['id'], 'reference_shape': list(reference_codes.shape),
                'predicted_shape': list(predicted_codes.shape), 'batched_shape': list(paired_codes.shape),
                'reference_matches_saved': reference_codes[0].tolist() == output['codes']}
            data.write(ROOT/'current_case.json', diagnostic)
            if reference_codes.shape != (1,641) or predicted_codes.shape != (1,641) or not diagnostic['reference_matches_saved']:
                raise ValueError('singleton reference must reproduce original evaluation')
            if paired_codes.shape != (2,641): raise ValueError('complete paired encoding required')
            identical = torch.equal(gold,predicted)
            if identical != output['exact_active_sequence']: raise ValueError('active-frame control differs')
            changed = reference_codes[0] != predicted_codes[0]; start = 0; counts = []
            for width in levels:
                counts.append(int(changed[start:start+width].sum())); start += width
            if identical and any(counts): raise ValueError('identical singleton frames changed codes')
            records.append({**diagnostic, 'identical_active_frame': identical,
                'changed_code_ids': sum(counts), 'changed_code_ids_by_level': counts,
                'reference_codes': reference_codes[0].tolist(), 'decoded_frame_codes': predicted_codes[0].tolist(),
                'distinct_frame_same_codes': not identical and not any(counts),
                'batched_reference_differences': int((reference_codes[0] != paired_codes[0]).sum()),
                'batched_decoded_differences': int((predicted_codes[0] != paired_codes[1]).sum())})
            data.write(ROOT/'cases.json', records)
    if study.old.frozen_hash(codec) != frozen: raise ValueError('encoding changed frozen tensors')
    if torch.cuda.max_memory_allocated() > study.runtime.LIMIT: raise ValueError('fixed memory cap exceeded')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    identical = sum(r['identical_active_frame'] for r in records)
    if identical != 117: raise ValueError('all117 exact controls required')
    value = {'schema': proof['schema'], 'status': 'all128_singleton_confusions_and_batch_differences_measured',
        'cases': 128, 'identical_active_frame_controls': identical, 'distinct_active_frames': 128-identical,
        'distinct_frame_same_codes': sum(r['distinct_frame_same_codes'] for r in records),
        'distinct_frame_different_codes': sum(not r['identical_active_frame'] and r['changed_code_ids'] > 0 for r in records),
        'batched_reference_changed_cases': sum(r['batched_reference_differences'] > 0 for r in records),
        'batched_reference_changed_codes': sum(r['batched_reference_differences'] for r in records),
        'all_singleton_references_match_saved_codes': True, 'levels': levels,
        'frozen_encoder_VQ_unchanged': True, 'new_requests': 0, 'task_accuracy': None,
        'global_injectivity_proven': False, 'final_test_opened': False,
        'artifacts': {str(p): data.sha(p) for p in (ROOT/'producer.json', ROOT/'cases.json', ROOT/'current_case.json')}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
